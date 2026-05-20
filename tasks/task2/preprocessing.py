import os
import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Stations in order WEY → WAT. Used for both directions:
# WEY2WAT: position increases as train travels toward WAT
# WAT2WEY: position increases as train travels toward WEY (we reverse the list)
ROUTE_WEY2WAT = [
    "WEY", "DCH", "WRM", "HAM", "POO", "PKS", "BSM", "BMH",
    "BCU", "SOU", "SOA", "WIN", "BSK", "WOK", "GLD", "CLJ", "WAT",
]
ROUTE_WAT2WEY = list(reversed(ROUTE_WEY2WAT))

# All unique route stations (same set, both directions)
ROUTE = ROUTE_WEY2WAT

TARGET_COL = "target_delay_at_destination"
FEATURE_COLS = [
    "station_encoded",
    "destination_encoded",
    "arrival_delay",
    "departure_delay",       # was computed but never used before
    "planned_arr_mins",
    "day_of_week",
    "week_of_year",
    "is_peak_hour",
    "stops_remaining",       # distance to destination along the route
    "has_delay_reason",      # 1 if late_canc_reason is present
]


def load_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath, dtype={"rid": str})
    df.columns = df.columns.str.strip()
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows that would corrupt training:

    1. Cancelled trains — RIDs where every actual_arrival_time is NaN.
       The train never ran; there is no delay signal to learn from.

    2. Duplicate rid+location stops — happen when a timetable is revised
       mid-operation and two schedule entries exist for the same stop.
       Keep the row that has actual_arrival_time filled in; if both are
       filled in (rare), keep the one with the earlier planned_arrival_time.
    """
    before = len(df)

    # 1. Drop fully cancelled trains
    has_any_actual = df.groupby("rid")["actual_arrival_time"].transform(
        lambda x: x.notna().any()
    )
    df = df[has_any_actual].copy()
    cancelled_rows = before - len(df)

    # 2. Resolve duplicate rid+location entries
    # Sort so rows with actual_arrival_time come first, then by planned time
    df["_has_actual"] = df["actual_arrival_time"].notna().astype(int)
    df = (
        df.sort_values(
            ["rid", "location", "_has_actual", "planned_arrival_time"],
            ascending=[True, True, False, True],
        )
        .drop_duplicates(subset=["rid", "location"], keep="first")
        .drop(columns=["_has_actual"])
    )
    dup_rows = (before - cancelled_rows) - len(df)

    print(f"  Cleaned: removed {cancelled_rows} cancelled-train rows, "
          f"{dup_rows} duplicate stop rows")
    return df


def parse_time_to_minutes(t) -> float:
    """Convert HH:MM or HH:MM:SS string to integer minutes since midnight."""
    if pd.isna(t) or str(t).strip() == "":
        return np.nan
    try:
        parts = str(t).strip().split(":")
        return int(parts[0]) * 60 + int(parts[1])
    except (ValueError, AttributeError, IndexError):
        return np.nan


def fix_midnight_wraparound(delay: float) -> float:
    """Trains crossing midnight produce ±1440-minute errors — undo them."""
    if pd.isna(delay):
        return delay
    if delay < -200:
        return delay + 1440
    if delay > 1000:
        return delay - 1440
    return delay


def _infer_direction(df: pd.DataFrame) -> str:
    """Return 'WEY2WAT' or 'WAT2WEY' based on which terminus appears as origin."""
    # The origin station has no planned_arrival_time (first stop)
    origins = df[df["planned_arrival_time"].isna()]["location"].value_counts()
    if origins.empty:
        return "WEY2WAT"
    top = origins.index[0]
    return "WAT2WEY" if top == "WAT" else "WEY2WAT"


def compute_delays(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["planned_arr_mins"] = df["planned_arrival_time"].apply(parse_time_to_minutes)
    df["planned_dep_mins"] = df["planned_departure_time"].apply(parse_time_to_minutes)
    df["actual_arr_mins"]  = df["actual_arrival_time"].apply(parse_time_to_minutes)
    df["actual_dep_mins"]  = df["actual_departure_time"].apply(parse_time_to_minutes)

    df["arrival_delay"]   = (df["actual_arr_mins"] - df["planned_arr_mins"]).apply(fix_midnight_wraparound).clip(-30, 120)
    df["departure_delay"] = (df["actual_dep_mins"] - df["planned_dep_mins"]).apply(fix_midnight_wraparound).clip(-30, 120)
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date_of_service"] = pd.to_datetime(df["date_of_service"], dayfirst=False, errors="coerce")
    df["day_of_week"]  = df["date_of_service"].dt.dayofweek
    df["week_of_year"] = df["date_of_service"].dt.isocalendar().week.astype(int)
    df["month"]        = df["date_of_service"].dt.month

    def is_peak(mins):
        if pd.isna(mins):
            return 0
        return int((420 <= mins <= 570) or (1020 <= mins <= 1170))

    df["is_peak_hour"] = df["planned_dep_mins"].apply(is_peak)
    return df


def add_delay_reason_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Binary flag: 1 if a delay/cancellation reason code is recorded."""
    df = df.copy()
    df["has_delay_reason"] = df["late_canc_reason"].notna().astype(int)
    return df


def create_journey_id(df: pd.DataFrame, direction: str) -> pd.DataFrame:
    """
    Build a stable journey ID for each train run.

    For WEY2WAT anchor on the WEY departure time.
    For WAT2WEY anchor on the WAT departure time.
    This avoids the 'unknown' problem when the anchor station is the destination.
    """
    df = df.copy()
    origin_station = "WEY" if direction == "WEY2WAT" else "WAT"

    origin_times = (
        df[df["location"] == origin_station][["rid", "date_of_service", "planned_departure_time"]]
        .drop_duplicates(subset=["rid", "date_of_service", "planned_departure_time"])
        .rename(columns={"planned_departure_time": "origin_departure"})
    )
    df = df.merge(origin_times, on=["rid", "date_of_service"], how="left")

    # Journeys with no origin stop recorded get a unique fallback via rid
    df["journey_id"] = (
        df["date_of_service"].astype(str) + "_"
        + df["origin_departure"].fillna(df["rid"].astype(str))
    )

    df = df.sort_values("planned_arr_mins").drop_duplicates(
        subset=["journey_id", "location"], keep="first"
    )
    return df


def extract_all_station_delays(df: pd.DataFrame) -> pd.DataFrame:
    route_stations = set(ROUTE)
    return (
        df[df["location"].isin(route_stations)]
        .dropna(subset=["arrival_delay"])
        [["journey_id", "location", "arrival_delay"]]
        .drop_duplicates(subset=["journey_id", "location"], keep="first")
    )


def build_features_for_chunk(df: pd.DataFrame, station_delays: pd.DataFrame, direction: str) -> pd.DataFrame:
    route = ROUTE_WEY2WAT if direction == "WEY2WAT" else ROUTE_WAT2WEY
    station_position = {s: i for i, s in enumerate(route)}

    # All intermediate stations (exclude the very first terminus)
    mid_df = df[df["location"].isin(route[1:])].copy()
    mid_df = mid_df.dropna(subset=["arrival_delay"])
    mid_df["position"] = mid_df["location"].map(station_position)

    dest_df = station_delays.copy()
    dest_df["position_dest"] = dest_df["location"].map(station_position)
    dest_df = dest_df.rename(columns={
        "location":      "destination",
        "arrival_delay": TARGET_COL,
    })

    merged = mid_df.merge(dest_df, on="journey_id")
    # Only keep pairs where destination is further along the route
    merged = merged[merged["position_dest"] > merged["position"]]
    merged["stops_remaining"] = merged["position_dest"] - merged["position"]
    return merged.dropna(subset=FEATURE_COLS + [TARGET_COL])


def encode_stations(df: pd.DataFrame, encoder=None, fit: bool = True):
    df = df.copy()
    if fit:
        encoder = LabelEncoder()
        encoder.fit(ROUTE)
    df["station_encoded"]     = encoder.transform(df["location"])
    df["destination_encoded"] = encoder.transform(df["destination"])
    return df, encoder


def get_X_y(df: pd.DataFrame):
    return df[FEATURE_COLS].values, df[TARGET_COL].values


def run_preprocessing(data_dir: str, save_dir: str = "../../models/task2", force: bool = False):
    """
    Process raw CSVs into features and save to disk.

    On subsequent calls (e.g. running a second model), if processed_data.csv
    and station_encoder.pkl already exist, skip all the heavy work and load
    them directly. Pass force=True to reprocess from scratch.
    """
    processed_csv  = os.path.join(save_dir, "processed_data.csv")
    encoder_pkl    = os.path.join(save_dir, "station_encoder.pkl")

    if not force and os.path.exists(processed_csv) and os.path.exists(encoder_pkl):
        print(f"Loading cached processed data from {save_dir}/")
        merged = pd.read_csv(processed_csv)
        with open(encoder_pkl, "rb") as f:
            encoder = pickle.load(f)
        print(f"  Loaded: {len(merged):,} rows, {merged['journey_id'].nunique():,} journeys")
        X, y = get_X_y(merged)
        return X, y, encoder, merged

    files = sorted([f for f in os.listdir(data_dir) if f.endswith(".csv")])
    if not files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    chunks = []
    for fname in files:
        filepath = os.path.join(data_dir, fname)
        print(f"Processing: {fname}")
        df = load_data(filepath)
        direction = _infer_direction(df)
        print(f"  Direction: {direction}")
        df = clean_data(df)

        df = compute_delays(df)
        df = add_time_features(df)
        df = add_delay_reason_feature(df)
        df = create_journey_id(df, direction)

        station_delays = extract_all_station_delays(df)
        chunk = build_features_for_chunk(df, station_delays, direction)
        chunks.append(chunk)
        print(f"  → {len(chunk):,} rows, {chunk['journey_id'].nunique():,} journeys")

    merged = pd.concat(chunks, ignore_index=True)
    merged, encoder = encode_stations(merged, fit=True)

    print(f"\nReady: {len(merged):,} rows, {merged['journey_id'].nunique():,} journeys")
    print(f"Destinations covered: {sorted(merged['destination'].unique())}")

    X, y = get_X_y(merged)

    os.makedirs(save_dir, exist_ok=True)
    with open(encoder_pkl, "wb") as f:
        pickle.dump(encoder, f)

    merged.to_csv(processed_csv, index=False)
    print(f"Saved to {save_dir}/")

    return X, y, encoder, merged


if __name__ == "__main__":
    DATA_DIR = str(__import__("pathlib").Path(__file__).resolve().parents[2] / "data" / "raw")
    X, y, encoder, df = run_preprocessing(DATA_DIR)
    print(f"\nX: {X.shape}, y: {y.shape}")
    print(df[FEATURE_COLS + [TARGET_COL]].head(10))
