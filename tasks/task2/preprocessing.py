import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pickle
import os

# Full route order Weymouth → Waterloo
ROUTE = ["WEY","DCH","WRM","HAM","POO","PKS","BSM","BMH","BCU","SOU","SOA","WIN","BSK","WOK","GLD","CLJ","WAT"]

TARGET_COL   = "target_delay_at_destination"
FEATURE_COLS = [
    "station_encoded",       # current station as number
    "destination_encoded",   # destination station as number
    "arrival_delay",         # current delay in minutes
    "planned_arr_mins",      # scheduled arrival (mins since midnight)
    "day_of_week",           # 0=Mon, 6=Sun
    "month",
    "is_peak_hour",
]


def load_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()
    return df


def load_all_years(data_dir: str) -> pd.DataFrame:
    """Load and combine all yearly CSV files from a directory."""
    files = sorted([f for f in os.listdir(data_dir) if f.endswith("_service_details.csv")])
    dfs = []
    for f in files:
        print(f"Loading: {f}")
        dfs.append(load_data(os.path.join(data_dir, f)))
    combined = pd.concat(dfs, ignore_index=True)
    print(f"Combined: {len(combined)} rows across {len(files)} files")
    return combined


def parse_time_to_minutes(t):
    """Convert HH:MM to minutes since midnight. Returns NaN if missing."""
    if pd.isna(t) or str(t).strip() == "":
        return np.nan
    try:
        h, m = str(t).strip().split(":")
        return int(h) * 60 + int(m)
    except:
        return np.nan


def compute_delays(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate arrival and departure delays (actual - planned) in minutes."""
    df = df.copy()
    df["planned_arr_mins"] = df["planned_arrival_time"].apply(parse_time_to_minutes)
    df["planned_dep_mins"] = df["planned_departure_time"].apply(parse_time_to_minutes)
    df["actual_arr_mins"]  = df["actual_arrival_time"].apply(parse_time_to_minutes)
    df["actual_dep_mins"]  = df["actual_departure_time"].apply(parse_time_to_minutes)
    df["arrival_delay"]    = df["actual_arr_mins"] - df["planned_arr_mins"]
    df["departure_delay"]  = df["actual_dep_mins"] - df["planned_dep_mins"]
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add day of week, month, and peak hour flag."""
    df = df.copy()
    df["date_of_service"] = pd.to_datetime(df["date_of_service"], dayfirst=True, errors="coerce")
    df["day_of_week"] = df["date_of_service"].dt.dayofweek
    df["month"]       = df["date_of_service"].dt.month

    def is_peak(mins):
        if pd.isna(mins): return 0
        return int((420 <= mins <= 570) or (1020 <= mins <= 1170))

    df["is_peak_hour"] = df["planned_dep_mins"].apply(is_peak)
    return df


def extract_all_station_delays(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each journey (rid), get the arrival delay at every station.
    Returns a wide-format lookup: rid → {station: delay}
    """
    route_stations = set(ROUTE)
    station_delays = (
        df[df["location"].isin(route_stations)]
        .dropna(subset=["arrival_delay"])
        [["rid", "location", "arrival_delay"]]
        .drop_duplicates(subset=["rid", "location"], keep="first")
    )
    return station_delays


def build_features(df: pd.DataFrame, station_delays: pd.DataFrame) -> pd.DataFrame:
    """
    For each mid-journey station row, create one training row per valid destination
    that is ahead of the current station on the route.

    Example: train is at SOU (index 9) → destinations can be SOA, WIN, BSK, WOK, GLD, CLJ, WAT
    """
    rows = []

    # Build a fast lookup: (rid, station) → delay
    delay_lookup = station_delays.set_index(["rid", "location"])["arrival_delay"].to_dict()

    # Only keep mid-journey stations (exclude origin WEY)
    mid_stations = ROUTE[1:]  # DCH onward

    for _, row in df[df["location"].isin(mid_stations)].iterrows():
        current_station = row["location"]
        rid = row["rid"]

        if pd.isna(row["arrival_delay"]):
            continue

        current_idx = ROUTE.index(current_station)

        # All stations ahead of current position (must be at least one stop ahead)
        for dest in ROUTE[current_idx + 1:]:
            dest_delay = delay_lookup.get((rid, dest), None)
            if dest_delay is None:
                continue  # no recorded arrival at this destination for this journey

            rows.append({
                "rid":               rid,
                "location":          current_station,
                "destination":       dest,
                "arrival_delay":     row["arrival_delay"],
                "planned_arr_mins":  row["planned_arr_mins"],
                "day_of_week":       row["day_of_week"],
                "month":             row["month"],
                "is_peak_hour":      row["is_peak_hour"],
                TARGET_COL:          dest_delay,
            })

    return pd.DataFrame(rows)


def encode_stations(df: pd.DataFrame, encoder=None, fit=True):
    """
    Encode both current station and destination using the same LabelEncoder
    so they share the same integer space.
    """
    df = df.copy()
    if fit:
        encoder = LabelEncoder()
        encoder.fit(ROUTE)  # fit on full route so encoding is consistent
    df["station_encoded"]     = encoder.transform(df["location"])
    df["destination_encoded"] = encoder.transform(df["destination"])
    return df, encoder


def get_X_y(df: pd.DataFrame):
    return df[FEATURE_COLS].values, df[TARGET_COL].values


def run_preprocessing(data_dir: str, save_dir: str = "../../models/task2"):
    df = load_all_years(data_dir)
    df = compute_delays(df)
    df = add_time_features(df)

    station_delays  = extract_all_station_delays(df)
    merged          = build_features(df, station_delays)
    merged, encoder = encode_stations(merged, fit=True)

    print(f"Ready: {len(merged)} rows, {merged['rid'].nunique()} journeys")
    print(f"Destinations covered: {sorted(merged['destination'].unique())}")

    X, y = get_X_y(merged)

    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "station_encoder.pkl"), "wb") as f:
        pickle.dump(encoder, f)

    merged.to_csv(os.path.join(save_dir, "processed_data.csv"), index=False)
    print(f"Saved to {save_dir}/")

    return X, y, encoder, merged


if __name__ == "__main__":
    DATA_DIR = "../../data/raw/"
    X, y, encoder, df = run_preprocessing(DATA_DIR)
    print(f"X: {X.shape}, y: {y.shape}")
    print(df[FEATURE_COLS + [TARGET_COL]].head(10))