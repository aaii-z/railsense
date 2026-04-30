import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pickle
import os

ROUTE = ["WEY","DCH","WRM","HAM","POO","PKS","BSM","BMH","BCU","SOU","SOA","WIN","BSK","WOK","GLD","CLJ","WAT"]

TARGET_COL   = "target_delay_at_destination"
FEATURE_COLS = [
    "station_encoded",
    "destination_encoded",
    "arrival_delay",
    "planned_arr_mins",
    "day_of_week",
    "week_of_year",
    "is_peak_hour",
]


def load_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath, dtype={"rid": str})
    df.columns = df.columns.str.strip()
    return df


def parse_time_to_minutes(t):
    if pd.isna(t) or str(t).strip() == "":
        return np.nan
    try:
        h, m = str(t).strip().split(":")
        return int(h) * 60 + int(m)
    except:
        return np.nan


def fix_midnight_wraparound(delay):
    """Fix delays caused by trains crossing midnight."""
    if pd.isna(delay):
        return delay
    if delay < -200:
        return delay + 1440
    if delay > 1000:
        return delay - 1440
    return delay


def compute_delays(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["planned_arr_mins"] = df["planned_arrival_time"].apply(parse_time_to_minutes)
    df["planned_dep_mins"] = df["planned_departure_time"].apply(parse_time_to_minutes)
    df["actual_arr_mins"]  = df["actual_arrival_time"].apply(parse_time_to_minutes)
    df["actual_dep_mins"]  = df["actual_departure_time"].apply(parse_time_to_minutes)
    df["arrival_delay"]    = df["actual_arr_mins"] - df["planned_arr_mins"]
    df["departure_delay"]  = df["actual_dep_mins"] - df["planned_dep_mins"]

    # Fix midnight wraparound first, then clip remaining outliers
    df["arrival_delay"]   = df["arrival_delay"].apply(fix_midnight_wraparound).clip(-30, 120)
    df["departure_delay"] = df["departure_delay"].apply(fix_midnight_wraparound).clip(-30, 120)
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date_of_service"] = pd.to_datetime(df["date_of_service"], dayfirst=True, errors="coerce")
    df["day_of_week"]  = df["date_of_service"].dt.dayofweek
    df["week_of_year"] = df["date_of_service"].dt.isocalendar().week.astype(int)

    def is_peak(mins):
        if pd.isna(mins): return 0
        return int((420 <= mins <= 570) or (1020 <= mins <= 1170))

    df["is_peak_hour"] = df["planned_dep_mins"].apply(is_peak)
    return df


def create_journey_id(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    wey_times = (
        df[df["location"] == "WEY"][["rid", "date_of_service", "planned_departure_time"]]
        .drop_duplicates(subset=["rid", "date_of_service", "planned_departure_time"])
        .rename(columns={"planned_departure_time": "wey_departure"})
    )
    df = df.merge(wey_times, on=["rid", "date_of_service"], how="left")
    df["journey_id"] = df["date_of_service"].astype(str) + "_" + df["wey_departure"].fillna("unknown")
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


def build_features_for_chunk(df: pd.DataFrame, station_delays: pd.DataFrame) -> pd.DataFrame:
    station_position = {s: i for i, s in enumerate(ROUTE)}

    mid_df = df[df["location"].isin(ROUTE[1:])].copy()
    mid_df = mid_df.dropna(subset=["arrival_delay"])
    mid_df["position"] = mid_df["location"].map(station_position)

    dest_df = station_delays.copy()
    dest_df["position_dest"] = dest_df["location"].map(station_position)
    dest_df = dest_df.rename(columns={
        "location":      "destination",
        "arrival_delay": TARGET_COL,
    })

    merged = mid_df.merge(dest_df, on="journey_id")
    merged = merged[merged["position_dest"] > merged["position"]]
    return merged.dropna(subset=["arrival_delay", TARGET_COL])


def encode_stations(df: pd.DataFrame, encoder=None, fit=True):
    df = df.copy()
    if fit:
        encoder = LabelEncoder()
        encoder.fit(ROUTE)
    df["station_encoded"]     = encoder.transform(df["location"])
    df["destination_encoded"] = encoder.transform(df["destination"])
    return df, encoder


def get_X_y(df: pd.DataFrame):
    return df[FEATURE_COLS].values, df[TARGET_COL].values


def run_preprocessing(data_dir: str, save_dir: str = "../../models/task2"):
    files = sorted([f for f in os.listdir(data_dir) if f.endswith("_service_details.csv")])

    chunks = []
    for f in files:
        print(f"Processing: {f}")
        df = load_data(os.path.join(data_dir, f))
        df = compute_delays(df)
        df = add_time_features(df)
        df = create_journey_id(df)

        station_delays = extract_all_station_delays(df)
        chunk = build_features_for_chunk(df, station_delays)
        chunks.append(chunk)
        print(f"  → {len(chunk)} rows, {chunk['journey_id'].nunique()} journeys")

    merged = pd.concat(chunks, ignore_index=True)
    merged, encoder = encode_stations(merged, fit=True)

    print(f"\nReady: {len(merged)} rows, {merged['journey_id'].nunique()} journeys")
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