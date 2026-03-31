import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pickle
import os

DESTINATION  = "WAT"
TARGET_COL   = "target_delay_at_WAT"
FEATURE_COLS = [
    "station_encoded",   # current station as number
    "arrival_delay",     # current delay in minutes
    "planned_arr_mins",  # scheduled arrival (mins since midnight)
    "day_of_week",       # 0=Mon, 6=Sun
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


def extract_waterloo_delay(df: pd.DataFrame) -> pd.DataFrame:
    """Get final arrival delay at WAT for each journey — this is the prediction target."""
    wat = df[df["location"] == DESTINATION][["rid", "arrival_delay"]].copy()
    wat = wat.rename(columns={"arrival_delay": TARGET_COL})
    return wat.drop_duplicates(subset="rid", keep="first")


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


def build_features(df: pd.DataFrame, wat_delays: pd.DataFrame) -> pd.DataFrame:
    """Combine mid-journey station rows with WAT target delay."""
    exclude = {DESTINATION, "WEY"}
    mid_df  = df[~df["location"].isin(exclude)].copy()
    mid_df  = mid_df.dropna(subset=["arrival_delay"])
    merged  = mid_df.merge(wat_delays, on="rid", how="inner")
    return merged.dropna(subset=[TARGET_COL])


def encode_stations(df: pd.DataFrame, encoder=None, fit=True):
    """Convert station codes to integers."""
    df = df.copy()
    if fit:
        encoder = LabelEncoder()
        df["station_encoded"] = encoder.fit_transform(df["location"])
    else:
        df["station_encoded"] = encoder.transform(df["location"])
    return df, encoder


def get_X_y(df: pd.DataFrame):
    return df[FEATURE_COLS].values, df[TARGET_COL].values


def run_preprocessing(data_dir: str, save_dir: str = "../../models/task2"):
    df = load_all_years(data_dir)
    df = compute_delays(df)
    df = add_time_features(df)

    wat_delays      = extract_waterloo_delay(df)
    merged          = build_features(df, wat_delays)
    merged, encoder = encode_stations(merged, fit=True)

    print(f"Ready: {len(merged)} rows, {merged['rid'].nunique()} journeys")

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
