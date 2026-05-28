"""
MSM1 Ensemble Selection -- loads pre-trained models, no retraining.
Greedy: add models one-by-one (best val-R2 first), stop when val MAE stops improving.
"""

import sys
import os
import pickle
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tasks.task2.preprocessing import run_preprocessing

DATA_DIR  = str(_ROOT / "data" / "raw")
MODEL_DIR = str(_ROOT / "models" / "task2")

MODEL_FILES = {
    "Ridge":             "ridge.pkl",
    "Random Forest":     "random_forest.pkl",
    "Gradient Boosting": "gradient_boosting.pkl",
    "XGBoost":           "xgboost.pkl",
    "LightGBM":          "lightgbm.pkl",
    "KNN":               "knn.pkl",
    "MLP":               "mlp.pkl",
}

def load_models(model_dir):
    models = {}
    for name, fname in MODEL_FILES.items():
        path = os.path.join(model_dir, fname)
        if not os.path.exists(path):
            print(f"  [skip] {name} -- {fname} not found")
            continue
        with open(path, "rb") as f:
            models[name] = pickle.load(f)
        print(f"  loaded {name}")
    return models


def msm1_select(models, X_val, y_val):
    """Greedy MSM1: pre-compute all val predictions, then select by MAE improvement."""
    print("  Pre-computing val predictions...")
    val_preds = {}
    val_r2    = {}
    for n, m in models.items():
        p = m.predict(X_val)
        val_preds[n] = p
        val_r2[n]    = r2_score(y_val, p)
        print(f"    {n:<22} R2={val_r2[n]:.4f}")

    ranked   = sorted(val_r2, key=lambda n: val_r2[n], reverse=True)
    selected = []
    best_mae = float("inf")

    print("\n  Greedy selection:")
    for name in ranked:
        candidate = selected + [name]
        w = np.array([max(val_r2[n], 0.0) for n in candidate])
        w = w / w.sum()
        y_cand   = sum(wi * val_preds[n] for wi, n in zip(w, candidate))
        cand_mae = mean_absolute_error(y_val, y_cand)

        if cand_mae < best_mae:
            best_mae = cand_mae
            selected = candidate
            print(f"  + add '{name}'  -> val MAE={cand_mae:.4f}  (size={len(selected)})")
        else:
            print(f"    skip '{name}' -> val MAE={cand_mae:.4f}  (no improvement)")

    return selected, {n: max(val_r2[n], 0.0) for n in selected}


if __name__ == "__main__":
    print("--- Loading data ---")
    X, y, _, df = run_preprocessing(DATA_DIR, MODEL_DIR)
    groups = df["journey_id"].values

    imputer = SimpleImputer(strategy="median")
    X = imputer.fit_transform(X)

    gss_test = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=14)
    trainval_idx, test_idx = next(gss_test.split(X, y, groups=groups))
    gss_val = GroupShuffleSplit(n_splits=1, test_size=0.176, random_state=14)
    rel_train_idx, rel_val_idx = next(
        gss_val.split(X[trainval_idx], y[trainval_idx], groups=groups[trainval_idx])
    )
    val_idx = trainval_idx[rel_val_idx]

    X_val,  y_val  = X[val_idx],  y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    print(f"  Val: {len(X_val):,}  |  Test: {len(X_test):,}")

    print("\n--- Loading pre-trained models ---")
    models = load_models(MODEL_DIR)
    if len(models) < 2:
        raise RuntimeError("Need at least 2 saved models. Run model_comparison.py first.")

    print("\n--- MSM1 Greedy Selection (on val set) ---")
    selected_names, weights = msm1_select(models, X_val, y_val)
    print(f"\n  Final selected: {selected_names}")

    print("\n  Computing test predictions...")
    w_arr  = np.array([weights[n] for n in selected_names])
    w_arr  = w_arr / w_arr.sum()
    y_pred = sum(w * models[n].predict(X_test) for w, n in zip(w_arr, selected_names))

    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)
    w2   = np.mean(np.abs(y_pred - y_test) <= 2)  * 100
    w5   = np.mean(np.abs(y_pred - y_test) <= 5)  * 100
    w10  = np.mean(np.abs(y_pred - y_test) <= 10) * 100

    print("\n--- MSM1 Ensemble -- Test Results ---")
    total_w = sum(weights.values())
    for n in selected_names:
        print(f"  {n:<22} weight={weights[n]/total_w:.3f}")
    print(f"  MAE:           {mae:.3f} min")
    print(f"  RMSE:          {rmse:.3f} min")
    print(f"  R2:            {r2:.4f}")
    print(f"  Within 2 min:  {w2:.1f}%")
    print(f"  Within 5 min:  {w5:.1f}%")
    print(f"  Within 10 min: {w10:.1f}%")

    out_path = os.path.join(MODEL_DIR, "msm1_ensemble.pkl")
    with open(out_path, "wb") as f:
        pickle.dump({"models": {n: models[n] for n in selected_names}, "weights": weights}, f)
    print(f"\n  Saved -> {out_path}")
