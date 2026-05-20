import sys
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# make project root importable when running this file directly
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import xgboost as xgb
from sklearn.model_selection import GroupShuffleSplit, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from tasks.task2.preprocessing import run_preprocessing, FEATURE_COLS

_REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = str(_REPO_ROOT / "data" / "raw")
SAVE_DIR = str(_REPO_ROOT / "models" / "task2")
PLOT_DIR = str(_REPO_ROOT / "models" / "task2" / "plots")


def train_with_tuning(X_train, y_train, X_val, y_val):
    # Step 1: tune on a 20% sample — finds good params in minutes not hours
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(len(X_train), size=int(len(X_train) * 0.2), replace=False)
    X_sample, y_sample = X_train[sample_idx], y_train[sample_idx]
    print(f"  Tuning on {len(X_sample):,} rows (20% sample) ...")

    param_dist = {
        "max_depth":        [4, 6, 8],
        "min_child_weight": [1, 5, 20],
        "subsample":        [0.7, 0.8, 1.0],
    }
    base = xgb.XGBRegressor(
        n_estimators=200,       # enough to rank params; early stopping handles depth in final fit
        learning_rate=0.05,
        colsample_bytree=0.8,
        random_state=14,
        n_jobs=1,               # let RandomizedSearchCV parallelize across folds instead
        verbosity=0,
    )
    search = RandomizedSearchCV(
        base,
        param_distributions=param_dist,
        n_iter=9,
        cv=3,
        scoring="neg_mean_absolute_error",
        random_state=14,
        n_jobs=-1,
        verbose=1,
    )
    search.fit(X_sample, y_sample)
    print(f"\n  Best params: {search.best_params_}")

    # Step 2: retrain with best params on the FULL training data
    print(f"  Retraining on full {len(X_train):,} rows ...")
    best = xgb.XGBRegressor(
        **search.best_params_,
        n_estimators=2000,
        learning_rate=0.05,
        colsample_bytree=0.8,
        early_stopping_rounds=50,
        eval_metric="mae",
        random_state=14,
        n_jobs=-1,
        verbosity=0,
    )
    best.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=100,
    )
    return best


def evaluate(model, X_test, y_test, plot_dir):
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)

    for n in [2, 5, 10]:
        within = np.mean(np.abs(y_pred - y_test) <= n) * 100
        print(f"  Within {n:>2} mins : {within:.1f}%")

    print(f"\n  RMSE : {rmse:.2f} minutes")
    print(f"  MAE  : {mae:.2f} minutes")
    print(f"  R²   : {r2:.4f}")

    os.makedirs(plot_dir, exist_ok=True)
    _plot_residuals(y_test, y_pred, plot_dir)
    _plot_feature_importances(model, plot_dir)

    return rmse, mae, r2


def _plot_residuals(y_test, y_pred, plot_dir):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].scatter(y_test, y_pred, alpha=0.3, s=5, color="#2980b9")
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    axes[0].plot(lims, lims, "r--", linewidth=1.5, label="Perfect prediction")
    axes[0].set_xlabel("Actual Delay (mins)")
    axes[0].set_ylabel("Predicted Delay (mins)")
    axes[0].set_title("Actual vs Predicted", fontweight="bold")
    axes[0].legend()

    residuals = y_pred - y_test
    axes[1].hist(residuals, bins=60, color="#8e44ad", edgecolor="white", alpha=0.85)
    axes[1].axvline(0, color="red", linestyle="--", linewidth=1.5)
    axes[1].set_xlabel("Residual (Predicted - Actual)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Residuals Distribution", fontweight="bold")

    plt.suptitle("XGBoost - Model Evaluation", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "xgb_residuals.png"), dpi=150)
    plt.close()
    print(f"  Residual plot saved to {plot_dir}")


def _plot_feature_importances(model, plot_dir):
    importances = model.feature_importances_
    sorted_idx   = np.argsort(importances)[::-1]
    sorted_names = [FEATURE_COLS[i] for i in sorted_idx]
    sorted_vals  = importances[sorted_idx]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(sorted_names, sorted_vals, color="#2980b9", edgecolor="white")
    ax.set_title("Feature Importances - XGBoost", fontweight="bold")
    ax.set_ylabel("Importance")
    ax.set_xlabel("Feature")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "xgb_feature_importances.png"), dpi=150)
    plt.close()
    print(f"  Feature importance plot saved to {plot_dir}")


def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"  Model saved to {path}")


if __name__ == "__main__":
    print("--- Preprocessing ---")
    X, y, encoder, df = run_preprocessing(DATA_DIR, SAVE_DIR)

    print("\n--- Splitting by journey ---")
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=14)
    train_idx, test_idx = next(gss.split(X, y, groups=df["journey_id"]))

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    print(f"  Train rows: {len(X_train):,}")
    print(f"  Test rows : {len(X_test):,}")

    # Carve a validation set from train for early stopping
    gss_val = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=0)
    tr_idx, val_idx = next(gss_val.split(
        X_train, y_train,
        groups=df.iloc[train_idx]["journey_id"].values,
    ))
    X_tr, X_val = X_train[tr_idx], X_train[val_idx]
    y_tr, y_val = y_train[tr_idx], y_train[val_idx]
    print(f"  Val rows  : {len(X_val):,}  (early stopping)")

    print("\n--- Training XGBoost (with tuning) ---")
    model = train_with_tuning(X_tr, y_tr, X_val, y_val)

    print("\n--- Evaluation ---")
    evaluate(model, X_test, y_test, PLOT_DIR)

    print("\n--- Saving ---")
    save_model(model, os.path.join(SAVE_DIR, "xgboost.pkl"))
