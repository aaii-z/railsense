import sys
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GroupShuffleSplit, RandomizedSearchCV, GroupKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from tasks.task2.preprocessing import run_preprocessing, FEATURE_COLS

_REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = str(_REPO_ROOT / "data" / "raw")
SAVE_DIR = str(_REPO_ROOT / "models" / "task2")
PLOT_DIR = str(_REPO_ROOT / "models" / "task2" / "plots")


def train_with_tuning(X_train, y_train, g_train):
    cv = GroupKFold(n_splits=3)
    param_dist = {
        "n_estimators":  [50, 100, 200],
        "max_depth":     [3, 4, 5],
        "learning_rate": [0.01, 0.05, 0.1],
    }
    print(f"  Tuning on {len(X_train):,} rows (full training set) ...")
    search = RandomizedSearchCV(
        GradientBoostingRegressor(random_state=42),
        param_distributions=param_dist,
        n_iter=9,
        cv=cv,
        scoring="neg_mean_absolute_error",
        random_state=42,
        n_jobs=-1,
        verbose=1,
    )
    search.fit(X_train, y_train, groups=g_train)
    print(f"  Best params: {search.best_params_}")
    return search.best_estimator_


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

    axes[0].scatter(y_test, y_pred, alpha=0.3, s=5, color="#e74c3c")
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    axes[0].plot(lims, lims, "r--", linewidth=1.5, label="Perfect prediction")
    axes[0].set_xlabel("Actual Delay (mins)")
    axes[0].set_ylabel("Predicted Delay (mins)")
    axes[0].set_title("Actual vs Predicted", fontweight="bold")
    axes[0].legend()

    residuals = y_pred - y_test
    axes[1].hist(residuals, bins=60, color="#c0392b", edgecolor="white", alpha=0.85)
    axes[1].axvline(0, color="black", linestyle="--", linewidth=1.5)
    axes[1].set_xlabel("Residual (Predicted - Actual)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Residuals Distribution", fontweight="bold")

    plt.suptitle("Gradient Boosting - Model Evaluation", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "gb_residuals.png"), dpi=150)
    plt.close()
    print(f"  Residual plot saved to {plot_dir}")


def _plot_feature_importances(model, plot_dir):
    importances = model.feature_importances_
    sorted_idx   = np.argsort(importances)[::-1]
    sorted_names = [FEATURE_COLS[i] for i in sorted_idx]
    sorted_vals  = importances[sorted_idx]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(sorted_names, sorted_vals, color="#e74c3c", edgecolor="white")
    ax.set_title("Feature Importances - Gradient Boosting", fontweight="bold")
    ax.set_ylabel("Importance")
    ax.set_xlabel("Feature")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "gb_feature_importances.png"), dpi=150)
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
    groups = df["journey_id"].values

    print("\n--- Splitting by journey ---")
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    g_train         = groups[train_idx]
    print(f"  Train rows: {len(X_train):,}")
    print(f"  Test rows : {len(X_test):,}")

    print("\n--- Training Gradient Boosting (with tuning) ---")
    model = train_with_tuning(X_train, y_train, g_train)

    print("\n--- Evaluation ---")
    evaluate(model, X_test, y_test, PLOT_DIR)

    print("\n--- Saving ---")
    save_model(model, os.path.join(SAVE_DIR, "gradient_boosting.pkl"))
