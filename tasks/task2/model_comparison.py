import sys
import os
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GroupShuffleSplit, RandomizedSearchCV, GroupKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import xgboost as xgb
import lightgbm as lgb

from tasks.task2.preprocessing import run_preprocessing, FEATURE_COLS

DATA_DIR = str(_ROOT / "data" / "raw")
SAVE_DIR = str(_ROOT / "models" / "task2")
PLOT_DIR = str(_ROOT / "models" / "task2" / "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

PALETTE = ["#20808D", "#A84B2F", "#1B474D", "#944454", "#FFC553", "#848456", "#6C5B9E"]

plt.rcParams.update({
    "figure.dpi": 150,
    "font.family": "DejaVu Sans",
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#E0DDD8",
    "grid.linewidth": 0.6,
})


def evaluate(model, X_tr, y_tr, X_te, y_te, fit=True):
    if fit:
        model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    return dict(
        RMSE    = float(np.sqrt(mean_squared_error(y_te, y_pred))),
        MAE     = float(mean_absolute_error(y_te, y_pred)),
        R2      = float(r2_score(y_te, y_pred)),
        Within2 = float(np.mean(np.abs(y_pred - y_te) <= 2)  * 100),
        Within5 = float(np.mean(np.abs(y_pred - y_te) <= 5)  * 100),
        Within10= float(np.mean(np.abs(y_pred - y_te) <= 10) * 100),
        y_pred  = y_pred,
    )


def plot_before_after(baseline_results, tuned_results, model_names, plot_dir):
    x     = np.arange(len(model_names))
    width = 0.35
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Hyperparameter Tuning  Before vs After",
                 fontsize=14, fontweight="bold", y=1.02)

    for ax, metric, title, ylabel in [
        (axes[0], "MAE",  "Mean Absolute Error (MAE)",      "minutes"),
        (axes[1], "RMSE", "Root Mean Squared Error (RMSE)", "minutes"),
    ]:
        base_vals  = [baseline_results[n][metric] for n in model_names]
        tuned_vals = [tuned_results[n][metric]    for n in model_names]
        b1 = ax.bar(x - width/2, base_vals,  width, label="Baseline",
                    color=PALETTE[0], alpha=0.7, edgecolor="white")
        b2 = ax.bar(x + width/2, tuned_vals, width, label="Tuned",
                    color=PALETTE[1], alpha=0.9, edgecolor="white")
        for bar, v in zip(b1, base_vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=8)
        for bar, v in zip(b2, tuned_vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=9)
        ax.set_ylim(0, max(base_vals + tuned_vals) * 1.18)

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "07_tuning_before_after.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print("  saved 07_tuning_before_after.png")


def plot_r2(baseline_results, tuned_results, model_names, plot_dir):
    x     = np.arange(len(model_names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    base_r2  = [baseline_results[n]["R2"] for n in model_names]
    tuned_r2 = [tuned_results[n]["R2"]   for n in model_names]

    b1 = ax.bar(x - width/2, base_r2,  width, label="Baseline",
                color=PALETTE[0], alpha=0.7, edgecolor="white")
    b2 = ax.bar(x + width/2, tuned_r2, width, label="Tuned",
                color=PALETTE[1], alpha=0.9, edgecolor="white")

    all_r2  = base_r2 + tuned_r2
    y_range = max(abs(min(all_r2)), abs(max(all_r2)))
    for bar, v in zip(list(b1) + list(b2), all_r2):
        yp = v - y_range * 0.04 if v < 0 else v + y_range * 0.01
        ax.text(bar.get_x() + bar.get_width()/2, yp, f"{v:.3f}",
                ha="center", va="top" if v < 0 else "bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=9)
    ax.set_ylabel("R²")
    ax.set_title("R² Score  Baseline vs Tuned")
    ax.legend(fontsize=9)
    ax.axhline(0, color="#28251D", lw=0.8, linestyle="--")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "08_tuning_r2.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  saved 08_tuning_r2.png")


def plot_mae_improvement(baseline_results, tuned_results, model_names, plot_dir):
    fig, ax = plt.subplots(figsize=(10, 5))
    deltas     = [baseline_results[n]["MAE"] - tuned_results[n]["MAE"] for n in model_names]
    bar_colors = [PALETTE[3] if d < 0 else PALETTE[2] for d in deltas]
    bars = ax.bar(model_names, deltas, color=bar_colors, edgecolor="white", width=0.5)
    ax.axhline(0, color="#28251D", lw=1.0, linestyle="--")
    for bar, v in zip(bars, deltas):
        yp = v + 0.001 if v >= 0 else v - 0.001
        ax.text(bar.get_x() + bar.get_width()/2, yp, f"{v:+.3f} min",
                ha="center", va="bottom" if v >= 0 else "top",
                fontsize=9, fontweight="bold")
    ax.set_ylabel("MAE reduction (minutes)")
    ax.set_title("MAE Improvement from Tuning\n(positive = better after tuning)")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "09_mae_improvement.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  saved 09_mae_improvement.png")


def plot_within_n(tuned_results, model_names, plot_dir):
    fig, ax = plt.subplots(figsize=(12, 5))
    x2    = np.arange(len(model_names))
    width = 0.25

    for i, (tol, col) in enumerate([(2, PALETTE[0]), (5, PALETTE[1]), (10, PALETTE[2])]):
        vals = [tuned_results[n][f"Within{tol}"] for n in model_names]
        bars = ax.bar(x2 + (i - 1) * width, vals, width,
                      label=f"Within {tol} min", color=col,
                      edgecolor="white", linewidth=0.8)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.4,
                    f"{v:.0f}%", ha="center", va="bottom", fontsize=7.5)

    ax.set_xticks(x2)
    ax.set_xticklabels(model_names, fontsize=9)
    ax.set_ylabel("% of Predictions")
    ax.set_title("Within-N-Minutes Accuracy  Tuned Models")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_ylim(0, 110)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "10_tuned_within_n.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  saved 10_tuned_within_n.png")


def plot_actual_vs_predicted(tuned_results, model_names, y_test, plot_dir):
    n_cols = 3
    n_rows = (len(model_names) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    fig.suptitle("Actual vs Predicted  Tuned Models", fontsize=14, fontweight="bold")
    axes_flat = axes.flatten()
    sample    = np.random.RandomState(14).choice(
        len(y_test), size=min(5000, len(y_test)), replace=False)
    lim_lo, lim_hi = -15, 45

    for ax, (name, col) in zip(axes_flat, zip(model_names, PALETTE)):
        y_pred = tuned_results[name]["y_pred"]
        ax.scatter(y_test[sample], y_pred[sample],
                   alpha=0.25, s=4, color=col, rasterized=True)
        ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], "k--", lw=1.2, label="Perfect")
        ax.set_xlim(lim_lo, lim_hi)
        ax.set_ylim(lim_lo, lim_hi)
        ax.set_xlabel("Actual (min)")
        ax.set_ylabel("Predicted (min)")
        ax.set_title(f"{name}\nR²={tuned_results[name]['R2']:.3f}  "
                     f"MAE={tuned_results[name]['MAE']:.2f} min")
        ax.legend(fontsize=8)

    for ax in axes_flat[len(model_names):]:
        ax.set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "11_tuned_actual_vs_predicted.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print("  saved 11_tuned_actual_vs_predicted.png")


def plot_heatmap(tuned_results, model_names, plot_dir):
    metrics_display = ["RMSE\n(min)", "MAE\n(min)", "R²",
                       "Within\n2 min (%)", "Within\n5 min (%)", "Within\n10 min (%)"]
    metric_keys = ["RMSE", "MAE", "R2", "Within2", "Within5", "Within10"]

    raw_values = np.array([[tuned_results[n][k] for k in metric_keys] for n in model_names])
    norm = raw_values.copy().astype(float)
    for col_i, key in enumerate(metric_keys):
        lo, hi = norm[:, col_i].min(), norm[:, col_i].max()
        if hi == lo:
            norm[:, col_i] = 0.5
            continue
        if key in ("RMSE", "MAE"):
            norm[:, col_i] = 1 - (norm[:, col_i] - lo) / (hi - lo)
        else:
            norm[:, col_i] = (norm[:, col_i] - lo) / (hi - lo)

    fig, ax = plt.subplots(figsize=(13, max(5, len(model_names))))
    im = ax.imshow(norm, cmap=plt.get_cmap("YlGn"), aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(6))
    ax.set_xticklabels(metrics_display, fontsize=10)
    ax.set_yticks(range(len(model_names)))
    ax.set_yticklabels(model_names, fontsize=10)
    ax.set_title("Tuned Model Performance Heatmap\n(green = better, per-column normalised)",
                 fontsize=13, fontweight="bold")

    for i in range(len(model_names)):
        for j, key in enumerate(metric_keys):
            raw  = raw_values[i, j]
            text = f"{raw:.3f}" if key == "R2" else f"{raw:.1f}"
            ax.text(j, i, text, ha="center", va="center", fontsize=9,
                    color="black" if norm[i, j] > 0.45 else "white",
                    fontweight="bold")

    plt.colorbar(im, ax=ax, label="Relative performance (1=best)", shrink=0.8)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "12_tuned_heatmap.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  saved 12_tuned_heatmap.png")


if __name__ == "__main__":
    import time as _time
    _t0_total = _time.time()

    def _elapsed():
        m, s = divmod(int(_time.time() - _t0_total), 60)
        return f"[{m:02d}:{s:02d}]"

    print(f"{_elapsed()} --- Preprocessing ---")
    X, y, encoder, df = run_preprocessing(DATA_DIR, SAVE_DIR)
    groups = df["journey_id"].values
    print(f"  {len(df):,} rows | {df['journey_id'].nunique():,} journeys")

    imputer = SimpleImputer(strategy="median")
    X = imputer.fit_transform(X)

    print("\n--- Train / Validation / Test Split (70 / 15 / 15) ---")
    # Step 1: carve out 15% test
    gss_test = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=14)
    trainval_idx, test_idx = next(gss_test.split(X, y, groups=groups))
    # Step 2: from the remaining 85%, carve out ~17.6% as val → ~15% of total
    gss_val = GroupShuffleSplit(n_splits=1, test_size=0.176, random_state=14)
    rel_train_idx, rel_val_idx = next(
        gss_val.split(X[trainval_idx], y[trainval_idx], groups=groups[trainval_idx])
    )
    train_idx = trainval_idx[rel_train_idx]
    val_idx   = trainval_idx[rel_val_idx]

    X_train, y_train = X[train_idx], y[train_idx]
    X_val,   y_val   = X[val_idx],   y[val_idx]
    X_test,  y_test  = X[test_idx],  y[test_idx]
    g_train = groups[train_idx]
    print(f"  Train: {len(X_train):,}  |  Val: {len(X_val):,}  |  Test: {len(X_test):,}")

    cv = GroupKFold(n_splits=3)

    # ── Baseline models ────────────────────────────────────────────────────────
    print(f"\n{_elapsed()} --- Baseline Models (1/5) ---")
    baseline_models = {
        "Ridge": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  Ridge(alpha=1.0)),
        ]),
        "Random Forest": RandomForestRegressor(
            n_estimators=100, max_depth=10, random_state=14, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=100, max_depth=4, learning_rate=0.05, random_state=14
        ),
        "XGBoost": xgb.XGBRegressor(
            n_estimators=100, max_depth=6, learning_rate=0.05,
            random_state=14, n_jobs=-1, verbosity=0
        ),
        "LightGBM": lgb.LGBMRegressor(
            n_estimators=100, max_depth=6, learning_rate=0.05,
            random_state=14, n_jobs=-1, verbose=-1
        ),
        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  KNeighborsRegressor(n_neighbors=10)),
        ]),
        "MLP": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  MLPRegressor(
                hidden_layer_sizes=(128, 64),
                max_iter=100,
                early_stopping=True,
                random_state=14,
            )),
        ]),
    }

    baseline_results = {}
    n_baselines = len(baseline_models)
    for i, (name, model) in enumerate(baseline_models.items(), 1):
        print(f"  {_elapsed()} [{i}/{n_baselines}] {name}...", end=" ", flush=True)
        _t0 = _time.time()
        res = evaluate(model, X_train, y_train, X_test, y_test)
        baseline_results[name] = res
        print(f"MAE={res['MAE']:.3f}  RMSE={res['RMSE']:.3f}  R²={res['R2']:.4f}  ({_time.time()-_t0:.0f}s)")

    # ── Hyperparameter tuning ──────────────────────────────────────────────────
    print(f"\n{_elapsed()} --- Hyperparameter Tuning (2/5) ---")
    search_configs = {
        "Ridge": {
            "estimator": Pipeline([("scaler", StandardScaler()), ("model", Ridge(solver="lsqr"))]),
            "param_dist": {
                "model__alpha":  [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
            },
            "n_iter": 6,
        },
        "Random Forest": {
            "estimator": RandomForestRegressor(random_state=14, n_jobs=1),
            "param_dist": {
                "n_estimators": [50, 100, 200],
                "max_depth":    [5, 10, 20],
                "max_features": ["sqrt", "log2", 0.5],
            },
            "n_iter": 6,
        },
        "Gradient Boosting": {
            "estimator": GradientBoostingRegressor(random_state=14),
            "param_dist": {
                "n_estimators":  [50, 100, 200],
                "max_depth":     [3, 4, 5],
                "learning_rate": [0.01, 0.05, 0.1],
            },
            "n_iter": 6,
        },
        "XGBoost": {
            "estimator": xgb.XGBRegressor(random_state=14, n_jobs=1, verbosity=0),
            "param_dist": {
                "max_depth":     [4, 6, 8],
                "learning_rate": [0.01, 0.05, 0.1],
                "subsample":     [0.7, 0.8, 1.0],
            },
            "n_iter": 6,
        },
        "LightGBM": {
            "estimator": lgb.LGBMRegressor(random_state=14, n_jobs=1, verbose=-1),
            "param_dist": {
                "num_leaves":    [31, 63, 127],
                "learning_rate": [0.01, 0.05, 0.1],
                "subsample":     [0.7, 0.8, 1.0],
            },
            "n_iter": 6,
        },
        "KNN": {
            "estimator": Pipeline([
                ("scaler", StandardScaler()),
                ("model",  KNeighborsRegressor()),
            ]),
            "param_dist": {
                "model__n_neighbors": [5, 10, 20, 50],
                "model__weights":     ["uniform", "distance"],
                "model__metric":      ["euclidean", "manhattan"],
            },
            "n_iter": 6,
        },
        "MLP": {
            "estimator": Pipeline([
                ("scaler", StandardScaler()),
                ("model",  MLPRegressor(max_iter=100, early_stopping=True, random_state=14)),
            ]),
            "param_dist": {
                "model__hidden_layer_sizes": [(64,), (128, 64)],
                "model__learning_rate_init": [0.001, 0.01],
            },
            "n_iter": 4,
        },
    }

    tuned_models  = {}
    tuned_results = {}
    best_params   = {}
    n_configs = len(search_configs)

    for i, (name, cfg) in enumerate(search_configs.items(), 1):
        print(f"\n  {_elapsed()} [{i}/{n_configs}] Tuning {name} ({cfg['n_iter']} iters × 3-fold CV)...")
        _t0 = _time.time()
        search = RandomizedSearchCV(
            estimator           = cfg["estimator"],
            param_distributions = cfg["param_dist"],
            n_iter              = cfg["n_iter"],
            cv                  = cv,
            scoring             = "neg_mean_absolute_error",
            random_state        = 14,
            n_jobs              = -1,
            verbose             = 0,
            refit               = True,
        )
        search.fit(X_train, y_train, groups=g_train)
        best_params[name] = search.best_params_
        print(f"    Best params: {search.best_params_}  ({_time.time()-_t0:.0f}s)")

        tuned_model = search.best_estimator_
        tuned_models[name] = tuned_model
        res = evaluate(tuned_model, X_train, y_train, X_test, y_test, fit=False)
        tuned_results[name] = res
        improvement = baseline_results[name]["MAE"] - res["MAE"]
        print(f"    MAE={res['MAE']:.3f}  RMSE={res['RMSE']:.3f}  R²={res['R2']:.4f}  "
              f"(ΔMAE={improvement:+.3f})")

        slug = name.lower().replace(" ", "_")
        model_path = os.path.join(SAVE_DIR, f"{slug}.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(tuned_model, f)
        print(f"    Saved → {model_path}")

    with open(os.path.join(SAVE_DIR, "best_params.json"), "w") as f:
        json.dump(best_params, f, indent=2, default=str)
    print(f"\n  Best params saved to {SAVE_DIR}/best_params.json")

    # ── Pick best single model and save as best_<model>.pkl (3/5) ──────────────
    winner      = min(tuned_results, key=lambda n: tuned_results[n]["MAE"])
    winner_slug = winner.lower().replace(" ", "_")
    best_path   = os.path.join(SAVE_DIR, f"best_{winner_slug}.pkl")
    with open(best_path, "wb") as f:
        pickle.dump(tuned_models[winner], f)
    print(f"\n  ✓ Best single model : {winner}")
    print(f"    MAE={tuned_results[winner]['MAE']:.3f}  "
          f"R²={tuned_results[winner]['R2']:.4f}")
    print(f"    Saved as best_{winner_slug}.pkl → {best_path}")

    # ── Weighted Ensemble (WE) ─────────────────────────────────────────────────
    print(f"\n{_elapsed()} --- Weighted Ensemble (3/5) ---")
    val_r2 = {}
    for name, model in tuned_models.items():
        val_r2[name] = float(r2_score(y_val, model.predict(X_val)))
        print(f"  {name:<20} val R²={val_r2[name]:.4f}")

    # Clip negative R² to 0 so a bad model doesn't pull predictions the wrong way
    weights    = {name: max(v, 0.0) for name, v in val_r2.items()}
    total_w    = sum(weights.values()) or 1.0  # guard against all-zero edge case
    norm_w     = {name: w / total_w for name, w in weights.items()}

    y_pred_we = sum(
        norm_w[name] * tuned_models[name].predict(X_test)
        for name in tuned_models
    )
    we_mae    = float(mean_absolute_error(y_test, y_pred_we))
    we_rmse   = float(np.sqrt(mean_squared_error(y_test, y_pred_we)))
    we_r2     = float(r2_score(y_test, y_pred_we))
    we_w2     = float(np.mean(np.abs(y_pred_we - y_test) <= 2)  * 100)
    we_w5     = float(np.mean(np.abs(y_pred_we - y_test) <= 5)  * 100)
    we_w10    = float(np.mean(np.abs(y_pred_we - y_test) <= 10) * 100)

    print(f"\n  Weighted Ensemble  MAE={we_mae:.3f}  RMSE={we_rmse:.3f}  R²={we_r2:.4f}")
    print(f"  Within 2/5/10 min: {we_w2:.1f}% / {we_w5:.1f}% / {we_w10:.1f}%")
    print(f"  Weights: { {n: f'{w:.3f}' for n, w in norm_w.items()} }")

    ensemble_path = os.path.join(SAVE_DIR, "weighted_ensemble.pkl")
    with open(ensemble_path, "wb") as f:
        pickle.dump({"models": tuned_models, "weights": norm_w}, f)
    print(f"  Saved → {ensemble_path}")

    # ── Plots ──────────────────────────────────────────────────────────────────
    print(f"\n{_elapsed()} --- Plots (4/5) ---")
    model_names = list(baseline_results.keys())
    plot_before_after(baseline_results, tuned_results, model_names, PLOT_DIR)
    plot_r2(baseline_results, tuned_results, model_names, PLOT_DIR)
    plot_mae_improvement(baseline_results, tuned_results, model_names, PLOT_DIR)
    plot_within_n(tuned_results, model_names, PLOT_DIR)
    plot_actual_vs_predicted(tuned_results, model_names, y_test, PLOT_DIR)
    plot_heatmap(tuned_results, model_names, PLOT_DIR)

    # ── Summary table ──────────────────────────────────────────────────────────
    print(f"\n{_elapsed()} --- Summary (5/5) ---")
    print("="*70)
    print(f"{'Model':<22} {'Base MAE':>9} {'Tuned MAE':>10} {'ΔMAE':>8} {'Tuned R²':>10}")
    print("-"*70)
    for name in model_names:
        b = baseline_results[name]["MAE"]
        t = tuned_results[name]["MAE"]
        r = tuned_results[name]["R2"]
        print(f"{name:<22} {b:>9.3f} {t:>10.3f} {t-b:>+8.3f} {r:>10.4f}")
    print("-"*70)
    print(f"{'Weighted Ensemble':<22} {'':>9} {we_mae:>10.3f} {'':>8} {we_r2:>10.4f}")
    print("="*70)
    print(f"\nAll plots saved to: {PLOT_DIR}")
    m, s = divmod(int(_time.time() - _t0_total), 60)
    print(f"\nDone in {m}m {s}s.")
