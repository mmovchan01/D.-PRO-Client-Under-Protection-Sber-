"""Check whether the numeric part of customer_id carries signal about protection_score.

Steps:
  1. extract the digits of customer_id (e.g. CUST100123 -> 100123) as int;
  2. plot protection_score against the numeric ID and save the figure;
  3. Pearson and Spearman correlations;
  4. models that use only the numeric ID, compared with a constant baseline:
     - 5-fold shuffled CV (the same kind of split as in experiments.py);
     - "extrapolation" check: train on the first 80% of IDs, validate on the last 20%,
       which mirrors the test set (its IDs lie above all training IDs).

Example:
    python id_analysis.py --train-csv hard_train.csv --output-dir reports/id_analysis
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

from model_utils import ID_COLUMN, TARGET_COLUMN, set_global_seed

SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-csv", default="hard_train.csv")
    parser.add_argument("--output-dir", default="reports/id_analysis")
    return parser.parse_args()


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def main() -> None:
    args = parse_args()
    set_global_seed(SEED)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(args.train_csv)
    digits = data[ID_COLUMN].astype(str).str.extract(r"(\d+)", expand=False)
    if digits.isna().any():
        raise ValueError(f"{ID_COLUMN} values without digits were found")
    data["id_num"] = digits.astype(int)
    if data["id_num"].duplicated().any():
        raise ValueError("Numeric IDs are not unique")

    x = data[["id_num"]].to_numpy(dtype=float)
    y = data[TARGET_COLUMN].to_numpy(dtype=float)
    order = np.argsort(data["id_num"].to_numpy())
    print(f"Numeric ID range: {data['id_num'].min()} .. {data['id_num'].max()}, unique: {data['id_num'].nunique()}")
    print(f"Is the file sorted by ID: {bool((np.diff(data['id_num'].to_numpy()) > 0).all())}")

    # 1. Plot.
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    axes[0].scatter(data["id_num"], y, s=4, alpha=0.4)
    rolling = pd.Series(y[order], index=data["id_num"].to_numpy()[order]).rolling(200, center=True, min_periods=50).mean()
    axes[0].plot(rolling.index, rolling.values, color="crimson", lw=2, label="rolling mean (200 rows)")
    axes[0].set_xlabel("numeric customer_id")
    axes[0].set_ylabel("protection_score")
    axes[0].set_title("protection_score vs numeric ID")
    axes[0].legend()

    # Mean target in consecutive blocks of 500 IDs; a flat profile means no trend.
    block = (data["id_num"] - data["id_num"].min()) // 500
    block_means = data.groupby(block)[TARGET_COLUMN].agg(["mean", "std"])
    axes[1].errorbar(block_means.index * 500, block_means["mean"], yerr=block_means["std"] / np.sqrt(500),
                     fmt="o-", ms=3, capsize=2)
    axes[1].set_xlabel("block start (500 consecutive IDs)")
    axes[1].set_ylabel("mean protection_score (±1 s.e.)")
    axes[1].set_title("Mean target by ID block")
    fig.tight_layout()
    fig.savefig(out / "id_vs_protection_score.png", dpi=130)
    plt.close(fig)

    # 2. Correlations.
    pearson_r, pearson_p = pearsonr(data["id_num"], y)
    spearman_r, spearman_p = spearmanr(data["id_num"], y)
    print(f"Pearson r = {pearson_r:.4f} (p = {pearson_p:.3g})")
    print(f"Spearman rho = {spearman_r:.4f} (p = {spearman_p:.3g})")

    # Autocorrelation of target in file order (used to check block structure).
    lag1 = float(np.corrcoef(y[:-1], y[1:])[0, 1])
    print(f"Lag-1 autocorrelation of target in ID order: {lag1:.4f}")

    # 3. Models on the numeric ID only.
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    models = {
        "linear regression": make_pipeline(StandardScaler(), LinearRegression()),
        "spline (cubic, 8 knots) + linear": make_pipeline(
            StandardScaler(), SplineTransformer(n_knots=8, degree=3), LinearRegression()
        ),
        "kNN (k=200)": make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=200)),
        "gradient boosting": GradientBoostingRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=3, subsample=0.8, random_state=SEED
        ),
    }

    results = []
    # K-fold: the same rows in train and validation, as in experiments.py.
    oof_const = np.zeros(len(y))
    for tr, va in kf.split(x):
        oof_const[va] = y[tr].mean()
    results.append({"model": "constant mean", "cv5_rmse": rmse(oof_const, y)})
    for name, model in models.items():
        pred = cross_val_predict(model, x, y, cv=kf)
        results.append({"model": name, "cv5_rmse": rmse(pred, y)})

    # Temporal split: the validation block lies above all training IDs (like the test set).
    cut = int(0.8 * len(data))
    train_idx, valid_idx = order[:cut], order[cut:]
    temporal = [
        {"model": "constant mean", "tail_rmse": rmse(np.full(len(valid_idx), y[train_idx].mean()), y[valid_idx])}
    ]
    for name, model in models.items():
        model.fit(x[train_idx], y[train_idx])
        temporal.append({"model": name, "tail_rmse": rmse(model.predict(x[valid_idx]), y[valid_idx])})
    temporal_df = pd.DataFrame(temporal)

    cv_df = pd.DataFrame(results)
    summary = cv_df.merge(temporal_df, on="model", how="outer")
    print("\nModels on the numeric ID only (RMSE on the 0-100 scale):")
    print(summary.round(3).to_string(index=False))
    summary.to_csv(out / "id_only_models.csv", index=False)

    # Save the numbers used in the report.
    stats = {
        "id_min": int(data["id_num"].min()),
        "id_max": int(data["id_num"].max()),
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "spearman_rho": float(spearman_r),
        "spearman_p": float(spearman_p),
        "lag1_autocorrelation": lag1,
        "target_std": float(y.std()),
        "seed": SEED,
    }
    with (out / "id_stats.json").open("w", encoding="utf-8") as file:
        json.dump(stats, file, indent=2)
        file.write("\n")
    print(f"\nSaved: {out / 'id_vs_protection_score.png'}, {out / 'id_only_models.csv'}, {out / 'id_stats.json'}")


if __name__ == "__main__":
    main()
