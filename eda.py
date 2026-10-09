"""Exploratory data analysis for the protection-score task.

Reads only the labeled training file and the unlabeled open test file (for
the missing-value and distribution comparison). Writes a Markdown summary and
PNG plots to ``reports/eda``.

Example:
    python eda.py --train-csv hard_train.csv --test-csv hard_test.csv --output-dir reports/eda
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402
from scipy.special import expit, logit  # noqa: E402
from sklearn.linear_model import RidgeCV  # noqa: E402
from sklearn.model_selection import KFold  # noqa: E402

from model_utils import ID_COLUMN, TARGET_COLUMN  # noqa: E402

CATEGORICAL = ["gender", "region", "city_type", "education", "family_status", "employment", "wealth_segment"]
INSURANCE_FLAGS = [
    "life_insurance",
    "property_insurance",
    "health_insurance",
    "travel_insurance",
    "car_insurance",
    "gadget_insurance",
    "cyber_protection",
    "identity_protection",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", default="hard_train.csv")
    parser.add_argument("--test-csv", default="hard_test.csv")
    parser.add_argument("--output-dir", default="reports/eda")
    parser.add_argument("--seed", type=int, default=0, help="Seed for the CV folds used in the noise diagnostics")
    return parser.parse_args()


def missing_table(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    table = pd.DataFrame(
        {
            "train_missing_share": train.isna().mean(),
            "test_missing_share": test.reindex(columns=train.columns).isna().mean(),
        }
    )
    return table[(table > 0).any(axis=1)].round(4)


def consistency_checks(train: pd.DataFrame) -> list[str]:
    lines = []
    flags_sum = train[INSURANCE_FLAGS].sum(axis=1)
    lines.append(f"- `insurance_products` equals the sum of the 8 insurance flags: {(flags_sum == train['insurance_products']).mean():.2%} of rows")
    years = train["years_with_company"]
    freq = np.where(years > 0, train["number_of_claims"] / years.clip(lower=1), 0.0)
    lines.append(
        "- `claim_frequency` equals `number_of_claims / years_with_company` (4-decimal rounding): "
        f"{np.isclose(freq, train['claim_frequency'], atol=6e-5).mean():.2%} of rows"
    )
    lines.append(
        "- `average_claim_cost` is missing exactly when `number_of_claims == 0`: "
        f"{((train['average_claim_cost'].isna()) == (train['number_of_claims'] == 0)).mean():.2%} of rows"
    )
    lines.append(
        "- `house_value` is missing for "
        f"{train['house_value'].isna().mean():.1%} of rows (dataset card: ~62%); `car_value` for "
        f"{train['car_value'].isna().mean():.1%} (~58%)"
    )
    segments = train.groupby("wealth_segment")["income"].agg(["min", "max"]).round(0)
    lines.append("- `wealth_segment` vs `income` ranges:\n\n" + segments.to_markdown())
    lines.append(f"- Duplicated feature rows in train: {train.drop(columns=[ID_COLUMN, TARGET_COLUMN]).duplicated().sum()}")
    return lines


def logit_noise_diagnostics(train: pd.DataFrame, seed: int, output_dir: Path) -> list[str]:
    """Check whether the target looks like 100*sigmoid(linear + Gaussian noise)."""
    y = train[TARGET_COLUMN].to_numpy(dtype=float)
    z = logit(np.clip(y / 100.0, 1e-4, 1 - 1e-4))
    features = pd.get_dummies(train.drop(columns=[ID_COLUMN, TARGET_COLUMN]), drop_first=True).astype(float)
    features = features.fillna(features.median())

    oof_logit = np.zeros(len(y))
    for train_idx, valid_idx in KFold(5, shuffle=True, random_state=seed).split(features):
        model = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(features.iloc[train_idx], z[train_idx])
        oof_logit[valid_idx] = model.predict(features.iloc[valid_idx])
    residual = z - oof_logit
    r2_logit = 1.0 - residual.var() / z.var()
    rmse_raw = float(np.sqrt(np.mean((100.0 * expit(oof_logit) - y) ** 2)))

    bins = pd.qcut(oof_logit, 10, labels=False)
    raw_residual = y - 100.0 * expit(oof_logit)
    by_bin = pd.DataFrame(
        {
            "decile": bins,
            "logit_resid": residual,
            "raw_resid": raw_residual,
        }
    ).groupby("decile").agg(logit_resid_std=("logit_resid", "std"), raw_resid_std=("raw_resid", "std"))

    # Expected RMSE if the noise is N(0, sigma) in logit space around the fitted index.
    rng = np.random.default_rng(seed)
    sigma = float(residual.std())
    reference = 100.0 * expit(oof_logit)

    def expected_rmse(noise_sd: float, draws: int = 20) -> float:
        values = []
        for _ in range(draws):
            noisy = 100.0 * expit(oof_logit + rng.normal(0.0, noise_sd, len(y)))
            values.append(np.sqrt(np.mean((noisy - reference) ** 2)))
        return float(np.mean(values))

    floor = expected_rmse(sigma)
    sigma_grid = [0.2, 0.3, 0.4, 0.5, sigma]
    floor_table = pd.DataFrame(
        {"logit_noise_sd": sigma_grid, "expected_rmse": [expected_rmse(v) for v in sigma_grid]}
    ).round(3)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].scatter(oof_logit, residual, s=3, alpha=0.4)
    axes[0].axhline(0, color="k", lw=0.8)
    axes[0].set_title("Logit scale: residual vs fitted index")
    axes[0].set_xlabel("out-of-fold linear index")
    axes[0].set_ylabel("logit(target) residual")
    axes[1].scatter(100.0 * expit(oof_logit), raw_residual, s=3, alpha=0.4, color="tab:orange")
    axes[1].axhline(0, color="k", lw=0.8)
    axes[1].set_title("Raw 0-100 scale: residual vs prediction")
    axes[1].set_xlabel("prediction")
    axes[1].set_ylabel("target residual")
    fig.tight_layout()
    fig.savefig(output_dir / "noise_logit_vs_raw.png", dpi=120)
    plt.close(fig)

    lines = [
        f"- Logit target `z = logit(protection_score/100)`: std {z.std():.3f}, range [{z.min():.2f}, {z.max():.2f}]",
        f"- Out-of-fold ridge R^2 on the logit scale: {r2_logit:.3f}",
        f"- Logit residual: std {residual.std():.3f}, skew {stats.skew(residual):.3f}, excess kurtosis {stats.kurtosis(residual):.3f}",
        "- Residual std by decile of the fitted index (logit scale is roughly constant, raw scale is not):",
        "",
        by_bin.round(3).to_markdown(),
        "",
        f"- Out-of-fold RMSE of the linear logit model on the 0-100 scale: {rmse_raw:.3f}",
        f"- Expected RMSE from the logit-scale noise alone (sigma={sigma:.3f}): {floor:.2f}",
        "- Expected RMSE for other logit-noise levels (a model that recovers the true index would still be at this level):",
        "",
        floor_table.to_markdown(index=False),
        "",
        "Reaching RMSE < 7 on this data would require the logit-scale noise to be well below the observed ~0.6,",
        "i.e. the remaining signal must be explained by information that is not in the given features.",
    ]
    return lines


def make_plots(train: pd.DataFrame, output_dir: Path) -> None:
    y = train[TARGET_COLUMN]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(y, bins=60, color="tab:blue")
    axes[0].set_title("protection_score distribution")
    axes[0].set_xlabel("protection_score")
    axes[1].hist(logit(np.clip(y / 100.0, 1e-4, 1 - 1e-4)), bins=60, color="tab:green")
    axes[1].set_title("logit(protection_score / 100) distribution")
    axes[1].set_xlabel("logit")
    fig.tight_layout()
    fig.savefig(output_dir / "target_distribution.png", dpi=120)
    plt.close(fig)

    numeric = train.select_dtypes("number").drop(columns=[TARGET_COLUMN])
    correlations = numeric.corrwith(y).sort_values(key=np.abs, ascending=False).head(20)
    fig, ax = plt.subplots(figsize=(8, 6))
    correlations[::-1].plot.barh(ax=ax, color=["tab:red" if v < 0 else "tab:blue" for v in correlations[::-1]])
    ax.set_title("Top-20 Pearson correlations with protection_score")
    fig.tight_layout()
    fig.savefig(output_dir / "top_correlations.png", dpi=120)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    train.boxplot(column=TARGET_COLUMN, by="insurance_products", ax=axes[0])
    axes[0].set_title("protection_score by insurance_products")
    axes[0].set_xlabel("insurance_products")
    train.boxplot(column=TARGET_COLUMN, by="wealth_segment", ax=axes[1])
    axes[1].set_title("protection_score by wealth_segment")
    fig.suptitle("")
    fig.tight_layout()
    fig.savefig(output_dir / "target_by_groups.png", dpi=120)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(args.train_csv)
    test = pd.read_csv(args.test_csv)
    y = train[TARGET_COLUMN]

    lines: list[str] = ["# EDA: protection_score", ""]
    lines += [
        "## Shapes",
        f"- train: {train.shape[0]} rows x {train.shape[1]} columns",
        f"- open test: {test.shape[0]} rows x {test.shape[1]} columns",
        "",
        "## Target",
        y.describe().round(3).to_markdown(),
        "",
        "## Missing values (share of rows)",
        missing_table(train.drop(columns=[TARGET_COLUMN]), test).to_markdown(),
        "",
        "## Categorical features: target mean by level",
    ]
    for column in CATEGORICAL:
        group = train.groupby(column)[TARGET_COLUMN].agg(["mean", "std", "count"]).round(2)
        lines += [f"### {column}", group.to_markdown(), ""]
    lines += ["## Consistency checks", *consistency_checks(train), ""]
    lines += ["## Top correlations with the target (Pearson)"]
    numeric = train.select_dtypes("number").drop(columns=[TARGET_COLUMN])
    correlations = numeric.corrwith(y).sort_values(key=np.abs, ascending=False).round(3).rename("pearson_r")
    lines += [correlations.head(20).to_frame().to_markdown(), ""]
    lines += ["## Target noise structure"]
    lines += logit_noise_diagnostics(train, args.seed, output_dir)
    lines += [
        "",
        "## Outliers",
        "Non-binary numeric features were checked with the 1.5*IQR rule; the counts are informational only, "
        "no rows were removed because the residuals on the logit scale are close to Gaussian.",
    ]
    outlier_counts = {}
    for column in numeric.columns:
        if numeric[column].nunique() <= 2:  # binary flags have no meaningful IQR outliers
            continue
        q1, q3 = numeric[column].quantile([0.25, 0.75])
        iqr = q3 - q1
        count = int(((numeric[column] < q1 - 1.5 * iqr) | (numeric[column] > q3 + 1.5 * iqr)).sum())
        if count:
            outlier_counts[column] = count
    lines += [pd.Series(outlier_counts, name="outliers").sort_values(ascending=False).to_frame().to_markdown(), ""]

    make_plots(train, output_dir)
    (output_dir / "eda_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {output_dir / 'eda_summary.md'} and plots to {output_dir}")


if __name__ == "__main__":
    main()
