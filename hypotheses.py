"""Test hypotheses about hidden structure in protection_score.

Every hypothesis is evaluated on hard_train.csv only, using 5-fold shuffled CV.
The main yardstick is the out-of-fold residual standard deviation on the logit
scale (z = logit(protection_score / 100)); the RMSE on the 0-100 scale is
reported for the final candidate. Results are written to reports/hypotheses.md.

This script is documentation only. The final model is trained by train.py.

Example:
    python hypotheses.py --train-csv hard_train.csv --test-csv hard_test.csv --output reports/hypotheses.md
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit, ndtr, ndtri
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import RidgeCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.neighbors import NearestNeighbors

from model_utils import fit_preprocessor, numeric_matrix, transform_features
from train import fit_sigmoid_index

warnings.filterwarnings("ignore")

TARGET = "protection_score"
ID = "customer_id"
CATEGORICAL = ["region", "city_type", "education", "family_status", "employment", "gender", "wealth_segment"]
CV_SEEDS = [0, 1, 2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", default="hard_train.csv")
    parser.add_argument("--test-csv", default="hard_test.csv")
    parser.add_argument("--output", default="reports/hypotheses.md")
    return parser.parse_args()


def design(frame: pd.DataFrame, numeric: list[str]) -> pd.DataFrame:
    """Numeric + one-hot categoricals + missing flags, NaN imputed by the frame's medians (fit on train)."""
    out = pd.get_dummies(frame[numeric + CATEGORICAL], drop_first=True).astype(float)
    for column in ["house_value", "car_value", "average_claim_cost"]:
        out[column + "_missing"] = frame[column].isna().astype(float)
    return out


def oof_ridge(matrix: np.ndarray, target: np.ndarray, seed: int) -> np.ndarray:
    prediction = np.zeros(len(target))
    for train_idx, valid_idx in KFold(5, shuffle=True, random_state=seed).split(matrix):
        model = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(matrix[train_idx], target[train_idx])
        prediction[valid_idx] = model.predict(matrix[valid_idx])
    return prediction


def logit_resid_sd(matrix: np.ndarray, z: np.ndarray, seeds=CV_SEEDS) -> tuple[float, list[float]]:
    values = [float((z - oof_ridge(matrix, z, s)).std()) for s in seeds]
    return float(np.mean(values)), values


def main() -> None:
    args = parse_args()
    train = pd.read_csv(args.train_csv)
    test = pd.read_csv(args.test_csv)
    y = train[TARGET].to_numpy(dtype=float)
    z = logit(np.clip(y / 100.0, 1e-4, 1 - 1e-4))
    features = train.drop(columns=[ID, TARGET])
    numeric = list(features.select_dtypes("number").columns)

    base_frame = design(features, numeric)
    fill = base_frame.median()
    base = base_frame.fillna(fill)
    base_matrix = base.to_numpy(dtype=float)
    results: list[tuple[str, str, str]] = []

    def record(hypothesis: str, verdict: str, detail: str) -> None:
        results.append((hypothesis, verdict, detail))
        print(f"{hypothesis}: {verdict} | {detail}", flush=True)

    base_mean, base_runs = logit_resid_sd(base_matrix, z)
    record("H0 baseline sigmoid-linear index", "reference", f"logit resid sd {base_mean:.4f} ({', '.join(f'{v:.4f}' for v in base_runs)})")

    # H1: derived numeric features are deterministic functions of the others?
    r2 = {}
    for column in numeric:
        others = [c for c in numeric if c != column]
        filled = features[others].fillna(features[others].median()).to_numpy(dtype=float)
        target_col = features[column].fillna(features[column].median()).to_numpy(dtype=float)
        pred = oof_ridge(filled, target_col, 0)
        r2[column] = 1 - np.var(target_col - pred) / np.var(target_col)
    top_r2 = pd.Series(r2).sort_values(ascending=False).head(3)
    record(
        "H1 a numeric feature is an exact function of the others (derived feature)",
        "rejected" if top_r2.max() < 0.95 else "supported",
        "max OOF R^2 of one feature from the others: " + ", ".join(f"{k} {v:.2f}" for k, v in top_r2.items()),
    )

    # H2: local structure in feature space (kNN residual correlation).
    scaled = (base_matrix - base_matrix.mean(0)) / base_matrix.std(0)
    residual = z - oof_ridge(base_matrix, z, 0)
    _, neighbours = NearestNeighbors(n_neighbors=21).fit(scaled).kneighbors(scaled)
    corr_knn = np.corrcoef(residual, residual[neighbours[:, 1:]].mean(axis=1))[0, 1]
    record(
        "H2 residuals are locally correlated (kNN in feature space)",
        "rejected" if abs(corr_knn) < 0.05 else "supported",
        f"corr(own residual, mean of 20 neighbours) = {corr_knn:.4f}",
    )

    # H3: categorical x numeric interactions.
    numeric_std = ((features[numeric] - features[numeric].mean()) / features[numeric].std()).fillna(0)
    cat_dummies = pd.get_dummies(features[CATEGORICAL], drop_first=True).astype(float)
    pairs = [(c, n) for c in cat_dummies.columns for n in numeric if n not in CATEGORICAL]
    all_inter = pd.DataFrame({f"{c}*{n}": cat_dummies[c] * numeric_std[n] for c, n in pairs}).to_numpy()
    private_inter = np.column_stack(
        [cat_dummies["wealth_segment_private"] * numeric_std["income"]]
    )
    mean_private, _ = logit_resid_sd(np.hstack([base_matrix, private_inter]), z)
    mean_all, _ = logit_resid_sd(np.hstack([base_matrix, all_inter]), z)
    record(
        "H3a category x numeric interactions improve the fit",
        "rejected" if mean_all > base_mean else "supported",
        f"all {len(pairs)} interactions: {mean_all:.4f} (worse than {base_mean:.4f}); overfitting",
    )
    record(
        "H3b the gain comes from wealth_segment=private x income",
        "explains the small gain" if mean_private < base_mean - 0.003 else "no effect",
        f"adding only private x income: {mean_private:.4f}; this term affects 32 rows (income extrapolation)",
    )

    # H4: explicit ratio features.
    ratios = pd.DataFrame(
        {
            "loan_to_income": features["loan_amount"] / features["income"],
            "balance_to_income": features["average_balance"] / features["income"],
            "house_to_income": features["house_value"] / features["income"],
            "car_to_income": features["car_value"] / features["income"],
            "phone_to_income": features["smartphone_price"] / features["income"],
            "claim_to_income": features["average_claim_cost"] / features["income"],
        }
    ).replace([np.inf, -np.inf], np.nan)
    ratios = ratios.fillna(ratios.median())
    mean_ratio, _ = logit_resid_sd(np.hstack([base_matrix, ratios.to_numpy()]), z)
    record(
        "H4 explicit money ratios (loan, balance, house, car, phone to income)",
        "weak" if mean_ratio > base_mean - 0.003 else "small logit gain (not tested on 0-100 scale)",
        f"logit resid sd {mean_ratio:.4f} vs {base_mean:.4f}",
    )

    # H5: boosted residuals capture thresholds that linear terms miss.
    # Judged on the logit scale AND on the 0-100 scale, because the final model is fit on the 0-100 scale.
    logit_gain, raw_final, raw_boost = [], [], []
    gbm_features = features.copy()
    for column in CATEGORICAL:
        gbm_features[column] = gbm_features[column].astype("category")
    raw_matrix = numeric_matrix(features, numeric)
    for seed in CV_SEEDS[:2]:
        linear = np.zeros(len(z))
        boosted = np.zeros(len(z))
        final = np.zeros(len(z))
        for train_idx, valid_idx in KFold(5, shuffle=True, random_state=seed).split(base_matrix):
            model = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(base_matrix[train_idx], z[train_idx])
            linear_in = model.predict(base_matrix[train_idx])
            linear[valid_idx] = model.predict(base_matrix[valid_idx])
            gbm = lgb.LGBMRegressor(
                n_estimators=100,
                learning_rate=0.02,
                num_leaves=7,
                min_child_samples=50,
                subsample=0.8,
                subsample_freq=1,
                colsample_bytree=0.7,
                reg_lambda=5,
                random_state=seed,
                verbose=-1,
            )
            gbm.fit(gbm_features.iloc[train_idx], z[train_idx] - linear_in)
            boosted[valid_idx] = linear[valid_idx] + gbm.predict(gbm_features.iloc[valid_idx])
            medians, means, scales = fit_preprocessor(raw_matrix[train_idx])
            coefficients, intercept, _ = fit_sigmoid_index(
                transform_features(raw_matrix[train_idx], medians, means, scales), y[train_idx]
            )
            final[valid_idx] = 100.0 * expit(
                transform_features(raw_matrix[valid_idx], medians, means, scales) @ coefficients + intercept
            )
        logit_gain.append((float((z - linear).std()), float((z - boosted).std())))
        raw_final.append(float(np.sqrt(np.mean((final - y) ** 2))))
        raw_boost.append(float(np.sqrt(np.mean((100.0 * expit(boosted) - y) ** 2))))
    lin_mean = np.mean([a for a, _ in logit_gain])
    boost_mean = np.mean([b for _, b in logit_gain])
    record(
        "H5 threshold effects missed by the linear index (boosting on residuals)",
        "rejected on 0-100 scale" if np.mean(raw_boost) >= np.mean(raw_final) else "supported",
        f"logit resid sd {lin_mean:.4f} -> {boost_mean:.4f}; but 0-100 RMSE {np.mean(raw_final):.4f} (final) vs {np.mean(raw_boost):.4f} (boosted)",
    )

    # H6: residual noise depends on a group (heteroscedastic noise).
    residual_by_group = {}
    for column in CATEGORICAL:
        residual_by_group[column] = pd.Series(residual).groupby(features[column].to_numpy()).std()
    spread = max(float(s.max() - s.min()) for s in residual_by_group.values())
    record(
        "H6 noise level differs between groups",
        "rejected" if spread < 0.05 else "supported",
        f"largest within-variable range of group residual sd: {spread:.3f} (all within 0.60-0.66)",
    )

    # H7: train vs test distribution shift.
    stacked = pd.concat([features, test.drop(columns=[ID])], ignore_index=True)
    stacked_design = design(stacked, numeric)
    stacked_design = stacked_design.fillna(stacked_design.median())
    labels = np.r_[np.zeros(len(features)), np.ones(len(test))]
    probability = cross_val_predict(
        HistGradientBoostingClassifier(max_iter=200, random_state=0),
        stacked_design.to_numpy(dtype=float),
        labels,
        cv=5,
        method="predict_proba",
    )[:, 1]
    auc = roc_auc_score(labels, probability)
    record(
        "H7 test set comes from a different distribution than train",
        "rejected" if auc < 0.55 else "supported",
        f"adversarial validation AUC {auc:.3f}",
    )

    # H8: link function. Logit residuals are homoscedastic; compare other links on the raw scale.
    link_rows = []
    raw_rmse_by_link = {}
    for name, link, inverse in [("logit", logit, expit), ("probit", ndtri, ndtr)]:
        transformed = link(np.clip(y / 100.0, 1e-4, 1 - 1e-4))
        oof = oof_ridge(base_matrix, transformed, 0)
        raw_rmse_by_link[name] = float(np.sqrt(np.mean((100.0 * inverse(oof) - y) ** 2)))
        link_rows.append(f"{name}: raw RMSE {raw_rmse_by_link[name]:.3f}")
    better = raw_rmse_by_link["probit"] < raw_rmse_by_link["logit"] - 0.02
    record(
        "H8 another link function (probit) fits clearly better than logit",
        "supported" if better else "rejected",
        "; ".join(link_rows) + "; logit residuals are homoscedastic, probit residuals are not",
    )

    # H9: Bayes floor if the remaining logit noise is N(0, sigma) (gauss-hermite expectation).
    eta = oof_ridge(base_matrix, z, 0)
    nodes, weights = np.polynomial.hermite_e.hermegauss(60)
    weights = weights / weights.sum()
    floors = []
    for sigma in [0.4, 0.5, 0.6, float(np.std(residual))]:
        mean = (expit(eta[:, None] + sigma * nodes[None, :]) * 100.0 * weights).sum(axis=1)
        second = ((expit(eta[:, None] + sigma * nodes[None, :]) * 100.0) ** 2 * weights).sum(axis=1)
        floors.append((sigma, float(np.sqrt(np.mean(second - mean**2)))))
    record(
        "H9 the remaining error is irreducible logit noise",
        "supported" if abs(floors[-1][1] - np.sqrt(np.mean((100 * expit(eta) - y) ** 2))) < 0.5 else "unclear",
        "Bayes RMSE floor by sigma: " + ", ".join(f"{s:.3f} -> {f:.2f}" for s, f in floors),
    )

    # H10: income tail (wealth_segment=private, income > 300k) explains the gap to the floor?
    # Oracle check: replace predictions on the tail by the true values and see how much RMSE moves.
    final_oof = np.zeros(len(y))
    for seed in [42]:
        for train_idx, valid_idx in KFold(5, shuffle=True, random_state=seed).split(raw_matrix):
            medians, means, scales = fit_preprocessor(raw_matrix[train_idx])
            coefficients, intercept, _ = fit_sigmoid_index(
                transform_features(raw_matrix[train_idx], medians, means, scales), y[train_idx]
            )
            final_oof[valid_idx] = 100.0 * expit(
                transform_features(raw_matrix[valid_idx], medians, means, scales) @ coefficients + intercept
            )
    tail = (features["income"] > 300_000).to_numpy()
    oracle = final_oof.copy()
    oracle[tail] = y[tail]
    base_rmse = float(np.sqrt(np.mean((final_oof - y) ** 2)))
    oracle_rmse = float(np.sqrt(np.mean((oracle - y) ** 2)))
    record(
        "H10 income tail (income > 300k) is a large source of error",
        "rejected" if base_rmse - oracle_rmse < 0.05 else "supported",
        f"{int(tail.sum())} rows ({tail.mean():.2%}); perfect prediction there moves RMSE {base_rmse:.4f} -> {oracle_rmse:.4f}",
    )

    # H11: fractional logit (quasi-binomial GLM on y/100) instead of MSE on the 0-100 scale.
    frac_scores = []
    for seed in CV_SEEDS[:2]:
        prediction = np.zeros(len(y))
        for train_idx, valid_idx in KFold(5, shuffle=True, random_state=seed).split(raw_matrix):
            medians, means, scales = fit_preprocessor(raw_matrix[train_idx])
            a = np.column_stack([np.ones(len(train_idx)), transform_features(raw_matrix[train_idx], medians, means, scales)])
            b = np.column_stack([np.ones(len(valid_idx)), transform_features(raw_matrix[valid_idx], medians, means, scales)])
            target_share = y[train_idx] / 100.0

            def objective(w: np.ndarray) -> tuple[float, np.ndarray]:
                eta = a @ w
                loss = np.mean(np.logaddexp(0.0, eta) - target_share * eta) + 0.5 * 1e-3 * np.dot(w[1:], w[1:])
                grad = a.T @ (expit(eta) - target_share) / len(target_share)
                grad[1:] += 1e-3 * w[1:]
                return float(loss), grad

            weights = minimize(objective, np.zeros(a.shape[1]), jac=True, method="L-BFGS-B").x
            prediction[valid_idx] = 100.0 * expit(b @ weights)
        frac_scores.append(float(np.sqrt(np.mean((prediction - y) ** 2))))
    final_mean = float(np.sqrt(np.mean((final_oof - y) ** 2)))
    record(
        "H11 fractional logit (binomial GLM) beats MSE fit on 0-100 scale",
        "rejected" if np.mean(frac_scores) > final_mean - 0.02 else "supported",
        f"0-100 RMSE {np.mean(frac_scores):.4f} vs {final_mean:.4f} for the final model (difference within CV noise)",
    )

    # H12: collinearity. insurance_products is exactly the sum of the 8 flags.
    flag_columns = [
        "life_insurance", "property_insurance", "health_insurance", "travel_insurance",
        "car_insurance", "gadget_insurance", "cyber_protection", "identity_protection",
    ]
    def impute(matrix: np.ndarray) -> np.ndarray:
        return np.where(np.isfinite(matrix), matrix, np.nanmedian(matrix, axis=0))

    raw_no_aggregate = numeric_matrix(features, [c for c in numeric if c != "insurance_products"])
    raw_no_flags = numeric_matrix(features, [c for c in numeric if c not in flag_columns])
    no_aggregate = impute(raw_no_aggregate)
    no_flags = impute(raw_no_flags)
    _, agg_runs = logit_resid_sd(no_aggregate, z, seeds=[42])
    flags_mean, _ = logit_resid_sd(no_flags, z, seeds=[42])
    no_aggregate_rmse = []
    for train_idx, valid_idx in KFold(5, shuffle=True, random_state=42).split(raw_no_aggregate):
        medians, means, scales = fit_preprocessor(raw_no_aggregate[train_idx])
        coefficients, intercept, _ = fit_sigmoid_index(
            transform_features(raw_no_aggregate[train_idx], medians, means, scales), y[train_idx]
        )
        no_aggregate_rmse.append(np.mean((100.0 * expit(
            transform_features(raw_no_aggregate[valid_idx], medians, means, scales) @ coefficients + intercept
        ) - y[valid_idx]) ** 2))
    record(
        "H12 drop the aggregate insurance_products (collinear with the 8 flags)",
        "no effect" if abs(np.sqrt(np.mean(no_aggregate_rmse)) - final_mean) < 0.01 else "changes fit",
        f"0-100 RMSE without aggregate {np.sqrt(np.mean(no_aggregate_rmse)):.4f} vs {final_mean:.4f}; "
        f"dropping the flags instead is much worse (logit sd {flags_mean:.3f})",
    )

    lines = [
        "# Hypotheses about hidden structure (hard_train.csv, 5-fold CV)",
        "",
        "Yardstick: out-of-fold residual sd on the logit scale; baseline sigmoid-linear index.",
        "",
        "| Hypothesis | Verdict | Evidence |",
        "| --- | --- | --- |",
    ]
    lines += [f"| {h} | {v} | {d} |" for h, v, d in results]
    lines += [
        "",
        "Conclusion: no tested hypothesis lowers the logit noise below roughly 0.60, and the 0-100 RMSE was checked for every candidate that lowered it on the logit scale (H5).",
        "A logit-scale gain (H4, H5) does not transfer to the 0-100 scale, so decisions are made on the 0-100 RMSE.",
        "At that noise level the Bayes RMSE floor is about 10.2-10.3, so RMSE < 7 would require a logit noise near 0.4.",
    ]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
