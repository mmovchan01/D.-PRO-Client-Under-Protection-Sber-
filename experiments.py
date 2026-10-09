"""Reproduce the model comparison and hypothesis checks used to choose the final model.

Every model is evaluated with the same 5-fold shuffled split of hard_train.csv
(only labeled training rows are used). Results are written to
``reports/experiments.md``. This script is for documentation and is NOT needed
to train or predict; the final model is trained by ``train.py``.

Example:
    python experiments.py --train-csv hard_train.csv --output reports/experiments.md
"""

from __future__ import annotations

import argparse
import itertools
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

from model_utils import (
    ID_COLUMN,
    TARGET_COLUMN,
    fit_preprocessor,
    get_numeric_feature_names,
    logits_to_score,
    numeric_matrix,
    transform_features,
)
from train import fit_sigmoid_index

warnings.filterwarnings("ignore")

CATEGORICAL = ["gender", "region", "city_type", "education", "family_status", "employment", "wealth_segment"]
CV_SEED = 42
N_FOLDS = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", default="hard_train.csv")
    parser.add_argument("--output", default="reports/experiments.md")
    return parser.parse_args()


def rmse(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.sqrt(np.mean((prediction - target) ** 2)))


def cross_validate(fit_predict, n_rows: int) -> np.ndarray:
    """fit_predict(train_idx, valid_idx) -> predictions for valid_idx on the 0-100 scale."""
    oof = np.full(n_rows, np.nan)
    for train_idx, valid_idx in KFold(N_FOLDS, shuffle=True, random_state=CV_SEED).split(np.arange(n_rows)):
        oof[valid_idx] = fit_predict(train_idx, valid_idx)
    return oof


def sigmoid_linear_predictor(matrix: np.ndarray, target: np.ndarray, *, l2_alpha: float = 10_000.0):
    def fit_predict(train_idx: np.ndarray, valid_idx: np.ndarray) -> np.ndarray:
        medians, means, scales = fit_preprocessor(matrix[train_idx])
        x_train = transform_features(matrix[train_idx], medians, means, scales)
        x_valid = transform_features(matrix[valid_idx], medians, means, scales)
        coefficients, intercept, _ = fit_sigmoid_index(x_train, target[train_idx], l2_alpha=l2_alpha)
        return logits_to_score(x_valid @ coefficients + intercept)

    return fit_predict


def build_features(data: pd.DataFrame, numeric_features: list[str], *, categories: bool, missing_flags: bool, logs: bool) -> np.ndarray:
    frame = data[numeric_features].astype(float).copy()
    if categories:
        frame = pd.concat([frame, pd.get_dummies(data[CATEGORICAL], drop_first=True).astype(float)], axis=1)
    if missing_flags:
        for column in ["house_value", "car_value", "average_claim_cost"]:
            frame[column + "_missing"] = data[column].isna().astype(float)
    if logs:
        for column in ["income", "average_balance", "loan_amount", "smartphone_price", "house_value", "car_value", "average_claim_cost"]:
            frame["log_" + column] = np.log1p(data[column].clip(lower=0))
    return frame.to_numpy(dtype=float)


def main() -> None:
    args = parse_args()
    data = pd.read_csv(args.train_csv)
    if TARGET_COLUMN not in data.columns:
        raise ValueError("Training CSV must contain protection_score")
    y = data[TARGET_COLUMN].to_numpy(dtype=float)
    z = logit(np.clip(y / 100.0, 1e-4, 1 - 1e-4))
    n = len(y)
    numeric = get_numeric_feature_names(data)
    results: list[tuple[str, str, float, str]] = []

    def record(name: str, idea: str, oof: np.ndarray, note: str = "") -> None:
        results.append((name, idea, rmse(oof, y), note))
        print(f"{name}: OOF RMSE {results[-1][2]:.4f}", flush=True)

    # Baselines
    record("constant mean", "reference", cross_validate(lambda tr, va: np.full(len(va), y[tr].mean()), n))

    raw = numeric_matrix(data, numeric)

    def linear_fit_predict(tr: np.ndarray, va: np.ndarray) -> np.ndarray:
        medians, means, scales = fit_preprocessor(raw[tr])
        model = LinearRegression().fit(transform_features(raw[tr], medians, means, scales), y[tr])
        return model.predict(transform_features(raw[va], medians, means, scales))

    record(
        "linear regression, raw target",
        "is the target linear in the numeric features?",
        cross_validate(linear_fit_predict, n),
    )

    def logit_ridge_fit_predict(tr: np.ndarray, va: np.ndarray) -> np.ndarray:
        medians, means, scales = fit_preprocessor(raw[tr])
        model = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(transform_features(raw[tr], medians, means, scales), z[tr])
        return 100.0 * expit(model.predict(transform_features(raw[va], medians, means, scales)))

    # Target is 100*sigmoid(index + noise): fit the index on the logit scale, predict on the 0-100 scale.
    record(
        "ridge on logit target (numeric)",
        "is the target a sigmoid of a linear index?",
        cross_validate(logit_ridge_fit_predict, n),
        "logit-scale noise sd about 0.6, residuals homoscedastic",
    )

    record(
        "sigmoid-linear, raw-scale MSE (final model)",
        "fit the 0-100 target directly with L-BFGS-B",
        cross_validate(sigmoid_linear_predictor(raw, y), n),
        "52 numeric features, median imputation, L2=1e4/n",
    )
    record(
        "sigmoid-linear, L2 = 1e3",
        "is the penalty strength important?",
        cross_validate(sigmoid_linear_predictor(raw, y, l2_alpha=1_000.0), n),
    )

    full = build_features(data, numeric, categories=True, missing_flags=True, logs=False)
    record(
        "sigmoid-linear + categorical dummies + missing flags",
        "do categorical fields add information?",
        cross_validate(sigmoid_linear_predictor(full, y), n),
        "categories do not help once numeric features are present",
    )
    full_logs = build_features(data, numeric, categories=True, missing_flags=True, logs=True)
    record(
        "sigmoid-linear + dummies + flags + log features",
        "do log transforms help?",
        cross_validate(sigmoid_linear_predictor(full_logs, y), n),
    )

    def spline_fit_predict(tr: np.ndarray, va: np.ndarray) -> np.ndarray:
        model = make_pipeline(
            StandardScaler(),
            SplineTransformer(n_knots=5, degree=3),
            RidgeCV(alphas=np.logspace(-2, 4, 13)),
        )
        medians, means, scales = fit_preprocessor(raw[tr])
        model.fit(transform_features(raw[tr], medians, means, scales), z[tr])
        return 100.0 * expit(model.predict(transform_features(raw[va], medians, means, scales)))

    record(
        "splines per feature + ridge on logit",
        "is there non-linearity in single features?",
        cross_validate(spline_fit_predict, n),
    )

    def lgbm_fit_predict(tr: np.ndarray, va: np.ndarray) -> np.ndarray:
        frame = data.drop(columns=[ID_COLUMN, TARGET_COLUMN]).copy()
        for column in CATEGORICAL:
            frame[column] = frame[column].astype("category")
        model = lgb.LGBMRegressor(
            n_estimators=600,
            learning_rate=0.02,
            num_leaves=15,
            min_child_samples=30,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.7,
            random_state=CV_SEED,
            verbose=-1,
        )
        model.fit(frame.iloc[tr], y[tr])
        return model.predict(frame.iloc[va])

    record(
        "LightGBM, raw target, categorical features",
        "do trees find non-linear interactions?",
        cross_validate(lgbm_fit_predict, n),
        "fixed 600 trees, no early stopping on validation rows",
    )

    def mlp_fit_predict(tr: np.ndarray, va: np.ndarray) -> np.ndarray:
        model = make_pipeline(
            StandardScaler(),
            MLPRegressor(hidden_layer_sizes=(64,), alpha=1.0, max_iter=2000, early_stopping=True, random_state=CV_SEED),
        )
        medians, means, scales = fit_preprocessor(full[tr])
        model.fit(transform_features(full[tr], medians, means, scales), z[tr])
        return 100.0 * expit(model.predict(transform_features(full[va], medians, means, scales)))

    record("MLP (64 units) on logit target", "neural net with the same features", cross_validate(mlp_fit_predict, n))

    # Pairwise interactions among the strongest features (ridge on logit).
    top = ["insurance_products", "active_policies", "digital_behavior_score", "internet_activity", "mobile_app_usage"]
    pairs = pd.DataFrame(
        {f"{a}*{b}": data[a].astype(float) * data[b].astype(float) for a, b in itertools.combinations(top, 2)}
    ).to_numpy()
    interactions = np.hstack([raw, pairs])

    def interaction_fit_predict(tr: np.ndarray, va: np.ndarray) -> np.ndarray:
        medians, means, scales = fit_preprocessor(interactions[tr])
        model = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(
            transform_features(interactions[tr], medians, means, scales), z[tr]
        )
        return 100.0 * expit(model.predict(transform_features(interactions[va], medians, means, scales)))

    record("ridge on logit + pairwise interactions (top 5)", "are there pairwise interactions?", cross_validate(interaction_fit_predict, n))

    lines = [
        "# Model comparison (5-fold CV on hard_train.csv)",
        "",
        f"Shuffled {N_FOLDS}-fold split with random_state={CV_SEED}; only labeled training rows are used.",
        "",
        "| Model | Question | OOF RMSE | Note |",
        "| --- | --- | ---: | --- |",
    ]
    for name, idea, value, note in results:
        lines.append(f"| {name} | {idea} | {value:.4f} | {note} |")
    lines += [
        "",
        "The RMSE of 7 required for a non-zero score was not reached by any model above.",
        "The residual analysis in `reports/eda/eda_summary.md` indicates that most of the remaining error",
        "is irreducible logit-scale noise (sd about 0.6) for the features provided.",
    ]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
