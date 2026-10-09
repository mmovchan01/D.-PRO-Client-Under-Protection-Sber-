"""Train the final sigmoid-linear protection-score regression model.

Example:
    python train.py --train-csv hard_train.csv --model-dir artifacts   # seed is drawn at random
"""

from __future__ import annotations

import argparse
import json
import platform
import random
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.model_selection import KFold, train_test_split

from model_utils import (
    TARGET_COLUMN,
    fit_preprocessor,
    get_numeric_feature_names,
    logits_to_score,
    numeric_matrix,
    target_to_logit,
    transform_features,
)

MODEL_FILENAME = "protection_score_logistic.npz"
L2_ALPHA = 10_000.0
INITIAL_LOGIT_RIDGE_ALPHA = 100.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", default="hard_train.csv", help="Path to labeled training CSV")
    parser.add_argument("--model-dir", default="artifacts", help="Directory for model and metadata")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Random seed for the validation split and the submission name. "
            "If omitted, a new random seed is drawn for every run."
        ),
    )
    parser.add_argument(
        "--validation-size",
        type=float,
        default=0.20,
        help="Fraction of labeled rows held out for validation",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of shuffled folds for an out-of-fold check; use 0 to skip",
    )
    return parser.parse_args()


def fit_sigmoid_index(
    features: np.ndarray,
    target: np.ndarray,
    *,
    l2_alpha: float = L2_ALPHA,
    initialization_alpha: float = INITIAL_LOGIT_RIDGE_ALPHA,
) -> tuple[np.ndarray, float, dict[str, float | int | bool | str]]:
    """Fit score = 100 * sigmoid(intercept + X @ coefficients).

    The objective is mean squared error on the original 0-100 scale, with a
    small L2 penalty. A ridge model on the logit-transformed target provides a
    deterministic starting point for the nonlinear optimization.
    """
    x = np.asarray(features, dtype=float)
    y = np.asarray(target, dtype=float)
    if x.ndim != 2 or y.ndim != 1 or len(x) != len(y):
        raise ValueError("Invalid feature/target dimensions")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Model fitting received non-finite values")

    n_rows, n_features = x.shape
    target_logit = target_to_logit(y)

    # Ridge initialization, with an unpenalized intercept.
    x_mean = x.mean(axis=0)
    y_mean = target_logit.mean()
    centered_x = x - x_mean
    centered_y = target_logit - y_mean
    gram = centered_x.T @ centered_x
    right_hand_side = centered_x.T @ centered_y
    initial_coefficients = np.linalg.solve(
        gram + initialization_alpha * np.eye(n_features), right_hand_side
    )
    initial_intercept = float(y_mean - x_mean @ initial_coefficients)
    initial_parameters = np.concatenate(
        [initial_coefficients, np.asarray([initial_intercept])]
    )

    penalty = l2_alpha / n_rows

    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        coefficients = parameters[:n_features]
        intercept = parameters[n_features]
        probability = expit(x @ coefficients + intercept)
        prediction = 100.0 * probability
        residual = prediction - y

        loss = 0.5 * np.mean(residual**2) + 0.5 * penalty * np.dot(
            coefficients, coefficients
        )
        derivative_wrt_logit = (
            residual * 100.0 * probability * (1.0 - probability) / n_rows
        )
        gradient = np.concatenate(
            [
                x.T @ derivative_wrt_logit + penalty * coefficients,
                np.asarray([derivative_wrt_logit.sum()]),
            ]
        )
        return float(loss), gradient

    result = minimize(
        objective,
        initial_parameters,
        method="L-BFGS-B",
        jac=True,
        options={
            "maxiter": 5000,
            "ftol": 1e-12,
            "gtol": 1e-9,
            "maxls": 50,
        },
    )
    if not np.isfinite(result.x).all():
        raise RuntimeError("Optimization produced non-finite model parameters")
    if not result.success:
        raise RuntimeError(f"Model optimization did not converge: {result.message}")

    diagnostics: dict[str, float | int | bool | str] = {
        "success": bool(result.success),
        "message": str(result.message),
        "iterations": int(result.nit),
        "objective": float(result.fun),
        "l2_alpha": float(l2_alpha),
        "initial_logit_ridge_alpha": float(initialization_alpha),
    }
    return result.x[:n_features], float(result.x[n_features]), diagnostics


def train_model(
    data: pd.DataFrame,
    feature_names: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, dict[str, float | int | bool | str]]:
    raw = numeric_matrix(data, feature_names)
    medians, means, scales = fit_preprocessor(raw)
    transformed = transform_features(raw, medians, means, scales)
    coefficients, intercept, diagnostics = fit_sigmoid_index(
        transformed, data[TARGET_COLUMN].astype(float).to_numpy()
    )
    return medians, means, scales, coefficients, intercept, diagnostics


def main() -> None:
    args = parse_args()
    if not 0.0 < args.validation_size < 1.0:
        raise ValueError("--validation-size must be between 0 and 1")
    if args.cv_folds == 1 or args.cv_folds < 0:
        raise ValueError("--cv-folds must be 0 or at least 2")

    if args.seed is None:
        # A new seed is drawn on every run; it is printed, stored in the
        # metadata and written into the submission file name.
        args.seed = random.SystemRandom().randrange(1, 1_000_000)
    print(f"Using random seed: {args.seed}")
    random.seed(args.seed)
    np.random.seed(args.seed)

    train_path = Path(args.train_csv)
    if not train_path.exists():
        raise FileNotFoundError(f"Training CSV not found: {train_path}")

    data = pd.read_csv(train_path)
    if TARGET_COLUMN not in data.columns:
        raise ValueError(f"Training CSV must contain the target column {TARGET_COLUMN!r}")
    if data[TARGET_COLUMN].isna().any():
        raise ValueError(f"Training target {TARGET_COLUMN!r} contains missing values")
    if not data[TARGET_COLUMN].between(0.0, 100.0).all():
        raise ValueError(f"Training target {TARGET_COLUMN!r} must lie in [0, 100]")

    feature_names = get_numeric_feature_names(data)
    if not feature_names:
        raise ValueError("No numeric predictor columns were found")
    raw_matrix = numeric_matrix(data, feature_names)
    target = data[TARGET_COLUMN].astype(float).to_numpy()

    train_indices, validation_indices = train_test_split(
        np.arange(len(data)),
        test_size=args.validation_size,
        random_state=args.seed,
        shuffle=True,
    )

    # Validation and model selection use only rows from hard_train.csv.
    validation_medians, validation_means, validation_scales = fit_preprocessor(
        raw_matrix[train_indices]
    )
    validation_x_train = transform_features(
        raw_matrix[train_indices],
        validation_medians,
        validation_means,
        validation_scales,
    )
    validation_x = transform_features(
        raw_matrix[validation_indices],
        validation_medians,
        validation_means,
        validation_scales,
    )
    validation_coefficients, validation_intercept, validation_diagnostics = (
        fit_sigmoid_index(validation_x_train, target[train_indices])
    )
    validation_prediction = logits_to_score(
        validation_x @ validation_coefficients + validation_intercept
    )
    validation_rmse = float(
        np.sqrt(np.mean((target[validation_indices] - validation_prediction) ** 2))
    )

    # Out-of-fold estimates provide a second check that the holdout result is not
    # unusually favorable. Every fold is trained from hard_train.csv only.
    cross_validation_rmse = None
    if args.cv_folds >= 2:
        if args.cv_folds > len(data):
            raise ValueError("--cv-folds cannot exceed the number of training rows")
        oof_prediction = np.full(len(data), np.nan, dtype=float)
        splitter = KFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)
        for fold_number, (fold_train, fold_validation) in enumerate(
            splitter.split(raw_matrix), start=1
        ):
            fold_medians, fold_means, fold_scales = fit_preprocessor(
                raw_matrix[fold_train]
            )
            fold_x_train = transform_features(
                raw_matrix[fold_train], fold_medians, fold_means, fold_scales
            )
            fold_x_validation = transform_features(
                raw_matrix[fold_validation], fold_medians, fold_means, fold_scales
            )
            fold_coefficients, fold_intercept, _ = fit_sigmoid_index(
                fold_x_train, target[fold_train]
            )
            oof_prediction[fold_validation] = logits_to_score(
                fold_x_validation @ fold_coefficients + fold_intercept
            )
            print(f"Completed CV fold {fold_number}/{args.cv_folds}")
        cross_validation_rmse = float(np.sqrt(np.mean((target - oof_prediction) ** 2)))

    # Refit the selected, low-parameter model on every labeled row.
    medians, means, scales, coefficients, intercept, final_diagnostics = train_model(
        data, feature_names
    )

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / MODEL_FILENAME
    np.savez_compressed(
        model_path,
        feature_names=np.asarray(feature_names, dtype=str),
        medians=medians,
        means=means,
        scales=scales,
        coefficients=coefficients,
        intercept=np.asarray(intercept, dtype=float),
    )

    pd.DataFrame(
        {
            "feature": feature_names,
            "standardized_coefficient": coefficients,
            "absolute_coefficient": np.abs(coefficients),
        }
    ).sort_values("absolute_coefficient", ascending=False).to_csv(
        model_dir / "feature_coefficients.csv", index=False
    )

    metadata = {
        "model_file": MODEL_FILENAME,
        "model_type": "sigmoid-linear index, fit by L-BFGS-B on raw-scale RMSE",
        "target": TARGET_COLUMN,
        "prediction_formula": "100 * sigmoid(intercept + standardized_numeric_features @ coefficients)",
        "seed": int(args.seed),
        "training_rows": int(len(data)),
        "validation_rows": int(len(validation_indices)),
        "validation_size": float(args.validation_size),
        "validation_rmse": validation_rmse,
        "cross_validation": {
            "folds": int(args.cv_folds),
            "oof_rmse": cross_validation_rmse,
        },
        "features": feature_names,
        "parameters": {
            "feature_count": int(len(feature_names)),
            "l2_alpha": L2_ALPHA,
            "initial_logit_ridge_alpha": INITIAL_LOGIT_RIDGE_ALPHA,
            "optimizer": "L-BFGS-B",
            "validation_fit": validation_diagnostics,
            "final_fit": final_diagnostics,
        },
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "scipy": scipy.__version__,
        },
    }
    with (model_dir / "model_metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)
        file.write("\n")

    print(f"Validation RMSE (original 0-100 scale): {validation_rmse:.4f}")
    if cross_validation_rmse is not None:
        print(f"{args.cv_folds}-fold OOF RMSE: {cross_validation_rmse:.4f}")
    print(f"Numeric predictors: {len(feature_names)}")
    print(f"Final optimizer iterations: {final_diagnostics['iterations']}")
    print(f"Saved model: {model_path}")
    print(f"Saved metadata: {model_dir / 'model_metadata.json'}")


if __name__ == "__main__":
    main()
