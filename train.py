"""Train the final sigmoid-linear protection-score model on hard_train.csv.

Model: protection_score = 100 * sigmoid(intercept + standardized_features @ coefficients)
Fit: MSE on the original 0-100 scale (L-BFGS-B, small L2 penalty).

What is saved to ``--model-dir``:
  * ``protection_score_logistic.npz`` - coefficients, intercept, numeric
    medians/means/scales, categorical columns and their modes;
  * ``model_metadata.json`` - seed, feature list, metrics, library versions.

Seed handling:
  * ``--seed N`` fixes the seed (the value is written to the metadata and to the
    submission file name by ``predict.py``);
  * without ``--seed`` a new random seed is drawn and printed, as the task rules
    require. The final weights do not depend on the seed: the optimizer is
    deterministic and the final fit uses all labeled rows. The seed only affects
    the validation split and the CV folds, which are reported metrics.

Example:
    python train.py --train-csv hard_train.csv --model-dir artifacts --seed 434089
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
    CATEGORICAL_COLUMNS,
    TARGET_COLUMN,
    fit_categorical_modes,
    fit_preprocessor,
    generate_features,
    get_numeric_feature_names,
    logits_to_score,
    numeric_matrix,
    set_global_seed,
    target_to_logit,
    transform_features,
)

MODEL_FILENAME = "protection_score_logistic.npz"
L2_ALPHA = 10_000.0
INITIAL_LOGIT_RIDGE_ALPHA = 100.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-csv", default="hard_train.csv", help="Path to labeled training CSV")
    parser.add_argument("--model-dir", default="artifacts", help="Directory for model and metadata")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Fixed seed. If omitted, a new random seed is drawn and printed.",
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
    """Fit score = 100 * sigmoid(intercept + X @ coefficients) by raw-scale MSE.

    A ridge model on the logit-transformed target gives a deterministic starting
    point, so the result does not depend on random initialization.
    """
    x = np.asarray(features, dtype=float)
    y = np.asarray(target, dtype=float)
    if x.ndim != 2 or y.ndim != 1 or len(x) != len(y):
        raise ValueError("Invalid feature/target dimensions")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Model fitting received non-finite values")

    n_rows, n_features = x.shape
    target_logit = target_to_logit(y)

    # Ridge initialization with an unpenalized intercept.
    x_mean = x.mean(axis=0)
    y_mean = target_logit.mean()
    centered_x = x - x_mean
    centered_y = target_logit - y_mean
    gram = centered_x.T @ centered_x
    right_hand_side = centered_x.T @ centered_y
    initial_coefficients = np.linalg.solve(gram + initialization_alpha * np.eye(n_features), right_hand_side)
    initial_intercept = float(y_mean - x_mean @ initial_coefficients)
    initial_parameters = np.concatenate([initial_coefficients, np.asarray([initial_intercept])])

    penalty = l2_alpha / n_rows

    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        coefficients = parameters[:n_features]
        intercept = parameters[n_features]
        probability = expit(x @ coefficients + intercept)
        prediction = 100.0 * probability
        residual = prediction - y

        loss = 0.5 * np.mean(residual**2) + 0.5 * penalty * np.dot(coefficients, coefficients)
        derivative_wrt_logit = residual * 100.0 * probability * (1.0 - probability) / n_rows
        gradient = np.concatenate(
            [x.T @ derivative_wrt_logit + penalty * coefficients, np.asarray([derivative_wrt_logit.sum()])]
        )
        return float(loss), gradient

    result = minimize(
        objective,
        initial_parameters,
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": 5000, "ftol": 1e-12, "gtol": 1e-9, "maxls": 50},
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


def main() -> None:
    args = parse_args()
    if not 0.0 < args.validation_size < 1.0:
        raise ValueError("--validation-size must be between 0 and 1")
    if args.cv_folds == 1 or args.cv_folds < 0:
        raise ValueError("--cv-folds must be 0 or at least 2")

    if args.seed is None:
        args.seed = random.SystemRandom().randrange(1, 1_000_000)
        print(f"No --seed given; drawn random seed: {args.seed}")
    else:
        print(f"Fixed seed: {args.seed}")
    set_global_seed(args.seed)

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

    data = generate_features(data)

    feature_names = get_numeric_feature_names(data)
    if not feature_names:
        raise ValueError("No numeric predictor columns were found")
    raw_matrix = numeric_matrix(data, feature_names)
    target = data[TARGET_COLUMN].astype(float).to_numpy()

    # Holdout check: imputation and scaling are fit on the training part only.
    train_indices, validation_indices = train_test_split(
        np.arange(len(data)), test_size=args.validation_size, random_state=args.seed, shuffle=True
    )
    v_medians, v_means, v_scales = fit_preprocessor(raw_matrix[train_indices])
    v_x_train = transform_features(raw_matrix[train_indices], v_medians, v_means, v_scales)
    v_x = transform_features(raw_matrix[validation_indices], v_medians, v_means, v_scales)
    v_coefficients, v_intercept, validation_diagnostics = fit_sigmoid_index(v_x_train, target[train_indices])
    validation_prediction = logits_to_score(v_x @ v_coefficients + v_intercept)
    validation_rmse = float(np.sqrt(np.mean((target[validation_indices] - validation_prediction) ** 2)))

    # Out-of-fold check. Each fold refits preprocessing and the model on its own
    # training part, so the held-out rows are never used for fitting.
    cross_validation_rmse = None
    if args.cv_folds >= 2:
        if args.cv_folds > len(data):
            raise ValueError("--cv-folds cannot exceed the number of training rows")
        oof_prediction = np.full(len(data), np.nan, dtype=float)
        splitter = KFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)
        for fold_number, (fold_train, fold_validation) in enumerate(splitter.split(raw_matrix), start=1):
            f_medians, f_means, f_scales = fit_preprocessor(raw_matrix[fold_train])
            f_x_train = transform_features(raw_matrix[fold_train], f_medians, f_means, f_scales)
            f_x_validation = transform_features(raw_matrix[fold_validation], f_medians, f_means, f_scales)
            f_coefficients, f_intercept, _ = fit_sigmoid_index(f_x_train, target[fold_train])
            oof_prediction[fold_validation] = logits_to_score(f_x_validation @ f_coefficients + f_intercept)
            print(f"Completed CV fold {fold_number}/{args.cv_folds}")
        cross_validation_rmse = float(np.sqrt(np.mean((target - oof_prediction) ** 2)))

    # Final model: fit on all labeled rows.
    medians, means, scales = fit_preprocessor(raw_matrix)
    x_all = transform_features(raw_matrix, medians, means, scales)
    coefficients, intercept, final_diagnostics = fit_sigmoid_index(x_all, target)

    # Imputation values for categorical columns (stored for robustness; the
    # numeric model does not read them).
    categorical_modes = fit_categorical_modes(data, CATEGORICAL_COLUMNS)

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
        categorical_columns=np.asarray(list(categorical_modes.keys()), dtype=str),
        categorical_modes=np.asarray(list(categorical_modes.values()), dtype=str),
    )

    coefficients_df = pd.DataFrame(
        {
            "feature": feature_names,
            "standardized_coefficient": coefficients,
            "absolute_coefficient": np.abs(coefficients),
        }
    ).sort_values("absolute_coefficient", ascending=False)
    coefficients_df.to_csv(model_dir / "feature_coefficients.csv", index=False)

    metadata = {
        "model_file": MODEL_FILENAME,
        "model_type": "sigmoid-linear index, fit by L-BFGS-B on raw-scale RMSE",
        "target": TARGET_COLUMN,
        "prediction_formula": "100 * sigmoid(intercept + standardized_numeric_features @ coefficients)",
        "seed": int(args.seed),
        "seed_policy": "fixed with --seed" if args.seed is not None else "random per run",
        "training_rows": int(len(data)),
        "validation_rows": int(len(validation_indices)),
        "validation_size": float(args.validation_size),
        "validation_rmse": validation_rmse,
        "cross_validation": {"folds": int(args.cv_folds), "oof_rmse": cross_validation_rmse},
        "features": feature_names,
        "imputation": {
            "numeric": "median of training data",
            "categorical": categorical_modes,
            "categorical_used_by_model": False,
        },
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
