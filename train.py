"""Train the conservative regularized CatBoost ensemble model on hard_train.csv with Huber loss.

Model: 5-fold CatBoostRegressor ensemble with Huber loss ('Huber:delta=1.5'),
       shallow trees (depth 4), and strong L2 leaf regularization (l2_leaf_reg=30).
Training: 5-fold cross-validation with early stopping on each validation fold.

What is saved to ``--model-dir``:
  * ``catboost_fold_{k}.cbm`` - CatBoost model weights for fold k;
  * ``fold_preprocessors.npz`` - feature names, fold medians/means/scales, fold RMSEs, fold weights, categorical modes;
  * ``feature_coefficients.csv`` - average feature importances across folds;
  * ``model_metadata.json`` - seed, feature list, fold models, hyperparameters, library versions, fold weights.

Seed handling:
  * ``--seed N`` fixes the seed (the value is written to the metadata and to the
    submission file name by ``predict.py``);
  * without ``--seed`` a new random seed is drawn and printed.

Example:
    python train.py --train-csv hard_train.csv --model-dir artifacts --seed 434089
"""

from __future__ import annotations

import argparse
import json
import platform
import random
from pathlib import Path

from catboost import CatBoostRegressor
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.model_selection import KFold

from model_utils import (
    CATEGORICAL_COLUMNS,
    TARGET_COLUMN,
    fit_categorical_modes,
    fit_preprocessor,
    generate_features,
    get_numeric_feature_names,
    numeric_matrix,
    set_global_seed,
    transform_features,
)

DEFAULT_DEPTH = 4
DEFAULT_LEARNING_RATE = 0.03
DEFAULT_L2_LEAF_REG = 30.0
DEFAULT_N_ESTIMATORS = 500
DEFAULT_EARLY_STOPPING_ROUNDS = 50
DEFAULT_LOSS_FUNCTION = "Huber:delta=1.5"


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
        "--cv-folds",
        type=int,
        default=5,
        help="Number of folds for cross-validation and ensemble models",
    )
    parser.add_argument("--depth", type=int, default=DEFAULT_DEPTH, help="Tree depth (e.g. 3 or 4)")
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE, help="Learning rate")
    parser.add_argument("--l2-leaf-reg", type=float, default=DEFAULT_L2_LEAF_REG, help="L2 regularization on leaves")
    parser.add_argument("--n-estimators", type=int, default=DEFAULT_N_ESTIMATORS, help="Max iterations/trees")
    parser.add_argument(
        "--early-stopping-rounds",
        type=int,
        default=DEFAULT_EARLY_STOPPING_ROUNDS,
        help="Early stopping rounds on validation fold",
    )
    parser.add_argument(
        "--loss-function",
        default=DEFAULT_LOSS_FUNCTION,
        help="CatBoost loss function (e.g. Huber:delta=1.5)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.cv_folds < 2:
        raise ValueError("--cv-folds must be at least 2")

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

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    fold_medians_list: list[np.ndarray] = []
    fold_means_list: list[np.ndarray] = []
    fold_scales_list: list[np.ndarray] = []
    fold_model_files: list[str] = []
    fold_best_iterations: list[int] = []
    fold_rmses: list[float] = []
    fold_importances = np.zeros(len(feature_names), dtype=float)

    oof_prediction = np.full(len(data), np.nan, dtype=float)
    splitter = KFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)

    for fold_number, (fold_train, fold_validation) in enumerate(splitter.split(raw_matrix), start=1):
        f_medians, f_means, f_scales = fit_preprocessor(raw_matrix[fold_train])
        f_x_train = transform_features(raw_matrix[fold_train], f_medians, f_means, f_scales)
        f_x_validation = transform_features(raw_matrix[fold_validation], f_medians, f_means, f_scales)

        cb_model = CatBoostRegressor(
            loss_function=args.loss_function,
            eval_metric="RMSE",
            depth=args.depth,
            learning_rate=args.learning_rate,
            l2_leaf_reg=args.l2_leaf_reg,
            iterations=args.n_estimators,
            random_seed=args.seed + fold_number,
            verbose=False,
        )
        cb_model.fit(
            f_x_train,
            target[fold_train],
            eval_set=(f_x_validation, target[fold_validation]),
            early_stopping_rounds=args.early_stopping_rounds,
            verbose=False,
        )

        fold_pred = cb_model.predict(f_x_validation)
        oof_prediction[fold_validation] = fold_pred

        model_filename = f"catboost_fold_{fold_number - 1}.cbm"
        cb_model.save_model(str(model_dir / model_filename))
        fold_model_files.append(model_filename)

        best_iter = int(cb_model.get_best_iteration())
        fold_best_iterations.append(best_iter)
        fold_importances += cb_model.get_feature_importance() / args.cv_folds

        fold_medians_list.append(f_medians)
        fold_means_list.append(f_means)
        fold_scales_list.append(f_scales)

        fold_rmse = float(np.sqrt(np.mean((target[fold_validation] - fold_pred) ** 2)))
        fold_rmses.append(fold_rmse)
        print(f"Completed CV fold {fold_number}/{args.cv_folds} (best iteration: {best_iter}, RMSE: {fold_rmse:.4f})")

    cross_validation_rmse = float(np.sqrt(np.mean((target - oof_prediction) ** 2)))

    # Compute fold weights inversely proportional to validation RMSE
    inv_rmses = 1.0 / np.asarray(fold_rmses, dtype=float)
    fold_weights = (inv_rmses / np.sum(inv_rmses)).tolist()

    # Categorical modes (saved for completeness)
    categorical_modes = fit_categorical_modes(data, CATEGORICAL_COLUMNS)

    # Save fold preprocessing parameters, RMSEs, weights and feature info
    np.savez_compressed(
        model_dir / "fold_preprocessors.npz",
        feature_names=np.asarray(feature_names, dtype=str),
        fold_medians=np.asarray(fold_medians_list, dtype=float),
        fold_means=np.asarray(fold_means_list, dtype=float),
        fold_scales=np.asarray(fold_scales_list, dtype=float),
        fold_rmses=np.asarray(fold_rmses, dtype=float),
        fold_weights=np.asarray(fold_weights, dtype=float),
        categorical_columns=np.asarray(list(categorical_modes.keys()), dtype=str),
        categorical_modes=np.asarray(list(categorical_modes.values()), dtype=str),
    )

    # Feature importance table
    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": fold_importances,
            "standardized_coefficient": fold_importances,
            "absolute_coefficient": fold_importances,
        }
    ).sort_values("importance", ascending=False)
    importance_df.to_csv(model_dir / "feature_coefficients.csv", index=False)

    metadata = {
        "model_file": "fold_preprocessors.npz",
        "fold_model_files": fold_model_files,
        "model_type": f"CatBoostRegressor {args.cv_folds}-fold ensemble (loss={args.loss_function}, depth={args.depth}, l2_leaf_reg={args.l2_leaf_reg})",
        "target": TARGET_COLUMN,
        "prediction_formula": f"Weighted average of {args.cv_folds} CatBoost fold models (weights inversely proportional to fold RMSE), clipped to [0, 100]",
        "seed": int(args.seed),
        "seed_policy": "fixed with --seed" if args.seed is not None else "random per run",
        "training_rows": int(len(data)),
        "cross_validation": {
            "folds": int(args.cv_folds),
            "oof_rmse": cross_validation_rmse,
            "fold_rmses": fold_rmses,
            "fold_weights": fold_weights,
            "fold_best_iterations": fold_best_iterations,
        },
        "features": feature_names,
        "imputation": {
            "numeric": "median of training fold",
            "categorical": categorical_modes,
            "categorical_used_by_model": False,
        },
        "hyperparameters": {
            "loss_function": args.loss_function,
            "depth": args.depth,
            "learning_rate": args.learning_rate,
            "l2_leaf_reg": args.l2_leaf_reg,
            "n_estimators": args.n_estimators,
            "early_stopping_rounds": args.early_stopping_rounds,
        },
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "scipy": scipy.__version__,
            "catboost": CatBoostRegressor().__module__,
        },
    }
    with (model_dir / "model_metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)
        file.write("\n")

    print(f"\n{args.cv_folds}-fold OOF RMSE: {cross_validation_rmse:.4f}")
    print(f"Fold RMSEs: {[round(r, 4) for r in fold_rmses]}")
    print(f"Fold Weights: {[round(w, 4) for w in fold_weights]}")
    print(f"Numeric predictors: {len(feature_names)}")
    print(f"Saved {args.cv_folds} fold models and metadata to: {model_dir}")


if __name__ == "__main__":
    main()
