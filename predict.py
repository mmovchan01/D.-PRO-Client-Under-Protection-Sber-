"""Predict protection_score with the saved CatBoost ensemble model. No training happens here.

What it does:
  1. loads the fold models, preprocessing arrays, and validation weights saved by ``train.py``;
  2. generates engineered domain features on the test data;
  3. handles missing features by filling with saved training medians;
  4. computes predictions from all fold models and performs a weighted average
     (where models with smaller validation RMSE get proportionally higher weight);
  5. clips predictions to [0, 100] and writes ``submission_seed_{SEED}.csv``.

Examples:
    python predict.py --input-csv hard_test.csv --model-dir artifacts
    python predict.py --input-csv private_test.csv --model-dir artifacts
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from catboost import CatBoostRegressor
import numpy as np
import pandas as pd

from model_utils import (
    CATEGORICAL_COLUMNS,
    ID_COLUMN,
    TARGET_COLUMN,
    ensure_columns,
    generate_features,
    impute_categorical,
    numeric_matrix,
    set_global_seed,
    transform_features,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-csv", default="hard_test.csv", help="Unlabeled test CSV")
    parser.add_argument("--model-dir", default="artifacts", help="Directory containing the saved model")
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Submission path (default: submission_seed_<model seed>.csv in the current directory)",
    )
    return parser.parse_args()


def load_ensemble(
    model_dir: Path,
) -> tuple[
    dict,
    list[str],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[CatBoostRegressor],
    dict[str, str],
]:
    """Return metadata, feature names, preprocessing parameters, fold weights, fold models, and categorical modes."""
    metadata_path = model_dir / "model_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Model metadata not found: {metadata_path}; run train.py first")
    with metadata_path.open(encoding="utf-8") as file:
        metadata = json.load(file)

    preprocessor_path = model_dir / "fold_preprocessors.npz"
    if not preprocessor_path.exists():
        raise FileNotFoundError(f"Fold preprocessors not found: {preprocessor_path}")
    with np.load(preprocessor_path, allow_pickle=False) as saved:
        feature_names = saved["feature_names"].astype(str).tolist()
        fold_medians = saved["fold_medians"]
        fold_means = saved["fold_means"]
        fold_scales = saved["fold_scales"]
        if "fold_weights" in saved:
            fold_weights = saved["fold_weights"]
        elif "fold_rmses" in saved:
            inv_r = 1.0 / saved["fold_rmses"]
            fold_weights = inv_r / np.sum(inv_r)
        else:
            fold_weights = np.ones(len(fold_medians), dtype=float) / len(fold_medians)
        modes = dict(zip(saved["categorical_columns"].astype(str), saved["categorical_modes"].astype(str)))

    if feature_names != metadata["features"]:
        raise ValueError("Model feature names do not match model_metadata.json")

    fold_model_files = metadata["fold_model_files"]
    models: list[CatBoostRegressor] = []
    for model_filename in fold_model_files:
        model_path = model_dir / model_filename
        if not model_path.exists():
            raise FileNotFoundError(f"Fold model file not found: {model_path}")
        cb = CatBoostRegressor()
        cb.load_model(str(model_path))
        models.append(cb)

    return metadata, feature_names, fold_medians, fold_means, fold_scales, fold_weights, models, modes


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_csv)
    if not input_path.exists():
        raise FileNotFoundError(f"Test CSV not found: {input_path}")

    model_dir = Path(args.model_dir)
    (
        metadata,
        feature_names,
        fold_medians,
        fold_means,
        fold_scales,
        fold_weights,
        models,
        modes,
    ) = load_ensemble(model_dir)
    seed = int(metadata["seed"])
    set_global_seed(seed)

    data = pd.read_csv(input_path)
    if ID_COLUMN not in data.columns:
        raise ValueError(f"Test CSV must contain the identifier column {ID_COLUMN!r}")
    if data[ID_COLUMN].isna().any():
        raise ValueError(f"Identifier column {ID_COLUMN!r} contains missing values")

    data = generate_features(data)

    # Report what had to be filled, so that the jury can see it.
    absent = ensure_columns(data, feature_names)
    if absent:
        print(f"WARNING: {len(absent)} feature column(s) absent from input, filled with training medians: {absent}")
    numeric_frame = data[feature_names]
    nan_counts = numeric_frame.apply(pd.to_numeric, errors="coerce").isna().sum()
    nan_counts = nan_counts[nan_counts > 0]
    if len(nan_counts):
        print("Numeric features filled with training medians (count of missing/invalid values):")
        for column, count in nan_counts.items():
            print(f"  {column}: {int(count)}")
    else:
        print("No missing numeric feature values in input.")

    data = impute_categorical(data, {c: m for c, m in modes.items() if c in CATEGORICAL_COLUMNS})
    raw = numeric_matrix(data, feature_names)

    fold_predictions = np.zeros((len(models), len(data)), dtype=float)
    for fold_index, model in enumerate(models):
        transformed = transform_features(raw, fold_medians[fold_index], fold_means[fold_index], fold_scales[fold_index])
        fold_predictions[fold_index] = model.predict(transformed)

    # Weighted ensemble prediction
    weights = np.asarray(fold_weights, dtype=float)
    weights = weights / np.sum(weights)
    weighted_prediction = np.sum(fold_predictions * weights[:, np.newaxis], axis=0)
    prediction = np.clip(weighted_prediction, 0.0, 100.0)
    if not np.isfinite(prediction).all():
        raise ValueError("Model produced a non-finite prediction")

    submission = pd.DataFrame({ID_COLUMN: data[ID_COLUMN].astype(str), TARGET_COLUMN: prediction})
    output_path = Path(args.output_csv) if args.output_csv else Path(f"submission_seed_{seed}.csv")
    submission.to_csv(output_path, index=False, float_format="%.4f")

    print(f"Folds combined: {len(models)}")
    print(f"Fold weights applied: {[round(float(w), 4) for w in weights]}")
    print(f"Rows written: {len(submission)}")
    print(f"Prediction range: {prediction.min():.4f} .. {prediction.max():.4f}")
    print(f"Saved submission: {output_path}")


if __name__ == "__main__":
    main()
