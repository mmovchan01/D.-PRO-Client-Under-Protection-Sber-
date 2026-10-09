"""Predict protection_score with the saved model. No training happens here.

What it does:
  1. loads the weights and the imputation values saved by ``train.py``;
  2. adds any feature column that is absent from the input (filled with the
     saved median), so a missing column does not stop the run;
  3. fills NaN and non-numeric values in numeric features with the saved
     training medians, and NaN in categorical columns with the saved modes;
  4. writes ``submission_seed_{SEED}.csv`` with the seed from the metadata.

Examples:
    python predict.py --input-csv hard_test.csv --model-dir artifacts
    python predict.py --input-csv private_test.csv --model-dir artifacts
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

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


def load_model(model_dir: Path) -> tuple[dict, dict]:
    """Return (metadata, arrays) for the saved model, with consistency checks."""
    metadata_path = model_dir / "model_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Model metadata not found: {metadata_path}; run train.py first")
    with metadata_path.open(encoding="utf-8") as file:
        metadata = json.load(file)

    model_path = model_dir / metadata["model_file"]
    if not model_path.exists():
        raise FileNotFoundError(f"Saved model not found: {model_path}")
    with np.load(model_path, allow_pickle=False) as saved:
        arrays = {name: saved[name] for name in saved.files}

    feature_names = arrays["feature_names"].astype(str).tolist()
    if feature_names != metadata["features"]:
        raise ValueError("Model feature names do not match model_metadata.json")
    if len(feature_names) != len(arrays["coefficients"]):
        raise ValueError("Saved feature list and coefficient vector have different lengths")
    for key in ("medians", "means", "scales"):
        if len(arrays[key]) != len(feature_names):
            raise ValueError(f"Saved {key!r} has the wrong length")
    if not np.isfinite(arrays["coefficients"]).all() or not np.isfinite(arrays["intercept"]):
        raise ValueError("Saved model parameters are not finite")
    return metadata, arrays


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_csv)
    if not input_path.exists():
        raise FileNotFoundError(f"Test CSV not found: {input_path}")

    metadata, arrays = load_model(Path(args.model_dir))
    seed = int(metadata["seed"])
    set_global_seed(seed)  # prediction is deterministic; the seed is fixed anyway

    feature_names = arrays["feature_names"].astype(str).tolist()
    coefficients = arrays["coefficients"]
    intercept = float(arrays["intercept"])
    medians, means, scales = arrays["medians"], arrays["means"], arrays["scales"]
    modes = dict(zip(arrays["categorical_columns"].astype(str), arrays["categorical_modes"].astype(str)))

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
    transformed = transform_features(raw, medians, means, scales)
    prediction = np.clip(100.0 * expit(transformed @ coefficients + intercept), 0.0, 100.0)
    if not np.isfinite(prediction).all():
        raise ValueError("Model produced a non-finite prediction")

    submission = pd.DataFrame({ID_COLUMN: data[ID_COLUMN].astype(str), TARGET_COLUMN: prediction})
    output_path = Path(args.output_csv) if args.output_csv else Path(f"submission_seed_{seed}.csv")
    submission.to_csv(output_path, index=False, float_format="%.4f")

    print(f"Rows written: {len(submission)}")
    print(f"Prediction range: {prediction.min():.4f} .. {prediction.max():.4f}")
    print(f"Saved submission: {output_path}")


if __name__ == "__main__":
    main()
