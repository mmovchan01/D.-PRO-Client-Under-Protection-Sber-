"""Load the saved sigmoid-linear model and write a submission without training.

Example:
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

from model_utils import ID_COLUMN, TARGET_COLUMN, numeric_matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", default="hard_test.csv", help="Unlabeled test CSV")
    parser.add_argument("--model-dir", default="artifacts", help="Directory containing the saved model")
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Submission path (default: submission_seed_<model seed>.csv)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_csv)
    model_dir = Path(args.model_dir)
    metadata_path = model_dir / "model_metadata.json"
    if not input_path.exists():
        raise FileNotFoundError(f"Test CSV not found: {input_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Model metadata not found: {metadata_path}; run train.py first")

    with metadata_path.open(encoding="utf-8") as file:
        metadata = json.load(file)
    model_path = model_dir / metadata["model_file"]
    if not model_path.exists():
        raise FileNotFoundError(f"Saved model not found: {model_path}")

    data = pd.read_csv(input_path)
    if ID_COLUMN not in data.columns:
        raise ValueError(f"Test CSV must contain the identifier column {ID_COLUMN!r}")

    with np.load(model_path, allow_pickle=False) as saved_model:
        feature_names = saved_model["feature_names"].astype(str).tolist()
        medians = saved_model["medians"]
        means = saved_model["means"]
        scales = saved_model["scales"]
        coefficients = saved_model["coefficients"]
        intercept = float(saved_model["intercept"])

    if feature_names != metadata.get("features"):
        raise ValueError("Model feature names do not match model_metadata.json")
    if len(feature_names) != len(coefficients):
        raise ValueError("Saved feature list and coefficient vector have different lengths")

    raw = numeric_matrix(data, feature_names)
    imputed = np.where(np.isfinite(raw), raw, medians)
    transformed = (imputed - means) / scales
    prediction = 100.0 * expit(transformed @ coefficients + intercept)
    prediction = np.clip(prediction, 0.0, 100.0)
    if not np.isfinite(prediction).all():
        raise ValueError("Model produced a non-finite prediction")

    submission = pd.DataFrame(
        {
            ID_COLUMN: data[ID_COLUMN].astype(str),
            TARGET_COLUMN: prediction,
        }
    )
    output_path = (
        Path(args.output_csv)
        if args.output_csv
        else Path(f"submission_seed_{int(metadata['seed'])}.csv")
    )
    submission.to_csv(output_path, index=False, float_format="%.4f")

    print(f"Rows written: {len(submission)}")
    print(f"Prediction range: {prediction.min():.4f} .. {prediction.max():.4f}")
    print(f"Saved submission: {output_path}")


if __name__ == "__main__":
    main()
