"""Utilities for the sigmoid-linear protection-score model."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import expit

ID_COLUMN = "customer_id"
TARGET_COLUMN = "protection_score"


def get_numeric_feature_names(data: pd.DataFrame) -> list[str]:
    """Return numeric predictor columns, excluding IDs and the label."""
    excluded = {ID_COLUMN, TARGET_COLUMN}
    return [
        column
        for column in data.columns
        if column not in excluded and pd.api.types.is_numeric_dtype(data[column])
    ]


def numeric_matrix(data: pd.DataFrame, feature_names: list[str]) -> np.ndarray:
    """Convert named predictors to a float matrix; non-finite values become NaN."""
    missing = [column for column in feature_names if column not in data.columns]
    if missing:
        raise ValueError(
            "Input data is missing required feature columns: " + ", ".join(missing)
        )
    frame = data[feature_names].apply(pd.to_numeric, errors="coerce")
    matrix = frame.to_numpy(dtype=float)
    matrix[~np.isfinite(matrix)] = np.nan
    return matrix


def fit_preprocessor(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit median imputation and standardization, returning their parameters."""
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("Expected a non-empty two-dimensional feature matrix")

    medians = np.zeros(values.shape[1], dtype=float)
    for column_index in range(values.shape[1]):
        column = values[:, column_index]
        finite = column[np.isfinite(column)]
        medians[column_index] = np.median(finite) if finite.size else 0.0

    imputed = np.where(np.isfinite(values), values, medians)
    means = imputed.mean(axis=0)
    scales = imputed.std(axis=0)
    scales[~np.isfinite(scales) | (scales < 1e-12)] = 1.0
    return medians, means, scales


def transform_features(
    matrix: np.ndarray,
    medians: np.ndarray,
    means: np.ndarray,
    scales: np.ndarray,
) -> np.ndarray:
    """Apply the saved imputation and standardization parameters."""
    values = np.asarray(matrix, dtype=float)
    imputed = np.where(np.isfinite(values), values, medians)
    transformed = (imputed - means) / scales
    if not np.isfinite(transformed).all():
        raise ValueError("Feature preprocessing produced non-finite values")
    return transformed


def target_to_logit(target: np.ndarray) -> np.ndarray:
    """Logit transform used only to initialize the raw-scale fit."""
    values = np.asarray(target, dtype=float)
    probability = np.clip(values / 100.0, 1e-4, 1.0 - 1e-4)
    return np.log(probability / (1.0 - probability))


def logits_to_score(logits: np.ndarray) -> np.ndarray:
    """Convert model logits to valid protection percentages."""
    return 100.0 * expit(np.asarray(logits, dtype=float))
