"""Shared helpers for the sigmoid-linear protection-score model.

Used by both ``train.py`` and ``predict.py`` so that feature handling is
identical in training and prediction:

* numeric predictors are converted to floats; any non-finite value becomes NaN;
* NaNs are filled with the medians learned on the training data;
* numeric predictors are standardized with training means and scales;
* categorical columns get modes learned on the training data (stored with the
  model, filled in when missing). The current model uses numeric predictors
  only, because categorical dummies did not improve the out-of-fold RMSE in
  ``reports/experiments.md``; the categorical imputation is kept so that a
  future model can use these columns safely.
"""

from __future__ import annotations

import os
import random

import numpy as np
import pandas as pd
from scipy.special import expit

ID_COLUMN = "customer_id"
TARGET_COLUMN = "protection_score"

# Categorical columns from the task description (object dtype in the CSV files).
CATEGORICAL_COLUMNS = [
    "region",
    "city_type",
    "gender",
    "education",
    "family_status",
    "employment",
    "wealth_segment",
]


def set_global_seed(seed: int) -> None:
    """Fix every random generator that might be used (Python, NumPy, hashing)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def generate_features(data: pd.DataFrame) -> pd.DataFrame:
    """Generate domain-specific engineered features.

    1. Cross-interactions of risks and insurances:
       - property_exposure = crime_rate * (1 - property_insurance)
       - cyber_exposure = cyber_risk * (1 - cyber_protection)

    2. Digital vulnerability index:
       - cyber_vulnerability = digital_behavior_score * (3 - (password_manager + two_factor_auth + security_training))

    3. Financial leverage and credit load:
       - income_to_balance_ratio = average_balance / (income + 1)
       - credit_load_ratio = loan_amount / (income + 1)

    4. Digital and policy activity patterns:
       - digital_channel_ratio = website_visits / (mobile_sessions + 1)
       - net_active_policies = active_policies - expired_policies

    5. Age patterns:
       - age_squared = age ** 2
       - is_young = (age < 25).astype(int)
    """
    df = data.copy()

    # 1. Cross-interactions of risks and insurances
    if "crime_rate" in df.columns and "property_insurance" in df.columns:
        df["property_exposure"] = df["crime_rate"] * (1 - df["property_insurance"])
    if "cyber_risk" in df.columns and "cyber_protection" in df.columns:
        df["cyber_exposure"] = df["cyber_risk"] * (1 - df["cyber_protection"])

    # 2. Digital vulnerability index
    if (
        "digital_behavior_score" in df.columns
        and "password_manager" in df.columns
        and "two_factor_auth" in df.columns
        and "security_training" in df.columns
    ):
        df["cyber_vulnerability"] = df["digital_behavior_score"] * (
            3 - (df["password_manager"] + df["two_factor_auth"] + df["security_training"])
        )

    # 3. Financial leverage and credit load
    if "average_balance" in df.columns and "income" in df.columns:
        df["income_to_balance_ratio"] = df["average_balance"] / (df["income"] + 1)
    if "loan_amount" in df.columns and "income" in df.columns:
        df["credit_load_ratio"] = df["loan_amount"] / (df["income"] + 1)

    # 4. Digital and policy activity patterns
    if "website_visits" in df.columns and "mobile_sessions" in df.columns:
        df["digital_channel_ratio"] = df["website_visits"] / (df["mobile_sessions"] + 1)
    if "active_policies" in df.columns and "expired_policies" in df.columns:
        df["net_active_policies"] = df["active_policies"] - df["expired_policies"]

    # 5. Age patterns
    if "age" in df.columns:
        df["age_squared"] = df["age"] ** 2
        df["is_young"] = (df["age"] < 25).astype(int)

    return df


# Aliases for flexibility
engineer_features = generate_features
create_features = generate_features
add_features = generate_features
add_engineered_features = generate_features


def get_numeric_feature_names(data: pd.DataFrame) -> list[str]:
    """Return numeric predictor columns, excluding the ID and the target."""
    excluded = {ID_COLUMN, TARGET_COLUMN}
    return [
        column
        for column in data.columns
        if column not in excluded and pd.api.types.is_numeric_dtype(data[column])
    ]


def numeric_matrix(data: pd.DataFrame, feature_names: list[str]) -> np.ndarray:
    """Convert named numeric predictors to a float matrix.

    Values that cannot be parsed as numbers (for example the text ``"n/a"``)
    and infinite values become NaN, so that they are imputed like any gap.
    Missing columns are not allowed here; see ``ensure_columns``.
    """
    frame = data[feature_names].apply(pd.to_numeric, errors="coerce")
    matrix = frame.to_numpy(dtype=float)
    matrix[~np.isfinite(matrix)] = np.nan
    return matrix


def ensure_columns(
    data: pd.DataFrame, columns: list[str], *, fill_value: float = np.nan
) -> list[str]:
    """Add absent columns in place, filled with NaN, and return their names.

    A missing column is treated as a column that is entirely missing, so that
    the saved median is used for every row instead of raising an error.
    """
    absent = [column for column in columns if column not in data.columns]
    for column in absent:
        data[column] = fill_value
    return absent


def fit_preprocessor(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Learn median imputation and standardization parameters from a matrix."""
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("Expected a non-empty two-dimensional feature matrix")

    medians = np.zeros(values.shape[1], dtype=float)
    for column_index in range(values.shape[1]):
        column = values[:, column_index]
        finite = column[np.isfinite(column)]
        # A column that is entirely NaN in training gets 0 as a neutral fallback.
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
    """Impute with saved medians, then standardize with saved parameters."""
    values = np.asarray(matrix, dtype=float)
    imputed = np.where(np.isfinite(values), values, medians)
    transformed = (imputed - means) / scales
    if not np.isfinite(transformed).all():
        raise ValueError("Feature preprocessing produced non-finite values")
    return transformed


def fit_categorical_modes(data: pd.DataFrame, columns: list[str]) -> dict[str, str]:
    """Most frequent non-missing value per categorical column (ties: alphabetical)."""
    modes: dict[str, str] = {}
    for column in columns:
        if column not in data.columns:
            continue
        counts = data[column].dropna().astype(str).value_counts()
        if counts.empty:
            continue
        top = counts[counts == counts.max()].index.sort_values()
        modes[column] = str(top[0])
    return modes


def impute_categorical(data: pd.DataFrame, modes: dict[str, str]) -> pd.DataFrame:
    """Fill missing categorical values with saved modes (in a copy of ``data``)."""
    filled = data.copy()
    for column, mode in modes.items():
        if column not in filled.columns:
            filled[column] = mode
        else:
            filled[column] = filled[column].where(filled[column].notna(), mode)
    return filled


def target_to_logit(target: np.ndarray) -> np.ndarray:
    """Logit transform used only to initialize the raw-scale fit."""
    values = np.asarray(target, dtype=float)
    probability = np.clip(values / 100.0, 1e-4, 1.0 - 1e-4)
    return np.log(probability / (1.0 - probability))


def logits_to_score(logits: np.ndarray) -> np.ndarray:
    """Convert model logits to valid protection percentages in [0, 100]."""
    return 100.0 * expit(np.asarray(logits, dtype=float))
