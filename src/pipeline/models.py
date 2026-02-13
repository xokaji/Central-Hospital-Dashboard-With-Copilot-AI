"""Train lightweight tabular models and score patients."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
from joblib import dump
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

NUMERIC_FEATURES = [
    "age",
    "length_of_stay",
    "icu_flag",
    "complication_flag",
    "mortality_flag",
    "lab_score",
    "vital_risk_score",
    "risk_score",
    "cost_per_day",
    "treatment_cost",
    "is_inpatient",
    "opd_visit",
]
CATEGORICAL_FEATURES = ["department", "treatment_category", "admission_type", "gender"]


def _expand_categoricals(df: pd.DataFrame, categories: Iterable[str]) -> pd.DataFrame:
    return pd.get_dummies(df, columns=list(categories), drop_first=True)


def _build_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    expanded = _expand_categoricals(df, CATEGORICAL_FEATURES)
    cat_columns = [c for c in expanded.columns if any(c.startswith(f"{cat}_") for cat in CATEGORICAL_FEATURES)]
    features = NUMERIC_FEATURES + cat_columns
    return expanded[features].fillna(0), features


def train_readmission_model(
    df: pd.DataFrame,
    model_output_path: Path | str,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Fit a readmission classifier and persist the model."""

    X, feature_columns = _build_feature_matrix(df)
    y = df["readmitted"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    model = HistGradientBoostingClassifier(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "roc_auc": round(roc_auc_score(y_test, y_proba), 3),
        "test_accuracy": round((y_test == y_pred).mean(), 3),
    }

    df_scored = df.copy()
    df_scored["predicted_readmission_prob"] = model.predict_proba(X)[:, 1]
    df_scored["predicted_readmission_class"] = model.predict(X)

    model_output_path = Path(model_output_path)
    model_output_path.parent.mkdir(parents=True, exist_ok=True)
    dump(model, model_output_path)

    return df_scored, metrics
