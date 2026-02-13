"""Preprocess raw records into KPI-ready artifacts."""
from __future__ import annotations

from os import PathLike

import pandas as pd


def load_raw_data(raw_file: str | PathLike[str]) -> pd.DataFrame:
    """Load the generated patient events and parse timestamps."""

    return pd.read_csv(raw_file, parse_dates=["admission_date", "discharge_date"])


def preprocess_patient_data(
    raw_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, float], pd.DataFrame, pd.DataFrame]:
    """Return feature-rich dataset plus KPI summaries for the dashboard."""

    df = raw_df.copy()
    df["admission_week"] = df["admission_date"].dt.to_period("W").dt.start_time
    df["discharge_week"] = df["discharge_date"].dt.to_period("W").dt.start_time
    df["is_inpatient"] = (df["admission_type"] == "Inpatient").astype(int)
    df["cost_per_day"] = (df["treatment_cost"] / df["length_of_stay"]).round(2)

    occupancy_rate = df["is_inpatient"].mean()
    kpis = {
        "occupancy_rate": round(occupancy_rate, 3),
        "icu_rate": round(df["icu_flag"].mean(), 3),
        "avg_length_of_stay": round(df.loc[df["is_inpatient"] == 1, "length_of_stay"].mean(), 2),
        "readmission_rate": round(df["readmitted"].mean(), 3),
        "mortality_rate": round(df["mortality_flag"].mean(), 3),
        "complication_rate": round(df["complication_flag"].mean(), 3),
        "avg_treatment_cost": round(df["treatment_cost"].mean(), 2),
        "opd_share": round(1 - occupancy_rate, 3),
    }

    dept_summary = (
        df.groupby("department")
        .agg(
            admissions=("patient_id", "count"),
            avg_length_of_stay=("length_of_stay", "mean"),
            readmission_rate=("readmitted", "mean"),
            avg_treatment_cost=("treatment_cost", "mean"),
            icu_rate=("icu_flag", "mean"),
        )
        .round(3)
        .reset_index()
    )

    weekly_trend = (
        df.groupby("admission_week")
        .agg(
            admissions=("patient_id", "count"),
            avg_treatment_cost=("treatment_cost", "mean"),
        )
        .reset_index()
    )
    weekly_trend["admission_week"] = weekly_trend["admission_week"].dt.strftime("%Y-%m-%d")

    return df, kpis, dept_summary, weekly_trend
