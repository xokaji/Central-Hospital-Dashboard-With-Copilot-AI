"""Generate synthetic patient events for dashboard development."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

DEPARTMENTS = [
    "Cardiology",
    "Oncology",
    "Orthopedics",
    "Neurology",
    "Emergency",
    "Gastroenterology",
    "Pulmonology",
]
TREATMENT_CATEGORIES = [
    "Surgery",
    "Medication",
    "Therapy",
    "Observation",
    "Diagnostics",
]
ADMISSION_TYPES = ["Inpatient", "OPD"]


def _build_clinical_note(department: str, patient_risk: float) -> str:
    severity = "high" if patient_risk > 0.6 else "moderate" if patient_risk > 0.3 else "low"
    return (
        f"Patient admitted to {department} with {severity} acuity."
        " Clinical team monitoring vitals, labs, and response to therapy."
    )


def generate_patient_data(
    output_path: Path | str,
    num_records: int = 2000,
    seed: int = 42,
) -> pd.DataFrame:
    """Create synthetic patient and OPD events and persist CSV for reuse."""

    rng = np.random.default_rng(seed)
    base_date = pd.Timestamp("2025-01-01")
    admission_offsets = rng.integers(0, 365, size=num_records)

    patient_id = np.arange(1, num_records + 1)
    admission_dates = base_date + pd.to_timedelta(admission_offsets, unit="D")
    length_of_stay = rng.poisson(lam=3, size=num_records)
    discharge_dates = admission_dates + pd.to_timedelta(length_of_stay.clip(1), unit="D")

    departments = rng.choice(DEPARTMENTS, size=num_records)
    treatment_categories = rng.choice(TREATMENT_CATEGORIES, size=num_records)
    admission_types = rng.choice(ADMISSION_TYPES, size=num_records, p=[0.6, 0.4])

    age = rng.integers(1, 95, size=num_records)
    gender = rng.choice(["Male", "Female", "Other"], size=num_records, p=[0.45, 0.45, 0.10])

    lab_score = rng.normal(loc=0.5, scale=0.15, size=num_records).clip(0, 1)
    vital_risk_score = rng.normal(loc=0.4, scale=0.2, size=num_records).clip(0, 1)
    risk_score = 0.5 * lab_score + 0.5 * vital_risk_score + rng.normal(0, 0.05, num_records)
    risk_score = risk_score.clip(0, 1)

    icu_flag = (risk_score > 0.7) | (admission_types == "Inpatient") & (length_of_stay > 3)
    complication_flag = rng.choice([0, 1], size=num_records, p=[0.82, 0.18]).astype(bool)
    mortality_flag = rng.choice([0, 1], size=num_records, p=[0.97, 0.03]).astype(bool)
    readmitted = rng.choice([0, 1], size=num_records, p=[0.8, 0.2]).astype(bool)

    treatment_cost = (
        5000
        + age * 30
        + length_of_stay * 1000
        + (icu_flag.astype(int) * 4000)
        + rng.normal(0, 800, size=num_records)
    ).clip(800, None)

    opd_visit = admission_types == "OPD"

    df = pd.DataFrame(
        {
            "patient_id": patient_id,
            "admission_type": admission_types,
            "department": departments,
            "treatment_category": treatment_categories,
            "age": age,
            "gender": gender,
            "admission_date": admission_dates,
            "discharge_date": discharge_dates,
            "length_of_stay": length_of_stay.clip(1),
            "icu_flag": icu_flag.astype(int),
            "complication_flag": complication_flag.astype(int),
            "mortality_flag": mortality_flag.astype(int),
            "readmitted": readmitted.astype(int),
            "treatment_cost": treatment_cost.round(2),
            "lab_score": lab_score.round(3),
            "vital_risk_score": vital_risk_score.round(3),
            "risk_score": risk_score.round(3),
            "note_text": [
                _build_clinical_note(dep, risk) for dep, risk in zip(departments, risk_score)
            ],
            "opd_visit": opd_visit.astype(int),
        }
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df
