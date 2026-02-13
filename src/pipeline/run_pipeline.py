"""Main entry point for data preparation, KPI computation, and modeling."""
from __future__ import annotations

import json
from pathlib import Path

from .data_generator import generate_patient_data
from .models import train_readmission_model
from .preprocess import load_raw_data, preprocess_patient_data


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    raw_file = project_root / "data" / "raw" / "patient_events.csv"
    processed_dir = project_root / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    if raw_file.exists():
        raw_df = load_raw_data(raw_file)
    else:
        raw_df = generate_patient_data(raw_file, num_records=2500)

    processed_df, kpi_summary, department_summary, weekly_trend = preprocess_patient_data(raw_df)
    processed_df.to_csv(processed_dir / "processed_patients.csv", index=False)
    department_summary.to_csv(processed_dir / "department_summary.csv", index=False)
    weekly_trend.to_csv(processed_dir / "weekly_trend.csv", index=False)
    (processed_dir / "kpi_summary.json").write_text(json.dumps(kpi_summary, indent=2))

    model_output_path = processed_dir / "models" / "readmission_model.joblib"
    scored_df, model_metrics = train_readmission_model(processed_df, model_output_path)
    scored_df.to_csv(processed_dir / "predictions.csv", index=False)
    (processed_dir / "model_metrics.json").write_text(json.dumps(model_metrics, indent=2))

    print("Pipeline complete. KPIs, predictions, and model saved to data/processed.")


if __name__ == "__main__":
    main()
