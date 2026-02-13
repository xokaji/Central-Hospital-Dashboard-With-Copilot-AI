from pathlib import Path

from src.pipeline.data_generator import generate_patient_data
from src.pipeline.preprocess import load_raw_data, preprocess_patient_data


def test_preprocess_returns_kpi(tmp_path: Path) -> None:
    raw_path = tmp_path / "patient_events.csv"
    generate_patient_data(raw_path, num_records=80)
    raw_df = load_raw_data(raw_path)
    processed_df, kpi_summary, dept_summary, weekly_trend = preprocess_patient_data(raw_df)

    assert not processed_df.empty
    assert "occupancy_rate" in kpi_summary
    assert dept_summary.shape[0] > 0
    assert weekly_trend.shape[0] > 0
