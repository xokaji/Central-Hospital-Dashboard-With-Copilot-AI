from pathlib import Path

from src.pipeline.data_generator import generate_patient_data


def test_data_generator_creates_csv(tmp_path: Path) -> None:
    output = tmp_path / "patient_events.csv"
    df = generate_patient_data(output, num_records=100)

    assert output.exists()
    assert not df.empty
    assert set(["patient_id", "admission_date", "department"]) <= set(df.columns)
