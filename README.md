# Central Hospital Data Insights

This project scaffolds a Python-based analytics and dashboard suite for Central Hospital. It synthesizes structured, semi-structured, and metadata-driven patient records for operational, clinical, financial, and predictive reporting. The core components are:

- **Synthetic data generation** of admission/OPD records with demographics, clinical notes, costs, and metadata.
- **ETL and preprocessing** that derives KPIs (occupancy, readmission, complications, revenue per department) and prepares features for modeling.
- **Predictive modeling** (readmission risk, ICU readjustment) driven by modern tabular models (HistGradientBoosting) with placeholders to extend toward Large Tabular/LLM services.
- **Streamlit dashboard** that visualizes KPIs, treatment efficiency, and live risk predictions on the existing infrastructure without heavy cloud investments.

## Structure

- `src/pipeline/` — data generator, preprocessing, and modeling scripts plus an orchestration module.
- `src/dashboard/` — Streamlit application rendering KPIs, trends, and risk tables.
- `data/raw/` & `data/processed/` — storage for generated patient events and derived outputs (KPIs, predictions).
- `notebooks/` — workspace for exploratory analysis (empty placeholder).
- `tests/` — lightweight smoke tests for generator/pipeline.

## Setup

1. Create an isolated Python environment (recommended):
   ```bash
   python -m venv .venv
   .\.venv\Scripts\Activate
   pip install -r requirements.txt
   ```
2. Run the pipeline to generate synthetic data, KPIs, and a trained readmission predictor:
   ```bash
   python -m src.pipeline.run_pipeline
   ```
   This writes `data/raw/patient_events.csv`, `data/processed/processed_patients.csv`, KPIs (JSON), department summaries, and prediction scores.
3. Launch the dashboard:
   ```bash
   streamlit run src/dashboard/app.py
   ```
   The app reads the processed artefacts, surfaces occupancy/treatment KPIs, and highlights patients flagged by the tabular model.

## Extending

- Swap `models.train_readmission_model` with an LLM-backed tabular model (TabPFN, Hugging Face Tabular API, etc.) by keeping the input/output schema constant.
- Hook real clinical text by replacing the synthetic note generator with actual de-identified EHR notes; the NLP pipeline expects a `note_text` column for enrichment.
- Deploy the Streamlit app inside a lightweight container or wrap it with Dash for richer interactions if required.
-new version
