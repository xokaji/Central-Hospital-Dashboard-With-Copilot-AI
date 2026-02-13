"""Streamlit dashboard presenting hospital insights."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"

st.set_page_config(
    page_title="Central Hospital Insights",
    layout="wide",
    page_icon="🏥",
)


def _load_json(path: Path) -> dict | None:
    if path.exists():
        return json.loads(path.read_text())
    return None


@st.cache_data
def load_csv(name: str) -> pd.DataFrame | None:
    path = DATA_DIR / name
    if path.exists():
        return pd.read_csv(path)
    return None


def display_kpis(kpis: dict[str, float]) -> None:
    st.markdown("""
    <div style='display:flex; gap:1rem;'>
      <div style='flex:1; padding:1rem; background:#f4f7fb; border-radius:12px;'>
        <h4 style='margin:0; color: black'>Occupancy</h4>
        <p style='font-size:32px; margin:0; font-weight:600; color: black'>{:.1f}%</p>
        <small style='color: black'>Inpatient share of visits</small>
      </div>
      <div style='flex:1; padding:1rem; background:#fff1f0; border-radius:12px;'>
        <h4 style='margin:0; color: black'>ICU Rate</h4>
        <p style='font-size:32px; margin:0; font-weight:600; color: black'>{:.1f}%</p>
        <small style='color: black'>Patients under critical care</small>
      </div>
      <div style='flex:1; padding:1rem; background:#f0fcf5; border-radius:12px;'>
        <h4 style='margin:0; color: black'>Avg Treatment Cost</h4>
        <p style='font-size:32px; margin:0; font-weight:600; color: black'>${:,.0f}</p>
        <small style='color: black'>Across admissions & OPD</small>
      </div>
    </div>
    """.format(
        kpis.get("occupancy_rate", 0) * 100,
        kpis.get("icu_rate", 0) * 100,
        kpis.get("avg_treatment_cost", 0),
    ), unsafe_allow_html=True)

    cols = st.columns(3)
    cols[0].metric("Readmissions", f"{kpis.get('readmission_rate', 0):.2f}")
    cols[1].metric("Complications", f"{kpis.get('complication_rate', 0):.2f}")
    cols[2].metric("Mortality", f"{kpis.get('mortality_rate', 0):.2f}")


def display_model_metrics(metrics: dict[str, float] | None) -> None:
    if not metrics:
        return
    cols = st.columns(2)
    cols[0].metric("Readmission AUROC", f"{metrics.get('roc_auc', 0):.3f}")
    cols[1].metric("Test Accuracy", f"{metrics.get('test_accuracy', 0):.3f}")


def display_predictions(predictions: pd.DataFrame, threshold: float) -> None:
    st.subheader("Patient Readmission Risk")
    high_risk = (
        predictions[predictions["predicted_readmission_prob"] >= threshold]
        .sort_values("predicted_readmission_prob", ascending=False)
    )
    if high_risk.empty:
        st.info("No patients exceed the selected risk threshold. Lower the slider to include more cases.")
        return

    left, right = st.columns((1, 1))
    with left:
        st.markdown("**High-risk patient roster**")
        st.dataframe(
            high_risk[
                ["patient_id", "department", "length_of_stay", "predicted_readmission_prob"]
            ].head(12)
            .style.format({"predicted_readmission_prob": "{:.2f}"})
        )
    with right:
        fig = px.bar(
            high_risk.head(8),
            x="patient_id",
            y="predicted_readmission_prob",
            color="department",
            title="Risk probability per patient",
            labels={"patient_id": "Patient", "predicted_readmission_prob": "Risk"},
        )
        st.plotly_chart(fig, use_container_width=True)


def display_trends(trend_df: pd.DataFrame) -> None:
    st.subheader("Weekly Admission & Cost Trends")
    left, right = st.columns(2)
    with left:
        fig = px.line(
            trend_df,
            x="admission_week",
            y="admissions",
            markers=True,
            labels={"admission_week": "Week of"},
        )
        st.plotly_chart(fig, use_container_width=True)
    with right:
        fig_cost = px.bar(
            trend_df,
            x="admission_week",
            y="avg_treatment_cost",
            labels={"avg_treatment_cost": "Avg Cost"},
        )
        st.plotly_chart(fig_cost, use_container_width=True)


def display_department_summary(summary_df: pd.DataFrame) -> None:
    st.subheader("Departmental Snapshot")
    st.dataframe(summary_df.sort_values("admissions", ascending=False))


def main() -> None:
    st.markdown("""
    <div style='background:#0c2d48; color:white; padding:1rem; border-radius:12px;'>
      <h1 style='margin:0;'>Central Hospital Data Insights</h1>
      <p style='margin:0;'>Operational, clinical, and predictive KPIs powered by the internal analytics pipeline.</p>
    </div>
    """, unsafe_allow_html=True)

    kpi_summary = _load_json(DATA_DIR / "kpi_summary.json")
    if not kpi_summary:
        st.warning("Run src.pipeline.run_pipeline to produce KPI summaries.")
        return

    display_kpis(kpi_summary)

    predictions = load_csv("predictions.csv")
    model_metrics = _load_json(DATA_DIR / "model_metrics.json")
    display_model_metrics(model_metrics)

    threshold = st.sidebar.slider(
        "Risk threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.65,
        step=0.01,
        help="Patients with a predicted probability above this value are considered high-risk.",
    )
    st.sidebar.markdown(
        "### Filters\nUse the slider to focus on the riskiest names. Refresh the pipeline if you need current data."
    )

    if predictions is not None:
        display_predictions(predictions, threshold)
    else:
        st.warning("Predictions CSV is missing. Run src.pipeline.run_pipeline first.")

    weekly_trend = load_csv("weekly_trend.csv")
    department_summary = load_csv("department_summary.csv")

    if weekly_trend is not None:
        display_trends(weekly_trend)

    if department_summary is not None:
        display_department_summary(department_summary)

    st.markdown("---")
    st.caption("Pipeline refresh captures the latest synthetic or imported events.")


if __name__ == "__main__":
    main()
