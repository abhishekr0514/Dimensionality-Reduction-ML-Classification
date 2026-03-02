import streamlit as st
import pandas as pd
from services.api_client import get_metrics
from components.charts import render_metric_bar
from components.charts import render_performance_heatmap

def render_comparison_page():
    """
    Renders model comparison dashboard.
    """

    st.title("📊 Model Comparison Dashboard")

    # -----------------------------
    # Fetch metrics from backend
    # -----------------------------
    metrics = get_metrics()

    if not metrics:
        st.error("Could not load metrics from backend.")
        return

    # Convert to DataFrame
    df = pd.DataFrame(metrics).T
    df = df.sort_values(by="accuracy", ascending=False)

    # -----------------------------
    # Performance Table
    # -----------------------------
    st.subheader("📋 Model Performance Table")
    st.dataframe(df.style.format("{:.4f}"))

    # -----------------------------
    # Best Model Highlight
    # -----------------------------
    best_model = df.index[0]
    best_accuracy = df.iloc[0]["accuracy"]

    st.success(
        f"🏆 Best Model: {best_model} | Accuracy: {best_accuracy:.4f}"
    )

    st.divider()

    # # -----------------------------
    # # Radar Chart
    # # -----------------------------
    # st.subheader("🕸 Model Performance Radar")

    # radar_fig = render_radar_chart(metrics)
    # st.plotly_chart(radar_fig, use_container_width=True)

    # st.divider()


    st.subheader("🔥 Performance Heatmap")

    heatmap_fig = render_performance_heatmap(metrics)
    st.plotly_chart(heatmap_fig, use_container_width=True)

    # -----------------------------
    # Metric Comparison Section
    # -----------------------------
    st.subheader("📊 Compare Specific Metric")

    metric_choice = st.selectbox(
        "Select Metric",
        ["accuracy", "precision", "f1", "roc_auc"]
    )

    bar_fig = render_metric_bar(metrics, metric_choice)
    st.plotly_chart(bar_fig, use_container_width=True)