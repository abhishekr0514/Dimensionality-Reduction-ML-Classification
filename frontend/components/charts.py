import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
# --------------------------------------------------
# Probability Bar Chart
# --------------------------------------------------

def render_probability_chart(prob_benign, prob_malignant):
    """
    Displays probability bar chart.
    """

    labels = ["Benign", "Malignant"]
    values = [prob_benign, prob_malignant]

    fig, ax = plt.subplots()
    ax.bar(labels, values)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Probability")
    ax.set_title("Prediction Confidence")

    st.pyplot(fig)


# --------------------------------------------------
# Model Comparison Chart
# --------------------------------------------------

def render_model_comparison(metrics_dict):
    """
    Displays model comparison as bar chart.

    metrics_dict format:
    {
        "baseline_svm": {"accuracy": 0.98},
        "pca_svm": {"accuracy": 0.97}
    }
    """

    models = list(metrics_dict.keys())
    accuracies = [metrics_dict[m]["accuracy"] for m in models]

    fig, ax = plt.subplots()
    ax.bar(models, accuracies)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy")
    ax.set_title("Model Accuracy Comparison")
    plt.xticks(rotation=45)

    st.pyplot(fig)

# MODEL_COLORS = [
#     "#1f77b4",  # blue
#     "#ff7f0e",  # orange
#     "#2ca02c",  # green
#     "#d62728",  # red
#     "#9467bd",  # purple
#     "#8c564b",  # brown
#     "#e377c2",  # pink
#     "#17becf",  # cyan
#     "#bcbd22"   # lime
# ]

# def render_radar_chart(metrics_dict):

#     categories = ["accuracy", "precision", "f1", "roc_auc"]

#     fig = go.Figure()

#     for i, (model, values) in enumerate(metrics_dict.items()):

#         fig.add_trace(go.Scatterpolar(
#             r=[values[m] for m in categories],
#             theta=categories,
#             fill='toself',
#             name=model,
#             line=dict(color=MODEL_COLORS[i % len(MODEL_COLORS)], width=2),
#             fillcolor=MODEL_COLORS[i % len(MODEL_COLORS)],
#             opacity=0.4
#         ))

#     fig.update_layout(
#         polar=dict(
#             radialaxis=dict(
#                 visible=True,
#                 range=[0, 1]
#             )
#         ),
#         showlegend=True,
#         title="Model Performance Radar Chart"
#     )

#     return fig


def render_performance_heatmap(metrics_dict):

    df = pd.DataFrame(metrics_dict).T

    fig = px.imshow(
        df,
        text_auto=".3f",
        color_continuous_scale="Viridis",
        aspect="auto",
        title="Model Performance Heatmap"
    )

    fig.update_layout(
        xaxis_title="Metric",
        yaxis_title="Model"
    )

    return fig


def render_metric_bar(metrics_dict, metric_name):

    data = []

    for model, values in metrics_dict.items():
        data.append({
            "Model": model,
            "Value": values[metric_name]
        })

    fig = px.bar(
        data,
        x="Model",
        y="Value",
        color="Model",  # color by model instead of value
        title=f"{metric_name.upper()} Comparison",
        color_discrete_sequence=px.colors.qualitative.Bold
    )

    fig.update_layout(
        xaxis_tickangle=-45,
        yaxis_range=[0, 1],
        showlegend=False
    )

    return fig