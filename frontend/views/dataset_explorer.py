import streamlit as st
import pandas as pd
import plotly.express as px
from services.api_client import get_dataset
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt

def render_dataset_explorer():

    st.title("📊 Dataset Explorer")

    data = get_dataset()
    df=pd.DataFrame(data)  

    # -----------------------------
    # Class Distribution
    # -----------------------------
    st.subheader("Class Distribution")

    class_counts = df["target"].value_counts()

    fig = px.pie(
        values=class_counts.values,
        names=["Benign", "Malignant"],
        title="Class Distribution"
    )

    st.plotly_chart(fig, use_container_width=True)

    # -----------------------------
    # Correlation Heatmap
    # -----------------------------
    st.subheader("Correlation Heatmap")

    corr = df.corr()

    fig2 = px.imshow(
        corr,
        title="Feature Correlation Matrix",
        color_continuous_scale="RdBu_r"
    )

    st.plotly_chart(fig2, use_container_width=True)

    # -----------------------------
    # Feature Distribution
    # -----------------------------
    st.subheader("Feature Distribution")

    feature_choice = st.selectbox(
        "Select Feature",
        df.columns[:-1]
    )

    fig3 = px.histogram(
        df,
        x=feature_choice,
        color="target",
        barmode="overlay",
        title=f"{feature_choice} Distribution"
    )

    st.plotly_chart(fig3, use_container_width=True)

    # -----------------------------
    # PCA 2D Visualization
    # -----------------------------
    st.subheader("PCA 2D Projection")

    X = df.drop("target", axis=1)

    pca = PCA(n_components=2)
    components = pca.fit_transform(X)

    pca_df = pd.DataFrame(
        components,
        columns=["PC1", "PC2"]
    )
    pca_df["target"] = df["target"]

    fig4 = px.scatter(
        pca_df,
        x="PC1",
        y="PC2",
        color="target",
        title="PCA 2D Scatter Plot"
    )

    st.plotly_chart(fig4, use_container_width=True)