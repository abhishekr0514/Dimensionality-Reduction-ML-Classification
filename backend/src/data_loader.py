import pandas as pd
import os


def load_dataset(path: str):
    """
    Loads dataset from CSV file and returns X, y.

    Parameters:
    path (str): Path to CSV file

    Returns:
    X (pd.DataFrame): Feature matrix
    y (pd.Series): Target labels
    """

    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}")

    df = pd.read_csv(path)

    if "target" not in df.columns:
        raise ValueError("Dataset must contain 'target' column.")

    X = df.drop("target", axis=1)
    y = df["target"]

    return X, y