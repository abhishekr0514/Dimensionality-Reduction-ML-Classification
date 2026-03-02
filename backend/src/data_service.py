import os
import pandas as pd


# Go THREE levels up:
# backend/src → backend → project_root
BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)

DATA_PATH = os.path.join(BASE_DIR, "data", "raw_data", "breast_cancer.csv")


def load_dataset():
    return pd.read_csv(DATA_PATH)


# Optional cache
_df = None


def get_dataset():
    global _df
    if _df is None:
        _df = load_dataset()
    return _df