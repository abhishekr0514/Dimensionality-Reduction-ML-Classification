import pandas as pd
from sklearn.datasets import load_breast_cancer

# Load dataset
data = load_breast_cancer()

# Convert to DataFrame
df = pd.DataFrame(data.data, columns=data.feature_names)
df["target"] = data.target

# Save to raw folder
df.to_csv(r"C:\ML PROJECT\Dimensionality-Reduction-Classification-Study\data\raw_data\breast_cancer.csv", index=False)

print("File saved successfully!")