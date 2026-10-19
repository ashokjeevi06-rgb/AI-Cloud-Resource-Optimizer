
from pathlib import Path
import joblib
import pandas as pd

base = Path(__file__).resolve().parent

model_path = base / "model" / "cloud_optimizer.pkl"
dataset_path = base / "dataset" / "cloud_workload_dataset.csv"

print("\n--- MODEL CHECK ---")

model = joblib.load(model_path)

print("Model type:", type(model))
print("Expected number of input features:",
      getattr(model, "n_features_in_", "Unknown"))
print("Feature names:",
      getattr(model, "feature_names_in_", "Not stored"))

print("\n--- DATASET CHECK ---")

df = pd.read_csv(dataset_path)

print("Dataset shape:", df.shape)
print("Dataset columns:")
print(df.columns.tolist())
print("\nFirst 3 rows:")
print(df.head(3).to_string(index=False))
