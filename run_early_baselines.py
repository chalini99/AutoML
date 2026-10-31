
from pathlib import Path
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.models import run_classification_baselines

df = pd.read_csv(
    ROOT / "data" / "raw" / "student-mat.csv",
    sep=";"
)

# Remove final grade and the two later-period grades
X = df.drop(columns=["G3", "G1", "G2"])
y = (df["G3"] >= 10).astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("EARLY-PREDICTION EXPERIMENT")
print("Input features:", X.shape[1])
print("G1 included:", "G1" in X.columns)
print("G2 included:", "G2" in X.columns)

results = run_classification_baselines(X_train, y_train)

output_path = ROOT / "reports" / "early_baseline_results.csv"
results.to_csv(output_path, index=False)

print("\nEarly-prediction leaderboard:")
print(results.to_string(index=False))
print(f"\nSaved results to: {output_path}")
