
from pathlib import Path
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.models import run_classification_baselines

# Load the Mathematics dataset
df = pd.read_csv(
    ROOT / "data" / "raw" / "student-mat.csv",
    sep=";"
)

# Classification target: pass = 1, fail = 0
X = df.drop(columns=["G3"])
y = (df["G3"] >= 10).astype(int)

# Reproducible held-out split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Run cross-validation on training data only
results = run_classification_baselines(X_train, y_train)

output_dir = ROOT / "reports"
output_dir.mkdir(parents=True, exist_ok=True)

output_path = output_dir / "baseline_results.csv"
results.to_csv(output_path, index=False)

print("\nBaseline leaderboard:")
print(results.to_string(index=False))
print(f"\nResults saved to: {output_path}")
print("Held-out test set has not been used for model selection.")
