
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score

from src.models import build_preprocessor, get_classification_models

# Load dataset
df = pd.read_csv("data/raw/student-mat.csv", sep=";")

# Use early prediction features (exclude G1 and G2)
X = df.drop(columns=["G1", "G2", "G3"])
y = (df["G3"] >= 10).astype(int)

# Same split used by AutoML
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

results = []

# Evaluate baseline models
models = get_classification_models()

for name, model in models.items():
    try:
        pipeline = Pipeline([
            ("preprocessor", build_preprocessor(X_train)),
            ("model", model)
        ])

        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)

        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, predictions),
            "Macro F1": f1_score(y_test, predictions, average="macro")
        })

        print(f"Evaluated: {name}")

    except Exception as e:
        print(f"Skipped {name}: {e}")

# Evaluate AutoML model
automl = joblib.load("models/automl_early.joblib")
predictions = automl.predict(X_test)

results.append({
    "Model": "AutoML",
    "Accuracy": accuracy_score(y_test, predictions),
    "Macro F1": f1_score(y_test, predictions, average="macro")
})

# Display and save results
results_df = pd.DataFrame(results).sort_values(
    by="Macro F1", ascending=False
)

print("\n===== TEST SET COMPARISON =====")
print(results_df.to_string(index=False))

results_df.to_csv("reports/automl_test_comparison.csv", index=False)
print("\nSaved to reports/automl_test_comparison.csv")
