
from pathlib import Path
import sys

import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.models import build_preprocessor

# Load dataset
df = pd.read_csv(
    ROOT / "data" / "raw" / "student-mat.csv",
    sep=";"
)

# Predict actual final grade (G3)
X = df.drop(columns=["G3"])
y = df["G3"]

# Hold out test data for final evaluation
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

models = {
    "Linear Regression": LinearRegression(),
    "KNN": KNeighborsRegressor(n_neighbors=5),
    "SVR": SVR(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    ),
    "Extra Trees": ExtraTreesRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    ),
    "XGBoost": XGBRegressor(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=2
    ),
    "LightGBM": LGBMRegressor(
        n_estimators=100,
        verbosity=-1,
        random_state=42,
        n_jobs=2
    )
}

# Five-fold cross-validation on training data only
cv = KFold(n_splits=5, shuffle=True, random_state=42)

scoring = {
    "mae": "neg_mean_absolute_error",
    "mse": "neg_mean_squared_error",
    "r2": "r2"
}

results = []

for name, model in models.items():
    pipeline = Pipeline([
        ("preprocessor", build_preprocessor(X_train)),
        ("model", model)
    ])

    scores = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        n_jobs=1
    )

    mae = -scores["test_mae"].mean()
    rmse = (-scores["test_mse"].mean()) ** 0.5
    r2 = scores["test_r2"].mean()

    results.append({
        "model": name,
        "MAE_mean": mae,
        "RMSE_mean": rmse,
        "R2_mean": r2
    })

    print(
        f"{name}: MAE={mae:.3f}, "
        f"RMSE={rmse:.3f}, R2={r2:.3f}"
    )

results_df = pd.DataFrame(results)
results_df = results_df.sort_values(
    by="MAE_mean",
    ascending=True
)

output_path = ROOT / "reports" / "regression_baseline_results.csv"
results_df.to_csv(output_path, index=False)

print("\nREGRESSION LEADERBOARD")
print(results_df.to_string(index=False))
print(f"\nSaved results to: {output_path}")
print("\nHeld-out test set reserved for final evaluation.")
