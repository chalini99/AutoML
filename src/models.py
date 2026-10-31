
from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


def build_preprocessor(X):
    categorical_cols = X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    numerical_cols = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    return ColumnTransformer([
        ("num", numeric_pipeline, numerical_cols),
        ("cat", categorical_pipeline, categorical_cols)
    ])


def get_classification_models():
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, random_state=42
        ),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "SVM": SVC(probability=True, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, random_state=42, n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            eval_metric="logloss",
            random_state=42,
            n_jobs=2
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=100,
            verbosity=-1,
            random_state=42,
            n_jobs=2
        )
    }


def run_classification_baselines(X_train, y_train):
    cv = StratifiedKFold(
        n_splits=5, shuffle=True, random_state=42
    )

    scoring = {
        "accuracy": "accuracy",
        "precision_macro": "precision_macro",
        "recall_macro": "recall_macro",
        "f1_macro": "f1_macro",
        "roc_auc": "roc_auc"
    }

    results = []

    for name, model in get_classification_models().items():
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
            n_jobs=1,
            error_score="raise"
        )

        row = {"model": name}

        for metric in scoring:
            row[f"{metric}_mean"] = scores[
                f"test_{metric}"
            ].mean()

            row[f"{metric}_std"] = scores[
                f"test_{metric}"
            ].std()

        results.append(row)

        print(
            f"{name}: "
            f"F1-macro={row['f1_macro_mean']:.4f}, "
            f"Accuracy={row['accuracy_mean']:.4f}"
        )

    return pd.DataFrame(results).sort_values(
        "f1_macro_mean", ascending=False
    )
