
from pathlib import Path

import joblib
import optuna
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, VarianceThreshold, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, classification_report

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from src.models import build_preprocessor

ROOT = Path(__file__).resolve().parent.parent


def create_model(trial):
    name = trial.suggest_categorical(
        "model",
        [
            "logistic",
            "random_forest",
            "extra_trees",
            "decision_tree",
            "svm",
            "xgboost",
            "lightgbm",
        ],
    )

    if name == "logistic":
        return LogisticRegression(
            C=trial.suggest_float("C", 0.01, 10.0, log=True),
            max_iter=2000,
            random_state=42,
        )

    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=trial.suggest_int("n_estimators", 100, 300),
            max_depth=trial.suggest_int("max_depth", 3, 20),
            min_samples_split=trial.suggest_int("min_samples_split", 2, 10),
            random_state=42,
            n_jobs=1,
        )

    if name == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=trial.suggest_int("n_estimators", 100, 300),
            max_depth=trial.suggest_int("max_depth", 3, 20),
            min_samples_split=trial.suggest_int("min_samples_split", 2, 10),
            random_state=42,
            n_jobs=1,
        )

    if name == "decision_tree":
        return DecisionTreeClassifier(
            max_depth=trial.suggest_int("max_depth", 2, 20),
            min_samples_split=trial.suggest_int("min_samples_split", 2, 10),
            random_state=42,
        )

    if name == "svm":
        return SVC(
            C=trial.suggest_float("C", 0.01, 10.0, log=True),
            kernel=trial.suggest_categorical("kernel", ["linear", "rbf"]),
            random_state=42,
        )

    if name == "xgboost":
        return XGBClassifier(
            n_estimators=trial.suggest_int("n_estimators", 100, 300),
            max_depth=trial.suggest_int("max_depth", 2, 8),
            learning_rate=trial.suggest_float(
                "learning_rate", 0.01, 0.3, log=True
            ),
            subsample=trial.suggest_float("subsample", 0.7, 1.0),
            colsample_bytree=trial.suggest_float(
                "colsample_bytree", 0.7, 1.0
            ),
            eval_metric="logloss",
            random_state=42,
            n_jobs=1,
        )

    return LGBMClassifier(
        n_estimators=trial.suggest_int("n_estimators", 100, 300),
        max_depth=trial.suggest_int("max_depth", 3, 12),
        learning_rate=trial.suggest_float(
            "learning_rate", 0.01, 0.3, log=True
        ),
        verbosity=-1,
        random_state=42,
        n_jobs=1,
    )


def build_pipeline(trial, X):
    model = create_model(trial)

    feature_method = trial.suggest_categorical(
        "feature_method", ["none", "variance", "mutual_info"]
    )

    steps = [("preprocessor", build_preprocessor(X))]

    if feature_method == "variance":
        steps.append(
            (
                "feature_selection",
                VarianceThreshold(
                    threshold=trial.suggest_float(
                        "variance_threshold", 0.0, 0.05
                    )
                ),
            )
        )

    elif feature_method == "mutual_info":
        steps.append(
            (
                "feature_selection",
                SelectKBest(
                    score_func=lambda X_data, y_data: mutual_info_classif(
                        X_data, y_data, random_state=42
                    ),
                    k=trial.suggest_int(
                        "k_features", 5, min(30, X.shape[1])
                    ),
                ),
            )
        )

    steps.append(("model", model))
    return Pipeline(steps)


def objective(trial, X, y, cv):
    pipeline = build_pipeline(trial, X)

    scores = cross_val_score(
        pipeline,
        X,
        y,
        cv=cv,
        scoring="f1_macro",
        n_jobs=1,
        error_score="raise",
    )

    return scores.mean()


def run_automl(n_trials=50, use_early_features=False):
    df = pd.read_csv(ROOT / "data" / "raw" / "student-mat.csv", sep=";")

    drop_columns = ["G3"]
    if use_early_features:
        drop_columns += ["G1", "G2"]

    X = df.drop(columns=drop_columns)
    y = (df["G3"] >= 10).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    cv = StratifiedKFold(
        n_splits=5, shuffle=True, random_state=42
    )

    mode = "early" if use_early_features else "grade_assisted"

    study = optuna.create_study(
        direction="maximize",
        study_name=f"student_performance_{mode}",
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    study.optimize(
        lambda trial: objective(trial, X_train, y_train, cv),
        n_trials=n_trials,
    )

    print("\nBest CV Macro F1:", round(study.best_value, 4))
    print("Best configuration:", study.best_params)

    # Rebuild the winning pipeline and fit only on training data.
    best_trial = optuna.trial.FixedTrial(study.best_params)
    final_pipeline = build_pipeline(best_trial, X_train)
    final_pipeline.fit(X_train, y_train)

    # Evaluate once on the held-out test set.
    predictions = final_pipeline.predict(X_test)

    print("\nHeld-out test accuracy:",
          round(accuracy_score(y_test, predictions), 4))
    print("Held-out test Macro F1:",
          round(f1_score(y_test, predictions, average="macro"), 4))
    print("\nClassification report:")
    print(classification_report(y_test, predictions, zero_division=0))

    reports_dir = ROOT / "reports"
    models_dir = ROOT / "models"
    reports_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(final_pipeline, models_dir / f"automl_{mode}.joblib")

    study.trials_dataframe().to_csv(
        reports_dir / f"automl_trials_{mode}.csv", index=False
    )

    print(f"\nSaved model: models/automl_{mode}.joblib")
    print(f"Saved trials: reports/automl_trials_{mode}.csv")

    return study, final_pipeline


if __name__ == "__main__":
    run_automl(n_trials=50, use_early_features=True)
