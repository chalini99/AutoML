"""Leakage-safe splitting, cross-validation, and evaluation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping
import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class EvaluationError(ValueError):
    """Raised when a dataset cannot support the requested evaluation."""


@dataclass(frozen=True)
class SplitData:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


def reproducible_split(
    X: pd.DataFrame,
    y: pd.Series,
    task: str,
    test_size: float,
    seed: int = 42,
) -> SplitData:
    """Create a validated, reproducible train/test split without fitting data."""
    if not isinstance(X, pd.DataFrame):
        raise EvaluationError("Features must be supplied as a pandas DataFrame.")
    if not isinstance(y, pd.Series):
        raise EvaluationError("The target must be supplied as a pandas Series.")
    if len(X) != len(y):
        raise EvaluationError("Features and target must contain the same number of rows.")
    if X.empty:
        raise EvaluationError("The dataset has no usable rows.")
    if X.shape[1] == 0:
        raise EvaluationError("At least one input feature is required.")
    if y.isna().any():
        raise EvaluationError("The target contains missing values.")
    if not isinstance(test_size, (float, int)) or not 0 < test_size < 1:
        raise EvaluationError("Test percentage must represent a fraction strictly between 0 and 1.")
    if task not in {"Classification", "Regression"}:
        raise EvaluationError("Task must be Classification or Regression.")

    split_kwargs = {"test_size": float(test_size), "random_state": int(seed)}
    if task == "Classification":
        counts = y.value_counts()
        if counts.size < 2:
            raise EvaluationError("Classification requires at least two target classes.")
        test_rows = max(1, int(np.ceil(len(y) * test_size)))
        train_rows = len(y) - test_rows
        if counts.min() < 2:
            raise EvaluationError(
                "Every target class needs at least two rows before a stratified split."
            )
        if train_rows < counts.size or test_rows < counts.size:
            raise EvaluationError(
                "The requested test size prevents every class appearing in both splits."
            )
        split_kwargs["stratify"] = y

    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, **split_kwargs)
    except ValueError as exc:
        raise EvaluationError(f"Train/test split is not feasible: {exc}") from exc

    return SplitData(X_train, X_test, y_train, y_test)


def make_cv(task: str, y_train: pd.Series, folds: int, seed: int = 42):
    """Create a shuffled CV object and validate support for the training data."""
    if task not in {"Classification", "Regression"}:
        raise EvaluationError("Task must be Classification or Regression.")
    if not isinstance(folds, int) or folds < 2:
        raise EvaluationError("Cross-validation folds must be an integer of at least 2.")
    if folds > len(y_train):
        raise EvaluationError(
            f"Requested {folds} folds, but only {len(y_train)} training rows are available."
        )

    if task == "Classification":
        smallest_class = int(y_train.value_counts().min())
        if smallest_class < folds:
            raise EvaluationError(
                f"Requested {folds} folds, but the smallest training class has only "
                f"{smallest_class} rows."
            )
        return StratifiedKFold(n_splits=folds, shuffle=True, random_state=int(seed))

    return KFold(n_splits=folds, shuffle=True, random_state=int(seed))


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Create an unfitted preprocessor for use inside a cross-validation pipeline."""
    numeric_columns = X.select_dtypes(include="number").columns.tolist()
    categorical_columns = X.select_dtypes(exclude="number").columns.tolist()
    transformers = []

    if numeric_columns:
        transformers.append((
            "numeric",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]),
            numeric_columns,
        ))
    if categorical_columns:
        transformers.append((
            "categorical",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]),
            categorical_columns,
        ))
    if not transformers:
        raise EvaluationError("No supported input features were found.")
    return ColumnTransformer(transformers)


def build_baseline_comparison(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    task: str,
    models: Mapping[str, object],
    folds: int,
    seed: int = 42,
    return_selected_pipeline: bool = False,
) -> (
    tuple[pd.DataFrame, dict[str, object], dict[str, object], str]
    | tuple[pd.DataFrame, dict[str, object], dict[str, object], str, Pipeline]
):
    """Compare baselines by training CV, then evaluate the winning pipeline once."""
    cv = make_cv(task, y_train, folds, seed)
    preprocessor = build_preprocessor(X_train)
    rows = []
    fitted = {}
    errors = {}
    scoring = (
        {
            "accuracy": "accuracy",
            "balanced_accuracy": "balanced_accuracy",
            "precision_macro": "precision_macro",
            "recall_macro": "recall_macro",
            "f1_macro": "f1_macro",
            "f1_weighted": "f1_weighted",
        }
        if task == "Classification"
        else {"mae": "neg_mean_absolute_error", "rmse": "neg_root_mean_squared_error", "r2": "r2"}
    )

    for name, model in models.items():
        try:
            pipeline = Pipeline([
                ("preprocessor", clone(preprocessor)),
                ("model", clone(model)),
            ])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UndefinedMetricWarning)
                scores = cross_validate(
                    pipeline,
                    X_train,
                    y_train,
                    cv=cv,
                    scoring=scoring,
                    n_jobs=1,
                    error_score="raise",
                )
            row = {"model": name}
            for metric in scoring:
                mean = float(np.mean(scores[f"test_{metric}"]))
                row[f"cv_{metric}_mean"] = mean
                row[f"cv_{metric}_std"] = float(np.std(scores[f"test_{metric}"]))
            rows.append(row)

            pipeline.fit(X_train, y_train)
            fitted[name] = pipeline
        except Exception as exc:
            errors[name] = str(exc)

    if not rows:
        raise EvaluationError(
            "No baseline model completed cross-validation: "
            + "; ".join(f"{name}: {error}" for name, error in errors.items())
        )

    comparison = pd.DataFrame(rows)
    selection_metric = "cv_f1_macro_mean" if task == "Classification" else "cv_mae_mean"
    ascending = task == "Regression"
    winner = comparison.sort_values(selection_metric, ascending=ascending).iloc[0]["model"]
    test_metrics = evaluate_predictions(
        y_test,
        fitted[winner].predict(X_test),
        task,
        fitted[winner],
        X_test,
    )
    if return_selected_pipeline:
        return comparison, test_metrics, errors, str(winner), fitted[str(winner)]
    return comparison, test_metrics, errors, str(winner)


def evaluate_predictions(
    y_true: pd.Series,
    y_pred: np.ndarray,
    task: str,
    fitted_model: object | None = None,
    X_test: pd.DataFrame | None = None,
) -> dict[str, object]:
    """Calculate final held-out metrics and plot-ready prediction data."""
    if task == "Classification":
        labels = sorted(pd.unique(y_true))
        metrics: dict[str, object] = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
            "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
            "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
            "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
            "classification_report": classification_report(
                y_true, y_pred, labels=labels, output_dict=True, zero_division=0
            ),
            "labels": labels,
            "predictions": pd.Series(y_pred),
            "actual": y_true.reset_index(drop=True),
        }

        if fitted_model is not None and X_test is not None:
            probability = _positive_or_multiclass_probability(fitted_model, X_test)
            if probability is not None:
                try:
                    classes = list(fitted_model.classes_)
                    if probability.shape[1] == 2:
                        positive_index = list(classes).index(labels[-1])
                        metrics["roc_auc"] = float(roc_auc_score(y_true, probability[:, positive_index]))
                    elif probability.shape[1] == len(labels):
                        metrics["roc_auc"] = float(
                            roc_auc_score(y_true, probability, multi_class="ovr", labels=labels)
                        )
                except (AttributeError, IndexError, ValueError):
                    pass
        return metrics

    residuals = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    metrics = {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "r2": float(r2_score(y_true, y_pred)),
        "actual": y_true.reset_index(drop=True),
        "predictions": pd.Series(y_pred),
        "residuals": pd.Series(residuals),
    }
    return metrics


def _positive_or_multiclass_probability(fitted_model: object, X_test: pd.DataFrame):
    predictor = getattr(fitted_model, "predict_proba", None)
    if callable(predictor):
        return predictor(X_test)
    decision = getattr(fitted_model, "decision_function", None)
    if callable(decision):
        scores = decision(X_test)
        return scores if scores.ndim == 2 else np.column_stack([-scores, scores])
    return None
