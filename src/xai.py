"""Lightweight, model-agnostic and model-specific explanations for fitted pipelines."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline


class ExplanationError(ValueError):
    """Raised when an explanation cannot be produced safely."""


@dataclass(frozen=True)
class LocalExplanation:
    prediction: object
    probabilities: dict[str, float] | None
    contributions: pd.DataFrame
    method: str


def _pipeline_steps(pipeline: Pipeline) -> tuple[list[str], object]:
    if not isinstance(pipeline, Pipeline) or not pipeline.steps:
        raise ExplanationError("A fitted scikit-learn Pipeline is required.")
    if "model" not in pipeline.named_steps:
        raise ExplanationError("The pipeline does not contain a named model step.")
    model = pipeline.named_steps["model"]
    if not hasattr(model, "predict"):
        raise ExplanationError("The model does not support prediction.")
    return pipeline.feature_names_in_.tolist() if hasattr(pipeline, "feature_names_in_") else [], model


def transformed_feature_names(pipeline: Pipeline) -> list[str]:
    """Return feature names after preprocessing and optional feature selection."""
    if not pipeline.steps:
        raise ExplanationError("A fitted pipeline is required.")

    preprocessor = pipeline.named_steps.get("preprocessor")
    if preprocessor is None or not hasattr(preprocessor, "get_feature_names_out"):
        raise ExplanationError("The pipeline has no readable fitted preprocessor.")
    try:
        names = list(preprocessor.get_feature_names_out())
    except Exception as exc:
        raise ExplanationError(f"Preprocessing feature names are unavailable: {exc}") from exc

    selector = pipeline.named_steps.get("feature_selection")
    if selector is not None:
        try:
            support = np.asarray(selector.get_support(), dtype=bool)
        except Exception as exc:
            raise ExplanationError(f"Feature-selection support is unavailable: {exc}") from exc
        if len(support) != len(names):
            raise ExplanationError("Feature-selection support does not align with transformed features.")
        names = [name for name, selected in zip(names, support) if selected]
    return names


def original_name_mapping(pipeline: Pipeline) -> list[str]:
    """Map selected encoded features to original input columns."""
    encoded = transformed_feature_names(pipeline)
    preprocessor = pipeline.named_steps["preprocessor"]
    raw_to_selected = {}
    for transformer_name, transformer, columns in preprocessor.transformers_:
        if transformer_name == "remainder" or transformer == "drop":
            continue
        try:
            output_names = list(transformer.get_feature_names_out(columns))
        except Exception:
            output_names = [f"{transformer_name}__{column}" for column in columns]
        for output_name in output_names:
            matching_column = next(
                (str(column) for column in columns if str(output_name) == str(column) or str(output_name).startswith(f"{column}_")),
                str(output_name),
            )
            raw_to_selected[f"{transformer_name}__{output_name}"] = matching_column

    mapping = []
    for name in encoded:
        original = raw_to_selected.get(name)
        if original is not None:
            mapping.append(original)
        elif "__" in name:
            transformer_name, suffix = name.split("__", 1)
            for column in pipeline.feature_names_in_.tolist():
                if suffix == column or suffix.startswith(f"{column}_"):
                    original = column
                    break
            else:
                original = transformer_name
            mapping.append(original)
        else:
            mapping.append(name)
    return mapping


def permutation_explanation(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    task: str,
    seed: int = 42,
    n_repeats: int = 5,
) -> pd.DataFrame:
    """Calculate held-out permutation importance and aggregate it by input column."""
    if X_test.empty or y_test.empty:
        raise ExplanationError("Held-out test data is required.")
    if len(X_test) != len(y_test):
        raise ExplanationError("Held-out features and target lengths differ.")
    if y_test.name and y_test.name in X_test.columns:
        raise ExplanationError("The target column must not be included in explanation features.")

    scoring = "f1_macro" if task == "Classification" else "neg_mean_absolute_error"
    try:
        result = permutation_importance(
            pipeline,
            X_test,
            y_test,
            scoring=scoring,
            n_repeats=max(2, int(n_repeats)),
            random_state=int(seed),
            n_jobs=1,
        )
    except Exception as exc:
        raise ExplanationError(f"Permutation importance is unavailable: {exc}") from exc

    names = X_test.columns.astype(str).tolist()
    if len(names) != len(result.importances_mean):
        raise ExplanationError("Permutation importance does not align with input columns.")
    return pd.DataFrame({
        "feature": names,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    }).sort_values("importance_mean", ascending=False, ignore_index=True)


def model_specific_explanation(pipeline: Pipeline) -> tuple[pd.DataFrame, str]:
    """Return native tree importances or linear coefficients, when supported."""
    _, model = _pipeline_steps(pipeline)
    names = transformed_feature_names(pipeline)

    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_, dtype=float)
        method = "Native tree feature importance"
    elif hasattr(model, "coef_"):
        coef = np.asarray(model.coef_, dtype=float)
        if coef.ndim == 1:
            values = np.abs(coef)
            method = "Absolute linear coefficient (one target direction)"
        else:
            values = np.mean(np.abs(coef), axis=0)
            method = "Mean absolute linear coefficient across classes"
    else:
        raise ExplanationError(
            f"{type(model).__name__} does not expose tree feature importances or coefficients."
        )

    if len(values) != len(names):
        raise ExplanationError("Model-specific importance does not align with transformed features.")
    frame = pd.DataFrame({"feature": names, "importance": values})
    frame["original_feature"] = original_name_mapping(pipeline)
    return frame.sort_values("importance", ascending=False, ignore_index=True), method


def predict_with_explanation(pipeline: Pipeline, record: pd.DataFrame, task: str) -> LocalExplanation:
    """Explain one prediction using bounded, local one-feature perturbations."""
    if not isinstance(pipeline, Pipeline):
        raise ExplanationError("A fitted pipeline is required for individual explanations.")
    if record.shape[0] != 1:
        raise ExplanationError("Exactly one prediction record is required.")
    feature_names = [str(name) for name in record.columns]
    if len(feature_names) != len(set(feature_names)):
        raise ExplanationError("Prediction input column names must be unique.")

    prediction = pipeline.predict(record)[0]
    probabilities = None
    if task == "Classification" and hasattr(pipeline, "predict_proba"):
        try:
            scores = np.asarray(pipeline.predict_proba(record))[0]
            probabilities = {
                str(label): float(score)
                for label, score in zip(pipeline.classes_, scores)
            }
        except Exception:
            probabilities = None

    selected_class = str(prediction)
    base_score = None
    if probabilities is not None:
        base_score = probabilities[selected_class]
    elif task == "Regression":
        base_score = float(prediction)

    rows = []
    for feature in record.columns:
        values = record[[feature]]
        original = record.at[record.index[0], feature]
        perturbed = record.copy()
        if pd.api.types.is_numeric_dtype(record[feature].dtype):
            training_feature = feature
            spread = float(values[training_feature].std()) if values[training_feature].notna().sum() > 1 else 0.0
            replacement = float(original) + (spread or 1.0)
        else:
            replacement = "<perturbed unseen category>"
            perturbed[feature] = perturbed[feature].astype(object)
        perturbed.at[record.index[0], feature] = replacement

        try:
            perturbed_prediction = pipeline.predict(perturbed)[0]
            if base_score is not None:
                if probabilities is not None:
                    perturbed_probabilities = dict(zip(pipeline.classes_, pipeline.predict_proba(perturbed)[0]))
                    perturbed_score = float(perturbed_probabilities.get(str(prediction), 0.0))
                else:
                    perturbed_score = float(perturbed_prediction)
                effect = perturbed_score - base_score
            else:
                effect = 0.0 if perturbed_prediction == prediction else float(perturbed_prediction != prediction)
            rows.append({
                "feature": str(feature),
                "current_value": original,
                "perturbed_value": replacement,
                "effect": float(effect),
            })
        except Exception:
            continue

    if not rows:
        raise ExplanationError("No local perturbation produced a model-compatible result.")
    frame = pd.DataFrame(rows).sort_values("effect", key=lambda s: s.abs(), ascending=False, ignore_index=True)
    method = (
        "Local one-feature perturbation of selected output. For classification, changes are shown "
        "in predicted-class probability; for regression, changes are shown in the numeric prediction."
        if probabilities is not None or task == "Regression"
        else "Local one-feature prediction-change indicator"
    )
    return LocalExplanation(prediction, probabilities, frame, method)

