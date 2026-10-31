import numpy as np
import pandas as pd
import pytest
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

from src.evaluation import build_preprocessor, reproducible_split
from src.xai import (
    ExplanationError,
    model_specific_explanation,
    original_name_mapping,
    permutation_explanation,
    predict_with_explanation,
    transformed_feature_names,
)


def classification_pipeline():
    rng = np.random.default_rng(14)
    X = pd.DataFrame({
        "hours": rng.normal(5, 2, 80),
        "age": rng.integers(15, 20, 80),
        "school": rng.choice(["GP", "MS"], 80),
    })
    y = pd.Series((X["hours"] > X["hours"].median()).astype(int), name="target")
    pipe = Pipeline([
        ("preprocessor", build_preprocessor(X)),
        ("feature_selection", SelectFromModel(LogisticRegression(max_iter=1000), threshold="median")),
        ("model", LogisticRegression(max_iter=1000, random_state=42)),
    ])
    split = reproducible_split(X, y, "Classification", 0.25, 42)
    pipe.fit(split.X_train, split.y_train)
    return pipe, split


def test_feature_names_align_after_preprocessing_and_selection():
    pipe, split = classification_pipeline()
    transformed = transformed_feature_names(pipe)
    original = original_name_mapping(pipe)

    assert len(transformed) == len(original)
    assert set(original).issubset(set(split.X_train.columns))
    assert all("target" not in name for name in transformed)


def test_permutation_importance_uses_held_out_data_and_aligns_features():
    pipe, split = classification_pipeline()
    result = permutation_explanation(pipe, split.X_test, split.y_test, "Classification")

    assert set(result.columns) == {"feature", "importance_mean", "importance_std"}
    assert set(result["feature"]) == set(split.X_test.columns)
    assert len(result) == split.X_test.shape[1]
    assert (result["importance_std"] >= 0).all()


def test_linear_model_specific_explanation_and_individual_prediction():
    pipe, split = classification_pipeline()
    table, method = model_specific_explanation(pipe)

    assert "coefficient" in method.lower()
    assert set(table["original_feature"]).issubset(set(split.X_train.columns))

    explanation = predict_with_explanation(pipe, split.X_test.iloc[[0]], "Classification")
    assert explanation.prediction in {0, 1}
    assert set(explanation.probabilities) == {"0", "1"}
    assert explanation.contributions["feature"].is_unique
    assert "perturbation" in explanation.method.lower()


def test_regression_explanation_and_prediction():
    rng = np.random.default_rng(31)
    X = pd.DataFrame({"x": rng.normal(size=60), "z": rng.normal(size=60)})
    y = pd.Series(3 * X["x"] - X["z"] + rng.normal(scale=0.1, size=60), name="target")
    split = reproducible_split(X, y, "Regression", 0.25, 42)
    pipe = Pipeline([
        ("preprocessor", build_preprocessor(X)),
        ("model", Ridge()),
    ])
    pipe.fit(split.X_train, split.y_train)

    importance = permutation_explanation(pipe, split.X_test, split.y_test, "Regression")
    native, method = model_specific_explanation(pipe)
    local = predict_with_explanation(pipe, split.X_test.iloc[[0]], "Regression")

    assert len(importance) == 2
    assert "coefficient" in method.lower()
    assert native["importance"].ge(0).all()
    assert isinstance(local.prediction, (float, np.floating))
    assert local.probabilities is None


def test_target_column_is_not_accepted_for_importance():
    pipe, split = classification_pipeline()
    X_with_target = split.X_test.copy()
    X_with_target["target"] = split.y_test.values

    with pytest.raises(ExplanationError, match="target"):
        permutation_explanation(pipe, X_with_target, split.y_test, "Classification")


def test_unsupported_native_explanation_reports_method_limit():
    X = pd.DataFrame({"x": [0, 1, 2, 3, 4, 5], "y": [1, 1, 0, 1, 0, 0]})
    y = pd.Series([0, 0, 1, 1, 1, 1])
    pipe = Pipeline([
        ("preprocessor", build_preprocessor(X)),
        ("model", KNeighborsClassifier()),
    ])
    pipe.fit(X, y)

    with pytest.raises(ExplanationError, match="does not expose"):
        model_specific_explanation(pipe)
