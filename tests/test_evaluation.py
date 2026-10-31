import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier, DummyRegressor

from src.evaluation import (
    EvaluationError,
    build_baseline_comparison,
    evaluate_predictions,
    make_cv,
    reproducible_split,
)


def classification_data(seed=42, rows=60):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"age": rng.integers(15, 20, rows), "hours": rng.normal(5, 2, rows)})
    y = pd.Series(rng.choice(["fail", "pass"], rows), name="target")
    X["numeric_missing"] = rng.choice([1.0, np.nan], rows)
    X["category"] = rng.choice(["A", "B", "C"], rows)
    return X, y


def regression_data(seed=42, rows=60):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"age": rng.integers(15, 20, rows), "hours": rng.normal(5, 2, rows)})
    y = pd.Series(10 + 0.5 * X["age"] + X["hours"] + rng.normal(0, 1, rows), name="G3")
    return X, y


def test_reproducible_stratified_split():
    X, y = classification_data()
    first = reproducible_split(X, y, "Classification", 0.2, 42)
    second = reproducible_split(X, y, "Classification", 0.2, 42)
    pd.testing.assert_frame_equal(first.X_train, second.X_train)
    pd.testing.assert_series_equal(first.y_test, second.y_test)
    assert set(first.y_train) == {"fail", "pass"}
    assert set(first.y_test) == {"fail", "pass"}


def test_regression_split_is_reproducible():
    X, y = regression_data()
    first = reproducible_split(X, y, "Regression", 0.25, 7)
    second = reproducible_split(X, y, "Regression", 0.25, 7)
    pd.testing.assert_frame_equal(first.X_train, second.X_train)
    pd.testing.assert_series_equal(first.y_test, second.y_test)


@pytest.mark.parametrize("test_size", [0, 1, -0.1, 1.2])
def test_invalid_test_sizes(test_size):
    X, y = classification_data()
    with pytest.raises(EvaluationError, match="between 0 and 1"):
        reproducible_split(X, y, "Classification", test_size, 42)


def test_rare_class_is_not_silently_split():
    X = pd.DataFrame({"x": range(10)})
    y = pd.Series(["rare"] + ["common"] * 9)
    with pytest.raises(EvaluationError, match="at least two"):
        reproducible_split(X, y, "Classification", 0.2, 42)


def test_missing_target_and_fold_errors():
    X = pd.DataFrame({"x": range(10)})
    y = pd.Series([0, 1, 0, 1, 0, 1, 0, 1, 0, np.nan])
    with pytest.raises(EvaluationError, match="missing"):
        reproducible_split(X, y, "Classification", 0.2, 42)

    y = pd.Series([0, 1] * 5)
    split = reproducible_split(X, y, "Classification", 0.2, 42)
    with pytest.raises(EvaluationError, match="smallest training class"):
        make_cv("Classification", split.y_train, 5, 42)


def test_classification_metrics_and_report():
    y_true = pd.Series(["fail", "pass", "fail", "pass"])
    metrics = evaluate_predictions(y_true, np.array(["fail", "fail", "fail", "pass"]), "Classification")
    assert metrics["confusion_matrix"] == [[2, 0], [1, 1]]
    assert metrics["classification_report"]
    assert 0 <= metrics["balanced_accuracy"] <= 1
    assert "roc_auc" not in metrics


def test_regression_metrics_and_residuals():
    y_true = pd.Series([1.0, 2.0, 3.0])
    metrics = evaluate_predictions(y_true, np.array([1.5, 2.0, 2.5]), "Regression")
    np.testing.assert_allclose(metrics["residuals"], [-0.5, 0.0, 0.5])
    assert metrics["mae"] == pytest.approx(1 / 3)
    assert metrics["rmse"] >= metrics["mae"]


def test_baseline_comparison_uses_same_split_and_continues_after_failure():
    X, y = classification_data(rows=120)
    split = reproducible_split(X, y, "Classification", 0.25, 42)

    comparison, final, errors, winner = build_baseline_comparison(
        split.X_train,
        split.y_train,
        split.X_test,
        split.y_test,
        "Classification",
        {
            "Dummy": DummyClassifier(strategy="most_frequent"),
            "Broken": _BrokenClassifier(),
        },
        folds=3,
        seed=42,
    )

    assert comparison["model"].tolist() == ["Dummy"]
    assert winner == "Dummy"
    assert "Broken" in errors
    assert final["confusion_matrix"]
    assert len(split.X_test) == 30


def test_identical_splits_across_model_comparisons():
    X, y = regression_data(rows=80)
    split = reproducible_split(X, y, "Regression", 0.2, 42)
    models = {"Dummy": DummyRegressor()}
    first = build_baseline_comparison(
        split.X_train, split.y_train, split.X_test, split.y_test,
        "Regression", models, folds=4, seed=42,
    )
    second = build_baseline_comparison(
        split.X_train, split.y_train, split.X_test, split.y_test,
        "Regression", models, folds=4, seed=42,
    )
    pd.testing.assert_frame_equal(first[0], second[0])
    assert first[2] == second[2]
    assert first[1]["residuals"].tolist() == second[1]["residuals"].tolist()


def test_target_is_never_added_to_feature_matrix():
    X, y = classification_data()
    split = reproducible_split(X, y, "Classification", 0.2, 42)
    assert y.name not in split.X_train.columns
    assert y.name not in split.X_test.columns


class _BrokenClassifier(DummyClassifier):
    def fit(self, X, y):
        raise RuntimeError("deliberate test failure")
