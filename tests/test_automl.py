from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import optuna
from sklearn.pipeline import Pipeline

import src.automl as automl
from src.evaluation import EvaluationError, reproducible_split


def classification_data(rows=40):
    rng = np.random.default_rng(7)
    X = pd.DataFrame({
        "hours": rng.normal(size=rows),
        "category": rng.choice(["A", "B"], size=rows),
    })
    y = pd.Series(["fail", "pass"] * (rows // 2), name="target")
    return X, y


def regression_data(rows=40):
    rng = np.random.default_rng(11)
    X = pd.DataFrame({"x": rng.normal(size=rows), "z": rng.normal(size=rows)})
    y = pd.Series(2 * X["x"] + rng.normal(scale=0.1, size=rows), name="target")
    return X, y


def test_automl_classification_is_reproducible(monkeypatch):
    monkeypatch.setattr(automl, "_model_options", lambda task: ("logistic",))
    X, y = classification_data()
    split = reproducible_split(X, y, "Classification", 0.25, 42)

    first = automl.run_automl_search(
        split.X_train, split.y_train, split.X_test, split.y_test,
        "Classification", "f1_macro", n_trials=2, folds=2,
    )
    second = automl.run_automl_search(
        split.X_train, split.y_train, split.X_test, split.y_test,
        "Classification", "f1_macro", n_trials=2, folds=2,
    )

    assert first.best_pipeline.predict(split.X_test).tolist() == second.best_pipeline.predict(split.X_test).tolist()
    assert first.metadata["input_features"] == ["hours", "category"]
    assert "target" not in first.metadata["input_features"]


def test_automl_regression_uses_regression_score(monkeypatch):
    monkeypatch.setattr(automl, "_model_options", lambda task: ("ridge",))
    X, y = regression_data()
    split = reproducible_split(X, y, "Regression", 0.25, 42)

    result = automl.run_automl_search(
        split.X_train, split.y_train, split.X_test, split.y_test,
        "Regression", "neg_mae", n_trials=2, folds=2,
    )

    assert result.final_metrics["mae"] >= 0
    assert result.metadata["scoring"] == "neg_mean_absolute_error"
    assert result.comparison.loc[0, "cv_score"] >= 0


def test_feature_selection_pipeline_is_inside_cv(monkeypatch):
    monkeypatch.setattr(automl, "_model_options", lambda task: ("logistic",))
    X, y = classification_data()
    trial = optuna.trial.FixedTrial({
        "model": "logistic", "C": 1.0,
        "feature_method": "mutual_info", "feature_percentile": 80,
    })
    pipeline = automl.build_automl_pipeline(trial, X, "Classification", 42, ["mutual_info"])

    assert isinstance(pipeline, Pipeline)
    assert [step[0] for step in pipeline.steps] == ["preprocessor", "feature_selection", "model"]
    assert pipeline.named_steps["feature_selection"].percentile == 80
    pipeline.fit(X, y)


def test_save_automl_result_creates_timestamped_artifacts(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(automl, "_model_options", lambda task: ("logistic",))
    X, y = classification_data()
    split = reproducible_split(X, y, "Classification", 0.25, 42)
    result = automl.run_automl_search(
        split.X_train, split.y_train, split.X_test, split.y_test,
        "Classification", "f1_macro", n_trials=1, folds=2,
    )

    first = automl.save_automl_result(result, tmp_path, "synthetic")
    second = automl.save_automl_result(result, tmp_path, "synthetic")

    assert first["model"] != second["model"]
    assert joblib.load(first["model"])
    assert len(list(tmp_path.glob("automl_*_trials.csv"))) == 2


def test_failed_trials_report_error_without_hiding_all_failure(monkeypatch):
    monkeypatch.setattr(automl, "_model_options", lambda task: ("broken",))
    monkeypatch.setattr(
        automl,
        "_classification_model",
        lambda trial, model_name, seed: (_ for _ in ()).throw(RuntimeError("deliberate failure")),
    )
    X, y = classification_data()
    split = reproducible_split(X, y, "Classification", 0.25, 42)

    try:
        automl.run_automl_search(
            split.X_train, split.y_train, split.X_test, split.y_test,
            "Classification", "f1_macro", n_trials=2, folds=2,
        )
    except EvaluationError as exc:
        assert "All Optuna trials failed" in str(exc)
    else:
        raise AssertionError("Expected all-failed trials to raise a clear error")


def test_automl_rejects_bad_trials_and_scoring(monkeypatch):
    monkeypatch.setattr(automl, "_model_options", lambda task: ("logistic",))
    X, y = classification_data()
    split = reproducible_split(X, y, "Classification", 0.25, 42)

    for trials in (0, -1, True):
        try:
            automl.run_automl_search(
                split.X_train, split.y_train, split.X_test, split.y_test,
                "Classification", "f1_macro", n_trials=trials, folds=2,
            )
        except EvaluationError as exc:
            assert "trial count" in str(exc)
        else:
            raise AssertionError("Expected invalid trial count to fail")

    try:
        automl.run_automl_search(
            split.X_train, split.y_train, split.X_test, split.y_test,
            "Classification", "roc_auc", n_trials=1, folds=2,
        )
    except EvaluationError as exc:
        assert "Unsupported Classification scoring" in str(exc)
    else:
        raise AssertionError("Expected invalid scoring to fail")
