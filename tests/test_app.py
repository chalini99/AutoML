from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

import src.automl as automl
import src.evaluation as evaluation


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def test_app_starts_without_an_upload():
    app = AppTest.from_file(str(APP_PATH), default_timeout=10)
    app.run()
    assert not app.exception
    assert app.info
    assert app.radio(key="student_automl_workspace").value == "Student Success Dashboard"
    assert any(metric.label == "Students in cohort" for metric in app.metric)


def test_app_runs_optuna_when_enabled(monkeypatch, tmp_path):
    comparison = pd.DataFrame([{
        "model": "Optuna logistic",
        "feature_method": "none",
        "cv_score": 0.75,
        "scoring": "f1_macro",
    }])
    trials = pd.DataFrame([{
        "number": 0,
        "status": "COMPLETE",
        "model": "logistic",
        "feature_method": "none",
        "cv_score": 0.75,
    }])

    class FakeResult:
        def __init__(self):
            self.study = None
            self.best_pipeline = None
            self.comparison = comparison
            self.trials = trials
            self.final_metrics = {"accuracy": 0.8, "f1_macro": 0.75}
            self.metadata = {"model_family": "logistic", "feature_method": "none"}

    monkeypatch.setattr(automl, "run_automl_search", lambda *args, **kwargs: FakeResult())
    monkeypatch.setattr(
        automl,
        "save_automl_result",
        lambda *args, **kwargs: {"model": str(tmp_path / "model.joblib"), "metadata": str(tmp_path / "metadata.json")},
    )
    rows = ["G1;G2;G3;school"]
    for i in range(60):
        grade = 10 + (i % 8)
        rows.append(f"{grade - 2};{grade - 1};{grade};GP")

    app = AppTest.from_file(str(APP_PATH), default_timeout=30)
    app.run()
    app.file_uploader[0].set_value(("student.csv", "\n".join(rows).encode(), "text/csv"))
    app.run()
    app.checkbox[0].set_value(True)
    app.button[0].click()
    app.run()

    assert not app.exception
    assert any("Optuna final held-out evaluation" in text.value for text in app.subheader)
    assert any("Explainable AI" in text.value for text in app.subheader)
    assert any("Dataset Overview" == tab.label for tab in app.tabs)
    assert any("Saved Experiments" == tab.label for tab in app.tabs)
    assert any("Rigorous Validation" == tab.label for tab in app.tabs)


def test_app_runs_semicolon_upload_with_cross_validation():
    rows = ["G1;G2;G3;school"]
    for i in range(60):
        grade = 10 + (i % 8)
        rows.append(f"{grade - 2};{grade - 1};{grade};GP")

    app = AppTest.from_file(str(APP_PATH), default_timeout=30)
    app.run()
    app.file_uploader[0].set_value(
        ("student.csv", "\n".join(rows).encode("utf-8"), "text/csv")
    )
    app.run()
    assert not app.exception
    assert app.success

    app.button[0].click()
    app.run()
    assert not app.exception
    assert any("Selected from training CV" in text.value for text in app.success)


def test_student_predictions_score_batch_and_ignore_untrained_columns():
    rows = ["G3;studytime;absences;school"]
    for i in range(90):
        grade = 12 if i % 2 else 8
        rows.append(f"{grade};{i % 4 + 1};{i % 7};{'GP' if i % 3 else 'MS'}")

    app = AppTest.from_file(str(APP_PATH), default_timeout=30)
    app.run()
    app.file_uploader[0].set_value(("students.csv", "\n".join(rows).encode(), "text/csv"))
    app.run()
    app.button[0].click()
    app.run()
    assert not app.exception

    app.radio(key="student_automl_workspace").set_value("Student Predictions")
    app.run()
    single_student_button = next(
        button for button in app.button if button.label == "Predict student outcome"
    )
    single_student_button.click()
    app.run()
    assert not app.exception
    assert any(metric.label == "Predicted outcome" for metric in app.metric)

    app.file_uploader(key="prediction_csv").set_value(
        (
            "new_students.csv",
            b"studytime,absences,school,G3\n2,0,GP,18\n4,3,MS,7\n",
            "text/csv",
        )
    )
    app.run()
    app.button(key="generate_predictions").click()
    app.run()

    assert not app.exception
    assert any("Students scored" == metric.label for metric in app.metric)
    assert any(
        "Additional columns were ignored: G3" in element.value
        for element in app.info
    )
    assert app.download_button(key="download_prediction_results")

    app.radio(key="student_automl_workspace").set_value("Explainable AI")
    app.run()
    assert not app.exception
    assert any("Global permutation importance" in item.value for item in app.subheader)


def test_dashboard_does_not_retrain_for_another_run(monkeypatch):
    calls = 0
    real_comparison = evaluation.build_baseline_comparison

    def counted_comparison(*args, **kwargs):
        nonlocal calls
        calls += 1
        return real_comparison(*args, **kwargs)

    monkeypatch.setattr(evaluation, "build_baseline_comparison", counted_comparison)
    rows = ["G1;G2;G3;school"]
    for i in range(60):
        grade = 10 + (i % 8)
        rows.append(f"{grade - 2};{grade - 1};{grade};GP")

    app = AppTest.from_file(str(APP_PATH), default_timeout=30)
    app.run()
    app.file_uploader[0].set_value(("student.csv", "\n".join(rows).encode(), "text/csv"))
    app.run()
    app.button[0].click()
    app.run()
    app.run()

    assert not app.exception
    assert calls == 1


def test_saved_experiments_display_metadata_without_loading_models(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    models = tmp_path / "models"
    models.mkdir()
    (models / "experiment.json").write_text(
        '{"timestamp_utc":"2026-01-01","task":"Classification","model_family":"Logistic Regression",'
        '"feature_method":"none","scoring":"f1_macro","cv_folds":5,"cv_score":0.8,"dataset_identifier":"student.csv"}',
        encoding="utf-8",
    )
    (models / "experiment.joblib").write_bytes(b"not a valid or trusted model")

    app = AppTest.from_file(str(APP_PATH), default_timeout=10)
    app.run()
    app.file_uploader[0].set_value(("student.csv", b"G1;G2;G3\n10;11;12\n", "text/csv"))
    app.run()

    assert not app.exception
    assert any("Metadata is read from JSON files only" in text.value for text in app.caption)
