from pathlib import Path

import numpy as np
import pandas as pd

from src.early_classification import (
    early_classification_features,
    run_early_classification,
    save_early_classification_report,
)
from src.evaluation import EvaluationError


def synthetic_frame(rows=80):
    rng = np.random.default_rng(17)
    frame = pd.DataFrame({
        "school": rng.choice(["GP", "MS"], rows),
        "sex": rng.choice(["F", "M"], rows),
        "age": rng.integers(15, 20, rows),
        "studytime": rng.integers(1, 5, rows),
        "failures": rng.integers(0, 4, rows),
    })
    frame["G1"] = rng.integers(5, 20, rows)
    frame["G2"] = rng.integers(5, 20, rows)
    frame["G3"] = np.where(
        0.25 * frame["G1"] + 0.55 * frame["G2"] + frame["studytime"] + rng.normal(0, 2, rows) >= 10,
        rng.integers(10, 20, rows),
        rng.integers(0, 10, rows),
    )
    return frame


def test_early_features_exclude_all_post_prediction_outcomes():
    X, y = early_classification_features(synthetic_frame())
    assert set(X.columns).isdisjoint({"G1", "G2", "G3"})
    assert set(y) == {0, 1}
    assert y.name == "G3_pass_fail"


def test_early_features_exclude_target_derived_aliases_and_case_variants():
    frame = synthetic_frame()
    frame["G3_pass_fail"] = frame["G3"].ge(10).astype("int8")
    frame["G3_category"] = frame["G3"].ge(10).map({True: "Pass", False: "Fail"})
    frame["pass_fail"] = frame["G3"].ge(10).astype("int8")
    frame["g1"] = frame["G1"]
    X, _ = early_classification_features(frame)
    assert set(X.columns).isdisjoint({
        "G1", "G2", "G3", "G3_pass_fail", "G3_category", "pass_fail", "g1",
    })


def test_early_feature_validation_rejects_missing_grade_columns():
    frame = synthetic_frame().drop(columns="G2")
    try:
        early_classification_features(frame)
    except EvaluationError:
        pass
    else:
        raise AssertionError("Missing G1/G2/G3 must fail")


def test_early_comparison_uses_identical_held_out_split_and_requested_models(tmp_path):
    report = run_early_classification(
        synthetic_frame(), "synthetic.csv", seeds=(42,), folds=3, optuna_trials=2,
    )
    assert report["excluded_features"] == ["G1", "G2", "G3"]
    assert report["selection_metric"] == "training_cv_f1_macro"
    assert report["feature_columns"] == ["school", "sex", "age", "studytime", "failures"]
    runs = report["seed_runs"]
    assert len(runs) == 1
    assert len(runs[0]["test_indices"]) == 16
    models = {row["model"] for row in runs[0]["models"]}
    assert "Majority baseline" in models
    assert {
        "Logistic Regression", "Class-weighted Logistic Regression",
        "Random Forest", "Class-weighted Random Forest", "Extra Trees", "SVM",
    }.issubset(models)
    selected = report["seed_runs"][0]["selected_model"]
    selected_row = next(row for row in report["seed_runs"][0]["models"] if row["model"] == selected)
    assert selected_row["cv_macro_f1_mean"] == max(
        row["cv_macro_f1_mean"] for row in report["seed_runs"][0]["models"]
    )
    assert report["selected_model_held_out_summary"]["held_out_macro_f1"]["mean"] == selected_row["held_out_macro_f1"]
    json_path = save_early_classification_report(report, tmp_path)
    csv_path = json_path.with_suffix(".csv")
    assert json_path.is_file()
    assert csv_path.is_file()
    original_json = json_path.read_text(encoding="utf-8")
    second_json_path = save_early_classification_report(report, tmp_path)
    assert second_json_path != json_path
    assert json_path.read_text(encoding="utf-8") == original_json
    assert second_json_path.with_suffix(".csv").is_file()
    csv_report = pd.read_csv(csv_path)
    assert "selected_by_training_cv" in csv_report
    assert set(csv_report["model"]) == models
    for row in runs[0]["models"]:
        assert len(row["held_out_confusion_matrix"]["matrix"]) == 2
        assert "held_out_balanced_accuracy" in row
        assert set(row["held_out_f1_by_class"]) == {"0", "1"}
        assert sum(row["held_out_predicted_class_distribution"].values()) == len(runs[0]["test_indices"])


def test_no_early_source_file_is_modified_by_report_generation(tmp_path):
    source = synthetic_frame()
    before = source.to_csv(index=False)
    run_early_classification(source, "synthetic.csv", seeds=(42,), folds=3, optuna_trials=1)
    assert source.to_csv(index=False) == before


def test_bundled_data_paths_exist_for_focused_experiment():
    root = Path(__file__).resolve().parents[1]
    assert (root / "data" / "raw" / "student-mat.csv").is_file()
    assert (root / "data" / "raw" / "student-por.csv").is_file()
