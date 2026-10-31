from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation import EvaluationError
from src.validation import assess_overlap, profile_dataset, run_validation_experiment


def student_frame(rows=60, seed=3):
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({
        "school": rng.choice(["GP", "MS"], rows),
        "sex": rng.choice(["F", "M"], rows),
        "age": rng.integers(15, 20, rows),
        "studytime": rng.integers(1, 5, rows),
        "failures": rng.integers(0, 4, rows),
        "G1": rng.integers(5, 20, rows),
        "G2": rng.integers(5, 20, rows),
    })
    frame["G3"] = np.clip(0.3 * frame["G1"] + 0.5 * frame["G2"] + rng.normal(0, 2, rows), 0, 20).round().astype(int)
    return frame


def test_profile_existing_datasets():
    root = Path(__file__).resolve().parents[1]
    math = profile_dataset(root / "data" / "raw" / "student-mat.csv")
    portuguese = profile_dataset(root / "data" / "raw" / "student-por.csv")
    assert math.shape == (395, 33)
    assert portuguese.shape == (649, 33)
    assert math.delimiter == portuguese.delimiter == ";"
    assert math.target_available and portuguese.target_available


def test_early_scenario_excludes_grade_leakage_features():
    frame = student_frame()
    report = run_validation_experiment(frame, "synthetic.csv", "Classification", "early", seeds=(42,), folds=2)
    assert report["excluded_features"] == ["G1", "G2"]
    assert report["target_rule"] == "G3 >= 10 -> Pass; otherwise Fail"
    assert all("G3" not in row for row in report["seed_runs"][0]["cv_rows"])


def test_grade_assisted_includes_grades_but_not_target():
    frame = student_frame()
    report = run_validation_experiment(frame, "synthetic.csv", "Regression", "grade_assisted", seeds=(42,), folds=2)
    assert report["excluded_features"] == []
    assert report["target_rule"] is None
    assert all("G3" not in row for row in report["seed_runs"][0]["cv_rows"])


def test_validation_reproducible_across_same_seeds():
    frame = student_frame()
    first = run_validation_experiment(frame, "synthetic.csv", "Classification", "early", seeds=(42, 43), folds=2)
    second = run_validation_experiment(frame, "synthetic.csv", "Classification", "early", seeds=(42, 43), folds=2)
    assert first["cv_summary"] == second["cv_summary"]
    assert first["held_out_summary"] == second["held_out_summary"]


def test_overlap_is_not_treated_as_identity():
    left = pd.DataFrame({"school": ["GP"], "age": [16], "G1": [10], "G2": [11], "G3": [12]})
    right = pd.DataFrame({"school": ["GP"], "age": [16], "G1": [10], "G2": [11], "G3": [14]})
    overlap = assess_overlap(left, right)
    assert overlap["attribute_overlap_rows"] == 1
    assert overlap["exact_shared_row_overlap"] == 0


def test_report_metadata_records_required_experiment_settings():
    report = run_validation_experiment(
        student_frame(), "synthetic.csv", "Classification", "early", seeds=(42, 43), folds=3,
    )
    assert report["dataset"] == "synthetic.csv"
    assert report["target"] == "G3"
    assert report["seeds"] == [42, 43]
    assert report["cv_folds"] == 3
    assert report["test_size"] == 0.20
    assert report["excluded_features"] == ["G1", "G2"]
    assert report["cv_configuration"] == "StratifiedKFold shuffled"


def test_invalid_scenario_and_missing_seeds_raise():
    frame = student_frame()
    try:
        run_validation_experiment(frame, "synthetic.csv", "Classification", "wrong", seeds=(42,), folds=2)
    except EvaluationError:
        pass
    else:
        raise AssertionError("Invalid scenario should fail")
    try:
        run_validation_experiment(frame, "synthetic.csv", "Classification", "early", seeds=(), folds=2)
    except EvaluationError:
        pass
    else:
        raise AssertionError("No seeds should fail")
