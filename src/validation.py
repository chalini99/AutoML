"""Rigorous, non-mutating validation experiments for the bundled student datasets."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression

from src.data_utils import DatasetValidationError, prepare_target, select_features
from src.evaluation import EvaluationError, build_baseline_comparison, evaluate_predictions, reproducible_split


FEATURE_COLUMNS = (
    "school", "sex", "age", "address", "famsize", "Pstatus", "Medu", "Fedu", "Mjob", "Fjob",
    "reason", "guardian", "traveltime", "studytime", "failures", "schoolsup", "famsup", "paid",
    "activities", "nursery", "higher", "internet", "romantic", "famrel", "freetime", "goout",
    "Dalc", "Walc", "health", "absences", "G1", "G2",
)


@dataclass(frozen=True)
class DatasetProfile:
    dataset: str
    path: str
    sha256: str
    delimiter: str
    shape: tuple[int, int]
    columns: list[str]
    missing_values: int
    duplicate_rows: int
    target_available: bool
    g3_distribution: dict[str, int]
    pass_fail_distribution: dict[str, int]
    warnings: list[str]

    def as_dict(self) -> dict:
        return asdict(self)


def profile_dataset(path: Path | str) -> DatasetProfile:
    """Read a bundled dataset without modifying it and return audit facts."""
    path = Path(path)
    if not path.is_file():
        raise DatasetValidationError(f"Required dataset not found: {path}")
    raw = path.read_bytes()
    delimiter = ";" if raw.count(b";") > raw.count(b",") else ","
    try:
        df = pd.read_csv(path, sep=delimiter)
    except Exception as exc:
        raise DatasetValidationError(f"Could not read {path}: {exc}") from exc
    if df.empty:
        raise DatasetValidationError(f"Dataset is empty: {path}")
    if "G3" not in df.columns:
        raise DatasetValidationError(f"G3 target is missing from {path}")

    g3_counts = df["G3"].value_counts(dropna=False).sort_index()
    pass_fail = (df["G3"] >= 10).map({True: "Pass", False: "Fail"}).value_counts()
    warnings = []
    if df.isna().sum().sum():
        warnings.append("Missing values are present.")
    if df.duplicated().any():
        warnings.append("Duplicate rows are present.")
    if pass_fail.min() < 20:
        warnings.append("A pass/fail class is small; hold-out estimates may be unstable.")

    return DatasetProfile(
        dataset=path.name,
        path=str(path),
        sha256=hashlib.sha256(raw).hexdigest(),
        delimiter=delimiter,
        shape=df.shape,
        columns=df.columns.tolist(),
        missing_values=int(df.isna().sum().sum()),
        duplicate_rows=int(df.duplicated().sum()),
        target_available=True,
        g3_distribution={str(key): int(value) for key, value in g3_counts.items()},
        pass_fail_distribution={str(key): int(value) for key, value in pass_fail.items()},
        warnings=warnings,
    )


def assess_overlap(left: pd.DataFrame, right: pd.DataFrame) -> dict:
    """Assess overlap conservatively using the public student attributes in UCI metadata."""
    shared = [column for column in FEATURE_COLUMNS if column in left.columns and column in right.columns]
    overlaps = pd.merge(left, right, on=shared, how="inner", suffixes=("_math", "_por"))
    exact_keys = list(left.columns.intersection(right.columns))
    exact = pd.merge(left, right, on=exact_keys, how="inner")
    return {
        "method": "inner joins on shared pre-outcome student attributes",
        "shared_attribute_columns": shared,
        "attribute_overlap_rows": int(len(overlaps)),
        "exact_shared_row_overlap": int(len(exact)),
        "note": (
            "Overlap indicates possible repeated students, not identity; datasets should be "
            "analysed and held out separately rather than combining them as independent rows."
        ),
    }


def _metric_summary(values: Iterable[float]) -> tuple[float, float]:
    array = np.asarray(list(values), dtype=float)
    return float(np.mean(array)), float(np.std(array, ddof=1)) if len(array) > 1 else 0.0


def _models(task: str, seed: int, include_learning_model: bool = True) -> dict:
    if task == "Classification":
        models = {"Majority baseline": DummyClassifier(strategy="most_frequent")}
        if include_learning_model:
            models.update({
                "Logistic regression": LogisticRegression(max_iter=2000, random_state=seed),
                "Extra trees": ExtraTreesClassifier(n_estimators=120, max_depth=12, random_state=seed, n_jobs=1),
            })
        return models

    models = {"Mean baseline": DummyRegressor(strategy="mean")}
    if include_learning_model:
        models.update({
            "Linear regression": LinearRegression(),
            "Extra trees": ExtraTreesRegressor(n_estimators=120, max_depth=12, random_state=seed, n_jobs=1),
        })
    return models


def run_validation_experiment(
    frame: pd.DataFrame,
    dataset: str,
    task: str,
    scenario: str,
    seeds: tuple[int, ...] = (42, 43, 44),
    test_size: float = 0.20,
    folds: int = 5,
) -> dict:
    """Run repeated, leakage-safe baseline experiments on one scenario."""
    if task not in {"Classification", "Regression"}:
        raise EvaluationError("Task must be Classification or Regression.")
    if scenario not in {"early", "grade_assisted"}:
        raise EvaluationError("Scenario must be early or grade_assisted.")
    if not seeds:
        raise EvaluationError("At least one random seed is required.")

    derive_pass_fail = task == "Classification"
    X, y = prepare_target(frame, "G3", task, derive_pass_fail=derive_pass_fail)
    X = select_features(X, "G3", "Early prediction" if scenario == "early" else "Grade-assisted prediction")
    if "G3" in X.columns:
        raise EvaluationError("Target G3 leaked into input features.")

    excluded = [column for column in ("G1", "G2") if column not in X.columns and column in frame.columns]
    seed_runs = []
    model_cv: dict[str, list[dict]] = {}
    model_test: dict[str, list[dict]] = {}

    for seed in seeds:
        split = reproducible_split(X, y, task, test_size=test_size, seed=seed)
        comparison, test_metrics, errors, winner = build_baseline_comparison(
            split.X_train,
            split.y_train,
            split.X_test,
            split.y_test,
            task,
            _models(task, seed),
            folds=folds,
            seed=seed,
        )
        seed_runs.append({
            "seed": seed,
            "train_rows": len(split.X_train),
            "test_rows": len(split.X_test),
            "selected_model": winner,
            "cv_rows": comparison.to_dict(orient="records"),
            "held_out_metrics": test_metrics,
            "errors": errors,
        })
        for row in comparison.to_dict(orient="records"):
            model_cv.setdefault(row["model"], []).append(row)

    selection_metric = "cv_f1_macro_mean" if task == "Classification" else "cv_mae_mean"
    summary_rows = []
    selected_models = [run["selected_model"] for run in seed_runs]
    for model in model_cv:
        rows = model_cv[model]
        cv_values = [row[selection_metric] for row in rows]
        cv_mean, cv_std = _metric_summary(cv_values)
        summary_rows.append({
            "model": model,
            "cv_mean": cv_mean,
            "cv_std_across_seeds": cv_std,
            "selection_metric": selection_metric,
            "selected_seed_count": selected_models.count(model),
        })

    for run in seed_runs:
        model = run["selected_model"]
        metrics = run["held_out_metrics"]
        model_test.setdefault(model, []).append(metrics)

    test_summary = []
    for model, metrics_rows in model_test.items():
        if task == "Classification":
            keys = ("accuracy", "balanced_accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_weighted")
            summary = {"model": model}
            for key in keys:
                mean, std = _metric_summary(row[key] for row in metrics_rows)
                summary[f"held_out_{key}_mean"] = mean
                summary[f"held_out_{key}_std"] = std
            prevalences = []
            for row in metrics_rows:
                labels = row["labels"]
                report = row["classification_report"]
                prevalences.append({str(label): report[str(label)]["support"] for label in labels})
            summary["class_prevalence_supports"] = prevalences
        else:
            summary = {"model": model}
            for key in ("mae", "rmse", "r2"):
                mean, std = _metric_summary(row[key] for row in metrics_rows)
                summary[f"held_out_{key}_mean"] = mean
                summary[f"held_out_{key}_std"] = std
        test_summary.append(summary)

    return {
        "dataset": dataset,
        "task": task,
        "target": "G3",
        "target_rule": "G3 >= 10 -> Pass; otherwise Fail" if derive_pass_fail else None,
        "scenario": scenario,
        "excluded_features": excluded,
        "seeds": list(seeds),
        "test_size": test_size,
        "cv_folds": folds,
        "cv_configuration": "StratifiedKFold shuffled" if task == "Classification" else "KFold shuffled",
        "cv_summary": summary_rows,
        "held_out_summary": test_summary,
        "seed_runs": seed_runs,
        "warnings": [
            "Results are single-dataset estimates and do not establish generalization.",
            "The mathematics and Portuguese datasets may contain overlapping students; datasets are never combined.",
            "CV variability is across seeds; test-set values are also averaged across fixed held-out splits.",
        ],
    }


def run_bundled_validation(root: Path | str, seeds: tuple[int, ...] = (42, 43, 44)) -> dict:
    """Run and write a fresh, uniquely named Phase 6 report without touching datasets."""
    root = Path(root)
    raw_dir = root / "data" / "raw"
    paths = [raw_dir / "student-mat.csv", raw_dir / "student-por.csv"]
    profiles = [profile_dataset(path) for path in paths]
    frames = [pd.read_csv(path, sep=";") for path in paths]
    overlap = assess_overlap(frames[0], frames[1])

    experiments = []
    for frame, profile in zip(frames, profiles):
        for task in ("Classification", "Regression"):
            for scenario in ("early", "grade_assisted"):
                experiments.append(run_validation_experiment(frame, profile.dataset, task, scenario, seeds=seeds))

    report = {
        "report_type": "phase6_rigorous_validation",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_profiles": [profile.as_dict() for profile in profiles],
        "dataset_overlap": overlap,
        "experiments": experiments,
        "limitations": [
            "This report validates only the bundled datasets; it does not establish performance on other schools or populations.",
            "The scenarios are not comparable in information availability when G1/G2 are included.",
            "Metrics are based on modest repeated fixed splits, not subject-level clustering across possible overlaps.",
            "No evidence is provided that predictive accuracy is improved over prior work or research baselines.",
        ],
    }

    output_dir = root / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    output = output_dir / f"phase6_validation_{stamp}.json"
    output.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report
