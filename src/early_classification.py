"""Focused, leakage-safe early-prediction classification experiments."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re
import time

import numpy as np
import optuna
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectPercentile, VarianceThreshold
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from src.evaluation import EvaluationError, build_preprocessor, evaluate_predictions, make_cv, reproducible_split

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except Exception:
    LGBMClassifier = None

EARLY_EXCLUDED = ("G1", "G2", "G3")
TARGET_DERIVED_FEATURE_NAMES = {
    "g3passfail",
    "g3pass",
    "g3result",
    "passfail",
    "passfailbinary",
    "passfailtarget",
    "targetderived",
}


def _is_target_derived_feature(column: object) -> bool:
    normalized = re.sub(r"[^a-z0-9]", "", str(column).casefold())
    return (
        normalized in {"g1", "g2"}
        or normalized in TARGET_DERIVED_FEATURE_NAMES
        or normalized.startswith(("g3", "grade3"))
    )


def _excluded_early_feature_columns(frame: pd.DataFrame) -> list[str]:
    return [
        column for column in frame.columns
        if column in EARLY_EXCLUDED or _is_target_derived_feature(column)
    ]


def early_classification_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Create G3 pass/fail and strictly early-only features without derived leakage."""
    missing = [column for column in EARLY_EXCLUDED if column not in frame.columns]
    if missing:
        raise EvaluationError(f"Required early-classification columns missing: {missing}")
    X = frame.drop(columns=_excluded_early_feature_columns(frame)).copy()
    if X.empty or X.shape[1] == 0:
        raise EvaluationError("No early-prediction input features remain.")
    y = frame["G3"].ge(10).astype("int8")
    y.name = "G3_pass_fail"
    if y.isna().any() or y.nunique() != 2:
        raise EvaluationError("The G3 pass/fail target must contain both classes and no missing values.")
    return X, y


def _model_candidates(seed: int) -> dict[str, object]:
    candidates: dict[str, object] = {
        "Majority baseline": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=seed),
        "Class-weighted Logistic Regression": LogisticRegression(
            class_weight="balanced", max_iter=2000, random_state=seed,
        ),
        "Random Forest": RandomForestClassifier(n_estimators=180, random_state=seed, n_jobs=1),
        "Class-weighted Random Forest": RandomForestClassifier(
            n_estimators=180, class_weight="balanced", random_state=seed, n_jobs=1,
        ),
        "Extra Trees": ExtraTreesClassifier(n_estimators=180, random_state=seed, n_jobs=1),
        "SVM": SVC(random_state=seed),
        "HistGradientBoosting": HistGradientBoostingClassifier(random_state=seed),
    }
    if XGBClassifier is not None:
        candidates["XGBoost"] = XGBClassifier(random_state=seed, n_jobs=1, verbosity=0, eval_metric="logloss")
    if LGBMClassifier is not None:
        candidates["LightGBM"] = LGBMClassifier(random_state=seed, n_jobs=1, verbosity=-1)
    return candidates


def _tuned_model(trial: optuna.Trial, name: str, seed: int):
    if name in {"Logistic Regression", "Class-weighted Logistic Regression"}:
        class_weight = (
            "balanced"
            if name == "Class-weighted Logistic Regression"
            else trial.suggest_categorical("class_weight", [None, "balanced"])
        )
        return LogisticRegression(
            C=trial.suggest_float("C", 1e-3, 100, log=True),
            class_weight=class_weight,
            max_iter=3000,
            random_state=seed,
        )
    if name in {"Random Forest", "Class-weighted Random Forest"}:
        class_weight = (
            "balanced"
            if name == "Class-weighted Random Forest"
            else trial.suggest_categorical("class_weight", [None, "balanced"])
        )
        return RandomForestClassifier(
            n_estimators=trial.suggest_int("n_estimators", 80, 160),
            max_depth=trial.suggest_int("max_depth", 3, 18),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 8),
            max_features=trial.suggest_categorical("max_features", ["sqrt", 0.7]),
            class_weight=class_weight,
            random_state=seed,
            n_jobs=1,
        )
    if name == "Extra Trees":
        return ExtraTreesClassifier(
            n_estimators=trial.suggest_int("n_estimators", 80, 160),
            max_depth=trial.suggest_int("max_depth", 3, 18),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 8),
            max_features=trial.suggest_categorical("max_features", ["sqrt", 0.7]),
            class_weight=trial.suggest_categorical("class_weight", [None, "balanced"]),
            random_state=seed,
            n_jobs=1,
        )
    if name == "SVM":
        return SVC(
            C=trial.suggest_float("C", 1e-2, 100, log=True),
            gamma=trial.suggest_categorical("gamma", ["scale", "auto"]),
            kernel=trial.suggest_categorical("kernel", ["rbf", "linear"]),
            class_weight=trial.suggest_categorical("class_weight", [None, "balanced"]),
            random_state=seed,
        )
    if name == "HistGradientBoosting":
        return HistGradientBoostingClassifier(
            learning_rate=trial.suggest_float("learning_rate", 0.02, 0.25, log=True),
            max_iter=trial.suggest_int("max_iter", 80, 240),
            max_leaf_nodes=trial.suggest_int("max_leaf_nodes", 15, 63),
            l2_regularization=trial.suggest_float("l2_regularization", 1e-8, 10, log=True),
            class_weight=trial.suggest_categorical("class_weight", [None, "balanced"]),
            random_state=seed,
        )
    if name == "XGBoost" and XGBClassifier is not None:
        return XGBClassifier(
            n_estimators=trial.suggest_int("n_estimators", 80, 240),
            max_depth=trial.suggest_int("max_depth", 2, 8),
            learning_rate=trial.suggest_float("learning_rate", 0.02, 0.3, log=True),
            subsample=trial.suggest_float("subsample", 0.7, 1),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.7, 1),
            min_child_weight=trial.suggest_float("min_child_weight", 0.5, 8),
            # Label 1 is the majority Pass class; values below one reduce its
            # relative weight and therefore emphasize Fail (label 0).
            scale_pos_weight=trial.suggest_categorical(
                "scale_pos_weight", [0.15, 0.3, 0.5, 1.0],
            ),
            random_state=seed,
            n_jobs=1,
            verbosity=0,
            eval_metric="logloss",
        )
    if name == "LightGBM" and LGBMClassifier is not None:
        return LGBMClassifier(
            n_estimators=trial.suggest_int("n_estimators", 80, 240),
            max_depth=trial.suggest_int("max_depth", 3, 12),
            learning_rate=trial.suggest_float("learning_rate", 0.02, 0.3, log=True),
            num_leaves=trial.suggest_int("num_leaves", 15, 63),
            subsample=trial.suggest_float("subsample", 0.7, 1),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.7, 1),
            class_weight=trial.suggest_categorical("class_weight", [None, "balanced"]),
            random_state=seed,
            n_jobs=1,
            verbosity=-1,
        )
    raise EvaluationError(f"Unsupported tuned model: {name}")


def _pipeline(trial: optuna.Trial, X: pd.DataFrame, model_name: str, seed: int) -> Pipeline:
    feature_method = trial.suggest_categorical("feature_method", ["none", "variance", "mutual_info"])
    steps = [("preprocessor", build_preprocessor(X))]
    if feature_method == "variance":
        steps.append(("feature_selection", VarianceThreshold(threshold=trial.suggest_float("variance_threshold", 0.0, 0.03))))
    elif feature_method == "mutual_info":
        steps.append((
            "feature_selection",
            SelectPercentile(
                score_func=lambda values, target: mutual_info_classif(values, target, random_state=seed),
                percentile=trial.suggest_int("feature_percentile", 50, 100),
            ),
        ))
    steps.append(("model", _tuned_model(trial, model_name, seed)))
    return Pipeline(steps)


def _cv_result(estimator, X_train: pd.DataFrame, y_train: pd.Series, cv) -> dict:
    scores = cross_validate(
        estimator,
        X_train,
        y_train,
        cv=cv,
        scoring={"accuracy": "accuracy", "macro_f1": "f1_macro"},
        n_jobs=1,
        error_score="raise",
    )
    return {
        "cv_accuracy_mean": float(scores["test_accuracy"].mean()),
        "cv_accuracy_std": float(scores["test_accuracy"].std()),
        "cv_macro_f1_mean": float(scores["test_macro_f1"].mean()),
        "cv_macro_f1_std": float(scores["test_macro_f1"].std()),
    }


def tune_one_model(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    seed: int,
    folds: int,
    trials: int,
) -> dict:
    """Tune one model using training-only CV without accessing the held-out test set."""
    cv = make_cv("Classification", y_train, folds, seed)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    start = time.perf_counter()

    def objective(trial: optuna.Trial) -> float:
        pipeline = _pipeline(trial, X_train, model_name, seed)
        scores = _cv_result(pipeline, X_train, y_train, cv)
        trial.set_user_attr("cv_accuracy_mean", scores["cv_accuracy_mean"])
        trial.set_user_attr("cv_accuracy_std", scores["cv_accuracy_std"])
        trial.set_user_attr("cv_macro_f1_std", scores["cv_macro_f1_std"])
        return scores["cv_macro_f1_mean"]

    study.optimize(
        objective,
        n_trials=trials,
        n_jobs=1,
        catch=(Exception,),
        show_progress_bar=False,
    )
    complete = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
    if not complete:
        errors = [str(trial.value) for trial in study.trials if trial.state == optuna.trial.TrialState.FAIL]
        raise EvaluationError(f"All {model_name} trials failed: {errors}")

    best = study.best_trial
    return {
        "model": model_name,
        "selection_metric": "training_cv_f1_macro",
        "cv_macro_f1_mean": float(best.value),
        "cv_macro_f1_std": float(best.user_attrs["cv_macro_f1_std"]),
        "cv_accuracy_mean": float(best.user_attrs["cv_accuracy_mean"]),
        "cv_accuracy_std": float(best.user_attrs["cv_accuracy_std"]),
        "best_params": best.params,
        "optuna_completed_trials": len(complete),
        "optuna_failed_trials": len(study.trials) - len(complete),
        "runtime_seconds": time.perf_counter() - start,
    }


def _evaluate_final_candidate(
    row: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    seed: int,
) -> None:
    """Evaluate one already-selected CV candidate in the final held-out stage."""
    if row["model"] == "Majority baseline":
        fitted_model = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
    else:
        fitted_model = _pipeline(
            optuna.trial.FixedTrial(row["best_params"]),
            X_train,
            row["model"],
            seed,
        ).fit(X_train, y_train)
    predictions = fitted_model.predict(X_test)
    metrics = evaluate_predictions(y_test, predictions, "Classification", fitted_model, X_test)
    labels = metrics["labels"]
    classification_report = metrics["classification_report"]
    row.update({
        "held_out_accuracy": metrics["accuracy"],
        "held_out_balanced_accuracy": float(balanced_accuracy_score(y_test, predictions)),
        "held_out_macro_f1": metrics["f1_macro"],
        "held_out_precision_macro": metrics["precision_macro"],
        "held_out_recall_macro": metrics["recall_macro"],
        "held_out_recall_by_class": {
            str(label): float(classification_report[str(label)]["recall"])
            for label in labels
        },
        "held_out_precision_by_class": {
            str(label): float(classification_report[str(label)]["precision"])
            for label in labels
        },
        "held_out_f1_by_class": {
            str(label): float(classification_report[str(label)]["f1-score"])
            for label in labels
        },
        "held_out_predicted_class_distribution": {
            str(label): int((predictions == label).sum())
            for label in labels
        },
        "held_out_confusion_matrix": {
            "labels": [str(label) for label in labels],
            "matrix": confusion_matrix(y_test, predictions, labels=labels).tolist(),
        },
    })


def run_early_classification(
    frame: pd.DataFrame,
    dataset: str,
    seeds: tuple[int, ...] = (42, 43),
    test_size: float = 0.20,
    folds: int = 5,
    optuna_trials: int = 8,
) -> dict:
    """Tune candidates on training CV and compare them on identical held-out splits."""
    X, y = early_classification_features(frame)
    seed_runs = []

    for seed in seeds:
        split = reproducible_split(X, y, "Classification", test_size=test_size, seed=seed)
        models = {
            name: clone(model)
            for name, model in _model_candidates(seed).items()
            if name != "Majority baseline"
        }
        baseline_model = DummyClassifier(strategy="most_frequent")
        cv = make_cv("Classification", split.y_train, folds, seed)
        baseline_cv = _cv_result(baseline_model, split.X_train, split.y_train, cv)
        baseline_row = {
            "model": "Majority baseline",
            **baseline_cv,
            "selection_metric": "training_cv_f1_macro",
        }
        rows = [baseline_row]
        errors = {}
        for name in models:
            try:
                rows.append(tune_one_model(name, split.X_train, split.y_train, seed, folds, optuna_trials))
            except Exception as exc:
                errors[name] = f"{type(exc).__name__}: {exc}"
        selected_model = max(
            rows,
            key=lambda row: row["cv_macro_f1_mean"],
        )
        selected_model_name = selected_model["model"]
        for row in rows:
            _evaluate_final_candidate(
                row,
                split.X_train,
                split.y_train,
                split.X_test,
                split.y_test,
                seed,
            )
        seed_runs.append({
            "seed": seed,
            "train_rows": len(split.X_train),
            "test_rows": len(split.X_test),
            "test_indices": split.X_test.index.tolist(),
            "selected_model": selected_model_name,
            "selected_model_cv_macro_f1": selected_model["cv_macro_f1_mean"],
            "models": rows,
            "model_errors": errors,
        })

    summary = []
    names = sorted({row["model"] for run in seed_runs for row in run["models"]})
    for name in names:
        rows = [row for run in seed_runs for row in run["models"] if row["model"] == name]
        summary_row = {"model": name}
        metric_columns = {
            "cv_accuracy": "cv_accuracy_mean",
            "cv_macro_f1": "cv_macro_f1_mean",
            "held_out_accuracy": "held_out_accuracy",
            "held_out_balanced_accuracy": "held_out_balanced_accuracy",
            "held_out_macro_f1": "held_out_macro_f1",
        }
        for metric, column in metric_columns.items():
            values = [row[column] for row in rows]
            summary_row[f"{metric}_mean"] = float(pd.Series(values).mean())
            summary_row[f"{metric}_std"] = float(pd.Series(values).std(ddof=1)) if len(values) > 1 else 0.0
        for metric in ("precision", "recall", "f1"):
            for label, class_name in (("0", "fail"), ("1", "pass")):
                values = [
                    row[f"held_out_{metric}_by_class"][label]
                    for row in rows
                ]
                summary_row[f"held_out_{class_name}_{metric}_mean"] = float(pd.Series(values).mean())
                summary_row[f"held_out_{class_name}_{metric}_std"] = (
                    float(pd.Series(values).std(ddof=1)) if len(values) > 1 else 0.0
                )
        matrices = [row["held_out_confusion_matrix"]["matrix"] for row in rows]
        summary_row["held_out_confusion_matrix_total"] = [
            [sum(matrix[i][j] for matrix in matrices) for j in range(2)]
            for i in range(2)
        ]
        summary_row["held_out_predicted_class_distribution_total"] = {
            str(label): sum(row["held_out_predicted_class_distribution"][str(label)] for row in rows)
            for label in (0, 1)
        }
        summary.append(summary_row)

    selected_rows = [
        next(row for row in run["models"] if row["model"] == run["selected_model"])
        for run in seed_runs
    ]
    best_model_summary = max(summary, key=lambda row: row["cv_macro_f1_mean"])
    return {
        "experiment_protocol": "cv-selection-before-held-out-v1",
        "dataset": dataset,
        "task": "Classification",
        "target": "G3_pass_fail",
        "target_rule": "G3 >= 10 -> Pass (1); otherwise Fail (0)",
        "scenario": "early_prediction",
        "excluded_features": _excluded_early_feature_columns(frame),
        "feature_columns": X.columns.tolist(),
        "class_distribution": {
            "Fail": int((y == 0).sum()),
            "Pass": int((y == 1).sum()),
        },
        "class_prevalence": {
            "Fail": float((y == 0).mean()),
            "Pass": float((y == 1).mean()),
        },
        "seeds": list(seeds),
        "test_size": test_size,
        "cv_folds": folds,
        "cv_configuration": "StratifiedKFold(shuffle=True)",
        "selection_metric": "training_cv_f1_macro",
        "test_set_policy": (
            "No held-out features or labels are accessed until every candidate has completed "
            "training-CV tuning and the CV-macro-F1 winner is frozen for that seed."
        ),
        "optuna_trials_per_model_per_seed": optuna_trials,
        "preprocessing": "Fitted inside each CV pipeline",
        "feature_selection": ["none", "variance", "mutual_information"],
        "class_imbalance_methods": [
            "class_weight=balanced for Logistic Regression, Random Forest, Extra Trees, SVM, HistGradientBoosting, and LightGBM",
            "Optuna-tuned scale_pos_weight values below one for XGBoost, since Pass is the positive majority class",
            "No resampling; all balancing is estimator-native and fitted within each CV fold",
        ],
        "installed_gradients": [name for name in ("HistGradientBoosting", "XGBoost", "LightGBM") if name in _model_candidates(seeds[0])],
        "class_mapping": {"0": "Fail", "1": "Pass"},
        "selected_model_by_seed": [
            {
                "seed": run["seed"],
                "model": run["selected_model"],
                "training_cv_macro_f1": run["selected_model_cv_macro_f1"],
                "held_out_accuracy": selected_rows[index]["held_out_accuracy"],
                "held_out_macro_f1": selected_rows[index]["held_out_macro_f1"],
            }
            for index, run in enumerate(seed_runs)
        ],
        "selected_model_held_out_summary": {
            metric: {
                "mean": float(pd.Series([row[metric] for row in selected_rows]).mean()),
                "std": (
                    float(pd.Series([row[metric] for row in selected_rows]).std(ddof=1))
                    if len(selected_rows) > 1 else 0.0
                ),
            }
            for metric in ("held_out_accuracy", "held_out_macro_f1", "held_out_balanced_accuracy")
        },
        "best_model_by_mean_training_cv_macro_f1": {
            "model": best_model_summary["model"],
            "training_cv_macro_f1_mean": best_model_summary["cv_macro_f1_mean"],
            "training_cv_macro_f1_std_across_seeds": best_model_summary["cv_macro_f1_std"],
            "held_out_accuracy_mean": best_model_summary["held_out_accuracy_mean"],
            "held_out_accuracy_std": best_model_summary["held_out_accuracy_std"],
            "held_out_macro_f1_mean": best_model_summary["held_out_macro_f1_mean"],
            "held_out_macro_f1_std": best_model_summary["held_out_macro_f1_std"],
            "held_out_balanced_accuracy_mean": best_model_summary["held_out_balanced_accuracy_mean"],
            "held_out_balanced_accuracy_std": best_model_summary["held_out_balanced_accuracy_std"],
        },
        "selected_model_fail_error_analysis": [
            {
                "seed": run["seed"],
                "model": run["selected_model"],
                "actual_fail_count": int(run["models"][0]["held_out_confusion_matrix"]["matrix"][0][0]
                                         + run["models"][0]["held_out_confusion_matrix"]["matrix"][0][1]),
                "fail_misclassified_as_pass": int(
                    next(row for row in run["models"] if row["model"] == run["selected_model"])
                    ["held_out_confusion_matrix"]["matrix"][0][1]
                ),
                "fail_recall": next(
                    row for row in run["models"] if row["model"] == run["selected_model"]
                )["held_out_recall_by_class"]["0"],
                "fail_predicted_as_pass_rate": (
                    1.0 - next(
                        row for row in run["models"] if row["model"] == run["selected_model"]
                    )["held_out_recall_by_class"]["0"]
                ),
            }
            for run in seed_runs
        ],
        "seed_runs": seed_runs,
        "summary": summary,
        "warnings": [
            "Each seed's model is selected only from training-CV macro F1; held-out metrics never select or tune a model.",
            "Each model's reported held-out score is a diagnostic evaluation after its CV tuning; use selected_model_by_seed for the CV-selected model.",
            "Small-sample and class-prevalence effects remain possible.",
            "No claim of generalization beyond these bundled datasets is made.",
        ],
    }


def save_early_classification_report(report: dict, output_dir: Path | str) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    path = output_dir / f"early_classification_{stamp}.json"
    while path.exists() or path.with_suffix(".csv").exists():
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
        path = output_dir / f"early_classification_{stamp}.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    csv_rows = []
    for seed_run in report["seed_runs"]:
        for model in seed_run["models"]:
            csv_rows.append({
                "dataset": report["dataset"],
                "seed": seed_run["seed"],
                "model": model["model"],
                "selected_by_training_cv": model["model"] == seed_run["selected_model"],
                "cv_macro_f1_mean": model["cv_macro_f1_mean"],
                "cv_macro_f1_std": model["cv_macro_f1_std"],
                "held_out_accuracy": model["held_out_accuracy"],
                "held_out_balanced_accuracy": model["held_out_balanced_accuracy"],
                "held_out_macro_f1": model["held_out_macro_f1"],
                "fail_precision": model["held_out_precision_by_class"]["0"],
                "fail_recall": model["held_out_recall_by_class"]["0"],
                "fail_f1": model["held_out_f1_by_class"]["0"],
                "pass_precision": model["held_out_precision_by_class"]["1"],
                "pass_recall": model["held_out_recall_by_class"]["1"],
                "pass_f1": model["held_out_f1_by_class"]["1"],
                "confusion_matrix": json.dumps(model["held_out_confusion_matrix"]["matrix"]),
                "predicted_class_distribution": json.dumps(model["held_out_predicted_class_distribution"]),
                "best_params": json.dumps(model.get("best_params", {})),
                "optuna_completed_trials": model.get("optuna_completed_trials", 0),
            })
    pd.DataFrame(csv_rows).to_csv(path.with_suffix(".csv"), index=False)
    return path
