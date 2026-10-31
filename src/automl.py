"""Reproducible Optuna search for classification and regression pipelines."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import time
from typing import Callable

import joblib
import optuna
import pandas as pd
from sklearn.base import clone
from sklearn.feature_selection import SelectFromModel, SelectPercentile, VarianceThreshold
from sklearn.feature_selection import f_regression, mutual_info_classif, mutual_info_regression
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Lasso, LogisticRegression, Ridge
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from src.evaluation import EvaluationError, build_preprocessor, evaluate_predictions, make_cv, reproducible_split

ROOT = Path(__file__).resolve().parent.parent


def _optional_model(model: str):
    try:
        if model == "xgboost_classifier":
            from xgboost import XGBClassifier
            return XGBClassifier
        if model == "xgboost_regressor":
            from xgboost import XGBRegressor
            return XGBRegressor
        if model == "lightgbm_classifier":
            from lightgbm import LGBMClassifier
            return LGBMClassifier
        if model == "lightgbm_regressor":
            from lightgbm import LGBMRegressor
            return LGBMRegressor
    except Exception:
        return None
    return None


@dataclass
class AutoMLResult:
    study: optuna.study.Study
    best_pipeline: Pipeline
    trials: pd.DataFrame
    comparison: pd.DataFrame
    final_metrics: dict
    metadata: dict


def supported_scoring(task: str) -> dict[str, str]:
    """Return scikit-learn scoring names and their reported directions."""
    if task == "Classification":
        return {
            "f1_macro": "f1_macro",
            "balanced_accuracy": "balanced_accuracy",
            "accuracy": "accuracy",
            "f1_weighted": "f1_weighted",
        }
    return {
        "neg_mae": "neg_mean_absolute_error",
        "neg_rmse": "neg_root_mean_squared_error",
        "r2": "r2",
    }


def _classification_model(trial, model_name: str, seed: int):
    if model_name == "logistic":
        return LogisticRegression(
            C=trial.suggest_float("C", 0.01, 100.0, log=True),
            max_iter=2000,
            random_state=seed,
        )
    if model_name == "decision_tree":
        return DecisionTreeClassifier(
            max_depth=trial.suggest_int("max_depth", 2, 20),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10),
            random_state=seed,
        )
    if model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=trial.suggest_int("n_estimators", 80, 220),
            max_depth=trial.suggest_int("max_depth", 3, 18),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 5),
            random_state=seed,
            n_jobs=1,
        )
    if model_name == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=trial.suggest_int("n_estimators", 80, 220),
            max_depth=trial.suggest_int("max_depth", 3, 18),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 5),
            random_state=seed,
            n_jobs=1,
        )
    if model_name == "knn":
        return KNeighborsClassifier(
            n_neighbors=trial.suggest_int("n_neighbors", 3, 15),
            weights=trial.suggest_categorical("weights", ["uniform", "distance"]),
            p=trial.suggest_int("p", 1, 2),
        )
    if model_name == "svc":
        return SVC(
            C=trial.suggest_float("C", 0.01, 100.0, log=True),
            kernel=trial.suggest_categorical("kernel", ["linear", "rbf"]),
            probability=True,
            random_state=seed,
        )
    if model_name == "hist_gradient_boosting":
        try:
            from sklearn.ensemble import HistGradientBoostingClassifier
            return HistGradientBoostingClassifier(
                learning_rate=trial.suggest_float("learning_rate", 0.03, 0.25, log=True),
                max_iter=trial.suggest_int("max_iter", 80, 220),
                max_leaf_nodes=trial.suggest_int("max_leaf_nodes", 15, 63),
                l2_regularization=trial.suggest_float("l2_regularization", 1e-8, 10.0, log=True),
                random_state=seed,
            )
        except Exception as exc:
            raise RuntimeError(f"HistGradientBoostingClassifier unavailable: {exc}") from exc

    if model_name == "xgboost":
        cls = _optional_model("xgboost_classifier")
        if cls is None:
            raise RuntimeError("xgboost is not installed")
        return cls(
            n_estimators=trial.suggest_int("n_estimators", 80, 220),
            max_depth=trial.suggest_int("max_depth", 2, 8),
            learning_rate=trial.suggest_float("learning_rate", 0.02, 0.3, log=True),
            subsample=trial.suggest_float("subsample", 0.7, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.7, 1.0),
            eval_metric="logloss",
            random_state=seed,
            n_jobs=1,
        )

    cls = _optional_model("lightgbm_classifier")
    if cls is None:
        raise RuntimeError("lightgbm is not installed")
    return cls(
        n_estimators=trial.suggest_int("n_estimators", 80, 220),
        max_depth=trial.suggest_int("max_depth", 3, 12),
        learning_rate=trial.suggest_float("learning_rate", 0.02, 0.3, log=True),
        num_leaves=trial.suggest_int("num_leaves", 15, 63),
        verbosity=-1,
        random_state=seed,
        n_jobs=1,
    )


def _regression_model(trial, model_name: str, seed: int):
    if model_name == "linear":
        return LinearRegression()
    if model_name == "ridge":
        return Ridge(alpha=trial.suggest_float("alpha", 1e-3, 100.0, log=True))
    if model_name == "lasso":
        return Lasso(
            alpha=trial.suggest_float("alpha", 1e-4, 10.0, log=True),
            max_iter=5000,
            random_state=seed,
        )
    if model_name == "decision_tree":
        return DecisionTreeRegressor(
            max_depth=trial.suggest_int("max_depth", 2, 20),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10),
            random_state=seed,
        )
    if model_name == "random_forest":
        return RandomForestRegressor(
            n_estimators=trial.suggest_int("n_estimators", 80, 220),
            max_depth=trial.suggest_int("max_depth", 3, 18),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 5),
            random_state=seed,
            n_jobs=1,
        )
    if model_name == "extra_trees":
        return ExtraTreesRegressor(
            n_estimators=trial.suggest_int("n_estimators", 80, 220),
            max_depth=trial.suggest_int("max_depth", 3, 18),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 5),
            random_state=seed,
            n_jobs=1,
        )
    if model_name == "svr":
        return SVR(
            C=trial.suggest_float("C", 0.01, 100.0, log=True),
            epsilon=trial.suggest_float("epsilon", 1e-3, 2.0, log=True),
            kernel=trial.suggest_categorical("kernel", ["linear", "rbf"]),
        )
    if model_name == "hist_gradient_boosting":
        try:
            from sklearn.ensemble import HistGradientBoostingRegressor
            return HistGradientBoostingRegressor(
                learning_rate=trial.suggest_float("learning_rate", 0.03, 0.25, log=True),
                max_iter=trial.suggest_int("max_iter", 80, 220),
                max_leaf_nodes=trial.suggest_int("max_leaf_nodes", 15, 63),
                l2_regularization=trial.suggest_float("l2_regularization", 1e-8, 10.0, log=True),
                random_state=seed,
            )
        except Exception as exc:
            raise RuntimeError(f"HistGradientBoostingRegressor unavailable: {exc}") from exc

    if model_name == "xgboost":
        cls = _optional_model("xgboost_regressor")
        if cls is None:
            raise RuntimeError("xgboost is not installed")
        return cls(
            n_estimators=trial.suggest_int("n_estimators", 80, 220),
            max_depth=trial.suggest_int("max_depth", 2, 8),
            learning_rate=trial.suggest_float("learning_rate", 0.02, 0.3, log=True),
            random_state=seed,
            n_jobs=1,
        )

    cls = _optional_model("lightgbm_regressor")
    if cls is None:
        raise RuntimeError("lightgbm is not installed")
    return cls(
        n_estimators=trial.suggest_int("n_estimators", 80, 220),
        max_depth=trial.suggest_int("max_depth", 3, 12),
        learning_rate=trial.suggest_float("learning_rate", 0.02, 0.3, log=True),
        verbosity=-1,
        random_state=seed,
        n_jobs=1,
    )


def build_automl_pipeline(trial: optuna.trial.FrozenTrial | optuna.Trial, X: pd.DataFrame, task: str, seed: int, feature_methods: list[str]):
    model_names = list(_model_options(task))
    model_name = trial.suggest_categorical("model", model_names)
    model = _classification_model(trial, model_name, seed) if task == "Classification" else _regression_model(trial, model_name, seed)
    feature_method = trial.suggest_categorical("feature_method", feature_methods)
    steps = [("preprocessor", build_preprocessor(X))]

    if feature_method == "variance":
        steps.append(("feature_selection", VarianceThreshold(threshold=trial.suggest_float("variance_threshold", 0.0, 0.02))))
    elif feature_method == "mutual_info":
        score = mutual_info_classif if task == "Classification" else mutual_info_regression
        score_func = lambda values, target: score(values, target, random_state=seed)
        steps.append(("feature_selection", SelectPercentile(score_func=score_func, percentile=trial.suggest_int("feature_percentile", 50, 100))))
    elif feature_method == "model_based":
        estimator = LogisticRegression(max_iter=1000, random_state=seed) if task == "Classification" else Ridge(alpha=1.0)
        steps.append(("feature_selection", SelectFromModel(estimator=estimator, threshold="median")))

    steps.append(("model", model))
    return Pipeline(steps)


def _model_options(task: str) -> tuple[str, ...]:
    if task == "Classification":
        options = (
            "logistic", "decision_tree", "random_forest", "extra_trees", "knn", "svc",
            "hist_gradient_boosting",
        )
        if _optional_model("xgboost_classifier") is not None:
            options += ("xgboost",)
        if _optional_model("lightgbm_classifier") is not None:
            options += ("lightgbm",)
    else:
        options = (
            "linear", "ridge", "lasso", "decision_tree", "random_forest", "extra_trees", "svr",
            "hist_gradient_boosting",
        )
        if _optional_model("xgboost_regressor") is not None:
            options += ("xgboost",)
        if _optional_model("lightgbm_regressor") is not None:
            options += ("lightgbm",)
    return options


def run_automl_search(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    task: str,
    scoring: str,
    n_trials: int = 12,
    folds: int = 5,
    seed: int = 42,
    feature_methods: list[str] | None = None,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> AutoMLResult:
    """Tune training-only CV and evaluate once on the provided untouched test set."""
    scoring_options = supported_scoring(task)
    if task not in {"Classification", "Regression"}:
        raise EvaluationError("Task must be Classification or Regression.")
    if scoring not in scoring_options and scoring not in scoring_options.values():
        raise EvaluationError(f"Unsupported {task} scoring metric: {scoring}")
    sklearn_scoring = scoring_options.get(scoring, scoring)
    if not isinstance(n_trials, int) or isinstance(n_trials, bool) or n_trials < 1:
        raise EvaluationError("Optuna trial count must be an integer of at least 1.")

    feature_methods = feature_methods or ["none"]
    allowed = {"none", "variance", "mutual_info", "model_based"}
    feature_methods = [method for method in feature_methods if method in allowed]
    if not feature_methods:
        raise EvaluationError("At least one supported feature-selection strategy is required.")
    cv = make_cv(task, y_train, folds, seed)
    study = optuna.create_study(
        study_name=f"student_automl_{task.lower()}_{seed}",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )

    start = time.perf_counter()
    def objective(trial: optuna.Trial) -> float:
        trial_start = time.perf_counter()
        try:
            pipeline = build_automl_pipeline(trial, X_train, task, seed, feature_methods)
            scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring=sklearn_scoring, n_jobs=1, error_score="raise")
            score = float(scores.mean())
            trial.set_user_attr("runtime_seconds", time.perf_counter() - trial_start)
            trial.set_user_attr("model_family", trial.params["model"])
            trial.set_user_attr("feature_method", trial.params["feature_method"])
            return score
        except Exception as exc:
            trial.set_user_attr("error", f"{type(exc).__name__}: {exc}")
            trial.set_user_attr("runtime_seconds", time.perf_counter() - trial_start)
            raise

    def callback(study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> None:
        if progress_callback:
            progress_callback(trial.number + 1, n_trials, trial.params.get("model", "failed"))

    study.optimize(objective, n_trials=max(1, int(n_trials)), n_jobs=1, catch=(Exception,), callbacks=[callback])
    completed = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        raise EvaluationError("All Optuna trials failed. Check trial errors and model compatibility.")

    best = study.best_trial
    best_pipeline = build_automl_pipeline(optuna.trial.FixedTrial(best.params), X_train, task, seed, feature_methods)
    best_pipeline.fit(X_train, y_train)
    predictions = best_pipeline.predict(X_test)
    final_metrics = evaluate_predictions(y_test, predictions, task, best_pipeline, X_test)

    trials = pd.DataFrame([{
        "number": trial.number,
        "status": trial.state.name,
        "model": trial.params.get("model"),
        "feature_method": trial.params.get("feature_method"),
        "cv_score": trial.value,
        "runtime_seconds": trial.user_attrs.get("runtime_seconds"),
        "error": trial.user_attrs.get("error"),
        "params": json.dumps(trial.params, sort_keys=True),
    } for trial in study.trials])

    reported_score = best.value
    if sklearn_scoring in {"neg_mean_absolute_error", "neg_root_mean_squared_error"}:
        reported_score = -reported_score
    comparison = pd.DataFrame([{
        "model": f"Optuna {best.params['model']}",
        "feature_method": best.params["feature_method"],
        "cv_score": reported_score,
        "scoring": scoring,
        "trials": len(study.trials),
        "completed_trials": len(completed),
        "runtime_seconds": time.perf_counter() - start,
        "params": json.dumps(best.params, sort_keys=True),
        **({key: final_metrics[key] for key in ("accuracy", "balanced_accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_weighted") if key in final_metrics}
           if task == "Classification" else {key: final_metrics[key] for key in ("mae", "rmse", "r2")}),
    }])

    metadata = {
        "task": task,
        "model_family": best.params["model"],
        "feature_method": best.params["feature_method"],
        "hyperparameters": best.params,
        "scoring": sklearn_scoring,
        "cv_folds": folds,
        "cv_seed": seed,
        "cv_score": reported_score,
        "trial_count": len(study.trials),
        "completed_trial_count": len(completed),
        "runtime_seconds": time.perf_counter() - start,
        "input_features": X_train.columns.tolist(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "sklearn": importlib.metadata.version("scikit-learn"),
    }
    return AutoMLResult(study, best_pipeline, trials, comparison, final_metrics, metadata)


def save_automl_result(result: AutoMLResult, output_dir: Path | str, dataset_identifier: str = "uploaded") -> dict[str, str]:
    """Save a pipeline, metadata, and trial report without overwriting prior artifacts."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    task = result.metadata["task"].lower()
    stem = f"automl_{task}_{dataset_identifier}_{stamp}"
    model_path = output_dir / f"{stem}.joblib"
    metadata = {**result.metadata, "dataset_identifier": dataset_identifier, "model_path": str(model_path)}
    joblib.dump(result.best_pipeline, model_path)
    (output_dir / f"{stem}.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    result.trials.to_csv(output_dir / f"{stem}_trials.csv", index=False)
    result.comparison.to_csv(output_dir / f"{stem}_comparison.csv", index=False)
    return {"model": str(model_path), "metadata": str(output_dir / f"{stem}.json")}


def run_automl(n_trials: int = 12, use_early_features: bool = True):
    """Backward-compatible command-line entry point using the mathematics dataset."""
    df = pd.read_csv(ROOT / "data" / "raw" / "student-mat.csv", sep=";")
    target = "G3"
    X = df.drop(columns=[target])
    if use_early_features:
        X = X.drop(columns=["G1", "G2"])
    y = (df[target] >= 10).astype(int)
    split = reproducible_split(X, y, "Classification", 0.2, 42)
    result = run_automl_search(
        split.X_train,
        split.y_train,
        split.X_test,
        split.y_test,
        "Classification",
        "f1_macro",
        n_trials=n_trials,
        folds=5,
        seed=42,
    )
    save_automl_result(result, ROOT / "models", "student-mat")
    print(result.trials.to_string(index=False))
    print("Held-out test metrics:", result.final_metrics)
    return result


if __name__ == "__main__":
    run_automl()
