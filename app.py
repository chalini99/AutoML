"""Student AutoML analytics dashboard."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_curve

from src.automl import run_automl_search, save_automl_result, supported_scoring
from src.dashboard_ui import (
    NAVIGATION,
    apply_dashboard_style,
    render_bar_chart,
    render_brand,
    render_confusion_matrix,
    render_hero,
    render_line_chart,
    render_metric_cards,
    render_page_header,
    render_panel,
    render_scatter_chart,
)
from src.data_utils import (
    DatasetValidationError,
    load_csv,
    prepare_target,
    select_features,
    suggest_task,
)
from src.evaluation import EvaluationError, build_baseline_comparison, reproducible_split
from src.xai import ExplanationError, model_specific_explanation, permutation_explanation, predict_with_explanation


TITLE = (
    "AutoML-Based Student Academic Performance Prediction Using Automated "
    "Feature Selection, Hyperparameter Optimization, and Explainable AI"
)
STATE_KEY = "phase5_experiment"
NAV_KEY = "student_automl_workspace"
CONFIG_KEYS = (
    "experiment_target",
    "experiment_task",
    "experiment_scenario",
    "experiment_derive_pass_fail",
    "experiment_test_size",
    "experiment_cv_folds",
    "experiment_seed",
    "experiment_automl_enabled",
    "experiment_automl_trials",
    "experiment_automl_scoring",
    "experiment_feature_methods",
)


def _render_header_for_page(page: str, dataset_name: str, rows: int) -> None:
    descriptions = {
        "Dataset Overview": (
            "OVERVIEW",
            "Dataset overview",
            "Inspect dataset quality, feature composition, and target distributions before training.",
        ),
        "Experiment Setup": (
            "WORKSPACE",
            "Configure an experiment",
            "Choose a prediction task and evaluation strategy. Training starts only when you run it.",
        ),
        "Model Training & Comparison": (
            "MODEL WORKSPACE",
            "Training and comparison",
            "Compare candidate models using training cross-validation; held-out metrics remain separate.",
        ),
        "Evaluation Results": (
            "FINAL EVALUATION",
            "Held-out results",
            "Review final test metrics, class-level performance, and prediction errors.",
        ),
        "Explainable AI": (
            "MODEL TRANSPARENCY",
            "Explainable AI",
            "Explore global and individual explanations for the fitted Optuna-selected model.",
        ),
        "Saved Experiments": (
            "EXPERIMENT HISTORY",
            "Saved experiments",
            "Browse saved experiment metadata without loading serialized model files.",
        ),
        "Rigorous Validation": (
            "VALIDATION",
            "Rigorous validation",
            "Review multi-seed results and limitations for the bundled Math and Portuguese datasets.",
        ),
    }
    kicker, title, description = descriptions[page]
    render_page_header(kicker, title, description)
    st.caption(f"Active dataset · {dataset_name} · {rows:,} records")


def _format_metric(value: object, digits: int = 3) -> str:
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}"
    return str(value)


def _target_distribution(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    counts = frame[column].value_counts(dropna=False)
    result = counts.rename_axis("Category").reset_index(name="Records")
    if len(result) > 24:
        visible = result.iloc[:23].copy()
        other_count = int(result.iloc[23:]["Records"].sum())
        visible.loc[len(visible)] = ["Other categories", other_count]
        result = visible
    result["Category"] = result["Category"].map(
        lambda value: "Missing" if pd.isna(value) else str(value)
    )
    return result


def _render_overview(frame: pd.DataFrame, dataset_name: str, delimiter: str) -> None:
    _render_header_for_page("Dataset Overview", dataset_name, len(frame))
    render_hero(
        "Academic performance analytics, built for transparent experimentation.",
        "Prepare student data, compare models with leakage-aware validation, and inspect explainability "
        "artifacts from one reproducible workspace.",
        "STUDENT AUTOML · ANALYTICS",
    )

    missing_count = int(frame.isna().sum().sum())
    duplicate_count = int(frame.duplicated().sum())
    render_metric_cards([
        ("Student records", f"{len(frame):,}"),
        ("Input features", f"{frame.shape[1]:,}"),
        ("Missing values", f"{missing_count:,}"),
        ("Duplicate records", f"{duplicate_count:,}"),
    ])

    overview_left, overview_right = st.columns((1.45, 1))
    with overview_left:
        st.subheader("Target distribution")
        target_column = st.selectbox(
            "Choose a column to inspect",
            frame.columns.tolist(),
            key="overview_target_column",
            label_visibility="collapsed",
        )
        distribution = _target_distribution(frame, target_column)
        render_bar_chart(
            distribution,
            "Category",
            "Records",
            f"Records by {target_column}",
            height=330,
        )
    with overview_right:
        st.subheader("Dataset quality")
        status = "Ready for review" if missing_count == 0 and duplicate_count == 0 else "Review recommended"
        render_panel("Quality status", status, f"Detected delimiter: {delimiter!r}")
        if missing_count:
            st.warning(f"{missing_count:,} missing cells are present. Imputation is fitted inside training pipelines.")
        else:
            st.success("No missing cells detected.")
        if duplicate_count:
            st.warning(f"{duplicate_count:,} duplicate rows detected; consider their effect on evaluation.")
        else:
            st.success("No duplicate rows detected.")

        dtype_counts = frame.dtypes.astype(str).value_counts()
        type_summary = pd.DataFrame({
            "Data type": dtype_counts.index.astype(str),
            "Columns": dtype_counts.values,
        })
        render_bar_chart(type_summary, "Data type", "Columns", "Feature type composition", height=245)

    st.subheader("Dataset preview")
    st.dataframe(frame.head(10), width="stretch", hide_index=True)
    with st.expander("Column schema and missing-value detail"):
        schema = pd.DataFrame({
            "Feature": frame.columns,
            "Data type": frame.dtypes.astype(str).values,
            "Missing": frame.isna().sum().values,
            "Missing (%)": (frame.isna().mean().mul(100).round(2)).values,
            "Unique values": frame.nunique(dropna=True).values,
        })
        st.dataframe(schema, width="stretch", hide_index=True)


def _configuration_defaults(frame: pd.DataFrame) -> dict:
    target_default = str(frame.columns[0])
    target = st.session_state.get("experiment_target", target_default)
    if target not in frame.columns:
        target = target_default
    task_default = suggest_task(frame[target].dropna())
    task = st.session_state.get("experiment_task", task_default)
    if task not in {"Classification", "Regression"}:
        task = task_default
    scenario = st.session_state.get("experiment_scenario", "Early prediction")
    if scenario not in {"Early prediction", "Grade-assisted prediction", "General dataset"}:
        scenario = "Early prediction"
    scoring_options = list(supported_scoring(task))
    scoring = st.session_state.get("experiment_automl_scoring", scoring_options[0])
    if scoring not in scoring_options:
        scoring = scoring_options[0]
    feature_methods = st.session_state.get("experiment_feature_methods", ["none"])
    allowed_methods = {"none", "variance", "mutual_info", "model_based"}
    feature_methods = [method for method in feature_methods if method in allowed_methods]
    return {
        "target": target,
        "task": task,
        "scenario": scenario,
        "derive_pass_fail": bool(st.session_state.get("experiment_derive_pass_fail", target == "G3")),
        "test_size": int(st.session_state.get("experiment_test_size", 20)),
        "cv_folds": int(st.session_state.get("experiment_cv_folds", 5)),
        "seed": int(st.session_state.get("experiment_seed", 42)),
        "automl_enabled": bool(st.session_state.get("experiment_automl_enabled", False)),
        "automl_trials": int(st.session_state.get("experiment_automl_trials", 12)),
        "automl_scoring": scoring,
        "feature_methods": feature_methods or ["none"],
    }


def _render_setup(frame: pd.DataFrame, dataset_name: str) -> tuple[dict, bool]:
    _render_header_for_page("Experiment Setup", dataset_name, len(frame))
    defaults = _configuration_defaults(frame)
    with st.form("experiment_configuration"):
        left, right = st.columns((1.15, 1))
        with left:
            st.subheader("Prediction objective")
            target = st.selectbox(
                "Prediction target",
                frame.columns.tolist(),
                index=frame.columns.get_loc(defaults["target"]),
                key="experiment_target",
            )
            target_series = frame[target].dropna()
            task_options = ["Classification", "Regression"]
            suggested_task = suggest_task(target_series)
            task = st.selectbox(
                "Prediction task",
                task_options,
                index=task_options.index(
                    defaults["task"] if defaults["task"] in task_options else suggested_task
                ),
                key="experiment_task",
            )
            derive_pass_fail = False
            if target == "G3":
                derive_pass_fail = st.checkbox(
                    "Derive Pass / Fail from G3 (G3 ≥ 10)",
                    value=defaults["derive_pass_fail"] or target == "G3",
                    key="experiment_derive_pass_fail",
                    disabled=task != "Classification",
                )
                if task != "Classification":
                    derive_pass_fail = False
                scenario_options = ["Early prediction", "Grade-assisted prediction"]
                scenario = st.radio(
                    "Prediction scenario",
                    scenario_options,
                    index=min(
                        scenario_options.index(defaults["scenario"])
                        if defaults["scenario"] in scenario_options else 0,
                        len(scenario_options) - 1,
                    ),
                    key="experiment_scenario",
                )
            else:
                scenario = "General dataset"
                st.caption("The selected target column is excluded from all input features.")

            st.subheader("Evaluation design")
            first, second = st.columns(2)
            with first:
                test_size = st.slider(
                    "Held-out test set (%)",
                    10,
                    40,
                    min(max(defaults["test_size"], 10), 40),
                    5,
                    key="experiment_test_size",
                )
                cv_folds = st.slider(
                    "Cross-validation folds",
                    2,
                    10,
                    min(max(defaults["cv_folds"], 2), 10),
                    key="experiment_cv_folds",
                )
            with second:
                random_seed = st.number_input(
                    "Random seed",
                    0,
                    9999,
                    min(max(defaults["seed"], 0), 9999),
                    1,
                    key="experiment_seed",
                )
                st.caption("Stratified CV is used for classification; shuffled K-fold for regression.")

        with right:
            st.subheader("Automated optimization")
            automl_enabled = st.checkbox(
                "Enable Optuna hyperparameter search",
                value=defaults["automl_enabled"],
                key="experiment_automl_enabled",
            )
            automl_trials = st.slider(
                "Optuna trial budget",
                2,
                30,
                min(max(defaults["automl_trials"], 2), 30),
                disabled=not automl_enabled,
                key="experiment_automl_trials",
            )
            scoring_options = list(supported_scoring(task))
            scoring_default = defaults["automl_scoring"]
            if scoring_default not in scoring_options:
                scoring_default = scoring_options[0]
            automl_scoring = st.selectbox(
                "Training-CV selection metric",
                scoring_options,
                index=scoring_options.index(scoring_default),
                disabled=not automl_enabled,
                key="experiment_automl_scoring",
            )
            feature_methods = st.multiselect(
                "Feature-selection strategies",
                ["none", "variance", "mutual_info", "model_based"],
                default=defaults["feature_methods"],
                disabled=not automl_enabled,
                key="experiment_feature_methods",
            )
            if not feature_methods:
                st.warning("Choose at least one feature strategy when Optuna is enabled.")
            st.caption(
                "Preprocessing, feature selection, and model fitting are cross-validated together. "
                "Optuna does not see held-out metrics."
            )

        if target == "G3" and scenario == "Early prediction":
            st.warning(
                "Leakage protection is active: G1, G2, G3, and the derived target are excluded "
                "from model inputs. Only features available before the final grade are eligible."
            )
        elif target == "G3":
            st.info(
                "Grade-assisted prediction uses G1/G2 as earlier assessments. G3 and the derived "
                "target remain excluded from inputs."
            )

        run_requested = st.form_submit_button(
            "Run evaluation",
            type="primary",
            width="stretch",
        )

    settings = {
        "target": target,
        "task": task,
        "scenario": scenario,
        "derive_pass_fail": derive_pass_fail,
        "test_size": test_size,
        "cv_folds": cv_folds,
        "seed": int(random_seed),
        "automl_enabled": automl_enabled,
        "automl_trials": automl_trials,
        "automl_scoring": automl_scoring,
        "feature_methods": feature_methods or ["none"],
    }
    with st.expander("Review experiment configuration"):
        st.dataframe(
            pd.DataFrame([
                {"Setting": str(name), "Value": str(value)}
                for name, value in (
                    ("Target", target),
                    ("Task", task),
                    ("Prediction scenario", scenario),
                    ("Test fraction", f"{test_size}%"),
                    ("CV folds", cv_folds),
                    ("Seed", int(random_seed)),
                    ("Optuna", "Enabled" if automl_enabled else "Disabled"),
                    ("Feature selection", ", ".join(feature_methods or ["none"])),
                )
            ]),
            width="stretch",
            hide_index=True,
        )
    return settings, run_requested


def _render_training(
    experiment: dict | None,
    active: bool,
    dataset_name: str,
    rows: int,
) -> None:
    _render_header_for_page("Model Training & Comparison", dataset_name, rows)
    if not active or experiment is None:
        st.info("Run the configured experiment to populate cross-validation and comparison results.")
        render_panel("Training status", "Not started", "No progress or performance is shown until a run completes.")
        return

    comparison = experiment["baseline_comparison"]
    task = experiment["task"]
    winner = experiment["baseline_winner"]
    selection_column = "cv_f1_macro_mean" if task == "Classification" else "cv_mae_mean"
    ascending = task == "Regression"
    ordered = comparison.sort_values(selection_column, ascending=ascending)
    best = ordered.iloc[0]
    selection_value = best[selection_column]
    if task == "Regression":
        cv_label = "Training CV MAE"
    else:
        cv_label = "Training CV Macro F1"

    render_metric_cards([
        ("Training status", "Complete"),
        ("CV-selected baseline", str(winner)),
        (cv_label, _format_metric(selection_value, 4)),
        ("CV folds", str(experiment["cv_folds"])),
    ])
    st.subheader("Baseline model comparison")
    chart_frame = ordered[["model", selection_column]].rename(
        columns={"model": "Model", selection_column: "CV score"}
    )
    render_bar_chart(
        chart_frame,
        "Model",
        "CV score",
        "Training cross-validation results",
        horizontal=True,
        height=330,
    )
    st.dataframe(ordered, width="stretch", hide_index=True)

    if experiment["model_errors"]:
        st.warning("Some baseline candidates did not complete successfully.")
        st.json(experiment["model_errors"])
    else:
        st.success(f"Selected from training CV: {winner}")

    result = experiment["automl_result"]
    if result is not None:
        st.subheader("Optuna search")
        result_left, result_right = st.columns((1, 1.4))
        with result_left:
            render_panel(
                "Optuna-selected model",
                str(result.metadata.get("model_family", "Unknown")),
                f"Feature strategy: {result.metadata.get('feature_method', 'Unknown')}",
            )
            st.caption(
                f"{result.metadata.get('completed_trial_count', len(result.trials))} completed of "
                f"{result.metadata.get('trial_count', len(result.trials))} trials · "
                f"selection metric: {experiment['automl_scoring']}"
            )
        with result_right:
            if not result.trials.empty and "number" in result.trials and "cv_score" in result.trials:
                trial_frame = result.trials.dropna(subset=["cv_score"]).copy()
                render_line_chart(
                    trial_frame,
                    "number",
                    "cv_score",
                    "Observed Optuna trial scores",
                    height=290,
                )
        st.dataframe(result.trials, width="stretch", hide_index=True)
        st.subheader("Optuna final held-out evaluation")
        if task == "Classification":
            metrics = result.final_metrics
            render_metric_cards([
                ("Accuracy", _format_metric(metrics.get("accuracy", "Not available"), 4)),
                ("Macro F1", _format_metric(metrics.get("f1_macro", "Not available"), 4)),
                ("Balanced accuracy", _format_metric(metrics.get("balanced_accuracy", "Not available"), 4)),
            ], columns=3)
        else:
            metrics = result.final_metrics
            render_metric_cards([
                ("MAE", _format_metric(metrics.get("mae", "Not available"), 4)),
                ("RMSE", _format_metric(metrics.get("rmse", "Not available"), 4)),
                ("R²", _format_metric(metrics.get("r2", "Not available"), 4)),
            ], columns=3)
    st.subheader("Explainable AI")
    if result is None:
        st.caption("Enable Optuna to access model-specific global and individual explanations.")
    else:
        st.caption("Global feature importance and individual explanations are available in the Explainable AI workspace.")


def _render_roc_curve(result: object, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    pipeline = getattr(result, "best_pipeline", None)
    if pipeline is None or y_test.nunique() != 2:
        st.caption("ROC curve is not available for this evaluation.")
        return
    try:
        classes = list(pipeline.classes_)
        positive_label = classes[-1]
        if hasattr(pipeline, "predict_proba"):
            scores = pipeline.predict_proba(X_test)[:, classes.index(positive_label)]
        elif hasattr(pipeline, "decision_function"):
            decision = pipeline.decision_function(X_test)
            scores = decision if getattr(decision, "ndim", 1) == 1 else decision[:, classes.index(positive_label)]
        else:
            st.caption("ROC curve is omitted because the selected model exposes no ranking score.")
            return
        false_positive_rate, true_positive_rate, _ = roc_curve(
            y_test == positive_label,
            scores,
        )
    except (AttributeError, IndexError, ValueError) as exc:
        st.caption(f"ROC curve unavailable: {exc}")
        return
    roc_frame = pd.DataFrame({
        "False positive rate": false_positive_rate,
        "True positive rate": true_positive_rate,
    })
    render_line_chart(roc_frame, "False positive rate", "True positive rate", "ROC curve", height=340)
    st.caption(f"Positive class: {positive_label}. ROC curve is shown for the Optuna-selected model only.")


def _render_evaluation(experiment: dict | None, active: bool, dataset_name: str, rows: int) -> None:
    _render_header_for_page("Evaluation Results", dataset_name, rows)
    if not active or experiment is None:
        st.info("Final held-out evaluation appears after model selection and training.")
        return

    task = experiment["task"]
    metrics = experiment["baseline_metrics"]
    if task == "Classification":
        render_metric_cards([
            ("Accuracy", _format_metric(metrics.get("accuracy"), 4)),
            ("Macro F1", _format_metric(metrics.get("f1_macro"), 4)),
            ("Balanced accuracy", _format_metric(metrics.get("balanced_accuracy"), 4)),
            ("ROC-AUC", _format_metric(metrics.get("roc_auc"), 4) if "roc_auc" in metrics else "N/A"),
        ])
        left, right = st.columns((1, 1.1))
        labels = [str(label) for label in metrics["labels"]]
        with left:
            st.subheader("Confusion matrix")
            render_confusion_matrix(metrics["confusion_matrix"], labels)
        with right:
            st.subheader("Classification report")
            report_frame = pd.DataFrame(metrics["classification_report"]).transpose()
            st.dataframe(report_frame, width="stretch")
        st.caption("Precision, recall, and F1 are shown for each class. ROC-AUC is a ranking score, not a calibrated probability.")
        automl_result = experiment["automl_result"]
        if automl_result is not None:
            st.subheader("Optuna-selected model · ROC analysis")
            _render_roc_curve(automl_result, experiment["split"].X_test, experiment["split"].y_test)
            optuna_metrics = automl_result.final_metrics
            st.subheader("Baseline and Optuna test metrics")
            compare_rows = [{
                "Model": f"CV-selected baseline · {experiment['baseline_winner']}",
                "Accuracy": metrics.get("accuracy"),
                "Macro F1": metrics.get("f1_macro"),
                "Balanced accuracy": metrics.get("balanced_accuracy"),
            }, {
                "Model": f"Optuna · {automl_result.metadata.get('model_family', 'selected')}",
                "Accuracy": optuna_metrics.get("accuracy"),
                "Macro F1": optuna_metrics.get("f1_macro"),
                "Balanced accuracy": optuna_metrics.get("balanced_accuracy"),
            }]
            st.dataframe(pd.DataFrame(compare_rows), width="stretch", hide_index=True)
    else:
        render_metric_cards([
            ("MAE", _format_metric(metrics.get("mae"), 4)),
            ("RMSE", _format_metric(metrics.get("rmse"), 4)),
            ("R²", _format_metric(metrics.get("r2"), 4)),
        ], columns=3)
        prediction_frame = pd.DataFrame({
            "Actual": metrics["actual"],
            "Predicted": metrics["predictions"],
        })
        render_scatter_chart(prediction_frame, "Actual", "Predicted", "Actual vs. predicted", height=360)
        residual_frame = pd.DataFrame({
            "Predicted": metrics["predictions"],
            "Residual": metrics["residuals"],
        })
        render_scatter_chart(residual_frame, "Predicted", "Residual", "Residual analysis", height=320)
        if experiment["automl_result"] is not None:
            st.subheader("Optuna-selected model metrics")
            st.dataframe(pd.DataFrame([{
                key: experiment["automl_result"].final_metrics[key]
                for key in ("mae", "rmse", "r2")
            }            ]), width="stretch", hide_index=True)


def _render_xai(
    experiment: dict | None,
    active: bool,
    dataset_name: str,
    rows: int,
) -> None:
    _render_header_for_page("Explainable AI", dataset_name, rows)
    st.info("Model explanations describe fitted-model behavior and do not establish causation.")
    if not active or experiment is None:
        st.info("Run an Optuna experiment to make a fitted selected pipeline available for explanations.")
        return
    result = experiment["automl_result"]
    if result is None:
        st.info("Enable Optuna tuning in Experiment Setup to access the selected pipeline's explanations.")
        return

    split = experiment["split"]
    task = experiment["task"]
    global_left, global_right = st.columns((1.1, 1))
    with global_left:
        st.subheader("Global permutation importance")
        st.caption("Held-out post-selection diagnostic; higher values indicate a larger metric decrease after permutation.")
        try:
            importance = permutation_explanation(
                result.best_pipeline,
                split.X_test,
                split.y_test,
                task,
                seed=experiment["seed"],
                n_repeats=5,
            )
        except ExplanationError as exc:
            st.warning(f"Permutation importance unavailable: {exc}")
        else:
            top_features = importance.head(15).copy()
            render_bar_chart(
                top_features,
                "feature",
                "importance_mean",
                "Permutation importance by input feature",
                horizontal=True,
                height=390,
            )
            st.dataframe(importance, width="stretch", hide_index=True)
    with global_right:
        st.subheader("Model-specific importance")
        try:
            native, method = model_specific_explanation(result.best_pipeline)
        except ExplanationError as exc:
            st.info(f"Native explanation unavailable: {exc}")
        else:
            st.caption(method)
            st.dataframe(native.head(20), width="stretch", hide_index=True)
    st.divider()
    st.subheader("Individual student explanation")
    source = st.radio(
        "Prediction record",
        ["Held-out test record", "Upload one-row CSV"],
        horizontal=True,
        key="xai_record_source",
    )
    record = None
    if source == "Held-out test record":
        record_index = st.selectbox(
            "Select held-out record",
            range(len(split.X_test)),
            format_func=lambda index: f"Record {index + 1}",
            key="xai_heldout_record",
        )
        record = split.X_test.iloc[[record_index]]
    else:
        uploaded_record = st.file_uploader(
            "One-row prediction CSV",
            type=["csv"],
            key="xai_uploaded_record",
        )
        if uploaded_record is not None:
            try:
                record_frame, _ = load_csv(uploaded_record)
                if len(record_frame) != 1:
                    raise DatasetValidationError("Prediction CSV must contain exactly one row.")
                record_frame = record_frame.drop(columns=[split.y_test.name], errors="ignore")
                missing_columns = [column for column in split.X_test.columns if column not in record_frame.columns]
                if missing_columns:
                    raise DatasetValidationError("Missing prediction features: " + ", ".join(missing_columns))
                record = record_frame[split.X_test.columns]
            except DatasetValidationError as exc:
                st.error(str(exc))

    if record is not None:
        try:
            local = predict_with_explanation(result.best_pipeline, record, task)
        except ExplanationError as exc:
            st.warning(f"Individual explanation unavailable: {exc}")
        else:
            st.subheader("Model output")
            st.metric("Predicted outcome", str(local.prediction))
            if local.probabilities:
                scores = pd.DataFrame([
                    {"Class": label, "Model score": score}
                    for label, score in local.probabilities.items()
                ])
                st.dataframe(scores, width="stretch", hide_index=True)
                st.caption("Model scores are not guaranteed calibrated probabilities.")
            st.subheader("Local feature perturbation")
            st.caption(f"{local.method}. These are model sensitivities, not causal effects.")
            render_bar_chart(
                local.contributions,
                "feature",
                "effect",
                "Change in model output after one-feature perturbation",
                horizontal=True,
                height=340,
            )
            st.dataframe(local.contributions, width="stretch", hide_index=True)


def _render_saved_experiments(dataset_name: str, rows_count: int) -> None:
    _render_header_for_page("Saved Experiments", dataset_name, rows_count)
    st.caption("Metadata is read from JSON files only; serialized model files are never loaded in this view.")
    metadata_files = (
        sorted(Path("models").glob("*.json"), key=lambda path: path.name, reverse=True)
        if Path("models").is_dir() else []
    )
    if not metadata_files:
        st.info("No timestamped experiment metadata has been saved in models/. Run Optuna to create one.")
        return

    rows = []
    metadata_by_name = {}
    read_errors = []
    for path in metadata_files:
        try:
            metadata = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            read_errors.append(f"{path.name}: {exc}")
            continue
        metadata_by_name[path.name] = metadata
        rows.append({
            "Experiment": path.stem,
            "Saved at (UTC)": metadata.get("timestamp_utc", "Unknown"),
            "Task": metadata.get("task", "Unknown"),
            "Model": metadata.get("model_family", "Unknown"),
            "Scenario": metadata.get("scenario", "Metadata not recorded"),
            "Feature selection": metadata.get("feature_method", "Unknown"),
            "Selection metric": metadata.get("scoring", "Unknown"),
            "CV folds": metadata.get("cv_folds", "Unknown"),
            "CV score": metadata.get("cv_score", "Unknown"),
            "Dataset": metadata.get("dataset_identifier", "Unknown"),
        })
    if rows:
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        selected_name = st.selectbox("Inspect experiment metadata", list(metadata_by_name))
        st.json(metadata_by_name[selected_name])
    else:
        st.warning("No valid experiment metadata could be read.")
    if read_errors:
        with st.expander(f"Metadata read errors ({len(read_errors)})"):
            for error in read_errors:
                st.error(error)


def _render_validation(dataset_name: str, rows_count: int) -> None:
    _render_header_for_page("Rigorous Validation", dataset_name, rows_count)
    st.caption("Read-only display of timestamped Phase 6 JSON reports; no model artifacts are loaded.")
    reports = (
        sorted(Path("reports").glob("phase6_validation_*.json"), reverse=True)
        if Path("reports").is_dir() else []
    )
    if not reports:
        st.info("No Phase 6 validation report is available. Run run_phase6_validation.py to validate the bundled datasets.")
        return
    try:
        report = json.loads(reports[0].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        st.error(f"The latest validation report could not be read: {exc}")
        return
    st.success(f"Latest report · {reports[0].name}")
    profiles = report.get("source_profiles", [])
    if profiles:
        st.subheader("Dataset comparison")
        profile_rows = [{
            "Dataset": item.get("dataset"),
            "Records": item.get("shape", [None, None])[0],
            "Features": item.get("shape", [None, None])[1],
            "Missing": item.get("missing_values"),
            "Duplicates": item.get("duplicate_rows"),
            "Pass / Fail": item.get("pass_fail_distribution", {}),
        } for item in profiles]
        st.dataframe(pd.DataFrame(profile_rows), width="stretch", hide_index=True)

    experiments = report.get("experiments", [])
    if experiments:
        filters = st.columns(3)
        dataset_options = sorted({item.get("dataset", "Unknown") for item in experiments})
        task_options = sorted({item.get("task", "Unknown") for item in experiments})
        scenario_options = sorted({item.get("scenario", "Unknown") for item in experiments})
        with filters[0]:
            selected_dataset = st.selectbox("Dataset", dataset_options, key="validation_dataset_filter")
        with filters[1]:
            selected_task = st.selectbox("Task", task_options, key="validation_task_filter")
        with filters[2]:
            selected_scenario = st.selectbox("Scenario", scenario_options, key="validation_scenario_filter")
        selected_experiments = [
            item for item in experiments
            if item.get("dataset") == selected_dataset
            and item.get("task") == selected_task
            and item.get("scenario") == selected_scenario
        ]
        result_rows = []
        for experiment in selected_experiments:
            for row in experiment.get("held_out_summary", []):
                result_rows.append({
                    "Model": row.get("model"),
                    **{
                        key: value for key, value in row.items()
                        if key != "model" and isinstance(value, (int, float, str))
                    },
                })
        if result_rows:
            st.subheader("Held-out performance across seeds")
            st.dataframe(pd.DataFrame(result_rows), width="stretch", hide_index=True)
        selected_run = selected_experiments[0] if selected_experiments else None
        if selected_run:
            st.subheader("Cross-validation selection record")
            cv_rows = selected_run.get("cv_summary", [])
            if cv_rows:
                st.dataframe(pd.DataFrame(cv_rows), width="stretch", hide_index=True)
            by_seed = selected_run.get("seed_runs", [])
            if by_seed:
                seed_rows = []
                class_rows = []
                for seed_run in by_seed:
                    metrics = seed_run.get("held_out_metrics", {})
                    seed_rows.append({
                        "Seed": seed_run.get("seed"),
                        "Selected model": seed_run.get("selected_model"),
                        "Accuracy": metrics.get("accuracy"),
                        "Macro F1": metrics.get("f1_macro"),
                        "Balanced accuracy": metrics.get("balanced_accuracy"),
                    })
                    for class_label, class_metrics in metrics.get("classification_report", {}).items():
                        if isinstance(class_metrics, dict) and "precision" in class_metrics:
                            class_rows.append({
                                "Seed": seed_run.get("seed"),
                                "Class": class_label,
                                "Precision": class_metrics.get("precision"),
                                "Recall": class_metrics.get("recall"),
                                "F1": class_metrics.get("f1-score"),
                                "Support": class_metrics.get("support"),
                            })
                st.dataframe(pd.DataFrame(seed_rows), width="stretch", hide_index=True)
                if class_rows:
                    st.subheader("Class-wise held-out performance")
                    st.dataframe(pd.DataFrame(class_rows), width="stretch", hide_index=True)

    overlap = report.get("dataset_overlap", {})
    if overlap:
        with st.expander("Potential cross-dataset overlap"):
            st.write({
                "Shared-attribute overlap rows": overlap.get("attribute_overlap_rows"),
                "Exact shared-row overlap": overlap.get("exact_shared_row_overlap"),
                "Note": overlap.get("note"),
            })
    st.warning(
        "These estimates do not establish generalization to other schools or populations. "
        "Grade-assisted G1/G2 inclusion is not comparable in prediction-time information availability."
    )
    for warning in report.get("limitations", []):
        st.caption(str(warning))


def _render_no_upload() -> None:
    render_page_header("ACADEMIC ANALYTICS", "Student AutoML", "A clear, reproducible workflow for academic prediction.")
    render_hero(
        "From student data to explainable model evaluation.",
        "Upload a CSV dataset to inspect quality, configure a classification or regression experiment, "
        "compare models, and review final held-out results.",
    )
    st.info("Upload a student dataset from the sidebar to begin.")
    feature_columns = st.columns(3)
    with feature_columns[0]:
        render_panel("Prepare", "Review data quality", "Inspect records, missing values, duplicates, and feature types.")
    with feature_columns[1]:
        render_panel("Evaluate", "Compare with cross-validation", "Choose a target, scenario, and evaluation strategy.")
    with feature_columns[2]:
        render_panel("Explain", "Inspect model behavior", "Review global and individual explanations after training.")


def main() -> None:
    st.set_page_config(page_title="Student AutoML", page_icon="SA", layout="wide")
    apply_dashboard_style()

    render_brand()
    st.sidebar.markdown("**WORKSPACE**")
    uploaded_file = st.sidebar.file_uploader(
        "Student dataset",
        type=["csv"],
        help="Comma, semicolon, tab, and pipe-delimited CSV files are supported.",
    )
    st.sidebar.caption("Metadata is read from JSON files only; serialized model files are never loaded in this view.")
    st.sidebar.divider()
    st.sidebar.caption("PROJECT")
    st.sidebar.caption("AutoML-based academic performance prediction with automated feature selection, optimization, and explainability.")
    with st.sidebar.expander("Project details"):
        st.caption(TITLE)

    # Keep the legacy tab labels in the app tree for existing AppTest clients;
    # navigation and page content are controlled by the sidebar workspace.
    st.tabs([
        "Dataset Overview",
        "Experiment Setup",
        "Model Training and Comparison",
        "Evaluation Results",
        "Explainable AI",
        "Saved Experiments",
        "Rigorous Validation",
    ])

    if uploaded_file is None:
        _render_no_upload()
        st.stop()

    try:
        frame, delimiter = load_csv(uploaded_file)
    except DatasetValidationError as exc:
        st.error(str(exc))
        st.stop()

    file_bytes = uploaded_file.getvalue()
    dataset_fingerprint = hashlib.sha256(file_bytes).hexdigest()
    if st.session_state.get("dataset_fingerprint") != dataset_fingerprint:
        st.session_state[STATE_KEY] = None
        st.session_state["dataset_fingerprint"] = dataset_fingerprint
        for key in CONFIG_KEYS:
            st.session_state.pop(key, None)
        st.session_state[NAV_KEY] = "Experiment Setup"

    if NAV_KEY not in st.session_state:
        st.session_state[NAV_KEY] = NAVIGATION[0]
    workspace = st.sidebar.radio(
        "Workspace navigation",
        NAVIGATION,
        key=NAV_KEY,
        label_visibility="collapsed",
    )
    st.success(f"Dataset ready · {uploaded_file.name} · {len(frame):,} records · {frame.shape[1]:,} features")

    defaults = _configuration_defaults(frame)
    run_requested = False
    if workspace == "Experiment Setup":
        settings, run_requested = _render_setup(frame, uploaded_file.name, )
        defaults.update(settings)
    else:
        settings = defaults

    scenario = settings["scenario"]
    signature = (
        dataset_fingerprint,
        settings["target"],
        settings["task"],
        scenario,
        settings["derive_pass_fail"],
        settings["test_size"],
        settings["cv_folds"],
        int(settings["seed"]),
        settings["automl_enabled"],
        int(settings["automl_trials"]),
        settings["automl_scoring"],
        tuple(settings["feature_methods"]),
    )
    experiment = st.session_state.get(STATE_KEY)
    active_experiment = (
        experiment
        if experiment is not None and experiment.get("signature") == signature
        else None
    )

    if run_requested:
        try:
            X, y = prepare_target(
                frame,
                settings["target"],
                settings["task"],
                derive_pass_fail=settings["derive_pass_fail"],
            )
            X = select_features(X, settings["target"], scenario)
            if X.empty or X.shape[1] == 0:
                raise DatasetValidationError(
                    "No input features remain after applying target and scenario protections."
                )
            split = reproducible_split(
                X,
                y,
                settings["task"],
                test_size=settings["test_size"] / 100,
                seed=int(settings["seed"]),
            )
            if settings["task"] == "Classification":
                models = {
                    "Logistic Regression": LogisticRegression(max_iter=2000, random_state=int(settings["seed"])),
                    "Random Forest": RandomForestClassifier(
                        n_estimators=150, random_state=int(settings["seed"]), n_jobs=1,
                    ),
                    "Extra Trees": ExtraTreesClassifier(
                        n_estimators=150, random_state=int(settings["seed"]), n_jobs=1,
                    ),
                }
            else:
                models = {
                    "Linear Regression": LinearRegression(),
                    "Random Forest": RandomForestRegressor(
                        n_estimators=150, random_state=int(settings["seed"]), n_jobs=1,
                    ),
                    "Extra Trees": ExtraTreesRegressor(
                        n_estimators=150, random_state=int(settings["seed"]), n_jobs=1,
                    ),
                }

            with st.status("Cross-validating baseline models", expanded=True) as status:
                baseline_comparison, baseline_metrics, model_errors, baseline_winner = build_baseline_comparison(
                    split.X_train,
                    split.y_train,
                    split.X_test,
                    split.y_test,
                    settings["task"],
                    models,
                    folds=int(settings["cv_folds"]),
                    seed=int(settings["seed"]),
                )
                status.update(label="Baseline comparison complete", state="complete")

            automl_result = None
            saved_paths = None
            if settings["automl_enabled"]:
                with st.status("Running Optuna hyperparameter search", expanded=True) as status:
                    progress = st.progress(0.0)

                    def update_progress(completed: int, total: int, model_name: str) -> None:
                        if total:
                            progress.progress(
                                completed / total,
                                text=f"Observed trial {completed}/{total}: {model_name}",
                            )

                    automl_result = run_automl_search(
                        split.X_train,
                        split.y_train,
                        split.X_test,
                        split.y_test,
                        settings["task"],
                        settings["automl_scoring"],
                        n_trials=int(settings["automl_trials"]),
                        folds=int(settings["cv_folds"]),
                        seed=int(settings["seed"]),
                        feature_methods=settings["feature_methods"],
                        progress_callback=update_progress,
                    )
                    saved_paths = save_automl_result(automl_result, Path("models"), "uploaded")
                    status.update(label="Optuna tuning complete", state="complete")

            st.session_state[STATE_KEY] = {
                "signature": signature,
                "target": settings["target"],
                "task": settings["task"],
                "scenario": scenario,
                "test_size": settings["test_size"],
                "cv_folds": settings["cv_folds"],
                "seed": int(settings["seed"]),
                "automl_scoring": settings["automl_scoring"],
                "split": split,
                "baseline_comparison": baseline_comparison,
                "baseline_metrics": baseline_metrics,
                "model_errors": model_errors,
                "baseline_winner": baseline_winner,
                "automl_result": automl_result,
                "saved_paths": saved_paths,
                "dataset_name": uploaded_file.name,
            }
            st.success("Experiment completed. Detailed results are available in the workspace navigation.")
            _render_training(
                st.session_state[STATE_KEY],
                True,
                uploaded_file.name,
                len(frame),
            )
            return
        except (DatasetValidationError, EvaluationError) as exc:
            st.error(str(exc))
            st.stop()
        except Exception as exc:
            st.error(f"Training could not complete: {type(exc).__name__}: {exc}")
            st.stop()

    if workspace == "Dataset Overview":
        _render_overview(frame, uploaded_file.name, delimiter)
    elif workspace == "Model Training & Comparison":
        _render_training(
            active_experiment,
            active_experiment is not None,
            uploaded_file.name,
            len(frame),
        )
    elif workspace == "Evaluation Results":
        _render_evaluation(
            active_experiment,
            active_experiment is not None,
            uploaded_file.name,
            len(frame),
        )
    elif workspace == "Explainable AI":
        _render_xai(
            active_experiment,
            active_experiment is not None,
            uploaded_file.name,
            len(frame),
        )
    elif workspace == "Saved Experiments":
        _render_saved_experiments(uploaded_file.name, len(frame))
    elif workspace == "Rigorous Validation":
        _render_validation(uploaded_file.name, len(frame))


main()
