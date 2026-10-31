from pathlib import Path

import pandas as pd
import streamlit as st

from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression

from src.data_utils import (
    DatasetValidationError,
    load_csv,
    prepare_target,
    select_features,
    suggest_task,
)
from src.automl import run_automl_search, save_automl_result, supported_scoring
from src.evaluation import (
    EvaluationError,
    build_baseline_comparison,
    reproducible_split,
)
from src.xai import (
    ExplanationError,
    model_specific_explanation,
    permutation_explanation,
    predict_with_explanation,
)


st.set_page_config(page_title="Student AutoML", page_icon="🤖", layout="wide")
st.title("🤖 Student AutoML")
st.write(
    "Upload a dataset, choose a prediction target and task, and review the data before training."
)

uploaded_file = st.file_uploader(
    "Upload CSV",
    type=["csv"],
    help="Comma, semicolon, tab, and pipe delimited CSV files are supported.",
)

if uploaded_file is None:
    st.info("Upload a CSV file to begin.")
    st.stop()

try:
    df, delimiter = load_csv(uploaded_file)
except DatasetValidationError as exc:
    st.error(str(exc))
    st.stop()

st.success(f"Loaded CSV using delimiter {delimiter!r}.")
overview, quality = st.tabs(["Dataset Overview", "Data Quality"])

with overview:
    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Duplicate rows", int(df.duplicated().sum()))

    st.subheader("Preview")
    st.dataframe(df.head(10), width="stretch")

    st.subheader("Column types")
    st.dataframe(
        pd.DataFrame(
            {"Column": df.columns, "Data type": df.dtypes.astype(str)}
        ).reset_index(drop=True),
        width="stretch",
        hide_index=True,
    )

with quality:
    missing = df.isna().sum()
    missing_df = pd.DataFrame(
        {"Missing values": missing, "Missing %": missing / len(df) * 100}
    ).reset_index(names="Column")
    st.dataframe(missing_df, width="stretch", hide_index=True)

st.subheader("Prediction configuration")
with st.form("phase1_configuration"):
    target = st.selectbox("Target column", df.columns)
    suggested = suggest_task(df[target].dropna())
    task = st.selectbox(
        "Prediction task",
        ["Classification", "Regression"],
        index=["Classification", "Regression"].index(suggested),
        help=f"The data suggests {suggested}; confirm this choice explicitly.",
    )

    derive_pass_fail = False
    if target == "G3":
        derive_pass_fail = st.checkbox(
            "Create binary pass/fail target (G3 >= 10)",
            value=True,
            help="Pass means G3 is 10 or greater; fail means G3 is below 10.",
        )

    scenario = "General dataset"
    if target == "G3":
        scenario = st.radio(
            "Student prediction scenario",
            ["Early prediction", "Grade-assisted prediction"],
            horizontal=True,
        )
        if scenario == "Early prediction":
            st.warning("G1 and G2 will be excluded to avoid using prior grades for early prediction.")
        else:
            st.warning(
                "G1 and G2 are prior grades and may reveal outcomes unavailable for genuinely early prediction."
            )

    test_size = st.slider("Held-out test set (%)", 10, 40, 20, 5)
    cv_folds = st.slider("Cross-validation folds", 2, 10, 5)
    random_seed = st.number_input("Random seed", 0, 9999, 42, 1)
    automl_enabled = st.checkbox(
        "Run Optuna tuning after baselines",
        value=False,
        help="Uses only training-fold data for tuning and then evaluates the tuned pipeline once on the held-out set.",
    )
    automl_trials = st.slider("Optuna trials", 2, 30, 12, disabled=not automl_enabled)
    automl_scoring = st.selectbox(
        "Optuna selection metric",
        list(supported_scoring(task)),
        index=0,
        disabled=not automl_enabled,
    )
    feature_methods = st.multiselect(
        "Feature-selection strategies",
        ["none", "variance", "mutual_info", "model_based"],
        default=["none"],
        disabled=not automl_enabled,
        help="Selection methods are fitted inside every cross-validation fold.",
    )
    run_requested = st.form_submit_button("Run evaluation")

if not run_requested:
    st.stop()

try:
    X, y = prepare_target(
        df,
        target,
        task,
        derive_pass_fail=derive_pass_fail,
    )
    X = select_features(X, target, scenario)
except DatasetValidationError as exc:
    st.error(str(exc))
    st.stop()

if X.shape[1] == 0:
    st.error("No input features remain after applying the selected target and scenario.")
    st.stop()

if task == "Classification" and y.nunique() > 30:
    st.error("Classification targets should not contain more than 30 distinct values.")
    st.stop()

try:
    split = reproducible_split(
        X,
        y,
        task,
        test_size=test_size / 100,
        seed=int(random_seed),
    )
except EvaluationError as exc:
    st.error(str(exc))
    st.stop()

if task == "Classification":
    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=int(random_seed)),
        "Random Forest": RandomForestClassifier(n_estimators=150, random_state=int(random_seed), n_jobs=1),
        "Extra Trees": ExtraTreesClassifier(n_estimators=150, random_state=int(random_seed), n_jobs=1),
    }
else:
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=150, random_state=int(random_seed), n_jobs=1),
        "Extra Trees": ExtraTreesRegressor(n_estimators=150, random_state=int(random_seed), n_jobs=1),
    }

with st.status("Cross-validating baseline models...", expanded=True) as status:
    try:
        comparison, final_metrics, model_errors, winner = build_baseline_comparison(
            split.X_train,
            split.y_train,
            split.X_test,
            split.y_test,
            task,
            models,
            folds=int(cv_folds),
            seed=int(random_seed),
        )
        status.update(label="Baseline comparison complete", state="complete")
    except EvaluationError as exc:
        status.update(label="Baseline comparison failed", state="error")
        st.error(str(exc))
        st.stop()

automl_result = None
if automl_enabled:
    with st.status("Tuning an Optuna search inside training-only cross-validation...", expanded=True) as status:
        try:
            progress = st.progress(0.0)

            def update_progress(completed, total, model_name):
                progress.progress(completed / total, text=f"Trial {completed}/{total}: {model_name}")

            automl_result = run_automl_search(
                split.X_train,
                split.y_train,
                split.X_test,
                split.y_test,
                task,
                automl_scoring,
                n_trials=int(automl_trials),
                folds=int(cv_folds),
                seed=int(random_seed),
                feature_methods=feature_methods,
                progress_callback=update_progress,
            )
            saved_paths = save_automl_result(automl_result, Path("models"), "uploaded")
            status.update(label="Optuna tuning complete", state="complete")
        except EvaluationError as exc:
            status.update(label="Optuna tuning failed", state="error")
            st.error(str(exc))
            st.stop()

st.subheader("Experimental configuration")
st.write(
    f"Train/test split: {len(split.X_train)} training rows / {len(split.X_test)} held-out rows; "
    f"test fraction {test_size / 100:.2f}; seed {int(random_seed)}; {int(cv_folds)}-fold "
    f"{task.lower()} cross-validation."
)
st.info(
    f"Model selection metric: {'macro F1' if task == 'Classification' else 'mean absolute error'} "
    "(training-only cross-validation). The held-out test set is used only after model selection."
)

comparison = comparison.rename(columns={
    "model": "Model",
    "cv_f1_macro_mean": "CV Macro F1",
    "cv_f1_macro_std": "CV Macro F1 SD",
    "cv_mae_mean": "CV MAE",
    "cv_mae_std": "CV MAE SD",
})
selection_column = "CV Macro F1" if task == "Classification" else "CV MAE"
comparison = comparison.sort_values(selection_column, ascending=task == "Regression")

st.subheader("Training cross-validation comparison")
st.dataframe(comparison, width="stretch", hide_index=True)
st.bar_chart(comparison.set_index("Model")[[selection_column]])
if model_errors:
    st.warning("Some compatible baselines could not complete. See details below.")
    st.json(model_errors)

st.success(f"Selected model using training CV: {winner}")
st.subheader("Final held-out test evaluation")
st.caption("This is the untouched test set and must not be used to change model selection.")

if task == "Classification":
    test_values = pd.DataFrame([{
        key: value
        for key, value in final_metrics.items()
        if isinstance(value, (int, float))
    }])
    st.dataframe(test_values, width="stretch", hide_index=True)
    st.write("Confusion matrix")
    confusion = pd.DataFrame(
        final_metrics["confusion_matrix"],
        index=[f"Actual {label}" for label in final_metrics["labels"]],
        columns=[f"Predicted {label}" for label in final_metrics["labels"]],
    )
    st.dataframe(confusion, width="stretch")
    st.write("Per-class classification report")
    report = pd.DataFrame(final_metrics["classification_report"]).transpose()
    st.dataframe(report, width="stretch")
    if "roc_auc" not in final_metrics:
        st.caption("ROC-AUC is not shown because the held-out set or model did not support a valid calculation.")
else:
    test_values = pd.DataFrame([{
        key: final_metrics[key] for key in ("mae", "rmse", "r2")
    }])
    st.dataframe(test_values, width="stretch", hide_index=True)
    scatter = pd.DataFrame({
        "Actual": final_metrics["actual"],
        "Predicted": final_metrics["predictions"],
    })
    st.scatter_chart(scatter, x="Actual", y="Predicted")
    residuals = pd.DataFrame({
        "Predicted": final_metrics["predictions"],
        "Residual": final_metrics["residuals"],
    })
    st.scatter_chart(residuals, x="Predicted", y="Residual")

@st.fragment
def render_explanation_section(
    result,
    X_test,
    y_test,
    task,
    seed,
):
    st.subheader("Explainable AI")
    st.caption("Explanations describe the fitted model and test data; they do not establish causation.")

    try:
        global_importance = permutation_explanation(
            result.best_pipeline,
            X_test,
            y_test,
            task,
            seed=seed,
            n_repeats=5,
        )
    except ExplanationError as exc:
        st.warning(f"Held-out permutation importance is unavailable: {exc}")
    else:
        st.write("Global permutation importance")
        st.caption(
            "Uses the untouched held-out test set only after model selection. Larger positive values mean "
            "a feature's permutation caused more loss of the selected metric."
        )
        st.bar_chart(
            global_importance.set_index("feature")[["importance_mean"]],
            horizontal=True,
        )
        st.dataframe(global_importance, width="stretch", hide_index=True)

    try:
        native_importance, native_method = model_specific_explanation(result.best_pipeline)
    except ExplanationError as exc:
        st.info(f"Model-specific explanation unavailable: {exc}")
    else:
        st.write(native_method)
        st.caption(
            "This native explanation reflects fitted model parameters and should not be interpreted as causal effect."
        )
        st.dataframe(native_importance, width="stretch", hide_index=True)

    st.write("Individual prediction explanation")
    record_source = st.radio(
        "Prediction record source",
        ["Held-out test record", "Upload a one-row CSV"],
        horizontal=True,
        key="xai_record_source",
    )
    record = None
    if record_source == "Held-out test record":
        record_index = st.selectbox(
            "Held-out row",
            range(len(X_test)),
            format_func=lambda index: f"Row {index + 1}",
            key="xai_test_row",
        )
        record = X_test.iloc[[record_index]].copy()
    else:
        record_file = st.file_uploader(
            "Upload one-row prediction CSV",
            type=["csv"],
            key="xai_record_upload",
        )
        if record_file is not None:
            try:
                uploaded_record, _ = load_csv(record_file)
                if len(uploaded_record) != 1:
                    raise DatasetValidationError("The prediction CSV must contain exactly one record.")
                target_in_record = getattr(y_test, "name", None)
                if target_in_record in uploaded_record.columns:
                    uploaded_record = uploaded_record.drop(columns=[target_in_record])
                    st.caption("The target column was excluded from the prediction input.")
                missing_columns = [column for column in X_test.columns if column not in uploaded_record.columns]
                if missing_columns:
                    raise DatasetValidationError(
                        "Missing prediction features: " + ", ".join(missing_columns)
                    )
                record = uploaded_record[X_test.columns].copy()
            except DatasetValidationError as exc:
                st.error(str(exc))

    if record is not None:
        try:
            local = predict_with_explanation(result.best_pipeline, record, task)
        except ExplanationError as exc:
            st.warning(f"Individual explanation unavailable: {exc}")
        else:
            st.success(f"Predicted outcome: {local.prediction}")
            if local.probabilities:
                probability_frame = pd.DataFrame(
                    [{"class": label, "score": score} for label, score in local.probabilities.items()]
                )
                st.dataframe(probability_frame, width="stretch", hide_index=True)
                st.caption(
                    "These are model probability scores, not evidence of calibrated clinical or academic risk. "
                    "Scoring validity depends on the model and available classes."
                )
            else:
                st.caption("The selected model does not provide usable prediction probabilities.")
            st.write("Local one-feature perturbation")
            st.caption(local.method)
            st.bar_chart(
                local.contributions.set_index("feature")[["effect"]],
                horizontal=True,
            )
            st.dataframe(local.contributions, width="stretch", hide_index=True)


if automl_result is not None:
    st.subheader("Optuna tuning")
    tuning_metrics = automl_result.metadata
    st.caption(
        f"Selected on {int(cv_folds)} training-only folds using {automl_scoring}; "
        f"seed {int(random_seed)}. Test metrics below are final held-out evaluation."
    )
    st.dataframe(automl_result.comparison, width="stretch", hide_index=True)
    st.write("Trial history")
    st.dataframe(automl_result.trials, width="stretch", hide_index=True)
    st.subheader("Optuna final held-out test evaluation")
    if task == "Classification":
        tuned_test_values = pd.DataFrame([{
            key: value
            for key, value in automl_result.final_metrics.items()
            if isinstance(value, (int, float))
        }])
        st.dataframe(tuned_test_values, width="stretch", hide_index=True)
    else:
        tuned_test_values = pd.DataFrame([{
            key: automl_result.final_metrics[key]
            for key in ("mae", "rmse", "r2")
        }])
        st.dataframe(tuned_test_values, width="stretch", hide_index=True)
    st.success(
        f"Optuna selected {tuning_metrics['model_family']} with "
        f"{tuning_metrics['feature_method']} feature handling."
    )
    st.caption(
        f"Timestamped model and metadata saved to {saved_paths['model']} and {saved_paths['metadata']}."
    )
    render_explanation_section(
        automl_result,
        split.X_test,
        split.y_test,
        task,
        int(random_seed),
    )
else:
    st.subheader("Explainable AI")
    st.info("Enable Optuna tuning and run the evaluation to explain the selected fitted pipeline. "
            "Permuting held-out test data is intentionally shown only after model selection.")

st.caption(
    "These metrics depend on the dataset size and the fixed split. They do not establish external validity."
)
st.download_button(
    "Download cross-validation comparison CSV",
    comparison.to_csv(index=False).encode("utf-8"),
    file_name="student_automl_cv_comparison.csv",
    mime="text/csv",
)
