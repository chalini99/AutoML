# Student AutoML

A Streamlit project for evaluating academic-performance data with classification and regression baselines. The current implementation supports robust CSV upload, target selection, student early versus grade-assisted prediction scenarios, leakage-safe baselines, and scientific held-out evaluation.

## Experimental methodology

1. The target column is removed from the input features.
2. Missing target rows are excluded before analysis.
3. A fixed-seed split reserves the configured test fraction. Classification uses stratified splitting when every class has enough rows; regression uses a random split.
4. Input preprocessing, including missing-value imputation, scaling, and categorical encoding, is implemented inside the scikit-learn pipeline. It is fitted separately in every cross-validation fold.
5. Classification uses shuffled `StratifiedKFold`; regression uses shuffled `KFold`. Requested folds are accepted only when the available training data supports them.
6. Baseline models are compared using training-only cross-validation. The selected model is the highest training-only macro-F1 model for classification or lowest training-only MAE model for regression.
7. The untouched test set is evaluated exactly once after model selection. Its metrics are explicitly labelled as final held-out evaluation.

## Classification metrics

The dashboard reports accuracy, balanced accuracy, macro precision, macro recall, macro F1, weighted F1, ROC-AUC when a valid probability or decision score is available, a confusion matrix, and a per-class classification report.

## Regression metrics

The dashboard reports MAE, RMSE, and R². It also provides actual-versus-predicted and residual plots.

## Phase 3 Optuna tuning

The optional Optuna workflow uses the same target preparation and held-out split as the baselines. It tunes model-family hyperparameters and optional feature-selection strategies using only shuffled training-fold cross-validation. Supported feature handling is no selection, variance thresholding, mutual information filtering, or model-based selection; preprocessing and selection both occur inside each CV pipeline.

Classification tuning maximizes macro F1, balanced accuracy, accuracy, or weighted F1. Regression tuning maximizes scikit-learn R² or converts negative MAE/RMSE training scores back to positive error values for reporting. Optuna study selection does not use held-out test data. The tuned pipeline is fitted on all training rows and evaluated once on the untouched test set.

When enabled, timestamped pipeline, metadata, comparison, and trial files are written beneath `models/`. Existing artifacts are never overwritten. Trial failures are recorded and do not stop successful trials. The CLI entry point remains available:

```powershell
.\venv\Scripts\python.exe run_automl.py
```

Optuna tuning is CPU-bound and may take longer than baseline evaluation. The configured trial count, folds, feature strategies, random seed, and dataset size affect runtime and metrics. It is experimental model selection, not evidence of external validity.

## Phase 4 Explainable AI

After the Optuna workflow selects a fitted pipeline, the Explainable AI section calculates held-out post-selection explanations without using test data for tuning or model selection:

- Permutation importance measures how much the selected metric worsens when each original input feature is shuffled repeatedly. Importance variability is shown as its standard deviation.
- Tree-based pipelines expose native feature importance. Linear pipelines expose coefficients: coefficients for a binary classification model or mean absolute coefficients across multiclass directions. Unsupported model families show a clear limitation instead of inventing an explanation.
- One held-out record can be selected for a local explanation, or a one-row CSV can be uploaded. The local method changes one input at a time and reports the change in predicted probability or numerical prediction.

Preprocessing and feature selection are respected by reading the fitted pipeline. Encoded feature names are mapped back to original input columns when unambiguous. SHAP is not required; the Phase 4 interface uses CPU-lightweight scikit-learn tools and does not introduce a mandatory dependency.

Prediction probabilities are model output scores, not proof of calibration or clinical/academic risk. Explanations are associations from the fitted model, not causal effects. A KNN or unsupported model may support permutation and individual prediction explanations but cannot provide native coefficient or tree-importance values. If no Optuna run is enabled, the interface prompts the user to run one rather than explaining an unrelated fitted model.

## Phase 5 dashboard workflow

The Streamlit interface uses a sidebar-led workspace navigation:

1. **Dataset Overview**: row/column counts, missing values, duplicate rows, data types, preview, target distribution, and data quality indicators.
2. **Experiment Setup**: target and task, early or grade-assisted scenario, excluded columns, test fraction, CV folds, seed, feature selection, and Optuna settings.
3. **Model Training and Comparison**: baseline status and training-only CV scores, with failed model summaries kept separate from successful results.
4. **Evaluation Results**: clearly labeled held-out metrics, classification report, confusion matrix, and plots for the selected baseline workflow, plus Optuna results when enabled.
5. **Explainable AI**: held-out permutation importance, supported native explanations, and optional individual predictions.
6. **Saved Experiments**: metadata-only discovery of timestamped JSON sidecars from `models/`. Serialized `.joblib` model files are not loaded by the dashboard.
7. **Rigorous Validation**: latest validation report, multi-seed performance, class-wise results, and validation limitations.

The interface uses a navy-and-white visual theme, reusable metric and panel components, and compact responsive charts. Plotly charts are used when Plotly is available; otherwise, the dashboard falls back to native Streamlit charts without requiring an additional dependency.

Experiment training runs only after **Run evaluation**. Completed results are retained in Streamlit session state, so changing a display setting or interacting with the explanation fragment does not repeat Optuna or baseline training. Starting a new CSV upload creates a separate result context. Workspace and chart changes are presentation-only and never affect model selection.

Saved metadata is informational and may be incomplete or malformed. The dashboard skips unreadable files and does not silently fill in missing fields or overwrite existing artifacts. A model artifact should be loaded only through a separately trusted, explicit workflow.

## Phase 6 rigorous validation

The project includes a read-only validation runner for the bundled UCI Student Performance data:

```powershell
.\venv\Scripts\python.exe run_phase6_validation.py
```

The runner reads `data/raw/student-mat.csv` and `data/raw/student-por.csv` without modifying them, validates dimensions and schemas, records missing values and duplicates, checks G3/pass-fail distributions, and reports a conservative overlap indicator across shared student attributes. It does not combine the two datasets or treat overlapping attributes as proof of student identity.

For each dataset and task, the runner uses both prediction scenarios:

- **Early prediction** excludes `G1` and `G2`.
- **Grade-assisted prediction** includes `G1` and `G2`, only after those earlier assessments are available.
- Pass/fail derives from `G3 >= 10`: G3 itself is removed from features and never used as an input. A G3 value of at least 10 is **Pass**; every lower value is **Fail**.

Classification compares a majority baseline, logistic regression, and extra trees using five-fold shuffled stratified CV and held-out test splits at seeds 42, 43, and 44. The report records accuracy, balanced accuracy, macro precision/recall/F1, weighted F1, per-class support, and confusion-matrix test values. Regression compares a mean baseline, linear regression, and extra trees using five-fold shuffled KFold, MAE, RMSE, and R². All preprocessing, imputation, encoding, scaling, and models stay inside the CV pipeline.

A fresh uniquely timestamped JSON report is written to `reports/`, preserving prior artifacts. The Streamlit **Rigorous Validation** tab displays the latest report only from JSON metadata and never loads serialized models. Cross-CV variability is reported across seeds. Held-out values are based on the untouched test partitions and are not used for model selection.

The current report is a small-sample validation of these two datasets, not a claim of performance improvement or generalization to other schools or populations. Classification and regression scenarios are also not interchangeable: including G1/G2 changes the information available at prediction time. Negative R² and unstable estimates may occur and must be interpreted as evidence of poor or limited generalization rather than fabricated performance.

## Focused early-classification experiment

The focused experiment compares the existing classifiers with class-weighted Logistic Regression and Random Forest, Extra Trees, SVM, HistGradientBoosting, XGBoost, and LightGBM. Run both datasets with:

```powershell
.\venv\Scripts\python.exe run_early_classification.py
```

Use `--dataset student-mat.csv` or `--dataset student-por.csv` to run one dataset. The experiment uses identical stratified 80/20 splits for all candidates at seeds 42, 43, and 44; five-fold shuffled stratified training CV; and eight Optuna trials per model family, seed, and dataset. Optuna maximizes training-CV macro F1. Preprocessing, feature selection (none, variance threshold, or mutual information), and estimator-native class weighting are fitted inside CV pipelines. No resampling is used. G1, G2, G3, and recognized G3/pass-fail target-derived columns are excluded.

The final test partitions are not accessed until every candidate's tuning is complete and the CV winner for that seed is frozen. Per-seed selected classifiers varied, so the primary three-seed held-out scores below average the individual CV-selected classifiers; the winning model family by mean CV macro F1 and its separate diagnostics are also shown. Standard deviations are across the three seeded splits, not independent external validation samples.

| Dataset | Per-seed CV-selected models | Held-out accuracy | Held-out macro F1 | Balanced accuracy |
|---|---|---:|---:|---:|
| Math | Class-weighted Logistic Regression; Class-weighted Random Forest; Extra Trees | 0.6498 +/- 0.0773 | 0.6004 +/- 0.0967 | 0.6051 +/- 0.1026 |
| Portuguese | Class-weighted Random Forest; Random Forest; XGBoost | 0.7872 +/- 0.0311 | 0.6341 +/- 0.0703 | 0.6561 +/- 0.0898 |

Across all seeds, **Class-weighted Random Forest** had the highest mean training-CV macro F1 on both datasets: 0.6589 +/- 0.0407 for Math and 0.7318 +/- 0.0385 for Portuguese. Its held-out accuracy and macro F1 were 0.6498 +/- 0.0824 and 0.5858 +/- 0.1237 for Math, and 0.7897 +/- 0.0311 and 0.6267 +/- 0.0768 for Portuguese. This is distinct from the per-seed CV-selected aggregate above.

For that mean-CV-winning model family, mean per-class precision / recall / F1 and pooled confusion matrices (rows are actual Fail, Pass; columns are predicted Fail, Pass) were:

| Dataset | Fail precision / recall / F1 | Pass precision / recall / F1 | Pooled confusion matrix | Pooled predicted Fail / Pass |
|---|---|---|---|---|
| Math | 0.4377 / 0.4231 / 0.4252 | 0.7361 / 0.7610 / 0.7464 | `[[33, 45], [38, 121]]` | 71 / 166 |
| Portuguese | 0.3438 / 0.4500 / 0.3808 | 0.8971 / 0.8515 / 0.8726 | `[[27, 33], [49, 281]]` | 76 / 314 |

Fail students continue to be missed as Pass: this model predicted 45 of 78 Math Fail test cases and 33 of 60 Portuguese Fail test cases as Pass across the three splits. The Pass share is 67.1% in Math and 84.6% in Portuguese; the majority classifier predicts every case as Pass and has zero Fail recall. That imbalance explains why accuracy alone is misleading. Early-only demographic, family, and study-behavior features omit G1/G2 and may not separate the outcomes well; this experiment identifies predictive limitations, not causal explanations.

For context, the previous Phase 6 CV-selected Logistic Regression reported held-out accuracy / macro F1 of 0.6751 / 0.5743 for Math and 0.8179 / 0.5896 for Portuguese. The previous early-classification reports' CV-selected families were Random Forest for Math (0.6582 / 0.5702) and Extra Trees for Portuguese (0.8026 / 0.6780). The current per-seed CV-selected macro F1 is higher than Phase 6 on both datasets, while its accuracy is lower; compared with the prior early-classification baselines, Math macro F1 is higher but Portuguese macro F1 and accuracy are lower. No result reaches 90% accuracy.

The uniquely timestamped experiment reports preserve prior files: [Math JSON](./reports/early_classification_20261031-044633-681161.json), [Math CSV](./reports/early_classification_20261031-044633-681161.csv), [Portuguese JSON](./reports/early_classification_20261031-045431-786094.json), and [Portuguese CSV](./reports/early_classification_20261031-045431-786094.csv). The JSON reports include per-model and per-seed metrics, Optuna parameters, confusion matrices, prediction distributions, errors, and historical baseline references.

## Dataset limitations

The Student Performance datasets are relatively small and may contain overlapping students between the mathematics and Portuguese course records. Performance metrics are single-split estimates and do not establish external validity for other schools or populations. Prior grades (`G1` and `G2`) make grade-assisted prediction easier and may not be available for genuinely early prediction.

## Run locally on Windows

```powershell
cd "c:\Users\Chalini\Downloads\AutoML_Student_Performance_App\Student AutoML"
.\venv\Scripts\activate
python -m streamlit run app.py
```

## Tests

```powershell
.\venv\Scripts\python.exe -m pytest -q
.\venv\Scripts\python.exe -m py_compile app.py src\data_utils.py src\evaluation.py src\automl.py
```

The test suite uses synthetic data for splitting and evaluation checks. Serialized models or datasets are not uploaded by tests.
