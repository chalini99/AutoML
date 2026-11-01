# Student AutoML

### AutoML-Based Student Academic Performance Prediction Using Automated Feature Selection, Hyperparameter Optimization, and Explainable AI

The Automated Student Performance Prediction System is a demo-ready school analytics platform built with Streamlit. It opens on a student-success dashboard backed by a bundled cohort, supports individual and batch predictions, and keeps model configuration, evaluation, and explainability available as supporting workspaces.

## Project highlights

- **Demo-first school dashboard:** explore the bundled Math or Portuguese cohort immediately, without preparing an upload.
- **Student-focused prediction scenarios:** configure early prediction or prediction with prior-grade information.
- **Individual and cohort scoring:** enter a student profile or upload a batch CSV and download the results.
- **Cohort support indicators:** review observed achievement and attendance patterns, with historical outcomes clearly distinguished from model predictions.
- **Automated model development:** compare established classifiers and regressors, then use Optuna to search model hyperparameters.
- **Automated feature selection:** evaluate multiple feature-selection strategies as part of the model-training workflow.
- **Reproducible evaluation:** use seeded data splits and cross-validation to compare candidate models consistently.
- **Class-aware analysis:** evaluate per-class precision, recall, and F1 alongside aggregate metrics and confusion matrices.
- **Explainable predictions:** inspect permutation importance, supported native feature importance, and individual prediction explanations.
- **School-facing workspace:** navigate the student-success dashboard, predictions, cohort overview, model studio, evaluation, explainability, saved experiments, and rigorous validation from the sidebar.
- **Experiment history:** save timestamped result artifacts and review experiment metadata in the dashboard.

## Project contribution

The project combines a student-performance workflow with automated experimentation and model transparency. It supports both early and grade-assisted prediction, integrates preprocessing and feature selection with cross-validation, and presents model evaluation and explanations in one interactive workspace.

Early-prediction experiments explicitly exclude `G1`, `G2`, `G3`, and recognized target-derived columns from model inputs. This allows experiments to focus on information available before the final grade. Grade-assisted prediction is configured separately for scenarios where prior grades are available.

## Dashboard workspaces

| Workspace | What it offers |
|---|---|
| **Student Success Dashboard** | Cohort-level achievement and attendance snapshot, observed support watchlist, and direct links to prediction/model workflows |
| **Student Predictions** | Score an individual student using a profile form or score a batch from CSV; download a prediction template and results |
| **Dataset Overview** | Record and feature counts, data quality summaries, target distributions, feature types, and a dataset preview |
| **Model Studio** | Task and target selection, prediction scenario, split and cross-validation settings, random seed, feature selection, and Optuna controls |
| **Model Training & Comparison** | Baseline comparisons, training cross-validation scores, and Optuna trial results |
| **Evaluation Results** | Held-out metrics, classification reports, confusion matrices, and visual comparisons |
| **Explainable AI** | Global feature importance, supported model-specific explanations, and individual predictions for available fitted models |
| **Saved Experiments** | Timestamped experiment metadata and result history |
| **Rigorous Validation** | Multi-seed results and class-wise evaluation for the Math and Portuguese datasets |

The dashboard uses a navy-and-white visual theme with reusable metric cards, consistent page layouts, and responsive charts. It uses Plotly charts when available and Streamlit-native charts otherwise.

## Demonstration workflow

1. Launch the app and explore the bundled Math cohort; choose the Portuguese cohort in the sidebar or upload your own CSV.
2. Open **Model Studio**, select the prediction target and early or grade-assisted scenario, then run an evaluation.
3. Review cross-validation comparisons and held-out evaluation in their dedicated workspaces.
4. Open **Student Predictions** to enter one student's profile or upload a batch using the downloadable feature template.
5. Download prediction results and inspect model explanations when needed.

The final held-out test partition is reserved for evaluation after model selection. Preprocessing, feature selection, and model fitting are performed within the training and cross-validation workflow.

Dashboard pass rates and the support watchlist are based on observed historical `G3` values in the source cohort. They are descriptive summaries, not model predictions. Model scores are decision-support signals and should be reviewed by educators; they are not guaranteed to be calibrated probabilities.

## Early-classification experiment

The focused early-classification experiment evaluates the Math and Portuguese datasets separately. It compares existing classifiers with class-weighted Logistic Regression and Random Forest, Extra Trees, SVM, and suitable gradient-boosting models.

The experiment uses seeded stratified splits, five-fold stratified training cross-validation, and Optuna search to optimize training-CV macro F1. Candidate preprocessing, feature selection, and estimator-native class weighting are fitted inside the cross-validation pipelines. Final held-out results are reported separately from the training-CV model-selection process.

Run both datasets:

```powershell
.\venv\Scripts\python.exe run_early_classification.py
```

Run one dataset:

```powershell
.\venv\Scripts\python.exe run_early_classification.py --dataset student-mat.csv
.\venv\Scripts\python.exe run_early_classification.py --dataset student-por.csv
```

Each run writes uniquely timestamped JSON and CSV reports under `reports/`. The reports include per-model and per-seed metrics, per-class results, confusion matrices, prediction distributions, and experiment settings. The completed Math and Portuguese reports are available here:

- [Math experiment report (JSON)](./reports/early_classification_20261031-044633-681161.json)
- [Math experiment report (CSV)](./reports/early_classification_20261031-044633-681161.csv)
- [Portuguese experiment report (JSON)](./reports/early_classification_20261031-045431-786094.json)
- [Portuguese experiment report (CSV)](./reports/early_classification_20261031-045431-786094.csv)

## Evaluation and validation

Classification evaluation includes accuracy, balanced accuracy, macro precision, macro recall, macro F1, weighted F1, per-class reports, and confusion matrices. ROC-AUC is included when supported by the model output. Regression evaluation includes MAE, RMSE, and R², with actual-versus-predicted and residual visualizations.

The rigorous-validation workflow evaluates the bundled UCI Student Performance Math and Portuguese datasets across multiple seeds and prediction scenarios. It records dataset profiles, class distributions, model comparisons, and per-class evaluation results in timestamped JSON reports under `reports/`.

## Explainable AI

The dashboard offers global permutation feature importance and supported native model explanations, such as tree-based importance or linear coefficients. It also provides individual prediction explanations using changes in model outputs when input features are varied. Explanations use the fitted preprocessing and feature-selection pipeline.

## Run locally on Windows

From the project directory:

```powershell
.\venv\Scripts\activate
python -m streamlit run app.py
```

The dashboard opens in the browser with a bundled demonstration cohort. Upload a CSV from the sidebar to use your own data.

## Tests

Run the test suite:

```powershell
.\venv\Scripts\python.exe -m pytest -q
```

Compile the application and source modules:

```powershell
.\venv\Scripts\python.exe -m compileall -q app.py src
```
