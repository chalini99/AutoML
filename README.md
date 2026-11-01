
# AutoML-Based Student Academic Performance Prediction

### Automated Feature Selection | Hyperparameter Optimization | Explainable AI

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Machine Learning](<https://img.shields.io/badge/Machine%20Learning-Scikit--learn-orange>)
![AutoML](https://img.shields.io/badge/AutoML-Optuna-purple)
![Dashboard](https://img.shields.io/badge/Dashboard-Streamlit-red)
![Testing](<https://img.shields.io/badge/Tests-51%20Passed-success>)

## 1. Project Overview

The **AutoML-Based Student Academic Performance Prediction System** is an interactive machine-learning platform designed to analyse student academic data and predict academic outcomes using automated machine-learning techniques.

The project integrates data preprocessing, automated feature selection, hyperparameter optimization, machine-learning model comparison, performance evaluation and Explainable Artificial Intelligence (XAI) into a unified application.

The system uses historical student information, including demographic, academic, family and social characteristics, to identify patterns associated with academic performance.

It supports two prediction scenarios:

- **Early Prediction:** Predicts academic outcomes using non-grade-related information, without using G1, G2 or G3 as input features.
- **Grade-Assisted Prediction:** Uses previous examination grades G1 and G2 to predict the final grade G3.

The application provides an interactive dashboard developed using Streamlit, allowing users to explore datasets, configure experiments, train models, evaluate performance and understand model predictions.

## 2. Problem Statement

Educational institutions collect large amounts of student information, including academic records, attendance, family background, study habits and social characteristics.

However, analysing this information manually and identifying students who may experience academic difficulties can be challenging.

Traditional machine-learning development also requires several manual steps, including data preprocessing, feature selection, algorithm selection, hyperparameter tuning and model evaluation.

There is a need for an integrated system that can automate important stages of machine-learning development, predict student academic outcomes and provide explanations for its predictions.

This project addresses the problem by developing an AutoML-based student academic-performance prediction platform with an interactive dashboard and Explainable AI capabilities.

## 3. Project Objectives

The main objectives are:

1. Analyse student academic datasets and identify meaningful patterns.
2. Implement automated data preprocessing for numerical and categorical variables.
3. Apply feature-selection techniques to identify relevant input variables.
4. Automate hyperparameter optimization using Optuna.
5. Train and compare machine-learning models using systematic evaluation.
6. Evaluate models using appropriate classification and regression metrics.
7. Implement Explainable AI techniques to understand model behaviour.
8. Develop an interactive Streamlit dashboard for dataset exploration, model training, evaluation and predictions.
9. Maintain reproducible experiments through controlled evaluation and saved experiment information.

## 4. Dataset Description

The project uses the **Student Performance dataset from the UCI Machine Learning Repository**.

Dataset source:

https://archive.ics.uci.edu/dataset/320/student+performance

Two datasets are used:

| Dataset                           | Records | Features / Columns |
| --------------------------------- | ------: | -----------------: |
| Mathematics (`student-mat.csv`) |     395 |                 33 |
| Portuguese (`student-por.csv`)  |     649 |                 33 |

### Dataset characteristics

- Structured tabular student information.
- Demographic and family-related attributes.
- Academic performance information.
- Study habits and lifestyle-related variables.
- Previous academic failures and absences.
- Final academic grade information.

The datasets use semicolon-separated CSV formatting.

The reported dataset validation found no missing values or duplicate rows in either original dataset.

### Important dataset attributes

| Feature   | Description                             |
| --------- | --------------------------------------- |
| Age       | Student age                             |
| Sex       | Student gender attribute in the dataset |
| Address   | Urban or rural residence                |
| Studytime | Weekly study-time category              |
| Failures  | Previous class failures                 |
| Absences  | Number of school absences               |
| Schoolsup | Extra educational support               |
| Famsup    | Family educational support              |
| Internet  | Internet access                         |
| G1        | First-period grade                      |
| G2        | Second-period grade                     |
| G3        | Final-period grade                      |

The dataset contains 33 columns, including G1, G2 and G3.

### Target Variable

The final grade G3 is used to derive a binary classification target.

The target is defined as:

- Pass: G3 >= 10
- Fail: G3 < 10

Mathematically:

\[
y =
\begin{cases}
1 & \text{if } G3 \geq 10\\
0 & \text{if } G3 < 10
\end{cases}
\]

The system also supports regression experiments where the target is a numerical grade.

## 5. Prediction Scenarios

The project supports two prediction scenarios.

### 5.1 Early Prediction

Early prediction aims to estimate student academic outcomes using information available before previous examination grades are known.

Input features exclude:

- G1
- G2
- G3

The model uses the remaining 30 original dataset variables.

This scenario is particularly relevant to identifying students who may require academic support at an earlier stage.

### 5.2 Grade-Assisted Prediction

Grade-assisted prediction uses previous academic grades as additional input information.

- G1: First-period grade.
- G2: Second-period grade.
- G3: Final-period grade and prediction target.

G1 and G2 are included as input features, while G3 is excluded from the input matrix.

This scenario evaluates prediction when previous examination results are available.

**Important:** Grade-assisted prediction is not equivalent to early prediction because it uses additional academic information.

## 6. Proposed System

The proposed platform integrates the following capabilities:

- Dataset loading and validation.
- Data preprocessing.
- Prediction-task configuration.
- Early and grade-assisted prediction scenarios.
- Feature selection.
- Hyperparameter optimization.
- Machine-learning model training.
- Cross-validation.
- Model comparison.
- Held-out test evaluation.
- Explainable AI.
- Individual and cohort-level prediction interfaces.
- Experiment persistence and validation reporting.

## 7. System Architecture

The system follows an integrated machine-learning workflow.

```text
                 INPUT DATASET
                       |
                       v
             DATA LOADING & VALIDATION
                       |
                       v
                DATA PREPROCESSING
                       |
                       v
             EXPERIMENT CONFIGURATION
                       |
                       v
              TRAIN / TEST SEPARATION
                       |
                       v
             CROSS-VALIDATION PIPELINE
                       |
             +---------+---------+
             |                   |
             v                   v
       FEATURE SELECTION    OPTUNA SEARCH
             |                   |
             +---------+---------+
                       |
                       v
               MODEL TRAINING
                       |
                       v
              MODEL COMPARISON
                       |
                       v
              MODEL SELECTION
                       |
                       v
             HELD-OUT EVALUATION
                       |
             +---------+---------+
             |                   |
             v                   v
         PREDICTIONS       EXPLAINABLE AI
             |                   |
             +---------+---------+
                       |
                       v
               STREAMLIT DASHBOARD
```

## 8. Methodology

### Step 1: Data Loading

The application loads the student datasets and validates their structure.

The user can work with the bundled demonstration dataset or upload a compatible CSV file.

### Step 2: Data Preprocessing

Preprocessing prepares the dataset for machine-learning algorithms.

Depending on the selected experiment, this includes:

- Numerical and categorical feature handling.
- Categorical encoding.
- Numerical preprocessing.
- Target preparation.
- Data validation.

Preprocessing is incorporated into the model-evaluation pipeline to reduce data leakage.

### Step 3: Experimental Design

The system configures:

- Prediction target.
- Classification or regression task.
- Early or grade-assisted scenario.
- Train/test separation.
- Cross-validation strategy.
- Random seed.
- Evaluation metric.

The test set is kept separate from model selection.

### Step 4: Automated Feature Selection

Feature selection identifies useful input variables and can reduce unnecessary features.

The platform provides feature-selection configuration options.

Feature selection is performed within the training and validation workflow to avoid using held-out test information during model development.

### Step 5: Hyperparameter Optimization

Optuna is used to search for suitable hyperparameter configurations.

The optimization process evaluates candidate configurations using a defined objective function.

For the reported early-classification experiments, Macro F1 is used as the model-selection metric.

### Step 6: Model Training and Comparison

Candidate machine-learning models are trained and evaluated.

The system compares their performance using the configured evaluation strategy.

### Step 7: Model Evaluation

The selected model is evaluated using the separate held-out test set.

Evaluation includes relevant classification or regression metrics.

### Step 8: Explainable AI

Explainability methods are used to understand model behaviour.

The implementation includes feature-importance analysis, permutation importance and local perturbation-based explanations.

### Step 9: Dashboard Presentation

The application displays dataset information, model configurations, training results, evaluation metrics, predictions and explanations through a Streamlit interface.

## 9. Machine Learning Algorithms

The project uses supervised machine-learning approaches for classification and regression.

Important model families and baselines include:

### 9.1 Logistic Regression

Logistic Regression is a supervised classification algorithm that estimates the probability of an observation belonging to a particular class.

It is useful as a classification baseline and provides a relatively interpretable model.

### 9.2 Decision Tree

A Decision Tree learns a series of decision rules by splitting the dataset according to feature values.

It can model nonlinear relationships and is relatively easy to interpret.

### 9.3 Random Forest

Random Forest is an ensemble-learning algorithm that combines multiple decision trees.

For classification, the predictions of individual trees are combined to generate the final class prediction.

Advantages:

- Captures nonlinear relationships.
- Models interactions between features.
- Reduces dependence on a single decision tree.
- Supports feature-importance analysis.
- Can use class weighting to address class imbalance.

### 9.4 Linear Regression

Linear Regression is used for numerical prediction tasks.

It models relationships between input variables and a continuous target variable.

### 9.5 Baseline Models

Baseline models provide a simple reference point for evaluating the performance of more complex machine-learning models.

Comparing against baselines helps determine whether the trained models provide meaningful improvements.

## 10. Automated Machine Learning (AutoML)

AutoML stands for Automated Machine Learning.

It automates important stages of the machine-learning development process, including:

- Preprocessing configuration.
- Feature selection.
- Hyperparameter optimization.
- Model training.
- Model comparison.
- Performance evaluation.

The objective is to reduce repetitive manual experimentation and create a systematic model-development workflow.

The user still defines the prediction objective, evaluation strategy and experiment settings.

## 11. Hyperparameter Optimization Using Optuna

Optuna is a Python framework for hyperparameter optimization.

Official documentation:

https://optuna.org/

Hyperparameters are configuration values that control how a machine-learning algorithm learns.

Examples include:

| Algorithm           | Example Hyperparameters                                   |
| ------------------- | --------------------------------------------------------- |
| Random Forest       | Number of trees, maximum depth, minimum samples per split |
| Logistic Regression | Regularization strength                                   |
| Decision Tree       | Maximum depth, minimum samples per split                  |

The optimization workflow is:

1. Define an objective function.
2. Define the hyperparameter search space.
3. Generate a candidate configuration.
4. Train and evaluate the candidate using cross-validation.
5. Record the objective value.
6. Repeat for the configured trial budget.
7. Select the best-performing configuration according to the objective metric.

For the reported early-classification experiments, cross-validation Macro F1 is the optimization and model-selection criterion.

## 12. Feature Selection

Feature selection identifies the most relevant input variables for a prediction task.

Benefits include:

- Reducing unnecessary features.
- Improving model interpretability.
- Potentially reducing training complexity.
- Reducing the effect of irrelevant variables.

The system supports configurable feature-selection strategies.

The selected strategy and its impact should be evaluated experimentally rather than assuming feature selection always improves performance.

## 13. Model Evaluation Metrics

The project uses multiple metrics to evaluate classification performance.

### Accuracy

Measures the proportion of correctly classified observations.

\[
Accuracy = \frac{TP+TN}{TP+TN+FP+FN}
\]

### Precision

Measures the proportion of predicted positive cases that are actually positive.

\[
Precision = \frac{TP}{TP+FP}
\]

### Recall

Measures the proportion of actual positive cases correctly identified.

\[
Recall = \frac{TP}{TP+FN}
\]

### F1-score

The harmonic mean of precision and recall.

\[
F1 = 2 \times \frac{Precision \times Recall}{Precision+Recall}
\]

### Macro F1

Calculates F1 separately for each class and takes the unweighted average.

This is important when the target classes are imbalanced.

### Balanced Accuracy

Calculates the average recall across classes.

### Confusion Matrix

A confusion matrix presents the number of correct and incorrect predictions for each class.

For regression experiments, metrics such as Mean Absolute Error (MAE), Root Mean Squared Error (RMSE) and R-squared are used where applicable.

## 14. Experimental Results

The latest reported early-classification experiments were conducted using three random seeds:

- 42
- 43
- 44

The selected model for both cohorts was class-weighted Random Forest.

### Early Prediction — Mathematics Dataset

| Metric            |                       Result |
| ----------------- | ---------------------------: |
| Selected Model    | Class-weighted Random Forest |
| Mean CV Macro F1  |             0.6589 ± 0.0407 |
| Test Accuracy     |             0.6498 ± 0.0824 |
| Test Macro F1     |             0.5858 ± 0.1237 |
| Balanced Accuracy |             0.5920 ± 0.1255 |
| Fail-class F1     |                       0.4252 |

### Early Prediction — Portuguese Dataset

| Metric            |                       Result |
| ----------------- | ---------------------------: |
| Selected Model    | Class-weighted Random Forest |
| Mean CV Macro F1  |             0.7318 ± 0.0385 |
| Test Accuracy     |             0.7897 ± 0.0311 |
| Test Macro F1     |             0.6267 ± 0.0768 |
| Balanced Accuracy |             0.6508 ± 0.1166 |
| Fail-class F1     |                       0.3808 |

### Result Interpretation

The Portuguese cohort achieved higher overall test accuracy than the Mathematics cohort in the reported experiments.

However, the Fail-class F1 scores were relatively low in both datasets.

This indicates that identifying students belonging to the minority Fail class remains challenging.

Therefore, accuracy alone is insufficient to describe model performance. Macro F1 and balanced accuracy are also important.

The reported values represent the results of the specific datasets, prediction scenario, model-selection process and experimental seeds. They should not be interpreted as guaranteed performance on other institutions' data.

## 15. Explainable Artificial Intelligence (XAI)

Explainable AI helps users understand the behaviour of machine-learning models.

The project incorporates explainability to make model predictions more transparent.

### 15.1 Feature Importance

Feature importance helps identify variables that contribute to model predictions.

It provides an overview of which features the trained model considers useful.

### 15.2 Permutation Importance

Permutation importance evaluates the importance of a feature by randomly shuffling its values and measuring the resulting change in model performance.

A larger performance decrease after shuffling suggests that the model relies more strongly on that feature.

### 15.3 Local Perturbation Explanations

Local perturbation explanations investigate individual predictions by modifying input features and observing how the model output changes.

This helps users inspect how individual feature values influence model behaviour.

### Importance of XAI

- Improves model transparency.
- Helps users understand predictions.
- Supports model inspection.
- Helps identify potentially unexpected model behaviour.
- Provides additional information for human decision-making.

Feature importance and model explanations indicate learned relationships or model reliance. They do not establish that a particular feature causes a student's academic outcome.

## 16. Application Dashboard

The project includes an interactive Streamlit dashboard.

The main workspaces include:

| Workspace                   | Purpose                                         |
| --------------------------- | ----------------------------------------------- |
| Project Details             | Provides project information and context        |
| Student Success Dashboard   | Displays cohort-level performance information   |
| Student Predictions         | Supports student-level prediction workflows     |
| Dataset Overview            | Explores dataset structure and distributions    |
| Model Studio                | Configures prediction experiments               |
| Model Training & Comparison | Trains and compares candidate models            |
| Evaluation Results          | Presents model evaluation metrics               |
| Explainable AI              | Displays model explanations                     |
| Saved Experiments           | Provides access to saved experiment information |
| Rigorous Validation         | Presents validation and robustness reports      |

The application supports a bundled demonstration dataset and CSV uploads.

It also provides dataset information, model configuration and interactive analysis.

## 17. Technology Stack

| Technology   | Purpose                                             |
| ------------ | --------------------------------------------------- |
| Python       | Primary programming language                        |
| Pandas       | Data manipulation and analysis                      |
| NumPy        | Numerical computation                               |
| Scikit-learn | Machine-learning algorithms and evaluation          |
| Optuna       | Hyperparameter optimization                         |
| Streamlit    | Interactive dashboard                               |
| Joblib       | Saving and loading supported Python model artifacts |
| Pytest       | Automated testing                                   |

## 18. Project Structure

The main project structure is organised as follows:

```text
StudentAutoML/
│
├── app.py
├── README.md
├── .gitignore
│
├── app/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── splits/
│
├── models/
│
├── notebooks/
│
├── reports/
│
├── src/
│   ├── early_classification.py
│   ├── xai.py
│   └── ...
│
├── tests/
│
├── check_dataset.py
├── compare_automl.py
├── run_automl.py
├── run_baselines.py
├── run_early_baselines.py
├── run_early_classification.py
├── run_phase6_validation.py
└── run_regression_baselines.py
```

The structure separates the application interface, source code, datasets, experiments, reports, models and tests.

Some folders may contain additional modules and artifacts as development progresses.

## 19. Installation and Setup

### Prerequisites

- Python 3.10 or later.
- pip.
- Git.
- Windows, Linux or macOS.

### Step 1: Clone the repository

```bash
git clone https://github.com/chalini99/AutoML.git
```

### Step 2: Navigate into the project folder

```bash
cd AutoML
```

### Step 3: Create a virtual environment

Windows:

```powershell
python -m venv venv
```

Linux / macOS:

```bash
python3 -m venv venv
```

### Step 4: Activate the environment

Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

Windows Command Prompt:

```cmd
venv\Scripts\activate.bat
```

Linux / macOS:

```bash
source venv/bin/activate
```

### Step 5: Install dependencies

Install the project's required dependencies using its dependency file if one is provided.

For example:

```bash
pip install -r requirements.txt
```

If the repository does not yet contain a complete `requirements.txt`, install the dependencies listed in the project's Python imports and package configuration before running the application.

## 20. Running the Application

Start the Streamlit dashboard:

```bash
python -m streamlit run app.py
```

Open the following address in your browser:

```text
http://localhost:8501
```

The dashboard should load locally.

To stop the application, press:

```text
Ctrl + C
```

## 21. Testing

The project uses Pytest for automated testing.

Run the test suite:

```bash
python -m pytest -q
```

The last completed full test run reported:

```text
51 passed
```

Additional application validation included:

- Streamlit AppTest navigation checks.
- Application startup verification.
- Health endpoint verification.
- Python compilation checks.

The last reported health endpoint check returned HTTP 200.

These tests validate application functionality and do not independently establish real-world predictive accuracy.

## 22. Data Leakage Prevention and Experimental Reliability

Data leakage prevention is important in academic-performance prediction.

The project addresses leakage through the following practices:

1. G1, G2 and G3 are excluded from early-prediction input features.
2. G3 is used only as the target in the classification experiment.
3. Grade-assisted prediction is treated as a separate scenario.
4. Preprocessing, feature selection and model fitting are performed within the cross-validation workflow.
5. The held-out test set is kept separate from model selection.
6. Experiments are evaluated across multiple random seeds.

The project also includes dataset-overlap assessment and validation reporting.

These measures improve the reliability of the reported experiments, although external validation remains necessary before real-world deployment.

## 23. Limitations

The current implementation has several limitations:

1. The UCI datasets are relatively small compared with large institutional student databases.
2. The target classes are imbalanced.
3. Fail-class prediction performance requires further improvement.
4. Model performance may vary across educational institutions.
5. Historical data may not represent current student populations.
6. Feature importance does not establish causation.
7. External validation on independent institutional datasets has not been established.
8. Predictions should not be used as the sole basis for academic decisions.

## 24. Future Enhancements

Potential future improvements include:

- Integrating larger datasets from multiple educational institutions.
- Improving minority-class prediction.
- Exploring additional machine-learning and ensemble algorithms.
- Performing external validation.
- Adding fairness and bias evaluation.
- Improving individual prediction explanations.
- Adding probability calibration and uncertainty analysis.
- Implementing secure institutional deployment.
- Monitoring changes in student data distributions over time.
- Supporting additional academic datasets and prediction objectives.

## 25. Ethical Considerations

Student academic data can contain sensitive personal information.

Therefore, a practical deployment should include:

- Appropriate access controls.
- Secure storage and processing.
- Data minimisation.
- Institutional permission for data use.
- Fairness assessment.
- Human oversight of academic decisions.

The platform is intended to support educators and academic counsellors. It should not replace professional judgement or be used to permanently label students based on model predictions.

## 26. Conclusion

The AutoML-Based Student Academic Performance Prediction System demonstrates an integrated approach to automated machine-learning development in the educational domain.

The platform combines data preprocessing, feature selection, hyperparameter optimization, model comparison, evaluation and Explainable AI within an interactive dashboard.

By supporting early and grade-assisted prediction scenarios, the system provides a structured way to investigate student academic outcomes under different information-availability conditions.

The experiments also highlight the challenges of predicting minority-class outcomes using relatively small educational datasets.

Overall, the project provides a foundation for data-driven academic analysis and can be extended through larger datasets, improved model evaluation and additional explainability and fairness techniques.

## 27. References

1. Cortez, P., and Silva, A. M. G. (2008). *Using Data Mining to Predict Secondary School Student Performance.* Proceedings of the 5th Annual Future Business Technology Conference.
2. UCI Machine Learning Repository. *Student Performance Dataset.*

   https://archive.ics.uci.edu/dataset/320/student+performance
3. Scikit-learn Documentation.

   https://scikit-learn.org/stable/
4. Optuna Documentation.

   https://optuna.org/
5. Streamlit Documentation.

   https://docs.streamlit.io/
6. Pandas Documentation.

   https://pandas.pydata.org/docs/
7. Pytest Documentation.

   https://docs.pytest.org/

---

## Project Information

**Project Name:** AutoML-Based Student Academic Performance Prediction

**GitHub Repository:** https://github.com/chalini99/AutoML

**Domain:** Artificial Intelligence and Machine Learning

**Core Technologies:** Python, Scikit-learn, Optuna, Streamlit and Explainable AI.

**Purpose:** Automated student academic-performance analysis and prediction.

---

*This project is an academic machine-learning prototype. Model outputs are estimates and should be interpreted with appropriate validation and human oversight.*
