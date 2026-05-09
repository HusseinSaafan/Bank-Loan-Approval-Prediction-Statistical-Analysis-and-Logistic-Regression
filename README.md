# Bank Loan Approval Prediction

An end-to-end machine learning project that analyzes key factors influencing loan approval decisions, trains and compares multiple classification models, and evaluates the best performer on held-out test data with full interpretability through SHAP values.

---

> ## Disclaimer
>
> This dataset represents a **sample of a broader population** and is used for skills demonstration purposes.
>
> This project is **not intended for real-world banking or production deployment**.
>
> The primary focus is demonstrating core data science competencies: statistical analysis, multi-model training and comparison, hyperparameter tuning, and business-oriented interpretation of results.

---

## Project Objective

- Analyze factors that influence loan approval decisions  
- Perform statistical hypothesis testing on applicant features  
- Train and compare three classification models: Logistic Regression, Random Forest, and XGBoost  
- Automatically select the best model based on cross-validation F1 score  
- Evaluate the best model on a held-out test set and interpret results  

---

## Dataset Overview

The dataset contains information about loan applicants and whether their loan was approved.

| Feature | Description |
|---|---|
| `Loan_ID` | Unique loan identifier |
| `Gender` | Applicant gender |
| `Married` | Marital status |
| `Dependents` | Number of dependents |
| `Education` | Education level |
| `Self_Employed` | Self-employment status |
| `ApplicantIncome` | Applicant income |
| `CoapplicantIncome` | Co-applicant income |
| `LoanAmount` | Requested loan amount |
| `Loan_Amount_Term` | Loan duration in months |
| `Credit_History` | Credit history record |
| `Property_Area` | Urban / Semiurban / Rural |
| `Loan_Status` | Target variable — Approved (1) / Rejected (0) |

---

## Project Structure

```
bank_loan_approval/
├── src/
│   ├── main.py                            # End-to-end pipeline entry point
│   ├── analysis/
│   │   ├── eda.py                         # Exploratory data analysis
│   │   └── feature_analysis.py            # Statistical hypothesis testing
│   ├── ingestion_preprocessing/
│   │   ├── feature_eng.py                 # Feature engineering
│   │   └── data_encoding_splitting.py     # Encoding and train/test split
│   ├── modeling/
│   │   ├── logistic_regression.py         # Logistic regression training
│   │   ├── random_forest.py               # Random forest training
│   │   └── xgboost.py                     # XGBoost training
│   ├── evaluation/
│   │   ├── best_model_selector.py         # Selects best model by CV F1
│   │   └── test_best_model.py             # Tests best model on held-out data
│   └── utils/
│       ├── config.py                      # Logger and global config
│       └── helpers.py                     # Shared utilities (CV, SHAP, GridSearch)
├── artifacts/
│   └── models/
│       ├── logistic_regression.pkl
│       ├── logistic_regression_f1_scores.json
│       ├── logistic_regression_cv_scores.html
│       ├── logistic_regression_shap_values.html
│       ├── random_forest.pkl
│       ├── random_forest_f1_scores.json
│       ├── random_forest_cv_scores.html
│       ├── random_forest_shap_values.html
│       ├── xgboost.pkl
│       ├── xgboost_f1_scores.json
│       ├── xgboost_cv_scores.html
│       ├── xgboost_shap_values.html
│       ├── best_model.pkl
│       ├── best_model_summary.json
│       └── best_model/
│           ├── test_metrics.json          # F1, ROC AUC, classification report
│           ├── confusion_matrix.json      # Confusion matrix with interpretation
│           ├── test_predictions.json      # Actual vs predicted labels
│           └── roc_curve.html             # Interactive ROC curve plot
├── database/
│   ├── Loan Approval Dataset.csv          # Raw dataset
│   ├── Loan Approval Dataset Cleaned.csv  # Cleaned dataset
│   ├── train_encoded.csv                  # Encoded training set
│   └── test_encoded.csv                   # Encoded test set
├── logs/                                  # Timestamped run logs
├── model_evaluation_results.md            # Detailed interpretation of test results
└── requirements.txt
```

---

## Pipeline Workflow

### 1. Data Cleaning
- Handled missing values and removed invalid entries
- Converted variables to appropriate types

### 2. Exploratory Data Analysis (EDA)
- Distribution analysis of loan status and key features
- Correlation analysis between features and target
- Interactive visualizations saved to `figures/eda/`

### 3. Statistical Hypothesis Testing
Applied statistical tests to validate feature-target relationships:

- **Shapiro-Wilk Test** — Normality of continuous features
- **Chi-Square Test** — Association between categorical features and loan status
- **T-Test** — Comparison of means across approval groups
- **Proportion Z-Test** — Approval rate differences across subgroups

### 4. Feature Engineering & Encoding
- Engineered new features from existing columns
- Applied `MinMaxScaler` on numerical features
- Applied `OneHotEncoder (drop='first')` on categorical features
- Saved encoded `train_encoded.csv` and `test_encoded.csv` to `database/`

### 5. Model Training

Three models were trained with `GridSearchCV` hyperparameter tuning and 5-fold stratified cross-validation:

| Model | Approach |
|---|---|
| **Logistic Regression** | `statsmodels` for interpretability + `sklearn` for tuning |
| **Random Forest** | Ensemble of decision trees with GridSearch |
| **XGBoost** | Gradient boosting with GridSearch |

For each model, the following are saved to `artifacts/models/`:
- Trained model (`.pkl`)
- Cross-validation F1 scores (`.json` + interactive `.html` chart)
- SHAP value plots (summary, dot, and dependence plots as `.html`)

### 6. Automatic Best Model Selection (`best_model_selector.py`)

- Loads the CV F1 scores from each model's JSON file
- Selects the model with the highest mean CV F1 score
- Saves the winning model as `best_model.pkl`
- Saves a comparison summary to `best_model_summary.json`

### 7. Best Model Testing on Held-Out Data (`test_best_model.py`)

- Loads `best_model.pkl` and `test_encoded.csv`
- Generates predictions and probability scores
- Calculates: **F1 Score**, **Confusion Matrix**, **ROC AUC**
- Saves results to `artifacts/models/best_model/`
- Generates an interactive ROC curve plot

---

## Results Summary

All three models achieved a cross-validation F1 score of **0.8792**. XGBoost was selected as the best model and evaluated on the 112-sample held-out test set:

| Metric | Value |
|---|---|
| Accuracy | 80.36% |
| F1 Score (Approved) | 0.8706 |
| ROC AUC | 0.7317 |

| | Predicted Rejected | Predicted Approved |
|---|---|---|
| **Actual Rejected** | 16 ✅ | 20 ❌ |
| **Actual Approved** | 2 ❌ | 74 ✅ |

The model achieves **97.4% recall on approvals** but only **44.4% recall on rejections**, reflecting a class imbalance bias toward approving loans.

See [model_evaluation_results.md](model_evaluation_results.md) for a full interpretation.

---

## How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run the full end-to-end pipeline
python -m src.main
```

This runs data loading, all three model training pipelines, automatic best model selection, and final test evaluation in sequence.

---

## Tech Stack

| Library | Purpose |
|---|---|
| `pandas`, `numpy` | Data manipulation |
| `scikit-learn` | Modeling, preprocessing, evaluation |
| `statsmodels` | Statistical logistic regression |
| `xgboost` | Gradient boosting model |
| `shap` | Model interpretability |
| `plotly` | Interactive visualizations |

---

## Skills Demonstrated

- Exploratory Data Analysis (EDA)
- Statistical hypothesis testing
- Feature engineering and preprocessing pipelines
- Multi-model training with hyperparameter tuning (GridSearchCV)
- Cross-validation and model comparison
- SHAP-based model interpretability
- Automated best model selection
- Held-out test evaluation (F1, Confusion Matrix, ROC AUC)
- Structured, modular Python project organization
- Structured logging across pipeline runs
