import json
import os

import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
import pickle
from src.utils.config import logger
from src.utils.helpers import compute_cv_scores, plot_cv_scores, plot_shap_values, run_grid_search

TRAIN_ENCODED_PATH = os.path.join('database', 'train_encoded.csv')
ARTIFACTS_MODELS_DIR = os.path.join('artifacts', 'models')


def build_logistic_regression(X_train, y_train):
    """
    Builds a logistic regression model using statsmodels for interpretability.
    Adds a constant term to the training data to include an intercept in the model.
    """
    logger.info("Building logistic regression model using statsmodels.")
    try:
        X_train_const = sm.add_constant(X_train)
        model = sm.Logit(y_train, X_train_const)
        result = model.fit()
        logger.info("Logistic regression model trained successfully.")
        logger.info(f"\n{result.summary()}")
        return result
    except Exception as e:
        logger.error(f"Error building logistic regression model: {e}")
        return None


def tune_logistic_regression(X_train, y_train):
    logger.info("Running GridSearchCV for logistic regression hyperparameter tuning.")
    try:
        param_grid = [
            {
                'C': [0.01, 0.1, 1, 10, 100],
                'solver': ['liblinear'],
                'penalty': ['l1', 'l2'],
                'max_iter': [500, 1000],
            },
            {
                'C': [0.01, 0.1, 1, 10, 100],
                'solver': ['lbfgs'],
                'penalty': ['l2'],
                'max_iter': [500, 1000],
            },
        ]

        base_model = LogisticRegression(random_state=42)
        grid_search = run_grid_search(base_model, param_grid, X_train, y_train)
        if grid_search is None:
            return None

        fold_scores, fold_labels = compute_cv_scores(grid_search.best_estimator_, X_train, y_train)
        plot_cv_scores(
            fold_labels, fold_scores,
            title='Logistic Regression — Cross-Validation F1 Score per Fold',
            color='steelblue',
            plot_path=os.path.join(ARTIFACTS_MODELS_DIR, 'logistic_regression_cv_scores.html'),
        )
        return grid_search
    except Exception as e:
        logger.error(f"Error during logistic regression grid search: {e}")
        return None


def run_logistic_regression():
    logger.info("Running logistic regression modeling pipeline.")

    # Step 1: Load pre-encoded training data from train_encoded.csv
    try:
        train_df = pd.read_csv(TRAIN_ENCODED_PATH)
        logger.info(f"Loaded training data from {TRAIN_ENCODED_PATH}: {train_df.shape}")
    except Exception as e:
        logger.error(f"Failed to load training data: {e}")
        return None

    target_col = 'Loan_Status'
    if target_col not in train_df.columns:
        logger.error(f"Target column '{target_col}' not found in training data.")
        return None

    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]

    # Step 2: Find suitable logistic regression hyperparameters with grid search.
    grid_search = tune_logistic_regression(X_train, y_train)
    if grid_search is None:
        logger.error("Modeling pipeline failed: grid search tuning did not complete.")
        return None

    plot_shap_values(
        estimator=grid_search.best_estimator_,
        X=X_train,
        title='Logistic Regression — Mean Absolute SHAP Values',
        plot_path=os.path.join(ARTIFACTS_MODELS_DIR, 'logistic_regression_shap_values.html'),
    )

    # Step 3: Fit the statsmodels logistic regression for statistical interpretation.
    result = build_logistic_regression(X_train, y_train)
    save_logistic_regression_model(grid_search.best_estimator_)

    # Step 4: Save best model F1 score to JSON.
    fold_scores, _ = compute_cv_scores(grid_search.best_estimator_, X_train, y_train)
    if fold_scores is not None:
        scores_dict = {
            'model': 'logistic_regression',
            'best_params': grid_search.best_params_,
            'best_cv_f1': round(float(grid_search.best_score_), 6),
            'cv_f1_per_fold': [round(float(s), 6) for s in fold_scores],
            'cv_f1_mean': round(float(fold_scores.mean()), 6),
            'cv_f1_std': round(float(fold_scores.std()), 6),
        }
        scores_path = os.path.join(ARTIFACTS_MODELS_DIR, 'logistic_regression_f1_scores.json')
        try:
            os.makedirs(ARTIFACTS_MODELS_DIR, exist_ok=True)
            with open(scores_path, 'w') as f:
                json.dump(scores_dict, f, indent=4)
            logger.info(f"Logistic regression F1 scores saved to: {scores_path}")
        except Exception as e:
            logger.error(f"Error saving F1 scores: {e}")

    return {
        'grid_search': grid_search,
        'best_model': grid_search.best_estimator_,
        'statsmodels_result': result,
    }

def save_logistic_regression_model(model, file_name='logistic_regression.pkl'):
    if model is None:
        logger.error("No logistic regression model available to save.")
        return

    file_name = os.path.basename(file_name)
    if not file_name.endswith('.pkl'):
        file_name = f"{file_name}.pkl"
    output_path = os.path.join(ARTIFACTS_MODELS_DIR, file_name)

    logger.info(f"Saving logistic regression model to: {output_path}")
    try:
        os.makedirs(ARTIFACTS_MODELS_DIR, exist_ok=True)
        with open(output_path, 'wb') as f:
            pickle.dump(model, f)
        logger.info("Model saved successfully.")
    except Exception as e:
        logger.error(f"Error saving logistic regression model: {e}")

# run_logistic_regression()



