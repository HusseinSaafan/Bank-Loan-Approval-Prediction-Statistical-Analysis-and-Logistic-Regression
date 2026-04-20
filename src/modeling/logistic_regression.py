import os

import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score

from src.utils.config import logger

TRAIN_ENCODED_PATH = os.path.join('database', 'train_encoded.csv')
ARTIFACTS_DIR = 'artifacts'


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
        stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

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
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=stratified_kfold,
            scoring='f1',
            n_jobs=-1,
            refit=True,
        )
        grid_search.fit(X_train, y_train)

        logger.info(f"Best parameters found: {grid_search.best_params_}")
        logger.info(f"Best cross-validation F1 score: {grid_search.best_score_:.4f}")

        # Compute per-fold F1 scores using the best estimator
        fold_scores = cross_val_score(
            grid_search.best_estimator_,
            X_train,
            y_train,
            cv=stratified_kfold,
            scoring='f1',
        )
        fold_df = pd.DataFrame({
            'fold': [f'fold_{i + 1}' for i in range(len(fold_scores))],
            'f1_score': fold_scores,
        })
        fold_df.loc[len(fold_df)] = ['mean', fold_scores.mean()]
        fold_df.loc[len(fold_df)] = ['std', fold_scores.std()]

        logger.info(f"Per-fold F1 scores:\n{fold_df.to_string(index=False)}")

        os.makedirs(ARTIFACTS_DIR, exist_ok=True)
        cv_output_path = os.path.join(ARTIFACTS_DIR, 'logistic_regression_cv_scores.csv')
        fold_df.to_csv(cv_output_path, index=False)
        logger.info(f"Cross-validation fold scores saved to: {cv_output_path}")

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

    # Drop columns where all values are NaN (no usable data)
    all_nan_cols = X_train.columns[X_train.isnull().all()].tolist()
    if all_nan_cols:
        X_train = X_train.drop(columns=all_nan_cols)
        logger.warning(f"Dropped fully-NaN columns: {all_nan_cols}")

    # Drop rows that still have any NaN
    before = len(X_train)
    mask = X_train.notna().all(axis=1) & y_train.notna()
    X_train = X_train[mask]
    y_train = y_train[mask]
    logger.info(f"Dropped {before - len(X_train)} rows with NaN. Remaining: {len(X_train)}")

    # Step 2: Find suitable logistic regression hyperparameters with grid search.
    grid_search = tune_logistic_regression(X_train, y_train)
    if grid_search is None:
        logger.error("Modeling pipeline failed: grid search tuning did not complete.")
        return None

    # Step 3: Fit the statsmodels logistic regression for statistical interpretation.
    result = build_logistic_regression(X_train, y_train)
    return {
        'grid_search': grid_search,
        'best_model': grid_search.best_estimator_,
        'statsmodels_result': result,
    }

run_logistic_regression()



