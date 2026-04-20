import os

import pandas as pd
import plotly.graph_objects as go
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score

from src.utils.config import logger

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
        fold_labels = [f'Fold {i + 1}' for i in range(len(fold_scores))]
        logger.info(f"Per-fold F1 scores: {dict(zip(fold_labels, fold_scores.round(4)))}")
        logger.info(f"Mean F1: {fold_scores.mean():.4f} | Std: {fold_scores.std():.4f}")

        os.makedirs(ARTIFACTS_MODELS_DIR, exist_ok=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=fold_labels, y=fold_scores,
            mode='lines+markers', name='F1 Score',
            line=dict(color='steelblue', width=2),
            marker=dict(size=8),
        ))
        fig.add_hline(
            y=fold_scores.mean(),
            line=dict(color='tomato', dash='dash', width=1.5),
            annotation_text=f'Mean F1 = {fold_scores.mean():.4f}',
            annotation_position='top right',
        )
        fig.update_layout(
            title='Logistic Regression — Cross-Validation F1 Score per Fold',
            xaxis_title='Fold',
            yaxis_title='F1 Score',
            yaxis=dict(range=[0, 1]),
            legend=dict(orientation='h'),
        )
        plot_path = os.path.join(ARTIFACTS_MODELS_DIR, 'logistic_regression_cv_scores.html')
        fig.write_html(plot_path)
        logger.info(f"Cross-validation fold scores plot saved to: {plot_path}")

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

    # Step 3: Fit the statsmodels logistic regression for statistical interpretation.
    result = build_logistic_regression(X_train, y_train)
    return {
        'grid_search': grid_search,
        'best_model': grid_search.best_estimator_,
        'statsmodels_result': result,
    }

run_logistic_regression()



