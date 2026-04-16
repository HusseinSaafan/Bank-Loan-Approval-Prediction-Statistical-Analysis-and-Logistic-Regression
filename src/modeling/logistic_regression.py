import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from src.ingestion_preprocessing.feature_eng import run_feature_eng
from src.ingestion_preprocessing.data_encoding_splitting import run_data_encoding_splitting
from src.utils.config import logger


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
            scoring='accuracy',
            n_jobs=-1,
            refit=True,
        )
        grid_search.fit(X_train, y_train)

        logger.info(f"Best parameters found: {grid_search.best_params_}")
        logger.info(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        return grid_search
    except Exception as e:
        logger.error(f"Error during logistic regression grid search: {e}")
        return None


def run_logistic_regression():
    logger.info("Running logistic regression modeling pipeline.")

    # Step 1: Run feature engineering to produce the cleaned dataset
    cleaned_df = run_feature_eng()
    if cleaned_df is None:
        logger.error("Modeling pipeline failed: could not produce cleaned data.")
        return None

    # Step 2: Encode and split the cleaned data into train/test sets
    X_train, X_test, y_train, y_test = run_data_encoding_splitting(cleaned_df)
    if any(item is None for item in [X_train, X_test, y_train, y_test]):
        logger.error("Modeling pipeline failed: data encoding or splitting produced None.")
        return None

    # Step 3: Find suitable logistic regression hyperparameters with grid search.
    grid_search = tune_logistic_regression(X_train, y_train)
    if grid_search is None:
        logger.error("Modeling pipeline failed: grid search tuning did not complete.")
        return None

    # Step 4: Fit the statsmodels logistic regression for statistical interpretation.
    result = build_logistic_regression(X_train, y_train)
    return {
        'grid_search': grid_search,
        'best_model': grid_search.best_estimator_,
        'statsmodels_result': result,
    }




