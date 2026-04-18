from sklearn.model_selection import GridSearchCV, StratifiedKFold
from xgboost import XGBClassifier

from src.ingestion_preprocessing.data_encoding_splitting import run_data_encoding_splitting
from src.ingestion_preprocessing.feature_eng import run_feature_eng
from src.utils.config import logger


def build_xgboost(X_train, y_train, best_params=None):
	logger.info("Building XGBoost classifier.")
	try:
		default_params = {
			'n_estimators': 200,
			'max_depth': 5,
			'learning_rate': 0.1,
			'subsample': 1.0,
			'colsample_bytree': 1.0,
			'objective': 'binary:logistic',
			'eval_metric': 'logloss',
			'random_state': 42,
		}

		if best_params is not None:
			default_params.update(best_params)

		model = XGBClassifier(**default_params)
		model.fit(X_train, y_train)
		logger.info("XGBoost model trained successfully.")
		return model
	except Exception as e:
		logger.error(f"Error building XGBoost model: {e}")
		return None


def tune_xgboost(X_train, y_train):
	logger.info("Running stratified GridSearchCV for XGBoost hyperparameter tuning.")
	try:
		stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

		param_grid = {
			'n_estimators': [100, 200, 300],
			'max_depth': [3, 5, 7],
			'learning_rate': [0.01, 0.1],
			'subsample': [0.8, 1.0],
			'colsample_bytree': [0.8, 1.0],
			'min_child_weight': [1, 3, 5],
		}

		base_model = XGBClassifier(
			objective='binary:logistic',
			eval_metric='logloss',
			random_state=42,
		)

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
		logger.error(f"Error during XGBoost grid search: {e}")
		return None


def run_xgboost():
	logger.info("Running XGBoost modeling pipeline.")

	cleaned_df = run_feature_eng()
	if cleaned_df is None:
		logger.error("Modeling pipeline failed: could not produce cleaned data.")
		return None

	X_train, X_test, y_train, y_test = run_data_encoding_splitting(cleaned_df)
	if any(item is None for item in [X_train, X_test, y_train, y_test]):
		logger.error("Modeling pipeline failed: data encoding or splitting produced None.")
		return None

	grid_search = tune_xgboost(X_train, y_train)
	if grid_search is None:
		logger.error("Modeling pipeline failed: grid search tuning did not complete.")
		return None

	model = build_xgboost(X_train, y_train, best_params=grid_search.best_params_)
	return {
		'grid_search': grid_search,
		'best_model': model,
		'best_params': grid_search.best_params_,
		'best_cv_score': grid_search.best_score_,
	}


