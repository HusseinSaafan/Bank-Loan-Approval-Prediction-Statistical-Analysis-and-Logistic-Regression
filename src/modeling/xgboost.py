import json
import os
import pickle

import pandas as pd
from xgboost import XGBClassifier

from src.utils.config import logger
from src.utils.helpers import compute_cv_scores, plot_cv_scores, plot_shap_values, run_grid_search

TRAIN_ENCODED_PATH = os.path.join('database', 'train_encoded.csv')
ARTIFACTS_MODELS_DIR = os.path.join('artifacts', 'models')


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

		grid_search = run_grid_search(base_model, param_grid, X_train, y_train)
		if grid_search is None:
			return None

		fold_scores, fold_labels = compute_cv_scores(grid_search.best_estimator_, X_train, y_train)
		plot_cv_scores(
			fold_labels, fold_scores,
			title='XGBoost — Cross-Validation F1 Score per Fold',
			color='darkorange',
			plot_path=os.path.join(ARTIFACTS_MODELS_DIR, 'xgboost_cv_scores.html'),
		)
		return grid_search
	except Exception as e:
		logger.error(f"Error during XGBoost grid search: {e}")
		return None


def run_xgboost():
	logger.info("Running XGBoost modeling pipeline.")

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

	grid_search = tune_xgboost(X_train, y_train)
	if grid_search is None:
		logger.error("Modeling pipeline failed: grid search tuning did not complete.")
		return None
	plot_shap_values(
		estimator=grid_search.best_estimator_,
		X=X_train,
		title='XGBoost — Mean Absolute SHAP Values',
		plot_path=os.path.join(ARTIFACTS_MODELS_DIR, 'xgboost_shap_values.html'),
	)

	model = build_xgboost(X_train, y_train, best_params=grid_search.best_params_)
	save_xgboost_model(model)

	# Save best model F1 score to JSON.
	fold_scores, _ = compute_cv_scores(grid_search.best_estimator_, X_train, y_train)
	if fold_scores is not None:
		scores_dict = {
			'model': 'xgboost',
			'best_params': grid_search.best_params_,
			'best_cv_f1': round(float(grid_search.best_score_), 6),
			'cv_f1_per_fold': [round(float(s), 6) for s in fold_scores],
			'cv_f1_mean': round(float(fold_scores.mean()), 6),
			'cv_f1_std': round(float(fold_scores.std()), 6),
		}
		scores_path = os.path.join(ARTIFACTS_MODELS_DIR, 'xgboost_f1_scores.json')
		try:
			os.makedirs(ARTIFACTS_MODELS_DIR, exist_ok=True)
			with open(scores_path, 'w') as f:
				json.dump(scores_dict, f, indent=4)
			logger.info(f"XGBoost F1 scores saved to: {scores_path}")
		except Exception as e:
			logger.error(f"Error saving F1 scores: {e}")

	return {
		'grid_search': grid_search,
		'best_model': model,
		'best_params': grid_search.best_params_,
		'best_cv_score': grid_search.best_score_,
	}


def save_xgboost_model(model, file_name='xgboost.pkl'):
	if model is None:
		logger.error("No XGBoost model available to save.")
		return

	output_path = os.path.join(ARTIFACTS_MODELS_DIR, file_name)
	logger.info(f"Saving XGBoost model to: {output_path}")
	try:
		os.makedirs(ARTIFACTS_MODELS_DIR, exist_ok=True)
		with open(output_path, 'wb') as f:
			pickle.dump(model, f)
		logger.info("XGBoost model saved successfully.")
	except Exception as e:
		logger.error(f"Error saving XGBoost model: {e}")


# run_xgboost()