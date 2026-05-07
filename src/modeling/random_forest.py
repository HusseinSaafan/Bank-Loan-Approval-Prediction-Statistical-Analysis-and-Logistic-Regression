import json
import os
import pickle

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.utils.config import logger
from src.utils.helpers import compute_cv_scores, plot_cv_scores, plot_shap_values, run_grid_search

TRAIN_ENCODED_PATH = os.path.join('database', 'train_encoded.csv')
ARTIFACTS_MODELS_DIR = os.path.join('artifacts', 'models')


def build_random_forest(X_train, y_train, best_params=None):
	logger.info("Building Random Forest classifier.")
	try:
		params = best_params if best_params is not None else {
			'n_estimators': 200,
			'max_depth': None,
			'min_samples_split': 2,
			'min_samples_leaf': 1,
			'max_features': 'sqrt',
		}

		model = RandomForestClassifier(**params, random_state=42)
		model.fit(X_train, y_train)
		logger.info("Random Forest model trained successfully.")
		return model
	except Exception as e:
		logger.error(f"Error building Random Forest model: {e}")
		return None


def tune_random_forest(X_train, y_train):
	logger.info("Running stratified GridSearchCV for Random Forest hyperparameter tuning.")
	try:
		param_grid = {
			'n_estimators': [100, 200, 300],
			'max_depth': [None, 5, 10, 20],
			'min_samples_split': [2, 5, 10],
			'min_samples_leaf': [1, 2, 4],
			'max_features': ['sqrt', 'log2'],
			'bootstrap': [True, False],
		}

		base_model = RandomForestClassifier(random_state=42)
		grid_search = run_grid_search(base_model, param_grid, X_train, y_train)
		if grid_search is None:
			return None

		fold_scores, fold_labels = compute_cv_scores(grid_search.best_estimator_, X_train, y_train)
		plot_cv_scores(
			fold_labels, fold_scores,
			title='Random Forest — Cross-Validation F1 Score per Fold',
			color='forestgreen',
			plot_path=os.path.join(ARTIFACTS_MODELS_DIR, 'random_forest_cv_scores.html'),
		)
		return grid_search
	except Exception as e:
		logger.error(f"Error during Random Forest grid search: {e}")
		return None


def run_random_forest():
	logger.info("Running Random Forest modeling pipeline.")

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

	grid_search = tune_random_forest(X_train, y_train)
	if grid_search is None:
		logger.error("Modeling pipeline failed: grid search tuning did not complete.")
		return None

	plot_shap_values(
		estimator=grid_search.best_estimator_,
		X=X_train,
		title='Random Forest — Mean Absolute SHAP Values',
		plot_path=os.path.join(ARTIFACTS_MODELS_DIR, 'random_forest_shap_values.html'),
	)

	model = build_random_forest(X_train, y_train, best_params=grid_search.best_params_)
	save_random_forest_model(model)

	# Save best model F1 score to JSON.
	fold_scores, _ = compute_cv_scores(grid_search.best_estimator_, X_train, y_train)
	if fold_scores is not None:
		scores_dict = {
			'model': 'random_forest',
			'best_params': grid_search.best_params_,
			'best_cv_f1': round(float(grid_search.best_score_), 6),
			'cv_f1_per_fold': [round(float(s), 6) for s in fold_scores],
			'cv_f1_mean': round(float(fold_scores.mean()), 6),
			'cv_f1_std': round(float(fold_scores.std()), 6),
		}
		scores_path = os.path.join(ARTIFACTS_MODELS_DIR, 'random_forest_f1_scores.json')
		try:
			os.makedirs(ARTIFACTS_MODELS_DIR, exist_ok=True)
			with open(scores_path, 'w') as f:
				json.dump(scores_dict, f, indent=4)
			logger.info(f"Random Forest F1 scores saved to: {scores_path}")
		except Exception as e:
			logger.error(f"Error saving F1 scores: {e}")

	return {
		'grid_search': grid_search,
		'best_model': model,
		'best_params': grid_search.best_params_,
		'best_cv_score': grid_search.best_score_,
	}


def save_random_forest_model(model, file_name='random_forest.pkl'):
	if model is None:
		logger.error("No Random Forest model available to save.")
		return

	output_path = os.path.join(ARTIFACTS_MODELS_DIR, file_name)
	logger.info(f"Saving Random Forest model to: {output_path}")
	try:
		os.makedirs(ARTIFACTS_MODELS_DIR, exist_ok=True)
		with open(output_path, 'wb') as f:
			pickle.dump(model, f)
		logger.info("Random Forest model saved successfully.")
	except Exception as e:
		logger.error(f"Error saving Random Forest model: {e}")

# run_random_forest()

