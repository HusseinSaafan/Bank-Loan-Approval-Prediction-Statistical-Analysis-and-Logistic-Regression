from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, log_loss, precision_score, recall_score, roc_auc_score

from src.ingestion_preprocessing.data_encoding_splitting import run_data_encoding_splitting
from src.ingestion_preprocessing.feature_eng import run_feature_eng
from src.modeling.xgboost import tune_xgboost
from src.utils.config import logger


def evaluate_xgboost(model, X_test, y_test):
	logger.info("Evaluating XGBoost model performance on the test set.")
	try:
		y_pred = model.predict(X_test)
		y_pred_proba = model.predict_proba(X_test)[:, 1]

		cm = confusion_matrix(y_test, y_pred)
		accuracy = accuracy_score(y_test, y_pred)
		precision = precision_score(y_test, y_pred)
		recall = recall_score(y_test, y_pred)
		f1 = f1_score(y_test, y_pred)
		roc_auc = roc_auc_score(y_test, y_pred_proba)
		logloss = log_loss(y_test, y_pred_proba)
		report = classification_report(y_test, y_pred)

		logger.info(f"Confusion Matrix:\n{cm}")
		logger.info(f"Accuracy: {accuracy:.4f}")
		logger.info(f"Precision: {precision:.4f}")
		logger.info(f"Recall: {recall:.4f}")
		logger.info(f"F1 Score: {f1:.4f}")
		logger.info(f"ROC-AUC: {roc_auc:.4f}")
		logger.info(f"Log Loss: {logloss:.4f}")
		logger.info(f"Classification Report:\n{report}")

		return {
			'confusion_matrix': cm,
			'accuracy': accuracy,
			'precision': precision,
			'recall': recall,
			'f1_score': f1,
			'roc_auc': roc_auc,
			'log_loss': logloss,
			'classification_report': report,
		}
	except Exception as e:
		logger.error(f"Error evaluating XGBoost model: {e}")
		return None


def run_xgboost_evaluation(random_state=42):
	logger.info("Running XGBoost evaluation pipeline.")

	cleaned_df = run_feature_eng()
	if cleaned_df is None:
		logger.error("Evaluation pipeline failed: could not produce cleaned data.")
		return None

	X_train, X_test, y_train, y_test = run_data_encoding_splitting(cleaned_df, random_state=random_state)
	if any(item is None for item in [X_train, X_test, y_train, y_test]):
		logger.error("Evaluation pipeline failed: data encoding or splitting produced None.")
		return None

	grid_search = tune_xgboost(X_train, y_train)
	if grid_search is None:
		logger.error("Evaluation pipeline failed: model tuning did not complete.")
		return None

	best_model = grid_search.best_estimator_
	return evaluate_xgboost(best_model, X_test, y_test)

