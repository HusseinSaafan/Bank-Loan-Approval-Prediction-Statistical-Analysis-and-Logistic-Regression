from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score

from src.ingestion_preprocessing.data_encoding_splitting import run_data_encoding_splitting
from src.ingestion_preprocessing.feature_eng import run_feature_eng
from src.modeling.logistic_regression import tune_logistic_regression
from src.utils.config import logger


def evaluate_logistic_regression(model, X_test, y_test):
	logger.info("Evaluating logistic regression model performance on the test set.")
	try:
		y_pred = model.predict(X_test)

		cm = confusion_matrix(y_test, y_pred)
		accuracy = accuracy_score(y_test, y_pred)
		precision = precision_score(y_test, y_pred)
		recall = recall_score(y_test, y_pred)
		f1 = f1_score(y_test, y_pred)
		report = classification_report(y_test, y_pred)

		logger.info(f"Confusion Matrix:\n{cm}")
		logger.info(f"Accuracy: {accuracy:.4f}")
		logger.info(f"Precision: {precision:.4f}")
		logger.info(f"Recall: {recall:.4f}")
		logger.info(f"F1 Score: {f1:.4f}")
		logger.info(f"Classification Report:\n{report}")

		return {
			'confusion_matrix': cm,
			'accuracy': accuracy,
			'precision': precision,
			'recall': recall,
			'f1_score': f1,
			'classification_report': report,
		}
	except Exception as e:
		logger.error(f"Error evaluating logistic regression model: {e}")
		return None


def run_logistic_regression_evaluation():
	logger.info("Running logistic regression evaluation pipeline.")

	cleaned_df = run_feature_eng()
	if cleaned_df is None:
		logger.error("Evaluation pipeline failed: could not produce cleaned data.")
		return None

	X_train, X_test, y_train, y_test = run_data_encoding_splitting(cleaned_df)
	if any(item is None for item in [X_train, X_test, y_train, y_test]):
		logger.error("Evaluation pipeline failed: data encoding or splitting produced None.")
		return None

	grid_search = tune_logistic_regression(X_train, y_train)
	if grid_search is None:
		logger.error("Evaluation pipeline failed: model tuning did not complete.")
		return None

	best_model = grid_search.best_estimator_
	return evaluate_logistic_regression(best_model, X_test, y_test)
