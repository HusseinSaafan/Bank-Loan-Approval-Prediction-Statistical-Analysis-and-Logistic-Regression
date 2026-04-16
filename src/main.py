from src.evaluation.evaluate_lg import run_logistic_regression_evaluation
from src.modeling.logistic_regression import run_logistic_regression
from src.utils.config import logger


def main():
	logger.info("Starting the logistic regression training and evaluation pipeline.")

	model_results = run_logistic_regression()
	if model_results is None:
		logger.error("Training pipeline failed. Skipping evaluation.")
		return None

	evaluation_results = run_logistic_regression_evaluation()
	if evaluation_results is None:
		logger.error("Evaluation pipeline failed.")
		return None

	logger.info("Training and evaluation pipeline completed successfully.")
	return {
		'model_results': model_results,
		'evaluation_results': evaluation_results,
	}


if __name__ == "__main__":
	main()
