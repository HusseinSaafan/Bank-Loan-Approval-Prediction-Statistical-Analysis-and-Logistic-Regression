from src.evaluation.evaluate_lg import run_logistic_regression_evaluation
from src.evaluation.evaluate_rf import run_random_forest_evaluation
from src.evaluation.evaluate_xgb import run_xgboost_evaluation
from src.evaluation.evaluation import compare_models
from src.modeling.logistic_regression import run_logistic_regression
from src.modeling.random_forest import run_random_forest
from src.modeling.xgboost import run_xgboost
from src.utils.config import logger


def main():
	logger.info("Starting end-to-end pipeline for Logistic Regression, Random Forest, and XGBoost.")

	logger.info("Running Logistic Regression training and evaluation.")
	run_logistic_regression()
	run_logistic_regression_evaluation()

	logger.info("Running Random Forest training and evaluation.")
	run_random_forest()
	run_random_forest_evaluation()

	logger.info("Running XGBoost training and evaluation.")
	run_xgboost()
	run_xgboost_evaluation()

	logger.info("Running final model comparison.")
	comparison_results = compare_models()

	if comparison_results is None:
		logger.error("Final comparison failed.")
		return None

	logger.info("End-to-end training and evaluation pipeline completed successfully.")
	# return {
	# 	'model_results': {
	# 		'logistic_regression': logistic_model_results,
	# 		'random_forest': random_forest_model_results,
	# 		'xgboost': xgboost_model_results,
	# 	},
	# 	'evaluation_results': {
	# 		'logistic_regression': logistic_eval_results,
	# 		'random_forest': random_forest_eval_results,
	# 		'xgboost': xgboost_eval_results,
	# 	},
	# 	'comparison_results': comparison_results,
	}

main()

