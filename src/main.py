# from src.evaluation.evaluate_lg import run_logistic_regression_evaluation
# from src.evaluation.evaluate_rf import run_random_forest_evaluation
# from src.evaluation.evaluate_xgb import run_xgboost_evaluation
# from src.evaluation.evaluation import compare_models
from src.modeling.logistic_regression import run_logistic_regression
from src.modeling.random_forest import run_random_forest
from src.modeling.xgboost import run_xgboost
from src.utils.config import logger
from src.utils.helpers import compute_cv_scores, plot_cv_scores, plot_shap_values, run_grid_search
from src.evaluation.best_model_selector import select_and_save_best_model
from src.evaluation.test_best_model import run_test_best_model

def main():
	logger.info("Starting end-to-end pipeline for Logistic Regression, Random Forest, and XGBoost.")

	logger.info("Running Logistic Regression training and evaluation.")
	run_logistic_regression()
	# run_logistic_regression_evaluation()

	logger.info("Running Random Forest training and evaluation.")
	run_random_forest()
	# run_random_forest_evaluation()

	logger.info("Running XGBoost training and evaluation.")
	run_xgboost()
	# run_xgboost_evaluation()

	logger.info("Running final model comparison.")
	comparison_results = select_and_save_best_model()
	if comparison_results is not None:
		run_test_best_model()
	else:
		logger.error("Best model selection failed. Skipping test evaluation.")
	
	logger.info("End-to-end training and evaluation pipeline completed successfully.")
	
if __name__ == "__main__":
	main()

