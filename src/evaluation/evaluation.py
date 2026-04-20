from src.evaluation.evaluate_lg import run_logistic_regression_evaluation
from src.evaluation.evaluate_rf import run_random_forest_evaluation
from src.evaluation.evaluate_xgb import run_xgboost_evaluation
from src.utils.config import logger

def run_evaluation():
    # load best model -  artifacts/model/best_model.pkl
    # load test data - 
# def compare_models(random_states=None):
# 	logger.info("Running repeated evaluation for Logistic Regression, Random Forest, and XGBoost.")

# 	if random_states is None:
# 		random_states = [42, 52, 62, 72, 82]

# 	per_run_results = {
# 		'logistic_regression': [],
# 		'random_forest': [],
# 		'xgboost': [],
# 	}

# 	for random_state in random_states:
# 		logger.info(f"Running evaluations with random_state={random_state}.")
# 		lg_result = run_logistic_regression_evaluation(random_state=random_state)
# 		rf_result = run_random_forest_evaluation(random_state=random_state)
# 		xgb_result = run_xgboost_evaluation(random_state=random_state)

# 		if lg_result is not None:
# 			per_run_results['logistic_regression'].append(lg_result)
# 		if rf_result is not None:
# 			per_run_results['random_forest'].append(rf_result)
# 		if xgb_result is not None:
# 			per_run_results['xgboost'].append(xgb_result)

# 	aggregated_results = {}
# 	for model_name, runs in per_run_results.items():
# 		if not runs:
# 			continue

# 		aggregated_results[model_name] = {
# 			'accuracy': sum(item['accuracy'] for item in runs) / len(runs),
# 			'precision': sum(item['precision'] for item in runs) / len(runs),
# 			'recall': sum(item['recall'] for item in runs) / len(runs),
# 			'f1_score': sum(item['f1_score'] for item in runs) / len(runs),
# 			'roc_auc': sum(item['roc_auc'] for item in runs) / len(runs),
# 			'log_loss': sum(item['log_loss'] for item in runs) / len(runs),
# 			'runs': len(runs),
# 		}

# 	if not aggregated_results:
# 		logger.error("No model evaluation results were generated across repeated splits.")
# 		return None

# 	logger.info("Model comparison summary (mean across repeated stratified splits):")
# 	for model_name, metrics in aggregated_results.items():
# 		logger.info(
# 			f"{model_name}: accuracy={metrics['accuracy']:.4f}, precision={metrics['precision']:.4f}, "
# 			f"recall={metrics['recall']:.4f}, f1_score={metrics['f1_score']:.4f}, "
# 			f"roc_auc={metrics['roc_auc']:.4f}, log_loss={metrics['log_loss']:.4f}, "
# 			f"runs={metrics['runs']}"
# 		)

# 	best_model_name, best_model_metrics = max(
# 		aggregated_results.items(),
# 		key=lambda item: (item[1]['f1_score'], item[1]['roc_auc'], item[1]['accuracy'], -item[1]['log_loss']),
# 	)

# 	logger.info(
# 		f"Best performing model: {best_model_name} "
# 		f"(F1={best_model_metrics['f1_score']:.4f}, "
# 		f"ROC-AUC={best_model_metrics['roc_auc']:.4f}, "
# 		f"Accuracy={best_model_metrics['accuracy']:.4f}, "
# 		f"LogLoss={best_model_metrics['log_loss']:.4f})"
# 	)

# 	return {
# 		'per_run_results': per_run_results,
# 		'all_results': aggregated_results,
# 		'best_model_name': best_model_name,
# 		'best_model_metrics': best_model_metrics,
# 	}
