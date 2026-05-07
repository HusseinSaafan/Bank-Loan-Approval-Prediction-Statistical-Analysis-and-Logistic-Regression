import json
import os
import pickle

from src.utils.config import logger

ARTIFACTS_MODELS_DIR = os.path.join('artifacts', 'models')

_MODEL_REGISTRY = {
    'logistic_regression': {
        'scores_file': 'logistic_regression_f1_scores.json',
        'pkl_file': 'logistic_regression.pkl',
    },
    'random_forest': {
        'scores_file': 'random_forest_f1_scores.json',
        'pkl_file': 'random_forest.pkl',
    },
    'xgboost': {
        'scores_file': 'xgboost_f1_scores.json',
        'pkl_file': 'xgboost.pkl',
    },
}


def select_and_save_best_model():
    """
    Loads the F1 score JSON files for all registered models, identifies the one
    with the highest mean CV F1 score, loads its corresponding .pkl file, and
    saves it as best_model.pkl in the artifacts/models directory.
    """
    logger.info("Selecting best model based on cross-validation F1 scores.")

    scores = {}
    for model_name, paths in _MODEL_REGISTRY.items():
        scores_path = os.path.join(ARTIFACTS_MODELS_DIR, paths['scores_file'])
        if not os.path.exists(scores_path):
            logger.warning(f"Scores file not found for '{model_name}': {scores_path}. Skipping.")
            continue
        try:
            with open(scores_path, 'r') as f:
                data = json.load(f)
            scores[model_name] = data.get('cv_f1_mean', 0.0)
            logger.info(f"  {model_name}: cv_f1_mean = {scores[model_name]:.6f}")
        except Exception as e:
            logger.error(f"Failed to load scores file for '{model_name}': {e}")

    if not scores:
        logger.error("No model score files could be loaded. Aborting best model selection.")
        return None

    best_model_name = max(scores, key=scores.get)
    best_f1 = scores[best_model_name]
    logger.info(f"Best model: '{best_model_name}' with cv_f1_mean = {best_f1:.6f}")

    pkl_path = os.path.join(ARTIFACTS_MODELS_DIR, _MODEL_REGISTRY[best_model_name]['pkl_file'])
    if not os.path.exists(pkl_path):
        logger.error(f"Model pkl file not found: {pkl_path}")
        return None

    try:
        with open(pkl_path, 'rb') as f:
            best_model = pickle.load(f)
        logger.info(f"Loaded model from: {pkl_path}")
    except Exception as e:
        logger.error(f"Failed to load model pkl for '{best_model_name}': {e}")
        return None

    output_path = os.path.join(ARTIFACTS_MODELS_DIR, 'best_model.pkl')
    try:
        os.makedirs(ARTIFACTS_MODELS_DIR, exist_ok=True)
        with open(output_path, 'wb') as f:
            pickle.dump(best_model, f)
        logger.info(f"Best model saved as: {output_path}")
    except Exception as e:
        logger.error(f"Failed to save best_model.pkl: {e}")
        return None

    return {
        'best_model_name': best_model_name,
        'best_cv_f1': best_f1,
        'model': best_model,
    }


select_and_save_best_model()
