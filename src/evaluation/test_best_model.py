"""
Test Best Model Evaluation Script

This script performs comprehensive evaluation of the best performing model
on test data. It:

1. Loads the best model previously identified and saved by best_model_selector
2. Loads the test dataset (test_encoded.csv)
3. Generates predictions on test data
4. Calculates evaluation metrics:
   - F1 Score
   - Confusion Matrix
   - ROC AUC Score
5. Saves all results to artifacts/models/best_model directory with:
   - Test results JSON file containing F1 score and ROC AUC
   - Confusion matrix saved as a JSON file
   - Predictions on test set

The script is model-agnostic and works with any scikit-learn compatible model
(LogisticRegression, RandomForest, XGBoost, etc.)
"""

import os
import pickle
import json

import pandas as pd
import numpy as np
from sklearn.metrics import (
    f1_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    auc,
    classification_report,
)
import plotly.graph_objects as go

from src.utils.config import logger

ARTIFACTS_MODELS_DIR = os.path.join('artifacts', 'models')
BEST_MODEL_DIR = os.path.join(ARTIFACTS_MODELS_DIR, 'best_model')
TEST_DATA_PATH = os.path.join('database', 'test_encoded.csv')


def load_test_data():
    """
    Load and prepare test data for evaluation.

    Returns:
        tuple: (X_test, y_test) - Features and labels for testing
    """
    try:
        logger.info(f"Loading test data from {TEST_DATA_PATH}")
        test_df = pd.read_csv(TEST_DATA_PATH)
        logger.info(f"Test data loaded successfully. Shape: {test_df.shape}")

        # Separate features and target
        target_col = 'Loan_Status'
        if target_col not in test_df.columns:
            logger.error(f"Target column '{target_col}' not found in test data.")
            return None, None

        y_test = test_df[target_col]
        X_test = test_df.drop(columns=[target_col])

        logger.info(f"Features shape: {X_test.shape}, Target shape: {y_test.shape}")
        return X_test, y_test

    except Exception as e:
        logger.error(f"Error loading test data: {e}")
        return None, None


def load_best_model():
    """
    Load the best model from artifacts/models/best_model.pkl.

    Returns:
        object: Trained model object or None if loading fails
    """
    best_model_path = os.path.join(ARTIFACTS_MODELS_DIR, 'best_model.pkl')

    if not os.path.exists(best_model_path):
        logger.error(f"Best model file not found at: {best_model_path}")
        return None

    try:
        with open(best_model_path, 'rb') as f:
            best_model = pickle.load(f)
        logger.info(f"Best model loaded successfully from: {best_model_path}")
        return best_model
    except Exception as e:
        logger.error(f"Error loading best model: {e}")
        return None


def generate_predictions(model, X_test):
    """
    Generate predictions and probability scores from the model.

    Args:
        model: Trained model object
        X_test (pd.DataFrame): Test features

    Returns:
        tuple: (predictions, probabilities) - Class predictions and probability scores
    """
    try:
        logger.info("Generating predictions on test data...")
        y_pred = model.predict(X_test)

        # Try to get probability scores (works for most sklearn models)
        try:
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        except AttributeError:
            logger.warning("Model does not support probability predictions. Using predicted scores.")
            # For some models, try to use decision_function or similar
            try:
                y_pred_proba = model.decision_function(X_test)
                # Normalize to [0, 1] range if needed
                y_pred_proba = (y_pred_proba - y_pred_proba.min()) / (y_pred_proba.max() - y_pred_proba.min())
            except AttributeError:
                logger.warning("Unable to extract probability scores.")
                y_pred_proba = None

        logger.info(f"Predictions generated. Shape: {y_pred.shape}")
        return y_pred, y_pred_proba

    except Exception as e:
        logger.error(f"Error generating predictions: {e}")
        return None, None


def calculate_metrics(y_test, y_pred, y_pred_proba):
    """
    Calculate evaluation metrics: F1 score, confusion matrix, and ROC AUC.

    Args:
        y_test: True test labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities

    Returns:
        dict: Dictionary containing all calculated metrics
    """
    try:
        logger.info("Calculating evaluation metrics...")

        # F1 Score
        f1 = f1_score(y_test, y_pred)
        logger.info(f"F1 Score: {f1:.6f}")

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        logger.info(f"Confusion Matrix:\n{cm}")

        # ROC AUC Score
        roc_auc = None
        fpr, tpr = None, None
        if y_pred_proba is not None:
            try:
                roc_auc = roc_auc_score(y_test, y_pred_proba)
                fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                logger.info(f"ROC AUC Score: {roc_auc:.6f}")
            except Exception as e:
                logger.warning(f"Could not calculate ROC AUC: {e}")
        else:
            logger.warning("Probability predictions unavailable. Skipping ROC AUC calculation.")

        # Additional metrics
        class_report = classification_report(y_test, y_pred, output_dict=True)

        metrics = {
            'f1_score': float(f1),
            'confusion_matrix': cm.tolist(),
            'roc_auc_score': float(roc_auc) if roc_auc is not None else None,
            'classification_report': class_report,
            'fpr': fpr.tolist() if fpr is not None else None,
            'tpr': tpr.tolist() if tpr is not None else None,
        }

        return metrics

    except Exception as e:
        logger.error(f"Error calculating metrics: {e}")
        return None


def save_results(metrics, y_test, y_pred, y_pred_proba):
    """
    Save evaluation results to artifacts/models/best_model directory.

    Args:
        metrics (dict): Dictionary containing calculated metrics
        y_test: True test labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities
    """
    try:
        # Create output directory
        os.makedirs(BEST_MODEL_DIR, exist_ok=True)
        logger.info(f"Output directory created/confirmed: {BEST_MODEL_DIR}")

        # Save metrics to JSON
        metrics_path = os.path.join(BEST_MODEL_DIR, 'test_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        logger.info(f"Metrics saved to: {metrics_path}")

        # Save confusion matrix separately for clarity
        cm_path = os.path.join(BEST_MODEL_DIR, 'confusion_matrix.json')
        cm_data = {
            'confusion_matrix': metrics['confusion_matrix'],
            'interpretation': {
                'true_negatives': int(metrics['confusion_matrix'][0][0]),
                'false_positives': int(metrics['confusion_matrix'][0][1]),
                'false_negatives': int(metrics['confusion_matrix'][1][0]),
                'true_positives': int(metrics['confusion_matrix'][1][1]),
            }
        }
        with open(cm_path, 'w') as f:
            json.dump(cm_data, f, indent=4)
        logger.info(f"Confusion matrix saved to: {cm_path}")

        # Save predictions
        predictions_path = os.path.join(BEST_MODEL_DIR, 'test_predictions.json')
        predictions_data = {
            'y_test': y_test.tolist(),
            'y_pred': y_pred.tolist(),
            'y_pred_proba': y_pred_proba.tolist() if y_pred_proba is not None else None,
        }
        with open(predictions_path, 'w') as f:
            json.dump(predictions_data, f, indent=4)
        logger.info(f"Predictions saved to: {predictions_path}")

        # Create ROC curve visualization if available
        if metrics['fpr'] is not None and metrics['tpr'] is not None:
            create_roc_curve_plot(metrics['fpr'], metrics['tpr'], metrics['roc_auc_score'])

        logger.info("All results saved successfully to artifacts/models/best_model/")

    except Exception as e:
        logger.error(f"Error saving results: {e}")


def create_roc_curve_plot(fpr, tpr, roc_auc):
    """
    Create and save a ROC curve visualization.

    Args:
        fpr: False positive rates
        tpr: True positive rates
        roc_auc: ROC AUC score
    """
    try:
        fig = go.Figure()

        # ROC Curve
        fig.add_trace(go.Scatter(
            x=fpr,
            y=tpr,
            mode='lines',
            name=f'ROC Curve (AUC = {roc_auc:.4f})',
            line=dict(color='#1f77b4', width=2),
        ))

        # Diagonal reference line
        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            name='Random Classifier',
            line=dict(color='red', width=2, dash='dash'),
        ))

        fig.update_layout(
            title='ROC Curve - Best Model Performance',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            hovermode='closest',
            template='plotly_white',
            width=800,
            height=600,
        )

        roc_plot_path = os.path.join(BEST_MODEL_DIR, 'roc_curve.html')
        fig.write_html(roc_plot_path)
        logger.info(f"ROC curve plot saved to: {roc_plot_path}")

    except Exception as e:
        logger.error(f"Error creating ROC curve plot: {e}")


def run_test_best_model():
    """
    Main function to orchestrate the complete model testing pipeline.

    Workflow:
    1. Load the best model
    2. Load test data
    3. Generate predictions
    4. Calculate metrics (F1, Confusion Matrix, ROC AUC)
    5. Save results to artifacts/models/best_model/
    """
    logger.info("=" * 60)
    logger.info("Starting Best Model Testing Pipeline")
    logger.info("=" * 60)

    # Step 1: Load best model
    best_model = load_best_model()
    if best_model is None:
        logger.error("Failed to load best model. Aborting.")
        return

    # Step 2: Load test data
    X_test, y_test = load_test_data()
    if X_test is None or y_test is None:
        logger.error("Failed to load test data. Aborting.")
        return

    # Step 3: Generate predictions
    y_pred, y_pred_proba = generate_predictions(best_model, X_test)
    if y_pred is None:
        logger.error("Failed to generate predictions. Aborting.")
        return

    # Step 4: Calculate metrics
    metrics = calculate_metrics(y_test, y_pred, y_pred_proba)
    if metrics is None:
        logger.error("Failed to calculate metrics. Aborting.")
        return

    # Step 5: Save results
    save_results(metrics, y_test, y_pred, y_pred_proba)

    logger.info("=" * 60)
    logger.info("Best Model Testing Pipeline Completed Successfully")
    logger.info(f"F1 Score: {metrics['f1_score']:.6f}")
    logger.info(f"ROC AUC: {metrics['roc_auc_score']:.6f}" if metrics['roc_auc_score'] else "ROC AUC: Not available")
    logger.info("=" * 60)


if __name__ == '__main__':
    run_test_best_model()
    