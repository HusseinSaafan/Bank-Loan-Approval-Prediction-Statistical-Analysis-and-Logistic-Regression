import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shap
from plotly.subplots import make_subplots
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score

from src.utils.config import logger


def load_data(file_path):
    logger.info(f"Loading data from {file_path}")
    try:
        # Do not force an index column here; different datasets may or may not
        # contain Loan_ID as a regular column.
        df = pd.read_csv(file_path, encoding='utf-8')
        logger.info("Data loaded successfully.")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return None


def _default_stratified_kfold():
    return StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


def run_grid_search(base_model, param_grid, X_train, y_train, cv=None, scoring='f1'):
    logger.info(f"Running GridSearchCV for {type(base_model).__name__}.")
    try:
        cv = cv if cv is not None else _default_stratified_kfold()
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            refit=True,
        )
        grid_search.fit(X_train, y_train)
        logger.info(f"Best parameters found: {grid_search.best_params_}")
        logger.info(f"Best cross-validation {scoring.upper()} score: {grid_search.best_score_:.4f}")
        return grid_search
    except Exception as e:
        logger.error(f"Error during GridSearchCV for {type(base_model).__name__}: {e}")
        return None


def compute_cv_scores(estimator, X_train, y_train, cv=None, scoring='f1'):
    logger.info(f"Computing per-fold {scoring.upper()} scores for {type(estimator).__name__}.")
    try:
        cv = cv if cv is not None else _default_stratified_kfold()
        fold_scores = cross_val_score(estimator, X_train, y_train, cv=cv, scoring=scoring)
        fold_labels = [f'Fold {i + 1}' for i in range(len(fold_scores))]
        logger.info(f"Per-fold {scoring.upper()} scores: {dict(zip(fold_labels, fold_scores.round(4)))}")
        logger.info(f"Mean {scoring.upper()}: {fold_scores.mean():.4f} | Std: {fold_scores.std():.4f}")
        return fold_scores, fold_labels
    except Exception as e:
        logger.error(f"Error computing CV scores: {e}")
        return None, None


def plot_cv_scores(fold_labels, fold_scores, title, color, plot_path):
    logger.info(f"Saving cross-validation plot to: {plot_path}")
    try:
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=fold_labels, y=fold_scores,
            mode='lines+markers', name='F1 Score',
            line=dict(color=color, width=2),
            marker=dict(size=8),
        ))
        fig.add_hline(
            y=fold_scores.mean(),
            line=dict(color='tomato', dash='dash', width=1.5),
            annotation_text=f'Mean F1 = {fold_scores.mean():.4f}',
            annotation_position='top right',
        )
        fig.update_layout(
            title=title,
            xaxis_title='Fold',
            yaxis_title='F1 Score',
            yaxis=dict(range=[0, 1]),
            legend=dict(orientation='h'),
        )
        fig.write_html(plot_path)
        logger.info(f"Cross-validation fold scores plot saved to: {plot_path}")
    except Exception as e:
        logger.error(f"Error saving CV plot: {e}")


def plot_shap_values(estimator, X, title, plot_path, max_samples=300):
    logger.info(f"Computing SHAP values for {type(estimator).__name__}.")
    try:
        if X is None or X.empty:
            logger.error("Input data for SHAP is empty.")
            return None

        X_sample = X if len(X) <= max_samples else X.sample(n=max_samples, random_state=42)

        explainer = shap.Explainer(estimator, X_sample)
        shap_values = explainer(X_sample)

        values = shap_values.values
        if np.ndim(values) == 3:
            # Binary/multiclass shape: (n_samples, n_features, n_classes).
            values = values[:, :, 0]

        mean_abs_shap = np.abs(values).mean(axis=0)
        shap_df = pd.DataFrame(
            {
                'feature': X_sample.columns,
                'mean_abs_shap': mean_abs_shap,
            }
        ).sort_values(by='mean_abs_shap', ascending=False)

        logger.info(
            f"Top SHAP features: {dict(zip(shap_df['feature'].head(10), shap_df['mean_abs_shap'].head(10).round(6)))}"
        )

        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        bar_fig = go.Figure()
        bar_fig.add_trace(go.Bar(
            x=shap_df['feature'],
            y=shap_df['mean_abs_shap'],
            marker=dict(color='indianred'),
            name='mean(|SHAP value|)',
        ))
        bar_fig.update_layout(
            title=title,
            xaxis_title='Feature',
            yaxis_title='mean(|SHAP value|)',
            xaxis_tickangle=-35,
            showlegend=False,
        )
        bar_fig.write_html(plot_path)
        logger.info(f"SHAP summary (bar) plot saved to: {plot_path}")

        # Build a summary dot plot (distribution of SHAP values per top feature).
        top_features = shap_df['feature'].head(8).tolist()
        dot_fig = go.Figure()
        for rank, feature_name in enumerate(top_features):
            feature_idx = X_sample.columns.get_loc(feature_name)
            feature_shap = values[:, feature_idx]
            feature_vals = X_sample[feature_name].to_numpy()
            jitter = np.random.default_rng(42).normal(0, 0.08, size=len(feature_shap))
            y_positions = np.full(len(feature_shap), len(top_features) - rank) + jitter

            dot_fig.add_trace(
                go.Scatter(
                    x=feature_shap,
                    y=y_positions,
                    mode='markers',
                    marker=dict(
                        size=7,
                        color=feature_vals,
                        colorscale='RdBu',
                        reversescale=True,
                        opacity=0.75,
                        showscale=(rank == 0),
                        colorbar=dict(title='Feature Value') if rank == 0 else None,
                    ),
                    name=feature_name,
                    showlegend=False,
                    text=[feature_name] * len(feature_shap),
                    hovertemplate='Feature: %{text}<br>SHAP: %{x:.5f}<extra></extra>',
                )
            )

        dot_fig.update_layout(
            title=f"{title} - Summary Dot Plot",
            xaxis_title='SHAP value (impact on model output)',
            yaxis=dict(
                title='Feature',
                tickmode='array',
                tickvals=list(range(1, len(top_features) + 1)),
                ticktext=list(reversed(top_features)),
            ),
            height=520,
        )
        dot_plot_path = plot_path.replace('.html', '_dot.html')
        dot_fig.write_html(dot_plot_path)
        logger.info(f"SHAP summary (dot) plot saved to: {dot_plot_path}")

        # Build dependence plots for top 3 features.
        dep_features = top_features[:3]
        dep_fig = make_subplots(rows=1, cols=len(dep_features), subplot_titles=dep_features)
        for col_idx, feature_name in enumerate(dep_features, start=1):
            feature_idx = X_sample.columns.get_loc(feature_name)
            dep_fig.add_trace(
                go.Scatter(
                    x=X_sample[feature_name],
                    y=values[:, feature_idx],
                    mode='markers',
                    marker=dict(size=7, color='royalblue', opacity=0.7),
                    showlegend=False,
                    hovertemplate=(
                        f"{feature_name}: %{{x:.5f}}<br>SHAP: %{{y:.5f}}<extra></extra>"
                    ),
                ),
                row=1,
                col=col_idx,
            )
            dep_fig.update_xaxes(title_text=feature_name, row=1, col=col_idx)
            dep_fig.update_yaxes(title_text='SHAP value', row=1, col=col_idx)

        dep_fig.update_layout(
            title=f"{title} - Dependence Plots (Top Features)",
            height=430,
        )
        dep_plot_path = plot_path.replace('.html', '_dependence.html')
        dep_fig.write_html(dep_plot_path)
        logger.info(f"SHAP dependence plot saved to: {dep_plot_path}")
        return shap_df
    except Exception as e:
        logger.error(f"Error computing/saving SHAP plot: {e}")
        return None


