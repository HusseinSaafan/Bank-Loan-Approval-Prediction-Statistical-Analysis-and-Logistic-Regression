import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from scipy.stats import shapiro, probplot
from src.utils.config import logger
from src.utils.helpers import load_data

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def explore_numerical_features(cleaned_df):
    logger.info("Exploring numerical features in the dataset.")
    try:
        numerical_cols = cleaned_df.select_dtypes(include=['int64', 'float64']).columns
        for col in numerical_cols:
            logger.info(f"Exploring {col}:")
            logger.info(cleaned_df[col].describe())
            logger.info(cleaned_df[col].isnull().sum())
    except Exception as e:
        logger.error(f"Error exploring numerical features: {e}")

def plot_numerical_distributions(cleaned_df):
    logger.info("Plotting distributions of numerical features.")
    try:
        numerical_cols = cleaned_df.select_dtypes(include=['int64', 'float64']).columns
        figures_path = PROJECT_ROOT / "figures" / "eda"
        figures_path.mkdir(parents=True, exist_ok=True)
        n_rows = 2
        n_cols = 3
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 10))
        axes = axes.flatten()
        for idx, name in enumerate(numerical_cols):
            if idx < len(axes):
                sns.kdeplot(cleaned_df[name].dropna(), ax=axes[idx])
                axes[idx].set_title(f"KDE Plot: {name}")
        for i in range(len(numerical_cols), len(axes)):
            fig.delaxes(axes[i])
        plt.tight_layout()
        plot_path = figures_path / "numerical_distributions_kde.png"
        plt.savefig(plot_path, dpi=100, bbox_inches='tight')
        logger.info(f"KDE plots saved: {plot_path}")
        plt.close()
    except Exception as e:
        logger.error(f"Error plotting numerical distributions: {e}")

def perform_normality_tests(cleaned_df):
    logger.info("Performing normality tests on numerical features.")
    try:
        numerical_cols = cleaned_df.select_dtypes(include=['int64', 'float64']).columns
        for column in numerical_cols:
            stat, p = shapiro(cleaned_df[column].dropna())  # Drop missing values
            if p > 0.05:
                logger.info(f"{column}: Probably Normal (p = {p:.3f})")
            else:
                logger.info(f"{column}: Probably Not Normal (p = {p:.3f})")
    except Exception as e:
        logger.error(f"Error performing normality tests: {e}")

def plot_qq_plots(cleaned_df):
    logger.info("Plotting Q-Q plots for numerical features.")
    try:
        numerical_cols = cleaned_df.select_dtypes(include=['int64', 'float64']).columns
        figures_path = PROJECT_ROOT / "figures"
        figures_path.mkdir(parents=True, exist_ok=True)
        n_rows = 2
        n_cols = 3
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 10))
        axes = axes.flatten()
        for idx, name in enumerate(numerical_cols):
            if idx < len(axes):
                probplot(cleaned_df[name].dropna(), dist="norm", plot=axes[idx])
                axes[idx].set_title(f"Q-Q Plot: {name}")
        for i in range(len(numerical_cols), len(axes)):
            fig.delaxes(axes[i])
        plt.tight_layout()
        plot_path = figures_path / "qq_plots.png"
        plt.savefig(plot_path, dpi=100, bbox_inches='tight')
        logger.info(f"Q-Q plots saved: {plot_path}")
        plt.close()
    except Exception as e:
        logger.error(f"Error plotting Q-Q plots: {e}")

def run_eda_num():
    file_path = PROJECT_ROOT / 'database' / 'Loan Approval Dataset Cleaned.csv'
    df = load_data(file_path)
    if df is not None:
        explore_numerical_features(df)
        plot_numerical_distributions(df)
        perform_normality_tests(df)
        plot_qq_plots(df)

# 1-plot bar plot for categorical features and save the plots in figures/eda/categorical_distributions.png
def plot_categorical_distributions(cleaned_df):
    logger.info("Plotting distributions of categorical features.")
    try:
        categorical_cols = cleaned_df.select_dtypes(include=['object']).columns
        figures_path = PROJECT_ROOT / "figures" / "eda"
        figures_path.mkdir(parents=True, exist_ok=True)
        n_rows = (len(categorical_cols) + 2) // 3  # Calculate rows needed for 3 columns
        n_cols = 3
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        axes = axes.flatten()
        for idx, name in enumerate(categorical_cols):
            if idx < len(axes):
                sns.countplot(x=cleaned_df[name], ax=axes[idx])
                axes[idx].set_title(f"Count Plot: {name}")
                axes[idx].tick_params(axis='x', rotation=45)
        for i in range(len(categorical_cols), len(axes)):
            fig.delaxes(axes[i])
        plt.tight_layout()
        plot_path = figures_path / "categorical_distributions.png"
        plt.savefig(plot_path, dpi=100, bbox_inches='tight')
        logger.info(f"Categorical distribution plots saved: {plot_path}")
        plt.close()
    except Exception as e:
        logger.error(f"Error plotting categorical distributions: {e}")
# 2 corsstab for each categorical features with the target variable and save the plots in figures/eda/categorical_target_relationships.png
def plot_categorical_target_relationships(cleaned_df):
    logger.info("Plotting relationships between categorical features and target variable.")
    try:
        categorical_cols = cleaned_df.select_dtypes(include=['object']).columns
        figures_path = PROJECT_ROOT / "figures" / "eda"
        figures_path.mkdir(parents=True, exist_ok=True)
        n_rows = (len(categorical_cols) + 2) // 3  # Calculate rows needed for 3 columns
        n_cols = 3
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        axes = axes.flatten()
        for idx, name in enumerate(categorical_cols):
            if idx < len(axes):
                sns.countplot(x=cleaned_df[name], hue=cleaned_df['Loan_Status'], ax=axes[idx])
                axes[idx].set_title(f"{name} vs Loan_Status")
                axes[idx].tick_params(axis='x', rotation=45)
        for i in range(len(categorical_cols), len(axes)):
            fig.delaxes(axes[i])
        plt.tight_layout()
        plot_path = figures_path / "categorical_target_relationships.png"
        plt.savefig(plot_path, dpi=100, bbox_inches='tight')
        logger.info(f"Categorical-target relationship plots saved: {plot_path}")
        plt.close()
    except Exception as e:
        logger.error(f"Error plotting categorical-target relationships: {e}")


def run_eda_cat():
    file_path = PROJECT_ROOT / 'database' / 'Loan Approval Dataset Cleaned.csv'
    df = load_data(file_path)
    if df is not None:
        plot_categorical_distributions(df)
        plot_categorical_target_relationships(df)

def run_eda():
    run_eda_num()
    run_eda_cat()

run_eda()