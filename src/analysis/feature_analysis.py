import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from src.utils.config import logger
from src.utils.helpers import load_data

# def  explore_features(df):
#     logger.info("Exploring features in the dataset.")
#     try:
#         pd.set_option('display.float_format', '{:.2f}'.format)
#         # Display basic information about the dataset
#         logger.info("Dataset Head:")
#         logger.info(df.head())
#         logger.info("Dataset Information:")
#         logger.info(df.shape)
#         logger.info("Dataset Description:")
#         logger.info(df.describe())
#     except Exception as e:
#         logger.error(f"Error during EDA: {e}")
# def split_categorical_numerical(df):
#     logger.info("Splitting columns into categorical and numerical.")
#     try:
#         categorical_cols = df.select_dtypes(include=['object']).columns
#         numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
#         logger.info(f"Categorical columns: {categorical_cols}")
#         logger.info(f"Numerical columns: {numerical_cols}")
#         return categorical_cols, numerical_cols
#     except Exception as e:
#         logger.error(f"Error splitting columns: {e}")
#         return None, None
# def perform_cat_num_eda(df):
#     logger.info("Performing EDA on categorical and numerical features.")
#     try:
#         categorical_cols, numerical_cols = split_categorical_numerical(df)
#         if categorical_cols is not None and numerical_cols is not None:
#             # Perform EDA on categorical columns
#             for col in categorical_cols:
#                 logger.info(f"Exploring {col}:")
#                 logger.info(df[col].value_counts())
#                 logger.info(df[col].isnull().sum())
#                 logger.info(df[col].unique())
#             # Perform EDA on numerical columns
#             for col in numerical_cols:
#                 logger.info(f"Exploring {col}:")
#                 logger.info(df[col].describe())
#                 logger.info(df[col].isnull().sum())
#     except Exception as e:
#         logger.error(f"Error during EDA: {e}")

# def run_feature_analysis():
#     file_path = 'database/Loan Approval Dataset.csv'  # Update with your actual file path
#     df = load_data(file_path)
#     if df is not None:
#         explore_features(df)
#         perform_cat_num_eda(df)




def  explore_cleaned_features(cleaned_df):
    logger.info("Exploring features in the dataset.")
    try:
        pd.set_option('display.float_format', '{:.2f}'.format)
        # Display basic information about the dataset
        logger.info("Dataset Head:")
        logger.info(cleaned_df.head())
        logger.info("Dataset Information:")
        logger.info(cleaned_df.shape)
        logger.info("Dataset Description:")
        logger.info(cleaned_df.describe())
    except Exception as e:
        logger.error(f"Error during EDA: {e}")
def split_cleaned_categorical_numerical(cleaned_df):
    logger.info("Splitting columns into categorical and numerical.")
    try:
        categorical_cols = [
    col for col in cleaned_df.select_dtypes(include=['object']).columns
    if col != 'Loan_ID'
]
        numerical_cols = cleaned_df.select_dtypes(include=['int64', 'float64']).columns
        logger.info(f"Categorical columns: {categorical_cols}")
        logger.info(f"Numerical columns: {numerical_cols}")
        return categorical_cols, numerical_cols
    except Exception as e:
        logger.error(f"Error splitting columns: {e}")
        return None, None
def perform_cleaned_cat_num_eda(cleaned_df):
    logger.info("Performing EDA on categorical and numerical features.")
    try:
        categorical_cols, numerical_cols = split_cleaned_categorical_numerical(cleaned_df)
        if categorical_cols is not None and numerical_cols is not None:
            # Perform EDA on categorical columns
            for col in categorical_cols:
                logger.info(f"Exploring {col}:")
                logger.info(cleaned_df[col].value_counts())
                logger.info(cleaned_df[col].isnull().sum())
                logger.info(cleaned_df[col].unique())
            # Perform EDA on numerical columns
            for col in numerical_cols:
                logger.info(f"Exploring {col}:")
                logger.info(cleaned_df[col].describe())
                logger.info(cleaned_df[col].isnull().sum())
    except Exception as e:
        logger.error(f"Error during EDA: {e}")

def run_cleaned_feature_analysis():
    file_path = 'database/Loan Approval Dataset Cleaned.csv'  # Update with your actual file path
    cleaned_df = load_data(file_path)
    if cleaned_df is not None:
        explore_cleaned_features(cleaned_df)
        perform_cleaned_cat_num_eda(cleaned_df)
run_cleaned_feature_analysis()







