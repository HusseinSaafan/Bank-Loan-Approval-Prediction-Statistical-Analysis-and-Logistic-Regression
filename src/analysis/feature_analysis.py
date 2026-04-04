import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from src.utils.config import logger
from src.utils.helpers import load_data

def perform_eda(df):
    logger.info("Performing Exploratory Data Analysis (EDA).")
    try:
        # Display basic information about the dataset
        logger.info("Dataset Information:")
        logger.info(df.shape)
        logger.info("Dataset Description:")
        logger.info(df.describe())
        # Check for missing values
        logger.info("Missing Values:")
        logger.info(df.isnull().sum())
        # Create figures/eda directory if it doesn't exist
        os.makedirs('figures/eda', exist_ok=True)
        # Visualize the distribution of the target variable and save as HTML
        fig = px.histogram(df, x='Loan_Status', title='Distribution of Loan Status')
        fig.write_html('figures/eda/loan_status_distribution.html')
        logger.info("Plot saved to figures/eda/loan_status_distribution.html")
    except Exception as e:
        logger.error(f"Error during EDA: {e}")
def run_feature_analysis():
    file_path = 'database/Loan Approval Dataset.csv'  # Update with your actual file path
    df = load_data(file_path)
    if df is not None:
        perform_eda(df)
run_feature_analysis()