import os
from pathlib import Path
import numpy as np
import pandas as pd
from src.utils.config import logger
from src.utils.helpers import load_data

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def create_new_cleaned_df(df):
    logger.info("Creating a new cleaned DataFrame.")
    try:
        # Example of creating a new cleaned DataFrame
        cleaned_df = df.copy()
        logger.info("New cleaned DataFrame created successfully.")
        return cleaned_df
    except Exception as e:
        logger.error(f"Error creating new cleaned DataFrame: {e}")
        return None
    
def remove_duplicate_rows(cleaned_df):
    logger.info("Removing duplicate rows from the DataFrame based on Loan_ID.")
    try:
        before_count = cleaned_df.shape[0]
        cleaned_df.drop_duplicates(subset=['Loan_ID'], inplace=True)
        after_count = cleaned_df.shape[0]
        logger.info(f"Removed {before_count - after_count} duplicate rows based on Loan_ID.")
        return cleaned_df
    except Exception as e:
        logger.error(f"Error removing duplicate rows: {e}")
        return cleaned_df
    
def retype_columns(cleaned_df, col_name, new_type):
    ## change creadithistory to categorical (boolean)
    logger.info("Retyping columns to appropriate data types.")
    try:
        cleaned_df[col_name] = cleaned_df[col_name].astype(new_type)
        logger.info("Columns retyped successfully.")
        return cleaned_df
    except Exception as e:
        logger.error(f"Error retyping columns: {e}")
        return cleaned_df
    
def treat_inconsistent_data(cleaned_df):
    logger.info("Treating inconsistent data in the DataFrame.")
    try:
        cleaned_df['Gender'] = cleaned_df['Gender'].replace({
            'M': 'Male',
            'F': 'Female',
            'male': 'Male',
            'female': 'Female'
        })
        cleaned_df['Education'] = cleaned_df['Education'].replace({
            'graduate': 'Graduate',
            'Not-Graduate': 'Not Graduate'
        })
        cleaned_df['Property_Area'] = cleaned_df['Property_Area'].replace({
            'rban': 'Urban'
        })
        cleaned_df['ApplicantIncome'] = cleaned_df['ApplicantIncome'].apply(lambda x: np.nan if x < 0 else x)
        cleaned_df['LoanAmount'] = cleaned_df['LoanAmount'].apply(lambda x: np.nan if x < 0 else x)
        logger.info("Inconsistent data treated successfully.")
        return cleaned_df
    except Exception as e:
        logger.error(f"Error treating inconsistent data: {e}")
        return cleaned_df
    
def feature_handling(cleaned_df):
    logger.info("Starting feature handling process.")
    try:
        cleaned_df['Total_Income'] = cleaned_df['ApplicantIncome'] + cleaned_df['CoapplicantIncome']
        cleaned_df.drop(['ApplicantIncome', 'CoapplicantIncome'], axis=1, inplace=True)
        logger.info("Feature handling completed successfully: Total_Income created and original income columns removed.")
        return cleaned_df
    except Exception as e:
        logger.error(f"Error during feature handling: {e}")
        return cleaned_df
    
def drop_missing_target(cleaned_df):
    logger.info("Dropping rows with missing target variable.")
    try:
                before_count = cleaned_df.shape[0]
                cleaned_df.dropna(subset=['Loan_Status'], inplace=True)
                after_count = cleaned_df.shape[0]
                logger.info(f"Removed {before_count - after_count} rows with missing target variable.")
    except Exception as e:
        logger.error(f"Error dropping rows with missing target variable: {e}")
    return cleaned_df

def run_cleaning_pipeline():
    file_path = 'database/Loan Approval Dataset.csv'  # Update with your actual file path
    df = load_data(file_path)
    if df is not None:
        cleaned_df = create_new_cleaned_df(df)
        cleaned_df = remove_duplicate_rows(cleaned_df)
        cleaned_df = treat_inconsistent_data(cleaned_df)
        cleaned_df = feature_handling(cleaned_df)
        cleaned_df = drop_missing_target(cleaned_df)
        cleaned_df = retype_columns(cleaned_df, 'Credit_History', 'boolean')
        logger.info("cleaned df types after retyping:")
        logger.info(cleaned_df.dtypes)
        output_path = PROJECT_ROOT / "database" / "Loan Approval Dataset Cleaned.csv"
        cleaned_df.to_csv(output_path, index=False)
        logger.info(f"Final cleaned DataFrame exported to {output_path}.")
        return cleaned_df
    else:
        logger.error("Failed to load data for cleaning pipeline.")
        return None
run_cleaning_pipeline()
