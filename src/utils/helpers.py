import pandas as pd
from src.utils.config import logger


def load_data(file_path):
    logger.info(f"Loading data from {file_path}")
    try:
        df = pd.read_csv(file_path, encoding='utf-8', index_col='Loan_ID')
        logger.info("Data loaded successfully.")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return None
    
