from src.utils.config import logger
from src.utils.helpers import load_data
from src.ingestion_preprocessing.feature_eng import run_cleaning_pipeline
def main():
    logger.info("Starting the Bank Loan Approval Prediction project.")
    # Your main code for data loading, preprocessing, model training, etc. goes here.
    cleaned_df = run_cleaning_pipeline()
    train_df, test_df = run_data_splitting(cleaned_df)
    logger.info("Finished processing.")
if __name__ == "__main__":
    main()