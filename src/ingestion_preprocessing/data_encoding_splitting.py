import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from src.ingestion_preprocessing.feature_eng import run_feature_eng
from src.utils.config import logger


def split_X_y(cleaned_df):
    logger.info("Splitting cleaned DataFrame into predictors (X) and target (y).")
    try:
        predictor_cols = ['Property_Area', 'Education', 'Married', 'Credit_History']
        X = cleaned_df[predictor_cols].copy()
        y = cleaned_df['Loan_Status'].map({'Y': 1, 'N': 0})
        logger.info(f"Predictors selected: {predictor_cols}")
        logger.info("Target variable mapped: Y -> 1, N -> 0")
        return X, y
    except Exception as e:
        logger.error(f"Error splitting predictors and target: {e}")
        return None, None


def encode_predictors(X):
    logger.info("Encoding predictor columns using OneHotEncoder.")
    try:
        encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
        encoded_array = encoder.fit_transform(X)
        encoded_columns = encoder.get_feature_names_out(X.columns)
        encoded_X = pd.DataFrame(encoded_array, columns=encoded_columns, index=X.index)

        logger.info("Predictor columns encoded successfully.")
        return encoded_X, encoder
    except Exception as e:
        logger.error(f"Error encoding predictor columns: {e}")
        return None, None


def split_train_test(X, y, test_size=0.2, random_state=42):
    logger.info("Splitting X and y into training and testing sets.")
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y,
        )
        logger.info(
            f"Train-test split completed: X_train={X_train.shape}, X_test={X_test.shape}, "
            f"y_train={y_train.shape}, y_test={y_test.shape}"
        )
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logger.error(f"Error splitting X and y into train and test sets: {e}")
        return None, None, None, None

def run_data_encoding_splitting(cleaned_df):
    logger.info("Running data encoding and splitting pipeline.")
    X, y = split_X_y(cleaned_df)
    if X is not None and y is not None:
        encoded_X, encoder = encode_predictors(X)
        if encoded_X is not None:
            X_train, X_test, y_train, y_test = split_train_test(encoded_X, y)
            return X_train, X_test, y_train, y_test
    logger.error("Data encoding and splitting pipeline failed.")
    return None, None, None, None


