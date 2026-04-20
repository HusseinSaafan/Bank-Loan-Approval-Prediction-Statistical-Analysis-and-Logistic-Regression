import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from src.ingestion_preprocessing.feature_eng import run_feature_eng
from src.utils.config import logger


PROJECT_ROOT = Path(__file__).resolve().parents[2]


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


# def encode_predictors(X):
#     logger.info("Encoding predictor columns using OneHotEncoder.")
#     try:
#         encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
#         encoded_array = encoder.fit_transform(X)
#         encoded_columns = encoder.get_feature_names_out(X.columns)
#         encoded_X = pd.DataFrame(encoded_array, columns=encoded_columns, index=X.index)

#         logger.info("Predictor columns encoded successfully.")
#         return encoded_X, encoder
#     except Exception as e:
#         logger.error(f"Error encoding predictor columns: {e}")
#         return None, None


def encode_train_test(X_train, X_test):
#Only Property_Area is one-hot encoded.
#Education is mapped as Graduate = 1 and Not Graduate = 0.
#Married is mapped as Yes = 1 and No = 0.
#Credit_History is mapped as True = 1 and False = 0.
    logger.info("Encoding predictor columns for training and testing sets.")
    try:
        encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
        X_train_encoded_array = encoder.fit_transform(X_train[['Property_Area']])
        X_test_encoded_array = encoder.transform(X_test[['Property_Area']])
        encoded_columns = encoder.get_feature_names_out(['Property_Area'])
        X_train_encoded = pd.DataFrame(X_train_encoded_array, columns=encoded_columns, index=X_train.index)
        X_test_encoded = pd.DataFrame(X_test_encoded_array, columns=encoded_columns, index=X_test.index)

        # Map other categorical variables
        for col in ['Education', 'Married', 'Credit_History']:
            mapping = {
                'Education': {'Graduate': 1, 'Not Graduate': 0},
                'Married': {'Yes': 1, 'No': 0},
                'Credit_History': {True: 1, False: 0}
            }[col]
            X_train_encoded[col] = X_train[col].map(mapping)
            X_test_encoded[col] = X_test[col].map(mapping)

        logger.info("Predictor columns encoded successfully for training and testing sets.")
        return X_train_encoded, X_test_encoded, encoder
    except Exception as e:
        logger.error(f"Error encoding predictor columns for training and testing sets: {e}")
        return None, None, None

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

def run_and_save_data_encoding_splitting(cleaned_df, test_size=0.2, random_state=42):
    logger.info("Running data encoding and splitting pipeline.")
    X, y = split_X_y(cleaned_df)
    if X is not None and y is not None:
        X_train_raw, X_test_raw, y_train, y_test = split_train_test(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
        )
        if not any(item is None for item in [X_train_raw, X_test_raw, y_train, y_test]):
            X_train, X_test, _ = encode_train_test(X_train_raw, X_test_raw)
            if X_train is not None and X_test is not None:
                database_path = PROJECT_ROOT / 'database'
                database_path.mkdir(parents=True, exist_ok=True)

                train_encoded_path = database_path / 'train_encoded.csv'
                test_encoded_path = database_path / 'test_encoded.csv'

                train_output = X_train.copy()
                train_output['Loan_Status'] = y_train
                test_output = X_test.copy()
                test_output['Loan_Status'] = y_test

                train_output.to_csv(train_encoded_path, index=False)
                test_output.to_csv(test_encoded_path, index=False)

                logger.info(f"Saved encoded train dataset: {train_encoded_path}")
                logger.info(f"Saved encoded test dataset: {test_encoded_path}")
                return X_train, X_test, y_train, y_test
    logger.error("Data encoding and splitting pipeline failed.")
    return None, None, None, None


if __name__ == '__main__':
    cleaned_df = run_feature_eng()
    if cleaned_df is not None:
        run_and_save_data_encoding_splitting(cleaned_df)
    else:
        logger.error("Feature engineering pipeline failed. Data encoding and splitting was not run.")

