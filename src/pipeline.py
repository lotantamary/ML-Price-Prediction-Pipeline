import pandas as pd
import numpy as np
import joblib
import os
import logging
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, root_mean_squared_log_error
from sklearn.preprocessing import OrdinalEncoder, FunctionTransformer
from sklearn.pipeline import Pipeline

# Logging configuration for production-level monitoring
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Dynamic path management relative to project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "bluebook-for-bulldozers")
MODEL_OUTPUT_PATH = os.path.join(BASE_DIR, "models", "bulldozer_model_v1.joblib")

TRAIN_DATA_PATH = os.path.join(DATA_PATH, "Train.csv")
VALID_DATA_PATH = os.path.join(DATA_PATH, "Valid.csv")
VALID_SOLUTION_PATH = os.path.join(DATA_PATH, "ValidSolution.csv")

def add_datetime_features(df):
    """
    Feature engineering: Extracts time-based attributes from 'saledate'.
    """
    df = df.copy()
    df["saledate"] = pd.to_datetime(df["saledate"])
    
    df["saleYear"] = df["saledate"].dt.year
    df["saleMonth"] = df["saledate"].dt.month
    df["saleDay"] = df["saledate"].dt.day
    df["saleDayOfWeek"] = df["saledate"].dt.dayofweek
    df["saleDayOfYear"] = df["saledate"].dt.dayofyear
    
    return df.drop("saledate", axis=1)

def load_and_prepare_data():
    """
    Loads training and validation datasets from local CSV files.
    """
    logging.info("Loading datasets from data directory...")
    train_df = pd.read_csv(TRAIN_DATA_PATH, low_memory=False)
    valid_df = pd.read_csv(VALID_DATA_PATH, low_memory=False)
    valid_solution = pd.read_csv(VALID_SOLUTION_PATH)

    X_train = train_df.drop("SalePrice", axis=1)
    y_train = train_df["SalePrice"]

    X_valid = valid_df.copy()
    y_valid = valid_solution["SalePrice"]

    return X_train, y_train, X_valid, y_valid

def build_pipeline():
    """
    Constructs an end-to-end Scikit-Learn Pipeline including preprocessing and regressor.
    """
    # Preprocessing for numerical data
    numeric_transformer = SimpleImputer(strategy="median")
    
    # Preprocessing for categorical data
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, make_column_selector(dtype_exclude="object")),
        ("cat", categorical_transformer, make_column_selector(dtype_include="object"))
    ])

    # Final Pipeline assembly
    pipeline = Pipeline(steps=[
        ("datetime_features", FunctionTransformer(add_datetime_features)),
        ("preprocessor", preprocessor),
        ("model", RandomForestRegressor(n_jobs=-1, random_state=42, n_estimators=100))
    ])
    
    return pipeline

def evaluate(model, X_train, y_train, X_valid, y_valid):
    """
    Evaluates the model using MAE, RMSLE, and R^2 metrics.
    """
    train_preds = model.predict(X_train)
    valid_preds = model.predict(X_valid)

    scores = {
        "Train MAE": mean_absolute_error(y_train, train_preds),
        "Valid MAE": mean_absolute_error(y_valid, valid_preds),
        "Train RMSLE": root_mean_squared_log_error(y_train, train_preds),
        "Valid RMSLE": root_mean_squared_log_error(y_valid, valid_preds),
        "Valid R^2": model.score(X_valid, y_valid)
    }
    return scores

def main():
    """
    Main execution flow: Data loading, Pipeline fitting, Evaluation, and Model serialization.
    """
    # Ensure the models directory exists
    os.makedirs(os.path.dirname(MODEL_OUTPUT_PATH), exist_ok=True)

    try:
        # 1. Data Ingestion
        X_train, y_train, X_valid, y_valid = load_and_prepare_data()

        # 2. Model Training
        logging.info("Starting Pipeline training (RandomForestRegressor)...")
        pipeline = build_pipeline()
        pipeline.fit(X_train, y_train)

        # 3. Performance Metrics
        logging.info("Model training complete. Evaluating performance...")
        results = evaluate(pipeline, X_train, y_train, X_valid, y_valid)
        for metric, value in results.items():
            logging.info(f"{metric}: {value:.4f}")

        # 4. Model Serialization
        logging.info(f"Saving serialized model to: {MODEL_OUTPUT_PATH}")
        joblib.dump(pipeline, MODEL_OUTPUT_PATH)
        logging.info("Pipeline execution finished successfully.")

    except Exception as e:
        logging.error(f"Pipeline execution failed: {e}", exc_info=True)

if __name__ == "__main__":
    main()