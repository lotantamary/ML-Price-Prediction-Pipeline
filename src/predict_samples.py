import pandas as pd
import joblib
import os
import logging

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Dynamic path management
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "bulldozer_model_v1.joblib")
DATA_PATH = os.path.join(BASE_DIR, "data", "bluebook-for-bulldozers")

def add_datetime_features(df):
    """
    REQUIRED: This function must be present for the pipeline to load,
    as it was used during training.
    """
    df = df.copy()
    df["saledate"] = pd.to_datetime(df["saledate"])
    df["saleYear"] = df.saledate.dt.year
    df["saleMonth"] = df.saledate.dt.month
    df["saleDay"] = df.saledate.dt.day
    df["saleDayOfWeek"] = df.saledate.dt.dayofweek
    df["saleDayOfYear"] = df.saledate.dt.dayofyear
    df.drop("saledate", axis=1, inplace=True)
    return df

def run_prediction_samples(num_samples=10):
    """
    Loads the trained model and predicts prices for a subset of the validation set,
    comparing predictions with actual market prices.
    """
    # 1. Check if model exists
    if not os.path.exists(MODEL_PATH):
        logging.error(f"Model not found at {MODEL_PATH}. Please run pipeline.py first.")
        return

    # 2. Load model and validation data
    logging.info("Loading model and validation datasets...")
    model = joblib.load(MODEL_PATH)
    
    X_valid = pd.read_csv(os.path.join(DATA_PATH, "Valid.csv"), low_memory=False)
    y_valid = pd.read_csv(os.path.join(DATA_PATH, "ValidSolution.csv"))

    # 3. Select random samples for demonstration
    indices = X_valid.sample(num_samples, random_state=100).index
    samples_x = X_valid.loc[indices]
    samples_y = y_valid.loc[indices]["SalePrice"].values

    # 4. Perform Inference
    logging.info(f"Generating predictions for {num_samples} samples...")
    predictions = model.predict(samples_x)

    # 5. Format and display results
    results = pd.DataFrame({
        "SalesID": samples_x["SalesID"].values,
        "Actual_Price": samples_y,
        "Predicted_Price": predictions.round(2),
        "Difference": (predictions - samples_y).round(2),
        "Error_Pct": (((predictions - samples_y) / samples_y) * 100).round(2)
    })

    print("\n" + "="*70)
    print("MODEL PREDICTION EXAMPLES (Actual vs. Predicted)")
    print("="*70)
    print(results.to_string(index=False))
    print("="*70)

if __name__ == "__main__":
    run_prediction_samples()