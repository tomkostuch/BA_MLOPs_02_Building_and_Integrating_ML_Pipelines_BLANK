# app.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import joblib
import os # Keep for os.path.join if needed elsewhere, but pathlib is preferred for this
from pathlib import Path # Import Path for robust path handling
from typing import List, Dict, Any

# --- Configuration (Imported from src/config.py) ---
# Adjusted import path based on the folder structure: app.py at root, config.py in src/
from src.config import MODEL_FILENAME, NUMERIC_COLUMNS, CATEGORICAL_COLUMNS

# Define the path to the model using pathlib for robustness
# Get the directory of the current script (app.py)
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR_PATH = BASE_DIR / "model_store" # Absolute path to the model_store directory
MODEL_PATH = MODEL_DIR_PATH / MODEL_FILENAME # Absolute path to the model file
# --------------------------------------------------------

# Global variable to hold the loaded model pipeline (will be set in app.state in a moment)
app = FastAPI(
    title="Customer Churn Prediction API",
    description="API for predicting customer churn based on various attributes."
)

# Global variable to hold the loaded model pipeline
model_pipeline = None

@app.on_event("startup")
async def load_model():
    """
    Load the pre-trained model when the FastAPI application starts up.
    """
    try:
        # In a deployed environment, the 'model_store' directory with the model
        # should already be present from your build process (e.g., Docker COPY).
        # This check is mostly for local testing if you run app.py without ensuring
        # the model_store directory exists and contains the model.
        if not MODEL_PATH.exists(): # Use pathlib's exists() method
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. "
                                    f"Please ensure the model is trained and saved in the '{MODEL_DIR_PATH}' directory.")

        # Store the model in app.state, which is the recommended way for FastAPI
        app.state.model_pipeline = joblib.load(MODEL_PATH)
        print(f"Model loaded successfully from {MODEL_PATH}")
    except FileNotFoundError as e:
        raise RuntimeError(f"Failed to load model: {e}")
    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred while loading the model: {e}")

# Define the input data model for Pydantic
class ChurnPredictRequest(BaseModel):
    # Numeric columns
    tenure: int = Field(..., description="Number of months the customer has been with the company.")
    MonthlyCharges: float = Field(..., description="The amount charged to the customer monthly.")
    TotalCharges: float = Field(..., description="The total amount charged to the customer.")

    # Categorical columns
    gender: str = Field(..., description="Customer's gender (Male/Female).")
    SeniorCitizen: int = Field(..., description="Whether the customer is a senior citizen (0/1).")
    Partner: str = Field(..., description="Whether the customer has a partner (Yes/No).")
    Dependents: str = Field(..., description="Whether the customer has dependents (Yes/No).")
    PhoneService: str = Field(..., description="Whether the customer has phone service (Yes/No).")
    MultipleLines: str = Field(..., description="Whether the customer has multiple lines (Yes/No/No phone service).")
    InternetService: str = Field(..., description="Customer's internet service provider (DSL/Fiber optic/No).")
    OnlineSecurity: str = Field(..., description="Whether the customer has online security (Yes/No/No internet service).")
    OnlineBackup: str = Field(..., description="Whether the customer has online backup (Yes/No/No internet service).")
    DeviceProtection: str = Field(..., description="Whether the customer has device protection (Yes/No/No internet service).")
    TechSupport: str = Field(..., description="Whether the customer has tech support (Yes/No/No internet service).")
    StreamingTV: str = Field(..., description="Whether the customer has streaming TV (Yes/No/No internet service).")
    StreamingMovies: str = Field(..., description="Whether the customer has streaming movies (Yes/No/No internet service).")
    Contract: str = Field(..., description="The customer's contract type (Month-to-month/One year/Two year).")
    PaperlessBilling: str = Field(..., description="Whether the customer has paperless billing (Yes/No).")
    PaymentMethod: str = Field(..., description="The customer's payment method (Electronic check/Mailed check/Bank transfer (automatic)/Credit card (automatic)).")

    class Config:
        extra = "forbid" # Ensures only defined fields are accepted

@app.post("/predict")
async def predict_churn(request: ChurnPredictRequest) -> Dict[str, Any]:
    """
    Endpoint to predict customer churn based on input features.
    """
    if not hasattr(app.state, "model_pipeline") or app.state.model_pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    input_data_dict = request.dict()

    # Combine all feature columns in the correct order as per your config.py
    all_features = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS
    try:
        # Create a DataFrame from the input data, ensuring correct column order
        input_df = pd.DataFrame([input_data_dict], columns=all_features)
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Missing expected input feature: {e}. Please provide all required features.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing input data: {e}")

    # Handle 'TotalCharges' as it might come in as string and needs numeric conversion
    if 'TotalCharges' in input_df.columns:
        # If 'TotalCharges' contains spaces or is empty string, convert to NaN, then fill with 0
        # This handles cases where 'TotalCharges' is ' ' for new customers with 0 tenure
        input_df['TotalCharges'] = pd.to_numeric(input_df['TotalCharges'], errors='coerce').fillna(0)

    try: # Access the model from app.state
        prediction = app.state.model_pipeline.predict(input_df)
        prediction_proba = app.state.model_pipeline.predict_proba(input_df)

        churn_status = "Churn" if prediction[0] == 1 else "No Churn"
        churn_probability = float(prediction_proba[0][1]) # Probability of churn (class 1)
        no_churn_probability = float(prediction_proba[0][0]) # Probability of no churn (class 0)

        return {
            "prediction": churn_status,
            "churn_probability": churn_probability,
            "no_churn_probability": no_churn_probability,
            "probabilities": {
                "No Churn": no_churn_probability,
                "Churn": churn_probability
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}. Ensure input data types and values are correct.")