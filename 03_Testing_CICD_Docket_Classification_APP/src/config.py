"""
config.py

This script stores configuration variables and constants used across the ML project.
It includes definitions for column names, test sizes, random states, and file names
for models and logs, and paths to datasets.
"""
import os

# Define column structure based on the churn dataset
TARGET_COLUMN = 'Churn'  # Target variable
NUMERIC_COLUMNS = ['tenure', 'MonthlyCharges', 'TotalCharges']
CATEGORICAL_COLUMNS = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents',
    'PhoneService', 'MultipleLines', 'InternetService',
    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'TechSupport', 'StreamingTV', 'StreamingMovies',
    'Contract', 'PaperlessBilling', 'PaymentMethod'
]

# Define file and directory names
MODEL_FILENAME = "churn_prediction_model_v1.joblib"
LOG_FILENAME = "churn_model_run_log.json"
MODEL_STORE_DIR = "model_store"
