# ml_project/tests/integration/test_data_preprocessing_integration.py
"""
Objective:
    Integration test for the data loading and initial preprocessing steps,
    now implemented using Python's built-in `unittest` framework.
    Verifies that the output of `data_loader` is correctly handled by `preprocessing.clean_churn_data`.

Tests Performed:
    - test_data_loading_and_cleaning_flow:
        Loads a representative dummy CSV, passes it to the cleaning function,
        and asserts on key characteristics of the resulting cleaned DataFrame.
"""
import unittest
import pandas as pd
import numpy as np
import os
import sys
import tempfile
import shutil

# Add the project root to sys.path to allow imports from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Import functions from src
from src.data_loader import load_churn_dataset
from src.preprocessing import clean_churn_data
from src.config import TARGET_COLUMN, NUMERIC_COLUMNS, CATEGORICAL_COLUMNS

class TestDataPreprocessingIntegration(unittest.TestCase):

    def setUp(self):
        # Create a temporary directory for test files
        self.tmp_dir = tempfile.mkdtemp()
        self.representative_raw_csv = os.path.join(self.tmp_dir, "representative_raw_churn_data.csv")

        # Creates a dummy CSV file that simulates the raw churn data.
        data = """customerID,gender,SeniorCitizen,Partner,Dependents,tenure,PhoneService,MultipleLines,InternetService,OnlineSecurity,OnlineBackup,DeviceProtection,TechSupport,StreamingTV,StreamingMovies,Contract,PaperlessBilling,PaymentMethod,MonthlyCharges,TotalCharges,Churn
7590-VHVEG,Female,0,Yes,No,1,No,No phone service,DSL,No,Yes,No,No,No,No,Month-to-month,Yes,Electronic check,29.85,29.85,No
5575-GNVDE,Male,0,No,No,34,Yes,No,DSL,Yes,No,Yes,No,No,No,One year,No,Mailed check,56.95,1889.5,No
3668-QPYAX,Male,0,No,No,2,Yes,No,Fiber optic,Yes,Yes,No,No,No,No,Month-to-month,Yes,Electronic check,53.85,108.15,Yes
9237-HQITU,Female,0,No,No,2,Yes,No,Fiber optic,No,No,No,No,No,No,Month-to-month,Yes,Electronic check,70.70,,No
9305-CDHLH,Female,0,No,No,8,Yes,Yes,Fiber optic,No,No,Yes,No,Yes,Yes,Month-to-month,Yes,Electronic check,99.65,820.5,Yes
"""
        with open(self.representative_raw_csv, 'w') as f:
            f.write(data)

    def tearDown(self):
        # Clean up the temporary directory
        shutil.rmtree(self.tmp_dir)

    def test_data_loading_and_cleaning_flow(self):
        """
        Integration test: Verifies the end-to-end flow from loading raw data
        to obtaining a cleaned DataFrame.
        """
        print(f"\n--- Integration Test: Data Loading and Cleaning Flow from {self.representative_raw_csv} ---")

        # 1. Load data using data_loader
        raw_df = load_churn_dataset(self.representative_raw_csv)

        self.assertIsInstance(raw_df, pd.DataFrame)
        self.assertFalse(raw_df.empty)
        print(f"Raw DF loaded with shape: {raw_df.shape}")

        # 2. Clean data using preprocessing
        cleaned_df = clean_churn_data(
            raw_df.copy(), # Pass a copy to avoid modifying the original
            TARGET_COLUMN,
            NUMERIC_COLUMNS,
            CATEGORICAL_COLUMNS
        )

        # Assertions on the cleaned_df
        self.assertIsInstance(cleaned_df, pd.DataFrame)
        self.assertIn('churn_binary', cleaned_df.columns)
        self.assertTrue(pd.api.types.is_integer_dtype(cleaned_df['churn_binary']))

        # Check TotalCharges NaN handling and type conversion
        self.assertTrue(pd.api.types.is_numeric_dtype(cleaned_df['TotalCharges']))
        self.assertEqual(cleaned_df['TotalCharges'].isnull().sum(), 1) # One NaN due to '' in original data

        # Ensure no unexpected columns and all required are present
        expected_cols = set(NUMERIC_COLUMNS + CATEGORICAL_COLUMNS + ['churn_binary'])
        self.assertEqual(set(cleaned_df.columns), expected_cols)
        print(f"Cleaned DF has expected columns and shape: {cleaned_df.shape}")

        # Check a specific value for cleaning using original row indices
        # Ensure 'TotalCharges' for the row with original empty string is now NaN
        self.assertTrue(pd.isna(cleaned_df.loc[3, 'TotalCharges'])) # 4th row (index 3) had empty TotalCharges

        # Verify 'SeniorCitizen' is handled as categorical (0 or 1, but its content remains as such)
        # It's an integer but treated as categorical in config; clean_churn_data doesn't change its type.
        self.assertTrue(pd.api.types.is_integer_dtype(cleaned_df['SeniorCitizen']))
        self.assertTrue(all(cleaned_df['SeniorCitizen'].isin([0, 1])))

        print("--- Integration Test Complete ---")

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False, verbosity=2) # exit=False prevents sys.exit() from being called