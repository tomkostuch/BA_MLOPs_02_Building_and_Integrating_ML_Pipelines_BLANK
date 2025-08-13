# ml_project/tests/test_data_loader.py
"""
Objective:
    This script contains unit tests for the functions in `src/data_loader.py`,
    now implemented using Python's built-in `unittest` framework.
    It focuses on verifying the correct loading of data and proper error handling.

Tests Performed:
    - test_load_churn_dataset_success: Verifies that a well-formed CSV file is loaded
      correctly into a pandas DataFrame with expected properties.
    - test_load_churn_dataset_file_not_found: Checks that a RuntimeError is raised
      when attempting to load a non-existent file.
    - test_load_churn_dataset_empty_csv: Ensures that an empty CSV file (with headers)
      is handled gracefully, resulting in an empty DataFrame with correct columns.
"""
import unittest
import pandas as pd
import os
import sys
import tempfile
import shutil

# Add the project root to sys.path to allow imports from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Import the function to be tested
from src.data_loader import load_churn_dataset

class TestDataLoader(unittest.TestCase):

    def setUp(self):
        # Create a temporary directory for test files
        self.tmp_dir = tempfile.mkdtemp()
        
        # Create a dummy CSV file
        self.dummy_csv_filepath = os.path.join(self.tmp_dir, "test_churn_data.csv")
        dummy_data = """customerID,gender,SeniorCitizen,Partner,Dependents,tenure,PhoneService,MultipleLines,InternetService,OnlineSecurity,OnlineBackup,DeviceProtection,TechSupport,StreamingTV,StreamingMovies,Contract,PaperlessBilling,PaymentMethod,MonthlyCharges,TotalCharges,Churn
7590-VHVEG,Female,0,Yes,No,1,No,No phone service,DSL,No,Yes,No,No,No,No,Month-to-month,Yes,Electronic check,29.85,29.85,No
5575-GNVDE,Male,0,No,No,34,Yes,No,DSL,Yes,No,Yes,No,No,No,One year,No,Mailed check,56.95,1889.5,No
3668-QPYAX,Male,0,No,No,2,Yes,No,DSL,Yes,Yes,No,No,No,No,Month-to-month,Yes,Mailed check,53.85,108.15,Yes
"""
        with open(self.dummy_csv_filepath, 'w') as f:
            f.write(dummy_data)

        # Create an empty CSV file with only headers
        self.empty_csv_filepath = os.path.join(self.tmp_dir, "empty_churn_data.csv")
        empty_data = """customerID,gender,SeniorCitizen,Partner,Dependents,tenure,PhoneService,MultipleLines,InternetService,OnlineSecurity,OnlineBackup,DeviceProtection,TechSupport,StreamingTV,StreamingMovies,Contract,PaperlessBilling,PaymentMethod,MonthlyCharges,TotalCharges,Churn
"""
        with open(self.empty_csv_filepath, 'w') as f:
            f.write(empty_data)

    def tearDown(self):
        # Clean up the temporary directory
        shutil.rmtree(self.tmp_dir)

    def test_load_churn_dataset_success(self):
        """Test if the dataset is loaded successfully as a DataFrame."""
        df = load_churn_dataset(self.dummy_csv_filepath)

        self.assertIsInstance(df, pd.DataFrame)
        self.assertFalse(df.empty)
        self.assertEqual(len(df), 3)
        self.assertIn('customerID', df.columns)
        self.assertIn('Churn', df.columns)
        self.assertIn('TotalCharges', df.columns)

    def test_load_churn_dataset_file_not_found(self):
        """Test if RuntimeError is raised for a non-existent file."""
        non_existent_file = os.path.join(self.tmp_dir, "non_existent_file.csv")
        with self.assertRaisesRegex(RuntimeError, "Failed to load churn dataset: .*No such file or directory.*"):
            load_churn_dataset(non_existent_file)

    def test_load_churn_dataset_empty_csv(self):
        """Test loading an empty CSV file."""
        df = load_churn_dataset(self.empty_csv_filepath)

        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 0)
        self.assertIn('customerID', df.columns) # Should still have columns from header

# This block allows you to run the tests directly from the script
if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False, verbosity=2) # exit=False prevents sys.exit() from being called