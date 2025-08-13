# ml_project/tests/e2e_test.py
"""
Objective:
    End-to-end test for the main application pipeline (`main.py`),
    implemented using Python's built-in `unittest` framework.
    This test verifies the entire ML pipeline from data loading to model saving
    and log creation, using a real (albeit tiny) dataset.

Tests Performed:
    - test_main_pipeline_end_to_end:
        - Creates a temporary directory structure simulating the project.
        - Prepares a small dummy dataset within this structure.
        - Calls the `main` function to run the full pipeline.
        - Asserts that the model file and log file are correctly created
          in the expected output directory.
        - Verifies that the logged metrics are valid (e.g., accuracy, f1_score
          are within reasonable bounds).
"""
import unittest
import os
import shutil
import json
import joblib # Used for loading/saving models, not directly tested but part of flow
from pathlib import Path
import sys
import tempfile # For creating temporary directories

# Add the project root to sys.path to allow imports from src and main
# Assumes e2e_test.py is in 'ml_project/tests/'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from main import main
from src.config import LOG_FILENAME, MODEL_FILENAME, DATASET_FILENAME, DATA_DIR_NAME, RAW_DATA_DIR_NAME


class TestE2EPipeline(unittest.TestCase):

    def setUp(self):
        """
        Prepare a temporary directory to simulate the project root
        and create a tiny dataset in the expected structure:
        tmp_project_root/data/raw/churn_data.csv
        """
        # Create a temporary directory for the entire simulated project
        self.tmp_project_root = Path(tempfile.mkdtemp())

        # Create required data directory structure within the temporary root
        self.raw_data_dir = self.tmp_project_root / DATA_DIR_NAME / RAW_DATA_DIR_NAME
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)

        # Create a tiny valid dataset
        sample_data = """customerID,gender,SeniorCitizen,Partner,Dependents,tenure,PhoneService,MultipleLines,InternetService,OnlineSecurity,OnlineBackup,DeviceProtection,TechSupport,StreamingTV,StreamingMovies,Contract,PaperlessBilling,PaymentMethod,MonthlyCharges,TotalCharges,Churn
7590-VHVEG,Female,0,Yes,No,1,No,No phone service,DSL,No,Yes,No,No,No,No,Month-to-month,Yes,Electronic check,29.85,29.85,No
5575-GNVDE,Male,0,No,No,34,Yes,No,DSL,Yes,No,Yes,No,No,No,One year,No,Mailed check,56.95,1889.5,No
3668-QPYAX,Male,0,No,No,2,Yes,No,Fiber optic,Yes,Yes,No,No,No,No,Month-to-month,Yes,Electronic check,53.85,108.15,Yes
9237-HQITU,Female,0,No,No,2,Yes,No,Fiber optic,No,No,No,No,No,No,Month-to-month,Yes,Electronic check,70.70,,No
9305-CDHLH,Female,0,No,No,8,Yes,Yes,Fiber optic,No,No,Yes,No,Yes,Yes,Month-to-month,Yes,Electronic check,99.65,820.5,Yes
"""
        # Use the filename from the config to ensure consistency with the pipeline
        self.sample_file_path = self.raw_data_dir / DATASET_FILENAME
        self.sample_file_path.write_text(sample_data)

        print(f"\n--- E2E Test: Temporary project root set up at: {self.tmp_project_root} ---")
        print(f"Sample data created at: {self.sample_file_path}")


    def tearDown(self):
        """Clean up the temporary directory and restore original working directory."""
        if os.path.exists(self.tmp_project_root):
            shutil.rmtree(self.tmp_project_root)
            print(f"--- E2E Test: Cleaned up temporary directory: {self.tmp_project_root} ---")

    def test_main_pipeline_end_to_end(self):
        """
        End-to-end test for the main training pipeline using a real dataset.
        Verifies model and log file creation, and that metrics are valid.
        """
        print("\n--- Running End-to-End Pipeline Test ---")

        # Run the pipeline (main function)
        # Pass the temporary directory as the output base to ensure files are saved there
        main(output_base_dir=self.tmp_project_root)

        # Expected output file paths (relative to self.tmp_project_root)
        model_path = self.tmp_project_root / "model_store" / MODEL_FILENAME
        log_path = self.tmp_project_root / "model_store" / LOG_FILENAME

        # Assert files were created
        self.assertTrue(model_path.exists(), f"Model file was not saved at {model_path}")
        self.assertTrue(log_path.exists(), f"Log file was not saved at {log_path}")
        print("Model and log files confirmed to exist.")

        # Load and verify log content
        with open(log_path, "r") as f:
            metrics_log = json.load(f)
            # The log file contains a list of run details, each with a 'metrics' key
            self.assertIsInstance(metrics_log, list, "Log file content should be a list")
            self.assertGreater(len(metrics_log), 0, "Log file should contain at least one run detail")
            
            first_run_metrics = metrics_log[0].get("metrics")
            self.assertIsNotNone(first_run_metrics, "First run detail should contain 'metrics' key")

            # Assert specific metrics are present and have reasonable values
            self.assertIn("accuracy", first_run_metrics)
            self.assertIsInstance(first_run_metrics["accuracy"], float)
            self.assertGreaterEqual(first_run_metrics["accuracy"], 0.0)
            self.assertLessEqual(first_run_metrics["accuracy"], 1.0)
            print("Logged metrics verified (accuracy range).")

            self.assertIn("f1_score", first_run_metrics)
            self.assertIsInstance(first_run_metrics["f1_score"], float)
            self.assertGreaterEqual(first_run_metrics["f1_score"], 0.0)
            self.assertLessEqual(first_run_metrics["f1_score"], 1.0)
            print("Logged metrics verified (f1_score range).")

            self.assertIn("confusion_matrix", first_run_metrics)
            self.assertIsInstance(first_run_metrics["confusion_matrix"], list)
            self.assertEqual(len(first_run_metrics["confusion_matrix"]), 2)
            self.assertEqual(len(first_run_metrics["confusion_matrix"][0]), 2)
            print("Logged metrics verified (confusion matrix).")

        print("--- End-to-End Pipeline Test Complete ---")


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False, verbosity=2) # exit=False prevents sys.exit() from being called