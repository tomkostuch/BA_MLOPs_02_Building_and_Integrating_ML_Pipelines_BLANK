# ml_project/tests/test_main_functional.py
"""
Objective:
    Functional tests for the main application entry point (`main.py`),
    implemented using Python's built-in `unittest` framework.
    These tests verify that the `main` function correctly orchestrates the
    ML pipeline, handles its outputs, and manages exceptions gracefully.

Tests Performed:
    - test_main_orchestrates_pipeline_and_reports_success:
        Verifies that `main` calls the pipeline with correctly constructed
        paths and prints a success message using the pipeline's results.
    - test_main_handles_pipeline_exceptions_gracefully:
        Verifies that `main` catches exceptions from the pipeline, prints
        an error message to stderr, and re-raises the exception.
"""
import unittest
from unittest.mock import patch, MagicMock
import os
import sys
from pathlib import Path
import io # For capturing stdout/stderr

# Add the project root to sys.path to allow imports from src and main
# Assumes test_main_functional.py is in 'ml_project/tests/'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Import the main function to be tested
from main import main
from src.config import (
    TARGET_COLUMN, TEST_SIZE, DATA_DIR_NAME, RAW_DATA_DIR_NAME,
    DATASET_FILENAME, MODEL_STORE_DIR
)


class TestMainFunctional(unittest.TestCase):

    # These patches are applied to each test method in this class.
    # The 'return_value' for os.path mocks helps simulate a consistent project root.
    @patch('main.run_churn_pipeline')
    @patch('os.path.abspath', return_value='/mock/project/path/main.py')
    @patch('os.path.dirname', return_value='/mock/project/path')
    def test_main_orchestrates_pipeline_and_reports_success(
        self,
        mock_dirname, # Mocks from decorators are passed as arguments
        mock_abspath,
        mock_run_churn_pipeline
    ):
        """
        Functional test to verify `main` correctly orchestrates the pipeline
        and reports success based on the pipeline's output.
        """
        print("\n--- Functional Test: Main orchestrates pipeline successfully ---")
        # --- Arrange ---
        # Define the mock return value for the entire pipeline run
        dummy_model = MagicMock()
        dummy_metrics = {"accuracy": 0.95123}
        mock_run_churn_pipeline.return_value = (dummy_model, dummy_metrics)

        # Use patch as a context manager to capture stdout/stderr cleanly
        with patch('sys.stdout', new_callable=io.StringIO) as mock_stdout, \
             patch('sys.stderr', new_callable=io.StringIO) as mock_stderr:
            # --- Act ---
            main()

            # --- Assert ---
            # 1. Verify that the pipeline was called once with correctly constructed paths
            mock_run_churn_pipeline.assert_called_once()

            # Inspect the keyword arguments of the call
            _, kwargs = mock_run_churn_pipeline.call_args
            expected_data_path = str(Path('/mock/project/path') / DATA_DIR_NAME / RAW_DATA_DIR_NAME / DATASET_FILENAME)
            expected_model_dir = str(Path('/mock/project/path') / MODEL_STORE_DIR)

            self.assertEqual(kwargs['data_file_path'], expected_data_path)
            self.assertEqual(kwargs['model_dir_path'], expected_model_dir)
            self.assertEqual(kwargs['target_column'], TARGET_COLUMN)
            self.assertEqual(kwargs['test_size'], TEST_SIZE)
            print("Pipeline called with correct arguments.")

            # 2. Verify console output for success messages
            stdout_output = mock_stdout.getvalue()
            stderr_output = mock_stderr.getvalue()

            self.assertIn("Starting Customer Churn Prediction Pipeline...", stdout_output)
            self.assertIn("Pipeline completed successfully!", stdout_output)
            self.assertIn(f"Final Model Accuracy: {dummy_metrics['accuracy']:.4f}", stdout_output)
            self.assertEqual(stderr_output, "") # Ensure no errors were printed to stderr
            print("Console output verified for success messages.")
        print("--- Functional Test Complete ---")


    @patch('main.run_churn_pipeline')
    @patch('os.path.abspath', return_value='/mock/project/path/main.py')
    @patch('os.path.dirname', return_value='/mock/project/path')
    def test_main_handles_pipeline_exceptions_gracefully(
        self,
        mock_dirname,
        mock_abspath,
        mock_run_churn_pipeline
    ):
        """
        Functional test to verify `main` catches exceptions from the pipeline,
        logs an error, and re-raises the exception.
        """
        print("\n--- Functional Test: Main handles pipeline exceptions gracefully ---")
        # --- Arrange ---
        # Configure the mock pipeline to raise an exception
        error_message = "Something went wrong during training"
        mock_run_churn_pipeline.side_effect = ValueError(error_message)

        # Use patch as a context manager to capture stdout/stderr cleanly
        with patch('sys.stdout', new_callable=io.StringIO) as mock_stdout, \
             patch('sys.stderr', new_callable=io.StringIO) as mock_stderr:

            # --- Act & Assert ---
            # Verify that the exception is caught and then re-raised by main
            with self.assertRaisesRegex(ValueError, error_message):
                main()

            # Verify that an error message was printed to stderr
            stdout_output = mock_stdout.getvalue()
            stderr_output = mock_stderr.getvalue()

            self.assertIn(f"ERROR: Pipeline failed with exception: {error_message}", stderr_output)
            self.assertNotIn("Pipeline completed successfully!", stdout_output)
            print("Error message printed to stderr.")
            print("Exception correctly re-raised.")
        print("--- Functional Test Complete ---")


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False, verbosity=2) # exit=False prevents sys.exit() from being called
