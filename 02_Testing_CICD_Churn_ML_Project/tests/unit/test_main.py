"""
Objective:
    This script contains unit tests for the main entrypoint script `main.py`,
    implemented using Python's built-in `unittest` framework.
    It verifies that the main function correctly orchestrates the pipeline
    execution based on its input parameters.

Tests Performed:
    - test_main_success:
        - Mocks the `run_churn_pipeline` function to isolate the `main` function's logic.
        - Calls `main` with a specified output directory.
        - Asserts that `run_churn_pipeline` is called exactly once with the correctly
          constructed file paths and configuration values.
    - test_main_pipeline_error_handling:
        - Simulates an error occurring within `run_churn_pipeline`.
        - Asserts that the `main` function correctly catches and re-raises the
          exception, ensuring proper error propagation.
"""
import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
import os
import tempfile
import sys

# Add project root to sys.path for import resolution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from main import main
from src.config import (
    TARGET_COLUMN, NUMERIC_COLUMNS, CATEGORICAL_COLUMNS,
    TEST_SIZE, RANDOM_STATE, MODEL_FILENAME, LOG_FILENAME,
    DATA_DIR_NAME, RAW_DATA_DIR_NAME, DATASET_FILENAME, MODEL_STORE_DIR
)

class TestMainFunction(unittest.TestCase):

    def setUp(self):
        """Set up a temporary directory for test artifacts."""
        # We manually manage the TemporaryDirectory to use it across the test method.
        self.tmpdir_context = tempfile.TemporaryDirectory()
        self.output_dir = Path(self.tmpdir_context.name)

    def tearDown(self):
        """Clean up the temporary directory after the test."""
        self.tmpdir_context.cleanup()

    @patch('main.run_churn_pipeline')
    def test_main_success(self, mock_run_pipeline):
        """
        Test that `main` correctly constructs paths and calls the pipeline
        when an output directory is provided.
        """
        dummy_model = MagicMock()
        dummy_metrics = {"accuracy": 0.92}
        mock_run_pipeline.return_value = (dummy_model, dummy_metrics)

        with patch('builtins.print'): # Suppress print statements for a cleaner test output
            main(output_base_dir=self.output_dir)

        expected_data_path = str(self.output_dir / DATA_DIR_NAME / RAW_DATA_DIR_NAME / DATASET_FILENAME)
        expected_model_store_path = str(self.output_dir / MODEL_STORE_DIR)

        mock_run_pipeline.assert_called_once_with(
            data_file_path=expected_data_path,
            target_column=TARGET_COLUMN,
            numeric_columns=NUMERIC_COLUMNS,
            categorical_columns=CATEGORICAL_COLUMNS,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            model_dir_path=expected_model_store_path,
            model_filename=MODEL_FILENAME,
            log_filename=LOG_FILENAME
        )

    @patch('main.run_churn_pipeline', side_effect=ValueError("Simulated pipeline error"))
    def test_main_pipeline_error_handling(self, mock_run_pipeline):
        """Test that `main` propagates exceptions raised from the pipeline."""
        # We expect the ValueError from the mock to be re-raised by main()
        with self.assertRaisesRegex(ValueError, "Simulated pipeline error"):
            main(output_base_dir=self.output_dir)

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False, verbosity=2) # exit=False prevents sys.exit() from being called
