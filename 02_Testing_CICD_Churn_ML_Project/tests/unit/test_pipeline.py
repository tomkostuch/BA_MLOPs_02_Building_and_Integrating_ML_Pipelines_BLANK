# ml_project/tests/test_pipeline.py
"""
Objective:
    This script contains an integration-style test for the `src/pipeline.py` module,
    now implemented using Python's built-in `unittest` framework.
    It focuses on verifying the correct orchestration and data flow of the entire ML pipeline.

Tests Performed:
    - test_pipeline_unit_orchestration_success:
        - Mocks *all* external and internal function calls to isolate the
          `run_churn_pipeline` function.
        - Verifies that each step of the pipeline is called in the correct order.
        - Asserts that the output of each step is correctly passed as input
          to the subsequent step, confirming the data handoff logic.
        - Verifies that the `run_churn_pipeline` function returns the trained model
          and evaluation metrics as expected.
        - Confirms that the model and log saving methods are invoked.
    - test_run_churn_pipeline_error_handling:
        - Simulates an error at an early stage in the pipeline (e.g., data loading).
        - Asserts that the `run_churn_pipeline` correctly propagates or re-raises the error.
"""
import unittest
import pandas as pd
import numpy as np
import os
from unittest.mock import patch, MagicMock
import sys
import tempfile
import shutil

# Add the project root to sys.path to allow imports from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Import functions to be tested (and internal functions that will now run unmocked)
from src.pipeline import run_churn_pipeline
from src.model import ChurnPredictionModel # Import the actual class for spec
from sklearn.compose import ColumnTransformer

from src.config import (
    TARGET_COLUMN, NUMERIC_COLUMNS, CATEGORICAL_COLUMNS,
    TEST_SIZE, RANDOM_STATE, MODEL_FILENAME, LOG_FILENAME
)


class TestPipeline(unittest.TestCase):

    def setUp(self):
        # Create a temporary directory for test outputs
        self.tmp_dir = tempfile.mkdtemp()
        self.data_path = os.path.join(self.tmp_dir, "test.csv")
        self.model_store_path = os.path.join(self.tmp_dir, "model_store")

        # The pipeline expects the model store directory to exist.
        os.makedirs(self.model_store_path, exist_ok=True)

    def tearDown(self):
        # Clean up the temporary directory
        shutil.rmtree(self.tmp_dir)

    # --- Unit Test: Mock all dependencies to test orchestration ---
    @patch('src.pipeline.report_classification_metrics')
    @patch('src.pipeline.compute_classification_metrics')
    @patch('src.pipeline.ChurnPredictionModel')
    @patch('src.pipeline.stratified_split')
    @patch('src.pipeline.split_features_and_target')
    @patch('src.pipeline.transform_features')
    @patch('src.pipeline.clean_churn_data')
    @patch('src.pipeline.load_churn_dataset')
    def test_pipeline_unit_orchestration_success(
        self,
        mock_load_data,
        mock_clean_data,
        mock_transform_features,
        mock_split_features,
        mock_stratified_split,
        mock_churn_model_class,
        mock_compute_metrics,
        mock_report_metrics,
    ):
        """
        Test the successful orchestration flow of the pipeline as a pure unit test.
        """
        # Arrange: Create mock objects for each stage's output
        mock_raw_df = MagicMock(spec=pd.DataFrame)
        mock_clean_df = MagicMock(spec=pd.DataFrame)
        mock_transformed_df = MagicMock(spec=pd.DataFrame)
        mock_preprocessor = MagicMock(spec=ColumnTransformer)
        mock_X, mock_y = MagicMock(spec=pd.DataFrame), MagicMock(spec=pd.Series)
        mock_X_train, mock_X_test = MagicMock(spec=pd.DataFrame), MagicMock(spec=pd.DataFrame)
        mock_y_train, mock_y_test = MagicMock(spec=pd.Series), MagicMock(spec=pd.Series)

        # Configure mocks to handle method calls inside the pipeline that would otherwise fail
        mock_y_train.mean.return_value = 0.25  # Return a float for the .mean() call
        mock_y_test.mean.return_value = 0.25
        mock_y_test.to_numpy.return_value = np.array([0, 1, 1, 0]) # Return a numpy array for the .to_numpy() call

        # Arrange: Define the return values for the mocked functions
        mock_load_data.return_value = mock_raw_df
        mock_clean_data.return_value = mock_clean_df
        mock_transform_features.return_value = (mock_transformed_df, mock_preprocessor)
        mock_split_features.return_value = (mock_X, mock_y)
        mock_stratified_split.return_value = (mock_X_train, mock_X_test, mock_y_train, mock_y_test)

        # Arrange: Set up the mock model *instance* returned by the mocked class
        mock_model_instance = MagicMock(spec=ChurnPredictionModel)
        mock_model_instance.predict.return_value = np.array([0, 1])
        mock_churn_model_class.return_value = mock_model_instance # What the constructor returns

        # Arrange: Set up mock for compute_classification_metrics
        mock_compute_metrics.return_value = {"accuracy": 0.95, "f1_score": 0.90}

        # Act
        returned_model, returned_metrics = run_churn_pipeline(
            data_file_path=self.data_path,
            target_column=TARGET_COLUMN,
            numeric_columns=NUMERIC_COLUMNS,
            categorical_columns=CATEGORICAL_COLUMNS,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            model_dir_path=self.model_store_path,
            model_filename=MODEL_FILENAME,
            log_filename=LOG_FILENAME
        )

        # Assert: Verify the orchestration and data handoffs
        mock_load_data.assert_called_once_with(self.data_path)
        mock_clean_data.assert_called_once_with(mock_raw_df, TARGET_COLUMN, NUMERIC_COLUMNS, CATEGORICAL_COLUMNS)
        mock_transform_features.assert_called_once_with(mock_clean_df, TARGET_COLUMN, NUMERIC_COLUMNS, CATEGORICAL_COLUMNS)
        mock_split_features.assert_called_once_with(mock_transformed_df)
        mock_stratified_split.assert_called_once_with(mock_X, mock_y, test_size=TEST_SIZE, seed=RANDOM_STATE)

        # Assert model initialization and training
        mock_churn_model_class.assert_called_once_with(preprocessor=mock_preprocessor, random_state=RANDOM_STATE)
        mock_model_instance.fit.assert_called_once_with(mock_X_train, mock_y_train)

        # Assert prediction and evaluation
        mock_model_instance.predict.assert_called_once_with(mock_X_test)
        mock_compute_metrics.assert_called_once()
        # Check that the correct y_true and y_pred were passed to metrics
        y_true_arg = mock_compute_metrics.call_args[0][0]
        y_pred_arg = mock_compute_metrics.call_args[0][1]
        self.assertIs(y_true_arg, mock_y_test.to_numpy.return_value)
        self.assertIs(y_pred_arg, mock_model_instance.predict.return_value)

        mock_report_metrics.assert_called_once_with(mock_compute_metrics.return_value)

        # Assert artifacts were saved
        mock_model_instance.save.assert_called_once_with(os.path.join(self.model_store_path, MODEL_FILENAME))
        mock_model_instance.log_run.assert_called_once()

        self.assertIs(returned_model, mock_model_instance)
        self.assertEqual(returned_metrics, {"accuracy": 0.95, "f1_score": 0.90})

    def test_run_churn_pipeline_error_handling(self):
        """Test that run_churn_pipeline correctly handles exceptions."""
        with patch('src.pipeline.load_churn_dataset', side_effect=RuntimeError("Test load error")):
            with self.assertRaisesRegex(RuntimeError, "Test load error"):
                run_churn_pipeline(
                    data_file_path="dummy_path.csv",
                    target_column=TARGET_COLUMN,
                    numeric_columns=NUMERIC_COLUMNS,
                    categorical_columns=CATEGORICAL_COLUMNS,
                    test_size=TEST_SIZE,
                    random_state=RANDOM_STATE,
                    model_dir_path=self.model_store_path, # Use self.model_store_path from setUp
                    model_filename=MODEL_FILENAME,
                    log_filename=LOG_FILENAME
                )

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False, verbosity=2) # exit=False prevents sys.exit() from being called