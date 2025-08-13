"""
Objective:
    This script contains unit tests for the `src/model.py` module, implemented
    using Python's built-in `unittest` framework. It ensures that the
    `ChurnPredictionModel` class and its associated metric functions operate correctly.

Tests Performed:
    - TestChurnPredictionModel:
        - `test_model_init_defaults`: Verifies that the model initializes with the
          correct default classifier (LogisticRegression) and pipeline structure.
        - `test_model_fit_predict`: Confirms that the model can be trained and can
          produce predictions and probabilities of the correct shape and format.
        - `test_model_save_and_load`: Ensures that the model's `save` method
          creates a file and that a loaded model retains its predictive ability.
        - `test_model_log_run`: Checks that model run information, including metrics
          and parameters, is correctly logged to a JSON file.
    - TestMetricsFunctions:
        - `test_compute_classification_metrics`: Validates the accuracy of metric
          calculations for both perfect and imperfect prediction scenarios.
        - `test_report_classification_metrics`: Asserts that the metric reporting
          function generates the expected console output.
"""
import unittest
import pandas as pd
import numpy as np
import os
import json
import joblib
from unittest.mock import patch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import sys

# Allow imports from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.model import ChurnPredictionModel, compute_classification_metrics, report_classification_metrics

class TestChurnPredictionModel(unittest.TestCase):
    """Unit tests for the ChurnPredictionModel class."""

    def setUp(self):
        """Set up a dummy dataset and a simple preprocessor for use in tests."""
        self.X = pd.DataFrame(np.random.rand(100, 5), columns=[f'feature_{i}' for i in range(5)])
        self.y = pd.Series(np.random.randint(0, 2, 100))
        self.preprocessor = StandardScaler()

    def test_model_init_defaults(self):
        """Test that the model initializes with correct default components."""
        model = ChurnPredictionModel()
        self.assertIsInstance(model.classifier, LogisticRegression)
        self.assertIn('scaler', model.pipe.named_steps)
        self.assertIn('classifier', model.pipe.named_steps)

    def test_model_fit_predict(self):
        """Test the model's fit, predict, and predict_proba methods."""
        model = ChurnPredictionModel(preprocessor=self.preprocessor)
        model.fit(self.X, self.y)
        preds = model.predict(self.X)
        probs = model.predict_proba(self.X)
        self.assertEqual(preds.shape, self.y.shape)
        self.assertTrue(np.all(np.isin(preds, [0, 1])))
        self.assertEqual(probs.shape, (len(self.X), 2))
        self.assertTrue(np.all((probs >= 0) & (probs <= 1)))
        self.assertTrue(np.allclose(np.sum(probs, axis=1), 1))

    def test_model_save_and_load(self):
        """Test that the model can be saved and loaded correctly."""
        model = ChurnPredictionModel(preprocessor=self.preprocessor)
        model.fit(self.X, self.y)
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.joblib")
            model.save(path)
            self.assertTrue(os.path.exists(path))
            loaded_model = joblib.load(path)
            self.assertIsInstance(loaded_model, Pipeline)
            self.assertEqual(loaded_model.predict(self.X).shape, self.y.shape)

    def test_model_log_run(self):
        """Test the logging of a model run to a JSON file."""
        model = ChurnPredictionModel()
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as tmpdir:
            log_dir = os.path.join(tmpdir, "logs") # The log_run method should create this directory
            log_file = os.path.join(log_dir, "log.json")
            metrics = {"accuracy": 0.85}
            dataset_info = {"samples": 100}
            model.log_run(log_dir, metrics, dataset_info, log_filename="log.json")
            self.assertTrue(os.path.exists(log_file))
            with open(log_file, 'r') as f:
                logs = json.load(f)
            self.assertIsInstance(logs, list)
            self.assertEqual(logs[0]["metrics"], metrics)

class TestMetricsFunctions(unittest.TestCase):
    """Unit tests for the metric computation and reporting functions."""

    def test_compute_classification_metrics_perfect(self):
        """Test metric computation with perfect predictions."""
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])
        metrics = compute_classification_metrics(y_true, y_pred)
        self.assertEqual(metrics['accuracy'], 1.0)
        self.assertEqual(metrics['precision'], 1.0)
        self.assertEqual(metrics['recall'], 1.0)
        self.assertEqual(metrics['f1_score'], 1.0)

    def test_compute_classification_metrics_imperfect(self):
        """Test metric computation with imperfect predictions."""
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 0, 0, 1])
        metrics = compute_classification_metrics(y_true, y_pred)
        self.assertEqual(metrics['accuracy'], 0.75)
        self.assertEqual(metrics['precision'], 1.0)
        self.assertEqual(metrics['recall'], 0.5)

    def test_report_classification_metrics_output(self):
        """Test that the metrics report is printed correctly."""
        metrics = {
            "accuracy": 0.85,
            "precision": 0.75,
            "recall": 0.7,
            "f1_score": 0.72,
            "confusion_matrix": [[80, 5], [10, 5]],
            "classification_report": "Some classification report text"
        }
        with patch('builtins.print') as mock_print:
            report_classification_metrics(metrics)
            printed = "\n".join([call.args[0] for call in mock_print.call_args_list])
            self.assertIn("CLASSIFICATION METRICS", printed)
            self.assertIn("Accuracy : 0.8500", printed)

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False, verbosity=2) # exit=False prevents sys.exit() from being called
