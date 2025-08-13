# ml_project/tests/integration/test_model_metrics_integration.py
"""
Objective:
    Integration test for the core model functionality: training, prediction,
    and subsequent metric computation, now implemented using Python's
    built-in `unittest` framework.
    Verifies that the `ChurnPredictionModel` can be fitted and its predictions
    are correctly processed by `compute_classification_metrics`.

Tests Performed:
    - test_model_training_prediction_and_metrics_flow:
        Creates a simple dummy dataset, simulates preprocessing, trains the model,
        makes predictions, and asserts on the calculated metrics, ensuring the
        entire flow works cohesively.
"""
import unittest
import pandas as pd
import numpy as np
import os
import sys

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Add the project root to sys.path to allow imports from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Import components
from src.model import ChurnPredictionModel, compute_classification_metrics
from src.preprocessing import stratified_split # To get X, y, splits

class TestModelMetricsIntegration(unittest.TestCase):

    def setUp(self):
        # Create preprocessed dummy data directly in setUp
        X = pd.DataFrame(np.random.rand(100, 5), columns=[f'feature_{i}' for i in range(5)])
        y = pd.Series(np.random.randint(0, 2, 100))

        # Apply a simple preprocessor to X, similar to how it would be done in the pipeline
        preprocessor = StandardScaler()
        X_scaled = pd.DataFrame(preprocessor.fit_transform(X), columns=X.columns)

        # Attach preprocessor and feature_columns as attributes, as `transform_features` would
        X_scaled.attrs['preprocessor'] = preprocessor
        X_scaled.attrs['feature_columns'] = list(X.columns)

        self.X_full = X_scaled
        self.y_full = y

    def test_model_training_prediction_and_metrics_flow(self):
        """
        Integration test: Verifies the flow from preprocessed data to model training,
        prediction, and metric computation.
        """
        print("\n--- Integration Test: Model Training, Prediction, and Metrics Flow ---")

        # 1. Split data (simulating pipeline's splitting after preprocessing)
        X_train, X_test, y_train, y_test = stratified_split(self.X_full, self.y_full, test_size=0.3, seed=42)
        print(f"Integration Test: Data split into train/test. Train shape: {X_train.shape}, Test shape: {X_test.shape}")

        # 2. Instantiate and train model
        # Use the preprocessor from the X_full's attributes (simulating transform_features)
        model = ChurnPredictionModel(
            classifier=LogisticRegression(random_state=42),
            preprocessor=self.X_full.attrs['preprocessor'] # Pass the fitted preprocessor
        )
        fitted_model = model.fit(X_train, y_train)
        self.assertIs(fitted_model, model)
        print("Model fitted successfully.")

        # 3. Make predictions
        y_pred = fitted_model.predict(X_test)
        self.assertIsInstance(y_pred, np.ndarray)
        self.assertEqual(y_pred.shape, y_test.shape)
        print("Predictions made.")

        # 4. Compute metrics
        metrics = compute_classification_metrics(y_test.to_numpy(), y_pred)
        self.assertIn('accuracy', metrics)
        self.assertIn('precision', metrics)
        self.assertIn('recall', metrics)
        self.assertIn('f1_score', metrics)
        self.assertIn('confusion_matrix', metrics)
        print(f"Metrics computed: {metrics}")

        # Basic assertion on metric values (expect them to be reasonable, not perfect)
        self.assertGreaterEqual(metrics['accuracy'], 0.4) # Should be better than random guess
        self.assertLessEqual(metrics['accuracy'], 1.0) # Should not exceed 1.0

        print("--- Integration Test Complete ---")

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False, verbosity=2) # exit=False prevents sys.exit() from being called