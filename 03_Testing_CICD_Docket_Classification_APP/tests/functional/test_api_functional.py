# ml_project/tests/test_api_functional.py
"""
Objective:
    Functional tests for the FastAPI application's API endpoints,
    implemented using Python's built-in `unittest` framework and FastAPI's `TestClient`.
    It focuses on verifying that each endpoint (`/predict`) behaves as expected
    given specific inputs, returning correct HTTP status codes and response bodies.
    The ML model's prediction logic is typically mocked in these tests to isolate
    the API's routing and response handling from the model's internal logic.

Tests Performed:
    - test_predict_endpoint_success:
        Verifies that the /predict endpoint returns a 200 status code and
        a valid prediction structure when the model is loaded and returns a mock prediction.
    - test_predict_endpoint_model_not_loaded:
        Ensures the /predict endpoint handles the scenario where the ML model
        is not yet loaded, returning a 503 status code.
    - test_predict_endpoint_invalid_input_data:
        Tests that the /predict endpoint returns a 422 status code for missing
        required fields in the input.
    - test_predict_endpoint_incorrect_data_type:
        Tests that the /predict endpoint returns a 422 status code for incorrect
        data types in the input fields.
    - test_predict_endpoint_model_prediction_error:
        Verifies that the /predict endpoint gracefully handles exceptions
        originating from the underlying model's prediction method, returning a 500 status.
"""
import unittest
import os
import sys
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

# Add the project root to sys.path to allow imports from app.py
# Assumes test_api_functional.py is in 'ml_project/tests/'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Import app as app_module to patch its internal attributes more easily if needed,
# and also import the app instance directly.
import app as app_module
from app import app # Import the FastAPI app instance


class TestAPIFunctional(unittest.TestCase):

    # Class-level setup for the TestClient
    @classmethod
    def setUpClass(cls):
        """
        Set up the FastAPI TestClient once for all tests in this class.
        """
        print("\n--- Functional Test Setup: Initializing FastAPI TestClient ---")
        # Initialize TestClient and manually trigger the app's startup events
        cls.client = TestClient(app)
        cls.client.__enter__()
        # Define common valid prediction data
        cls.valid_prediction_data = {
            "gender": "Male", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
            "tenure": 1, "PhoneService": "No", "MultipleLines": "No phone service",
            "InternetService": "DSL", "OnlineSecurity": "No", "OnlineBackup": "Yes",
            "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
            "StreamingMovies": "No", "Contract": "Month-to-month",
            "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check",
            "MonthlyCharges": 29.85, "TotalCharges": 29.85
        }
        print("FastAPI TestClient initialized. Valid prediction data prepared.")

    @classmethod
    def tearDownClass(cls):
        """Clean up resources after all tests in the class have run."""
        print("\n--- Functional Test Teardown: Shutting down FastAPI TestClient ---")
        # Manually trigger the app's shutdown events
        cls.client.__exit__(None, None, None)
        print("FastAPI TestClient resources cleaned up.")

    def setUp(self):
        """
        Set up a fresh mock for the model pipeline before each test.
        This ensures tests are isolated from each other's changes to the mock.
        """
        print("--- Per-test Setup: Setting up mock model pipeline ---")
        # Create a mock pipeline object for each test
        self.mock_pipeline = MagicMock()
        # Configure mock with predictable output for successful prediction tests
        self.mock_pipeline.predict.return_value = [1]  # Predicts 'Churn'
        self.mock_pipeline.predict_proba.return_value = [[0.210779946276635, 0.789220053723365]] # [P(No Churn), P(Churn)]

        # Use patch.object to replace the model in the app's state.
        # self.addCleanup ensures that the patch is automatically stopped after the test.
        patcher = patch.object(app_module.app.state, 'model_pipeline', self.mock_pipeline)
        patcher.start()
        self.addCleanup(patcher.stop)
        print("Mock model pipeline set as app.state.model_pipeline.")


    def test_predict_endpoint_success(self):
        """
        Test /predict endpoint with valid data when the model is mocked
        to return a successful prediction.
        """
        print("\n--- Test: /predict endpoint success with mocked model ---")
        response = self.client.post("/predict", json=self.valid_prediction_data)

        self.assertEqual(response.status_code, 200, f"Expected status 200, got {response.status_code}")
        data = response.json()

        self.assertIn("prediction", data)
        self.assertEqual(data["prediction"], "Churn") # Based on mock_pipeline.predict.return_value
        self.assertIn("no_churn_probability", data)
        self.assertIsInstance(data["no_churn_probability"], float)
        self.assertIn("churn_probability", data)
        self.assertIsInstance(data["churn_probability"], float)
        self.assertEqual(data["no_churn_probability"], 0.210779946276635)
        self.assertEqual(data["churn_probability"], 0.789220053723365)
        self.mock_pipeline.predict.assert_called_once()
        self.mock_pipeline.predict_proba.assert_called_once()
        print("'/predict' endpoint tested successfully with mocked model.")
        print(f"Response: {data}")
        print("--- Test Complete ---")

    def test_predict_endpoint_model_not_loaded(self):
        """
        Test /predict endpoint when the model pipeline is explicitly set to None.
        Should return 503 Service Unavailable.
        """
        print("\n--- Test: /predict endpoint - model not loaded ---")
        app_module.app.state.model_pipeline = None # Simulate model not being loaded

        response = self.client.post("/predict", json=self.valid_prediction_data)

        self.assertEqual(response.status_code, 503, f"Expected status 503, got {response.status_code}")
        data = response.json()
        self.assertIn("detail", data)
        self.assertEqual(data["detail"], "Model not loaded yet.")
        print("'/predict' endpoint correctly returned 503 when model not loaded.")
        print(f"Response: {data}")
        print("--- Test Complete ---")

    def test_predict_endpoint_invalid_input_data(self):
        """Test /predict endpoint with missing required field."""
        print("\n--- Test: /predict endpoint - invalid input (missing field) ---")
        invalid_data = self.valid_prediction_data.copy()
        del invalid_data["gender"] # Missing required field

        response = self.client.post("/predict", json=invalid_data)

        self.assertEqual(response.status_code, 422, f"Expected status 422, got {response.status_code}")
        data = response.json()
        self.assertIn("detail", data)
        self.assertIsInstance(data["detail"], list)
        self.assertIn("Field required", str(data["detail"]))
        self.assertIn("gender", str(data["detail"]))
        print("'/predict' endpoint correctly returned 422 for missing field.")
        print(f"Response: {data}")
        print("--- Test Complete ---")

    def test_predict_endpoint_incorrect_data_type(self):
        """Test /predict endpoint with wrong data type for field."""
        print("\n--- Test: /predict endpoint - invalid input (incorrect type) ---")
        invalid_data = self.valid_prediction_data.copy()
        invalid_data["tenure"] = "one" # Incorrect type

        response = self.client.post("/predict", json=invalid_data)

        self.assertEqual(response.status_code, 422, f"Expected status 422, got {response.status_code}")
        data = response.json()
        self.assertIn("detail", data)
        self.assertIsInstance(data["detail"], list)
        self.assertIn("Input should be a valid integer", str(data["detail"]))
        print("'/predict' endpoint correctly returned 422 for incorrect data type.")
        print(f"Response: {data}")
        print("--- Test Complete ---")

    def test_predict_endpoint_model_prediction_error(self):
        """Test /predict if the model.predict method raises an error."""
        print("\n--- Test: /predict endpoint - model prediction error ---")
        # Configure the mock model to raise an exception during prediction
        self.mock_pipeline.predict.side_effect = Exception("Mock prediction error")
        # Ensure the app is using this mock
        app_module.app.state.model_pipeline = self.mock_pipeline

        response = self.client.post("/predict", json=self.valid_prediction_data)

        self.assertEqual(response.status_code, 500, f"Expected status 500, got {response.status_code}")
        data = response.json()
        self.assertIn("detail", data)
        # Check for a more specific error message that indicates a prediction failure
        self.assertIn("Prediction failed", data["detail"])
        self.assertIn("Mock prediction error", data["detail"])
        self.mock_pipeline.predict.assert_called_once()
        print("'/predict' endpoint correctly returned 500 for model prediction error.")
        print(f"Response: {data}")
        print("--- Test Complete ---")


if __name__ == '__main__':
    unittest.main()