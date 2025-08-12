# ml_project/tests/test_api_integration.py
"""
Objective:
    Integration tests for the FastAPI application, implemented using Python's
    built-in `unittest` framework and FastAPI's `TestClient`.
    It focuses on validating the interaction between the FastAPI API endpoints
    and the actual, loaded machine learning model. These tests ensure that
    the model is correctly loaded and can make valid predictions based on
    the data received through the API, verifying the end-to-end data flow
    from API request parsing to model inference.

Tests Performed:
    - test_predict_endpoint_with_real_model_success:
        Verifies that the /predict endpoint returns a 200 status code and
        a valid prediction structure for a typical input, using the actual loaded model.
    - test_predict_endpoint_validation_failure:
        Verifies that the /predict endpoint correctly handles invalid input data
        by returning a 422 status code (Unprocessable Entity).
    - test_predict_endpoint_with_real_model_edge_case_input:
        Tests the /predict endpoint with an edge case input to ensure robustness.
"""
import unittest
import os
import sys
import json # Used if loading data from a file for fixtures, though here it's direct dicts

# Import TestClient and app from FastAPI
from fastapi.testclient import TestClient

# Add the project root to sys.path to allow imports from app.py
# Assumes test_api_integration.py is in 'ml_project/tests/'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Import app and MODEL_PATH to check if the real model is loaded
from app import app, MODEL_PATH


class TestAPIIntegration(unittest.TestCase):

    # Class-level setup for the TestClient, equivalent to pytest fixture(scope="module")
    @classmethod
    def setUpClass(cls):
        """
        Set up the FastAPI TestClient once for all tests in this class.
        Also, verifies that the model is loaded during app startup.
        """
        print("\n--- Integration Test Setup: Initializing FastAPI TestClient and checking model ---")
        # Initialize TestClient and manually trigger the app's startup events
        cls.client = TestClient(app)
        cls.client.__enter__()

        if not hasattr(app.state, "model_pipeline") or app.state.model_pipeline is None:
            # This will fail the test suite early if the model isn't loaded, as expected
            raise AssertionError(
                f"Integration test requires actual model loaded at {MODEL_PATH}. "
                "Please ensure the model file exists and is correctly loaded into app.state by app.py "
                "during application startup.")
        print("FastAPI TestClient initialized. Model confirmed to be loaded.")

        # Define valid and edge-case prediction data as class attributes
        cls.valid_prediction_data = {
            "gender": "Male", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
            "tenure": 1, "PhoneService": "No", "MultipleLines": "No phone service",
            "InternetService": "DSL", "OnlineSecurity": "No", "OnlineBackup": "Yes",
            "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
            "StreamingMovies": "No", "Contract": "Month-to-month",
            "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check",
            "MonthlyCharges": 29.85, "TotalCharges": 29.85
        }

        cls.edge_case_prediction_data = {
            "gender": "Female", "SeniorCitizen": 1, "Partner": "Yes", "Dependents": "Yes",
            "tenure": 72, "PhoneService": "Yes", "MultipleLines": "Yes",
            "InternetService": "Fiber optic", "OnlineSecurity": "Yes", "OnlineBackup": "Yes",
            "DeviceProtection": "Yes", "TechSupport": "Yes", "StreamingTV": "Yes",
            "StreamingMovies": "Yes", "Contract": "Two year",
            "PaperlessBilling": "No", "PaymentMethod": "Credit card (automatic)",
            "MonthlyCharges": 118.75, "TotalCharges": 8684.8
        }
        print("Test data for integration tests prepared.")


    @classmethod
    def tearDownClass(cls):
        """Clean up resources after all tests in the class have run."""
        print("\n--- Integration Test Teardown: Shutting down FastAPI TestClient ---")
        # Manually trigger the app's shutdown events
        cls.client.__exit__(None, None, None)
        print("FastAPI TestClient resources cleaned up.")

    def test_predict_endpoint_with_real_model_success(self):
        """
        Tests the /predict endpoint with a real model using valid input,
        ensuring a 200 status code and expected prediction structure.
        """
        print("\n--- Test: /predict endpoint with valid input ---")
        response = self.client.post("/predict", json=self.valid_prediction_data)

        self.assertEqual(response.status_code, 200, f"Expected status code 200, got {response.status_code}")
        data = response.json()

        self.assertIn("prediction", data, "Response should contain 'prediction' key")
        self.assertIn(data["prediction"], ["Churn", "No Churn"], "Prediction should be 'Churn' or 'No Churn'")
        self.assertIn("no_churn_probability", data, "Response should contain 'no_churn_probability' key")
        self.assertIsInstance(data["no_churn_probability"], float, "no_churn_probability should be a float")
        self.assertIn("churn_probability", data, "Response should contain 'churn_probability' key")
        self.assertIsInstance(data["churn_probability"], float, "churn_probability should be a float")
        self.assertGreaterEqual(data["no_churn_probability"], 0.0)
        self.assertLessEqual(data["no_churn_probability"], 1.0)
        self.assertGreaterEqual(data["churn_probability"], 0.0)
        self.assertLessEqual(data["churn_probability"], 1.0)
        print("'/predict' endpoint with valid input tested successfully.")
        print(f"Response: {data}")
        print("--- Test Complete ---")

    def test_predict_endpoint_validation_failure(self):
        """
        Tests the /predict endpoint with invalid input data to ensure
        it returns a 422 status code (Unprocessable Entity).
        """
        print("\n--- Test: /predict endpoint validation failure ---")
        invalid_data = self.valid_prediction_data.copy()
        invalid_data["tenure"] = "not_an_integer" # Introduce an invalid data type

        response = self.client.post("/predict", json=invalid_data)

        self.assertEqual(response.status_code, 422, f"Expected status code 422 for invalid data, got {response.status_code}")
        data = response.json()
        self.assertIn("detail", data, "Error response should contain 'detail' key")
        self.assertIsInstance(data["detail"], list, "'detail' should be a list of errors")
        self.assertGreater(len(data["detail"]), 0, "Error detail list should not be empty")
        self.assertIn("Input should be a valid integer", str(data["detail"]), "Error message should indicate invalid type")
        print("'/predict' endpoint correctly handled invalid input with 422 status.")
        print(f"Response: {data}")
        print("--- Test Complete ---")

    def test_predict_endpoint_with_real_model_edge_case_input(self):
        """
        Tests the /predict endpoint with a real model using an edge case or
        potentially challenging input to ensure robustness.
        """
        print("\n--- Test: /predict endpoint with edge case input ---")
        response = self.client.post("/predict", json=self.edge_case_prediction_data)

        self.assertEqual(response.status_code, 200, f"Expected status code 200, got {response.status_code}")
        data = response.json()
        self.assertIn("prediction", data)
        self.assertIn(data["prediction"], ["Churn", "No Churn"])
        self.assertIn("no_churn_probability", data)
        self.assertIsInstance(data["no_churn_probability"], float)
        print("'/predict' endpoint with edge case input tested successfully.")
        print(f"Response: {data}")
        print("--- Test Complete ---")


if __name__ == '__main__':
    unittest.main()