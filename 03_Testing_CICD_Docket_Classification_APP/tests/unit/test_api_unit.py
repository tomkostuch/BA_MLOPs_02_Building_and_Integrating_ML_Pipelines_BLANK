# ml_project/tests/test_api_unit.py
"""
Objective:
    Unit tests specifically for the FastAPI application's Pydantic models
    and any small, isolated utility functions within app.py,
    implemented using Python's built-in `unittest` framework.
    It focuses on validating data structures and basic logic in isolation,
    without interacting with the FastAPI server or the ML model's prediction logic.

Tests Performed:
    - test_churn_predict_request_validation_success:
        Verifies that valid data correctly validates against the Pydantic model.
    - test_churn_predict_request_validation_failure_invalid_type:
        Tests that the Pydantic model raises an error for incorrect data types.
    - test_churn_predict_request_validation_failure_missing_field:
        Tests that the Pydantic model raises an error for missing required fields.
"""
import unittest
import os
import sys
from pydantic import ValidationError # Specific exception for Pydantic validation

# Add the project root to sys.path to allow imports from app.py
# Assumes test_api_unit.py is in 'ml_project/tests/'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from app import ChurnPredictRequest # Import the Pydantic model


class TestChurnPredictRequest(unittest.TestCase):

    def setUp(self):
        """
        Set up common valid prediction data for tests.
        """
        self.valid_prediction_data = {
            "gender": "Male",
            "SeniorCitizen": 0,
            "Partner": "No",
            "Dependents": "No",
            "tenure": 1,
            "PhoneService": "No",
            "MultipleLines": "No phone service",
            "InternetService": "DSL",
            "OnlineSecurity": "No",
            "OnlineBackup": "Yes",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV": "No",
            "StreamingMovies": "No",
            "Contract": "Month-to-month",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": 29.85,
            "TotalCharges": 29.85
        }
        print("\n--- Setup: Valid prediction data prepared ---")

    def test_churn_predict_request_validation_success(self):
        """Test that valid data correctly validates against the Pydantic model."""
        print("\n--- Test: Valid data validation success ---")
        # The test will fail automatically if a ValidationError is raised here,
        # which is the desired behavior for a success test.
        request_model = ChurnPredictRequest(**self.valid_prediction_data)

        # Assert that the data was loaded correctly into the model's attributes.
        self.assertEqual(request_model.gender, "Male")
        self.assertEqual(request_model.tenure, 1)
        self.assertEqual(request_model.TotalCharges, 29.85)
        print("Pydantic model successfully validated with valid data.")
        print("--- Test Complete ---")

    def test_churn_predict_request_validation_failure_invalid_type(self):
        """Test Pydantic model raises error for incorrect data types."""
        print("\n--- Test: Validation failure - invalid type ---")
        invalid_data = self.valid_prediction_data.copy()
        invalid_data["tenure"] = "not_an_int" # Type mismatch

        with self.assertRaises(ValidationError) as cm:
            ChurnPredictRequest(**invalid_data)

        # Check if the error message contains a relevant type error message
        error_message = str(cm.exception)
        self.assertIn("Input should be a valid integer", error_message)
        print(f"Pydantic model correctly raised ValidationError for invalid type: {error_message.splitlines()[0]}")
        print("--- Test Complete ---")

    def test_churn_predict_request_validation_failure_missing_field(self):
        """Test Pydantic model raises error for missing required fields."""
        print("\n--- Test: Validation failure - missing field ---")
        invalid_data = self.valid_prediction_data.copy()
        del invalid_data["gender"] # Remove a required field

        with self.assertRaises(ValidationError) as cm:
            ChurnPredictRequest(**invalid_data)

        # Check if the error message contains a relevant missing field message
        error_message = str(cm.exception)
        self.assertIn("Field required", error_message)
        self.assertIn("gender", error_message)
        print(f"Pydantic model correctly raised ValidationError for missing field: {error_message.splitlines()[0]}")
        print("--- Test Complete ---")


if __name__ == '__main__':
    unittest.main()