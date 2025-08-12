# churn_classification_app_unittest/tests_unittest/test_e2e_suite.py
"""
Objective:
    End-to-End (E2E) tests for the deployed FastAPI application,
    implemented using Python's built-in `unittest` framework.
    It orchestrates the setup of a test environment, including:
    1. Verifying the existence of a pre-trained ML model on the host.
    2. Building a Docker image for the FastAPI app.
    3. Starting the Docker container based on the built image.
    4. Waiting for the API service to become reachable.
    5. Executing actual HTTP requests against the running service.
    6. Performing necessary cleanup (stopping/removing container, deleting Docker image).

    These tests verify the complete system behavior, including interaction with
    the actual loaded ML model and the network layer, mimicking a real-world scenario.

Tests Performed:
    - test_e2e_predict_success:
        Deploys the FastAPI app in Docker, makes a valid prediction request to its API,
        and asserts on the 200 status code and the structure/values of the prediction response.
    - test_e2e_predict_invalid_input:
        Deploys the FastAPI app, makes a prediction request with invalid input,
        and asserts on the 422 status code and the error details.
"""
import unittest
import requests
import time
import subprocess
import json
import os
from pathlib import Path
import sys
# shutil is no longer needed as we are not deleting model_store or log files
# from the host, but managing Docker artifacts.

# Add project root to sys.path
# Use pathlib for robust path handling
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT)) # sys.path needs a string

# Import config for consistent paths (assuming src.config exists)
try:
    from src.config import MODEL_FILENAME, MODEL_STORE_DIR # LOG_FILENAME removed
except ImportError:
    # Fallback for testing if config is not strictly available or if we're in a strange env
    print("Warning: Could not import MODEL_FILENAME or MODEL_STORE_DIR from src.config. Using defaults.")
    MODEL_FILENAME = "churn_prediction_model_v1.joblib"
    MODEL_STORE_DIR = "model_store"


# Configuration for the E2E test
API_URL = "http://localhost:8000"
PREDICT_ENDPOINT = f"{API_URL}/predict"
DOCKER_IMAGE_NAME = "churn-api-e2e-test"
CONTAINER_NAME = "churn-api-e2e-container"
API_STARTUP_TIMEOUT = 120 # seconds for API to become reachable
API_HEALTH_CHECK_INTERVAL = 2 # seconds for checking health

# Paths for model relative to the project root on the HOST machine
MODEL_DIR = PROJECT_ROOT / MODEL_STORE_DIR
MODEL_PATH_FULL = MODEL_DIR / MODEL_FILENAME

# Sample data for a valid prediction request
sample_e2e_prediction_data = {
    "tenure": 1,
    "MonthlyCharges": 29.85,
    "TotalCharges": 29.85,
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
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
    "PaymentMethod": "Electronic check"
}


class TestE2ESuite(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """
        Sets up the E2E test environment by:
        1. Verifying the pre-trained ML model exists on the host (prerequisite for Docker build).
        2. Builds the Docker image for the FastAPI app.
        3. Starts the Docker container.
        4. Waits for the API to be reachable (via /docs endpoint).
        """
        print("\n--- E2E Setup: Preparing environment and starting services ---")

        # 1. Verifying the pre-trained ML model exists on the host.
        # This is a prerequisite for the Docker build process to copy it.
        print("\n--- E2E Setup: Verifying Pre-trained Model on Host ---")
        if not MODEL_PATH_FULL.exists():
            raise AssertionError(
                f"Pre-trained model not found at {MODEL_PATH_FULL}. "
                "Please ensure the ML pipeline has been run and the model is saved to this location "
                "before running E2E tests, as it needs to be copied into the Docker image."
            )
        print(f"Using pre-trained model found on host at: {MODEL_PATH_FULL}")

        # Ensure no pre-existing container with the same name is running
        print("\n--- E2E Setup: Cleaning up any old container/image ---")
        subprocess.run(["docker", "stop", CONTAINER_NAME], check=False, capture_output=True, text=True)
        subprocess.run(["docker", "rm", CONTAINER_NAME], check=False, capture_output=True, text=True)
        subprocess.run(["docker", "rmi", DOCKER_IMAGE_NAME], check=False, capture_output=True, text=True)
        print("Ensured no pre-existing container or image with the same name.")


        # 2. Builds the Docker image for the FastAPI app.
        print("\n--- E2E Setup: Building Docker Image ---")
        try:
            # Build the Docker image from the project root (where Dockerfile, app.py, src/, model_store/ are)
            # check=True raises CalledProcessError if command returns non-zero exit status
            result = subprocess.run(["docker", "build", "-t", DOCKER_IMAGE_NAME, "."],
                                    check=True, cwd=PROJECT_ROOT, capture_output=True, text=True)
            print("Docker build stdout:\n", result.stdout)
            if result.stderr: # Only print stderr if there's actual content
                print("Docker build stderr:\n", result.stderr)
            print(f"Docker image '{DOCKER_IMAGE_NAME}' built successfully.")
        except subprocess.CalledProcessError as e:
            raise AssertionError(f"Docker image build failed. STDOUT: {e.stdout}\nSTDERR: {e.stderr}")
        except FileNotFoundError:
            raise AssertionError("Docker command not found. Is Docker installed and in your system's PATH?")


        # 3. Starts the Docker container.
        print("\n--- E2E Setup: Starting Docker Container ---")
        try:
            cmd = [
                "docker", "run", "-d", # Run in detached mode
                "--name", CONTAINER_NAME,
                "-p", "8000:8000", # Map host port 8000 to container port 8000
                DOCKER_IMAGE_NAME # Use the built image
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(f"Container start stdout: {result.stdout}")
            if result.stderr: # Only print stderr if there's actual content
                print(f"Container start stderr: {result.stderr}")
            cls.container_id = result.stdout.strip()
            print(f"Container {cls.container_id} started.")

        except subprocess.CalledProcessError as e:
            print(f"Docker run failed. STDOUT: {e.stdout}\nSTDERR: {e.stderr}")
            cls.tearDownClass() # Attempt cleanup before raising
            raise AssertionError(f"Docker container failed to start: {e}")
        except Exception as e:
            cls.tearDownClass() # Attempt cleanup before raising
            raise AssertionError(f"An unexpected error occurred during Docker container startup: {e}")

        # 4. Waits for the API to be reachable (e.g., by checking the /docs endpoint).
        print("Waiting for API to become reachable...")
        for i in range(API_STARTUP_TIMEOUT // API_HEALTH_CHECK_INTERVAL):
            try:
                # Use a higher timeout for the request itself, separate from total startup timeout
                response = requests.get(f"{API_URL}/docs", timeout=10)
                if response.status_code == 200:
                    print("API is reachable!")
                    break
            except requests.exceptions.ConnectionError as e:
                print(f"Connection error: {e}. Retrying...")
            except requests.exceptions.Timeout:
                print("Request timed out. Retrying...")
            except Exception as e: # Catch any other request-related errors
                print(f"An unexpected error occurred during health check: {e}. Retrying...")

            time.sleep(API_HEALTH_CHECK_INTERVAL)
        else:  # This block runs if the loop completes without a 'break'
            # If the API did not become ready, get logs before failing.
            print("\n--- Docker Container Logs (from failed startup) ---")
            logs_result = subprocess.run(
                ["docker", "logs", CONTAINER_NAME],
                capture_output=True, text=True, check=False
            )
            print(logs_result.stderr + logs_result.stdout) # Print both stdout and stderr logs
            print("---------------------------------------------------\n")
            cls.tearDownClass() # Ensure cleanup before failing
            raise AssertionError(f"API did not become reachable within {API_STARTUP_TIMEOUT} seconds.")

        print("--- E2E Setup Complete ---")

    @classmethod
    def tearDownClass(cls):
        """
        Teardown for all E2E tests:
        1. Stop and remove the Docker container.
        2. Remove the Docker image.
        """
        print("\n--- E2E Teardown: Stopping services and cleaning up artifacts ---")

        # 1. Stop and remove Docker container
        if hasattr(cls, 'container_id') and cls.container_id:
            print(f"Stopping and removing Docker container: {CONTAINER_NAME} ({cls.container_id[:12]}) ...")
            subprocess.run(["docker", "stop", CONTAINER_NAME], check=False, capture_output=True, text=True)
            subprocess.run(["docker", "rm", CONTAINER_NAME], check=False, capture_output=True, text=True)
            print("Docker container stopped and removed.")

        # 2. Remove Docker image
        print(f"Removing Docker image: {DOCKER_IMAGE_NAME}...")
        try:
            # -f (force) is often useful for test cleanup to remove even if in use (though it shouldn't be)
            subprocess.run(["docker", "rmi", DOCKER_IMAGE_NAME, "-f"], check=False, capture_output=True, text=True)
            print("Docker image removed.")
        except Exception as e:
            print(f"Error during image removal: {e}") # Don't fail the test cleanup for this

        print("--- E2E Teardown Complete ---")


    def test_e2e_predict_success(self):
        """E2E test for the /predict endpoint with valid data, matching the provided output format."""
        print("\n--- Test: E2E /predict successful ---")
        response = requests.post(PREDICT_ENDPOINT, json=sample_e2e_prediction_data)

        self.assertEqual(response.status_code, 200, f"Expected status 200, got {response.status_code}. Response: {response.text}")
        data = response.json()

        # Assert prediction string
        self.assertIn("prediction", data)
        self.assertIn(data["prediction"], ["Churn", "No Churn"], "Prediction should be 'Churn' or 'No Churn'")

        # Assert direct probability keys
        self.assertIn("churn_probability", data)
        self.assertIsInstance(data["churn_probability"], float)
        self.assertGreaterEqual(data["churn_probability"], 0.0)
        self.assertLessEqual(data["churn_probability"], 1.0)

        self.assertIn("no_churn_probability", data)
        self.assertIsInstance(data["no_churn_probability"], float)
        self.assertGreaterEqual(data["no_churn_probability"], 0.0)
        self.assertLessEqual(data["no_churn_probability"], 1.0)

        # Assert the 'probabilities' dictionary
        self.assertIn("probabilities", data)
        self.assertIsInstance(data["probabilities"], dict)
        self.assertIn("No Churn", data["probabilities"])
        self.assertIn("Churn", data["probabilities"])
        self.assertIsInstance(data["probabilities"]["No Churn"], float)
        self.assertIsInstance(data["probabilities"]["Churn"], float)
        self.assertGreaterEqual(data["probabilities"]["No Churn"], 0.0)
        self.assertLessEqual(data["probabilities"]["No Churn"], 1.0)
        self.assertGreaterEqual(data["probabilities"]["Churn"], 0.0)
        self.assertLessEqual(data["probabilities"]["Churn"], 1.0)

        # Assert probabilities sum to ~1.0
        self.assertAlmostEqual(data["churn_probability"] + data["no_churn_probability"], 1.0, places=5)
        self.assertAlmostEqual(data["probabilities"]["No Churn"] + data["probabilities"]["Churn"], 1.0, places=5)


        print(f"E2E /predict successful. Response: {data}")
        print("--- Test Complete ---")


    def test_e2e_predict_invalid_input(self):
        """E2E test for /predict with invalid input."""
        print("\n--- Test: E2E /predict invalid input ---")
        invalid_data = sample_e2e_prediction_data.copy()
        invalid_data["tenure"] = "not_a_number" # Invalid type

        response = requests.post(PREDICT_ENDPOINT, json=invalid_data)
        self.assertEqual(response.status_code, 422, f"Expected status 422, got {response.status_code}. Response: {response.text}")
        data = response.json()
        self.assertIn("detail", data)
        # Check for a validation error related to the 'tenure' field being an invalid integer.
        self.assertTrue(
            any("valid integer" in err["msg"].lower() and err["loc"][-1] == "tenure"
                for err in data["detail"]),
            f"Expected validation error for 'tenure' field, got: {data['detail']}"
        )
        print(f"E2E /predict correctly handled invalid input. Response: {data}")
        print("--- Test Complete ---")


if __name__ == '__main__':
    unittest.main()