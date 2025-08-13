"""
model.py

This script defines the ChurnPredictionModel class, which encapsulates
the machine learning model (e.g., Logistic Regression) and its associated
preprocessing pipeline. It also includes functions for computing and reporting
classification metrics, and for logging model run details.
"""
import pandas as pd
import numpy as np
import joblib
import os
import json
from datetime import datetime
from typing import Any, cast, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.preprocessing import StandardScaler # Import for fallback in __init__


class ChurnPredictionModel:
    """
    Customer churn prediction model using configurable classifier with preprocessing pipeline.
    """

    def __init__(
        self,
        classifier=None,
        preprocessor=None,
        random_state: int = 42
    ):
        """
        Initialize the churn prediction pipeline.

        Args:
            classifier: Scikit-learn classifier instance. If None, uses LogisticRegression.
            preprocessor: Scikit-learn preprocessing pipeline. If None, uses StandardScaler only.
            random_state (int): Random seed for reproducibility.
        """
        self.random_state = random_state
        self.classifier = classifier if classifier is not None else LogisticRegression(
            random_state=self.random_state,
            max_iter=1000,
            class_weight='balanced'
        )
        self.pipe = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', self.classifier)
        ]) if preprocessor is not None else Pipeline([
            ('scaler', StandardScaler()), # Fallback if no preprocessor provided
            ('classifier', self.classifier)
        ])

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'ChurnPredictionModel':
        """
        Fit the model to training data.

        Args:
            X (pd.DataFrame): Training features.
            y (pd.Series): Training target.

        Returns:
            ChurnPredictionModel: Self for method chaining.
        """
        print(f"Training model with {len(X)} samples and {len(X.columns)} features...")

        # Fit the pipeline
        self.pipe.fit(X, y)

        print(f"Model trained successfully!")
        print(f"Features used: {list(X.columns)}")

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on input data.

        Args:
            X (pd.DataFrame): Input features.

        Returns:
            np.ndarray: Predicted class labels.
        """
        return cast(np.ndarray, self.pipe.predict(X))

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            X (pd.DataFrame): Input features.

        Returns:
            np.ndarray: Predicted class probabilities.
        """
        return cast(np.ndarray, self.pipe.predict_proba(X))

    def save(self, filepath: str) -> None:
        """
        Save the trained model to a file.

        Args:
            filepath (str): Full path to save the model.
        """
        directory = os.path.dirname(filepath)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

        joblib.dump(self.pipe, filepath)
        print(f"Model saved to {filepath}")

    def log_run(
        self,
        directory: str,
        metrics: dict[str, Any],
        dataset_info: dict[str, Any],
        log_filename: str = "churn_model_run_log.json"
    ) -> None:
        """
        Save model configuration and performance metrics to a JSON file.

        Args:
            directory (str): Directory where the JSON file will be stored.
            metrics (Dict[str, Any]): Evaluation metrics to save.
            dataset_info (Dict[str, Any]): Information about the dataset used.
            log_filename (str): Name of the log file.
        """
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

        run_info = {
            "timestamp": datetime.now().isoformat(),
            "model_class": self.__class__.__name__,
            "classifier": str(type(self.classifier).__name__),
            "dataset": "Customer Churn",
            "dataset_info": dataset_info,
            "parameters": {
                "random_state": self.random_state,
                "classifier_params": self.classifier.get_params()
            },
            "metrics": metrics
        }

        log_file = os.path.join(directory, log_filename)

        # Load existing logs or create new list
        if os.path.exists(log_file):
            try:
                with open(log_file, "r") as f:
                    logs = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                logs = []
        else:
            logs = []

        logs.append(run_info)

        # Save updated logs
        with open(log_file, "w") as f:
            json.dump(logs, f, indent=4, default=str)

        print(f"Run log saved to {log_file}")


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: Optional[list[str]] = None
) -> dict[str, Any]:
    """
    Computes classification metrics for binary classification.

    Args:
        y_true (np.ndarray): True labels.
        y_pred (np.ndarray): Predicted labels.
        target_names (Optional[list[str]]): Names of target classes.

    Returns:
        Dict[str, Any]: Dictionary containing all computed metrics.
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average='binary'),
        "recall": recall_score(y_true, y_pred, average='binary'),
        "f1_score": f1_score(y_true, y_pred, average='binary'),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(
            y_true, y_pred,
            target_names=target_names,
            output_dict=False
        )
    }

    return metrics

def report_classification_metrics(metrics: dict[str, Any]) -> None:
    """
    Prints formatted classification metrics.

    Args:
        metrics (Dict[str, Any]): Model evaluation metrics.
    """
    output_lines = []
    output_lines.append("\n" + "="*40)
    output_lines.append("CLASSIFICATION METRICS")
    output_lines.append("="*40)

    # Check for key existence to prevent KeyErrors if metrics dict is incomplete
    output_lines.append(f"Accuracy : {metrics.get('accuracy', float('nan')):.4f}")
    output_lines.append(f"Precision: {metrics.get('precision', float('nan')):.4f}")
    output_lines.append(f"Recall   : {metrics.get('recall', float('nan')):.4f}")
    output_lines.append(f"F1-Score : {metrics.get('f1_score', float('nan')):.4f}")

    output_lines.append(f"\nConfusion Matrix:")
    confusion_matrix_data = metrics.get('confusion_matrix')
    if isinstance(confusion_matrix_data, list):
        for row in confusion_matrix_data:
            output_lines.append(f"  {row}")
    elif confusion_matrix_data is not None:
        output_lines.append(f"  Unexpected format for confusion matrix: {confusion_matrix_data}")
    else:
        output_lines.append(f"  Confusion matrix data not available.")

    print("\n".join(output_lines))