"""
pipeline.py

This script defines and orchestrates the end-to-end machine learning pipeline
for customer churn prediction. It encapsulates the sequential steps from
data loading and preprocessing to model training and evaluation.
"""
import pandas as pd
import os
from typing import Any, Dict, Tuple

from src.data_loader import load_churn_dataset
from src.preprocessing import clean_churn_data, transform_features, split_features_and_target, stratified_split
from src.model import ChurnPredictionModel, compute_classification_metrics, report_classification_metrics


def run_churn_pipeline(
    data_file_path: str,
    target_column: str,
    numeric_columns: list[str],
    categorical_columns: list[str],
    test_size: float,
    random_state: int,
    model_dir_path: str,
    model_filename: str,
    log_filename: str
) -> Tuple[ChurnPredictionModel, Dict[str, Any]]:
    """
    Runs the complete customer churn prediction pipeline.

    Args:
        data_file_path (str): Path to the raw dataset CSV file.
        target_column (str): Name of the target column.
        numeric_columns (list[str]): List of numeric feature column names.
        categorical_columns (list[str]): List of categorical feature column names.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Random seed for reproducibility.
        model_dir_path (str): Directory where the trained model and logs will be saved.
        model_filename (str): Name of the file to save the trained model.
        log_filename (str): Name of the file to save the run logs.

    Returns:
        Tuple[ChurnPredictionModel, Dict[str, Any]]:
            - The trained ChurnPredictionModel instance.
            - A dictionary containing the evaluation metrics.
    """
    print("\n1. Loading dataset...")
    df_raw = load_churn_dataset(data_file_path)

    print("\n2. Cleaning data...")
    df_clean = clean_churn_data(df_raw, target_column, numeric_columns, categorical_columns)

    print("\n3. Transforming features...")
    # If transform_features now returns only 2 items, adjust unpacking:
    df_transformed, preprocessor = transform_features(
        df_clean, target_column, numeric_columns, categorical_columns
    )

    # Extract dataset information
    dataset_info = {
        "total_samples": len(df_transformed),
        "n_features": len(df_transformed.attrs.get('feature_columns', [])),
        "target_mapping": df_transformed.attrs.get('target_mapping', {}),
        "target_names": df_transformed.attrs.get('target_names', []),
        "churn_distribution": df_transformed['target_encoded'].value_counts().to_dict(),
        "feature_columns": df_transformed.attrs.get('feature_columns', []),
        "numeric_features": df_transformed.attrs.get('numeric_features', []),
        "categorical_features": df_transformed.attrs.get('categorical_features', [])
    }

    print("\n4. Preparing features and target...")
    X, y = split_features_and_target(df_transformed)

    print("\n5. Splitting data...")
    X_train, X_test, y_train, y_test = stratified_split(X, y, test_size=test_size, seed=random_state)
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    print(f"Train churn rate: {y_train.mean():.3f}")
    print(f"Test churn rate: {y_test.mean():.3f}")

    print("\n6. Training model...")
    model = ChurnPredictionModel(preprocessor=preprocessor, random_state=random_state)
    model.fit(X_train, y_train)

    print("\n7. Making predictions...")
    y_pred = model.predict(X_test)

    print("\n8. Evaluating model...")
    metrics = compute_classification_metrics(
        y_test.to_numpy(),
        y_pred,
        target_names=dataset_info['target_names']
    )
    report_classification_metrics(metrics)

    print("\n9. Saving model and logs...")
    model_file_path = os.path.join(model_dir_path, model_filename)
    model.save(model_file_path)
    model.log_run(model_dir_path, metrics, dataset_info, log_filename=log_filename)

    print(f"Model saved to: {model_file_path}")
    print(f"Run log saved to: {os.path.join(model_dir_path, log_filename)}")

    return model, metrics