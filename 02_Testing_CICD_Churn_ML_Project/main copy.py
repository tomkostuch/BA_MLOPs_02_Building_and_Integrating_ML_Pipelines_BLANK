"""
main.py

This is the main entry point for the Customer Churn Prediction ML pipeline.
It orchestrates the entire process by calling the main pipeline function
and handling overall execution flow.
"""
from typing import Optional
from pathlib import Path
import os
import sys
from src.pipeline import run_churn_pipeline # Import the new pipeline function
from src.config import (
    TARGET_COLUMN, NUMERIC_COLUMNS, CATEGORICAL_COLUMNS, TEST_SIZE,
    RANDOM_STATE, MODEL_FILENAME, LOG_FILENAME, DATA_DIR_NAME,
    RAW_DATA_DIR_NAME, DATASET_FILENAME, MODEL_STORE_DIR)


def main(output_base_dir: Optional[Path] = None) -> None:
    """
    Main function to orchestrate the entire ML pipeline.

    Args:
        output_base_dir (Path, optional): The base directory where
                                         data and model artifacts should be
                                         read from/saved to. If None,
                                         defaults to the script's directory.
    """
    print("Starting Customer Churn Prediction Pipeline...")
    print("="*60)

    try:
        # Construct necessary paths
        if output_base_dir is None:
            # Default to script's directory for normal runs
            base_path = Path(os.path.dirname(os.path.abspath(__file__)))
        else:
            # Use the provided base directory for testing
            base_path = output_base_dir

        data_file_path = base_path / DATA_DIR_NAME / RAW_DATA_DIR_NAME / DATASET_FILENAME
        model_dir_path = base_path / MODEL_STORE_DIR

        # Run the entire churn prediction pipeline
        trained_model, evaluation_metrics = run_churn_pipeline(
            data_file_path=str(data_file_path),
            target_column=TARGET_COLUMN,
            numeric_columns=NUMERIC_COLUMNS,
            categorical_columns=CATEGORICAL_COLUMNS,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            model_dir_path=str(model_dir_path),
            model_filename=MODEL_FILENAME,
            log_filename=LOG_FILENAME
        )

        print(f"\n{'='*60}")
        print("Pipeline completed successfully!")
        print(f"Final Model Accuracy: {evaluation_metrics['accuracy']:.4f}") # Using metrics from pipeline output

    except Exception as e:
        #print(f"\nERROR: Pipeline failed with exception: {e}")
        print(f"ERROR: Pipeline failed with exception: {e}", file=sys.stderr) # Direct output to stderr
        raise

if __name__ == "__main__":
    main()