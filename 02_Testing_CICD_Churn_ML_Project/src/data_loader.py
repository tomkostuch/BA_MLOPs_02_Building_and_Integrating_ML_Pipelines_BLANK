"""
data_loader.py

This script is responsible for loading the raw customer churn dataset.
It contains functions to read the data from a specified file path.
"""
import pandas as pd

# Assuming interfaces.py exists and defines DataLoader
# from src.interfaces import DataLoader # Not needed for static type checking in runtime if not inheriting

def load_churn_dataset(filepath: str) -> pd.DataFrame:
    """
    Loads the Customer Churn dataset from a CSV file.

    Args:
        filepath (str): Path to the CSV file.

    Returns:
        pd.DataFrame: The raw churn dataset.
    """
    try:
        df = pd.read_csv(filepath)
        print(f"Dataset loaded from CSV: {filepath}")
        print(f"Raw dataset: {len(df)} samples, {len(df.columns)} features")
        print(f"Available columns: {list(df.columns)}")
        print(f"Missing values per column:\n{df.isnull().sum()}")

        return df

    except Exception as e:
        raise RuntimeError(f"Failed to load churn dataset: {e}")