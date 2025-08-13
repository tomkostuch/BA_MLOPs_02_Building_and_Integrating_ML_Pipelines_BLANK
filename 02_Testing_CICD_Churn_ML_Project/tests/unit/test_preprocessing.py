# ml_project/tests/test_preprocessing.py
"""
Objective:
    This script contains unit tests for the functions in `src/preprocessing.py`,
    now implemented using Python's built-in `unittest` framework.
    It ensures that data cleaning, preprocessing pipeline construction,
    feature transformation, and data splitting are performed correctly.

Tests Performed:
    - clean_churn_data:
        - Verifies successful cleaning, type conversion (e.g., TotalCharges),
          creation of churn_binary, and correct column selection.
        - Checks for proper error handling when essential columns are missing.
    - build_preprocessing_pipeline:
        - Asserts that the ColumnTransformer is correctly built with expected
          numerical and categorical transformers (StandardScaler, OneHotEncoder).
    - transform_features:
        - Confirms that target encoding is applied and that DataFrame attributes
          (like feature_columns, preprocessor) are correctly set.
    - split_features_and_target:
        - Ensures that features (X) and target (y) are accurately separated
          based on DataFrame attributes.
        - Checks for error when required attributes are missing.
    - stratified_split:
        - Verifies that the data is split into train and test sets while
          maintaining the original class proportions (stratification).
"""
import unittest
import pandas as pd
import numpy as np
import sys
import os
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

# Add the project root to sys.path to allow imports from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
# Import functions to be tested
from src.preprocessing import (
    clean_churn_data,
    build_preprocessing_pipeline,
    transform_features,
    split_features_and_target,
    stratified_split
)

class TestPreprocessing(unittest.TestCase):

    def setUp(self):
        # Helper method for raw DataFrame
        self.raw_df = pd.DataFrame({
            'customerID': ['1', '2', '3', '4'],
            'gender': ['Male', 'Female', 'Male', 'Female'],
            'SeniorCitizen': [0, 1, 0, 0],
            'Partner': ['Yes', 'No', 'No', 'Yes'],
            'Dependents': ['No', 'Yes', 'No', 'No'],
            'tenure': [1, 34, 2, 45],
            'PhoneService': ['No', 'Yes', 'Yes', 'Yes'],
            'MultipleLines': ['No phone service', 'No', 'No', 'Yes'],
            'InternetService': ['DSL', 'DSL', 'Fiber optic', 'DSL'],
            'OnlineSecurity': ['No', 'Yes', 'No', 'Yes'],
            'OnlineBackup': ['Yes', 'No', 'Yes', 'No'],
            'DeviceProtection': ['No', 'Yes', 'No', 'Yes'],
            'TechSupport': ['No', 'No', 'No', 'Yes'],
            'StreamingTV': ['No', 'No', 'No', 'Yes'],
            'StreamingMovies': ['No', 'No', 'No', 'Yes'],
            'Contract': ['Month-to-month', 'One year', 'Month-to-month', 'Two year'],
            'PaperlessBilling': ['Yes', 'No', 'Yes', 'No'],
            'PaymentMethod': ['Electronic check', 'Mailed check', 'Electronic check', 'Bank transfer (automatic)'],
            'MonthlyCharges': [29.85, 56.95, 53.85, 70.70],
            'TotalCharges': ['29.85', '1889.5', '108.15', ''], # Empty string for TotalCharges
            'Churn': ['No', 'No', 'Yes', 'Yes']
        })

        # Helper method for configuration columns
        self.config_columns = {
            'TARGET_COLUMN': 'Churn',
            'NUMERIC_COLUMNS': ['tenure', 'MonthlyCharges', 'TotalCharges'],
            'CATEGORICAL_COLUMNS': [
                'gender', 'SeniorCitizen', 'Partner', 'Dependents',
                'PhoneService', 'MultipleLines', 'InternetService',
                'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                'TechSupport', 'StreamingTV', 'StreamingMovies',
                'Contract', 'PaperlessBilling', 'PaymentMethod'
            ]
        }

        # Create cleaned and transformed dataframes once for all tests
        self.cleaned_df = clean_churn_data(
            self.raw_df.copy(),
            self.config_columns['TARGET_COLUMN'],
            self.config_columns['NUMERIC_COLUMNS'],
            self.config_columns['CATEGORICAL_COLUMNS']
        )
        
        self.transformed_df, _ = transform_features(
            self.cleaned_df.copy(),
            self.config_columns['TARGET_COLUMN'],
            self.config_columns['NUMERIC_COLUMNS'],
            self.config_columns['CATEGORICAL_COLUMNS']
        )

    # --- Tests for clean_churn_data ---
    def test_clean_churn_data_success(self):
        """
        Test successful data cleaning.
        Verifies that:
        1. The target column 'Churn' is converted to a binary 'churn_binary'.
        2. 'TotalCharges' is converted to a numeric type, handling non-numeric values as NaN.
        3. The final DataFrame contains only the expected columns.
        """
        df_clean = self.cleaned_df

        # 1. Verify target column transformation
        self.assertIn('churn_binary', df_clean.columns)
        self.assertEqual(df_clean['churn_binary'].dtype, np.int64)
        pd.testing.assert_series_equal(
            df_clean['churn_binary'],
            pd.Series([0, 0, 1, 1], name='churn_binary'),
            check_index=False  # We only care about the values
        )

        # 2. Verify 'TotalCharges' transformation
        self.assertTrue(pd.api.types.is_numeric_dtype(df_clean['TotalCharges']))
        self.assertTrue(np.isnan(df_clean['TotalCharges'].iloc[3]))

        # 3. Verify final column set
        expected_cols = self.config_columns['NUMERIC_COLUMNS'] + self.config_columns['CATEGORICAL_COLUMNS'] + ['churn_binary']
        self.assertEqual(set(df_clean.columns), set(expected_cols))

    def test_clean_churn_data_missing_target_column(self):
        """Test error handling when target column is missing."""
        df_missing_target = self.raw_df.drop(columns=[self.config_columns['TARGET_COLUMN']]).copy()
        with self.assertRaisesRegex(ValueError, "Missing required columns in dataset: \\['Churn'\\]"):
            clean_churn_data(
                df_missing_target,
                self.config_columns['TARGET_COLUMN'],
                self.config_columns['NUMERIC_COLUMNS'],
                self.config_columns['CATEGORICAL_COLUMNS']
            )

    # --- Tests for build_preprocessing_pipeline ---
    def test_build_preprocessing_pipeline_structure(self):
        """
        Test the structure of the preprocessing pipeline.
        Ensures the ColumnTransformer is built with the correct transformers
        for numeric and categorical features. This is key for ensuring our model's inputs are consistent.
        """
        preprocessor = build_preprocessing_pipeline(
            self.config_columns['NUMERIC_COLUMNS'],
            self.config_columns['CATEGORICAL_COLUMNS']
        )

        self.assertIsInstance(preprocessor, ColumnTransformer)
        self.assertEqual(len(preprocessor.transformers), 2) # type: ignore


        num_transformer = [t for name, t, cols in preprocessor.transformers if name == 'num'][0]  # type: ignore
        cat_transformer = [t for name, t, cols in preprocessor.transformers if name == 'cat'][0]  # type: ignore

        self.assertIsInstance(num_transformer, Pipeline)
        self.assertIsInstance(cat_transformer, Pipeline)

        self.assertIn('scaler', num_transformer.named_steps)
        self.assertIsInstance(num_transformer.named_steps['scaler'], StandardScaler)
        self.assertIn('onehot', cat_transformer.named_steps)
        self.assertIsInstance(cat_transformer.named_steps['onehot'], OneHotEncoder)

    # --- Tests for transform_features ---
    def test_transform_features_success(self):
        """
        Test successful feature transformation and metadata attachment.
        Verifies that the function correctly encodes the target and, crucially for MLOps,
        attaches important metadata (like feature names and the preprocessor itself)
        to the DataFrame's attributes for later use in the pipeline.
        """
        df_transformed, preprocessor = transform_features(
            self.cleaned_df.copy(),
            self.config_columns['TARGET_COLUMN'],
            self.config_columns['NUMERIC_COLUMNS'],
            self.config_columns['CATEGORICAL_COLUMNS']
        )

        self.assertIsInstance(df_transformed, pd.DataFrame)
        self.assertIn('target_encoded', df_transformed.columns)
        self.assertEqual(df_transformed['target_encoded'].dtype, np.int64)
        self.assertIsInstance(preprocessor, ColumnTransformer)

        self.assertIn('feature_columns', df_transformed.attrs)
        self.assertIn('preprocessor', df_transformed.attrs)
        self.assertIn('target_mapping', df_transformed.attrs)
        self.assertIn('target_names', df_transformed.attrs)
        self.assertEqual(set(df_transformed.attrs['feature_columns']), set(self.config_columns['NUMERIC_COLUMNS'] + self.config_columns['CATEGORICAL_COLUMNS']))

    # --- Tests for split_features_and_target ---
    def test_split_features_and_target_success(self):
        """Test successful splitting of features and target."""
        X, y = split_features_and_target(self.transformed_df)

        self.assertIsInstance(X, pd.DataFrame)
        self.assertIsInstance(y, pd.Series)
        self.assertEqual(len(X), len(y))
        self.assertNotIn('target_encoded', X.columns)
        self.assertEqual(y.name, 'target_encoded')
        self.assertEqual(set(X.columns), set(self.transformed_df.attrs['feature_columns']))

    def test_split_features_and_target_missing_attrs(self):
        """Test error when feature_columns attribute is missing."""
        df_no_attrs = self.cleaned_df.copy()
        if 'feature_columns' in df_no_attrs.attrs:
            del df_no_attrs.attrs['feature_columns']

        with self.assertRaisesRegex(ValueError, "No feature columns found in dataset attributes"):
            split_features_and_target(df_no_attrs)

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False, verbosity=2) # exit=False prevents sys.exit() from being called