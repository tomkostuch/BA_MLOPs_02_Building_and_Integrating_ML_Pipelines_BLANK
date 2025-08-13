# ml_project/tests/integration/test_preprocessing_transform_integration.py
"""
Objective:
    Integration test for the preprocessing pipeline building and feature transformation,
    now implemented using Python's built-in `unittest` framework.
    Verifies that the preprocessor built by `build_preprocessing_pipeline`
    is correctly applied by `transform_features`, and attributes are set.

Tests Performed:
    - test_preprocessing_pipeline_and_transform_flow:
        Creates a dummy cleaned DataFrame, constructs a preprocessor,
        applies transformation, and asserts on the transformed DataFrame's
        structure, content, and attached attributes.
"""
import unittest
import pandas as pd
import numpy as np
import os
import sys
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# Add the project root to sys.path to allow imports from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Import functions and config
from src.preprocessing import build_preprocessing_pipeline, transform_features
from src.config import TARGET_COLUMN, NUMERIC_COLUMNS, CATEGORICAL_COLUMNS

class TestPreprocessingTransformIntegration(unittest.TestCase):

    def setUp(self):
        # Provides a dummy cleaned DataFrame for testing.
        # This simulates the output of `clean_churn_data`.
        self.dummy_cleaned_df = pd.DataFrame({
            # Numeric columns
            'tenure': [1, 34, 5, 20],
            'MonthlyCharges': [29.85, 56.95, 30.15, 80.0],
            'TotalCharges': [29.85, 1889.5, np.nan, 1600.0],
            # Categorical columns - ensuring all from config are present
            'gender': ['Male', 'Female', 'Male', 'Female'],
            'SeniorCitizen': [0, 1, 0, 0],
            'Partner': ['Yes', 'No', 'No', 'Yes'],
            'Dependents': ['No', 'No', 'Yes', 'No'],
            'PhoneService': ['No', 'Yes', 'Yes', 'Yes'],
            'MultipleLines': ['No phone service', 'No', 'Yes', 'No'],
            'InternetService': ['DSL', 'Fiber optic', 'DSL', 'No'],
            'OnlineSecurity': ['No', 'Yes', 'No', 'Yes'],
            'OnlineBackup': ['Yes', 'No', 'Yes', 'No'],
            'DeviceProtection': ['No', 'Yes', 'No', 'Yes'],
            'TechSupport': ['No', 'No', 'Yes', 'Yes'],
            'StreamingTV': ['No', 'Yes', 'No', 'Yes'],
            'StreamingMovies': ['No', 'Yes', 'No', 'Yes'],
            'Contract': ['Month-to-month', 'One year', 'Two year', 'Month-to-month'],
            'PaperlessBilling': ['Yes', 'No', 'Yes', 'No'],
            'PaymentMethod': ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'],
            # Original Churn column for target encoding
            'Churn': ['No', 'Yes', 'No', 'Yes']
        })
        # Add 'churn_binary' which `clean_churn_data` would add
        le = LabelEncoder()
        self.dummy_cleaned_df['churn_binary'] = le.fit_transform(self.dummy_cleaned_df['Churn'])
        self.dummy_cleaned_df.drop(columns=['Churn'], inplace=True)


    def test_preprocessing_pipeline_and_transform_flow(self):
        """
        Integration test: Verifies the flow from cleaned data to preprocessor
        building and feature transformation.
        """
        print("\n--- Integration Test: Preprocessing Pipeline and Transform Flow ---")

        # 1. Build preprocessing pipeline
        preprocessor = build_preprocessing_pipeline(NUMERIC_COLUMNS, CATEGORICAL_COLUMNS)
        self.assertIsInstance(preprocessor, ColumnTransformer)
        print("Preprocessing pipeline built successfully.")

        # 2. Transform features using the built pipeline
        df_transformed, fitted_preprocessor = transform_features(
            self.dummy_cleaned_df.copy(), # Pass a copy to avoid modifying original
            TARGET_COLUMN, # 'Churn' in original, 'churn_binary' in cleaned. transform_features expects original target
            NUMERIC_COLUMNS,
            CATEGORICAL_COLUMNS
        )

        # Assertions on the transformed DataFrame
        self.assertIsInstance(df_transformed, pd.DataFrame)
        self.assertIn('target_encoded', df_transformed.columns)
        self.assertTrue(pd.api.types.is_integer_dtype(df_transformed['target_encoded']))
        # Verify target encoding is just the churn_binary column
        self.assertTrue(df_transformed['target_encoded'].equals(self.dummy_cleaned_df['churn_binary']))
        print(f"DataFrame transformed. Shape: {df_transformed.shape}, Target encoded column present.")

        # Assertions on attributes
        self.assertIn('feature_columns', df_transformed.attrs)
        self.assertIn('preprocessor', df_transformed.attrs)
        self.assertIn('target_mapping', df_transformed.attrs)
        self.assertIn('target_names', df_transformed.attrs)
        self.assertIsInstance(df_transformed.attrs['preprocessor'], ColumnTransformer)
        print("DataFrame attributes correctly set.")

        # Check that the imputer within the fitted preprocessor has learned from the data.
        # This is a more direct way to verify that the NaN was handled during fitting.
        num_pipeline = fitted_preprocessor.named_transformers_['num'] # type: ignore
        imputer = num_pipeline.named_steps['imputer']
        
        # A successfully fitted SimpleImputer will have a `statistics_` attribute.
        self.assertTrue(hasattr(imputer, 'statistics_'))
        self.assertEqual(len(imputer.statistics_), len(NUMERIC_COLUMNS))
        print("Numeric imputer was successfully fitted on the data, handling NaNs.")


        # Check that feature_columns attribute stores the input feature names
        self.assertEqual(
            set(df_transformed.attrs['feature_columns']),
            set(NUMERIC_COLUMNS + CATEGORICAL_COLUMNS)
        )
        print("Feature columns attribute correctly stores original feature names.")
        print("--- Integration Test Complete ---")

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False, verbosity=2) # exit=False prevents sys.exit() from being called