"""
preprocessing.py

This script contains functions for data cleaning and preprocessing.
It handles tasks such as converting data types, encoding categorical variables,
scaling numerical features, and splitting the data into training and testing sets.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder


def clean_churn_data(
    df: pd.DataFrame,
    target_column: str,
    numeric_columns: list[str],
    categorical_columns: list[str]
) -> pd.DataFrame:
    """
    Cleans the churn dataset by validating required columns and basic data type handling.

    Args:
        df (pd.DataFrame): Raw dataset.
        target_column (str): Target column that must be present.
        numeric_columns (list[str]): Numeric feature columns.
        categorical_columns (list[str]): Categorical feature columns.

    Returns:
        pd.DataFrame: Cleaned dataset.
    """
    df_clean = df.copy()

    print(f"Initial data shape: {df_clean.shape}")

    # Validate that required columns are present
    all_required_columns = [target_column] + numeric_columns + categorical_columns
    missing_columns = [col for col in all_required_columns if col not in df_clean.columns]

    if missing_columns:
        raise ValueError(f"Missing required columns in dataset: {missing_columns}")

    print(f"All required columns found: {all_required_columns}")

    # Keep only the required columns
    df_clean = df_clean[all_required_columns].copy()
    print(f"Kept only required columns: {list(df_clean.columns)}")

    print(f"Missing values before cleaning:\n{df_clean.isnull().sum()}")

    # Convert TotalCharges to numeric (it might be stored as string)
    if 'TotalCharges' in df_clean.columns:
        df_clean['TotalCharges'] = pd.to_numeric(df_clean['TotalCharges'], errors='coerce')
        print(f"Converted TotalCharges to numeric. New missing values: {df_clean['TotalCharges'].isnull().sum()}")

    # Clean target variable - standardize Yes/No to 1/0
    if target_column in df_clean.columns:
        df_clean['churn_binary'] = df_clean[target_column].map({'Yes': 1, 'No': 0})
        print(f"Target variable distribution:\n{df_clean['churn_binary'].value_counts()}")

    # Define the set of columns that should be in the final output DataFrame
    final_output_columns = numeric_columns + categorical_columns + ['churn_binary']

    # Select only these columns
    # Use .copy() to ensure this is a new DataFrame and avoid potential warnings later
    df_clean = df_clean[final_output_columns].copy()
    
    print(f"Final columns in returned DataFrame: {list(df_clean.columns)}")

    print(f"Final data shape after cleaning: {df_clean.shape}")

    return df_clean

def build_preprocessing_pipeline(numeric_features: list[str], categorical_features: list[str]) -> ColumnTransformer:
    """
    Builds a ColumnTransformer for preprocessing numerical and categorical features.

    Args:
        numeric_features (list[str]): list of numerical feature names.
        categorical_features (list[str]): list of categorical feature names.

    Returns:
        ColumnTransformer: The configured ColumnTransformer.
    """
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'
    )
    return preprocessor

def transform_features(
    df: pd.DataFrame,
    target_column: str,
    numeric_columns: list[str],
    categorical_columns: list[str]
) -> tuple[pd.DataFrame, ColumnTransformer]:
    
    #tuple[pd.DataFrame, ColumnTransformer, LabelEncoder]
    """
    Transforms features by encoding target variable and creating preprocessing pipeline.

    Args:
        df (pd.DataFrame): Cleaned dataset.
        target_column (str): Target column name.
        numeric_columns (list[str]): Numeric feature columns.
        categorical_columns (list[str]): Categorical feature columns.

    Returns:
        Tuple[pd.DataFrame, ColumnTransformer, LabelEncoder]:
            - Dataset with encoded target
            - Preprocessing pipeline for features
            - Label encoder for target variable
    """
    df_transformed = df.copy()

    # Encode target variable (churn_binary already created in cleaning)
    if 'churn_binary' in df_transformed.columns:
        # For churn, we don't need label encoding since it's already 0/1
        # But we'll create a dummy encoder for consistency

        #label_encoder = LabelEncoder()
        df_transformed['target_encoded'] = df_transformed['churn_binary']

        # Store mapping information
        churn_mapping = {0: 'No Churn', 1: 'Churn'}
        df_transformed.attrs['target_mapping'] = churn_mapping
        df_transformed.attrs['target_names'] = ['No Churn', 'Churn']

        print(f"Target encoding - Churn mapping: {churn_mapping}")
    else:
        raise ValueError("Churn binary column not found in dataset")

    # Filter available features
    available_numeric = [col for col in numeric_columns if col in df_transformed.columns]
    available_categorical = [col for col in categorical_columns if col in df_transformed.columns]

    print(f"Available numeric features: {available_numeric}")
    print(f"Available categorical features: {available_categorical}")

    # Build preprocessing pipeline
    preprocessor = build_preprocessing_pipeline(available_numeric, available_categorical)

    # Fit the preprocessor on the feature data
    features_to_fit = df_transformed[available_numeric + available_categorical]
    if not features_to_fit.empty: # Ensure there are features to fit
        preprocessor.fit(features_to_fit)

    # Store feature information for later use
    all_features = available_numeric + available_categorical
    df_transformed.attrs['feature_columns'] = all_features
    df_transformed.attrs['numeric_features'] = available_numeric
    df_transformed.attrs['categorical_features'] = available_categorical
    df_transformed.attrs['preprocessor'] = preprocessor

    print(f"Features for modeling: {all_features}")
    print(f"Preprocessing pipeline created with {len(available_numeric)} numeric and {len(available_categorical)} categorical features")

    #return df_transformed, preprocessor, label_encoder
    #return df_transformed, preprocessor, label_encoder
    return df_transformed, preprocessor

def split_features_and_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Splits the DataFrame into features and target variable.

    Args:
        df (pd.DataFrame): Transformed dataset.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: Features (X) and target (y).
    """
    # Get feature columns from transformation step
    feature_columns = df.attrs.get('feature_columns', [])

    if not feature_columns:
        raise ValueError("No feature columns found in dataset attributes")

    # Ensure all required features are present
    missing_features = [f for f in feature_columns if f not in df.columns]
    if missing_features:
        raise ValueError(f"Missing required features: {missing_features}")

    X = df[feature_columns].copy()
    y = df['target_encoded'].copy()

    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Features used: {list(X.columns)}")

    return X, y

def stratified_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.25,
    seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Splits the data into train and test sets with stratification.

    Args:
        X (pd.DataFrame): Features.
        y (pd.Series): Target.
        test_size (float): Proportion of test data.
        seed (int): Random seed for reproducibility.

    Returns:
        Tuple: Split data - X_train, X_test, y_train, y_test
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    # Cast to tuple to match the type hint
    return X_train, X_test, y_train, y_test