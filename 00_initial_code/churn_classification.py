import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
from typing import Optional, Any, cast 

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.linear_model import LogisticRegression
import joblib

# ──────────────────────────────────────────────────
# STATELESS HELPER FUNCTIONS
# ──────────────────────────────────────────────────

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
        ('imputer', SimpleImputer(strategy='median')),  # Handle potential NaNs
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),  # Handle potential NaNs
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))  # sparse_output=False for easier handling
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'  # Drop any columns not specified
    )
    return preprocessor

def transform_features(
    df: pd.DataFrame,
    target_column: str,
    numeric_columns: list[str], 
    categorical_columns: list[str]
) -> tuple[pd.DataFrame, ColumnTransformer, LabelEncoder]:
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
        label_encoder = LabelEncoder()
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
    
    # Store feature information for later use
    all_features = available_numeric + available_categorical
    df_transformed.attrs['feature_columns'] = all_features
    df_transformed.attrs['numeric_features'] = available_numeric
    df_transformed.attrs['categorical_features'] = available_categorical
    df_transformed.attrs['preprocessor'] = preprocessor
    
    print(f"Features for modeling: {all_features}")
    print(f"Preprocessing pipeline created with {len(available_numeric)} numeric and {len(available_categorical)} categorical features")
    
    return df_transformed, preprocessor, label_encoder

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
            output_lines.append(f"   {row}")
    elif confusion_matrix_data is not None:
        output_lines.append(f"   Unexpected format for confusion matrix: {confusion_matrix_data}")
    else:
        output_lines.append(f"   Confusion matrix data not available.")
        
    print("\n".join(output_lines))

# ──────────────────────────────────────────────────
# STATEFUL MODEL CLASS
# ──────────────────────────────────────────────────

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
        
        # Use provided classifier or default to LogisticRegression
        if classifier is None:
            self.classifier = LogisticRegression(
                random_state=self.random_state,
                max_iter=1000,
                class_weight='balanced'  # Handle potential class imbalance
            )
        else:
            self.classifier = classifier
        
        # Use provided preprocessor or create simple scaling pipeline
        if preprocessor is None:
            self.pipe = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', self.classifier)
            ])
        else:
            self.pipe = Pipeline([
                ('preprocessor', preprocessor),
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
    
    def save_run_log(self, directory: str, metrics: dict[str, Any], dataset_info: dict[str, Any]) -> None:
        """
        Save model configuration and performance metrics to a JSON file.
        
        Args:
            directory (str): Directory where the JSON file will be stored.
            metrics (Dict[str, Any]): Evaluation metrics to save.
            dataset_info (Dict[str, Any]): Information about the dataset used.
        """
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        
        run_info = {
            "timestamp": datetime.now().isoformat(),
            "model_class": "ChurnPredictionModel",
            "classifier": str(type(self.classifier).__name__),
            "dataset": "Customer Churn",
            "dataset_info": dataset_info,
            "parameters": {
                "random_state": self.random_state,
                "classifier_params": self.classifier.get_params()
            },
            "metrics": metrics
        }
        
        log_file = os.path.join(directory, "churn_model_run_log.json")
        
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

# ──────────────────────────────────────────────────
# MAIN ORCHESTRATOR
# ──────────────────────────────────────────────────

def main() -> None:
    """
    Main function to orchestrate the entire ML pipeline:
    1. Load and prepare data
    2. Split data
    3. Train model
    4. Evaluate performance
    5. Save model and logs
    """
    print("Starting Customer Churn Prediction Pipeline...")
    print("="*60)
    
    try:
        # Define column structure based on the churn dataset
        target_column = 'Churn'  # Target variable
        numeric_columns = ['tenure', 'MonthlyCharges', 'TotalCharges']
        categorical_columns = [
            'gender', 'SeniorCitizen', 'Partner', 'Dependents',
            'PhoneService', 'MultipleLines', 'InternetService',
            'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
            'TechSupport', 'StreamingTV', 'StreamingMovies',
            'Contract', 'PaperlessBilling', 'PaymentMethod'
        ]
        
        # Construct the absolute path to the data file (reverted to original path)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_file_path = os.path.join(script_dir, "data", "WA_Fn-UseC_-Telco-Customer-Churn.csv") # Path restored

        # Load and prepare data
        print("\n1. Loading dataset...")
        df_raw = load_churn_dataset(data_file_path)

        
        print("\n2. Cleaning data...")
        df_clean = clean_churn_data(df_raw, target_column, numeric_columns, categorical_columns)
        
        print("\n3. Transforming features...")
        df_transformed, preprocessor, label_encoder = transform_features(df_clean, target_column, numeric_columns, categorical_columns)
        
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
        
        # Split features and target
        print("\n4. Preparing features and target...")
        X, y = split_features_and_target(df_transformed)
        
        # Split into train/test
        print("\n5. Splitting data...")
        X_train, X_test, y_train, y_test = stratified_split(X, y, test_size=0.25, seed=42)
        print(f"Training set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        print(f"Train churn rate: {y_train.mean():.3f}")
        print(f"Test churn rate: {y_test.mean():.3f}")
        
        # Initialize and train model
        print("\n6. Training model...")
        model = ChurnPredictionModel(preprocessor=preprocessor, random_state=42)
        model.fit(X_train, y_train)
        
        # Make predictions
        print("\n7. Making predictions...")
        y_pred = model.predict(X_test)
        
        # Evaluate model
        print("\n8. Evaluating model...")
        metrics = compute_classification_metrics(
            y_test.to_numpy(),  # Use .to_numpy() for explicit np.ndarray conversion
            y_pred, 
            target_names=dataset_info['target_names']
        )
        
        # Report results
        report_classification_metrics(metrics)
        
        # Save model and logs
        print("\n9. Saving model and logs...")
        # Define the name of the directory for saved models
        saved_models_dirname = "saved_models"
        # Create the full path for the model directory relative to the script's location
        model_dir_path = os.path.join(script_dir, saved_models_dirname)
        
        # Create the full path for the model file
        model_file_path = os.path.join(model_dir_path, "churn_prediction_model_v1.joblib")
        
        model.save(model_file_path) # Pass the full path to the model file
        model.save_run_log(model_dir_path, metrics, dataset_info) # Pass the full path to the directory for logs
        
        print(f"\n{'='*60}")
        print("Pipeline completed successfully!")
        print(f"Model accuracy: {metrics['accuracy']:.4f}")
        print(f"Model saved to: {model_file_path}") # Updated to show the full path
        
    except Exception as e:
        print(f"\nERROR: Pipeline failed with exception: {e}")
        raise

# ──────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────

if __name__ == "__main__":
    main()