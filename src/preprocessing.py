"""
preprocessing.py

Description:
    This script handles data cleaning, feature engineering, and preprocessing for the Student Dropout Prediction model.
    It performs the following steps:
    1. Loads raw data.
    2. Handles missing values (imputation).
    3. Encodes categorical variables (One-Hot Encoding / Label Encoding).
    4. Scales numerical features (StandardScaler).
    5. Splits data into training and testing sets.
    6. Saves the processed dataset for modeling.

    Production Note: In a real deployment, we would save the scaler and encoder objects (pickles) to apply to new incoming data.

Author: Senior Data Analyst
Date: 2024-10-24
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import os

# Define paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_PATH = os.path.join(DATA_DIR, 'student_dropout_raw.csv')
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, 'student_dropout_processed.csv')

def load_data(path):
    """Loads the dataset from a CSV file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found at {path}")
    print(f"Loading data from {path}...")
    return pd.read_csv(path)

def preprocess_data(df):
    """
    Cleans and processes the dataframe.
    
    Args:
        df: Raw pandas DataFrame.
        
    Returns:
        X_train, X_test, y_train, y_test: Split datasets ready for modeling.
        df_processed: Full processed dataframe (optional, for verification).
    """
    
    # ---------------------------------------------------------
    # 1. HANDLING MISSING VALUES
    # ---------------------------------------------------------
    # Check for missing values
    missing_cols = df.columns[df.isnull().any()].tolist()
    if missing_cols:
        print(f"Missing values found in: {missing_cols}. Imputing...")
        # Numeric strategies: Mean/Median
        # Categorical strategies: Mode/Constant
        num_imputer = SimpleImputer(strategy='median')
        cat_imputer = SimpleImputer(strategy='most_frequent')
        
        # Separate numeric and categorical
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        df[numeric_cols] = num_imputer.fit_transform(df[numeric_cols])
        df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])
    else:
        print("No missing values detected.")

    # ---------------------------------------------------------
    # 2. ENCODING CATEGORICAL VARIABLES
    # ---------------------------------------------------------
    print("Encoding categorical variables...")
    
    # Binary encoding for simple categories
    le = LabelEncoder()
    df['Gender'] = le.fit_transform(df['Gender']) # 0=Female, 1=Male (or vice versa, check mapping)
    df['Internet_Access'] = le.fit_transform(df['Internet_Access']) # 0=No, 1=Yes
    
    # One-Hot Encoding for multi-class categories
    # Scholarship_Status (None, Merit, Need) -> We want dummy variables
    df = pd.get_dummies(df, columns=['Scholarship_Status'], drop_first=True)
    
    # ---------------------------------------------------------
    # 3. FEATURE SCALING
    # ---------------------------------------------------------
    print("Scaling numerical features...")
    
    # Features to scale (exclude Target and ID)
    features_to_scale = [
        'Age', 'Attendance_Percentage', 'Average_Test_Score', 
        'Assignment_Completion_Rate', 'Fee_Payment_Delay_Days',
        'Distance_From_School'
    ]
    
    scaler = StandardScaler()
    df[features_to_scale] = scaler.fit_transform(df[features_to_scale])
    
    # ---------------------------------------------------------
    # 4. TRAIN-TEST SPLIT
    # ---------------------------------------------------------
    print("Splitting data into Train and Test sets (80-20)...")
    
    X = df.drop(columns=['Dropout', 'Student_ID']) # Drop ID as it's not predictive
    y = df['Dropout']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"Train Set Shape: {X_train.shape}")
    print(f"Test Set Shape: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test, df

if __name__ == "__main__":
    # Load
    df_raw = load_data(RAW_DATA_PATH)
    
    # Preprocess
    _, _, _, _, df_processed = preprocess_data(df_raw)
    
    # Save processed full dataset
    df_processed.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"Processed dataset saved to {PROCESSED_DATA_PATH}")
