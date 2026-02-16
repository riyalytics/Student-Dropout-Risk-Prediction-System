
import nbformat as nbf
import os

nb = nbf.v4.new_notebook()

# Add meaningful cells
nb.cells = [
    nbf.v4.new_markdown_cell("""
# Student Dropout Risk Prediction & Early Warning System

**Project Overview:**
This notebook details the end-to-end process of building a machine learning model to predict student dropout risk.
The goal is to provide actionable insights for early intervention.

**Sections:**
1. Data Loading & Overview
2. Exploratory Data Analysis (EDA)
3. Data Preprocessing
4. Model Training & Evaluation
5. Feature Importance Analysis
6. Risk Scoring & Output
"""),
    
    nbf.v4.new_code_cell("""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

# Add src to system path to import modules
sys.path.append(os.path.abspath('../src'))

# Import custom modules
from preprocessing import load_data, preprocess_data
from modeling import train_and_evaluate, feature_importance_analysis, generate_risk_scores

# Set style
sns.set(style='whitegrid')
%matplotlib inline
"""),

    nbf.v4.new_markdown_cell("## 1. Data Loading"),

    nbf.v4.new_code_cell("""
# Load Raw Data
RAW_PATH = '../data/student_dropout_raw.csv'
df = load_data(RAW_PATH)
print(f"Dataset Shape: {df.shape}")
df.head()
"""),
    
    nbf.v4.new_markdown_cell("## 2. Exploratory Data Analysis (EDA)"),

    nbf.v4.new_code_cell("""
# Target Variable Distribution
plt.figure(figsize=(6,4))
sns.countplot(x='Dropout', data=df)
plt.title('Distribution of Dropout (Target)')
plt.show()

# Attendance vs Dropout
plt.figure(figsize=(10,6))
sns.histplot(data=df, x='Attendance_Percentage', hue='Dropout', kde=True, element='step')
plt.title('Attendance Percentage Distribution by Dropout Status')
plt.show()

# Boxplot of Fee Payment Delay
plt.figure(figsize=(10,6))
sns.boxplot(x='Dropout', y='Fee_Payment_Delay_Days', data=df)
plt.title('Fee Payment Delay vs Dropout')
plt.show()
"""),
    
    nbf.v4.new_code_cell("""
# Correlation Heatmap
# Select only numeric columns manually to avoid string errors
numeric_cols = df.select_dtypes(include=[np.number]).columns
plt.figure(figsize=(12,10))
sns.heatmap(df[numeric_cols].corr(), annot=False, cmap='coolwarm', fmt=".2f")
plt.title('Feature Correlation Heatmap')
plt.show()
"""),

    nbf.v4.new_markdown_cell("## 3. Data Preprocessing\nCleaning missing values, encoding categoricals, and scaling features."),

    nbf.v4.new_code_cell("""
# Run Preprocessing Pipeline
X_train, X_test, y_train, y_test, df_processed = preprocess_data(df)

print(f"Training Features Shape: {X_train.shape}")
print(f"Testing Features Shape: {X_test.shape}")
"""),

    nbf.v4.new_markdown_cell("## 4. Model Training & Evaluation\nWe train Logistic Regression and Random Forest models."),

    nbf.v4.new_code_cell("""
# Train and Evaluate Models
# This function trains both models and returns the Random Forest model
rf_model = train_and_evaluate(X_train, X_test, y_train, y_test)
"""),

    nbf.v4.new_markdown_cell("## 5. Feature Importance Analysis"),

    nbf.v4.new_code_cell("""
# Extract and Plot Feature Importance
feature_names = X_train.columns.tolist()
feature_importance_analysis(rf_model, feature_names)
"""),

    nbf.v4.new_markdown_cell("## 6. Risk Scoring & Deployment\nGenerating final risk scores for the dashboard."),

    nbf.v4.new_code_cell("""
# Generate Scores for the full dataset
# We use the full processed dataframe for scoring
feature_cols = X_train.columns.tolist()
generate_risk_scores(rf_model, df_processed, feature_cols)
"""),

    nbf.v4.new_code_cell("""
# Load and Inspect Final Output
df_final = pd.read_csv('../output/student_dropout_scored.csv')
print(df_final['Risk_Category'].value_counts(normalize=True))
df_final.head()
""")
]

# Write the notebook
with open('notebooks/dropout_modeling.ipynb', 'w') as f:
    nbf.write(nb, f)

print("Notebook created successfully.")
