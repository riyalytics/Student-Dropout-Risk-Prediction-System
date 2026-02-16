"""
modeling.py

Description:
    This script handles Model Training, Evaluation, and Risk Scoring.
    1. Loads processed data.
    2. Trains Logistic Regression and Random Forest Classifier.
    3. Evaluates models (Accuracy, Precision, Recall, F1, ROC-AUC).
    4. Selects the best mode (Random Forest).
    5. Extracts Feature Importance.
    6. Generates Risk Scores (Probability of Dropout).
    7. Assigns Risk Categories (Low, Medium, High).
    8. Exports final dataset for Power BI dashboard.

Author: Senior Data Scientist
Date: 2024-10-24
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import os
import joblib

# Define paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, 'student_dropout_processed.csv')
SCORED_DATA_PATH = os.path.join(OUTPUT_DIR, 'student_dropout_scored.csv')

def load_processed_data(path):
    print(f"Loading processed data from {path}...")
    return pd.read_csv(path)

def train_and_evaluate(X_train, X_test, y_train, y_test):
    print("\n---------------------------------------------------")
    print("TRAINING MODELS")
    print("---------------------------------------------------")
    
    # 1. Logistic Regression
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    y_prob_lr = lr.predict_proba(X_test)[:, 1]
    
    print("\nLogistic Regression Results:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred_lr):.4f}")
    print(f"ROC-AUC: {roc_auc_score(y_test, y_prob_lr):.4f}")
    
    # 2. Random Forest Classifier
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    y_prob_rf = rf.predict_proba(X_test)[:, 1]
    
    print("\nRandom Forest Results:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred_rf):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred_rf):.4f}")
    print(f"F1 Score: {f1_score(y_test, y_pred_rf):.4f}")
    print(f"ROC-AUC: {roc_auc_score(y_test, y_prob_rf):.4f}")
    
    print("\nClassification Report (Random Forest):")
    print(classification_report(y_test, y_pred_rf))
    
    # Save Confusion Matrix
    cm = confusion_matrix(y_test, y_pred_rf)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix - Random Forest')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'))
    print(f"Confusion Matrix saved to {os.path.join(OUTPUT_DIR, 'confusion_matrix.png')}")
    
    return rf

def feature_importance_analysis(model, feature_names):
    print("\n---------------------------------------------------")
    print("FEATURE IMPORTANCE ANALYSIS")
    print("---------------------------------------------------")
    
    importance = model.feature_importances_
    df_imp = pd.DataFrame({'Feature': feature_names, 'Importance': importance})
    df_imp = df_imp.sort_values(by='Importance', ascending=False)
    
    print("Top 5 Drivers of Dropout Risk:")
    print(df_imp.head(5))
    
    # Plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=df_imp.head(10), palette='viridis')
    plt.title('Top 10 Features Driving Student Dropout Risk')
    plt.xlabel('Importance Score')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'feature_importance.png'))
    print(f"Feature Importance Chart saved to {os.path.join(OUTPUT_DIR, 'feature_importance.png')}")

def generate_risk_scores(model, df_full, feature_cols):
    print("\n---------------------------------------------------")
    print("GENERATING RISK SCORES & CATEGORIES")
    print("---------------------------------------------------")
    
    # We need the original untransformed data (or just the IDs) to map back, 
    # but since our processed data has all rows, we can predict on the full dataset 
    # (Note: In strict ML, we shouldn't predict on Train, but for this "Score All" task it's fine 
    # as we want to output a Score for every student for the dashboard).
    
    # Predict Probability using the SCALED features
    X_full = df_full[feature_cols]
    probabilities = model.predict_proba(X_full)[:, 1]
    
    # ---------------------------------------------------------
    # SCALE BACK FOR BUSINESS OUTPUT
    # For the final CSV, we want readable numbers (Age=20, not -1.2)
    # So we load the RAW data and append predictions to it.
    # ---------------------------------------------------------
    raw_path = os.path.join(DATA_DIR, 'student_dropout_raw.csv')
    if os.path.exists(raw_path):
        print("Merging scores with Raw Data for business-friendly output...")
        df_final = pd.read_csv(raw_path)
    else:
        # Fallback if raw not found (unlikely)
        df_final = df_full.copy()
        
    df_final['Dropout_Probability'] = probabilities
    
    # Define Risk Categories
    def categorize_risk(prob):
        if prob > 0.6:
            return 'High Risk'
        elif prob > 0.3:
            return 'Medium Risk'
        else:
            return 'Low Risk'
            
    df_final['Risk_Category'] = df_final['Dropout_Probability'].apply(categorize_risk)
    
    # Reorder columns: ID, Risk, Prob, then the rest
    if 'Student_ID' in df_final.columns:
        cols = ['Student_ID', 'Risk_Category', 'Dropout_Probability'] + [c for c in df_final.columns if c not in ['Student_ID', 'Risk_Category', 'Dropout_Probability']]
        df_final = df_final[cols]

    df_final.to_csv(SCORED_DATA_PATH, index=False)
    print(f"Final Scored Dataset saved to {SCORED_DATA_PATH}")
    print(df_final['Risk_Category'].value_counts(normalize=True))

if __name__ == "__main__":
    df = load_processed_data(PROCESSED_DATA_PATH)
    
    # Prepare X and y
    # Note: 'Dropout' is the target. 'Student_ID' might not be in processed file based on my preprocessing script (I dropped it).
    # If Preprocessing dropped it, good. If not, drop it here.
    
    target_col = 'Dropout'
    if target_col not in df.columns:
         raise ValueError(f"Target column '{target_col}' not found in dataset.")
         
    X = df.drop(columns=[target_col])
    # Also drop Student_ID if it exists (it shouldn't based on previous script, but safety check)
    if 'Student_ID' in X.columns:
        X = X.drop(columns=['Student_ID'])
        
    y = df[target_col]
    
    feature_names = X.columns.tolist()
    
    # Split for valid evaluation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Train
    rf_model = train_and_evaluate(X_train, X_test, y_train, y_test)
    
    # Feature Importance
    feature_importance_analysis(rf_model, feature_names)
    
    # Score Full Dataset
    generate_risk_scores(rf_model, df, feature_names)
