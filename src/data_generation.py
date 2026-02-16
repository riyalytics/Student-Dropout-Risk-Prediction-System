"""
data_generation.py

Description:
    This script generates a synthetic but realistic dataset for the Student Dropout Risk Prediction project.
    It creates 200,000+ records with logical correlations (e.g., low attendance -> higher dropout probability).
    This dataset mimics real-world educational data often found in university databases.

Author: Senior Data Scientist
Date: 2024-10-24
"""

import pandas as pd
import numpy as np
import random
import os

# Set random seed for reproducibility
np.random.seed(42)

def generate_student_data(num_students=500000):
    """
    Generates a synthetic dataset for student dropout prediction.
    
    Parameters:
    - num_students: Number of records to generate.
    
    Returns:
    - DataFrame containing student data.
    """
    
    print(f"Generating data for {num_students} students...")

    # ---------------------------------------------------------
    # 1. BASE ATTRIBUTES
    # ---------------------------------------------------------
    student_ids = np.arange(1001, 1001 + num_students)
    
    # Age distribution: skew towards 18-22 (typical undergrad) but include some older students
    ages = np.random.choice(np.arange(17, 30), size=num_students, p=[0.1, 0.25, 0.25, 0.2, 0.1, 0.05, 0.02, 0.01, 0.01, 0.005, 0.003, 0.001, 0.001])
    
    genders = np.random.choice(['Male', 'Female'], size=num_students)
    
    # Grade levels: 0=Freshman, 1=Sophomore, 2=Junior, 3=Senior
    grade_levels = np.random.choice([0, 1, 2, 3], size=num_students, p=[0.3, 0.25, 0.25, 0.2])
    
    # ---------------------------------------------------------
    # 2. ACADEMIC PERFORMANCE (Correlated)
    # ---------------------------------------------------------
    # Attendance is key. We'll use a beta distribution to simulate real-world skew (most attend, some don't).
    attendance_pct = np.random.beta(a=7, b=2, size=num_students) * 100
    
    # Test scores correlated with attendance (Higher attendance -> Higher score generally, but with noise)
    # Base score + Attendance Bonus + Random Noise
    avg_test_score = np.random.normal(65, 10, num_students) + (attendance_pct * 0.2)
    avg_test_score = np.clip(avg_test_score, 30, 100) # Clip between 30 and 100
    
    # Assignment Completion Rate
    assignment_completion_rate = np.random.beta(a=6, b=2, size=num_students) * 100
    
    # Previous failures (Higher for lower attendance/scores)
    # Probability of failure increases if attendance < 60%
    fail_prob = np.where(attendance_pct < 60, 0.4, 0.05)
    previous_failures = np.random.poisson(lam=fail_prob)
    
    # ---------------------------------------------------------
    # 3. FINANCIAL & BEHAVIORAL FACTORS
    # ---------------------------------------------------------
    # Fee Payment Delay (Days) - Skewed distribution
    # Most pay on time (0 days), some delay.
    fee_payment_delay = np.random.exponential(scale=5, size=num_students).astype(int)
    # Add spikes for chronic defaulters
    random_defaulters = np.random.choice([0, 1], size=num_students, p=[0.9, 0.1])
    fee_payment_delay += random_defaulters * np.random.randint(30, 90, size=num_students)
    
    # Parent Meeting Count (More meetings -> often means trouble or high engagement, ambiguous feature)
    parent_meeting_count = np.random.poisson(lam=1, size=num_students)
    
    # Disciplinary Actions (0/1) - Correlated with low attendance
    disciplinary_prob = np.where(attendance_pct < 50, 0.3, 0.02)
    disciplinary_actions = np.random.binomial(n=1, p=disciplinary_prob)
    
    # ---------------------------------------------------------
    # 4. SOCIO-ECONOMIC FACTORS
    # ---------------------------------------------------------
    scholarship_status = np.random.choice(['None', 'Merit-Based', 'Need-Based'], size=num_students, p=[0.7, 0.15, 0.15])
    
    distance_from_school = np.random.exponential(scale=10, size=num_students) # distances in km
    
    internet_access = np.random.choice(['Yes', 'No'], size=num_students, p=[0.85, 0.15])
    
    # ---------------------------------------------------------
    # 5. TARGET VARIABLE: DROPOUT (0/1)
    # ---------------------------------------------------------
    # Logic defining risk (Probability of Dropout):
    # - Low Attendance (> 20% impact)
    # - Low Grades (> 20% impact)
    # - High Fee Delay (> 15% impact)
    # - Disciplinary Action (> 10% impact)
    # - Past Failures (> 10% impact)
    
    # Normalize features for risk calculation (approximate)
    norm_attendance = (100 - attendance_pct) / 100  # Low attendance -> High risk
    norm_score = (100 - avg_test_score) / 100       # Low score -> High risk
    norm_delay = np.clip(fee_payment_delay / 90, 0, 1)        # High delay -> High risk
    norm_fail = np.clip(previous_failures / 5, 0, 1)          # Many failures -> High risk
    
    risk_score = (
        (norm_attendance * 0.35) + 
        (norm_score * 0.25) + 
        (norm_delay * 0.20) + 
        (norm_fail * 0.10) +
        (disciplinary_actions * 0.10)
    )
    
    # Add some randomness to risk
    risk_score += np.random.normal(0, 0.05, num_students)
    
    # Threshold for dropout (Binary Target)
    # We want a realistic dropout rate, say ~15-20%
    threshold = np.percentile(risk_score, 80) # Top 20% risk score = Dropout
    dropout = (risk_score > threshold).astype(int)
    
    # Create DataFrame
    df = pd.DataFrame({
        'Student_ID': student_ids,
        'Age': ages,
        'Gender': genders,
        'Grade_Level': grade_levels,
        'Attendance_Percentage': np.round(attendance_pct, 1),
        'Average_Test_Score': np.round(avg_test_score, 1),
        'Assignment_Completion_Rate': np.round(assignment_completion_rate, 1),
        'Fee_Payment_Delay_Days': fee_payment_delay,
        'Previous_Failures': previous_failures,
        'Parent_Meeting_Count': parent_meeting_count,
        'Disciplinary_Actions': disciplinary_actions,
        'Scholarship_Status': scholarship_status,
        'Distance_From_School': np.round(distance_from_school, 1),
        'Internet_Access': internet_access,
        'Dropout': dropout
    })
    
    return df

if __name__ == "__main__":
    # Ensure data directory exists
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    os.makedirs(output_dir, exist_ok=True)
    
    df = generate_student_data(500000)
    
    output_path = os.path.join(output_dir, 'student_dropout_raw.csv')
    df.to_csv(output_path, index=False)
    
    print(f"Data generation complete. Saved to {output_path}")
    print(f"Dataset Shape: {df.shape}")
    print(df['Dropout'].value_counts(normalize=True))
