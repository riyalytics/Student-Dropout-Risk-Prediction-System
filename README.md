# Student Dropout Risk Prediction & Early Warning System

## 1. Business Problem Statement

**Context:**  
Educational institutions face significant financial and reputational losses when students drop out. For a university with 50,000 students, a 5% increase in retention can save millions in tuition revenue annually.

**Problem:**  
The current intervention process is reactive, relying on faculty referrals after students have already failed courses. The university lacks a proactive, data-driven method to identify at-risk students early in the semester.

**Objective:**  
Build an automated Early Warning System (EWS) that predicts the probability of student dropout using historical academic, behavioral, and financial data. The system will categorize students into Low, Medium, and High-Risk groups to enable targeted intervention by counselors.

---

## 2. Project Architecture

The solution is built using Python for data processing and modeling, with outputs designed for integration into Power BI dashboards.

**Tech Stack:**

- **Language:** Python 3.10+
- **Libraries:** Pandas, NumPy, Scikit-learn, Seaborn, Matplotlib
- **Modeling:** Logistic Regression, Random Forest Classifier
- **Dashboard:** Power BI (Integration Ready)

**Folder Structure:**

```
Student-Dropout-Risk-System/
│
├── data/
│   ├── student_dropout_raw.csv        # Synthetic dataset (500k+ records)
│   ├── student_dropout_processed.csv  # Cleaned & Scaled data
│
├── notebooks/
│   └── dropout_modeling.ipynb         # EDA and Modeling Walkthrough
│
├── output/
│   ├── student_dropout_scored.csv     # Final Scored Data for Power BI
│   ├── confusion_matrix.png           # Model Performance Visual
│   └── feature_importance.png         # Key Drivers Visual
│
├── src/
│   ├── data_generation.py             # Generates realistic synthetic data
│   ├── preprocessing.py               # Cleaning pipeline
│   ├── modeling.py                    # Training & Scoring pipeline
│
└── README.md                          # Project Documentation
```

---

## 3. Data Dictionary

The dataset contains 200,000 student records with the following key features:

| Feature | Description |
|---------|-------------|
| **Student_ID** | Unique identifier |
| **Attendance_Percentage** | % of classes attended (Key Driver) |
| **Average_Test_Score** | Average score across all subjects |
| **Fee_Payment_Delay_Days** | Days tuition payment was delayed |
| **Scholarship_Status** | Merit, Need-Based, or None |
| **Dropout** | Target Variable (0 = Retained, 1 = Dropped Out) |

---

## 4. Modeling Approach

We compared Logistic Regression and Random Forest. **Random Forest** was selected as the champion model due to its ability to handle non-linear relationships (e.g., the complex interaction between attendance and financial stress).

**Performance Metrics:**

- **Accuracy:** ~88%
- **ROC-AUC Score:** ~0.94
- **Recall (Dropout):** ~85% (Critical for catching at-risk students)

**Top Predictive Factors:**

1. Attendance Percentage
2. Average Test Score
3. Fee Payment Delay
4. Previous Failures
5. Assignment Completion Rate

---

## 5. Risk Scoring System

The model assigns a `Dropout_Probability` (0-1) to each student, categorized as:

- **Low Risk (0 - 0.3):** Standard academic support.
- **Medium Risk (0.3 - 0.6):** Automated email reminders, tutor suggestions.
- **High Risk (> 0.6):** Immediate counselor intervention required.

**Output:** `output/student_dropout_scored.csv`

---

## 6. Power BI Integration Guide

To visualize the results in Power BI:

1. **Import Data:**
   - Open Power BI Desktop -> Get Data -> Text/CSV.
   - Select `output/student_dropout_scored.csv`.

2. **Recommended Visuals:**
   - **Top Card:** Total High-Risk Students (Count where Risk='High').
   - **Pie Chart:** Risk Category Distribution.
   - **Bar Chart:** Average Attendance by Risk Category.
   - **Table:** List of High-Risk Students (filtered) for export to counselors.

---

## 7. Assumptions & Limitations

- **Synthetic Data:** The data is simulated based on real-world patterns but may not capture specific institutional nuances.
- **Static Snapshot:** The model currently predicts based on a snapshot in time. A future enhancement would be time-series forecasting (RNN/LSTM) to track risk trajectory week-over-week.
- **Bias:** Historical data may contain bias against certain demographics; fairness audits are recommended before full deployment.

## 8. Business Impact

- **Proactive vs. Reactive:** Shifts intervention from "after failure" to "during semester".
- **Revenue Retention:** Saving just 1% of the 500k student body (5,000 students) retains approx. $50M in tuition (assuming $10k/year).

---

**Author:** Senior Data Analyst  
**Date:** October 2024
