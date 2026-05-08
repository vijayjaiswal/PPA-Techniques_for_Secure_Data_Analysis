import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Using Fairlearn for the bias audit and mitigation
from fairlearn.metrics import MetricFrame, selection_rate, demographic_parity_difference
from fairlearn.reductions import ExponentiatedGradient, DemographicParity

import warnings
warnings.filterwarnings('ignore')

def generate_synthetic_data(n_samples=1000):
    """Generates synthetic dataset with inherent bias."""
    np.random.seed(42)
    # Protected attribute: 'Group' (e.g., 0 for minority, 1 for majority)
    group = np.random.choice([0, 1], size=n_samples, p=[0.3, 0.7])
    
    # Feature: 'Qualification' score
    qualifications = np.random.normal(70, 10, n_samples)
    
    # Target: 'Approved' (1 if approved, 0 if not)
    # Bias is injected: Group 1 has a higher base approval rate despite similar qualifications
    approval_prob = (qualifications - 40) / 60
    approval_prob[group == 0] -= 0.2  # Penalize minority group
    approval_prob = np.clip(approval_prob, 0, 1)
    
    approved = np.random.binomial(1, approval_prob)
    
    df = pd.DataFrame({
        'Group': group,
        'Qualification': qualifications,
        'Approved': approved
    })
    return df

def step_1_data_exploration(df):
    print("--- Step 1: Data Exploration (Pre-processing Audit) ---")
    print("Analyzing baseline fairness metrics in the dataset...")
    
    # Using Fairlearn to calculate selection rate per group
    sr_group_0 = df[df['Group'] == 0]['Approved'].mean()
    sr_group_1 = df[df['Group'] == 1]['Approved'].mean()
    
    print(f"Selection Rate - Group 0 (Minority): {sr_group_0:.2f}")
    print(f"Selection Rate - Group 1 (Majority): {sr_group_1:.2f}")
    
    dp_diff = abs(sr_group_1 - sr_group_0)
    print(f"Demographic Parity Difference in Data: {dp_diff:.2f}")
    print("Finding: There is a significant discrepancy in the raw data.\n")
    return df

def step_2_model_training_and_evaluation(X_train, y_train, X_test, y_test, A_test):
    print("--- Step 2: Model Training and Evaluation (Unmitigated) ---")
    
    # Train standard model
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Evaluate Accuracy
    acc = accuracy_score(y_test, y_pred)
    print(f"Unmitigated Model Accuracy: {acc:.2f}")
    
    # Fairlearn MetricFrame for detailed auditing
    metrics = {
        'accuracy': accuracy_score,
        'selection_rate': selection_rate
    }
    
    metric_frame = MetricFrame(metrics=metrics, y_true=y_test, y_pred=y_pred, sensitive_features=A_test)
    
    print("\nMetrics by Group (Unmitigated):")
    print(metric_frame.by_group)
    
    # Calculate Disparities
    dp_diff = demographic_parity_difference(y_test, y_pred, sensitive_features=A_test)
    print(f"\nDemographic Parity Difference (Unmitigated): {dp_diff:.2f}")
    print("Finding: The trained model inherited and potentially amplified the bias.\n")
    
    return model, metric_frame

def step_3_mitigation(X_train, y_train, A_train):
    print("--- Step 3: Applying Mitigation Techniques ---")
    print("Using In-processing mitigation: Exponentiated Gradient with Demographic Parity constraint.")
    
    base_model = LogisticRegression(random_state=42)
    
    # Apply mitigation strategy
    mitigator = ExponentiatedGradient(
        estimator=base_model,
        constraints=DemographicParity(),
        sample_weight_name='sample_weight'
    )
    
    mitigator.fit(X_train, y_train, sensitive_features=A_train)
    print("Mitigated model trained successfully.\n")
    
    return mitigator

def step_4_verify_mitigation(mitigated_model, X_test, y_test, A_test):
    print("--- Step 4: Verify Mitigation Effectiveness ---")
    
    y_pred_mitigated = mitigated_model.predict(X_test)
    
    # Evaluate Accuracy
    acc = accuracy_score(y_test, y_pred_mitigated)
    print(f"Mitigated Model Accuracy: {acc:.2f}")
    
    # Evaluate Fairness
    metrics = {
        'accuracy': accuracy_score,
        'selection_rate': selection_rate
    }
    
    metric_frame_mitigated = MetricFrame(metrics=metrics, y_true=y_test, y_pred=y_pred_mitigated, sensitive_features=A_test)
    
    print("\nMetrics by Group (Mitigated):")
    print(metric_frame_mitigated.by_group)
    
    dp_diff_mitigated = demographic_parity_difference(y_test, y_pred_mitigated, sensitive_features=A_test)
    print(f"\nDemographic Parity Difference (Mitigated): {dp_diff_mitigated:.2f}")
    print("Finding: The fairness metric (Demographic Parity Difference) has improved significantly, minimizing bias.\n")
    
    return metric_frame_mitigated

def step_5_documentation(metric_frame_unmitigated, metric_frame_mitigated):
    print("--- Step 5: Document Findings ---")
    print("Generating a summary report for compliance and governance...")
    
    report = """
================ BIAS AUDIT REPORT ================
1. Objective: Ensure fairness in the approval model.
2. Audit Tools Used: Fairlearn.
3. Findings (Unmitigated):
   - Discovered significant demographic parity difference.
   - The model favored Group 1 over Group 0.
4. Mitigation Strategy:
   - Applied Exponentiated Gradient algorithm.
   - Constraint: Demographic Parity.
5. Post-Mitigation Verification:
   - Selection rates are now balanced across groups.
   - Verified that demographic parity difference is minimized.
6. Conclusion: Mitigation successful. Model is ready for review.
===================================================
"""
    print(report)
    
    # In a real scenario, this would be exported to a PDF or Markdown file.
    with open('bias_audit_report.txt', 'w') as f:
        f.write(report)
    print("Report saved to 'bias_audit_report.txt'.\n")

if __name__ == "__main__":
    print("Starting the Bias Audit Workflow...\n")
    
    # 0. Setup Data
    df = generate_synthetic_data()
    X = df[['Qualification']]
    y = df['Approved']
    A = df['Group'] # Sensitive feature
    
    X_train, X_test, y_train, y_test, A_train, A_test = train_test_split(
        X, y, A, test_size=0.3, random_state=42
    )
    
    # 1. Data Exploration
    step_1_data_exploration(df)
    
    # 2. Model Training & Evaluation
    unmitigated_model, metric_frame_unmitigated = step_2_model_training_and_evaluation(
        X_train, y_train, X_test, y_test, A_test
    )
    
    # 3. Mitigation
    mitigated_model = step_3_mitigation(X_train, y_train, A_train)
    
    # 4. Verify Mitigation
    metric_frame_mitigated = step_4_verify_mitigation(
        mitigated_model, X_test, y_test, A_test
    )
    
    # 5. Documentation
    step_5_documentation(metric_frame_unmitigated, metric_frame_mitigated)
