import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from faker import Faker
from sdv.metadata import Metadata
from sdv.single_table import CTGANSynthesizer
from sdv.cag import Inequality
from sdv.evaluation.single_table import evaluate_quality, run_diagnostic

# Initialize Faker for realistic PII generation
fake = Faker()
Faker.seed(42)
np.random.seed(42)

def generate_healthcare_dataset(num_rows=1000):
    """
    Generates a mock healthcare dataset with realistic clinical and PII features.
    Demonstrates correlations (Age -> Health Condition -> Cost).
    """
    print(f"Generating {num_rows} mock healthcare records...")
    
    data = []
    conditions = ['Diabetes', 'Hypertension', 'Asthma', 'Healthy', 'Influenza']
    genders = ['Male', 'Female', 'Non-Binary']
    
    for i in range(num_rows):
        patient_id = f"PAT-{i+1:05d}"
        name = fake.name()
        email = fake.email()
        age = np.random.randint(18, 90)
        gender = np.random.choice(genders)
        
        # Correlate condition with age (simplified)
        if age > 60:
            condition = np.random.choice(conditions, p=[0.3, 0.4, 0.1, 0.1, 0.1])
        else:
            condition = np.random.choice(conditions, p=[0.1, 0.1, 0.2, 0.5, 0.1])
            
        # Correlate cost with condition
        base_cost = {'Diabetes': 1500, 'Hypertension': 800, 'Asthma': 500, 'Healthy': 100, 'Influenza': 300}
        cost = base_cost[condition] + np.random.normal(0, 50)
        cost = max(50.0, round(cost, 2))
        
        # Dates
        admission_date = fake.date_between(start_date='-2y', end_date='today')
        # Ensure discharge is after admission
        stay_duration = np.random.randint(1, 15)
        discharge_date = admission_date + timedelta(days=stay_duration)
        
        data.append({
            'patient_id': patient_id,
            'patient_name': name,
            'contact_email': email,
            'age': age,
            'gender': gender,
            'diagnosis': condition,
            'treatment_cost': cost,
            'admission_date': admission_date,
            'discharge_date': discharge_date
        })
        
    df = pd.DataFrame(data)
    # Ensure dates are datetime objects for SDV
    df['admission_date'] = pd.to_datetime(df['admission_date'])
    df['discharge_date'] = pd.to_datetime(df['discharge_date'])
    
    print("Mock Data Sample:")
    print(df.head())
    return df

def run_practical_synthetic_workflow():
    # 1. Prepare Real-World Like Data
    real_data = generate_healthcare_dataset(1000)
    
    # 2. Define Metadata and Handle PII
    # For healthcare data, anonymizing PII is a critical first step.
    print("\n--- Defining Metadata & Anonymization ---")
    metadata = Metadata.detect_from_dataframe(
        data=real_data,
        table_name='patients'
    )
    
    # Mark PII columns. SDV will replace these with fake values during sampling.
    metadata.update_column(
        column_name='patient_id',
        sdtype='id'
    )
    metadata.update_column(
        column_name='patient_name',
        sdtype='name',
        pii=True
    )
    metadata.update_column(
        column_name='contact_email',
        sdtype='email',
        pii=True
    )
    
    metadata.set_primary_key(column_name='patient_id')
    
    print("Metadata configured. PII columns: patient_name, contact_email (Anonymization Active)")
    
    # 3. Define Logical Constraints
    # In healthcare, data integrity rules must be preserved (e.g., Discharge > Admission).
    print("\n--- Adding Business Logic Constraints ---")
    discharge_constraint = Inequality(
        low_column_name='admission_date',
        high_column_name='discharge_date',
        strict_boundaries=False
    )
    
    # 4. Initialize and Train CTGAN
    # CTGAN is ideal for complex, correlated tabular data like clinical records.
    print("\n--- Training CTGAN Synthesizer ---")
    synthesizer = CTGANSynthesizer(
        metadata,
        epochs=300,
        verbose=True
    )
    
    # Apply constraints before training
    synthesizer.add_constraints(constraints=[discharge_constraint])
    
    synthesizer.fit(real_data)
    print("Training complete.")
    
    # 5. Generate Synthetic Records
    print("\n--- Generating Clean Synthetic Data ---")
    synthetic_data = synthesizer.sample(num_rows=1000)
    
    print("Synthetic Healthcare Data Sample:")
    print(synthetic_data.head())
    
    # Verify Anonymization: Check if any names from real data leaked into synthetic data
    leaked_names = set(real_data['patient_name']).intersection(set(synthetic_data['patient_name']))
    print(f"\nPrivacy Check: {len(leaked_names)} real names found in synthetic data.")
    
    # 6. Evaluation
    print("\n--- Running Utility & Quality Evaluation ---")
    
    diagnostic = run_diagnostic(
        real_data=real_data,
        synthetic_data=synthetic_data,
        metadata=metadata
    )
    
    quality_report = evaluate_quality(
        real_data=real_data,
        synthetic_data=synthetic_data,
        metadata=metadata
    )
    
    print(f"\nOverall Quality Score: {quality_report.get_score():.2%}")
    
    # Detailed utility for cost and age correlations
    details = quality_report.get_details('Column Shapes')
    print("\nClinical Utility (Column Distributions):")
    print(details[details['Column'].isin(['age', 'treatment_cost', 'diagnosis'])])

if __name__ == "__main__":
    run_practical_synthetic_workflow()
