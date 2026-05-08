import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sdv.metadata import Metadata
from sdv.single_table import CTGANSynthesizer
from sdv.evaluation.single_table import evaluate_quality, run_diagnostic

def generate_mock_data(num_rows=1000):
    """Generates a simulated customer transaction log dataset."""
    print(f"Creating mock dataset with {num_rows} rows...")
    
    np.random.seed(42)
    
    # Generate customer IDs
    customer_ids = [f'CUST-{i:03d}' for i in range(1, 101)]
    
    # Generate merchant categories
    categories = ['Electronics', 'Groceries', 'Dining', 'Travel', 'Health', 'Subscription']
    
    # Create data
    data = {
        'transaction_id': [f'TXN-{i:06d}' for i in range(1, num_rows + 1)],
        'customer_id': np.random.choice(customer_ids, num_rows),
        'amount': np.round(np.random.exponential(scale=50, size=num_rows) + 5, 2),
        'timestamp': [datetime(2023, 1, 1) + timedelta(minutes=int(x)) for x in np.random.randint(0, 525600, num_rows)],
        'merchant_category': np.random.choice(categories, num_rows),
        'is_fraud': np.random.choice([0, 1], num_rows, p=[0.98, 0.02])
    }
    
    df = pd.DataFrame(data)
    print("Mock Data Sample:")
    print(df.head())
    return df

def run_sdv_ctgan_workflow():
    # 1. Prepare Data
    real_data = generate_mock_data(1000)
    
    # 2. Define Metadata
    print("\n--- Defining Metadata ---")
    metadata = Metadata.detect_from_dataframe(
        data=real_data,
        table_name='transactions'
    )
    
    # Update sdtype for IDs and set primary key
    metadata.update_column(
        column_name='transaction_id',
        sdtype='id'
    )
    metadata.set_primary_key(column_name='transaction_id')
    
    print("Metadata detected successfully.")
    # metadata.visualize() # This would show a diagram in a notebook environment
    
    # 3. Initialize and Train CTGAN
    # CTGAN is a GAN-based model designed specifically for tabular data.
    print("\n--- Training CTGAN Model ---")
    model = CTGANSynthesizer(
        metadata,
        epochs=500, # Number of training iterations
        batch_size=500,
        verbose=True
    )
    
    model.fit(real_data)
    print("Training complete.")
    
    # 4. Generate Synthetic Data
    # Once trained, we can generate as many synthetic records as needed.
    print("\n--- Generating Synthetic Data ---")
    synthetic_data = model.sample(num_rows=1000)
    
    print("Synthetic Data Sample:")
    print(synthetic_data.head())
    
    # 5. Evaluation
    # SDV provides built-in tools to compare statistical properties and privacy.
    print("\n--- Evaluating Synthetic Data ---")
    
    # Diagnostic report: Checks for basic data integrity and structure
    diagnostic = run_diagnostic(
        real_data=real_data,
        synthetic_data=synthetic_data,
        metadata=metadata
    )
    
    # Quality report: Compares statistical shapes and correlations
    quality_report = evaluate_quality(
        real_data=real_data,
        synthetic_data=synthetic_data,
        metadata=metadata
    )
    
    print("\nEvaluation Summary:")
    print(f"Quality Score: {quality_report.get_score():.2%}")
    
    # We can also get detailed scores for each column
    column_shapes = quality_report.get_details('Column Shapes')
    print("\nColumn Shape Scores:")
    print(column_shapes)

if __name__ == "__main__":
    run_sdv_ctgan_workflow()
