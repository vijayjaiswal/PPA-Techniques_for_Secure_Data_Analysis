import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from snsynth import Synthesizer
import matplotlib.pyplot as plt

"""
Section 3.c: Implementing Differential Privacy within Generative Models
This reference script demonstrates how to strengthen privacy guarantees by 
integrating Differential Privacy (DP) into the training of a CTGAN model.

Key Techniques Demonstrated:
1. DP-SGD (Differentially Private Stochastic Gradient Descent)
2. Privacy Budget (Epsilon ε and Delta δ)
3. Noise Injection & Gradient Clipping
"""

def generate_sensitive_healthcare_data(num_rows=1000):
    """Generates mock healthcare data with sensitive medical and financial fields."""
    np.random.seed(42)
    categories = ['Cardiology', 'Oncology', 'Neurology', 'General']
    
    data = {
        'patient_id': np.arange(1, num_rows + 1),
        'age': np.random.randint(20, 85, num_rows),
        'department': np.random.choice(categories, num_rows),
        'treatment_cost': np.random.lognormal(mean=8, sigma=0.8, size=num_rows),
        'is_high_risk': np.random.choice([0, 1], num_rows, p=[0.8, 0.2])
    }
    
    df = pd.DataFrame(data)
    # Ensure treatment_cost is rounded (standard currency)
    df['treatment_cost'] = df['treatment_cost'].round(2)
    return df

def train_dp_generative_model():
    # 1. Load Sensitive Data
    real_data = generate_sensitive_healthcare_data(2000)
    print("Sample of Sensitive Healthcare Data:")
    print(real_data.head())

    # 2. Configure Privacy Parameters
    # Epsilon (ε) is the privacy budget. Lower ε = Stronger Privacy but higher noise.
    # Delta (δ) is the 'failure' probability. A common target is 1/n_samples.
    epsilon = 1.0  # Rigorous privacy target
    delta = 1e-5   # Probability of privacy breach occurring
    
    print(f"\n--- Initializing DP-CTGAN with epsilon={epsilon}, delta={delta} ---")
    
    # 3. Initialize the Differentially Private Synthesizer
    # SmartNoise (snsynth) uses DP-SGD internally. 
    # DP-SGD integrates DP by:
    #   a. Gradient Clipping: Ensuring no single record's gradient has an outsized influence.
    #   b. Noise Injection: Adding calibrated Laplacian or Gaussian noise to the gradients.
    synth = Synthesizer.create(
        "dpctgan", 
        epsilon=epsilon, 
        delta=delta, 
        verbose=True
    )

    # 4. Training (DP-SGD Integration)
    # The 'fit' process allocates the total ε budget.
    # PART of the budget (preprocessor_eps) is used to safely learn the data range/bins.
    print("\nTraining DP-CTGAN model...")
    synth.fit(
        real_data, 
        preprocessor_eps=0.1 # Using 10% of budget for metadata analysis
    )
    print("Differentially Private training complete.")

    # 5. Generate Synthetic Records
    # These records have a formal mathematical guarantee against reconstruction attacks.
    print("\n--- Generating DP Synthetic Data ---")
    synthetic_data = synth.sample(1000)
    
    print("DP Synthetic Data Sample:")
    print(synthetic_data.head())

    # 6. Privacy-Utility Tradeoff Analysis
    # Let's compare the mean treatment cost to see how DP noise affected accuracy.
    real_mean = real_data['treatment_cost'].mean()
    synth_mean = synthetic_data['treatment_cost'].mean()
    
    print("\n--- Privacy-Utility Analysis ---")
    print(f"Real Mean Treatment Cost: ${real_mean:,.2f}")
    print(f"Synthetic Mean Treatment Cost (epsilon={epsilon}): ${synth_mean:,.2f}")
    print(f"Difference (DP Noise Impact): ${abs(real_mean - synth_mean):,.2f}")

    # Note on Implementation:
    # While traditional models might memorize outliers (e.g., a patient with $1M cost),
    # the DP-SGD mechanism clips those gradients, ensuring the outlier's presence 
    # doesn't meaningfully alter the final model weights beyond the ε-bound.

if __name__ == "__main__":
    # Ensure smartnoise-synth is installed: pip install smartnoise-synth
    try:
        train_dp_generative_model()
    except ImportError:
        print("\n[ERROR] Library 'smartnoise-synth' not found.")
        print("To run this differentially private example, please install it via:")
        print("pip install smartnoise-synth")
