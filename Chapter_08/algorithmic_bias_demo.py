import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

def demonstrate_historical_bias():
    print("--- 1. Historical Bias ---")
    print("Scenario: Past discriminatory practices skew the target variable.")
    # Example: Qualifications are identical across groups, but historical hiring 
    # data favors Group A due to past prejudice.
    
    n_samples = 1000
    group = np.random.choice(['A', 'B'], size=n_samples)
    qualifications = np.random.normal(70, 10, n_samples)
    
    # Calculate baseline probability of being hired based purely on merit
    hired_prob = (qualifications - 40) / 60 
    
    # Apply historical bias: Group B is penalized
    hired_prob[group == 'B'] -= 0.3 
    hired_prob = np.clip(hired_prob, 0, 1)
    
    hired = np.random.binomial(1, hired_prob)
    df = pd.DataFrame({'Group': group, 'Qualification': qualifications, 'Hired': hired})
    
    print(f"Avg Qualification - Group A: {df[df['Group'] == 'A']['Qualification'].mean():.1f}")
    print(f"Avg Qualification - Group B: {df[df['Group'] == 'B']['Qualification'].mean():.1f}")
    print(f"Hiring Rate - Group A: {df[df['Group'] == 'A']['Hired'].mean() * 100:.1f}%")
    print(f"Hiring Rate - Group B: {df[df['Group'] == 'B']['Hired'].mean() * 100:.1f}%")
    print("Insight: An AI trained on this data will perpetuate the historical bias.")
    
    # Visualization
    plt.figure(figsize=(8, 5))
    sns.barplot(data=df, x='Group', y='Hired', errorbar=None, palette='Set2', hue='Group', legend=False)
    plt.title('Historical Bias: Hiring Rates by Group\n(Both groups have equal average qualifications)')
    plt.ylabel('Hiring Rate')
    plt.ylim(0, 1)
    plt.savefig('historical_bias_viz.png')
    print("-> Saved visualization to 'historical_bias_viz.png'\n")
    plt.close()


def demonstrate_representation_bias():
    print("--- 2. Representation Bias ---")
    print("Scenario: A minority group is underrepresented in the training data.")
    
    # 90% Group A, 10% Group B
    n_samples = 1000
    group = np.random.choice(['A', 'B'], size=n_samples, p=[0.9, 0.1])
    feature = np.random.normal(50, 15, n_samples)
    
    # The true underlying pattern is slightly different for Group B
    target = np.where(group == 'A', (feature > 50).astype(int), (feature > 40).astype(int))
    
    # Train a single model on this skewed dataset
    X = feature.reshape(-1, 1)
    y = target
    model = LogisticRegression().fit(X, y)
    preds = model.predict(X)
    
    acc_A = accuracy_score(y[group == 'A'], preds[group == 'A'])
    acc_B = accuracy_score(y[group == 'B'], preds[group == 'B'])
    
    print(f"Model Accuracy for Majority Group A (90% of data): {acc_A * 100:.1f}%")
    print(f"Model Accuracy for Minority Group B (10% of data): {acc_B * 100:.1f}%")
    print("Insight: The model optimizes for the majority, performing poorly on the minority.")

    # Visualization
    results_df = pd.DataFrame({'Group': ['A (Majority)', 'B (Minority)'], 'Accuracy': [acc_A, acc_B]})
    plt.figure(figsize=(8, 5))
    sns.barplot(data=results_df, x='Group', y='Accuracy', palette='Set1', hue='Group', legend=False)
    plt.title('Representation Bias: Model Accuracy by Group')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1.1)
    plt.savefig('representation_bias_viz.png')
    print("-> Saved visualization to 'representation_bias_viz.png'\n")
    plt.close()


def demonstrate_measurement_bias():
    print("--- 3. Measurement Bias ---")
    print("Scenario: A proxy feature is measured differently across groups.")
    # Example: True behavior is identical, but stricter monitoring of Group B 
    # leads to artificially inflated proxy measurements (e.g., arrest rates vs actual crime).
    
    n_samples = 1000
    group = np.random.choice(['A', 'B'], size=n_samples)
    
    # True underlying metric is the same for both groups
    true_metric = np.random.normal(50, 10, n_samples)
    
    # Measured proxy feature contains systemic error for Group B
    measured_proxy = true_metric.copy()
    measured_proxy[group == 'B'] += 15 # Over-measured due to bias
    
    df = pd.DataFrame({'Group': group, 'True_Metric': true_metric, 'Measured_Proxy': measured_proxy})
    
    print(f"True Average Metric - Group A: {df[df['Group'] == 'A']['True_Metric'].mean():.1f}")
    print(f"True Average Metric - Group B: {df[df['Group'] == 'B']['True_Metric'].mean():.1f}")
    print(f"Measured Proxy Feature - Group A: {df[df['Group'] == 'A']['Measured_Proxy'].mean():.1f}")
    print(f"Measured Proxy Feature - Group B: {df[df['Group'] == 'B']['Measured_Proxy'].mean():.1f}")
    print("Insight: Garbage in, garbage out. The flawed proxy becomes the source of truth for the AI.")

    # Visualization
    plt.figure(figsize=(10, 5))
    df_melted = df.melt(id_vars='Group', value_vars=['True_Metric', 'Measured_Proxy'], 
                        var_name='Metric Type', value_name='Value')
    sns.boxplot(data=df_melted, x='Group', y='Value', hue='Metric Type', palette='Set3')
    plt.title('Measurement Bias: True vs Measured Metric by Group')
    plt.savefig('measurement_bias_viz.png')
    print("-> Saved visualization to 'measurement_bias_viz.png'\n")
    plt.close()


if __name__ == "__main__":
    demonstrate_historical_bias()
    demonstrate_representation_bias()
    demonstrate_measurement_bias()
