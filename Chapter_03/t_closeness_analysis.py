import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance

# ==========================================
# 1. Setup & Helper Functions
# ==========================================

def get_distribution(values):
    """
    Converts a list/series of values into a probability distribution.
    Returns the values and their corresponding probabilities.
    """
    values = np.array(values)
    unique_vals, counts = np.unique(values, return_counts=True)
    probs = counts / len(values)
    return unique_vals, probs

def calculate_emd_numerical(global_series, local_series):
    """
    Calculates Earth Mover's Distance (Wasserstein Distance) for numerical data.
    This metric respects the magnitude of values (e.g., 30k is closer to 40k than 100k).
    """
    # scipy's wasserstein_distance accepts two arrays of observations
    return wasserstein_distance(global_series, local_series)

def check_t_closeness(df, partition_col, sensitive_col, t_threshold):
    """
    Checks if a partitioned dataset satisfies t-Closeness.
    
    Args:
        df: The dataset containing the data.
        partition_col: Column ID representing the equivalence class (Group ID).
        sensitive_col: The column containing private info (e.g., Salary).
        t_threshold: The maximum allowed EMD distance.
    """
    global_data = df[sensitive_col].values
    partitions = df.groupby(partition_col)
    
    results = []
    is_valid = True
    
    print(f"--- Checking t-Closeness (t = {t_threshold}) ---")
    
    for group_id, group_data in partitions:
        local_data = group_data[sensitive_col].values
        
        # Calculate EMD between this group and the global population
        emd = calculate_emd_numerical(global_data, local_data)
        
        group_valid = emd <= t_threshold
        if not group_valid:
            is_valid = False
            
        results.append({
            'partition_id': group_id,
            'size': len(group_data),
            'emd': emd,
            'valid': group_valid
        })
        
        status = "PASS" if group_valid else "FAIL"
        print(f"Group {group_id} (n={len(group_data)}): EMD = {emd:.4f} -> {status}")

    return is_valid, pd.DataFrame(results)

# ==========================================
# 2. Visualization
# ==========================================

def visualize_distributions(df, partition_col, sensitive_col, specific_group_id=None):
    """
    Plots the histogram of the global distribution vs a specific partition.
    """
    global_data = df[sensitive_col]
    
    plt.figure(figsize=(10, 6))
    
    # Plot Global Distribution
    plt.hist(global_data, bins=10, density=True, alpha=0.5, label='Global Population', color='gray', edgecolor='black')
    
    # Plot Specific Partition Distribution if provided
    if specific_group_id is not None:
        group_data = df[df[partition_col] == specific_group_id][sensitive_col]
        plt.hist(group_data, bins=10, density=True, alpha=0.5, label=f'Equivalence Class {specific_group_id}', color='blue', edgecolor='black')
        
        # Calculate EMD for title
        emd = calculate_emd_numerical(global_data, group_data)
        plt.title(f"Distribution Comparison (EMD: {emd:.4f})")
    else:
        plt.title("Global Sensitive Attribute Distribution")
        
    plt.xlabel(sensitive_col)
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# ==========================================
# 3. Main Execution Block
# ==========================================

if __name__ == "__main__":
    # --- A. generate Mock Data ---
    # Imagine a dataset of 20 employees with mapped "Salary" (Sensitive)
    # We partition them into 4 groups (Equivalence Classes)
    
    data = {
        'ID': range(1, 21),
        'Partition_ID': [1]*5 + [2]*5 + [3]*5 + [4]*5, # 4 groups of 5 people
        'Salary': [
            # Group 1: Balanced distribution (matches global structure)
            30, 50, 70, 90, 110,
            # Group 2: Balanced distribution
            30, 50, 70, 90, 110,
            # Group 3: Balanced distribution
            30, 50, 70, 90, 110,
            # Group 4: Balanced distribution
            30, 50, 70, 90, 110
        ]
    }
    
    df = pd.DataFrame(data)
    print("\n--- Sample Data ---")
    print(df)
    # The Global population has a specific distribution of salaries.
    # t-Closeness requires every subgroup (Partition) to have a distribution 
    # "close" (within distance t) to this global shape.
    
    # --- B. Run Check ---
    # We set a strict threshold t = 0.2
    is_compliant, results_df = check_t_closeness(df, 'Partition_ID', 'Salary', t_threshold=0.2)
    
    print("\n--- Summary Results ---")
    print(results_df)
    
    if is_compliant:
        print("\n✅ Dataset satisfies t-Closeness.")
    else:
        print("\n❌ Dataset does NOT satisfy t-Closeness.")

    # --- C. Visualize ---
    # Let's visualize Group 3 (High earners) vs Global. 
    # Since Global has low and high earners, Group 3 should look very different (High EMD).
    print("\nDisplaying visualization for Group 3 vs Global...")
    visualize_distributions(df, 'Partition_ID', 'Salary', specific_group_id=3)
    
    # Let's visualize Group 2 (Mixed) vs Global.
    # This should have a lower EMD.
    # Uncomment the line below to see Group 2
    # visualize_distributions(df, 'Partition_ID', 'Salary', specific_group_id=2)