import numpy as np
import diffprivlib.tools as dp_tools
import warnings

# Suppress warnings for cleaner output (optional)
warnings.filterwarnings('ignore')

# 1. Generate Sample Data
# Creating a dummy dataset of 1,000 ages ranging from 18 to 80
np.random.seed(42) # For reproducible exact data
ages = np.random.randint(18, 80, size=1000)

print("--- Sample Data ---")
print(f"Number of Ages: {len(ages)}")
print(f"First 10 Ages: {ages[:10]}\n")

# 2. Compute Standard (Non-Private) Statistics
exact_mean = np.mean(ages)
exact_std = np.std(ages)
exact_hist, exact_bins = np.histogram(ages, bins=5, range=(18, 80))

print("--- Exact Statistics (No Privacy) ---")
print(f"Mean: {exact_mean:.2f}")
print(f"Standard Deviation: {exact_std:.2f}")
print(f"Histogram Counts: {exact_hist}\n")

# 3. Compute Differentially Private Statistics
# Set the privacy budget (epsilon)
epsilon_value = 0.5 

# Set the bounds. This is CRITICAL for true differential privacy!
# It defines the maximum possible range of your data without looking at the data itself.
data_bounds = (18, 80)

# Note: diffprivlib adds randomness, so running these multiple times 
# will yield slightly different results.
dp_mean = dp_tools.mean(ages, epsilon=epsilon_value, bounds=data_bounds)
dp_std = dp_tools.std(ages, epsilon=epsilon_value, bounds=data_bounds)

# For histograms, the 'range' argument inherently acts as the bounds
dp_hist, dp_bins = dp_tools.histogram(ages, epsilon=epsilon_value, bins=5, range=(18, 80))

print(f"--- Differentially Private Statistics (\u03B5={epsilon_value}) ---")
print(f"DP Mean: {dp_mean:.2f}")
print(f"DP Standard Deviation: {dp_std:.2f}")
print(f"DP Histogram Counts: {dp_hist}\n")

# 4. Describe and Compare Results
print("--- Privacy Analysis & Description ---")
mean_error = abs(exact_mean - dp_mean)
std_error = abs(exact_std - dp_std)
hist_diff = np.sum(np.abs(exact_hist - dp_hist))

print(f"Analysis: With epsilon (\u03B5) = {epsilon_value}, the DP Mean differs by {mean_error:.4f} from the exact value.")
print(f"          The DP Standard Deviation differs by {std_error:.4f}.")
print(f"          Total variation in histogram counts: {hist_diff}")

print("\nInterpretation:")
print(f"- A lower epsilon (\u03B5) would provide stronger privacy but higher noise (less accuracy).")
print(f"- The 'bounds' {data_bounds} ensured that no individual age could be inferred beyond this range.")
print("- The noise added follows the Laplace/Gaussian mechanisms provided by 'diffprivlib'.")
