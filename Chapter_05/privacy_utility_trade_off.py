import numpy as np
import matplotlib.pyplot as plt
from diffprivlib.tools import mean as dp_mean

def plot_accuracy_tradeoff():
    # 1. Setup Data
    np.random.seed(42)
    # 1,000 salaries between 30k and 150k
    dataset = np.random.randint(30000, 150000, size=1000)
    true_mean = np.mean(dataset)
    bounds = (30000, 150000)
    
    # 2. Define a range of epsilon values (Logarithmic scale from 0.001 to 10)
    # We use log space because epsilon effects scale exponentially
    epsilons = np.logspace(-3, 1, 50) 
    average_errors = []

    print("Running queries across different epsilon values...")

    # 3. Test each epsilon
    for eps in epsilons:
        trial_errors = []
        
        # Run 50 trials per epsilon to find the *expected* error 
        # (Since DP relies on random noise, a single query might be luckily accurate)
        for _ in range(50):
            noisy_mean = dp_mean(dataset, epsilon=eps, bounds=bounds)
            
            # Calculate absolute error
            error = abs(noisy_mean - true_mean)
            trial_errors.append(error)
            
        # Store the average error for this specific epsilon
        average_errors.append(np.mean(trial_errors))

    # 4. Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(epsilons, average_errors, marker='o', linestyle='-', markersize=4, color='b')
    
    # Using log scales makes the exponential relationship easier to read
    plt.xscale('log')
    plt.yscale('log')
    
    plt.xlabel('Epsilon (ε) - Log Scale\n<-- Higher Privacy (More Noise) | Lower Privacy (Less Noise) -->')
    plt.ylabel('Mean Absolute Error - Log Scale\n<-- Higher Accuracy | Lower Accuracy -->')
    plt.title('Privacy-Utility Trade-off: Epsilon vs Error (Mean Query)')
    
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig('epsilon_tradeoff.png')
    print("Plot saved as 'epsilon_tradeoff.png'")

if __name__ == "__main__":
    plot_accuracy_tradeoff()