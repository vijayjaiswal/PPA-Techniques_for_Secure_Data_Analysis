import numpy as np
from diffprivlib import BudgetAccountant
from diffprivlib.tools import sum as dp_sum, count_nonzero as dp_count, mean as dp_mean
from diffprivlib.utils import BudgetError

def run_dp_analysis():
    # 1. Setup Data
    # ---------------------------------------------------------
    # Dataset: Salaries of 100 employees (randomly generated)
    # Range: 30k to 150k
    dataset = np.random.randint(30000, 150000, size=100)
    print(f"Dataset generated with {len(dataset)} records.")
    print(f"True Mean Salary: {np.mean(dataset):.2f}\n")

    # 2. Initialize the Privacy Accountant
    # ---------------------------------------------------------
    # We set a total epsilon budget of 2.0.
    # 'delta=0' implies Pure Differential Privacy.
    acc = BudgetAccountant(epsilon=2.0, delta=0)
    
    # Set this accountant as the default for all diffprivlib tools
    acc.set_default() 

    print("--- Starting Privacy Accounting ---")
    print(f"Total Budget: {acc.total()[0]} epsilon")
    print(f"Remaining:    {acc.remaining()[0]} epsilon\n")

    # 3. Single Query Execution
    # ---------------------------------------------------------
    print("--- Executing Single Queries ---")
    
    # Query A: Sum of Salaries
    # Note: 'bounds' is REQUIRED for sensitivity calculation in diffprivlib.
    # We estimate the salary range is between 0 and 200,000.
    try:
        # We spend 0.5 epsilon here
        noisy_sum = dp_sum(dataset, epsilon=0.5, bounds=(0, 200000))
        print(f"Query 1 (Sum): {noisy_sum:.2f} | Spent: 0.5")
    except BudgetError:
        print("Query 1 Failed: Insufficient Budget")

    # Query B: Mean Salary
    try:
        # We spend 0.5 epsilon here
        noisy_mean = dp_mean(dataset, epsilon=0.5, bounds=(0, 200000))
        print(f"Query 2 (Mean): {noisy_mean:.2f}   | Spent: 0.5")
    except BudgetError:
        print("Query 2 Failed: Insufficient Budget")

    print(f"\nRemaining Budget: {acc.remaining()[0]:.2f}\n")

    # 4. Multiple Query Execution Loop (Stress Test)
    # ---------------------------------------------------------
    print("--- Executing Multiple Low-Budget Queries ---")
    
    # We will try to run a count query repeatedly until budget runs out.
    # Each query costs 0.1 epsilon.
    query_cost = 0.1
    query_count = 1

    while True:
        try:
            # Running a Count Query
            # diffprivlib checks the global accountant before running this
            res = dp_count(dataset, epsilon=query_cost)
            
            # If successful, print status every 3 queries to avoid clutter
            if query_count % 3 == 0:
                print(f"  Query {query_count} successful. (Result: {res})")
            
            query_count += 1
            
        except BudgetError:
            # This block executes when the accountant denies the request
            print(f"\n[STOP] Budget Depleted at Query #{query_count}!")
            print("The accountant blocked this query to preserve privacy guarantees.")
            break

    # 5. Final Report
    # ---------------------------------------------------------
    print("\n--- Final Accounting Status ---")
    print(f"Total Budget    : {acc.total()[0]}")
    print(f"Remaining Budget: {acc.remaining()[0]:.4f}")
    # Verify the number of queries tracked
    print(f"Total Queries Tracked: {len(acc)}")

if __name__ == "__main__":
    run_dp_analysis()