import os
import warnings

# Suppress TF oneDNN warnings and resolve Protobuf TypeError
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
warnings.filterwarnings('ignore', category=FutureWarning)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# ---------------------------------------------------------------------------
# Note: The Google What-If Tool (WIT) is an interactive visual interface.
# It is designed to be executed within a Jupyter Notebook or Google Colab.
# This script provides the complete reference implementation.
# ---------------------------------------------------------------------------

try:
    from witwidget.notebook.visualization import WitWidget, WitConfigBuilder
except ImportError:
    print("Warning: witwidget is not installed. Please install using: pip install witwidget")

def create_synthetic_data(n_samples=1000):
    """
    Generates a synthetic dataset for loan approvals with a protected attribute.
    This simulates a scenario where bias might exist, ideal for stress-testing.
    """
    np.random.seed(42)
    
    # Basic features
    age = np.random.randint(18, 70, n_samples)
    income = np.random.normal(60000, 20000, n_samples)
    credit_score = np.random.normal(650, 80, n_samples)
    
    # Protected attribute (Demographic: Group A vs Group B)
    # Simulating a historical disadvantage for Group B
    group = np.random.choice(['Group A', 'Group B'], size=n_samples, p=[0.7, 0.3])
    
    income = np.where(group == 'Group B', income - 5000, income)
    credit_score = np.where(group == 'Group B', credit_score - 20, credit_score)
    
    # Target variable: Loan Approved (1) or Denied (0)
    # The true probability depends heavily on income and credit score
    prob = (income / 100000) * 0.4 + (credit_score / 800) * 0.6
    prob += np.random.normal(0, 0.1, n_samples) # Add some random noise
    
    approved = (prob > 0.65).astype(int)
    
    df = pd.DataFrame({
        'Age': age,
        'Income': income,
        'Credit_Score': credit_score,
        'Demographic_Group': group,
        'Loan_Approved': approved
    })
    
    return df

def train_model(df):
    """Trains a Random Forest classifier to predict loan approvals."""
    # Convert categorical variables to numerical for the model
    df_encoded = df.copy()
    df_encoded['Demographic_Group'] = df_encoded['Demographic_Group'].map({'Group A': 0, 'Group B': 1})
    
    # Features and Target
    X = df_encoded.drop('Loan_Approved', axis=1)
    y = df_encoded['Loan_Approved']
    
    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Return the model, test features, test target, and the original unencoded test data
    return model, X_test, y_test, df.iloc[X_test.index]

def setup_what_if_tool():
    """
    Sets up the What-If Tool configuration and renders the interactive widget.
    """
    print("1. Generating synthetic dataset...")
    df = create_synthetic_data()
    
    print("2. Training Random Forest model...")
    model, X_test, y_test, df_test_original = train_model(df)
    
    # The What-If Tool expects data as a list of dictionaries
    # We use the original dataframe (with string labels) for better UI readability
    examples = df_test_original.to_dict(orient='records')
    
    # Create a custom prediction function that WIT will call interactively
    def custom_predict(examples_to_predict):
        # Convert incoming dictionaries back to a DataFrame
        df_pred = pd.DataFrame(examples_to_predict)
        
        # Apply the exact same preprocessing used during training
        df_pred['Demographic_Group'] = df_pred['Demographic_Group'].map({'Group A': 0, 'Group B': 1})
        
        features = ['Age', 'Income', 'Credit_Score', 'Demographic_Group']
        X_pred = df_pred[features]
        
        # Return probability scores [prob_denied, prob_approved]
        preds = model.predict_proba(X_pred)
        return preds.tolist()

    print("3. Configuring the What-If Tool builder...")
    
    if 'WitConfigBuilder' not in globals():
        print("\n[Error] 'WitConfigBuilder' is not defined. Ensure witwidget is installed (pip install witwidget).")
        return

    # Configure the WIT widget
    # We limit to 200 examples to ensure smooth UI responsiveness
    config_builder = (WitConfigBuilder(examples[:200])
                      .set_custom_predict_fn(custom_predict)
                      .set_target_feature('Loan_Approved')
                      .set_label_vocab(['Denied', 'Approved']))
    
    print("\n--- EXECUTION INSTRUCTIONS ---")
    print("The What-If Tool requires a notebook environment to render its HTML/JS interface.")
    print("If you run this inside Jupyter Notebook, JupyterLab, or Google Colab, the tool will appear below.")
    print("You can use it to slice data by 'Demographic_Group', adjust decision thresholds, and explore fairness metrics (like Equal Opportunity).")
    
    try:
        print("\n[Important Note for VS Code Users]")
        print("VS Code Notebooks do not natively support Google's What-If Tool widget rendering.")
        print("If you see 'Error displaying widget', please run this notebook in JupyterLab in your browser.")
        print("To do this, run `jupyter lab` in your terminal and open this notebook.")
        
        # Render the interactive widget
        widget = WitWidget(config_builder, height=800)
        print("\n[Success] WitWidget rendered successfully in the notebook environment.")
        # return widget (deferred)
    except NameError:
         print("\n[Notice] 'WitWidget' is not defined. Ensure witwidget is installed.")
    except Exception as e:
        print(f"\n[Notice] Could not render WitWidget visually. Error: {e}")
        print("Please run this code block inside a Jupyter Notebook cell.")

    # Added: CLI Visualization & Metrics Fallback
    cli_visualization(model, X_test, y_test, df_test_original)
    
    if 'widget' in locals():
        return widget
    
from fairlearn.metrics import MetricFrame, selection_rate, false_positive_rate, false_negative_rate
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

def cli_visualization(model, X_test, y_test, df_test_original):
    """Provides a CLI-friendly static evaluation for fairness metrics."""
    print("\n" + "="*50)
    print("--- CLI VISUALIZATION & FAIRNESS METRICS ---")
    print("Since the interactive What-If Tool requires a Notebook,")
    print("here is a CLI-based static analysis summary:")
    
    y_pred = model.predict(X_test)
    sensitive_features = df_test_original['Demographic_Group']
    
    metrics = {
        'Accuracy': accuracy_score,
        'Selection Rate (Approval %)': selection_rate,
        'False Positive Rate': false_positive_rate,
        'False Negative Rate': false_negative_rate
    }
    
    metric_frame = MetricFrame(
        metrics=metrics,
        y_true=y_test,
        y_pred=y_pred,
        sensitive_features=sensitive_features
    )
    
    print("\n1. Fairness Metrics Breakdown by Group:")
    print("-" * 50)
    print(metric_frame.by_group.to_string(float_format="%.3f"))
    print("-" * 50)
    
    print("\n2. Generating static visualization plot...")
    # Plotting using Pandas wrapper over Matplotlib
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    metric_frame.by_group.plot(kind='bar', subplots=True, ax=axes, legend=False, rot=0, 
                               color=['#1f77b4', '#ff7f0e'])
    plt.suptitle("Fairness Metrics by Demographic Group", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig('cli_fairness_visualization.png', bbox_inches='tight')
    print("-> Saved visualization to 'cli_fairness_visualization.png'")
    plt.close('all')
    print("="*50 + "\n")

if __name__ == "__main__":
    widget = setup_what_if_tool()
    widget