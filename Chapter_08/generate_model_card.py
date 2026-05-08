import os
try:
    from huggingface_hub import ModelCard, ModelCardData
except ImportError:
    print("Please install huggingface_hub first: pip install huggingface_hub")
    exit(1)

def generate_card():
    card_data = ModelCardData(
        language="en",
        license="mit",
        tags=["tabular-classification", "fairness", "what-if-tool", "synthetic-data", "random-forest", "scikit-learn"],
        model_name="Loan Approval Fairness Model",
        datasets=["synthetic"],
    )
    
    # Define the content manually to append after YAML metadata
    content = f"""
# Model Card for Loan Approval Fairness Model

## Model Details
- **Model Type:** Random Forest Classifier
- **Task:** Tabular Classification (Binary)
- **Framework:** scikit-learn
- **Use Case:** Predicting loan approval (Approved or Denied) based on applicant demographics and financial history. This model is specifically used to demonstrate fairness analysis with the Google What-If Tool.

## Intended Use
- **Primary Use:** Educational demonstration of AI bias audits and the Google What-If Tool.
- **Out-of-Scope:** Not intended for real-world loan approval or financial decision making.

## Training Data
- **Dataset:** Synthetic dataset containing 1,000 samples.
- **Features:** `Age`, `Income`, `Credit_Score`, and `Demographic_Group`.
- **Target:** `Loan_Approved` (0 = Denied, 1 = Approved).
- **Protected Attribute:** `Demographic_Group` (Group A vs Group B). The dataset is explicitly designed to simulate historical bias against Group B.

## Evaluation Data
- **Split:** 20% test split from the synthetic dataset.

## Metrics
- **Accuracy:** ~73% on the hold-out test set.
- **Fairness Metrics:** Evaluated interactively using the Google What-If Tool to observe disparities between Demographic Groups.

## Ethical Considerations
This model demonstrates how historical biases in training data (e.g., artificially lowered income and credit scores for Group B) translate into disparate impact in model predictions. It serves as a teaching tool for understanding algorithmic fairness and model governance.

## Caveats and Recommendations
The synthetic data is highly simplified. Real-world financial models must navigate complex regulatory environments and multi-dimensional fairness constraints not captured here.
"""
    
    # Generate the yaml frontmatter from card data
    yaml_header = card_data.to_yaml()
    full_markdown = yaml_header + "\n" + content
    
    # Save the card as README.md in the current directory (Chapter_08)
    output_path = os.path.join(os.path.dirname(__file__), 'README.md')
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(full_markdown)
        
    print(f"Model card successfully generated at {output_path}")

if __name__ == "__main__":
    generate_card()
