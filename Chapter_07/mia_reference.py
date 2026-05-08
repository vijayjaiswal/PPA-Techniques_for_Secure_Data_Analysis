import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# Set seeds for reproducibility
np.random.seed(42)

def generate_healthcare_data(n_samples=2000):
    """Generates synthetic healthcare records for binary classification."""
    age = np.random.randint(20, 80, n_samples)
    bmi = np.random.uniform(18, 45, n_samples)
    blood_pressure = np.random.randint(80, 180, n_samples)
    glucose = np.random.randint(70, 200, n_samples)
    
    # Logic for ground truth (probability of readmission)
    logits = (age * 0.05 + bmi * 0.1 + blood_pressure * 0.02 + glucose * 0.03 - 10)
    prob = 1 / (1 + np.exp(-logits))
    labels = (prob > 0.5).astype(int)
    
    df = pd.DataFrame({
        'age': age,
        'bmi': bmi,
        'blood_pressure': blood_pressure,
        'glucose': glucose,
        'label': labels
    })
    return df

def train_target_model(train_df, test_df):
    """Trains the Target Model that we want to attack."""
    print("\n[Target] Training Target Model (Random Forest)...")
    X_train = train_df.drop('label', axis=1)
    y_train = train_df['label']
    
    # We use a slightly deeper forest to allow some overfitting (demonstrating the vulnerability)
    model = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42)
    model.fit(X_train, y_train)
    
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(test_df['label'], model.predict(test_df.drop('label', axis=1)))
    
    print(f"Target Model - Train Accuracy: {train_acc:.2%}, Test Accuracy: {test_acc:.2%}")
    return model

def shadow_model_attack(target_model, target_train, target_test, auxiliary_data):
    """
    Performs the Membership Inference Attack using the Shadow Model technique.
    
    1. Shadow Model: Trained on auxiliary data to mimic target behavior.
    2. Attack Dataset: Created from Shadow Model's predictions on its own training/test sets.
    3. Attack Model: Trained to distinguish 'Member' behavior from 'Non-Member' behavior.
    """
    print("\n[Shadow] Training Shadow Model on auxiliary data...")
    # Split auxiliary data into shadow_train (members of shadow) and shadow_test (non-members of shadow)
    sh_train, sh_test = train_test_split(auxiliary_data, test_size=0.5, random_state=42)
    
    # 1. Train Shadow Model (using same architecture as target if possible, or similar)
    shadow_clf = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=1)
    shadow_clf.fit(sh_train.drop('label', axis=1), sh_train['label'])
    
    # 2. Build Attack Dataset
    # We collect the prediction probabilities (confidence) for shadow members and non-members
    print("[Attack] Building Attack Dataset from Shadow Model signals...")
    
    def get_attack_features(model, df):
        # We use the sorted probabilities as features for the attack model
        probs = model.predict_proba(df.drop('label', axis=1))
        # Sort probabilities to make the attack invariant to class indices
        return np.sort(probs, axis=1)

    X_attack_members = get_attack_features(shadow_clf, sh_train)
    X_attack_nonmembers = get_attack_features(shadow_clf, sh_test)
    
    X_attack = np.vstack([X_attack_members, X_attack_nonmembers])
    y_attack = np.concatenate([np.ones(len(X_attack_members)), np.zeros(len(X_attack_nonmembers))])
    
    # 3. Train Attack Model
    # This model learns: "What does a model's output look like when it's seeing data it was trained on?"
    attack_model = MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=500, random_state=42)
    attack_model.fit(X_attack, y_attack)
    
    # 4. Inferencing on Target Model
    print("\n[Inferencing] Attacking Target Model to identify members...")
    
    # Get features for real members and non-members of the Target Model
    target_member_features = get_attack_features(target_model, target_train)
    target_nonmember_features = get_attack_features(target_model, target_test)
    
    test_X = np.vstack([target_member_features, target_nonmember_features])
    test_y = np.concatenate([np.ones(len(target_member_features)), np.zeros(len(target_nonmember_features))])
    
    predictions = attack_model.predict(test_X)
    
    print("\nMIA Results on Target Model:")
    print(classification_report(test_y, predictions, target_names=['Non-Member', 'Member']))
    
    return attack_model

def main():
    print("--- Membership Inference Attack (MIA) Educational Reference ---")
    
    # 1. Prepare Data
    data = generate_healthcare_data(3000)
    
    # Data Split:
    # Set 1: For the Target Model (A)
    # Set 2: Auxiliary Data for the Attacker (B)
    target_data, auxiliary_data = train_test_split(data, test_size=0.5, random_state=42)
    
    # Within Target Data, split into train (Members) and test (Non-Members)
    target_train, target_test = train_test_split(target_data, test_size=0.5, random_state=42)
    
    # 2. Train Target
    target_model = train_target_model(target_train, target_test)
    
    # 3. Perform MIA
    # In a real attack, the attacker doesn't have target_train. 
    # They only have auxiliary_data and access to the target_model's API.
    attack_model = shadow_model_attack(target_model, target_train, target_test, auxiliary_data)
    
    print("\nConclusion:")
    print("If the 'Member' recall is high, it means the model has 'memorized' the training data")
    print("sufficiently for the attacker to distinguish it from unseen data.")
    print("This violates the privacy of the individuals in the training set.")

if __name__ == "__main__":
    main()
