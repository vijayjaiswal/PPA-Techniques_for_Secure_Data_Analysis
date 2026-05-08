"""
=============================================================================
  Privacy-Aware Model Publishing — Educational Reference
=============================================================================
  Topic  : Sanitizing ML Models Before Deployment or Sharing
  Goal   : Demonstrate techniques to harden a model artifact against 
           privacy leakage (MIA), extraction, and over-memorization.
  
  Scenario:
  ---------
  A healthcare provider wants to share a pre-trained "Patient Risk" model 
  with a research partner for fine-tuning. To prevent the partner (or an 
  adversary) from reconstructing patient data from the model weights or 
  confident scores, the provider sanitizes the model before "publishing" it.

  Techniques Demonstrated:
  ------------------------
  1. Gradient Clipping (Training-time): Limits the influence of individual 
     training samples on the final parameters.
  2. Weight Perturbation (Post-training): Adds controlled noise to weights 
     to "blur" memorized signals.
  3. Weight Quantization/Rounding: Reduces weight precision to hide 
     fine-grained decision boundary details.
  4. Knowledge Distillation: Training a smaller, "sanitized" student 
     model that inherits logic but forgets specific training data instances.
=============================================================================
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy

# ─── Reproducibility ─────────────────────────────────────────────────────────
torch.manual_seed(42)
np.random.seed(42)

# ═══════════════════════════════════════════════════════════════════════════════
# 1. DATA PREPARATION (Synthetic Healthcare Data)
# ═══════════════════════════════════════════════════════════════════════════════

def generate_data(n_samples=2500):
    """Generate synthetic patient data for readmission risk."""
    X = np.random.randn(n_samples, 8).astype(np.float32)
    # Target depends on features + some non-linear interaction
    logits = X[:, 0] * 0.5 + X[:, 1] * -0.3 + (X[:, 2]**2) * 0.1 - 0.2
    y = (logits > 0).astype(np.int64)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return (torch.tensor(X_train), torch.tensor(y_train), 
            torch.tensor(X_test), torch.tensor(y_test))

# ═══════════════════════════════════════════════════════════════════════════════
# 2. MODEL DEFINITION
# ═══════════════════════════════════════════════════════════════════════════════

class RiskModel(nn.Module):
    def __init__(self, input_dim=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32), nn.ReLU(),
            nn.Linear(32, 16),        nn.ReLU(),
            nn.Linear(16, 2)
        )
    def forward(self, x):
        return self.net(x)

# ═══════════════════════════════════════════════════════════════════════════════
# 3. TECHNIQUE: GRADIENT CLIPPING (Training-Time Defense)
# ═══════════════════════════════════════════════════════════════════════════════

def train_with_clipping(model, X, y, max_grad_norm=1.0, epochs=50):
    """
    Standard training augmented with gradient clipping.
    Clipping prevents any single sample from causing large jumps in 
    weight updates, which helps resist Membership Inference Attacks.
    """
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        
        # ─── THE SANITIZATION STEP ───
        # Ensure gradients don't exceed a specific magnitude
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        
        optimizer.step()
    return model

# ═══════════════════════════════════════════════════════════════════════════════
# 4. TECHNIQUE: POST-TRAINING SANITIZATION (Weight Perturbation)
# ═══════════════════════════════════════════════════════════════════════════════

def sanitize_weights(model, noise_scale=0.01, rounding_decimals=3):
    """
    Sanitize a trained model artifact by:
    1. Injecting Gaussian noise into weights (Differential Privacy heuristic).
    2. Rounding weight values to reduce precision (hiding over-fitted details).
    """
    sanitized_model = copy.deepcopy(model)
    with torch.no_grad():
        for param in sanitized_model.parameters():
            # 1. Add Perturbation (Noise)
            noise = torch.randn_like(param) * noise_scale
            param.add_(noise)
            
            # 2. Quantization / Rounding
            # Rounding to N decimals removes the 'memorization' in the LSBs.
            param.copy_(torch.round(param * (10**rounding_decimals)) / (10**rounding_decimals))
            
    return sanitized_model

# ═══════════════════════════════════════════════════════════════════════════════
# 5. TECHNIQUE: SANITIZED DISTILLATION
# ═══════════════════════════════════════════════════════════════════════════════

def train_sanitized_student(teacher_model, X_train, y_train, epochs=50):
    """
    Knowledge Distillation as Sanitization.
    The Student model learns from the Teacher's *soft* predictions 
    (probabilities) rather than the raw data. This acts as a filter,
    as soft labels represent the general logic while discarding 
    instance-specific noise/memorization.
    """
    student_model = RiskModel()
    optimizer = optim.Adam(student_model.parameters(), lr=0.01)
    
    teacher_model.eval()
    with torch.no_grad():
        soft_targets = torch.softmax(teacher_model(X_train), dim=1)

    for epoch in range(epochs):
        optimizer.zero_grad()
        student_outputs = student_model(X_train)
        # Match student logs to teacher's soft probabilities
        loss = nn.KLDivLoss(reduction='batchmean')(
            torch.log_softmax(student_outputs, dim=1), soft_targets
        )
        loss.backward()
        optimizer.step()
        
    return student_model

# ═══════════════════════════════════════════════════════════════════════════════
# 6. PRIVACY AUDIT: CONFIDENCE-BASED MIA
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_privacy(model, X_train, y_train, X_test, y_test):
    """
    Evaluate Membership Inference vulnerability.
    We check the 'Confidence Gap' — how much more confident the model 
    is on training data vs unseen data. Higher gap = Lower privacy.
    """
    model.eval()
    with torch.no_grad():
        tr_probs = torch.softmax(model(X_train), dim=1)
        te_probs = torch.softmax(model(X_test), dim=1)
        
        # Get confidence for correct classes
        tr_conf = tr_probs.gather(1, y_train.view(-1, 1)).mean().item()
        te_conf = te_probs.gather(1, y_test.view(-1, 1)).mean().item()
        
        gap = tr_conf - te_conf
        acc = accuracy_score(y_test, model(X_test).argmax(dim=1))
        
    return acc, gap

# ═══════════════════════════════════════════════════════════════════════════════
# 7. MAIN EXPERIMENT
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("="*65)
    print("  MODEL PUBLISHING SANITIZATION — Privacy hardening")
    print("="*65)

    # -- 1. Data --
    X_tr, y_tr, X_te, y_te = generate_data()

    # -- 2. Baseline Model (Vulnerable) --
    print("\n[Phase 1] Training Baseline Model (No Sanitization)...")
    base_model = RiskModel()
    # standard training (no clipping)
    optimizer = optim.Adam(base_model.parameters(), lr=0.01)
    for _ in range(50):
        optimizer.zero_grad(); nn.CrossEntropyLoss()(base_model(X_tr), y_tr).backward(); optimizer.step()
    
    b_acc, b_gap = evaluate_privacy(base_model, X_tr, y_tr, X_te, y_te)
    print(f"   Baseline -> Acc: {b_acc:.2%}, Confidence Gap (MIA Risk): {b_gap:.4f}")

    # -- 3. Hardened: Gradient Clipping --
    print("\n[Phase 2] Training with Gradient Clipping (C=1.0)...")
    clip_model = RiskModel()
    train_with_clipping(clip_model, X_tr, y_tr, max_grad_norm=1.0)
    c_acc, c_gap = evaluate_privacy(clip_model, X_tr, y_tr, X_te, y_te)
    print(f"   Clipped  -> Acc: {c_acc:.2%}, Confidence Gap (MIA Risk): {c_gap:.4f}")

    # -- 4. Sanitized: Weight Perturbation --
    print("\n[Phase 3] Post-training Weight Sanitization (Noise+Rounding)...")
    san_model = sanitize_weights(base_model, noise_scale=0.05, rounding_decimals=2)
    s_acc, s_gap = evaluate_privacy(san_model, X_tr, y_tr, X_te, y_te)
    print(f"   Sanitized -> Acc: {s_acc:.2%}, Confidence Gap (MIA Risk): {s_gap:.4f}")

    # -- 5. Sanitized: Distillation --
    print("\n[Phase 4] Publishing via Sanitized Distillation...")
    dist_model = train_sanitized_student(base_model, X_tr, y_tr)
    d_acc, d_gap = evaluate_privacy(dist_model, X_tr, y_tr, X_te, y_te)
    print(f"   Distilled -> Acc: {d_acc:.2%}, Confidence Gap (MIA Risk): {d_gap:.4f}")

    # -- Visualization --
    labels = ['Baseline', 'Grad Clipping', 'Weight Perturb', 'Distillation']
    accs = [b_acc, c_acc, s_acc, d_acc]
    gaps = [b_gap, c_gap, s_gap, d_gap]

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()
    
    ax1.bar(labels, accs, alpha=0.3, color='blue', label='Utility (Accuracy)')
    ax2.plot(labels, gaps, color='red', marker='o', linewidth=2, label='Privacy Risk (MIA Gap)')
    
    ax1.set_ylabel('Accuracy', color='blue', fontsize=12)
    ax2.set_ylabel('Confidence Gap (MIA Risk)', color='red', fontsize=12)
    plt.title('Sanitization Trade-off: Utility vs. Privacy', fontsize=14)
    
    plt.tight_layout()
    plt.savefig('model_sanitization_results.png', dpi=150)
    print("\n[OK] Results plot saved to 'model_sanitization_results.png'")

    # -- Summary --
    print("\n" + "="*63)
    print("      HOW SANITIZATION PROTECTS PUBLISHED MODELS")
    print("="*63)
    print("""
  1. GRADIENT CLIPPING
     Ensures no single sample outlier can 'tilt' the model too far.
     This smoothes the decision boundary and reduces the gap between
     training and test confidence, making MIA harder.

  2. WEIGHT PERTURBATION (Noise Injection)
     Treats model weights like sensitive data. Adding noise acts as
     a heuristic version of Differential Privacy, 'washing out'
     highly specific feature patterns that might identify a patient.

  3. WEIGHT QUANTIZATION (Rounding)
     High-precision floating point numbers can store 'unintended
     memorization'. Rounding limits the bit-budget, forcing the
     model to keep only the most generalizable signals.

  4. KNOWLEDGE DISTILLATION
     The 'sanitization layer'. By training a student on teacher
     probabilities, we extract the *logic* without inheriting the
     raw data memorization.
    """)
    print("="*63)

if __name__ == "__main__":
    main()
