"""
=============================================================================
  Adversarial Training Defense — Educational Reference Implementation
=============================================================================
  Topic  : Defending Models and Building Robustness
  Method : Adversarial Training (AT)
  Goal   : Train a neural network on both clean AND adversarially perturbed
           inputs so the model learns robust decision boundaries that resist
           gradient-based evasion attacks (FGSM, PGD).

  How Adversarial Training Defends:
  ─────────────────────────────────
  Standard training minimizes loss on clean data only. This leaves the model
  vulnerable to small, carefully crafted input perturbations that cross
  decision boundaries. Adversarial Training reformulates the objective as
  a min-max optimization:

      min_θ  E[ max_{||δ||≤ε}  L(f_θ(x + δ), y) ]

  The INNER loop (max) finds the worst-case perturbation δ for each sample.
  The OUTER loop (min) updates model weights to be correct even under that
  worst-case perturbation. This forces the model to:
    1. Learn smoother, wider decision margins.
    2. Ignore adversarial noise within the ε-ball.
    3. Generalize to unseen perturbations at inference time.
=============================================================================
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# ─── Reproducibility ─────────────────────────────────────────────────────────
torch.manual_seed(42)
np.random.seed(42)

# ══════════════════════════════════════════════════════════════════════════════
# 1. DATA PREPARATION
# ══════════════════════════════════════════════════════════════════════════════

def prepare_data(n_samples=1500):
    """Generate a 2-class, 2D dataset (easy to visualize decision boundaries)."""
    X, y = make_moons(n_samples=n_samples, noise=0.2, random_state=42)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    def to_loader(X, y, batch=64):
        return DataLoader(
            TensorDataset(
                torch.tensor(X, dtype=torch.float32),
                torch.tensor(y, dtype=torch.long),
            ),
            batch_size=batch, shuffle=True,
        )

    return to_loader(X_train, y_train), to_loader(X_test, y_test), scaler


# ══════════════════════════════════════════════════════════════════════════════
# 2. MODEL DEFINITION
# ══════════════════════════════════════════════════════════════════════════════

class SimpleClassifier(nn.Module):
    """Small MLP — intentionally simple to make the defense effect obvious."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 2),
        )

    def forward(self, x):
        return self.net(x)


# ══════════════════════════════════════════════════════════════════════════════
# 3. ATTACK IMPLEMENTATIONS  (used both to ATTACK and to TRAIN-with)
# ══════════════════════════════════════════════════════════════════════════════

def fgsm_attack(model, x, y, epsilon=0.3):
    """
    Fast Gradient Sign Method (single-step).
    Perturbs input in the direction that maximizes the loss.
    """
    x_adv = x.clone().detach().requires_grad_(True)
    loss = nn.CrossEntropyLoss()(model(x_adv), y)
    loss.backward()
    perturbation = epsilon * x_adv.grad.sign()
    return (x_adv + perturbation).detach()


def pgd_attack(model, x, y, epsilon=0.3, alpha=0.05, steps=10):
    """
    Projected Gradient Descent (multi-step — stronger attack).
    Iteratively refines the perturbation, projecting back into ε-ball.
    """
    x_adv = x.clone().detach()
    for _ in range(steps):
        x_adv.requires_grad_(True)
        loss = nn.CrossEntropyLoss()(model(x_adv), y)
        loss.backward()
        x_adv = (x_adv + alpha * x_adv.grad.sign()).detach()
        # Project back into the ε-ball around the original input
        x_adv = torch.clamp(x_adv, x - epsilon, x + epsilon)
    return x_adv


# ══════════════════════════════════════════════════════════════════════════════
# 4. TRAINING ROUTINES
# ══════════════════════════════════════════════════════════════════════════════

def train_standard(model, loader, epochs=30, lr=0.01):
    """Standard (vanilla) training — no adversarial examples."""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        total_loss = 0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f"  [Standard] Epoch {epoch+1:>2}/{epochs}  Loss: {total_loss/len(loader):.4f}")


def train_adversarial(model, loader, epochs=30, lr=0.01, epsilon=0.3):
    """
    Adversarial Training — the DEFENSE.
    Each mini-batch goes through an inner maximization (PGD) before
    the model weights are updated on the worst-case examples.
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        total_loss = 0
        for xb, yb in loader:
            # ── Inner Maximization: find worst-case perturbation ──
            x_adv = pgd_attack(model, xb, yb, epsilon=epsilon, alpha=0.05, steps=7)

            # ── Outer Minimization: train on BOTH clean + adversarial ──
            optimizer.zero_grad()
            loss_clean = criterion(model(xb), yb)
            loss_adv   = criterion(model(x_adv), yb)
            loss = 0.5 * loss_clean + 0.5 * loss_adv   # balanced objective
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f"  [Adversarial] Epoch {epoch+1:>2}/{epochs}  Loss: {total_loss/len(loader):.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# 5. EVALUATION & VISUALIZATION
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate(model, loader, attack_fn=None, epsilon=0.3):
    """Returns accuracy — optionally under a specified attack."""
    correct, total = 0, 0
    for xb, yb in loader:
        if attack_fn is not None:
            model.eval()
            # Re-enable grads only for generating attack
            with torch.enable_grad():
                xb = attack_fn(model, xb, yb, epsilon=epsilon)
        preds = model(xb).argmax(dim=1)
        correct += (preds == yb).sum().item()
        total += yb.size(0)
    return correct / total


def plot_comparison(results, save_path="adversarial_training_defense.png"):
    """Bar chart comparing standard vs. adversarially trained model."""
    scenarios = list(results.keys())
    std_accs  = [results[s]["Standard"]     * 100 for s in scenarios]
    adv_accs  = [results[s]["Adversarial"]  * 100 for s in scenarios]

    x = np.arange(len(scenarios))
    width = 0.32

    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - width/2, std_accs, width, label="Standard Model",
                   color="#e74c3c", edgecolor="black", linewidth=0.6)
    bars2 = ax.bar(x + width/2, adv_accs, width, label="Adversarially Trained",
                   color="#27ae60", edgecolor="black", linewidth=0.6)

    # Annotate bars
    for bar in bars1 + bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                f"{bar.get_height():.1f}%", ha='center', va='bottom', fontsize=9)

    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Defense Effectiveness: Standard vs. Adversarial Training", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, fontsize=11)
    ax.set_ylim(0, 110)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"\n[OK] Comparison chart saved: {save_path}")


# ==============================================================================
# 6. MAIN -- RUN THE FULL EXPERIMENT
# ==============================================================================

def main():
    EPSILON = 0.3   # perturbation budget (Linf ball radius)

    print("=" * 65)
    print("  ADVERSARIAL TRAINING DEFENSE -- Educational Reference")
    print("=" * 65)

    # -- Data ------------------------------------------------------------------
    train_loader, test_loader, _ = prepare_data()

    # -- Train: Standard Model -------------------------------------------------
    print("\n>> Phase 1: Training Standard (Undefended) Model")
    std_model = SimpleClassifier()
    train_standard(std_model, train_loader, epochs=30)

    # -- Train: Adversarially Trained Model ------------------------------------
    print("\n>> Phase 2: Training Adversarially Hardened Model")
    adv_model = SimpleClassifier()
    train_adversarial(adv_model, train_loader, epochs=30, epsilon=EPSILON)

    # -- Evaluate both models under multiple conditions ------------------------
    print("\n>> Phase 3: Evaluating Robustness")
    print("-" * 55)

    results = {}

    # Scenario A -- Clean (no attack)
    std_clean = evaluate(std_model, test_loader)
    adv_clean = evaluate(adv_model, test_loader)
    results["Clean Data (No Attack)"] = {"Standard": std_clean, "Adversarial": adv_clean}
    print(f"  Clean Data       | Standard: {std_clean:.2%}  | Robust: {adv_clean:.2%}")

    # Scenario B -- FGSM attack
    std_fgsm = evaluate(std_model, test_loader, attack_fn=fgsm_attack, epsilon=EPSILON)
    adv_fgsm = evaluate(adv_model, test_loader, attack_fn=fgsm_attack, epsilon=EPSILON)
    results["Under FGSM Attack"] = {"Standard": std_fgsm, "Adversarial": adv_fgsm}
    print(f"  FGSM Attack      | Standard: {std_fgsm:.2%}  | Robust: {adv_fgsm:.2%}")

    # Scenario C -- PGD attack (stronger)
    std_pgd = evaluate(std_model, test_loader, attack_fn=pgd_attack, epsilon=EPSILON)
    adv_pgd = evaluate(adv_model, test_loader, attack_fn=pgd_attack, epsilon=EPSILON)
    results["Under PGD Attack"] = {"Standard": std_pgd, "Adversarial": adv_pgd}
    print(f"  PGD Attack       | Standard: {std_pgd:.2%}  | Robust: {adv_pgd:.2%}")

    print("-" * 55)

    # -- Visualize -------------------------------------------------------------
    plot_comparison(results)

    # -- Summary ---------------------------------------------------------------
    print("\n" + "=" * 63)
    print("       HOW ADVERSARIAL TRAINING DEFENDS")
    print("=" * 63)
    print("")
    print("  1. WIDER DECISION MARGINS")
    print("     By training on worst-case perturbations, the model")
    print("     pushes decision boundaries AWAY from data points,")
    print("     making small perturbations insufficient to flip class.")
    print("")
    print("  2. SMOOTHER LOSS LANDSCAPE")
    print("     Standard models have sharp, jagged loss surfaces.")
    print("     AT smooths these out, removing the 'footholds' that")
    print("     gradient-based attacks exploit.")
    print("")
    print("  3. REDUCED GRADIENT SENSITIVITY")
    print("     AT makes gradients less informative for attackers,")
    print("     which also partially defends against Model Inversion")
    print("     attacks that rely on gradient signals.")
    print("")
    print("  4. CERTIFIED ROBUSTNESS (within epsilon-ball)")
    print("     For any perturbation ||delta|| <= epsilon, the adversarially")
    print("     trained model is optimized to maintain its prediction,")
    print("     providing a practical robustness guarantee.")
    print("")
    print("-" * 63)
    print("  KEY METRICS:")
    fgsm_gain = (adv_fgsm - std_fgsm) * 100
    pgd_gain  = (adv_pgd  - std_pgd)  * 100
    print(f"  FGSM robustness gain: +{fgsm_gain:.1f}%  (AT vs Standard)")
    print(f"  PGD  robustness gain: +{pgd_gain:.1f}%  (AT vs Standard)")
    print("=" * 63)


if __name__ == "__main__":
    main()
