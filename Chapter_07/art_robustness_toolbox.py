"""
=============================================================================
  IBM Adversarial Robustness Toolbox (ART) — Educational Reference
=============================================================================
  Topic  : Standardised Adversarial Testing & Defense with ART
  Goal   : Demonstrate how to use ART's unified API to:
           1. Wrap a PyTorch model as an ART-compatible classifier
           2. Generate adversarial examples (FGSM, PGD, DeepFool, C&W)
           3. Evaluate model vulnerability under each attack
           4. Apply built-in defenses (Adversarial Training, preprocessors,
              postprocessors)
           5. Simulate Membership Inference Attacks (MIA) for privacy audit
           6. Compare defended vs. undefended model performance

  Why ART?
  ────────
  Rather than hand-coding each attack / defense (as in our earlier scripts),
  ART provides a *standardised, framework-agnostic* API with 100+ attacks
  and 40+ defenses.  This makes security auditing reproducible and lets
  practitioners focus on interpreting results rather than re-implementing
  algorithms.

  Dependencies:
    pip install adversarial-robustness-toolbox torch numpy matplotlib scikit-learn
=============================================================================
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# ─── Reproducibility ─────────────────────────────────────────────────────────
torch.manual_seed(42)
np.random.seed(42)

# ═══════════════════════════════════════════════════════════════════════════════
# 1. SYNTHETIC DATA PREPARATION
# ═══════════════════════════════════════════════════════════════════════════════

def prepare_data(n_samples=2000):
    """
    Generate a binary classification dataset, scale it, and split into
    train / test partitions.  Returns numpy arrays suitable for ART.
    """
    X, y = make_classification(
        n_samples=n_samples, n_features=10, n_informative=6,
        n_redundant=2, n_classes=2, random_state=42,
    )
    scaler = StandardScaler()
    X = scaler.fit_transform(X).astype(np.float32)
    y = y.astype(np.int64)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42,
    )
    return X_train, X_test, y_train, y_test


# ═══════════════════════════════════════════════════════════════════════════════
# 2. PYTORCH MODEL DEFINITION
# ═══════════════════════════════════════════════════════════════════════════════

class BinaryClassifier(nn.Module):
    """Simple MLP for binary classification — kept small for fast demos."""
    def __init__(self, input_dim=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),  nn.ReLU(),
            nn.Linear(64, 32),         nn.ReLU(),
            nn.Linear(32, 2),
        )

    def forward(self, x):
        return self.net(x)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. ART CLASSIFIER WRAPPER
# ═══════════════════════════════════════════════════════════════════════════════

def create_art_classifier(model, input_dim=10, lr=0.01):
    """
    Wrap a PyTorch model in ART's PyTorchClassifier.

    This is the key integration point: ART needs to compute gradients
    through the model to generate adversarial examples, so we provide
    the loss function and optimizer.
    """
    from art.estimators.classification import PyTorchClassifier

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    classifier = PyTorchClassifier(
        model=model,
        loss=criterion,
        optimizer=optimizer,
        input_shape=(input_dim,),
        nb_classes=2,
        clip_values=(-3.0, 3.0),   # approx range after StandardScaler
    )
    return classifier


# ═══════════════════════════════════════════════════════════════════════════════
# 4. STANDARD TRAINING (baseline — no adversarial hardening)
# ═══════════════════════════════════════════════════════════════════════════════

def train_standard(classifier, X_train, y_train, epochs=20, batch_size=64):
    """Train the ART classifier on clean data only."""
    print("  Training standard (undefended) model ...")
    classifier.fit(X_train, np.eye(2)[y_train], nb_epochs=epochs,
                   batch_size=batch_size, verbose=False)
    preds = classifier.predict(X_train).argmax(axis=1)
    acc = accuracy_score(y_train, preds)
    print(f"  Train accuracy: {acc:.2%}")
    return classifier


# ═══════════════════════════════════════════════════════════════════════════════
# 5. EVASION ATTACKS — Generate Adversarial Examples
# ═══════════════════════════════════════════════════════════════════════════════

def run_evasion_attacks(classifier, X_test, y_test, eps=0.3):
    """
    Generate adversarial examples using four ART attack modules:
      - FGSM  (Fast Gradient Sign Method — single step)
      - PGD   (Projected Gradient Descent — multi-step, stronger)
      - DeepFool (minimum perturbation to cross boundary)
      - C&W L2  (Carlini-Wagner — optimisation-based, very strong)

    Returns dict of {attack_name: (x_adv, clean_acc, adv_acc)}.
    """
    from art.attacks.evasion import (
        FastGradientMethod,
        ProjectedGradientDescent,
        DeepFool,
        CarliniL2Method,
    )

    clean_preds = classifier.predict(X_test).argmax(axis=1)
    clean_acc = accuracy_score(y_test, clean_preds)
    print(f"\n  Clean test accuracy: {clean_acc:.2%}")

    attacks = {
        "FGSM": FastGradientMethod(estimator=classifier, eps=eps),
        "PGD": ProjectedGradientDescent(
            estimator=classifier, eps=eps, eps_step=eps/5,
            max_iter=20, batch_size=64,
        ),
        "DeepFool": DeepFool(classifier=classifier, max_iter=50,
                              batch_size=64),
        "C&W L2": CarliniL2Method(classifier=classifier,
                                   max_iter=30, batch_size=64),
    }

    results = {}
    for name, attack in attacks.items():
        print(f"  Generating {name} adversarial examples ...", end=" ")
        x_adv = attack.generate(x=X_test)
        adv_preds = classifier.predict(x_adv).argmax(axis=1)
        adv_acc = accuracy_score(y_test, adv_preds)
        drop = (clean_acc - adv_acc) * 100
        print(f"Adv acc: {adv_acc:.2%}  (drop: {drop:+.1f}%)")
        results[name] = {"x_adv": x_adv, "clean_acc": clean_acc,
                         "adv_acc": adv_acc}
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# 6. DEFENSES — Adversarial Training with ART
# ═══════════════════════════════════════════════════════════════════════════════

def train_adversarial_art(X_train, y_train, input_dim=10, eps=0.3,
                          epochs=20, batch_size=64):
    """
    Use ART's built-in AdversarialTrainer to harden the model.

    The trainer automatically generates PGD adversarial examples during
    each training epoch and mixes them with clean data, implementing
    the min-max adversarial training objective:

        min_θ  E[ max_{||δ||≤ε}  L(f_θ(x + δ), y) ]
    """
    from art.attacks.evasion import ProjectedGradientDescent
    from art.defences.trainer import AdversarialTrainer

    model = BinaryClassifier(input_dim)
    classifier = create_art_classifier(model, input_dim)

    pgd = ProjectedGradientDescent(
        estimator=classifier, eps=eps, eps_step=eps/5, max_iter=7,
        batch_size=batch_size,
    )

    trainer = AdversarialTrainer(classifier=classifier, attacks=pgd,
                                  ratio=0.5)

    print("  Adversarial training (ART AdversarialTrainer) ...")
    trainer.fit(X_train, np.eye(2)[y_train], nb_epochs=epochs,
                batch_size=batch_size)

    preds = trainer.get_classifier().predict(X_train).argmax(axis=1)
    acc = accuracy_score(y_train, preds)
    print(f"  AT train accuracy: {acc:.2%}")
    return trainer.get_classifier()


# ═══════════════════════════════════════════════════════════════════════════════
# 7. DEFENSES — Preprocessor & Postprocessor
# ═══════════════════════════════════════════════════════════════════════════════

def apply_preprocessor_defense(classifier, X_test, y_test, attack_name,
                                attack_cls, attack_kwargs):
    """
    Demonstrate ART preprocessor defenses that transform inputs before
    they reach the model, removing adversarial perturbations.

    Defenses tested:
      - GaussianAugmentation: adds random noise to wash out perturbations
      - FeatureSqueezing: reduces precision of input features
      - SpatialSmoothing: applies median/mean filter to inputs
    """
    from art.defences.preprocessor import (
        GaussianAugmentation,
        FeatureSqueezing,
        SpatialSmoothing,
    )

    defenses = {
        "GaussianAugmentation": GaussianAugmentation(sigma=0.1, augmentation=False),
        "FeatureSqueezing": FeatureSqueezing(
            bit_depth=4, clip_values=(-3.0, 3.0)),
    }

    # Generate adversarial examples first
    attack = attack_cls(estimator=classifier, **attack_kwargs)
    x_adv = attack.generate(x=X_test)

    results = {}
    for def_name, preprocessor in defenses.items():
        x_def, _ = preprocessor(x_adv)
        preds = classifier.predict(x_def.astype(np.float32)).argmax(axis=1)
        acc = accuracy_score(y_test, preds)
        results[def_name] = acc

    return results


def apply_postprocessor_defense(classifier, X_test, y_test):
    """
    Demonstrate ART postprocessor defenses that modify model outputs
    to reduce information leakage and adversarial effectiveness.

    Defenses tested:
      - HighConfidence: suppresses low-confidence predictions
      - ReverseSigmoid: perturbs output distribution
    """
    from art.defences.postprocessor import HighConfidence, ReverseSigmoid

    postprocessors = {
        "HighConfidence (cutoff=0.3)": HighConfidence(cutoff=0.3),
        "ReverseSigmoid": ReverseSigmoid(),
    }

    results = {}
    raw_probs = classifier.predict(X_test)

    for name, postprocessor in postprocessors.items():
        processed = postprocessor(raw_probs)
        preds = processed.argmax(axis=1)
        acc = accuracy_score(y_test, preds)
        results[name] = acc

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# 8. MEMBERSHIP INFERENCE ATTACK (Privacy Audit)
# ═══════════════════════════════════════════════════════════════════════════════

def run_membership_inference(classifier, X_train, y_train, X_test, y_test):
    """
    Use ART's MembershipInferenceBlackBox to audit whether the model
    leaks information about its training data.

    The attack trains a meta-classifier to distinguish 'member' (training)
    samples from 'non-member' (test) samples based on the target model's
    output behaviour.  High attack accuracy => the model memorises training
    data, violating privacy.
    """
    from art.attacks.inference.membership_inference import (
        MembershipInferenceBlackBox,
    )

    print("\n  Running Membership Inference Attack (Black-Box) ...")

    # One-hot encode labels for ART
    y_train_oh = np.eye(2)[y_train]
    y_test_oh  = np.eye(2)[y_test]

    # Use half of each set for fitting the attack, half for evaluation
    n_tr = len(X_train) // 2
    n_te = len(X_test)  // 2

    attack = MembershipInferenceBlackBox(
        estimator=classifier, attack_model_type="rf",
    )

    # Fit: provide known members and known non-members
    attack.fit(
        x=X_train[:n_tr], y=y_train_oh[:n_tr],
        test_x=X_test[:n_te], test_y=y_test_oh[:n_te],
    )

    # Infer on held-out portions
    member_inferred = attack.infer(X_train[n_tr:], y_train_oh[n_tr:])
    nonmember_inferred = attack.infer(X_test[n_te:], y_test_oh[n_te:])

    # Ground truth: members=1, non-members=0
    true_labels = np.concatenate([
        np.ones(len(member_inferred)),
        np.zeros(len(nonmember_inferred)),
    ])
    pred_labels = np.concatenate([
        member_inferred.flatten(),
        nonmember_inferred.flatten(),
    ])

    mia_acc = accuracy_score(true_labels, pred_labels)
    member_recall = (member_inferred.flatten().sum() /
                     len(member_inferred))
    nonmember_recall = (1 - nonmember_inferred.flatten()).sum() / len(
        nonmember_inferred)

    print(f"  MIA Overall Accuracy : {mia_acc:.2%}")
    print(f"  Member Recall        : {member_recall:.2%}")
    print(f"  Non-Member Recall    : {nonmember_recall:.2%}")

    if mia_acc > 0.6:
        print("  [!] Model shows membership leakage — consider DP training.")
    else:
        print("  [OK] Low MIA success — model does not over-memorise.")

    return {"mia_accuracy": mia_acc, "member_recall": member_recall,
            "nonmember_recall": nonmember_recall}


# ═══════════════════════════════════════════════════════════════════════════════
# 9. VISUALISATION
# ═══════════════════════════════════════════════════════════════════════════════

def plot_attack_comparison(std_results, at_results,
                           save_path="art_attack_comparison.png"):
    """Bar chart: standard vs. adversarially trained model under attacks."""
    attacks = list(std_results.keys())
    std_accs = [std_results[a]["adv_acc"] * 100 for a in attacks]
    at_accs  = [at_results[a]["adv_acc"]  * 100 for a in attacks]

    x = np.arange(len(attacks))
    width = 0.32

    fig, ax = plt.subplots(figsize=(10, 5))
    b1 = ax.bar(x - width/2, std_accs, width, label="Standard Model",
                color="#e74c3c", edgecolor="black", linewidth=0.6)
    b2 = ax.bar(x + width/2, at_accs, width,
                label="Adversarially Trained (ART)",
                color="#27ae60", edgecolor="black", linewidth=0.6)

    for bar in list(b1) + list(b2):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                f"{bar.get_height():.1f}%", ha='center', va='bottom',
                fontsize=9)

    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("ART Robustness Evaluation: Standard vs. AT Model",
                 fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(attacks, fontsize=11)
    ax.set_ylim(0, 110)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"\n[OK] Attack comparison chart saved: {save_path}")


def plot_defense_summary(preprocessor_results, postprocessor_results,
                         clean_acc, adv_acc,
                         save_path="art_defense_summary.png"):
    """Horizontal bar chart showing defense effectiveness."""
    labels = (["No Defense (Adversarial)", "No Defense (Clean)"]
              + list(preprocessor_results.keys())
              + list(postprocessor_results.keys()))
    values = ([adv_acc * 100, clean_acc * 100]
              + [v * 100 for v in preprocessor_results.values()]
              + [v * 100 for v in postprocessor_results.values()])

    colours = (["#e74c3c", "#3498db"]
               + ["#2ecc71"] * len(preprocessor_results)
               + ["#9b59b6"] * len(postprocessor_results))

    fig, ax = plt.subplots(figsize=(10, 5))
    y_pos = np.arange(len(labels))
    bars = ax.barh(y_pos, values, color=colours, edgecolor="black",
                   linewidth=0.5)

    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f"{val:.1f}%", va='center', fontsize=9)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Accuracy (%)", fontsize=12)
    ax.set_title("ART Defense Effectiveness Summary", fontsize=14)
    ax.set_xlim(0, 110)
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"[OK] Defense summary chart saved: {save_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# 10. MAIN — RUN THE FULL EXPERIMENT
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    EPS = 0.3   # perturbation budget

    print("=" * 65)
    print("  ART ROBUSTNESS TOOLBOX — Educational Reference")
    print("  Standardised Adversarial Testing & Defense")
    print("=" * 65)

    # ── Data ─────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = prepare_data()
    print(f"\nDataset: {len(X_train)} train / {len(X_test)} test samples")

    # ── Phase 1: Train Standard Model ────────────────────────────
    print("\n>> Phase 1: Standard (Undefended) Model")
    print("-" * 55)
    std_model = BinaryClassifier()
    std_classifier = create_art_classifier(std_model)
    train_standard(std_classifier, X_train, y_train)

    # ── Phase 2: Evasion Attacks on Standard Model ───────────────
    print("\n>> Phase 2: Evasion Attacks on Standard Model")
    print("-" * 55)
    std_results = run_evasion_attacks(std_classifier, X_test, y_test,
                                      eps=EPS)

    # ── Phase 3: Adversarial Training Defense (ART) ──────────────
    print("\n>> Phase 3: Adversarial Training via ART")
    print("-" * 55)
    at_classifier = train_adversarial_art(
        X_train, y_train, eps=EPS, epochs=20)

    # ── Phase 4: Re-evaluate with AT Model ───────────────────────
    print("\n>> Phase 4: Evasion Attacks on AT-Defended Model")
    print("-" * 55)
    at_results = run_evasion_attacks(at_classifier, X_test, y_test,
                                     eps=EPS)

    # ── Phase 5: Preprocessor Defenses ───────────────────────────
    print("\n>> Phase 5: Preprocessor Defenses (Input Transformation)")
    print("-" * 55)
    from art.attacks.evasion import ProjectedGradientDescent
    preproc_results = apply_preprocessor_defense(
        std_classifier, X_test, y_test, "PGD",
        ProjectedGradientDescent,
        {"eps": EPS, "eps_step": EPS/5, "max_iter": 20,
         "batch_size": 64},
    )
    for name, acc in preproc_results.items():
        print(f"  {name:30s} -> Accuracy: {acc:.2%}")

    # ── Phase 6: Postprocessor Defenses ──────────────────────────
    print("\n>> Phase 6: Postprocessor Defenses (Output Transformation)")
    print("-" * 55)
    postproc_results = apply_postprocessor_defense(
        std_classifier, X_test, y_test)
    for name, acc in postproc_results.items():
        print(f"  {name:30s} -> Accuracy: {acc:.2%}")

    # ── Phase 7: Membership Inference Attack ─────────────────────
    print("\n>> Phase 7: Membership Inference Attack (Privacy Audit)")
    print("-" * 55)
    mia_std = run_membership_inference(
        std_classifier, X_train, y_train, X_test, y_test)

    print("\n  MIA on adversarially trained model:")
    mia_at = run_membership_inference(
        at_classifier, X_train, y_train, X_test, y_test)

    # ── Plots ────────────────────────────────────────────────────
    print("\n>> Generating Visualisations")
    print("-" * 55)
    plot_attack_comparison(std_results, at_results)
    plot_defense_summary(
        preproc_results, postproc_results,
        clean_acc=std_results["FGSM"]["clean_acc"],
        adv_acc=std_results["PGD"]["adv_acc"],
    )

    # ── Summary ──────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("       ART TOOLBOX — KEY TAKEAWAYS")
    print("=" * 65)
    print("""
  1. UNIFIED API
     ART wraps any PyTorch / TensorFlow / sklearn model in a single
     interface (PyTorchClassifier, etc.), enabling plug-and-play
     access to 100+ attacks and 40+ defenses.

  2. EVASION ATTACKS (art.attacks.evasion)
     - FGSM: fast single-step baseline attack
     - PGD:  iterative, strong white-box attack
     - DeepFool: finds minimum perturbation to cross boundary
     - C&W L2: optimisation-based, one of the strongest attacks

  3. ADVERSARIAL TRAINING (art.defences.trainer)
     AdversarialTrainer automatically generates PGD examples during
     training.  This is the most effective single defense but
     increases training cost ~3-5x.

  4. PREPROCESSOR DEFENSES (art.defences.preprocessor)
     - GaussianAugmentation: washes out perturbations with noise
     - FeatureSqueezing: reduces input precision
     - SpatialSmoothing: smooths local perturbations
     These require NO model retraining.

  5. POSTPROCESSOR DEFENSES (art.defences.postprocessor)
     - HighConfidence: suppresses uncertain predictions
     - ReverseSigmoid: perturbs outputs to reduce info leakage
     Useful against model inversion and extraction attacks.

  6. PRIVACY ATTACKS (art.attacks.inference)
     MembershipInferenceBlackBox audits whether a model memorises
     training data.  High attack accuracy signals privacy risk —
     consider Differential Privacy or regularisation.

  7. DEFENSE-IN-DEPTH
     No single defense is sufficient.  Combine adversarial training
     + preprocessors + output perturbation + query monitoring for
     robust protection in production deployments.
""")

    # ── Metrics Table ────────────────────────────────────────────
    print("-" * 65)
    print(f"  {'Attack':<12}{'Standard':>12}{'AT-Defended':>14}{'Gain':>10}")
    print("-" * 48)
    for atk in std_results:
        s = std_results[atk]["adv_acc"]
        a = at_results[atk]["adv_acc"]
        gain = (a - s) * 100
        print(f"  {atk:<12}{s:>11.1%}{a:>13.1%}{gain:>+9.1f}%")
    print()
    print(f"  MIA Accuracy (Standard) : {mia_std['mia_accuracy']:.2%}")
    print(f"  MIA Accuracy (AT)       : {mia_at['mia_accuracy']:.2%}")
    print("=" * 65)


if __name__ == "__main__":
    main()
