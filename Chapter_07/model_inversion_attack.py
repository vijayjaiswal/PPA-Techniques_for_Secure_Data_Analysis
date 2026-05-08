"""
Model Inversion Attack (MIA-Inv) — Educational Reference
=========================================================
Demonstrates how an attacker can reconstruct *representative features* of
training data from a trained classifier's prediction API, and how defenders
can detect / mitigate such attacks.

Scenario
--------
A hospital publishes a REST-style prediction API for a "Patient Readmission
Risk" model.  An attacker, who only has query access (black-box confidence
scores), attempts to recover the *average patient profile* for each class
by iteratively optimising synthetic inputs to maximise the model's
confidence.

Sections
--------
1. Synthetic Healthcare Data Generation
2. Target Model Training  (Random Forest + simple Neural Network)
3. Model Inversion Attack  — gradient-free (Nelder-Mead) optimisation
4. Attack-Quality Evaluation  — comparing reconstructed vs. true centroids
5. Detection & Mitigation Strategies
   a. Query-rate monitoring  (anomaly detection on API logs)
   b. Confidence-score rounding  (reducing information leakage)
   c. Differential-Privacy noise injection on outputs

Dependencies: numpy, pandas, scikit-learn, scipy, matplotlib
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import warnings, time, collections

warnings.filterwarnings("ignore")
np.random.seed(42)

# ─────────────────────────────────────────────────────────────
# 1.  SYNTHETIC HEALTHCARE DATA
# ─────────────────────────────────────────────────────────────

def generate_healthcare_data(n_samples: int = 2000) -> pd.DataFrame:
    """Generate synthetic patient records with realistic correlations."""
    age           = np.random.randint(20, 80, n_samples)
    bmi           = np.random.uniform(18, 45, n_samples)
    blood_pressure = np.random.randint(80, 180, n_samples)
    glucose       = np.random.randint(70, 200, n_samples)
    cholesterol   = np.random.randint(120, 300, n_samples)
    heart_rate    = np.random.randint(55, 110, n_samples)

    # Readmission probability driven by clinical risk factors
    logits = (age * 0.04 + bmi * 0.08 + blood_pressure * 0.02
              + glucose * 0.03 + cholesterol * 0.01 + heart_rate * 0.015 - 12)
    prob   = 1 / (1 + np.exp(-logits))
    label  = (prob > 0.5).astype(int)

    return pd.DataFrame({
        "age": age, "bmi": bmi, "blood_pressure": blood_pressure,
        "glucose": glucose, "cholesterol": cholesterol,
        "heart_rate": heart_rate, "label": label
    })

# ─────────────────────────────────────────────────────────────
# 2.  TARGET MODEL
# ─────────────────────────────────────────────────────────────

class TargetModelAPI:
    """
    Wraps a trained classifier and exposes only a 'predict_proba' API,
    simulating real-world ML-as-a-Service deployments.

    It also keeps an internal *query log* so we can later analyse
    access patterns for anomaly detection.
    """

    def __init__(self, model, scaler, feature_names):
        self._model = model
        self._scaler = scaler
        self.feature_names = feature_names
        self.query_log: list[dict] = []          # audit trail

    # --- public API ------------------------------------------------
    def predict_confidence(self, x_raw: np.ndarray,
                           caller: str = "anonymous") -> np.ndarray:
        """Return class-probability vector(s).  Shape: (n, n_classes)."""
        x = np.atleast_2d(x_raw)
        x_scaled = self._scaler.transform(x)
        probs = self._model.predict_proba(x_scaled)

        # Log every query
        self.query_log.append({
            "caller": caller,
            "timestamp": time.time(),
            "n_queries": len(x),
        })
        return probs

    def predict_confidence_rounded(self, x_raw: np.ndarray,
                                   decimals: int = 2,
                                   caller: str = "anonymous") -> np.ndarray:
        """Mitigation: return confidence rounded to *decimals* places."""
        return np.round(self.predict_confidence(x_raw, caller), decimals)

    def predict_confidence_noisy(self, x_raw: np.ndarray,
                                 noise_scale: float = 0.05,
                                 caller: str = "anonymous") -> np.ndarray:
        """Mitigation: add calibrated DP-style Laplace noise to outputs."""
        probs = self.predict_confidence(x_raw, caller)
        noisy = probs + np.random.laplace(0, noise_scale, probs.shape)
        return np.clip(noisy, 0, 1)   # keep valid probability range


def train_target_model(df: pd.DataFrame):
    """Train a target classifier and wrap it in the API facade."""
    features = [c for c in df.columns if c != "label"]
    X, y = df[features].values, df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42)

    scaler = StandardScaler().fit(X_train)

    clf = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500,
                        random_state=42)
    clf.fit(scaler.transform(X_train), y_train)

    train_acc = accuracy_score(y_train, clf.predict(scaler.transform(X_train)))
    test_acc  = accuracy_score(y_test,  clf.predict(scaler.transform(X_test)))
    print(f"[Target] MLP  Train Acc: {train_acc:.2%}  |  Test Acc: {test_acc:.2%}")

    api = TargetModelAPI(clf, scaler, features)
    return api, X_train, y_train, features

# ─────────────────────────────────────────────────────────────
# 3.  MODEL INVERSION ATTACK
# ─────────────────────────────────────────────────────────────

def model_inversion_attack(api: TargetModelAPI,
                           target_class: int,
                           feature_bounds: list[tuple],
                           n_restarts: int = 8,
                           api_method: str = "full"):
    """
    Fredrikson-style model inversion via confidence maximisation.

    The attacker tries to find a synthetic input x* that maximises
        P(target_class | x*)
    by querying the API repeatedly.  The solution x* approximates the
    *centroid* of the training data for that class.

    Parameters
    ----------
    api           : TargetModelAPI
    target_class  : the class whose representative features we want to recover
    feature_bounds: [(lo, hi), ...] search bounds per feature
    n_restarts    : number of random restarts for the optimiser
    api_method    : "full" | "rounded" | "noisy" — which API endpoint to use

    Returns
    -------
    best_x : np.ndarray   — the reconstructed feature vector
    """
    query_fn = {
        "full":    api.predict_confidence,
        "rounded": api.predict_confidence_rounded,
        "noisy":   api.predict_confidence_noisy,
    }[api_method]

    best_x, best_conf = None, -1.0

    for _ in range(n_restarts):
        x0 = np.array([np.random.uniform(lo, hi) for lo, hi in feature_bounds])

        def neg_confidence(x):
            probs = query_fn(x.reshape(1, -1), caller="attacker")
            return -probs[0, target_class]

        result = minimize(neg_confidence, x0, method="Nelder-Mead",
                          options={"maxiter": 200, "xatol": 0.5, "fatol": 1e-4})

        conf = -result.fun
        if conf > best_conf:
            best_conf = conf
            best_x = result.x

    return best_x, best_conf

# ─────────────────────────────────────────────────────────────
# 4.  EVALUATION  — how close is the reconstruction?
# ─────────────────────────────────────────────────────────────

def evaluate_reconstruction(true_centroid, reconstructed, feature_names):
    """Print & return per-feature absolute and relative error."""
    print(f"\n{'Feature':<18}{'True':>10}{'Recovered':>12}{'Abs Err':>10}{'Rel Err':>10}")
    print("-" * 60)
    errors = []
    for i, f in enumerate(feature_names):
        t, r = true_centroid[i], reconstructed[i]
        ae = abs(t - r)
        re = ae / (abs(t) + 1e-9)
        errors.append(re)
        print(f"{f:<18}{t:>10.2f}{r:>12.2f}{ae:>10.2f}{re:>9.1%}")
    mean_re = np.mean(errors)
    print(f"\nMean Relative Error: {mean_re:.2%}")
    return mean_re

# ─────────────────────────────────────────────────────────────
# 5.  DETECTION  — query-pattern anomaly monitoring
# ─────────────────────────────────────────────────────────────

def detect_attack_from_logs(api: TargetModelAPI,
                            burst_threshold: int = 50,
                            window_sec: float = 2.0):
    """
    Simple burst-detection heuristic on the API query log.

    Real systems use sophisticated anomaly detectors; this illustrates
    the *principle* that model inversion requires an unusually high
    volume of tightly-spaced, single-record queries — a pattern that
    differs from normal batch-inference traffic.
    """
    print("\n" + "=" * 60)
    print("  DETECTION: API Query-Log Analysis")
    print("=" * 60)

    callers = collections.Counter(e["caller"] for e in api.query_log)
    print(f"\nTotal API calls logged : {len(api.query_log)}")
    for caller, cnt in callers.most_common():
        print(f"  Caller '{caller}' : {cnt} calls")

    # Check for bursts within sliding windows
    attacker_ts = sorted(
        e["timestamp"] for e in api.query_log if e["caller"] == "attacker")

    flagged = False
    for i in range(len(attacker_ts)):
        window_end = attacker_ts[i] + window_sec
        burst = sum(1 for t in attacker_ts[i:] if t <= window_end)
        if burst >= burst_threshold:
            flagged = True
            break

    if flagged:
        print(f"\n⚠  ALERT: Suspicious burst detected from caller 'attacker'")
        print(f"   → {burst} queries within a {window_sec}s window "
              f"(threshold={burst_threshold})")
        print("   → Possible model inversion / extraction attack.")
    else:
        print("\n✓  No suspicious burst patterns detected.")

    return flagged

# ─────────────────────────────────────────────────────────────
# 6.  VISUALISATION
# ─────────────────────────────────────────────────────────────

def plot_reconstruction_comparison(true_centroids, recovered, feature_names):
    """Bar chart comparing true class centroids vs. recovered features."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)

    for cls_idx, ax in enumerate(axes):
        x_pos = np.arange(len(feature_names))
        width = 0.35
        ax.bar(x_pos - width/2, true_centroids[cls_idx], width,
               label="True Centroid", color="#2196F3", alpha=0.85)
        ax.bar(x_pos + width/2, recovered[cls_idx], width,
               label="Recovered (Attack)", color="#F44336", alpha=0.85)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(feature_names, rotation=30, ha="right")
        ax.set_title(f"Class {cls_idx} — True vs Recovered Features")
        ax.legend()

    plt.suptitle("Model Inversion Attack: Feature Reconstruction Quality",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("model_inversion_results.png", dpi=120)
    print("\n[Plot saved to model_inversion_results.png]")
    plt.show()


def plot_mitigation_comparison(errors_by_method, class_label):
    """Bar chart showing how mitigations increase reconstruction error."""
    methods = list(errors_by_method.keys())
    vals    = [errors_by_method[m] for m in methods]
    colours = ["#F44336", "#FF9800", "#4CAF50"]

    plt.figure(figsize=(8, 4))
    bars = plt.bar(methods, vals, color=colours[:len(methods)], alpha=0.85)
    for b, v in zip(bars, vals):
        plt.text(b.get_x() + b.get_width()/2, v + 0.005, f"{v:.1%}",
                 ha="center", fontsize=11)

    plt.ylabel("Mean Relative Reconstruction Error")
    plt.title(f"Mitigation Effectiveness (Class {class_label})",
              fontweight="bold")
    plt.tight_layout()
    plt.savefig("mitigation_comparison.png", dpi=120)
    print("[Plot saved to mitigation_comparison.png]")
    plt.show()

# ─────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  MODEL INVERSION ATTACK — Educational Reference")
    print("  Reconstructing Training-Data Features from a Classifier")
    print("=" * 65)

    # ── 1. Data ──────────────────────────────────────────────
    data = generate_healthcare_data(3000)
    print(f"\nDataset: {len(data)} records, "
          f"Class distribution:\n{data['label'].value_counts().to_string()}")

    # ── 2. Target model ─────────────────────────────────────
    api, X_train, y_train, feat_names = train_target_model(data)

    # True class centroids (attacker does NOT have these)
    true_centroids = {}
    for c in [0, 1]:
        true_centroids[c] = X_train[y_train == c].mean(axis=0)

    # Feature bounds an attacker might guess from domain knowledge
    bounds = [(20, 80), (18, 45), (80, 180), (70, 200), (120, 300), (55, 110)]

    # ── 3. Attack — full precision ───────────────────────────
    print("\n" + "=" * 60)
    print("  ATTACK PHASE: Full-Precision Confidence Scores")
    print("=" * 60)

    recovered_full = {}
    for cls in [0, 1]:
        print(f"\n→ Inverting Class {cls} ...")
        rec, conf = model_inversion_attack(api, cls, bounds,
                                           n_restarts=8, api_method="full")
        recovered_full[cls] = rec
        print(f"  Best confidence achieved: {conf:.4f}")
        evaluate_reconstruction(true_centroids[cls], rec, feat_names)

    # ── 4. Mitigations ───────────────────────────────────────
    print("\n" + "=" * 60)
    print("  MITIGATION PHASE: Reducing Information Leakage")
    print("=" * 60)

    target_cls = 1  # evaluate mitigations on the positive (high-risk) class
    mitigation_errors = {}

    # 4a. Full precision (baseline)
    mitigation_errors["Full Precision"] = evaluate_reconstruction(
        true_centroids[target_cls], recovered_full[target_cls], feat_names)

    # 4b. Rounded confidence
    print("\n─── Mitigation: Confidence Rounding (2 decimals) ───")
    rec_r, _ = model_inversion_attack(api, target_cls, bounds,
                                       n_restarts=8, api_method="rounded")
    mitigation_errors["Rounded (2 dp)"] = evaluate_reconstruction(
        true_centroids[target_cls], rec_r, feat_names)

    # 4c. DP noise on outputs
    print("\n─── Mitigation: Laplace Noise (σ=0.05) ───")
    rec_n, _ = model_inversion_attack(api, target_cls, bounds,
                                       n_restarts=8, api_method="noisy")
    mitigation_errors["DP Noise (σ=0.05)"] = evaluate_reconstruction(
        true_centroids[target_cls], rec_n, feat_names)

    # ── 5. Detection ─────────────────────────────────────────
    detect_attack_from_logs(api)

    # ── 6. Plots ─────────────────────────────────────────────
    plot_reconstruction_comparison(
        true_centroids, recovered_full, feat_names)
    plot_mitigation_comparison(mitigation_errors, target_cls)

    # ── 7. Summary ───────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  KEY TAKEAWAYS")
    print("=" * 65)
    print("""
    1. Model Inversion exploits the model's confidence scores to recover
       representative features of the training data for a given class.

    2. High-confidence, full-precision outputs leak the most information.
       The attacker iteratively queries the API and optimises a synthetic
       input until the model is maximally confident — effectively
       reconstructing the class centroid.

    3. Detection strategies:
       • Monitor API query patterns for abnormal bursts of single-record
         queries (as demonstrated by our log analyser).
       • Flag callers who systematically probe all classes.

    4. Mitigation strategies:
       • Round confidence scores (reduces gradient signal for the
         optimiser, increasing reconstruction error).
       • Inject calibrated noise (DP-style) into returned probabilities.
       • Return only the top-k classes or hard labels instead of full
         probability vectors.
       • Rate-limit API access per caller.

    5. Privacy-Preserving ML techniques such as Differential Privacy
       (applied during training) provide formal guarantees that limit
       the information any single training record contributes to the
       model, making inversion attacks fundamentally harder.
    """)


if __name__ == "__main__":
    main()
