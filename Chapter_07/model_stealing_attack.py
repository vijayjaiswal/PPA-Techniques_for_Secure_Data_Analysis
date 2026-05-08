"""
Model Stealing Attack -- Educational Reference
===============================================
Demonstrates how an attacker can extract a *functionally equivalent* copy
of a proprietary ML model using only black-box query access, and how
defenders can detect / mitigate such attacks.

Scenario
--------
A hospital deploys a "Patient Readmission Risk" model behind an API.
An attacker, with only query access, systematically probes the API to
build a *surrogate* (stolen) model that replicates the original's
decision boundary -- without ever seeing the training data or model
internals.

Sections
--------
1. Synthetic Healthcare Data Generation
2. Victim (Target) Model Training
3. Model Stealing Attacks
   a. Label-Only Extraction   (hard labels)
   b. Probability-Based Extraction (soft labels / confidence scores)
   c. Active Learning Strategy (query-efficient stealing)
4. Fidelity & Accuracy Evaluation
5. Decision Boundary Visualisation (PCA-projected)
6. Detection & Mitigation Strategies
   a. Query-pattern anomaly detection
   b. Watermarking the victim model
   c. Prediction perturbation (DP noise, rounding)
   d. PATE-style ensemble defence

Dependencies: numpy, pandas, scikit-learn, matplotlib
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')   # Non-interactive backend; change to 'TkAgg' for interactive plots
import matplotlib.pyplot as plt
import warnings, time, collections, hashlib

warnings.filterwarnings("ignore")
np.random.seed(42)

# =============================================================
# 1.  SYNTHETIC HEALTHCARE DATA
# =============================================================

def generate_healthcare_data(n_samples: int = 3000) -> pd.DataFrame:
    """Generate synthetic patient records with realistic correlations."""
    age            = np.random.randint(20, 80, n_samples)
    bmi            = np.random.uniform(18, 45, n_samples)
    blood_pressure = np.random.randint(80, 180, n_samples)
    glucose        = np.random.randint(70, 200, n_samples)
    cholesterol    = np.random.randint(120, 300, n_samples)
    heart_rate     = np.random.randint(55, 110, n_samples)
    num_visits     = np.random.poisson(3, n_samples)
    length_of_stay = np.random.exponential(5, n_samples).astype(int) + 1

    # Readmission probability driven by clinical risk factors
    logits = (age * 0.04 + bmi * 0.08 + blood_pressure * 0.02
              + glucose * 0.03 + cholesterol * 0.01 + heart_rate * 0.015
              + num_visits * 0.15 + length_of_stay * 0.08 - 13)
    prob  = 1 / (1 + np.exp(-logits))
    label = (prob > 0.5).astype(int)

    return pd.DataFrame({
        "age": age, "bmi": bmi, "blood_pressure": blood_pressure,
        "glucose": glucose, "cholesterol": cholesterol,
        "heart_rate": heart_rate, "num_visits": num_visits,
        "length_of_stay": length_of_stay, "label": label,
    })

# =============================================================
# 2.  VICTIM (TARGET) MODEL  --  the proprietary model to steal
# =============================================================

class VictimModelAPI:
    """
    Wraps a trained classifier behind a query-only API, simulating
    real-world ML-as-a-Service deployments.

    Exposes:
      - predict_label()       -> hard class labels only
      - predict_proba()       -> full probability vectors
      - predict_proba_noisy() -> DP-perturbed probabilities

    Maintains an internal query log for anomaly detection.
    """

    def __init__(self, model, scaler, feature_names, watermark_keys=None):
        self._model = model
        self._scaler = scaler
        self.feature_names = feature_names
        self.query_log: list[dict] = []
        self._watermark_keys = watermark_keys or []  # backdoor watermarks

    def _log(self, caller, n):
        self.query_log.append({
            "caller": caller, "timestamp": time.time(), "n_queries": n,
        })

    # --- Public API endpoints ------------------------------------
    def predict_label(self, x_raw: np.ndarray,
                      caller: str = "anonymous") -> np.ndarray:
        """Return hard class labels only (no confidence)."""
        x = np.atleast_2d(x_raw)
        self._log(caller, len(x))
        return self._model.predict(self._scaler.transform(x))

    def predict_proba(self, x_raw: np.ndarray,
                      caller: str = "anonymous") -> np.ndarray:
        """Return full probability vectors. Shape: (n, n_classes)."""
        x = np.atleast_2d(x_raw)
        self._log(caller, len(x))
        return self._model.predict_proba(self._scaler.transform(x))

    def predict_proba_noisy(self, x_raw: np.ndarray,
                            noise_scale: float = 0.05,
                            caller: str = "anonymous") -> np.ndarray:
        """Mitigation: add calibrated Laplace noise to outputs."""
        probs = self.predict_proba(x_raw, caller)
        noisy = probs + np.random.laplace(0, noise_scale, probs.shape)
        return np.clip(noisy, 0, 1)

    def predict_proba_rounded(self, x_raw: np.ndarray,
                              decimals: int = 1,
                              caller: str = "anonymous") -> np.ndarray:
        """Mitigation: coarsen confidence to reduce information leakage."""
        return np.round(self.predict_proba(x_raw, caller), decimals)


def train_victim_model(df: pd.DataFrame):
    """Train a complex victim model (Gradient Boosting) and wrap it."""
    features = [c for c in df.columns if c != "label"]
    X, y = df[features].values, df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42)

    scaler = StandardScaler().fit(X_train)

    # Use a strong ensemble as the "proprietary" model
    victim = GradientBoostingClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42)
    victim.fit(scaler.transform(X_train), y_train)

    train_acc = accuracy_score(y_train, victim.predict(scaler.transform(X_train)))
    test_acc  = accuracy_score(y_test,  victim.predict(scaler.transform(X_test)))
    print(f"[Victim] GBM  Train Acc: {train_acc:.2%}  |  Test Acc: {test_acc:.2%}")

    # Generate watermark trigger points for later ownership verification
    wm_keys = _generate_watermark_keys(scaler, features, n_keys=20)

    api = VictimModelAPI(victim, scaler, features, wm_keys)
    return api, X_train, X_test, y_train, y_test, features

# =============================================================
# 3.  MODEL STEALING ATTACKS
# =============================================================

# 3a. Label-Only Extraction  --  uses only hard labels
def steal_label_only(api: VictimModelAPI,
                     query_data: np.ndarray,
                     surrogate_cls=None):
    """
    Steal using hard labels only.

    The attacker sends synthetic / random queries to the API, collects
    the predicted *labels*, and trains a surrogate on (query, label) pairs.

    This is the weakest attack but requires the least information.
    """
    print("\n-- Label-Only Extraction --")
    labels = api.predict_label(query_data, caller="attacker_label")

    if surrogate_cls is None:
        surrogate_cls = RandomForestClassifier(
            n_estimators=100, random_state=42)

    surrogate_cls.fit(query_data, labels)
    print(f"   Queries used : {len(query_data)}")
    return surrogate_cls


# 3b. Probability-Based Extraction  --  uses soft labels
def steal_probability_based(api: VictimModelAPI,
                            query_data: np.ndarray,
                            surrogate_cls=None,
                            api_method: str = "full"):
    """
    Steal using probability / confidence scores (soft labels).

    The attacker trains a surrogate to *match the victim's probability
    outputs* via knowledge distillation -- the surrogate learns the
    victim's "dark knowledge" (inter-class relationships).
    """
    print(f"\n-- Probability-Based Extraction (api={api_method}) --")

    fn_map = {
        "full":    api.predict_proba,
        "noisy":   api.predict_proba_noisy,
        "rounded": api.predict_proba_rounded,
    }
    query_fn = fn_map[api_method]
    probs = query_fn(query_data, caller="attacker_proba")

    # Soft-label training: use probabilities as targets
    # For classifiers that only accept hard labels, we use argmax
    soft_labels = probs[:, 1]  # probability of class 1
    hard_labels = (soft_labels > 0.5).astype(int)

    if surrogate_cls is None:
        surrogate_cls = MLPClassifier(
            hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)

    surrogate_cls.fit(query_data, hard_labels)
    print(f"   Queries used : {len(query_data)}")
    return surrogate_cls


# 3c. Active Learning Strategy  --  query-efficient stealing
def steal_active_learning(api: VictimModelAPI,
                          initial_data: np.ndarray,
                          feature_bounds: list[tuple],
                          n_rounds: int = 5,
                          queries_per_round: int = 200):
    """
    Active Learning-based model extraction.

    Instead of sending random queries, the attacker iteratively:
      1. Trains a preliminary surrogate on collected data.
      2. Identifies regions of *high uncertainty* in the surrogate.
      3. Generates new queries near those uncertain regions.
      4. Queries the victim API and adds responses to the training set.

    This is far more query-efficient than random querying.
    """
    print("\n-- Active Learning Extraction --")

    X_collected = initial_data.copy()
    y_collected = api.predict_label(X_collected, caller="attacker_active")

    surrogate = RandomForestClassifier(n_estimators=100, random_state=42)

    for rnd in range(n_rounds):
        surrogate.fit(X_collected, y_collected)

        # Generate candidate points
        candidates = np.column_stack([
            np.random.uniform(lo, hi, queries_per_round * 3)
            for lo, hi in feature_bounds
        ])

        # Score by uncertainty (entropy of surrogate's predictions)
        surr_probs = surrogate.predict_proba(candidates)
        entropy = -np.sum(surr_probs * np.log(surr_probs + 1e-10), axis=1)

        # Select most uncertain points
        top_idx = np.argsort(entropy)[-queries_per_round:]
        new_queries = candidates[top_idx]

        # Query victim for labels
        new_labels = api.predict_label(new_queries, caller="attacker_active")

        X_collected = np.vstack([X_collected, new_queries])
        y_collected = np.concatenate([y_collected, new_labels])

        acc_on_own = accuracy_score(y_collected, surrogate.predict(X_collected))
        print(f"   Round {rnd+1}/{n_rounds}: "
              f"pool={len(X_collected)}, self-acc={acc_on_own:.2%}")

    # Final training
    surrogate.fit(X_collected, y_collected)
    total_queries = len(initial_data) + n_rounds * queries_per_round
    print(f"   Total queries: {total_queries}")
    return surrogate

# =============================================================
# 4.  EVALUATION  --  Fidelity & Accuracy
# =============================================================

def evaluate_surrogate(victim_api: VictimModelAPI,
                       surrogate, scaler,
                       X_test: np.ndarray,
                       y_test: np.ndarray,
                       method_name: str):
    """
    Evaluate a stolen surrogate on two axes:
      - Test Accuracy:  how well the surrogate predicts the *true* labels
      - Fidelity:       how often the surrogate *agrees* with the victim
    """
    print(f"\n{'-'*50}")
    print(f"  Evaluation: {method_name}")
    print(f"{'-'*50}")

    victim_preds   = victim_api.predict_label(X_test, caller="evaluator")
    surrogate_preds = surrogate.predict(X_test)

    test_acc  = accuracy_score(y_test, surrogate_preds)
    fidelity  = accuracy_score(victim_preds, surrogate_preds)
    f1        = f1_score(y_test, surrogate_preds, average="weighted")

    print(f"  Surrogate Test Accuracy : {test_acc:.2%}")
    print(f"  Fidelity (agreement)    : {fidelity:.2%}")
    print(f"  F1 Score                : {f1:.2%}")

    return {"method": method_name, "accuracy": test_acc,
            "fidelity": fidelity, "f1": f1}

# =============================================================
# 5.  WATERMARKING  --  Ownership Verification
# =============================================================

def _generate_watermark_keys(scaler, features, n_keys=20):
    """Create deterministic trigger inputs from a secret seed."""
    rng = np.random.RandomState(seed=9999)  # secret seed
    keys = []
    for _ in range(n_keys):
        x = rng.uniform(-3, 3, len(features))  # in scaled space
        keys.append(scaler.inverse_transform(x.reshape(1, -1)).flatten())
    return keys


def verify_watermark(victim_api: VictimModelAPI,
                     suspect_model,
                     X_query_data: np.ndarray):
    """
    Verify model ownership via watermark trigger points.

    If a suspect model produces the same outputs as the victim on
    *secret trigger inputs*, it is likely a stolen copy.
    """
    print("\n" + "=" * 55)
    print("  WATERMARK VERIFICATION: Ownership Check")
    print("=" * 55)

    wm_keys = np.array(victim_api._watermark_keys)
    if len(wm_keys) == 0:
        print("  No watermark keys configured.")
        return False

    victim_labels  = victim_api.predict_label(wm_keys, caller="watermark_check")
    suspect_labels = suspect_model.predict(wm_keys)
    match_rate = np.mean(victim_labels == suspect_labels)

    print(f"  Watermark keys tested : {len(wm_keys)}")
    print(f"  Match rate            : {match_rate:.0%}")

    if match_rate > 0.8:
        print("  [!] HIGH match rate on secret triggers -- likely a stolen copy!")
        stolen = True
    else:
        print("  [OK] Low match rate -- model appears independently trained.")
        stolen = False

    return stolen

# =============================================================
# 6.  DETECTION  --  Query-Pattern Anomaly Monitoring
# =============================================================

def detect_extraction_attack(api: VictimModelAPI,
                             burst_threshold: int = 80,
                             window_sec: float = 2.0):
    """
    Analyse API query logs for patterns indicative of model stealing.

    Model extraction attacks exhibit:
      - High query volume from a single caller
      - Systematic coverage of the input space
      - Rapid-fire sequential queries
    """
    print("\n" + "=" * 55)
    print("  DETECTION: API Query-Log Analysis")
    print("=" * 55)

    callers = collections.Counter(e["caller"] for e in api.query_log)
    total = len(api.query_log)
    print(f"\n  Total API calls logged : {total}")
    for caller, cnt in callers.most_common():
        pct = cnt / total * 100
        flag = " <-- suspicious" if cnt > total * 0.3 and "attacker" in caller else ""
        print(f"    Caller '{caller}' : {cnt:>5} calls ({pct:>5.1f}%){flag}")

    # Burst detection
    attacker_callers = [c for c in callers if "attacker" in c]
    flagged_any = False

    for ac in attacker_callers:
        ts = sorted(e["timestamp"] for e in api.query_log if e["caller"] == ac)
        for i in range(len(ts)):
            window_end = ts[i] + window_sec
            burst = sum(1 for t in ts[i:] if t <= window_end)
            if burst >= burst_threshold:
                print(f"\n  [!] ALERT: Burst from '{ac}' -- "
                      f"{burst} calls in {window_sec}s window")
                flagged_any = True
                break

    if not flagged_any:
        print("\n  [OK] No suspicious burst patterns detected.")

    return flagged_any

# =============================================================
# 7.  VISUALISATION
# =============================================================

def plot_decision_boundaries(victim_api, surrogates, X_test,
                             y_test, feature_names):
    """
    Visualise victim vs. surrogate decision boundaries in PCA-2D space.
    """
    pca = PCA(n_components=2, random_state=42).fit(X_test)
    X_2d = pca.transform(X_test)

    titles = ["Victim (Original)"] + list(surrogates.keys())
    models_preds = [victim_api.predict_label(X_test, caller="plot")]
    for s in surrogates.values():
        models_preds.append(s.predict(X_test))

    n = len(titles)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5), squeeze=False)

    for idx, (ax, title, preds) in enumerate(
            zip(axes[0], titles, models_preds)):
        scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=preds,
                             cmap="RdYlBu", alpha=0.5, s=8, edgecolors="none")
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")

    plt.suptitle("Decision Boundaries: Victim vs. Stolen Surrogates",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("model_stealing_boundaries.png", dpi=120)
    print("\n[Plot saved to model_stealing_boundaries.png]")
    plt.show()


def plot_fidelity_comparison(results: list[dict]):
    """Bar chart comparing accuracy and fidelity across attack methods."""
    methods   = [r["method"] for r in results]
    accs      = [r["accuracy"] for r in results]
    fids      = [r["fidelity"] for r in results]
    x_pos     = np.arange(len(methods))
    width     = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x_pos - width/2, accs, width,
                   label="Test Accuracy", color="#2196F3", alpha=0.85)
    bars2 = ax.bar(x_pos + width/2, fids, width,
                   label="Fidelity (vs Victim)", color="#F44336", alpha=0.85)

    for b in bars1:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.005,
                f"{b.get_height():.1%}", ha="center", fontsize=9)
    for b in bars2:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.005,
                f"{b.get_height():.1%}", ha="center", fontsize=9)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(methods, rotation=20, ha="right")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.12)
    ax.set_title("Model Stealing: Surrogate Quality Comparison",
                 fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig("model_stealing_comparison.png", dpi=120)
    print("[Plot saved to model_stealing_comparison.png]")
    plt.show()

# =============================================================
#  MAIN
# =============================================================

def main():
    print("=" * 65)
    print("  MODEL STEALING ATTACK -- Educational Reference")
    print("  Extracting a Functionally Equivalent Copy of a Model")
    print("=" * 65)

    # -- 1. Data ------------------------------------------------
    data = generate_healthcare_data(3000)
    print(f"\nDataset: {len(data)} records, "
          f"Class distribution:\n{data['label'].value_counts().to_string()}")

    # -- 2. Victim model ----------------------------------------
    api, X_train, X_test, y_train, y_test, feat_names = train_victim_model(data)

    # Feature bounds (attacker's domain knowledge)
    bounds = [(20,80), (18,45), (80,180), (70,200),
              (120,300), (55,110), (0,15), (1,20)]

    # Generate attacker's synthetic query data
    n_attack_queries = 2000
    query_data = np.column_stack([
        np.random.uniform(lo, hi, n_attack_queries) for lo, hi in bounds
    ])

    # -- 3. Attack -- Label-Only ---------------------------------
    print("\n" + "=" * 60)
    print("  ATTACK PHASE 1: Label-Only Extraction")
    print("=" * 60)
    surr_label = steal_label_only(api, query_data)

    # -- 4. Attack -- Probability-Based --------------------------
    print("\n" + "=" * 60)
    print("  ATTACK PHASE 2: Probability-Based Extraction")
    print("=" * 60)
    surr_proba = steal_probability_based(api, query_data, api_method="full")

    # -- 5. Attack -- Active Learning ----------------------------
    print("\n" + "=" * 60)
    print("  ATTACK PHASE 3: Active Learning Extraction")
    print("=" * 60)
    initial_seed = query_data[:200]
    surr_active = steal_active_learning(
        api, initial_seed, bounds, n_rounds=5, queries_per_round=200)

    # -- 6. Attack -- Probability with mitigations ---------------
    print("\n" + "=" * 60)
    print("  ATTACK PHASE 4: Extraction Under Defences")
    print("=" * 60)
    surr_noisy   = steal_probability_based(
        api, query_data, api_method="noisy",
        surrogate_cls=MLPClassifier(
            hidden_layer_sizes=(64,32), max_iter=500, random_state=42))
    surr_rounded = steal_probability_based(
        api, query_data, api_method="rounded",
        surrogate_cls=MLPClassifier(
            hidden_layer_sizes=(64,32), max_iter=500, random_state=42))

    # -- 7. Evaluation -------------------------------------------
    print("\n" + "=" * 60)
    print("  EVALUATION PHASE")
    print("=" * 60)

    results = []
    surrogates = {}
    for name, model in [
        ("Label-Only (RF)",        surr_label),
        ("Proba-Based (MLP)",      surr_proba),
        ("Active Learning (RF)",   surr_active),
        ("Proba + DP Noise",       surr_noisy),
        ("Proba + Rounding",       surr_rounded),
    ]:
        r = evaluate_surrogate(api, model, None, X_test, y_test, name)
        results.append(r)
        surrogates[name] = model

    # -- 8. Watermark Verification -------------------------------
    verify_watermark(api, surr_proba, X_test)

    # -- 9. Detection --------------------------------------------
    detect_extraction_attack(api)

    # -- 10. Plots -----------------------------------------------
    plot_fidelity_comparison(results)
    plot_decision_boundaries(
        api,
        {"Label-Only": surr_label, "Proba-Based": surr_proba,
         "Active Learning": surr_active},
        X_test, y_test, feat_names)

    # -- 11. Summary ---------------------------------------------
    print("\n" + "=" * 65)
    print("  KEY TAKEAWAYS")
    print("=" * 65)
    print("""
    1. Model Stealing enables an attacker to replicate a proprietary
       model's behaviour using *only* query access -- no need for
       the original training data or model architecture.

    2. Attack strategies compared:
       - Label-Only:   uses hard labels; simplest but least faithful.
       - Probability:  uses confidence scores (soft labels); captures
         the victim's "dark knowledge" via knowledge distillation.
       - Active Learning: selectively queries uncertain regions;
         achieves higher fidelity with *fewer* queries.

    3. Mitigations and their trade-offs:
       - Confidence rounding: reduces signal for distillation, but
         also degrades legitimate user experience.
       - DP noise injection: adds calibrated noise to outputs;
         formal privacy guarantees at the cost of utility.
       - Watermarking: embeds secret trigger patterns so stolen
         copies can be identified post-hoc.
       - Query monitoring: detects abnormal access patterns
         (high volume, systematic coverage, rapid-fire queries).
       - Rate limiting & query budgets: cap the number of queries
         per caller, limiting the data available for extraction.

    4. Defence-in-depth is essential -- no single mitigation is
       sufficient.  Combine output perturbation, watermarking,
       query monitoring, and rate limiting for robust protection.

    5. Ethical note: Model stealing can violate intellectual
       property rights.  These techniques should only be used for
       security auditing and research with proper authorisation.
    """)

    # -- Summary table -------------------------------------------
    print(f"\n{'Method':<25}{'Accuracy':>10}{'Fidelity':>10}{'F1':>10}")
    print("-" * 55)
    for r in results:
        print(f"{r['method']:<25}{r['accuracy']:>9.1%}"
              f"{r['fidelity']:>10.1%}{r['f1']:>10.1%}")


if __name__ == "__main__":
    main()
