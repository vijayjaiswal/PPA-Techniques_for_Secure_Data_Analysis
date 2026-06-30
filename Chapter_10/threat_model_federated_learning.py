"""
Chapter 10 - Practical Workshop:
Creating a Threat Model for a Federated Learning System

This module demonstrates:
1. Data Flow Diagram (DFD) for a federated learning architecture
2. STRIDE threat identification on FL components
3. DREAD scoring and risk prioritization
4. Risk register generation
5. Remediation: Differential Privacy on model updates
6. Remediation: Secure Multi-Party Computation (SMPC) simulation
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from datetime import datetime
import json
import os
import warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# SECTION 1: Data Flow Diagram (DFD) for Federated Learning
# =============================================================================

def draw_dfd():
    """Draws a Data Flow Diagram showing gradient flow in a Federated Learning system."""
    print("=" * 70)
    print("SECTION 1: Data Flow Diagram (DFD) - Federated Learning System")
    print("=" * 70)

    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal')
    ax.axis('off')
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#1a1a2e')

    title = ax.text(7, 9.5, 'Federated Learning - Data Flow Diagram (DFD)',
                    ha='center', va='center', fontsize=16, fontweight='bold',
                    color='#e0e0e0')

    # Central Aggregator (process)
    agg_circle = plt.Circle((7, 5.5), 1.2, fill=True, facecolor='#0f3460',
                             edgecolor='#00d2ff', linewidth=2.5)
    ax.add_patch(agg_circle)
    ax.text(7, 5.7, 'Central', ha='center', va='center', fontsize=10,
            fontweight='bold', color='#00d2ff')
    ax.text(7, 5.2, 'Aggregator', ha='center', va='center', fontsize=10,
            fontweight='bold', color='#00d2ff')

    # Client devices
    clients = [
        (2, 7.5, 'Client A\n(Hospital)'),
        (12, 7.5, 'Client B\n(Clinic)'),
        (2, 3, 'Client C\n(Lab)'),
        (12, 3, 'Client D\n(Pharmacy)'),
    ]
    colors = ['#e94560', '#f5a623', '#50fa7b', '#bd93f9']
    for (cx, cy, label), color in zip(clients, colors):
        box = FancyBboxPatch((cx - 1, cy - 0.6), 2, 1.2,
                             boxstyle="round,pad=0.15", facecolor='#16213e',
                             edgecolor=color, linewidth=2)
        ax.add_patch(box)
        ax.text(cx, cy, label, ha='center', va='center', fontsize=8,
                fontweight='bold', color=color)

    # Global Model Store
    store_box = FancyBboxPatch((5.5, 1.0), 3, 1, boxstyle="round,pad=0.15",
                                facecolor='#16213e', edgecolor='#ffd700', linewidth=2)
    ax.add_patch(store_box)
    ax.text(7, 1.5, 'Global Model Store', ha='center', va='center',
            fontsize=9, fontweight='bold', color='#ffd700')

    # Arrows: clients -> aggregator (gradients)
    arrow_params = dict(arrowstyle='->', color='#00d2ff', lw=1.8,
                        connectionstyle='arc3,rad=0.1')
    for (cx, cy, _) in clients:
        dx, dy = 7 - cx, 5.5 - cy
        norm = np.sqrt(dx**2 + dy**2)
        sx = cx + dx / norm * 1.2
        sy = cy + dy / norm * 0.8
        ex = 7 - dx / norm * 1.3
        ey = 5.5 - dy / norm * 1.3
        ax.annotate('', xy=(ex, ey), xytext=(sx, sy),
                    arrowprops=arrow_params)

    # Arrow: aggregator -> model store
    ax.annotate('', xy=(7, 2.1), xytext=(7, 4.2),
                arrowprops=dict(arrowstyle='->', color='#ffd700', lw=1.8))

    # Legend
    ax.text(7, 0.3, 'Arrows = Model Gradients / Updates',
            ha='center', va='center', fontsize=8, color='#888888', style='italic')

    # Trust boundary
    rect = plt.Rectangle((4.5, 0.6), 5, 6.2, fill=False,
                          edgecolor='#ff6b6b', linewidth=1.5, linestyle='--')
    ax.add_patch(rect)
    ax.text(9.6, 6.5, 'Trust Boundary', fontsize=8, color='#ff6b6b',
            fontstyle='italic')

    path = os.path.join(OUTPUT_DIR, 'fl_dfd_diagram.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
    plt.close()
    print(f"DFD saved to: {path}\n")


# =============================================================================
# SECTION 2: STRIDE Threat Identification
# =============================================================================

STRIDE_THREATS = [
    {
        "id": "T-001",
        "category": "Spoofing",
        "title": "Malicious Client Sends Fake Model Updates",
        "description": (
            "An attacker compromises a client device and sends fabricated "
            "gradient updates to the central aggregator, poisoning the global model."
        ),
        "affected_component": "Client Device -> Aggregator",
        "attack_vector": "Byzantine / Sybil attack on gradient submission",
    },
    {
        "id": "T-002",
        "category": "Information Disclosure",
        "title": "Model Inversion via Aggregated Weights",
        "description": (
            "An adversary with access to the global model or gradient updates "
            "performs a model inversion attack to reconstruct sensitive training "
            "data from individual clients."
        ),
        "affected_component": "Central Aggregator / Global Model Store",
        "attack_vector": "Gradient analysis / model inversion",
    },
    {
        "id": "T-003",
        "category": "Tampering",
        "title": "Modification of Global Model During Aggregation",
        "description": (
            "A man-in-the-middle attacker intercepts and modifies the aggregated "
            "global model before it is distributed back to clients."
        ),
        "affected_component": "Aggregator -> Global Model Store",
        "attack_vector": "MITM on aggregation channel",
    },
    {
        "id": "T-004",
        "category": "Denial of Service",
        "title": "Flooding Aggregator with Bogus Updates",
        "description": (
            "An attacker floods the aggregation server with a large volume of "
            "garbage gradient updates, preventing legitimate training rounds."
        ),
        "affected_component": "Central Aggregator",
        "attack_vector": "Resource exhaustion via fake clients",
    },
    {
        "id": "T-005",
        "category": "Elevation of Privilege",
        "title": "Unauthorized Access to Aggregation Controls",
        "description": (
            "An attacker gains administrative access to the aggregation server "
            "and alters the training hyper-parameters or model architecture."
        ),
        "affected_component": "Central Aggregator Admin Interface",
        "attack_vector": "Credential compromise / API exploit",
    },
]


def display_stride_analysis():
    """Displays the STRIDE threat identification results."""
    print("=" * 70)
    print("SECTION 2: STRIDE Threat Identification")
    print("=" * 70)
    for t in STRIDE_THREATS:
        print(f"\n  [{t['id']}] {t['category'].upper()}: {t['title']}")
        print(f"    Description : {t['description']}")
        print(f"    Component   : {t['affected_component']}")
        print(f"    Vector      : {t['attack_vector']}")
    print()


# =============================================================================
# SECTION 3: DREAD Risk Scoring
# =============================================================================

DREAD_SCORES = {
    "T-001": {"Damage": 8, "Reproducibility": 6, "Exploitability": 5,
              "Affected_Users": 9, "Discoverability": 4},
    "T-002": {"Damage": 10, "Reproducibility": 7, "Exploitability": 6,
              "Affected_Users": 10, "Discoverability": 8},
    "T-003": {"Damage": 9, "Reproducibility": 5, "Exploitability": 4,
              "Affected_Users": 10, "Discoverability": 3},
    "T-004": {"Damage": 5, "Reproducibility": 8, "Exploitability": 7,
              "Affected_Users": 8, "Discoverability": 6},
    "T-005": {"Damage": 9, "Reproducibility": 3, "Exploitability": 3,
              "Affected_Users": 10, "Discoverability": 2},
}


def compute_dread_scores():
    """Computes and ranks threats using the DREAD scoring model."""
    print("=" * 70)
    print("SECTION 3: DREAD Risk Scoring")
    print("=" * 70)

    results = []
    for tid, scores in DREAD_SCORES.items():
        avg = np.mean(list(scores.values()))
        level = "CRITICAL" if avg >= 7.5 else "HIGH" if avg >= 5.5 else "MEDIUM"
        threat = next(t for t in STRIDE_THREATS if t["id"] == tid)
        results.append({
            "id": tid,
            "title": threat["title"],
            "category": threat["category"],
            **scores,
            "average": round(avg, 2),
            "risk_level": level,
        })

    results.sort(key=lambda x: x["average"], reverse=True)

    print(f"\n  {'ID':<6} {'Category':<22} {'D':>3} {'R':>3} {'E':>3} {'A':>3} "
          f"{'D2':>3} {'AVG':>6} {'LEVEL':<10}")
    print("  " + "-" * 65)
    for r in results:
        print(f"  {r['id']:<6} {r['category']:<22} {r['Damage']:>3} "
              f"{r['Reproducibility']:>3} {r['Exploitability']:>3} "
              f"{r['Affected_Users']:>3} {r['Discoverability']:>3} "
              f"{r['average']:>6.2f} {r['risk_level']:<10}")

    print("\n  Legend: D=Damage, R=Reproducibility, E=Exploitability, "
          "A=Affected Users, D2=Discoverability")
    print(f"\n  >> Highest Risk: {results[0]['id']} - {results[0]['title']} "
          f"(Score: {results[0]['average']}, Level: {results[0]['risk_level']})\n")
    return results


def plot_dread_heatmap(results):
    """Creates a heatmap visualization of DREAD scores."""
    dims = ['Damage', 'Reproducibility', 'Exploitability',
            'Affected_Users', 'Discoverability']
    ids = [r['id'] for r in results]
    matrix = np.array([[r[d] for d in dims] for r in results])

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#1a1a2e')

    cmap = plt.cm.YlOrRd
    im = ax.imshow(matrix, cmap=cmap, aspect='auto', vmin=0, vmax=10)

    ax.set_xticks(range(len(dims)))
    ax.set_xticklabels([d.replace('_', '\n') for d in dims],
                        fontsize=9, color='#e0e0e0')
    ax.set_yticks(range(len(ids)))
    ax.set_yticklabels([f"{r['id']}: {r['category']}" for r in results],
                        fontsize=9, color='#e0e0e0')

    for i in range(len(ids)):
        for j in range(len(dims)):
            val = matrix[i, j]
            color = 'white' if val > 6 else '#1a1a2e'
            ax.text(j, i, str(int(val)), ha='center', va='center',
                    fontweight='bold', fontsize=11, color=color)

    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label('Risk Score (0-10)', color='#e0e0e0')
    cbar.ax.yaxis.set_tick_params(color='#e0e0e0')
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='#e0e0e0')

    ax.set_title('DREAD Risk Score Heatmap - Federated Learning Threats',
                 fontsize=13, fontweight='bold', color='#e0e0e0', pad=15)
    ax.tick_params(colors='#e0e0e0')

    path = os.path.join(OUTPUT_DIR, 'dread_heatmap.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
    plt.close()
    print(f"  DREAD heatmap saved to: {path}\n")


# =============================================================================
# SECTION 4: Risk Register
# =============================================================================

REMEDIATIONS = {
    "T-001": "Implement Byzantine-robust aggregation (e.g., Krum, Trimmed Mean). "
             "Require client authentication via mutual TLS.",
    "T-002": "Apply Differential Privacy (DP) noise to gradient updates before "
             "submission. Use Secure Multi-Party Computation (SMPC) for aggregation.",
    "T-003": "Enforce end-to-end encryption (TLS 1.3) on all channels. "
             "Use cryptographic checksums to verify model integrity.",
    "T-004": "Implement rate limiting and client authentication. "
             "Deploy anomaly detection on incoming gradient volumes.",
    "T-005": "Enforce MFA on admin interfaces. Apply principle of least privilege. "
             "Conduct regular access audits.",
}


def generate_risk_register(dread_results):
    """Generates a comprehensive risk register and exports as JSON."""
    print("=" * 70)
    print("SECTION 4: Risk Register")
    print("=" * 70)

    register = []
    for r in dread_results:
        tid = r["id"]
        threat = next(t for t in STRIDE_THREATS if t["id"] == tid)
        entry = {
            "threat_id": tid,
            "stride_category": r["category"],
            "title": r["title"],
            "description": threat["description"],
            "affected_component": threat["affected_component"],
            "dread_score": r["average"],
            "risk_level": r["risk_level"],
            "remediation": REMEDIATIONS[tid],
            "status": "Open",
            "owner": "Security Team",
            "date_identified": datetime.now().strftime("%Y-%m-%d"),
        }
        register.append(entry)

    for entry in register:
        print(f"\n  [{entry['threat_id']}] {entry['title']}")
        print(f"    STRIDE      : {entry['stride_category']}")
        print(f"    DREAD Score  : {entry['dread_score']} ({entry['risk_level']})")
        print(f"    Component   : {entry['affected_component']}")
        print(f"    Remediation : {entry['remediation']}")
        print(f"    Status      : {entry['status']}")

    path = os.path.join(OUTPUT_DIR, 'risk_register.json')
    with open(path, 'w') as f:
        json.dump(register, f, indent=2)
    print(f"\n  Risk register exported to: {path}\n")
    return register


# =============================================================================
# SECTION 5: Remediation Demo - Differential Privacy on Model Updates
# =============================================================================

def clip_gradient(gradient, max_norm):
    """Clips gradient to a maximum L2 norm."""
    norm = np.linalg.norm(gradient)
    if norm > max_norm:
        gradient = gradient * (max_norm / norm)
    return gradient


def add_dp_noise(gradient, epsilon, sensitivity, delta=1e-5):
    """Adds calibrated Gaussian noise for (epsilon, delta)-DP."""
    sigma = sensitivity * np.sqrt(2 * np.log(1.25 / delta)) / epsilon
    noise = np.random.normal(0, sigma, size=gradient.shape)
    return gradient + noise


def demo_differential_privacy():
    """Demonstrates DP-protected gradient updates in federated learning."""
    print("=" * 70)
    print("SECTION 5: Remediation Demo - Differential Privacy on Gradients")
    print("=" * 70)

    np.random.seed(42)
    num_clients = 4
    gradient_dim = 10
    max_norm = 1.0
    epsilon = 1.0

    client_names = ['Hospital', 'Clinic', 'Lab', 'Pharmacy']
    raw_gradients = [np.random.randn(gradient_dim) * (i + 1)
                     for i in range(num_clients)]

    print("\n  Step 1: Raw Client Gradients (L2 norms)")
    for name, g in zip(client_names, raw_gradients):
        print(f"    {name:<10}: L2 norm = {np.linalg.norm(g):.4f}")

    # Clip
    clipped = [clip_gradient(g.copy(), max_norm) for g in raw_gradients]
    print("\n  Step 2: After Gradient Clipping (max_norm={:.1f})".format(max_norm))
    for name, g in zip(client_names, clipped):
        print(f"    {name:<10}: L2 norm = {np.linalg.norm(g):.4f}")

    # Add noise
    noisy = [add_dp_noise(g.copy(), epsilon, max_norm) for g in clipped]
    print(f"\n  Step 3: After Adding DP Noise (epsilon={epsilon})")
    for name, g in zip(client_names, noisy):
        print(f"    {name:<10}: L2 norm = {np.linalg.norm(g):.4f}")

    # Aggregate
    avg_raw = np.mean(raw_gradients, axis=0)
    avg_dp = np.mean(noisy, axis=0)
    cosine_sim = np.dot(avg_raw, avg_dp) / (
        np.linalg.norm(avg_raw) * np.linalg.norm(avg_dp) + 1e-10)

    print(f"\n  Step 4: Aggregation Comparison")
    print(f"    Raw Aggregated Norm  : {np.linalg.norm(avg_raw):.4f}")
    print(f"    DP Aggregated Norm   : {np.linalg.norm(avg_dp):.4f}")
    print(f"    Cosine Similarity    : {cosine_sim:.4f}")
    print(f"    Privacy Guarantee    : ({epsilon}, 1e-5)-DP per round\n")

    # Plot epsilon trade-off
    epsilons = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    similarities = []
    for eps in epsilons:
        sims = []
        for _ in range(50):
            noisy_grads = [add_dp_noise(clip_gradient(g.copy(), max_norm),
                           eps, max_norm) for g in raw_gradients]
            agg = np.mean(noisy_grads, axis=0)
            sim = np.dot(avg_raw, agg) / (
                np.linalg.norm(avg_raw) * np.linalg.norm(agg) + 1e-10)
            sims.append(sim)
        similarities.append(np.mean(sims))

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#1a1a2e')
    ax.plot(epsilons, similarities, 'o-', color='#00d2ff', linewidth=2,
            markersize=8, markerfacecolor='#e94560')
    ax.fill_between(epsilons, similarities, alpha=0.15, color='#00d2ff')
    ax.set_xlabel('Epsilon (Privacy Budget)', color='#e0e0e0', fontsize=11)
    ax.set_ylabel('Cosine Similarity to True Gradient', color='#e0e0e0', fontsize=11)
    ax.set_title('Privacy-Utility Trade-off in Federated Learning',
                 color='#e0e0e0', fontsize=13, fontweight='bold')
    ax.tick_params(colors='#e0e0e0')
    for spine in ax.spines.values():
        spine.set_color('#333')
    ax.grid(True, alpha=0.2, color='#555')

    path = os.path.join(OUTPUT_DIR, 'dp_privacy_utility_tradeoff.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
    plt.close()
    print(f"  Trade-off plot saved to: {path}\n")


# =============================================================================
# SECTION 6: Remediation Demo - Secure Multi-Party Computation (SMPC)
# =============================================================================

def generate_additive_shares(secret, num_shares, modulus=2**31 - 1):
    """Splits a secret integer into additive shares mod a prime."""
    shares = [np.random.randint(0, modulus) for _ in range(num_shares - 1)]
    last_share = (secret - sum(shares)) % modulus
    shares.append(last_share)
    return shares


def reconstruct_from_shares(shares, modulus=2**31 - 1):
    """Reconstructs the secret from additive shares."""
    return sum(shares) % modulus


def demo_smpc_aggregation():
    """Demonstrates SMPC-based secure aggregation of model gradients."""
    print("=" * 70)
    print("SECTION 6: Remediation Demo - Secure Multi-Party Computation")
    print("=" * 70)

    np.random.seed(42)
    num_clients = 4
    client_names = ['Hospital', 'Clinic', 'Lab', 'Pharmacy']
    modulus = 2**31 - 1

    # Simulate scalar gradient values (quantized to integers)
    gradients = [np.random.randint(100, 1000) for _ in range(num_clients)]
    true_sum = sum(gradients)

    print("\n  Phase 1: Each client has a private gradient value")
    for name, g in zip(client_names, gradients):
        print(f"    {name:<10}: gradient = {g}")
    print(f"    True Sum (unknown to any single party): {true_sum}")

    # Each client generates shares and distributes
    print("\n  Phase 2: Secret sharing (each client splits its value)")
    all_shares = []
    for i, (name, g) in enumerate(zip(client_names, gradients)):
        shares = generate_additive_shares(g, num_clients, modulus)
        all_shares.append(shares)
        print(f"    {name:<10} shares: {shares}")

    # Each party sums the shares it received
    print("\n  Phase 3: Each party sums shares it received")
    party_sums = []
    for j in range(num_clients):
        partial = sum(all_shares[i][j] for i in range(num_clients)) % modulus
        party_sums.append(partial)
        print(f"    Party {j} partial sum: {partial}")

    # Reconstruct
    reconstructed = reconstruct_from_shares(party_sums, modulus)
    print(f"\n  Phase 4: Reconstruction")
    print(f"    Reconstructed Sum : {reconstructed}")
    print(f"    True Sum          : {true_sum}")
    print(f"    Match             : {'YES' if reconstructed == true_sum else 'NO'}")
    print(f"\n  Conclusion: The aggregator learns ONLY the sum, never individual values.\n")


# =============================================================================
# SECTION 7: Summary Report
# =============================================================================

def generate_summary_report(dread_results, risk_register):
    """Generates a final text-based summary report."""
    print("=" * 70)
    print("SECTION 7: Workshop Summary Report")
    print("=" * 70)

    report_lines = [
        "=" * 60,
        "THREAT MODEL REPORT - FEDERATED LEARNING SYSTEM",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 60,
        "",
        "1. SYSTEM OVERVIEW",
        "   Architecture: Federated Learning with central aggregation",
        "   Components: 4 client devices, 1 central aggregator, 1 model store",
        "   Data Flow: Local gradients -> Aggregator -> Global model",
        "",
        "2. THREAT ANALYSIS METHODOLOGY",
        "   - STRIDE for threat identification",
        "   - DREAD for risk scoring and prioritization",
        "",
        "3. THREATS IDENTIFIED: {}".format(len(STRIDE_THREATS)),
    ]
    for r in dread_results:
        report_lines.append(
            f"   [{r['id']}] {r['category']:<22} Score: {r['average']:.1f} "
            f"({r['risk_level']})")

    report_lines += [
        "",
        "4. PRIORITY REMEDIATION ACTIONS",
        "   a) Implement Differential Privacy (epsilon-DP) on gradient updates",
        "   b) Deploy Secure Multi-Party Computation for aggregation",
        "   c) Enforce mutual TLS and Byzantine-robust aggregation",
        "   d) Apply rate limiting and anomaly detection",
        "",
        "5. ARTIFACTS GENERATED",
        f"   - DFD Diagram       : {os.path.join(OUTPUT_DIR, 'fl_dfd_diagram.png')}",
        f"   - DREAD Heatmap     : {os.path.join(OUTPUT_DIR, 'dread_heatmap.png')}",
        f"   - Privacy Trade-off : {os.path.join(OUTPUT_DIR, 'dp_privacy_utility_tradeoff.png')}",
        f"   - Risk Register     : {os.path.join(OUTPUT_DIR, 'risk_register.json')}",
        f"   - Summary Report    : {os.path.join(OUTPUT_DIR, 'threat_model_report.txt')}",
        "",
        "=" * 60,
    ]

    report_text = "\n".join(report_lines)
    print(report_text)

    path = os.path.join(OUTPUT_DIR, 'threat_model_report.txt')
    with open(path, 'w') as f:
        f.write(report_text)
    print(f"\n  Report saved to: {path}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("\n" + "#" * 70)
    print("#  PRACTICAL WORKSHOP: Threat Model for Federated Learning System  #")
    print("#" * 70 + "\n")

    # 1. Draw DFD
    draw_dfd()

    # 2. STRIDE analysis
    display_stride_analysis()

    # 3. DREAD scoring
    dread_results = compute_dread_scores()
    plot_dread_heatmap(dread_results)

    # 4. Risk register
    risk_register = generate_risk_register(dread_results)

    # 5. DP remediation demo
    demo_differential_privacy()

    # 6. SMPC remediation demo
    demo_smpc_aggregation()

    # 7. Summary report
    generate_summary_report(dread_results, risk_register)

    print("\n" + "#" * 70)
    print("#  Workshop Complete - All artifacts saved to output directory.     #")
    print("#" * 70 + "\n")
