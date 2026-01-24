"""
Compare all models: baselines vs neural network.
UPDATED: v3.0 (The "Board-Ready" Edition)
- Restored & Optimized Visualizations (Radar, Bar, Gain Charts)
- Robust Metadata handling for Series B splits (Cold/Scaffold)
- Automatic "Verdict" generation based on performance deltas
"""

import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg') # Force headless mode for server compatibility
import matplotlib.pyplot as plt
import seaborn as sns
from math import pi

# Configurations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10
ARTIFACT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'artifacts')
BASELINE_PATH = os.path.join(ARTIFACT_DIR, 'baseline_results.json')
EVAL_PATH = os.path.join(ARTIFACT_DIR, 'eval_results.json')
META_PATH = os.path.join(ARTIFACT_DIR, 'metadata.json')

def load_data():
    """Load all result files with robust error handling."""
    data = {'baselines': {}, 'gnn': {}, 'meta': {}}

    # 1. Load Metadata
    if os.path.exists(META_PATH):
        with open(META_PATH, 'r') as f:
            data['meta'] = json.load(f)

    # 2. Load Baselines
    if os.path.exists(BASELINE_PATH):
        with open(BASELINE_PATH, 'r') as f:
            data['baselines'] = json.load(f)

    # 3. Load GNN Evaluation
    if os.path.exists(EVAL_PATH):
        with open(EVAL_PATH, 'r') as f:
            data['gnn'] = json.load(f)
            # Normalize keys to match baselines if needed
            if 'name' not in data['gnn']: data['gnn']['name'] = 'BioGuardGAT (GNN)'

    return data

def get_best_baseline(baselines):
    """Find the best performing baseline (by PR-AUC)."""
    if not baselines: return None, 0.0
    best_name = max(baselines, key=lambda k: baselines[k].get('pr_auc', 0))
    return baselines[best_name], baselines[best_name].get('pr_auc', 0)

def plot_radar_chart(baselines, gnn):
    """Generate the 'Board Meeting' Radar Chart."""
    print("Generating Radar Chart...")

    # Metrics to display
    labels = ['ROC-AUC', 'PR-AUC', 'F1', 'Accuracy']
    keys = ['roc_auc', 'pr_auc', 'f1', 'accuracy']
    N = len(labels)

    # Compute angles
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1] # Close the loop

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)

    plt.xticks(angles[:-1], labels)
    ax.set_rlabel_position(0)
    plt.yticks([0.2, 0.4, 0.6, 0.8], ["0.2", "0.4", "0.6", "0.8"], color="grey", size=7)
    plt.ylim(0, 1)

    # Plot Baselines (Lightly)
    for name, metrics in baselines.items():
        if name == 'metadata': continue
        values = [metrics.get(k, 0) for k in keys]
        values += values[:1]
        ax.plot(angles, values, linewidth=1, linestyle='solid', label=metrics['name'], alpha=0.4)
        ax.fill(angles, values, alpha=0.05)

    # Plot GNN (Bold)
    if gnn:
        values = [gnn.get(k, 0) for k in keys]
        values += values[:1]
        ax.plot(angles, values, linewidth=2, linestyle='solid', color='red', label=gnn['name'])
        ax.fill(angles, values, color='red', alpha=0.15)

    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title("Model Capabilities Profile", y=1.08)

    out_path = os.path.join(ARTIFACT_DIR, 'comparison_radar.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_bar_comparison(baselines, gnn):
    """Generate PR-AUC Bar Comparison."""
    print("Generating Bar Comparison...")

    models = []
    scores = []
    colors = []

    # Add Baselines
    for name, metrics in baselines.items():
        if name == 'metadata': continue
        models.append(metrics.get('name', name))
        scores.append(metrics.get('pr_auc', 0))
        colors.append('grey')

    # Add GNN
    if gnn:
        models.append(gnn.get('name', 'GNN'))
        scores.append(gnn.get('pr_auc', 0))
        colors.append('#d62728') # Standard red

    y_pos = np.arange(len(models))

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(y_pos, scores, align='center', color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(models)
    ax.invert_yaxis()
    ax.set_xlabel('PR-AUC (Precision-Recall Area Under Curve)')
    ax.set_title('Primary Metric Comparison (PR-AUC)')
    ax.set_xlim(0, 1.0)

    # Add text labels
    for i, v in enumerate(scores):
        ax.text(v + 0.01, i, f"{v:.3f}", color='black', va='center', fontweight='bold')

    out_path = os.path.join(ARTIFACT_DIR, 'comparison_bars.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

def print_executive_summary(baselines, gnn, meta):
    """Print the text investors care about."""
    print("\n" + "="*80)
    print(f"BIOGUARD EXECUTIVE SUMMARY (Split: {meta.get('split_type', 'UNKNOWN').upper()})")
    print("="*80)

    # 1. Identify Leader
    best_base_metric, best_base_score = get_best_baseline(baselines)
    gnn_score = gnn.get('pr_auc', 0) if gnn else 0

    print(f"{'Model':<30} | {'PR-AUC':<8} | {'ROC-AUC':<8} | {'Status'}")
    print("-" * 80)

    # Print Baselines
    for name, m in baselines.items():
        print(f"{m['name']:<30} | {m.get('pr_auc',0):.4f}   | {m.get('roc_auc',0):.4f}   | Baseline")

    # Print GNN
    if gnn:
        print("-" * 80)
        print(f"{gnn['name']:<30} | {gnn_score:.4f}   | {gnn.get('roc_auc',0):.4f}   | CHALLENGER")

    print("="*80)

    # 2. The Verdict
    if not gnn:
        print("Status: GNN not evaluated yet.")
        return

    delta = gnn_score - best_base_score
    pct_gain = (delta / best_base_score * 100) if best_base_score > 0 else 0

    print(f"\nPERFORMANCE DELTA: {delta:+.4f} PR-AUC ({pct_gain:+.1f}%)")

    if delta > 0.10:
        print("VERDICT: [GREEN LIGHT] Transformative improvement. Proceed to production.")
    elif delta > 0.03:
        print("VERDICT: [YELLOW LIGHT] Solid improvement. Justifies compute cost.")
    elif delta > 0:
        print("VERDICT: [ORANGE LIGHT] Marginal gain. Optimize hyperparameters or data.")
    else:
        print("VERDICT: [RED LIGHT] Failed to beat baseline. Do not deploy.")
        print("   -> Check for: Overfitting, Data Leakage in Baselines, or Featurization bugs.")

def main():
    data = load_data()

    if not data['baselines'] and not data['gnn']:
        print("No results found in artifacts/. Run 'train', 'baselines', and 'eval' first.")
        return

    # Text Report
    print_executive_summary(data['baselines'], data['gnn'], data['meta'])

    # Visualizations
    if data['baselines'] or data['gnn']:
        plot_bar_comparison(data['baselines'], data['gnn'])
        plot_radar_chart(data['baselines'], data['gnn'])
        print(f"\n[Artifacts Generated]")
        print(f"- {os.path.join(ARTIFACT_DIR, 'comparison_radar.png')}")
        print(f"- {os.path.join(ARTIFACT_DIR, 'comparison_bars.png')}")

if __name__ == "__main__":
    main()