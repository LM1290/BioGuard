"""
Compare all models: baselines vs neural network.

This script loads results from baselines and neural network evaluation
and presents a comprehensive comparison with visualizations.
"""

import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

ARTIFACT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'artifacts')
BASELINE_RESULTS = os.path.join(ARTIFACT_DIR, 'baseline_results.json')
METADATA_PATH = os.path.join(ARTIFACT_DIR, 'metadata.json')
EVAL_RESULTS = os.path.join(ARTIFACT_DIR, 'eval_results.json')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


def load_results():
    """Load baseline and neural network results."""
    results = {}

    # Load baselines
    if os.path.exists(BASELINE_RESULTS):
        with open(BASELINE_RESULTS, 'r') as f:
            baselines = json.load(f)
            results.update(baselines)
    else:
        print(f"WARNING: Baseline results not found at {BASELINE_RESULTS}")
        print("Run: python -m bioguard.main baselines")
        return None

    # Load neural network evaluation results
    if os.path.exists(EVAL_RESULTS):
        with open(EVAL_RESULTS, 'r') as f:
            nn_results = json.load(f)
            results['neural_network'] = nn_results

    # Load metadata
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, 'r') as f:
            metadata = json.load(f)
            results['metadata'] = metadata

    return results


def print_comparison_table(results):
    """Print formatted comparison table."""

    print("\n" + "="*100)
    print("COMPREHENSIVE MODEL COMPARISON")
    print("="*100)

    print(f"\n{'Model':<30} {'ROC-AUC':<10} {'PR-AUC':<10} {'F1':<10} {'Precision':<10} {'Recall':<10} {'Specificity':<12}")
    print("-"*100)

    # Baselines
    baseline_names = ['tanimoto', 'logistic_regression', 'random_forest', 'gradient_boosting', 'svm']
    for key in baseline_names:
        if key in results:
            r = results[key]
            spec = r.get('specificity', 0.0)
            print(f"{r['name']:<30} "
                  f"{r['roc_auc']:<10.4f} "
                  f"{r['pr_auc']:<10.4f} "
                  f"{r['f1']:<10.4f} "
                  f"{r['precision']:<10.4f} "
                  f"{r['recall']:<10.4f} "
                  f"{spec:<12.4f}")

    print("-"*100)

    # Neural network
    if 'neural_network' in results:
        r = results['neural_network']
        spec = r.get('specificity', 0.0)
        print(f"{r['name']:<30} "
              f"{r['roc_auc']:<10.4f} "
              f"{r['pr_auc']:<10.4f} "
              f"{r['f1']:<10.4f} "
              f"{r['precision']:<10.4f} "
              f"{r['recall']:<10.4f} "
              f"{spec:<12.4f}")

        # Highlight improvement
        baseline_pr_aucs = {k: results[k]['pr_auc'] for k in baseline_names if k in results}
        if baseline_pr_aucs:
            best_baseline_pr = max(baseline_pr_aucs.values())
            nn_pr = r['pr_auc']
            improvement = nn_pr - best_baseline_pr

            if improvement > 0.10:
                status = "[OK] EXCELLENT IMPROVEMENT"
            elif improvement > 0.05:
                status = "[OK] STRONG IMPROVEMENT"
            elif improvement > 0.02:
                status = "[WARNING] MARGINAL IMPROVEMENT"
            elif improvement > 0:
                status = "[WARNING] SLIGHT IMPROVEMENT"
            else:
                status = "[ERROR] NO IMPROVEMENT - USE SIMPLER MODEL"

            print(f"\n{status}")
            print(f"  Absolute gain: {improvement:+.4f} PR-AUC")
            print(f"  Relative gain: {(improvement/best_baseline_pr*100):+.1f}%")

            if improvement <= 0:
                print(f"\n  [WARNING]  WARNING: Neural network does not outperform baselines!")
                print(f"  → Consider using {max(baseline_pr_aucs, key=baseline_pr_aucs.get)} instead")
                print(f"  → Neural network adds complexity without benefit")
    else:
        print(f"{'BioGuardNet (Neural Network)':<30} {'[Not run]':<10}")
        print(f"\nRun: python -m bioguard.main eval")

    print("="*100)


def create_comparison_visualizations(results):
    """Create visualizations comparing all models including neural network."""

    if 'neural_network' not in results:
        print("\nSkipping visualizations - neural network not evaluated yet")
        return

    print("\n" + "="*60)
    print("GENERATING COMPARISON VISUALIZATIONS")
    print("="*60)

    # Extract model data
    models = []
    metrics = {
        'ROC-AUC': [],
        'PR-AUC': [],
        'F1': [],
        'Precision': [],
        'Recall': [],
        'Specificity': []
    }

    baseline_names = ['tanimoto', 'logistic_regression', 'random_forest', 'gradient_boosting', 'svm']
    for key in baseline_names:
        if key in results:
            r = results[key]
            models.append(r['name'])
            metrics['ROC-AUC'].append(r['roc_auc'])
            metrics['PR-AUC'].append(r['pr_auc'])
            metrics['F1'].append(r['f1'])
            metrics['Precision'].append(r['precision'])
            metrics['Recall'].append(r['recall'])
            metrics['Specificity'].append(r.get('specificity', 0.0))

    # Add neural network
    if 'neural_network' in results:
        r = results['neural_network']
        models.append(r['name'])
        metrics['ROC-AUC'].append(r['roc_auc'])
        metrics['PR-AUC'].append(r['pr_auc'])
        metrics['F1'].append(r['f1'])
        metrics['Precision'].append(r['precision'])
        metrics['Recall'].append(r['recall'])
        metrics['Specificity'].append(r.get('specificity', 0.0))

    # 1. Primary Metrics Comparison with Neural Network Highlight
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # ROC-AUC
    colors = ['steelblue'] * (len(models) - 1) + ['darkred']  # Highlight NN in red
    bars1 = ax1.bar(models, metrics['ROC-AUC'], color=colors, alpha=0.7)
    ax1.set_ylabel('ROC-AUC', fontsize=12, fontweight='bold')
    ax1.set_title('ROC-AUC Comparison (Including Neural Network)', fontsize=13, fontweight='bold')
    ax1.set_ylim([0, 1.05])
    ax1.grid(True, alpha=0.3, axis='y')
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)

    # PR-AUC
    colors = ['coral'] * (len(models) - 1) + ['darkgreen']  # Highlight NN in green
    bars2 = ax2.bar(models, metrics['PR-AUC'], color=colors, alpha=0.7)
    ax2.set_ylabel('PR-AUC', fontsize=12, fontweight='bold')
    ax2.set_title('PR-AUC Comparison (Primary Metric)', fontsize=13, fontweight='bold')
    ax2.set_ylim([0, 1.05])
    ax2.grid(True, alpha=0.3, axis='y')
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Add value labels
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(ARTIFACT_DIR, 'final_comparison_primary_metrics.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("[OK] Saved: final_comparison_primary_metrics.png")

    # 2. All Metrics Grouped Bar Chart
    fig, ax = plt.subplots(figsize=(16, 8))
    x = np.arange(len(models))
    width = 0.12

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    for i, (metric, values) in enumerate(metrics.items()):
        offset = width * (i - len(metrics)/2 + 0.5)
        ax.bar(x + offset, values, width, label=metric, color=colors[i % len(colors)])

    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Complete Model Comparison - All Metrics', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend(loc='lower right')
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(ARTIFACT_DIR, 'final_comparison_all_metrics.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("[OK] Saved: final_comparison_all_metrics.png")

    # 3. Radar Chart
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='polar')

    categories = ['ROC-AUC', 'PR-AUC', 'F1', 'Precision', 'Recall', 'Specificity']
    N = len(categories)

    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=11)

    # Plot baselines
    for key in baseline_names:
        if key in results:
            r = results[key]
            values = [r['roc_auc'], r['pr_auc'], r['f1'],
                     r['precision'], r['recall'], r.get('specificity', 0.0)]
            values += values[:1]
            ax.plot(angles, values, 'o-', linewidth=1.5, label=r['name'], alpha=0.6)
            ax.fill(angles, values, alpha=0.1)

    # Plot neural network (highlighted)
    if 'neural_network' in results:
        r = results['neural_network']
        values = [r['roc_auc'], r['pr_auc'], r['f1'],
                 r['precision'], r['recall'], r.get('specificity', 0.0)]
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=3, label=r['name'], color='red', alpha=0.9)
        ax.fill(angles, values, alpha=0.2, color='red')

    ax.set_ylim(0, 1)
    ax.set_title('Comprehensive Model Comparison\n(Neural Network vs Baselines)',
                size=14, fontweight='bold', y=1.08)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(ARTIFACT_DIR, 'final_comparison_radar.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("[OK] Saved: final_comparison_radar.png")

    # 4. Performance Gain Analysis
    if len(models) > 1:
        fig, ax = plt.subplots(figsize=(12, 6))

        # Calculate gains relative to best baseline
        baseline_pr_aucs = [metrics['PR-AUC'][i] for i in range(len(models)-1)]
        best_baseline_pr = max(baseline_pr_aucs)
        nn_pr = metrics['PR-AUC'][-1]

        gains = [(pr - best_baseline_pr) * 100 for pr in metrics['PR-AUC']]

        colors_gain = ['gray'] * (len(models) - 1) + ['green' if gains[-1] > 0 else 'red']
        bars = ax.bar(models, gains, color=colors_gain, alpha=0.7)

        ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax.axhline(y=2, color='orange', linestyle='--', linewidth=1, alpha=0.5, label='Marginal (+2%)')
        ax.axhline(y=5, color='green', linestyle='--', linewidth=1, alpha=0.5, label='Strong (+5%)')
        ax.axhline(y=10, color='darkgreen', linestyle='--', linewidth=1, alpha=0.5, label='Excellent (+10%)')

        ax.set_ylabel('PR-AUC Gain (%)', fontsize=12, fontweight='bold')
        ax.set_title(f'Performance Gain Analysis\n(Relative to Best Baseline: {best_baseline_pr:.4f})',
                    fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        ax.legend()

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:+.2f}%', ha='center', va='bottom' if height > 0 else 'top', fontsize=9)

        plt.tight_layout()
        plt.savefig(os.path.join(ARTIFACT_DIR, 'final_performance_gain.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("[OK] Saved: final_performance_gain.png")

    print("="*60)


def print_detailed_analysis(results):
    """Print detailed analysis and insights."""

    print("\n" + "="*80)
    print("DETAILED STATISTICAL ANALYSIS")
    print("="*80)

    baseline_names = ['tanimoto', 'logistic_regression', 'random_forest', 'gradient_boosting', 'svm']

    if not all(k in results for k in baseline_names[:3]):
        print("Missing baseline results. Run baselines first.")
        return

    tan = results.get('tanimoto')
    log = results.get('logistic_regression')
    rf = results.get('random_forest')
    gb = results.get('gradient_boosting')
    svm = results.get('svm')
    nn = results.get('neural_network')

    print("\n1. STRUCTURE-ONLY PERFORMANCE (Tanimoto Baseline)")
    print("-"*80)
    print(f"   PR-AUC: {tan['pr_auc']:.4f}")
    print(f"   This represents chemical similarity alone - no machine learning.")

    print("\n2. LINEAR MODEL (Logistic Regression)")
    print("-"*80)
    print(f"   PR-AUC: {log['pr_auc']:.4f}")
    print(f"   Gain over Tanimoto: {(log['pr_auc'] - tan['pr_auc']):.4f} "
          f"({((log['pr_auc'] - tan['pr_auc'])/tan['pr_auc']*100):.1f}%)")
    print(f"   → Enzyme features add predictive signal")

    print("\n3. NON-LINEAR MODELS (Tree Ensembles)")
    print("-"*80)
    print(f"   Random Forest:     PR-AUC = {rf['pr_auc']:.4f}")
    if gb:
        print(f"   Gradient Boosting: PR-AUC = {gb['pr_auc']:.4f}")
    best_tree_pr = max(rf['pr_auc'], gb['pr_auc'] if gb else 0)
    print(f"   Best tree improvement: {(best_tree_pr - log['pr_auc']):.4f} "
          f"({((best_tree_pr - log['pr_auc'])/log['pr_auc']*100):.1f}%)")
    print(f"   → Non-linearity captures complex interactions")

    if svm:
        print("\n4. KERNEL METHOD (SVM)")
        print("-"*80)
        print(f"   PR-AUC: {svm['pr_auc']:.4f}")

    if nn:
        print("\n5. DEEP LEARNING (Neural Network)")
        print("-"*80)
        print(f"   PR-AUC: {nn['pr_auc']:.4f}")
        print(f"   Gain over best baseline: {(nn['pr_auc'] - best_tree_pr):.4f} "
              f"({((nn['pr_auc'] - best_tree_pr)/best_tree_pr*100):.1f}%)")

        if nn['pr_auc'] > best_tree_pr + 0.10:
            print(f"   [OK] EXCELLENT: Neural network provides substantial improvement!")
            print(f"   → Deep learning complexity is justified")
        elif nn['pr_auc'] > best_tree_pr + 0.05:
            print(f"   [OK] GOOD: Neural network provides meaningful improvement")
            print(f"   → Consider deployment depending on infrastructure costs")
        elif nn['pr_auc'] > best_tree_pr + 0.02:
            print(f"   ~ MARGINAL: Neural network slightly better")
            print(f"   → Consider simpler models for production")
        else:
            print(f"   ✗ POOR: Neural network does not justify complexity")
            print(f"   → Use tree ensemble instead (simpler, faster, more interpretable)")

    print("\n" + "="*80)
    print("RECOMMENDATION")
    print("="*80)

    if nn:
        if nn['pr_auc'] > best_tree_pr + 0.05:
            print("\n[OK] RECOMMENDATION: Deploy Neural Network")
            print("   • Significant performance advantage")
            print("   • Justifies additional complexity")
            print(f"   • Expected PR-AUC in production: ~{nn['pr_auc']:.4f}")
        elif nn['pr_auc'] > best_tree_pr + 0.02:
            print("\n[WARNING] RECOMMENDATION: Consider Both Options")
            print("   • Neural network marginally better")
            print("   • Tree ensemble simpler and faster")
            print("   • Decision depends on:")
            print("     - Available infrastructure")
            print("     - Latency requirements")
            print("     - Interpretability needs")
        else:
            print("\n[ERROR] RECOMMENDATION: Use Tree Ensemble")
            print("   • Simpler model performs as well or better")
            print("   • Faster inference")
            print("   • More interpretable")
            print("   • Easier to maintain")
            print(f"   • Recommended model: {('Random Forest' if rf['pr_auc'] > gb['pr_auc'] else 'Gradient Boosting') if gb else 'Random Forest'}")
    else:
        print("\nNeural network not evaluated yet.")
        print("Run: python -m bioguard.main eval")

    print("="*80)


def main():
    """Main comparison function."""

    results = load_results()

    if results is None:
        print("\nERROR: Cannot proceed without baseline results.")
        print("Run: python -m bioguard.main baselines")
        sys.exit(1)

    print_comparison_table(results)
    print_detailed_analysis(results)
    create_comparison_visualizations(results)

    # Neural network status
    print("\n" + "="*80)
    print("FILES GENERATED")
    print("="*80)
    print("\nVisualization files:")
    print("  • final_comparison_primary_metrics.png")
    print("  • final_comparison_all_metrics.png")
    print("  • final_comparison_radar.png")
    print("  • final_performance_gain.png")
    print("\nData files:")
    print("  • baseline_results.json")
    if 'neural_network' in results:
        print("  • eval_results.json")
    print("  • metadata.json")
    print("="*80)


if __name__ == "__main__":
    main()
