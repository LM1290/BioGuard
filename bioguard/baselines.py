"""
Baseline models for DDI prediction to demonstrate neural network improvement.

Baselines:
1. Tanimoto Similarity (structure-only)
2. Logistic Regression (structure features only)
3. Random Forest (structure features only)

All baselines use PAIR-DISJOINT split for fair comparison with neural network.
"""

import numpy as np
import pandas as pd
import os
import json
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score
)
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from tqdm import tqdm
import warnings

from .data_loader import load_twosides_data
from .train import get_pair_disjoint_split
from .featurizer import BioFeaturizer

ARTIFACT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'artifacts')
BASELINE_RESULTS = os.path.join(ARTIFACT_DIR, 'baseline_results.json')


def compute_tanimoto_similarity(smiles_a, smiles_b):
    """
    Compute Tanimoto similarity between two SMILES strings.
    Returns similarity in [0, 1] or None if invalid.
    """
    try:
        mol_a = Chem.MolFromSmiles(smiles_a)
        mol_b = Chem.MolFromSmiles(smiles_b)
        
        if mol_a is None or mol_b is None:
            return None
        
        fp_a = AllChem.GetMorganFingerprintAsBitVect(mol_a, 2, nBits=1024)
        fp_b = AllChem.GetMorganFingerprintAsBitVect(mol_b, 2, nBits=1024)
        
        return DataStructs.TanimotoSimilarity(fp_a, fp_b)
    except:
        return None


def evaluate_tanimoto_baseline(test_df):
    """
    Baseline 1: Tanimoto Similarity
    
    Hypothesis: High structural similarity → DDI likely
    This is the simplest baseline using only chemical structure.
    """
    print("\n" + "="*60)
    print("BASELINE 1: TANIMOTO SIMILARITY")
    print("="*60)
    print("Computing pairwise Tanimoto similarities...")
    
    similarities = []
    valid_labels = []
    
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Computing similarities"):
        sim = compute_tanimoto_similarity(row['smiles_a'], row['smiles_b'])
        if sim is not None:
            similarities.append(sim)
            valid_labels.append(row['label'])
    
    similarities = np.array(similarities)
    valid_labels = np.array(valid_labels)
    
    print(f"Valid pairs: {len(similarities)}/{len(test_df)}")
    
    # Tanimoto similarity is the "score" (higher = more likely to interact)
    roc_auc = roc_auc_score(valid_labels, similarities)
    pr_auc = average_precision_score(valid_labels, similarities)
    
    # Find optimal threshold on this test set (for fair comparison)
    thresholds = np.linspace(0, 1, 100)
    best_f1 = 0
    best_threshold = 0.5
    
    for thresh in thresholds:
        preds = (similarities >= thresh).astype(int)
        f1 = f1_score(valid_labels, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh
    
    preds = (similarities >= best_threshold).astype(int)
    acc = accuracy_score(valid_labels, preds)
    precision = precision_score(valid_labels, preds, zero_division=0)
    recall = recall_score(valid_labels, preds, zero_division=0)
    
    results = {
        "name": "Tanimoto Similarity",
        "description": "Structure-only baseline using Morgan fingerprint similarity",
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "threshold": float(best_threshold),
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(best_f1),
        "n_test": len(similarities)
    }
    
    print(f"\nResults:")
    print(f"  ROC-AUC   : {roc_auc:.4f}")
    print(f"  PR-AUC    : {pr_auc:.4f}")
    print(f"  Best F1   : {best_f1:.4f} (threshold={best_threshold:.3f})")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    
    return results


def prepare_features(df):
    """
    Prepare feature matrix for sklearn models.
    Uses symmetric pair representation: [feat_a + feat_b, |feat_a - feat_b|, feat_a * feat_b]
    """
    print("Preparing features...")
    featurizer = BioFeaturizer()
    
    X = []
    y = []
    valid_indices = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Featurizing"):
        try:
            vec = featurizer.featurize_pair(
                row['smiles_a'], row['smiles_b'],
                row['drug_a'], row['drug_b']
            )
            X.append(vec)
            y.append(row['label'])
            valid_indices.append(idx)
        except Exception as e:
            # Skip invalid pairs
            continue
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"Valid pairs: {len(X)}/{len(df)}")
    print(f"Feature dimension: {X.shape[1]}")
    print(f"Positive rate: {y.mean():.2%}")
    
    return X, y, valid_indices


def evaluate_logistic_regression(train_df, test_df):
    """
    Baseline 2: Logistic Regression
    
    Uses structure features with linear model.
    Tests whether a simple linear combination is sufficient.
    """
    print("\n" + "="*60)
    print("BASELINE 2: LOGISTIC REGRESSION")
    print("="*60)
    
    # Prepare features
    X_train, y_train, _ = prepare_features(train_df)
    X_test, y_test, _ = prepare_features(test_df)
    
    # Train with class balancing
    print("\nTraining Logistic Regression...")
    model = LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        random_state=42,
        n_jobs=1
    )
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X_train, y_train)
    
    # Predict probabilities
    y_probs = model.predict_proba(X_test)[:, 1]
    
    # Metrics
    roc_auc = roc_auc_score(y_test, y_probs)
    pr_auc = average_precision_score(y_test, y_probs)
    
    # Find optimal threshold
    thresholds = np.linspace(0, 1, 100)
    best_f1 = 0
    best_threshold = 0.5
    
    for thresh in thresholds:
        preds = (y_probs >= thresh).astype(int)
        f1 = f1_score(y_test, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh
    
    preds = (y_probs >= best_threshold).astype(int)
    acc = accuracy_score(y_test, preds)
    precision = precision_score(y_test, preds, zero_division=0)
    recall = recall_score(y_test, preds, zero_division=0)
    
    results = {
        "name": "Logistic Regression",
        "description": "Linear model with structure features",
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "threshold": float(best_threshold),
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(best_f1),
        "n_train": len(X_train),
        "n_test": len(X_test)
    }
    
    print(f"\nResults:")
    print(f"  ROC-AUC   : {roc_auc:.4f}")
    print(f"  PR-AUC    : {pr_auc:.4f}")
    print(f"  Best F1   : {best_f1:.4f} (threshold={best_threshold:.3f})")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    
    return results


def evaluate_random_forest(train_df, test_df):
    """
    Baseline 3: Random Forest
    
    Non-linear tree-based ensemble with structure features.
    Tests whether non-linearity alone (without deep learning) is sufficient.
    """
    print("\n" + "="*60)
    print("BASELINE 3: RANDOM FOREST")
    print("="*60)
    
    # Prepare features
    X_train, y_train, _ = prepare_features(train_df)
    X_test, y_test, _ = prepare_features(test_df)
    
    # Train
    print("\nTraining Random Forest...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight='balanced',
        random_state=42,
        n_jobs=2,
        verbose=1
    )
    
    model.fit(X_train, y_train)
    
    # Predict probabilities
    y_probs = model.predict_proba(X_test)[:, 1]
    
    # Metrics
    roc_auc = roc_auc_score(y_test, y_probs)
    pr_auc = average_precision_score(y_test, y_probs)
    
    # Find optimal threshold
    thresholds = np.linspace(0, 1, 100)
    best_f1 = 0
    best_threshold = 0.5
    
    for thresh in thresholds:
        preds = (y_probs >= thresh).astype(int)
        f1 = f1_score(y_test, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh
    
    preds = (y_probs >= best_threshold).astype(int)
    acc = accuracy_score(y_test, preds)
    precision = precision_score(y_test, preds, zero_division=0)
    recall = recall_score(y_test, preds, zero_division=0)
    
    results = {
        "name": "Random Forest",
        "description": "Non-linear ensemble with structure features",
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "threshold": float(best_threshold),
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(best_f1),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "feature_importances": model.feature_importances_.tolist()[:50]
    }
    
    print(f"\nResults:")
    print(f"  ROC-AUC   : {roc_auc:.4f}")
    print(f"  PR-AUC    : {pr_auc:.4f}")
    print(f"  Best F1   : {best_f1:.4f} (threshold={best_threshold:.3f})")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    
    return results


def run_all_baselines(split_type='pair_disjoint'):
    """
    Run all baseline evaluations and save results.
    
    Args:
        split_type: Only 'pair_disjoint' is supported in production
    """
    if split_type != 'pair_disjoint':
        raise ValueError(f"Only 'pair_disjoint' split is supported in production. Got: {split_type}")
    
    print("="*60)
    print(f"BIOGUARD BASELINE EVALUATION (PAIR-DISJOINT)")
    print("="*60)
    print("Testing: Can we predict NEW combinations of KNOWN drugs?")
    print("="*60)
    
    # Load data with specified split
    print("\nLoading data...")
    df = load_twosides_data()
    train_df, val_df, test_df = get_pair_disjoint_split(df, seed=42)
    
    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_df)}")
    print(f"  Val:   {len(val_df)}")
    print(f"  Test:  {len(test_df)}")
    
    # Run baselines
    all_results = {}
    
    # 1. Tanimoto (structure-only)
    tanimoto_results = evaluate_tanimoto_baseline(test_df)
    all_results['tanimoto'] = tanimoto_results
    
    # 2. Logistic Regression (structure, linear)
    logreg_results = evaluate_logistic_regression(train_df, test_df)
    all_results['logistic_regression'] = logreg_results
    
    # 3. Random Forest (structure, non-linear)
    rf_results = evaluate_random_forest(train_df, test_df)
    all_results['random_forest'] = rf_results
    
    # Save results
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    with open(BASELINE_RESULTS, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n[OK] Baseline results saved to {BASELINE_RESULTS}")
    
    # Summary comparison
    print("\n" + "="*60)
    print("BASELINE COMPARISON")
    print("="*60)
    print(f"{'Model':<25} {'ROC-AUC':<10} {'PR-AUC':<10} {'F1':<10}")
    print("-"*60)
    
    for key, results in all_results.items():
        print(f"{results['name']:<25} "
              f"{results['roc_auc']:<10.4f} "
              f"{results['pr_auc']:<10.4f} "
              f"{results['f1']:<10.4f}")
    
    print("-"*60)
    
    # Analysis
    print("\n" + "="*60)
    print("ANALYSIS")
    print("="*60)
    
    # Compare PR-AUC (primary metric)
    pr_aucs = {name: res['pr_auc'] for name, res in all_results.items()}
    best_baseline = max(pr_aucs, key=pr_aucs.get)
    
    print(f"\nBest baseline: {all_results[best_baseline]['name']}")
    print(f"  PR-AUC: {pr_aucs[best_baseline]:.4f}")
    
    print("\nKey insights:")
    print(f"  1. Tanimoto (structure-only): PR-AUC = {tanimoto_results['pr_auc']:.4f}")
    print(f"     - Simple similarity metric")
    print(f"     - No mechanism knowledge")
    
    print(f"\n  2. Logistic Regression: PR-AUC = {logreg_results['pr_auc']:.4f}")
    print(f"     - Linear combination of features")
    gain_log = logreg_results['pr_auc'] - tanimoto_results['pr_auc']
    print(f"     - Improvement over Tanimoto: {gain_log:+.4f} ({gain_log/tanimoto_results['pr_auc']*100:+.1f}%)")
    
    print(f"\n  3. Random Forest: PR-AUC = {rf_results['pr_auc']:.4f}")
    print(f"     - Non-linear interactions")
    gain_rf = rf_results['pr_auc'] - logreg_results['pr_auc']
    print(f"     - Improvement over LogReg: {gain_rf:+.4f} ({gain_rf/logreg_results['pr_auc']*100:+.1f}%)")
    
    print("\n  → Neural network needed if it achieves PR-AUC > {:.4f}".format(pr_aucs[best_baseline]))
    print("  → Run evaluation to compare: python -m bioguard.main eval")
    
    print("\n" + "="*60)
    print("NEXT STEPS")
    print("="*60)
    print("1. Evaluate neural network: python -m bioguard.main eval")
    print("2. Compare all models: python -m bioguard.main compare")
    print("="*60)
    
    return all_results


if __name__ == "__main__":
    run_all_baselines()
