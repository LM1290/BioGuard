"""
Baseline models for DDI prediction.

Baselines:
1. Tanimoto Similarity (structure-only)
2. Logistic Regression (structure features only)
3. Random Forest (structure features only)

All baselines use PAIR-DISJOINT split for fair comparison with the GNN.
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
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit import RDLogger
from tqdm import tqdm
import warnings

from .data_loader import load_twosides_data
from .train import get_pair_disjoint_split
# CRITICAL: Use BioFeaturizer (Fingerprints) for baselines, NOT GraphFeaturizer
from .featurizer import BioFeaturizer

RDLogger.DisableLog('rdApp.*')

ARTIFACT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'artifacts')
BASELINE_RESULTS = os.path.join(ARTIFACT_DIR, 'baseline_results.json')


def compute_tanimoto_similarity(smiles_a, smiles_b):
    try:
        mol_a = Chem.MolFromSmiles(smiles_a)
        mol_b = Chem.MolFromSmiles(smiles_b)

        if mol_a is None or mol_b is None:
            return None

        # Standardize
        te = rdMolStandardize.TautomerEnumerator()
        te.SetMaxTautomers(20)
        te.SetMaxTransforms(20)

        def clean(m):
            try:
                m = rdMolStandardize.ChargeParent(m)
                try:
                    m = te.Canonicalize(m)
                except:
                    pass
                return m
            except:
                return m

        mol_a = clean(mol_a)
        mol_b = clean(mol_b)

        # Use 2048 bits to match BioFeaturizer defaults
        fp_a = AllChem.GetMorganFingerprintAsBitVect(mol_a, 2, nBits=2048)
        fp_b = AllChem.GetMorganFingerprintAsBitVect(mol_b, 2, nBits=2048)

        return DataStructs.TanimotoSimilarity(fp_a, fp_b)
    except:
        return None


def evaluate_tanimoto_baseline(test_df):
    """
    Baseline 1: Tanimoto Similarity
    Hypothesis: High structural similarity → DDI likely
    """
    print("\n" + "="*60)
    print("BASELINE 1: TANIMOTO SIMILARITY")
    print("="*60)
    print("Computing pairwise Tanimoto similarities...")

    similarities = []
    valid_labels = []

    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Computing similarities"):
        sim = compute_tanimoto_similarity(row['smiles_a'], row['smiles_b'])
        if sim is not None:
            similarities.append(sim)
            valid_labels.append(row['label'])

    similarities = np.array(similarities)
    valid_labels = np.array(valid_labels)

    print(f"Valid pairs: {len(similarities)}/{len(test_df)}")

    roc_auc = roc_auc_score(valid_labels, similarities)
    pr_auc = average_precision_score(valid_labels, similarities)

    # Find optimal threshold
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

    return results


def prepare_features(df):
    """
    Prepare feature matrix for sklearn models.
    Uses BioFeaturizer to create flat vectors (fingerprints + phys props).
    """
    print("Preparing features...")
    # CRITICAL: Use the legacy featurizer for baselines
    featurizer = BioFeaturizer()

    X = []
    y = []
    valid_indices = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Featurizing"):
        try:
            # Only pass SMILES. IDs are not needed by the featurizer.
            vec = featurizer.featurize_pair(
                row['smiles_a'], row['smiles_b']
            )
            X.append(vec)
            y.append(row['label'])
            valid_indices.append(idx)
        except Exception as e:
            continue

    X = np.array(X)
    y = np.array(y)

    print(f"Valid pairs: {len(X)}/{len(df)}")
    print(f"Feature dimension: {X.shape[1] if len(X) > 0 else 0}")
    print(f"Positive rate: {y.mean():.2%}")

    return X, y, valid_indices


def evaluate_logistic_regression(train_df, test_df):
    """
    Baseline 2: Logistic Regression
    """
    print("\n" + "="*60)
    print("BASELINE 2: LOGISTIC REGRESSION")
    print("="*60)

    X_train, y_train, _ = prepare_features(train_df)
    X_test, y_test, _ = prepare_features(test_df)

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

    y_probs = model.predict_proba(X_test)[:, 1]

    roc_auc = roc_auc_score(y_test, y_probs)
    pr_auc = average_precision_score(y_test, y_probs)

    # Thresholding logic
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

    return results


def evaluate_random_forest(train_df, test_df):
    """
    Baseline 3: Random Forest
    """
    print("\n" + "="*60)
    print("BASELINE 3: RANDOM FOREST")
    print("="*60)

    X_train, y_train, _ = prepare_features(train_df)
    X_test, y_test, _ = prepare_features(test_df)

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

    y_probs = model.predict_proba(X_test)[:, 1]

    roc_auc = roc_auc_score(y_test, y_probs)
    pr_auc = average_precision_score(y_test, y_probs)

    # Thresholding
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

    return results


def run_all_baselines(split_type='pair_disjoint'):
    """
    Run all baseline evaluations and save results.
    """
    if split_type != 'pair_disjoint':
        raise ValueError(f"Only 'pair_disjoint' split is supported. Got: {split_type}")

    print("="*60)
    print(f"BIOGUARD BASELINE EVALUATION (PAIR-DISJOINT)")
    print("="*60)

    print("\nLoading data...")
    df = load_twosides_data()
    train_df, val_df, test_df = get_pair_disjoint_split(df, seed=42)

    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_df)}")
    print(f"  Val:   {len(val_df)}")
    print(f"  Test:  {len(test_df)}")

    all_results = {}

    # 1. Tanimoto
    all_results['tanimoto'] = evaluate_tanimoto_baseline(test_df)

    # 2. Logistic Regression
    all_results['logistic_regression'] = evaluate_logistic_regression(train_df, test_df)

    # 3. Random Forest
    all_results['random_forest'] = evaluate_random_forest(train_df, test_df)

    # Save
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    with open(BASELINE_RESULTS, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n[OK] Baseline results saved to {BASELINE_RESULTS}")

    # Comparison
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

    return all_results


if __name__ == "__main__":
    run_all_baselines()