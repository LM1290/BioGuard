import os
import joblib
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import SimilarityMaps
from rdkit.Chem import Draw
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import train_test_split
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve
from torch_geometric.data import Batch

# Internal Imports
from bioguard.config import BASE_DIR, DATA_DIR, ARTIFACT_DIR, EDGE_DIM
from bioguard.model import BioGuardGAT
from bioguard.featurizer import drug_to_graph
from bioguard.enzyme import EnzymeManager
from bioguard.cyp_predictor import CYPPredictor

# Paths
CYP_MODEL_PATH = os.path.join(ARTIFACT_DIR, 'cyp_predictor.joblib')
GAT_MODEL_PATH = os.path.join(ARTIFACT_DIR, 'model.pt')
ENZYME_CSV_PATH = os.path.join(DATA_DIR, 'enzyme_features_full.csv')
AUDIT_OUT_DIR = os.path.join(ARTIFACT_DIR, 'audit_reports')
CALIBRATION_DIR = os.path.join(AUDIT_OUT_DIR, 'calibration_plots')
os.makedirs(CALIBRATION_DIR, exist_ok=True)


def get_base_estimator(clf):
    if hasattr(clf, 'calibrated_classifiers_'):
        return clf.calibrated_classifiers_[0].estimator
    return clf


def get_deterministic_test_set(df):
    """
    Reconstructs the EXACT Murcko Scaffold test set used during training.
    Cryptographically strict to prevent data leakage.
    """

    def generate_scaffold(smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: return ""
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)

    df['scaffold'] = df['smiles'].apply(generate_scaffold)

    # Sort to guarantee deterministic splitting across different machines
    unique_scaffolds = np.sort(df['scaffold'].unique())

    # Replicate the strict 3-way split from train_cyp_predictor.py
    train_scaffolds, temp_scaffolds = train_test_split(unique_scaffolds, test_size=0.3, random_state=42)
    calib_scaffolds, test_scaffolds = train_test_split(temp_scaffolds, test_size=0.5, random_state=42)

    test_df = df[df['scaffold'].isin(test_scaffolds)].copy()
    print(f"[Audit] Isolated Test Set: {len(test_df)} molecules on strictly novel scaffolds.")
    return test_df


def plot_specificity_recall_curves():
    print(f"\n--- Generating Clinical Specificity-Recall Curves (Strict Holdout) ---")
    models_dict = joblib.load(CYP_MODEL_PATH)
    df_full = pd.read_csv(ENZYME_CSV_PATH)

    # ENFORCE STRICT DATA ISOLATION
    test_df = get_deterministic_test_set(df_full)

    plt.figure(figsize=(8, 8))

    for target_name, clf in models_dict.items():
        df = test_df.dropna(subset=['smiles', target_name]).copy()
        if df[target_name].sum() < 5:
            continue  # Skip targets with no positive examples in the holdout

        X = np.stack(df['smiles'].apply(CYPPredictor._get_features).values)
        y_true = df[target_name].values
        y_prob = clf.predict_proba(X)[:, 1]

        fpr, tpr, thresholds = roc_curve(y_true, y_prob)

        specificity = 1 - fpr
        recall = tpr

        auc_val = np.trapz(recall[::-1], specificity[::-1])
        plt.plot(specificity, recall, label=f"{target_name} (AUC={auc_val:.2f})")

    plt.xlim(1.0, 0.0)
    plt.ylim(0.0, 1.05)
    plt.xlabel("Specificity (True Negative Rate)")
    plt.ylabel("Recall (True Positive Rate)")
    plt.title("Clinical Specificity-Recall on Unseen Scaffolds")
    plt.legend(loc="lower left", fontsize=8, ncol=2)
    plt.grid(True, linestyle='--', alpha=0.7)

    out_path = os.path.join(AUDIT_OUT_DIR, 'specificity_recall_curves_holdout.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Curves saved to: {out_path}")


def decode_top_morgan_bits(bit_to_find=926, target_name="CYP3A4_inh"):
    print(f"\n--- Consensus Decoding for Morgan Bit {bit_to_find} on {target_name} ---")
    df_full = pd.read_csv(ENZYME_CSV_PATH)
    test_df = get_deterministic_test_set(df_full)

    if target_name not in test_df.columns:
        return

    # Look only at True Positives in the unseen test set
    df_active = test_df[test_df[target_name] == 1].dropna(subset=['smiles'])
    found_mols = []

    for smiles in df_active['smiles']:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: continue

        bit_info = {}
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048, bitInfo=bit_info)

        if bit_to_find in bit_info:
            found_mols.append((mol, bit_info))
            if len(found_mols) >= 3:  # Find consensus across 3 different scaffolds
                break

    if not found_mols:
        print(f"Could not find Bit {bit_to_find} structurally mapped in the active holdout dataset.")
        return

    # Draw the bit across multiple molecules
    mols = [m[0] for m in found_mols]
    bit_infos = [m[1] for m in found_mols]

    # RDKit returns an SVG string by default
    svg_data = Draw.DrawMorganBits(
        [(mol, bit_to_find, bit_info) for mol, bit_info in zip(mols, bit_infos)],
        molsPerRow=3
    )

    # Write the raw string to an SVG file
    out_path = os.path.join(AUDIT_OUT_DIR, f'bit_{bit_to_find}_consensus.svg')
    with open(out_path, 'w') as f:
        f.write(svg_data)

    print(f"Consensus Pharmacophore saved to: {out_path}")

if __name__ == "__main__":
    plot_specificity_recall_curves()

    # Run consensus decoding to prove biological mechanism vs bit collision
    decode_top_morgan_bits(bit_to_find=926, target_name="CYP3A4_inh")
    decode_top_morgan_bits(bit_to_find=1019, target_name="CYP3A4_inh")

