import os
import joblib
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem.Draw import SimilarityMaps
from sklearn.calibration import calibration_curve
from captum.attr import IntegratedGradients

from bioguard.config import BASE_DIR, DATA_DIR, ARTIFACT_DIR, NODE_DIM, EDGE_DIM
from bioguard.model import BioGuardGAT
from bioguard.featurizer import drug_to_graph
from bioguard.cyp_predictor import CYPPredictor

CYP_MODEL_PATH = os.path.join(ARTIFACT_DIR, 'cyp_predictor.joblib')
GAT_MODEL_PATH = os.path.join(ARTIFACT_DIR, 'model.pt')
ENZYME_CSV_PATH = os.path.join(DATA_DIR, 'enzyme_features_full.csv')
AUDIT_OUT_DIR = os.path.join(ARTIFACT_DIR, 'audit_reports')
CALIBRATION_DIR = os.path.join(AUDIT_OUT_DIR, 'calibration_plots')
os.makedirs(CALIBRATION_DIR, exist_ok=True)


def get_base_estimator(clf):
    """Safely unwrap the CalibratedClassifierCV to get the LightGBM base."""
    if hasattr(clf, 'calibrated_classifiers_'):
        return clf.calibrated_classifiers_[0].estimator
    return clf


def audit_pure_topology():
    print("\n--- Running Pure Topology Pharmacophore Audit ---")
    models_dict = joblib.load(CYP_MODEL_PATH)

    for target_name, clf in models_dict.items():
        base_clf = get_base_estimator(clf)
        importances = base_clf.feature_importances_

        # Only Morgan bits exist now
        top_bits = np.argsort(importances)[::-1][:5]
        print(f"Target: {target_name} | Top 5 Morgan Drivers: {top_bits}")


def plot_calibrated_curves():
    print(f"\n--- Generating Calibrated Parity Plots ---")
    models_dict = joblib.load(CYP_MODEL_PATH)
    df_full = pd.read_csv(ENZYME_CSV_PATH)

    for target_name, clf in models_dict.items():
        df = df_full.dropna(subset=['smiles', target_name]).copy()
        if df[target_name].sum() < 10: continue

        df_sample = df.sample(min(1000, len(df)), random_state=42)
        X = np.stack(df_sample['smiles'].apply(CYPPredictor._get_features).values)
        y_true = df_sample[target_name].values
        y_prob = clf.predict_proba(X)[:, 1]

        fraction_of_positives, mean_predicted_value = calibration_curve(y_true, y_prob, n_bins=10)

        plt.figure(figsize=(6, 6))
        plt.plot(mean_predicted_value, fraction_of_positives, "s-", label=f"Calibrated {target_name}")
        plt.plot([0, 1], [0, 1], "k:", label="Perfect Calibration")
        plt.ylabel("Fraction of True Positives")
        plt.xlabel("Mean Predicted Probability")
        plt.title(f"Parity Plot: {target_name}")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(CALIBRATION_DIR, f'calibration_{target_name}.png'))
        plt.close()
    print("All Parity Plots generated successfully.\n")


# ... [Keep GATWrapper and the rest of the file the same to run the GAT attention heatmap] ...

if __name__ == "__main__":
    audit_pure_topology()
    plot_calibrated_curves()
    # audit_gat_attention() # <-- Un-comment to run your Aspirin heatmap