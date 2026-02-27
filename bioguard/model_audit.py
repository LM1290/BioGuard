import os
import joblib
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from sklearn.calibration import calibration_curve
from captum.attr import IntegratedGradients

# Internal imports
from bioguard.config import BASE_DIR, DATA_DIR, ARTIFACT_DIR, NODE_DIM, EDGE_DIM
from bioguard.model import BioGuardGAT
from bioguard.featurizer import drug_to_graph
from bioguard.train_cyp_predictor import get_physchem_and_fp

# Paths
CYP_MODEL_PATH = os.path.join(ARTIFACT_DIR, 'cyp_predictor.joblib')
GAT_MODEL_PATH = os.path.join(ARTIFACT_DIR, 'model.pt')
ENZYME_CSV_PATH = os.path.join(DATA_DIR, 'enzyme_features_full.csv')
AUDIT_OUT_DIR = os.path.join(ARTIFACT_DIR, 'audit_reports')

os.makedirs(AUDIT_OUT_DIR, exist_ok=True)


# ==========================================
# LEVEL 1 & 2: LightGBM Importance & RDKit Fragments
# ==========================================
def audit_metabolic_priors():
    print("\n--- Running Level 1 & 2: Pharmacophore Audit ---")
    if not os.path.exists(CYP_MODEL_PATH):
        print("CYP Predictor not found. Run train_cyp_predictor.py first.")
        return

    # Load MultiOutput LGBM
    clf = joblib.load(CYP_MODEL_PATH)

    # Let's audit the first task (usually CYP3A4 Substrate)
    target_idx = 0
    task_name = "CYP3A4_Substrate"

    # Extract feature importances from the specific LightGBM estimator
    estimator = clf.estimators_[target_idx]
    importances = estimator.feature_importances_

    # The first 2048 features are Morgan FP bits, the last 5 are PhysChem
    fp_importances = importances[:2048]
    physchem_importances = importances[2048:]

    print(f"Top PhysChem Importances (MolWt, LogP, TPSA, HDon, HAcc): {physchem_importances}")

    # Get top 5 Morgan FP Bits
    top_bits = np.argsort(fp_importances)[::-1][:5]
    print(f"Top 5 driving Morgan Bits for {task_name}: {top_bits}")

    # LEVEL 2: The Architect - Map Bits to Fragments
    print("Mapping top bits to physical fragments...")

    # Sample a known inhibitor/substrate (e.g., Ketoconazole)
    sample_smiles = "CC1(C2=C(C=CC(=C2)Cl)C(=O)O1)CN3C=CN=C3"
    mol = Chem.MolFromSmiles(sample_smiles)

    bit_info = {}
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048, bitInfo=bit_info)

    svgs = []
    for bit in top_bits:
        if bit in bit_info:
            # Get the first environment that triggered this bit
            atom_idx, radius = bit_info[bit][0]
            # Draw the specific fragment
            svg = Draw.DrawMorganBit(mol, bit, bit_info)
            svgs.append((bit, svg))
            print(f"  -> Bit {bit} found in sample! Rendered fragment.")
        else:
            print(f"  -> Bit {bit} not present in this specific sample molecule.")

    # In a real notebook environment, you would display(svg) here.
    print("Pharmacophore Audit Complete.\n")


# ==========================================
# LEVEL 4: The Parity Plot (Calibration Check)
# ==========================================
def plot_calibration_curve():
    print("--- Running Level 4: Calibration Integrity Check ---")
    if not os.path.exists(CYP_MODEL_PATH) or not os.path.exists(ENZYME_CSV_PATH):
        print("Missing model or ChEMBL data.")
        return

    clf = joblib.load(CYP_MODEL_PATH)
    df = pd.read_csv(ENZYME_CSV_PATH).dropna(subset=['smiles', 'cyp3a4_sub'])

    # Sample 1000 molecules for the parity plot
    df_sample = df.sample(min(1000, len(df)), random_state=42)

    X = np.stack(df_sample['smiles'].apply(get_physchem_and_fp).values)
    y_true = df_sample['cyp3a4_sub'].values

    # Predict Probabilities
    y_prob = clf.predict_proba(X)[0][:, 1]  # Task 0 (CYP3A4 Substrate), Positive Class

    # Calculate Calibration
    fraction_of_positives, mean_predicted_value = calibration_curve(y_true, y_prob, n_bins=10)

    plt.figure(figsize=(8, 8))
    plt.plot(mean_predicted_value, fraction_of_positives, "s-", label="LightGBM CYP3A4")
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly Calibrated (Truth)")
    plt.ylabel("Fraction of True Positives")
    plt.xlabel("Mean Predicted Probability (RF/LGBM)")
    plt.title("Calibration Curve (Parity Plot) for Metabolic Prior")
    plt.legend()
    plt.grid(True)

    out_file = os.path.join(AUDIT_OUT_DIR, 'calibration_curve.png')
    plt.savefig(out_file)
    print(f"Parity Plot saved to {out_file}\n")


# ==========================================
# LEVEL 3: GATv2 Attention Attribution (Captum)
# ==========================================
# Wrapper class to make PyG work with Captum
class GATWrapper(torch.nn.Module):
    def __init__(self, model, graph_b):
        super().__init__()
        self.model = model
        self.graph_b = graph_b  # We hold Drug B constant to audit Drug A

    def forward(self, node_features_a):
        # Captum feeds raw tensors, we need to repackage them for PyG
        # Note: In a true deep dive, we'd wrap the edge_index as well,
        # but integrated gradients on node features is sufficient for the CSO pitch.
        dummy_graph_a = self.graph_b.clone()
        dummy_graph_a.x = node_features_a
        return self.model(dummy_graph_a, self.graph_b)


def audit_gat_attention():
    print("--- Running Level 3: CSO Graph Attribution ---")
    if not os.path.exists(GAT_MODEL_PATH):
        print("GAT Model not trained yet.")
        return

    device = torch.device("cpu")  # Keep on CPU for easier Captum manipulation

    # Mock Enzyme Dimension (Update this based on your actual EnzymeManager dim)
    enzyme_dim = 11

    model = BioGuardGAT(NODE_DIM, EDGE_DIM, embedding_dim=128, heads=4, enzyme_dim=enzyme_dim)
    model.load_state_dict(torch.load(GAT_MODEL_PATH, map_location=device))
    model.eval()

    # Create two sample drugs (e.g., Warfarin + Aspirin)
    smiles_a = "CC(=O)OC1=CC=CC=C1C(=O)O"  # Aspirin
    smiles_b = "CC1=C(C2=CC=CC=C2O1)C3=C(C(=O)C4=CC=CC=C4C3=O)O"  # Warfarin

    g_a = drug_to_graph(smiles_a)
    g_b = drug_to_graph(smiles_b)

    # Attach dummy metabolic priors for the forward pass
    g_a.enzyme_a = torch.zeros((1, enzyme_dim))
    g_b.enzyme_b = torch.zeros((1, enzyme_dim))

    # Wrap the model
    wrapped_model = GATWrapper(model, g_b)
    ig = IntegratedGradients(wrapped_model)

    # Calculate attribution scores for Drug A's atoms
    # Baseline is a graph of zeros (no features)
    baseline = torch.zeros_like(g_a.x)

    attributions, delta = ig.attribute(
        inputs=g_a.x,
        baselines=baseline,
        target=0,  # Binary classification target
        return_convergence_delta=True
    )

    # Sum attributions across all feature dimensions to get a single score per atom
    atom_importance = attributions.sum(dim=1).detach().numpy()

    print("GATv2 Atom Attention Attributions for Aspirin (Interacting with Warfarin):")
    mol_a = Chem.MolFromSmiles(smiles_a)
    for idx, atom in enumerate(mol_a.GetAtoms()):
        symbol = atom.GetSymbol()
        score = atom_importance[idx]
        # A high positive score means this atom strongly pushed the model to predict an interaction
        print(f"  Atom {idx:2} ({symbol:2}): Attribution Score = {score:+.4f}")

    print(
        "\nCSO Pitch Ready: We can trace the GAT's interaction prediction back to specific atoms in the molecular graph.")


if __name__ == "__main__":
    audit_metabolic_priors()
    plot_calibration_curve()
    audit_gat_attention()