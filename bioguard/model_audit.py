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
from bioguard.cyp_predictor import CYPPredictor

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

    # Load the dictionary of LightGBM models
    models_dict = joblib.load(CYP_MODEL_PATH)

    # Grab the first successfully trained target (e.g., CYP3A4_inh)
    target_name = list(models_dict.keys())[0]
    clf = models_dict[target_name]

    # Extract feature importances
    importances = clf.feature_importances_

    # The first 2048 features are Morgan FP bits, the last 5 are PhysChem
    fp_importances = importances[:2048]
    physchem_importances = importances[2048:]

    print(f"Auditing Target: {target_name}")
    print(f"Top PhysChem Importances (MolWt, LogP, TPSA, HDon, HAcc): {physchem_importances}")

    # Get top 5 Morgan FP Bits
    top_bits = np.argsort(fp_importances)[::-1][:5]
    print(f"Top 5 driving Morgan Bits for {target_name}: {top_bits}")

    # LEVEL 2: The Architect - Map Bits to Fragments
    print("Mapping top bits to physical fragments...")
    sample_smiles = "CC1(C2=C(C=CC(=C2)Cl)C(=O)O1)CN3C=CN=C3"  # Ketoconazole
    mol = Chem.MolFromSmiles(sample_smiles)

    bit_info = {}
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048, bitInfo=bit_info)

    svgs = []
    for bit in top_bits:
        if bit in bit_info:
            atom_idx, radius = bit_info[bit][0]
            svg = Draw.DrawMorganBit(mol, bit, bit_info)
            svgs.append((bit, svg))
            print(f"  -> Bit {bit} found in sample! Rendered fragment.")
        else:
            print(f"  -> Bit {bit} not present in sample molecule.")

    print("Pharmacophore Audit Complete.\n")


# ==========================================
# LEVEL 4: The Parity Plot (Calibration Check)
# ==========================================
def plot_calibration_curve():
    print("--- Running Level 4: Calibration Integrity Check ---")
    if not os.path.exists(CYP_MODEL_PATH) or not os.path.exists(ENZYME_CSV_PATH):
        print("Missing model or ChEMBL data.")
        return

    models_dict = joblib.load(CYP_MODEL_PATH)
    target_name = list(models_dict.keys())[0]
    clf = models_dict[target_name]

    df = pd.read_csv(ENZYME_CSV_PATH).dropna(subset=['smiles', target_name])
    df_sample = df.sample(min(1000, len(df)), random_state=42)

    X = np.stack(df_sample['smiles'].apply(CYPPredictor._get_features).values)
    y_true = df_sample[target_name].values

    # Predict Probabilities
    y_prob = clf.predict_proba(X)[:, 1]

    fraction_of_positives, mean_predicted_value = calibration_curve(y_true, y_prob, n_bins=10)

    plt.figure(figsize=(8, 8))
    plt.plot(mean_predicted_value, fraction_of_positives, "s-", label=f"LightGBM {target_name}")
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly Calibrated (Truth)")
    plt.ylabel("Fraction of True Positives")
    plt.xlabel("Mean Predicted Probability (LightGBM)")
    plt.title(f"Calibration Curve for {target_name}")
    plt.legend()
    plt.grid(True)

    out_file = os.path.join(AUDIT_OUT_DIR, 'calibration_curve.png')
    plt.savefig(out_file)
    print(f"Parity Plot saved to {out_file}\n")


# ==========================================
# LEVEL 3: GATv2 Attention Attribution (Captum)
# ==========================================
class GATWrapper(torch.nn.Module):
    def __init__(self, model, graph_a_base, graph_b):
        super().__init__()
        self.model = model
        self.graph_a_base = graph_a_base
        self.graph_b = graph_b

    def forward(self, node_features_a):
        # MUST clone graph_a_base to preserve its topological edges, 3D global metrics,
        # and most importantly, its actual RF metabolic prior
        graph_a = self.graph_a_base.clone()
        graph_a.x = node_features_a

        # We need to simulate the batch vector required by PyG Pooling
        graph_a.batch = torch.zeros(graph_a.x.size(0), dtype=torch.long)
        self.graph_b.batch = torch.zeros(self.graph_b.x.size(0), dtype=torch.long)

        return self.model(graph_a, self.graph_b)


def audit_gat_attention():
    print("--- Running Level 3: CSO Graph Attribution ---")
    if not os.path.exists(GAT_MODEL_PATH):
        print("GAT Model not trained yet.")
        return

    device = torch.device("cpu")

    predictor = CYPPredictor()
    enzyme_dim = predictor.vector_dim

    model = BioGuardGAT(node_dim=NODE_DIM, edge_dim=EDGE_DIM, embedding_dim=128, heads=4, enzyme_dim=enzyme_dim)
    model.load_state_dict(torch.load(GAT_MODEL_PATH, map_location=device))
    model.eval()

    smiles_a = "CC(=O)OC1=CC=CC=C1C(=O)O"  # Aspirin
    smiles_b = "CC1=C(C2=CC=CC=C2O1)C3=C(C(=O)C4=CC=CC=C4C3=O)O"  # Warfarin

    g_a = drug_to_graph(smiles_a)
    g_b = drug_to_graph(smiles_b)

    # --- THE FIX: Real Inference Injection ---
    # We pass the real LightGBM metabolic probabilities so Captum tracks how the prior shifts the attention.
    g_a.enzyme_a = torch.tensor([predictor.predict(smiles_a)], dtype=torch.float)
    g_b.enzyme_b = torch.tensor([predictor.predict(smiles_b)], dtype=torch.float)

    # Wrap the model
    wrapped_model = GATWrapper(model, g_a, g_b)
    ig = IntegratedGradients(wrapped_model)

    # Baseline is a graph of zeros (no features)
    baseline = torch.zeros_like(g_a.x)

    attributions, delta = ig.attribute(
        inputs=g_a.x,
        baselines=baseline,
        target=0,
        return_convergence_delta=True
    )

    atom_importance = attributions.sum(dim=1).detach().numpy()

    print(f"\nReal Prior Injected: {g_a.enzyme_a.numpy().round(3)}")
    print("GATv2 Atom Attention Attributions for Aspirin (Interacting with Warfarin):")
    mol_a = Chem.MolFromSmiles(smiles_a)
    for idx, atom in enumerate(mol_a.GetAtoms()):
        symbol = atom.GetSymbol()
        score = atom_importance[idx]
        print(f"  Atom {idx:2} ({symbol:2}): Attribution Score = {score:+.4f}")

    print("\nThe attention heat is mathematically guided by the LightGBM prior.")


if __name__ == "__main__":
    audit_metabolic_priors()
    plot_calibration_curve()
    audit_gat_attention()