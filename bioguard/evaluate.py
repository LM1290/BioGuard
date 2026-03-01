import torch
import numpy as np
import os
import sys
import json
import joblib
import argparse
from tqdm import tqdm
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
from torch_geometric.loader import DataLoader as PyGDataLoader

# Internal Imports
from .model import BioGuardGAT
from .data_loader import load_twosides_data
from .enzyme import EnzymeManager
# Import the dataset class from your updated training script
from bioguard.train import BioGuardDataset

# --- CONFIG ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACT_DIR = os.path.join(BASE_DIR, 'artifacts')
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_PATH = os.path.join(ARTIFACT_DIR, 'model.pt')
CALIBRATOR_PATH = os.path.join(ARTIFACT_DIR, 'calibrator.joblib')
META_PATH = os.path.join(ARTIFACT_DIR, 'metadata.json')


def load_metadata():
    """Loads model training metadata to ensure architectural synchronization."""
    if not os.path.exists(META_PATH):
        return {}
    with open(META_PATH, 'r') as f:
        return json.load(f)


def evaluate_model(override_split=None):
    # Auto-detect device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. Load Metadata (The Source of Truth)
    meta = load_metadata()
    if not meta:
        print(f"[WARNING] No metadata found at {META_PATH}. Using defaults.")

    # Dynamically extract architecture parameters
    node_dim = meta.get('node_dim', 46)  # Using your verified 46-dim node features
    edge_dim = meta.get('edge_dim', 8)
    embedding_dim = meta.get('embedding_dim', 128)
    heads = meta.get('heads', 4)

    # Training Params
    train_split = meta.get('split_type', 'cold_drug')
    threshold = meta.get('threshold', 0.5)  # Standard clinical threshold

    # Allow overriding the split for cross-validation tests
    split_type = override_split if override_split else train_split

    print("--- BioGuard Final Clinical Evaluation ---")
    print(f"Architecture: Dim={embedding_dim}, Heads={heads}")
    print(f"Eval Split:   {split_type.upper()}")
    print(f"Threshold:    {threshold:.4f}")

    # 2. Load Data
    print("\nLoading holdout data...")
    df = load_twosides_data(split_method=split_type)
    test_df = df[df['split'] == 'test'].reset_index(drop=True)

    # 3. Initialize Enzyme Manager (Strict Dimension Check)
    enzyme_manager = EnzymeManager(allow_degraded=False)
    enzyme_dim = enzyme_manager.vector_dim

    # 4. Initialize Model (Adaptive Ensemble Gate Architecture)
    model = BioGuardGAT(
        node_dim=node_dim,
        edge_dim=edge_dim,
        embedding_dim=embedding_dim,
        heads=heads,
        enzyme_dim=enzyme_dim
    ).to(device)

    if not os.path.exists(MODEL_PATH):
        print(f"CRITICAL: Model weights missing at {MODEL_PATH}")
        sys.exit(1)

    print(f"Loading weights from {MODEL_PATH}...")
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
        print("[OK] Model weights loaded successfully")
    except Exception as e:
        print(f"[ERROR] Failed to load weights: {e}")
        sys.exit(1)

    model.eval()

    # 5. Initialize Dataset & Loader
    test_dataset = BioGuardDataset(root=DATA_DIR, df=test_df, split='test')
    test_loader = PyGDataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=0)

    print(f"[Test] Running inference on {len(test_dataset)} pairs...")

    y_true = []
    y_raw_probs = []
    alpha_vals = []  # Track the trust balance between GAT and Prior

    # 6. Inference Loop (Unpacking the Ensemble Tuple)
    with torch.no_grad():
        for batch_data in tqdm(test_loader, desc="Testing"):
            batch_a = batch_data[0].to(device)
            batch_b = batch_data[1].to(device)
            batch_y = batch_data[2].to(device)

            # Unpack the tuple: (final_logits, alpha_gate)
            logits, alpha = model(batch_a, batch_b)

            probs = torch.sigmoid(logits)

            y_true.extend(batch_y.cpu().numpy().flatten())
            y_raw_probs.extend(probs.cpu().numpy().flatten())
            alpha_vals.extend(alpha.cpu().numpy().flatten())

    y_true = np.array(y_true)
    y_raw_probs = np.array(y_raw_probs)
    mean_alpha = np.mean(alpha_vals)

    # 7. Final Metrics
    y_pred = (y_raw_probs >= threshold).astype(int)

    roc_auc = roc_auc_score(y_true, y_raw_probs)
    pr_auc = average_precision_score(y_true, y_raw_probs)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    print("\n" + "=" * 60)
    print("FINAL CLINICAL RESULTS SUMMARY")
    print("=" * 60)
    print(f"  ROC-AUC     : {roc_auc:.4f}")
    print(f"  PR-AUC      : {pr_auc:.4f} (AUPRC)")
    print(f"  Mean Alpha  : {mean_alpha:.4f} <-- (0.0=Prior, 1.0=Graph)")
    print("-" * 60)
    print(f"  Accuracy    : {acc:.4f}")
    print(f"  Precision   : {prec:.4f}")
    print(f"  Recall      : {rec:.4f}")
    print(f"  Specificity : {specificity:.4f}")
    print(f"  F1 Score    : {f1:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, default=None)
    args = parser.parse_args()
    evaluate_model(args.split)