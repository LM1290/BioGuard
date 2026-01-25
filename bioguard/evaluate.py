import torch
import numpy as np
import os
import sys
import json
import joblib
import argparse
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
# Import the Cached Dataset class we defined in train.py
from bioguard.train import BioGuardDataset

# --- CONFIG ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACT_DIR = os.path.join(BASE_DIR, 'artifacts')
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_PATH = os.path.join(ARTIFACT_DIR, 'model.pt')
CALIBRATOR_PATH = os.path.join(ARTIFACT_DIR, 'calibrator.joblib')
META_PATH = os.path.join(ARTIFACT_DIR, 'metadata.json')


def load_metadata():
    if not os.path.exists(META_PATH):
        return {}
    with open(META_PATH, 'r') as f:
        return json.load(f)


def evaluate_model(override_split=None):
    # Auto-detect device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    meta = load_metadata()
    train_node_dim = meta.get('node_dim', 41)
    train_split = meta.get('split_type', 'cold')
    threshold = meta.get('threshold', 0.5)

    # Allow overriding the split (e.g., testing on 'random' even if trained on 'cold')
    split_type = override_split if override_split else train_split

    print(f"--- BioGuard Evaluation ---")
    print(f"Model Node Dim: {train_node_dim}")
    print(f"Eval Split:   {split_type.upper()}")
    print(f"Threshold:    {threshold:.4f}")

    # 1. Load Data
    print("\nLoading data...")
    df = load_twosides_data(split_method=split_type)

    # We evaluate on the TEST set of the requested split
    test_df = df[df['split'] == 'test'].reset_index(drop=True)

    # 2. Initialize Model
    model = BioGuardGAT(node_dim=train_node_dim, edge_dim=6).to(device)

    if not os.path.exists(MODEL_PATH):
        print(f"CRITICAL: Model file not found at {MODEL_PATH}")
        sys.exit(1)

    print(f"Loading weights from {MODEL_PATH}...")
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
        print("[OK] Model weights loaded successfully")
    except Exception as e:
        print(f"[ERROR] Failed to load weights: {e}")
        sys.exit(1)

    model.eval()

    # 3. Load Calibrator (Optional)
    calibrator = None
    if os.path.exists(CALIBRATOR_PATH):
        calibrator = joblib.load(CALIBRATOR_PATH)
        print("[OK] Calibrator loaded")

    # 4. Initialize Dataset (RAM Cached)
    # We reuse the class from train.py to get the same caching benefits
    test_dataset = BioGuardDataset(root=DATA_DIR, df=test_df, split='test')

    # DataLoader (num_workers=0 is fastest for in-memory data)
    test_loader = PyGDataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=0)

    print(f"[Test] Running inference on {len(test_dataset)} pairs...")

    y_true = []
    y_raw_probs = []

    with torch.no_grad():
        for batch_data in test_loader:
            batch_a = batch_data[0].to(device)
            batch_b = batch_data[1].to(device)
            batch_y = batch_data[2].to(device)

            logits = model(batch_a, batch_b)
            probs = torch.sigmoid(logits)

            y_true.extend(batch_y.cpu().numpy().flatten())
            y_raw_probs.extend(probs.cpu().numpy().flatten())

    y_true = np.array(y_true)
    y_raw_probs = np.array(y_raw_probs)

    # 5. Calculate Metrics
    # Standard threshold 0.5 for raw predictions
    y_pred = (y_raw_probs >= 0.5).astype(int)

    roc_auc = roc_auc_score(y_true, y_raw_probs)
    pr_auc = average_precision_score(y_true, y_raw_probs)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # Specificity = TN / (TN + FP)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  ROC-AUC     : {roc_auc:.4f}")
    print(f"  PR-AUC      : {pr_auc:.4f}")
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