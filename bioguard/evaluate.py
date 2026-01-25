"""
Evaluation module for BioGuardGAT.
"""

import torch
import numpy as np
import os
import sys
import json
import joblib
import argparse
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, confusion_matrix
from torch_geometric.loader import DataLoader as PyGDataLoader

from .model import BioGuardGAT
from .data_loader import load_twosides_data
from bioguard.train import BioDataset
# Import all split functions to support whatever was used in training


ARTIFACT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'artifacts')
MODEL_PATH = os.path.join(ARTIFACT_DIR, 'model.pt')
CALIBRATOR_PATH = os.path.join(ARTIFACT_DIR, 'calibrator.joblib')
META_PATH = os.path.join(ARTIFACT_DIR, 'metadata.json')


def load_metadata():
    if not os.path.exists(META_PATH):
        print(f"[WARNING] Metadata not found at {META_PATH}. Assuming defaults.")
        return {}
    with open(META_PATH, 'r') as f:
        return json.load(f)


def evaluate_model(override_split=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load Metadata (The Source of Truth)
    meta = load_metadata()

    # Detect training configuration
    train_node_dim = meta.get('node_dim', 41)  # Default to 41 (Series B) if missing, or 22 (Legacy)
    train_split = meta.get('split_type', 'cold')
    threshold = meta.get('threshold', 0.5)

    # Allow CLI override for split, otherwise use what we trained on
    split_type = override_split if override_split else train_split

    print(f"--- BioGuard Evaluation ---")
    print(f"Device:       {device}")
    print(f"Model Node Dim: {train_node_dim}")
    print(f"Eval Split:   {split_type.upper()}")
    print(f"Threshold:    {threshold:.4f}")

    # 2. Load Data
    print("\nLoading data...")
    print(f"Loading test set from pre-computed {args.split} split...")
    df = load_twosides_data()

    # We strictly trust the loader now.
    # If you ran the loader in 'scaffold' mode, this returns the scaffold test set.
    test_df = df[df['split'] == 'test']

    if len(test_df) == 0:
        raise ValueError("Test set is empty! Check if data_loader generated splits correctly.")
    # 3. Initialize Model & Load Weights
    # CRITICAL: Use the node_dim found in metadata
    model = BioGuardGAT(
        node_dim=train_node_dim,
        embedding_dim=128,
        heads=4
    ).to(device)

    if not os.path.exists(MODEL_PATH):
        print("Model file not found. Train first!")
        sys.exit(1)

    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
        model.eval()
        print("[OK] Model weights loaded successfully")
    except RuntimeError as e:
        print(f"\n[FATAL ERROR] Weight mismatch!")
        print(f"The saved model and the code definition do not match.")
        print(f"Error details: {e}")
        print("Fix: Ensure 'node_dim' in evaluate.py matches 'node_dim' used in train.py.")
        sys.exit(1)

    # 4. Load Calibrator
    calibrator = None
    if os.path.exists(CALIBRATOR_PATH):
        calibrator = joblib.load(CALIBRATOR_PATH)
        print("[OK] Calibrator loaded")

    # 5. Inference Loop
    test_dataset = BioDataset(test_df)
    test_loader = PyGDataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=0)

    print("\nRunning inference...")
    y_true = []
    y_raw_probs = []

    with torch.no_grad():
        for batch_a, batch_b, batch_y in test_loader:
            batch_a = batch_a.to(device)
            batch_b = batch_b.to(device)

            logits = model(batch_a, batch_b)
            probs = torch.sigmoid(logits).cpu().numpy()

            y_true.extend(batch_y.numpy().flatten())
            y_raw_probs.extend(probs.flatten())

    y_true = np.array(y_true)
    y_raw_probs = np.array(y_raw_probs)

    # 6. Apply Calibration
    if calibrator:
        y_final_probs = calibrator.transform(y_raw_probs)
        y_final_probs = np.clip(y_final_probs, 0.0, 1.0)
    else:
        y_final_probs = y_raw_probs

    # 7. Metrics
    roc = roc_auc_score(y_true, y_final_probs)
    pr = average_precision_score(y_true, y_final_probs)

    y_pred = (y_final_probs >= threshold).astype(int)
    acc = accuracy_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0

    print(f"\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  ROC-AUC     : {roc:.4f}")
    print(f"  PR-AUC      : {pr:.4f}")
    print(f"  Accuracy    : {acc:.4f}")
    print(f"  Precision   : {precision:.4f}")
    print(f"  Recall      : {sensitivity:.4f}")
    print(f"  Specificity : {specificity:.4f}")
    print(f"  F1 Score    : {f1:.4f}")
    print("=" * 60)

    # Save
    results = {
        "roc_auc": float(roc),
        "pr_auc": float(pr),
        "accuracy": float(acc),
        "f1": float(f1),
        "split_used": split_type
    }
    with open(os.path.join(ARTIFACT_DIR, 'eval_results.json'), 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, default=None, choices=['random', 'cold', 'scaffold'],
                        help="Override split type for evaluation")
    args = parser.parse_args()

    evaluate_model(args.split)