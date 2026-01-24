import torch
import numpy as np
import os
import sys
import json
import joblib
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, confusion_matrix
# CRITICAL: Use PyG DataLoader for batched graph inference
from torch_geometric.loader import DataLoader as PyGDataLoader

from .model import BioGuardGAT
from .data_loader import load_twosides_data
from .train import get_pair_disjoint_split, BioDataset
from .featurizer import GraphFeaturizer

ARTIFACT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'artifacts')
MODEL_PATH = os.path.join(ARTIFACT_DIR, 'model.pt')
CALIBRATOR_PATH = os.path.join(ARTIFACT_DIR, 'calibrator.joblib')
META_PATH = os.path.join(ARTIFACT_DIR, 'metadata.json')


def evaluate_model(split_type='pair_disjoint'):
    """
    Evaluate BioGuardGAT (Graph Neural Network) on test set.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating BioGuardGAT on {device} (Split: {split_type})")

    # 1. Load Model
    if not os.path.exists(MODEL_PATH):
        print("Model not found. Please train first: python -m bioguard.main train")
        sys.exit(1)

    # Initialize GAT with the same parameters used in training
    # Note: These dimensions match the GraphFeaturizer and Model definitions from previous steps
    model = BioGuardGAT(node_dim=16, embedding_dim=128, heads=4).to(device)

    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
        model.eval()
        print("[OK] BioGuardGAT loaded successfully")
    except Exception as e:
        print(f"Failed to load model: {e}")
        sys.exit(1)

    # Load Calibrator
    calibrator = None
    if os.path.exists(CALIBRATOR_PATH):
        try:
            calibrator = joblib.load(CALIBRATOR_PATH)
            print("[OK] Calibrator loaded")
        except Exception as e:
            print(f"Warning: Calibrator load failed: {e}")

    # Load Threshold
    threshold = 0.5
    if os.path.exists(META_PATH):
        try:
            with open(META_PATH, 'r') as f:
                meta = json.load(f)
                threshold = meta.get('threshold', 0.5)
                print(f"[OK] Loaded optimal threshold: {threshold:.4f}")
        except Exception as e:
            print(f"Warning: Metadata load failed: {e}")

    # 2. Load Data
    print("\nLoading data...")
    df = load_twosides_data()
    # We only need the test set for evaluation
    _, _, test_df = get_pair_disjoint_split(df)

    if len(test_df) == 0:
        print("ERROR: Empty test set. Cannot evaluate.")
        sys.exit(0)

    print(f"Test Set Size: {len(test_df)}")

    # 3. Setup Graph Data Loading
    # Use PyGDataLoader to properly batch graph objects
    test_dataset = BioDataset(test_df)
    test_loader = PyGDataLoader(
        test_dataset,
        batch_size=128,  # Larger batch size is usually fine for inference
        shuffle=False,
        num_workers=0  # Set to 0 to avoid potential multiprocessing issues
    )

    # 4. Inference
    print("\nRunning graph inference on test set...")
    y_true = []
    y_raw_probs = []

    with torch.no_grad():
        for batch_a, batch_b, batch_y in test_loader:
            # Move graph batches to GPU
            batch_a = batch_a.to(device)
            batch_b = batch_b.to(device)

            # Forward pass: (Graph A, Graph B) -> Logits
            # The model now takes two separate graph batches
            logits = model(batch_a, batch_b)
            probs = torch.sigmoid(logits).cpu().numpy()

            y_true.extend(batch_y.numpy().flatten())
            y_raw_probs.extend(probs.flatten())

    y_true = np.array(y_true)
    y_raw_probs = np.array(y_raw_probs)

    # 5. Calibration
    if calibrator:
        y_final_probs = calibrator.transform(y_raw_probs)
        y_final_probs = np.clip(y_final_probs, 0.0, 1.0)
        print("[OK] Applied probability calibration")
    else:
        y_final_probs = y_raw_probs
        print("[WARNING] No calibration applied (using raw probabilities)")

    # 6. Metrics Calculation
    print("\n" + "=" * 60)
    print("TEST SET EVALUATION RESULTS (GNN)")
    print("=" * 60)

    # Discrimination metrics
    roc = roc_auc_score(y_true, y_final_probs)
    pr = average_precision_score(y_true, y_final_probs)

    print(f"\nDiscrimination Metrics:")
    print(f"  ROC-AUC     : {roc:.4f}")
    print(f"  PR-AUC      : {pr:.4f} (primary metric)")

    # Threshold-based metrics
    y_pred = (y_final_probs >= threshold).astype(int)
    acc = accuracy_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0.0

    print(f"\nClassification Metrics (threshold={threshold:.4f}):")
    print(f"  Accuracy    : {acc:.4f}")
    print(f"  Precision   : {precision:.4f}")
    print(f"  Sensitivity : {sensitivity:.4f} (Recall)")
    print(f"  Specificity : {specificity:.4f} (TNR)")
    print(f"  F1 Score    : {f1:.4f}")

    # Save results
    eval_results = {
        "name": "BioGuardGAT (GNN)",
        "description": "Graph Attention Network with symmetric architecture",
        "roc_auc": float(roc),
        "pr_auc": float(pr),
        "threshold": float(threshold),
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(sensitivity),
        "f1": float(f1),
        "specificity": float(specificity),
        "n_test": len(y_true),
        "calibrated": calibrator is not None
    }

    eval_results_path = os.path.join(ARTIFACT_DIR, 'eval_results.json')
    with open(eval_results_path, 'w') as f:
        json.dump(eval_results, f, indent=2)

    print(f"\n[OK] Evaluation results saved to {eval_results_path}")
    print("=" * 60)


if __name__ == "__main__":
    evaluate_model()