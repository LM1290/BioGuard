import torch
import numpy as np
import os
import sys
import json
import joblib
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, confusion_matrix
from torch.utils.data import DataLoader

from .model import BioGuardNet
from .data_loader import load_twosides_data
from .train import get_pair_disjoint_split, BioDataset
from .featurizer import BioFeaturizer

ARTIFACT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'artifacts')
MODEL_PATH = os.path.join(ARTIFACT_DIR, 'model.pt')
CALIBRATOR_PATH = os.path.join(ARTIFACT_DIR, 'calibrator.joblib')
META_PATH = os.path.join(ARTIFACT_DIR, 'metadata.json')


def evaluate_model(split_type='pair_disjoint'):
    """
    Evaluate BioGuardNet model on test set with proper batch inference.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on {device} (Split: {split_type})")

    # 1. Load Artifacts
    if not os.path.exists(MODEL_PATH):
        print("Model not found. Please train first: python -m bioguard.main train")
        sys.exit(1)

    featurizer = BioFeaturizer()
    model = BioGuardNet(input_dim=featurizer.total_dim).to(device)
    
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
        model.eval()
        print("[OK] Model loaded successfully")
    except Exception as e:
        print(f"Failed to load model: {e}")
        sys.exit(1)

    calibrator = None
    if os.path.exists(CALIBRATOR_PATH):
        try:
            calibrator = joblib.load(CALIBRATOR_PATH)
            print("[OK] Calibrator loaded")
        except Exception as e:
            print(f"Warning: Calibrator load failed: {e}")

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
    train_df, val_df, test_df = get_pair_disjoint_split(df)

    if len(test_df) == 0:
        print("ERROR: Empty test set. Cannot evaluate.")
        sys.exit(0)

    print(f"Dataset sizes:")
    print(f"  Train: {len(train_df)}")
    print(f"  Val:   {len(val_df)}")
    print(f"  Test:  {len(test_df)}")

    # Use batched inference
    test_loader = DataLoader(
        BioDataset(test_df), 
        batch_size=128, 
        shuffle=False, 
        num_workers=0
    )

    # 3. Inference
    print("\nRunning inference on test set...")
    y_true = []
    y_raw_probs = []

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            logits = model(batch_X.to(device))
            probs = torch.sigmoid(logits).cpu().numpy()
            y_true.extend(batch_y.numpy().flatten())
            y_raw_probs.extend(probs.flatten())

    y_true = np.array(y_true)
    y_raw_probs = np.array(y_raw_probs)

    # 4. Calibration
    if calibrator:
        y_final_probs = calibrator.transform(y_raw_probs)
        y_final_probs = np.clip(y_final_probs, 0.0, 1.0)
        print("[OK] Applied probability calibration")
    else:
        y_final_probs = y_raw_probs
        print("[WARNING] No calibration applied (using raw probabilities)")

    # 5. Metrics
    print("\n" + "="*60)
    print("TEST SET EVALUATION RESULTS")
    print("="*60)
    
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
    print(f"  Sensitivity : {sensitivity:.4f} (Recall/TPR)")
    print(f"  Specificity : {specificity:.4f} (TNR)")
    print(f"  F1 Score    : {f1:.4f}")
    
    print(f"\nConfusion Matrix:")
    print(f"                Predicted Negative  Predicted Positive")
    print(f"  Actual Negative     {tn:8d}           {fp:8d}")
    print(f"  Actual Positive     {fn:8d}           {tp:8d}")
    
    # Class distribution
    pos_rate = y_true.sum() / len(y_true)
    pred_pos_rate = y_pred.sum() / len(y_pred)
    
    print(f"\nClass Distribution:")
    print(f"  True positive rate      : {pos_rate:.2%}")
    print(f"  Predicted positive rate : {pred_pos_rate:.2%}")
    
    print("="*60)
    
    # Summary assessment
    print("\n" + "="*60)
    print("PERFORMANCE ASSESSMENT")
    print("="*60)
    
    if pr >= 0.85:
        assessment = "EXCELLENT"
    elif pr >= 0.75:
        assessment = "GOOD"
    elif pr >= 0.65:
        assessment = "ACCEPTABLE"
    else:
        assessment = "NEEDS IMPROVEMENT"
    
    print(f"Overall: {assessment}")
    print(f"  - PR-AUC is the primary metric for imbalanced DDI prediction")
    print(f"  - Current PR-AUC: {pr:.4f}")
    print(f"  - Sensitivity-Specificity tradeoff: {sensitivity:.4f} / {specificity:.4f}")
    
    if sensitivity < 0.7:
        print("\nWARNING: Low sensitivity - may miss many true interactions")
    if specificity < 0.7:
        print("\nWARNING: Low specificity - many false alarms")
    
    print("="*60)
    
    # Save results for comparison with baselines
    eval_results = {
        "name": "BioGuardNet (Neural Network)",
        "description": "Deep learning model with structure features + calibration",
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
    print("  Run 'python -m bioguard.main compare' to see comparison with baselines")
    print("="*60)


if __name__ == "__main__":
    evaluate_model()
