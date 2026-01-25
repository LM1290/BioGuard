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
# FIX: Import new class
from bioguard.train import BioGuardDataset 

# --- CONFIG ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACT_DIR = os.path.join(BASE_DIR, 'artifacts')
DATA_DIR = os.path.join(BASE_DIR, 'data') # Added Data Dir
MODEL_PATH = os.path.join(ARTIFACT_DIR, 'model.pt')
CALIBRATOR_PATH = os.path.join(ARTIFACT_DIR, 'calibrator.joblib')
META_PATH = os.path.join(ARTIFACT_DIR, 'metadata.json')

def load_metadata():
    if not os.path.exists(META_PATH):
        return {}
    with open(META_PATH, 'r') as f:
        return json.load(f)

def evaluate_model(override_split=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    meta = load_metadata()
    
    train_node_dim = meta.get('node_dim', 41)
    train_split = meta.get('split_type', 'cold')
    threshold = meta.get('threshold', 0.5)
    split_type = override_split if override_split else train_split

    print(f"--- BioGuard Evaluation ({split_type.upper()}) ---")

    # 1. Load Data
    print("Loading test set...")
    df = load_twosides_data(split_method=split_type)
    test_df = df[df['split'] == 'test'].reset_index(drop=True)

    # 2. Initialize Model
    model = BioGuardGAT(node_dim=train_node_dim, edge_dim=6).to(device)
    
    if not os.path.exists(MODEL_PATH):
        print("Model not found!")
        sys.exit(1)
        
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    # 3. Load Calibrator
    calibrator = None
    if os.path.exists(CALIBRATOR_PATH):
        calibrator = joblib.load(CALIBRATOR_PATH)

    # 4. Initialize Disk-Based Dataset
    # FIX: Use new signature
    test_dataset = BioGuardDataset(root=DATA_DIR, df=test_df, split='test')
    
    # Enable workers for evaluation too!
    test_loader = PyGDataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)

    print(f"Running inference on {len(test_dataset)} pairs...")
    y_true = []
    y_raw_probs = []

    with torch.no_grad():
        for batch_data in test_loader:
            batch_a = batch_data[0].to(device)
            batch_b = batch_data[1].to(device)
            batch_y = batch_data[2].to(device)

            logits = model(batch_a, batch_b)
            probs = torch.sigmoid(logits).cpu().numpy()

            y_true.extend(batch_y.cpu().numpy().flatten())
            y_raw_probs.extend(probs.flatten())

    y_true = np.array(y_true)
    y_raw_probs = np.array(y_raw_probs)

    # ... (Metrics calculation remains the same) ...
    # [Rest of the file is fine, just ensure indentation matches]
    
    roc = roc_auc_score(y_true, y_raw_probs)
    print(f"ROC-AUC: {roc:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, default=None)
    args = parser.parse_args()
    evaluate_model(args.split)