"""
Training module for BioGuardGAT DDI prediction model.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import os
import json
import random
import joblib
from sklearn.model_selection import train_test_split
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import precision_recall_curve
from joblib import dump
from tqdm import tqdm

# CRITICAL: Use PyG Loader for graphs
from torch_geometric.loader import DataLoader as PyGDataLoader

from .data_loader import load_twosides_data
from .featurizer import GraphFeaturizer
from .model import BioGuardGAT

CONFIG = {
    'BATCH_SIZE': 128,
    'LEARNING_RATE': 5e-4,
    'EPOCHS': 40,
    'PATIENCE': 8,
    'NUM_WORKERS': 0,
    'SEED': 42
}

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACT_DIR = os.path.join(BASE_DIR, 'artifacts')


def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_pair_disjoint_split(df, test_size=0.2, val_size=0.1, seed=42):
    if 'split' in df.columns:
        train_df = df[df['split'] == 'train'].copy()
        val_df = df[df['split'] == 'val'].copy()
        test_df = df[df['split'] == 'test'].copy()
        return train_df, val_df, test_df

    train_val, test = train_test_split(df, test_size=test_size, random_state=seed)
    train, val = train_test_split(train_val, test_size=val_size / (1 - test_size), random_state=seed)
    return train, val, test


class BioDataset(Dataset):
    """
    Optimized Dataset that pre-computes graphs into RAM.
    """

    def __init__(self, df, name="Dataset"):
        self.df = df.reset_index(drop=True)
        self.featurizer = GraphFeaturizer()
        self.cached_data = []

        print(f"[{name}] Pre-computing graphs into RAM (this happens once)...")
        # Pre-compute all graphs so we don't do it every epoch
        for _, row in tqdm(self.df.iterrows(), total=len(self.df)):
            data_a = self.featurizer.smiles_to_graph(row['smiles_a'])
            data_b = self.featurizer.smiles_to_graph(row['smiles_b'])
            label = torch.tensor(row['label'], dtype=torch.float32)
            self.cached_data.append((data_a, data_b, label))

    def __len__(self):
        return len(self.cached_data)

    def __getitem__(self, idx):
        # Zero computation here, just list access
        return self.cached_data[idx]


def run_training(split_type='pair_disjoint'):
    set_seed(CONFIG['SEED'])
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Training BioGuardGAT on {device} (Split: {split_type})")

    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    # 1. Load Data
    print("Loading TWOSIDES dataset...")
    df = load_twosides_data()
    train_df, val_df, test_df = get_pair_disjoint_split(df)

    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    # 2. Initialize Datasets with Caching
    # This will show a progress bar and take ~30-60s to start
    train_dataset = BioDataset(train_df, name="Train")
    val_dataset = BioDataset(val_df, name="Val")

    train_loader = PyGDataLoader(
        train_dataset,
        batch_size=CONFIG['BATCH_SIZE'],
        shuffle=True,
        num_workers=CONFIG['NUM_WORKERS']
    )

    val_loader = PyGDataLoader(
        val_dataset,
        batch_size=CONFIG['BATCH_SIZE'],
        shuffle=False,
        num_workers=CONFIG['NUM_WORKERS']
    )

    # 3. Initialize Model
    model = BioGuardGAT(
        node_dim=22,
        edge_dim=6,
        embedding_dim=128,
        heads=4
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=CONFIG['LEARNING_RATE'])
    criterion = nn.BCEWithLogitsLoss()

    # 4. Training Loop
    best_val_loss = float('inf')
    patience_counter = 0

    print("\nStarting training loop...")
    for epoch in range(CONFIG['EPOCHS']):
        # --- TRAIN ---
        model.train()
        train_loss = 0

        # This loop will now be very fast
        for batch_a, batch_b, batch_y in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{CONFIG['EPOCHS']}"):
            batch_a = batch_a.to(device)
            batch_b = batch_b.to(device)
            batch_y = batch_y.to(device).unsqueeze(1)

            optimizer.zero_grad()
            logits = model(batch_a, batch_b)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # --- VALIDATE ---
        model.eval()
        val_loss = 0
        y_true_val = []
        y_probs_val = []

        with torch.no_grad():
            for batch_a, batch_b, batch_y in val_loader:
                batch_a = batch_a.to(device)
                batch_b = batch_b.to(device)
                batch_y = batch_y.to(device).unsqueeze(1)

                logits = model(batch_a, batch_b)
                loss = criterion(logits, batch_y)
                val_loss += loss.item()

                probs = torch.sigmoid(logits).cpu().numpy()
                y_true_val.extend(batch_y.cpu().numpy())
                y_probs_val.extend(probs)

        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch + 1}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}")

        # --- CHECKPOINTING ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(ARTIFACT_DIR, 'model.pt'))

            y_true_val = np.array(y_true_val).flatten()
            y_probs_val = np.array(y_probs_val).flatten()
            np.savez(
                os.path.join(ARTIFACT_DIR, 'calibration_data.npz'),
                y_true=y_true_val,
                y_prob=y_probs_val
            )
        else:
            patience_counter += 1
            if patience_counter >= CONFIG['PATIENCE']:
                print(f"Early stopping triggered after {epoch + 1} epochs.")
                break

    # 5. Calibration
    print("\nTraining complete. Calibrating...")
    calib_data = np.load(os.path.join(ARTIFACT_DIR, 'calibration_data.npz'))
    y_true_val = calib_data['y_true']
    y_probs_val = calib_data['y_prob']

    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(y_probs_val, y_true_val)
    dump(iso, os.path.join(ARTIFACT_DIR, 'calibrator.joblib'))

    y_calibrated = iso.transform(y_probs_val)
    precision, recall, thresholds = precision_recall_curve(y_true_val, y_calibrated)

    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5

    print(f"Optimal threshold: {optimal_threshold:.4f} (F1={f1_scores[optimal_idx]:.4f})")

    metadata = {
        'version': f'2.0-GAT-{split_type}',
        'split_type': split_type,
        'config': CONFIG,
        'threshold': float(optimal_threshold),
        'model_type': 'GAT',
        'best_val_loss': float(best_val_loss),
        'split_info': {
            'train_size': len(train_df),
            'val_size': len(val_df),
            'test_size': len(test_df)
        }
    }

    with open(os.path.join(ARTIFACT_DIR, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"[OK] Model artifacts saved to {ARTIFACT_DIR}")


if __name__ == "__main__":
    run_training()