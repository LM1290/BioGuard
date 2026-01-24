"""
Training module for BioGuardGAT DDI prediction model.
FIXED: v2.0 - Removes redundant splitting logic. Respects data_loader.py splits.
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
import argparse
from tqdm import tqdm

from torch_geometric.loader import DataLoader as PyGDataLoader

from .data_loader import load_twosides_data
from .featurizer import GraphFeaturizer
from .model import BioGuardGAT

# Default Config
CONFIG = {
    'BATCH_SIZE': 128,
    'LEARNING_RATE': 5e-4,
    'EPOCHS': 40,
    'PATIENCE': 8,
    'NUM_WORKERS': 4, # UPDATED: Increased from 0 to 4 to prevent GPU starvation
    'SEED': 42
}

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACT_DIR = os.path.join(BASE_DIR, 'artifacts')


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class BioDataset(Dataset):
    def __init__(self, df, name="Dataset"):
        self.df = df.reset_index(drop=True)
        self.featurizer = GraphFeaturizer()
        self.cached_data = []

        print(f"[{name}] Pre-computing graphs into RAM...")
        # Note: For datasets >1M pairs, move this to __getitem__ (lazy loading)
        for _, row in tqdm(self.df.iterrows(), total=len(self.df)):
            try:
                data_a = self.featurizer.smiles_to_graph(row['smiles_a'])
                data_b = self.featurizer.smiles_to_graph(row['smiles_b'])
                label = torch.tensor(row['label'], dtype=torch.float32)
                self.cached_data.append((data_a, data_b, label))
            except Exception as e:
                print(f"Skipping error row: {e}")

    def __len__(self):
        return len(self.cached_data)

    def __getitem__(self, idx):
        return self.cached_data[idx]


def run_training(args):
    set_seed(CONFIG['SEED'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available(): device = torch.device("mps")

    print(f"--- BioGuard Training v2.0 (Fixed Pipeline) ---")
    print(f"Device: {device}")

    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    # 1. Load Data (Respecting the Loader's Split)
    print("Loading TWOSIDES dataset from data_loader...")
    # This loader performs the Strict Scaffold Split and Negative Partitioning
    df = load_twosides_data()

    # 2. Filter by 'split' column (The Single Source of Truth)
    if 'split' not in df.columns:
        raise ValueError("Dataframe missing 'split' column. Check data_loader.py version.")

    print("Applying pre-computed splits...")
    train_df = df[df['split'] == 'train']
    val_df = df[df['split'] == 'val']
    test_df = df[df['split'] == 'test']

    print(f"Final Counts -> Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    if len(train_df) == 0 or len(val_df) == 0:
        raise ValueError("Empty datasets detected! Check loader logic.")

    # 3. Initialize Datasets
    train_dataset = BioDataset(train_df, name="Train")
    val_dataset = BioDataset(val_df, name="Val")

    # 4. Initialize Loaders
    train_loader = PyGDataLoader(
        train_dataset,
        batch_size=CONFIG['BATCH_SIZE'],
        shuffle=True, # Shuffle ONLY train
        num_workers=CONFIG['NUM_WORKERS'],
        pin_memory=True if torch.cuda.is_available() else False
    )

    val_loader = PyGDataLoader(
        val_dataset,
        batch_size=CONFIG['BATCH_SIZE'],
        shuffle=False,
        num_workers=CONFIG['NUM_WORKERS'],
        pin_memory=True if torch.cuda.is_available() else False
    )

    # 5. Initialize Model
    model = BioGuardGAT(
        node_dim=41,
        edge_dim=6,
        embedding_dim=128,
        heads=4
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['LEARNING_RATE'])
    criterion = nn.BCEWithLogitsLoss()

    # 6. Training Loop
    best_val_loss = float('inf')
    patience_counter = 0

    print("\nStarting training loop...")
    for epoch in range(CONFIG['EPOCHS']):
        model.train()
        train_loss = 0

        for batch_a, batch_b, batch_y in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
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

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_a, batch_b, batch_y in val_loader:
                batch_a = batch_a.to(device)
                batch_b = batch_b.to(device)
                batch_y = batch_y.to(device).unsqueeze(1)

                logits = model(batch_a, batch_b)
                val_loss += criterion(logits, batch_y).item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch + 1}: Train={avg_train_loss:.4f}, Val={avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(ARTIFACT_DIR, 'model.pt'))
            print("  -> Model saved.")
        else:
            patience_counter += 1
            print(f"  -> Patience {patience_counter}/{CONFIG['PATIENCE']}")
            if patience_counter >= CONFIG['PATIENCE']:
                print("Early stopping triggered.")
                break

    # 7. Metadata
    metadata = {
        'version': '3.0-fixed',
        'config': CONFIG,
        'model_type': 'GAT',
        'node_dim': 41,
        'split_type': 'precomputed_scaffold', # Hardcoded as we now trust the loader
        'best_val_loss': float(best_val_loss)
    }
    with open(os.path.join(ARTIFACT_DIR, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"[OK] Training complete. Artifacts in {ARTIFACT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BioGuardGAT")
    # Removed --split argument to prevent confusion.
    args = parser.parse_args()

    run_training(args)