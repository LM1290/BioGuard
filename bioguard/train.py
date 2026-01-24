"""
Training module for BioGuardGAT DDI prediction model.
UPDATED: v1.2: Bemis-Murcko Support
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
import argparse  # <--- Added for CLI
from collections import defaultdict
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# RDKit Scaffolding
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from torch_geometric.loader import DataLoader as PyGDataLoader

from .data_loader import load_twosides_data
from .featurizer import GraphFeaturizer
from .model import BioGuardGAT

# Default Config (can be overridden by CLI)
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
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --- 1. PAIR SPLIT (Baseline/Weak) ---
def get_pair_disjoint_split(df, val_size=0.1, test_size=0.1, seed=42):
    print("Performing Pair Disjoint Split (Weak Baseline)...")
    train_val, test = train_test_split(df, test_size=test_size, random_state=seed)
    train, val = train_test_split(train_val, test_size=val_size / (1 - test_size), random_state=seed)
    return train, val, test


# --- 2. COLD DRUG SPLIT (Strict) ---
def get_cold_drug_split(df, val_size=0.1, test_size=0.1, seed=42):
    """
    Splits data by DRUG (scaffold), not by PAIR.
    Ensures that drugs in Val/Test have NEVER been seen in Train.
    """
    print("Performing Cold Drug Split (Strict Mode)...")

    unique_drugs = list(set(df['smiles_a']) | set(df['smiles_b']))
    train_drugs, temp_drugs = train_test_split(unique_drugs, test_size=(test_size + val_size), random_state=seed)
    val_drugs, test_drugs = train_test_split(temp_drugs, test_size=(test_size / (test_size + val_size)),
                                             random_state=seed)

    train_set, val_set, test_set = set(train_drugs), set(val_drugs), set(test_drugs)

    # Filter: Both drugs must be in the respective set
    train_df = df[df['smiles_a'].isin(train_set) & df['smiles_b'].isin(train_set)].copy()
    val_df = df[df['smiles_a'].isin(val_set) & df['smiles_b'].isin(val_set)].copy()
    test_df = df[df['smiles_a'].isin(test_set) & df['smiles_b'].isin(test_set)].copy()

    return train_df, val_df, test_df


# --- 3. BEMIS-MURCKO SCAFFOLD SPLIT (The Nuclear Option) ---
def generate_scaffold(smiles, include_chirality=False):
    """Compute the Bemis-Murcko scaffold for a SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return None
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)
    return scaffold


def get_scaffold_split(df, val_size=0.1, test_size=0.1, seed=42):
    """
    Splits drugs based on their core chemical scaffold.
    This tests if the model can generalize to completely new chemical families.
    """
    print("Performing Bemis-Murcko Scaffold Split (Nuclear Option)...")

    unique_drugs = list(set(df['smiles_a']) | set(df['smiles_b']))
    scaffold_to_drugs = defaultdict(list)

    print("Generating scaffolds...")
    for smiles in tqdm(unique_drugs):
        scaffold = generate_scaffold(smiles)
        if scaffold is not None:
            scaffold_to_drugs[scaffold].append(smiles)

    # Split the SCAFFOLDS (not the drugs)
    scaffolds = list(scaffold_to_drugs.keys())
    train_scaff, temp_scaff = train_test_split(scaffolds, test_size=(test_size + val_size), random_state=seed)
    val_scaff, test_scaff = train_test_split(temp_scaff, test_size=(test_size / (test_size + val_size)),
                                             random_state=seed)

    # Map back to drugs
    train_drugs = set([d for s in train_scaff for d in scaffold_to_drugs[s]])
    val_drugs = set([d for s in val_scaff for d in scaffold_to_drugs[s]])
    test_drugs = set([d for s in test_scaff for d in scaffold_to_drugs[s]])

    print(f"Scaffolds -> Train: {len(train_scaff)}, Val: {len(val_scaff)}, Test: {len(test_scaff)}")

    # Filter Dataframe (Both drugs must belong to the split's scaffold set)
    train_df = df[df['smiles_a'].isin(train_drugs) & df['smiles_b'].isin(train_drugs)].copy()
    val_df = df[df['smiles_a'].isin(val_drugs) & df['smiles_b'].isin(val_drugs)].copy()
    test_df = df[df['smiles_a'].isin(test_drugs) & df['smiles_b'].isin(test_drugs)].copy()

    return train_df, val_df, test_df


class BioDataset(Dataset):
    def __init__(self, df, name="Dataset"):
        self.df = df.reset_index(drop=True)
        self.featurizer = GraphFeaturizer()
        self.cached_data = []

        print(f"[{name}] Pre-computing graphs into RAM...")
        for _, row in tqdm(self.df.iterrows(), total=len(self.df)):
            data_a = self.featurizer.smiles_to_graph(row['smiles_a'])
            data_b = self.featurizer.smiles_to_graph(row['smiles_b'])
            label = torch.tensor(row['label'], dtype=torch.float32)
            self.cached_data.append((data_a, data_b, label))

    def __len__(self):
        return len(self.cached_data)

    def __getitem__(self, idx):
        return self.cached_data[idx]


def run_training(args):
    set_seed(CONFIG['SEED'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available(): device = torch.device("mps")

    print(f"--- BioGuard Training v3.0 ---")
    print(f"Device: {device}")
    print(f"Split Strategy: {args.split.upper()}")

    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    # 1. Load Data
    print("Loading TWOSIDES dataset...")
    df = load_twosides_data()

    # 2. Apply Selected Split
    if args.split == 'random':
        train_df, val_df, test_df = get_pair_disjoint_split(df)
    elif args.split == 'cold':
        train_df, val_df, test_df = get_cold_drug_split(df)
    elif args.split == 'scaffold':
        train_df, val_df, test_df = get_scaffold_split(df)
    else:
        raise ValueError(f"Unknown split type: {args.split}")

    print(f"Final Counts -> Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    if len(train_df) == 0 or len(val_df) == 0:
        raise ValueError("Selected split resulted in empty datasets. Dataset too sparse!")

    # 3. Initialize Datasets
    train_dataset = BioDataset(train_df, name="Train")
    val_dataset = BioDataset(val_df, name="Val")

    train_loader = PyGDataLoader(train_dataset, batch_size=CONFIG['BATCH_SIZE'], shuffle=True,
                                 num_workers=CONFIG['NUM_WORKERS'])
    val_loader = PyGDataLoader(val_dataset, batch_size=CONFIG['BATCH_SIZE'], shuffle=False,
                               num_workers=CONFIG['NUM_WORKERS'])

    # 4. Initialize Model
    # node_dim=41 (from v2 update), edge_dim=6
    model = BioGuardGAT(
        node_dim=41,
        edge_dim=6,
        embedding_dim=128,
        heads=4
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['LEARNING_RATE'])
    criterion = nn.BCEWithLogitsLoss()

    # 5. Training Loop
    best_val_loss = float('inf')
    patience_counter = 0

    print("\nStarting training loop...")
    for epoch in range(CONFIG['EPOCHS']):
        model.train()
        train_loss = 0

        for batch_a, batch_b, batch_y in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
            batch_a, batch_b, batch_y = batch_a.to(device), batch_b.to(device), batch_y.to(device).unsqueeze(1)

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
                batch_a, batch_b, batch_y = batch_a.to(device), batch_b.to(device), batch_y.to(device).unsqueeze(1)
                logits = model(batch_a, batch_b)
                val_loss += criterion(logits, batch_y).item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch + 1}: Train={avg_train_loss:.4f}, Val={avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(ARTIFACT_DIR, 'model.pt'))
        else:
            patience_counter += 1
            if patience_counter >= CONFIG['PATIENCE']:
                print("Early stopping.")
                break

    # 6. Metadata
    metadata = {
        'version': f'3.0-{args.split}',
        'config': CONFIG,
        'model_type': 'GAT',
        'node_dim': 41,
        'split_type': args.split,
        'best_val_loss': float(best_val_loss)
    }
    with open(os.path.join(ARTIFACT_DIR, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"[OK] Model artifacts saved to {ARTIFACT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BioGuardGAT")
    parser.add_argument('--split', type=str, default='cold', choices=['random', 'cold', 'scaffold'],
                        help="Data split strategy: 'random' (easy), 'cold' (hard), 'scaffold' (generalize to new chemical families)")
    args = parser.parse_args()

    run_training(args)