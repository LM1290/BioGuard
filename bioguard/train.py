import torch
import torch.nn as nn
import torch.optim as optim
# Use the correct Dataset class
from torch_geometric.data import Dataset as PyGDataset
from torch_geometric.loader import DataLoader as PyGDataLoader
from sklearn.metrics import precision_score, recall_score
import argparse
import os
import os.path as osp
import json
import numpy as np
from tqdm import tqdm
import pandas as pd

# Internal Imports
from bioguard.data_loader import load_twosides_data
from bioguard.featurizer import GraphFeaturizer
from bioguard.model import BioGuardGAT

# --- CONFIG ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACT_DIR = os.path.join(BASE_DIR, 'artifacts')
DATA_DIR = os.path.join(BASE_DIR, 'data')
os.makedirs(ARTIFACT_DIR, exist_ok=True)


class BioGuardDataset(PyGDataset):
    """
    On-Disk Dataset to prevent RAM explosion.
    Featurizes once, saves .pt files, loads lazily.
    """

    def __init__(self, root, df=None, split='train', transform=None, pre_transform=None):
        self.split = split
        self.df = df
        # The 'root' will contain /processed/train or /processed/val
        super().__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        # We handle raw data ingestion via DataFrame, so this can be dummy
        return []

    @property
    def processed_dir(self):
        return osp.join(self.root, 'processed', self.split)

    @property
    def processed_file_names(self):
        # We need to know how many files to expect.
        # If df is None (loading mode), we count files in dir.
        # If df is provided (creation mode), we use len(df).
        if self.df is not None:
            return [f'data_{i}.pt' for i in range(len(self.df))]
        else:
            # Fallback for reloading without DF
            files = [f for f in os.listdir(self.processed_dir) if f.startswith('data_') and f.endswith('.pt')]
            return sorted(files, key=lambda x: int(x.split('_')[1].split('.')[0]))

    def download(self):
        # Data is passed via DF, no download needed
        pass

    def process(self):
        if self.df is None:
            print(f"[{self.split}] No DataFrame provided and processing triggered. Assuming data exists.")
            return

        print(f"[{self.split}] Processing {len(self.df)} graphs to disk (One-time cost)...")
        os.makedirs(self.processed_dir, exist_ok=True)

        featurizer = GraphFeaturizer()

        for idx, row in tqdm(self.df.iterrows(), total=len(self.df)):
            g1 = featurizer.smiles_to_graph(row['smiles_a'])
            g2 = featurizer.smiles_to_graph(row['smiles_b'])

            if g1 and g2:
                # We store pairs in a generic Data object or custom dict-like
                # PyG Data object doesn't support nested Data well unless we batch carefully.
                # Better approach: Save them as a simple dictionary or tuple of Data objects
                # But PyG Dataset expects ONE Data object per file usually.
                # We will wrap them.
                data = (g1, g2, torch.tensor(row['label'], dtype=torch.float))
                torch.save(data, osp.join(self.processed_dir, f'data_{idx}.pt'))
            else:
                # Handle failure (empty graph) - simplified for brevity
                # In prod, we'd probably filter these out before loop
                pass

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, f'data_{idx}.pt'),weights_only=False)
        return data


def run_training(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. Load Data Frame (Lightweight Metadata)
    print("Loading TWOSIDES metadata...")
    df = load_twosides_data(split_method=args.split)

    train_df = df[df['split'] == 'train'].reset_index(drop=True)
    val_df = df[df['split'] == 'val'].reset_index(drop=True)

    # 2. Initialize Disk-Based Datasets
    # Pass 'root' as the data directory. The class will handle processed/train/ etc.
    train_dataset = BioGuardDataset(root=DATA_DIR, df=train_df, split='train')
    val_dataset = BioGuardDataset(root=DATA_DIR, df=val_df, split='val')

    # 3. Loaders with Multiprocessing
    # Now safe to use num_workers because we are loading files, not forking heavy objects
    num_workers = min(4, os.cpu_count())
    print(f"Data Loaders initialized with num_workers={num_workers}")

    train_loader = PyGDataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=num_workers)
    val_loader = PyGDataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers)

    # 4. Model Setup
    # Fetch dimensions from first graph of first pair
    sample_data = train_dataset[0]
    node_dim = sample_data[0].x.shape[1]

    with open(os.path.join(ARTIFACT_DIR, 'metadata.json'), 'w') as f:
        json.dump({'node_dim': node_dim}, f)

    model = BioGuardGAT(node_dim=node_dim, edge_dim=6).to(device)

    # Weighted Loss Setup
    # Note: We can't sum() the dataset labels cheaply anymore without iteration.
    # But we have the DataFrame! Use that.
    num_pos = train_df['label'].sum()
    num_neg = len(train_df) - num_pos
    pos_weight_val = num_neg / num_pos if num_pos > 0 else 1.0

    print(f"Loss Weighting: Neg={num_neg}, Pos={num_pos}, Ratio={pos_weight_val:.2f}")
    pos_weight_tensor = torch.tensor(pos_weight_val).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    # 5. Training Loop
    best_val_loss = float('inf')
    patience = 0
    max_patience = 8

    print("\nStarting training loop...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0

        for batch_data in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
            # Unpack tuple (g1, g2, label)
            # PyG DataLoader batches lists of tuples into lists of batches
            # Actually, standard PyG DataLoader collates Data objects.
            # Since we return a tuple (g1, g2, y), the loader returns (Batch_g1, Batch_g2, Batch_y)

            batch_a = batch_data[0].to(device)
            batch_b = batch_data[1].to(device)
            batch_y = batch_data[2].to(device).unsqueeze(1)

            optimizer.zero_grad()
            logits = model(batch_a, batch_b)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch_data in val_loader:
                batch_a = batch_data[0].to(device)
                batch_b = batch_data[1].to(device)
                batch_y = batch_data[2].to(device).unsqueeze(1)

                logits = model(batch_a, batch_b)
                loss = criterion(logits, batch_y)
                val_loss += loss.item()

                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()

                all_preds.extend(preds.cpu().numpy().flatten())
                all_labels.extend(batch_y.cpu().numpy().flatten())

        avg_val_loss = val_loss / len(val_loader)
        val_precision = precision_score(all_labels, all_preds, zero_division=0)
        val_recall = recall_score(all_labels, all_preds, zero_division=0)
        scheduler.step(avg_val_loss)

        print(f"Epoch {epoch}: Train Loss={avg_train_loss:.4f} | Val Loss={avg_val_loss:.4f}")
        print(f"    -> Val Precision: {val_precision:.4f} | Val Recall: {val_recall:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(ARTIFACT_DIR, 'model.pt'))
            print("    -> Model saved.")
            patience = 0
        else:
            patience += 1
            print(f"    -> Patience {patience}/{max_patience}")
            if patience >= max_patience:
                print("Early stopping triggered.")
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--split', type=str, default='scaffold')
    args = parser.parse_args()
    run_training(args)