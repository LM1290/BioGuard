import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.data import Dataset, Batch
import argparse
import os
import json
import numpy as np
from tqdm import tqdm

# Internal Imports
from bioguard.data_loader import load_twosides_data
from bioguard.featurizer import BioFeaturizer
from bioguard.model import BioGuardGAT

# --- CONFIG ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACT_DIR = os.path.join(BASE_DIR, 'artifacts')
os.makedirs(ARTIFACT_DIR, exist_ok=True)

class BioDataset(Dataset):
    def __init__(self, df, name='Train'):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.featurizer = BioFeaturizer()
        self.name = name

        # Pre-compute graphs to avoid bottleneck during training
        print(f"[{name}] Pre-computing graphs into RAM...")
        self.graphs_a = []
        self.graphs_b = []
        self.labels = []

        # Simple caching using lists (Memory heavy but fast)
        for _, row in tqdm(self.df.iterrows(), total=len(df)):
            g1 = self.featurizer.featurize(row['smiles_a'])
            g2 = self.featurizer.featurize(row['smiles_b'])

            if g1 and g2:
                self.graphs_a.append(g1)
                self.graphs_b.append(g2)
                self.labels.append(row['label'])

        self.labels = torch.tensor(self.labels, dtype=torch.float)

    def len(self):
        return len(self.labels)

    def get(self, idx):
        return self.graphs_a[idx], self.graphs_b[idx], self.labels[idx]

def run_training(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. Load Data
    print("Loading TWOSIDES dataset from data_loader...")
    df = load_twosides_data(split_method=args.split)

    train_df = df[df['split'] == 'train']
    val_df = df[df['split'] == 'val']

    print(f"Final Counts -> Train: {len(train_df)} | Val: {len(val_df)}")

    # 2. Datasets & Loaders
    train_dataset = BioDataset(train_df, name="Train")
    val_dataset = BioDataset(val_df, name="Val")

    train_loader = PyGDataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0) # Workers=0 for safety
    val_loader = PyGDataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # 3. Model Setup
    # Get dimension from first valid graph
    sample_graph = train_dataset[0][0]
    node_dim = sample_graph.x.shape[1]

    # Save metadata for API/Eval
    with open(os.path.join(ARTIFACT_DIR, 'metadata.json'), 'w') as f:
        json.dump({'node_dim': node_dim}, f)

    model = BioGuardGAT(node_dim=node_dim, edge_dim=6).to(device)

    # OPTIMIZER TWEAKS (For generalization)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

    # 4. Loop
    best_val_loss = float('inf')
    patience = 0
    max_patience = 8

    print("\nStarting training loop...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0

        for batch_a, batch_b, batch_y in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
            batch_a = batch_a.to(device)
            batch_b = batch_b.to(device)
            batch_y = batch_y.to(device)

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
        with torch.no_grad():
            for batch_a, batch_b, batch_y in val_loader:
                batch_a = batch_a.to(device)
                batch_b = batch_b.to(device)
                batch_y = batch_y.to(device)

                logits = model(batch_a, batch_b)
                loss = criterion(logits, batch_y)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)

        print(f"Epoch {epoch}: Train={avg_train_loss:.4f}, Val={avg_val_loss:.4f}")

        # Checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(ARTIFACT_DIR, 'model.pt'))
            print("  -> Model saved.")
            patience = 0
        else:
            patience += 1
            print(f"  -> Patience {patience}/{max_patience}")
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