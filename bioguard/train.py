import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.data import Dataset, Batch
from sklearn.metrics import precision_score, recall_score
import argparse
import os
import json
import numpy as np
from tqdm import tqdm

# Internal Imports
from bioguard.data_loader import load_twosides_data
from bioguard.featurizer import GraphFeaturizer
from bioguard.model import BioGuardGAT

# --- CONFIG ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACT_DIR = os.path.join(BASE_DIR, 'artifacts')
os.makedirs(ARTIFACT_DIR, exist_ok=True)


class BioDataset(Dataset):
    def __init__(self, df, name='Train'):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.featurizer = GraphFeaturizer()
        self.name = name

        # WARNING: TECHNICAL DEBT BOMB ----------------------------------------
        # This implementation pre-computes all graphs into RAM.
        # For large datasets (like full DrugBank), this WILL cause an OOM crash.
        # TODO: Refactor to on-disk lazy loading (torch_geometric.data.Dataset)
        # ---------------------------------------------------------------------
        print(f"[{name}] Pre-computing graphs into RAM...")
        self.graphs_a = []
        self.graphs_b = []
        self.labels = []

        for _, row in tqdm(self.df.iterrows(), total=len(df)):
            g1 = self.featurizer.smiles_to_graph(row['smiles_a'])
            g2 = self.featurizer.smiles_to_graph(row['smiles_b'])

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

    # num_workers=0 is safer for debugging; increase if stable
    train_loader = PyGDataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = PyGDataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # 3. Model Setup
    sample_graph = train_dataset[0][0]
    node_dim = sample_graph.x.shape[1]

    # Save metadata for API/Eval
    with open(os.path.join(ARTIFACT_DIR, 'metadata.json'), 'w') as f:
        json.dump({'node_dim': node_dim}, f)

    model = BioGuardGAT(node_dim=node_dim, edge_dim=6).to(device)

    # --- WEIGHTED LOSS CALCULATION ---
    num_pos = train_dataset.labels.sum().item()
    num_neg = len(train_dataset) - num_pos
    pos_weight_val = num_neg / num_pos if num_pos > 0 else 1.0

    print(f"Loss Weighting: Neg={num_neg}, Pos={num_pos}, Ratio={pos_weight_val:.2f}")
    pos_weight_tensor = torch.tensor(pos_weight_val).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

    # FIX: Removed 'verbose=True' which causes TypeError in newer PyTorch
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

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
            batch_y = batch_y.to(device).unsqueeze(1)

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
            for batch_a, batch_b, batch_y in val_loader:
                batch_a = batch_a.to(device)
                batch_b = batch_b.to(device)
                batch_y = batch_y.to(device).unsqueeze(1)

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
