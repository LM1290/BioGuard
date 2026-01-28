import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data, InMemoryDataset
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
from bioguard.config import NODE_DIM

# --- CONFIG ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACT_DIR = os.path.join(BASE_DIR, 'artifacts')
DATA_DIR = os.path.join(BASE_DIR, 'data')
os.makedirs(ARTIFACT_DIR, exist_ok=True)


class PairData(Data):
    """
    Custom Data object to hold a pair of graphs.
    Allows PyG to batch them correctly using 'follow_batch'.
    """

    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index_b':
            return self.x_b.size(0)
        return super().__inc__(key, value, *args, **kwargs)


class BioGuardDataset(InMemoryDataset):
    """
    In-Memory Dataset that loads 100x faster by saving a single collated .pt file.
    """

    def __init__(self, root, df=None, split='train', transform=None, pre_transform=None):
        self.split = split
        self.df = df
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_dir(self):
        # Separate processed files by split to avoid overwrites
        return osp.join(self.root, 'processed', self.split)

    @property
    def processed_file_names(self):
        return ['data.pt']

    def process(self):
        if self.df is None:
            print(f"[{self.split}] No DataFrame provided and processing triggered. Assuming data exists.")
            return

        print(f"[{self.split}] Processing {len(self.df)} graphs to single file...")
        os.makedirs(self.processed_dir, exist_ok=True)

        featurizer = GraphFeaturizer()
        data_list = []

        for idx, row in tqdm(self.df.iterrows(), total=len(self.df)):
            g1 = featurizer.smiles_to_graph(row['smiles_a'])
            g2 = featurizer.smiles_to_graph(row['smiles_b'])

            if g1 and g2:
                # Combine into one PairData object
                data = PairData(
                    x=g1.x,
                    edge_index=g1.edge_index,
                    edge_attr=g1.edge_attr,
                    x_b=g2.x,
                    edge_index_b=g2.edge_index,
                    edge_attr_b=g2.edge_attr,
                    y=torch.tensor(row['label'], dtype=torch.float)
                )
                data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        # Save one big .pt file
        torch.save(self.collate(data_list), self.processed_paths[0])


def run_training(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. Load Data Frame (Lightweight Metadata)
    print("Loading TWOSIDES metadata...")
    df = load_twosides_data(split_method=args.split)

    train_df = df[df['split'] == 'train'].reset_index(drop=True)
    val_df = df[df['split'] == 'val'].reset_index(drop=True)

    # 2. Initialize In-Memory Datasets
    train_dataset = BioGuardDataset(root=DATA_DIR, df=train_df, split='train')
    val_dataset = BioGuardDataset(root=DATA_DIR, df=val_df, split='val')

    # 3. Loaders with Multiprocessing
    num_workers = min(4, os.cpu_count())
    print(f"Data Loaders initialized with num_workers={num_workers}")

    # follow_batch=['x_b'] creates 'x_b_batch' vector for the second graph
    train_loader = PyGDataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                 num_workers=num_workers, follow_batch=['x_b'])
    val_loader = PyGDataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                               num_workers=num_workers, follow_batch=['x_b'])

    # 4. Model Setup
    # Use config for dimensions
    node_dim = NODE_DIM

    with open(os.path.join(ARTIFACT_DIR, 'metadata.json'), 'w') as f:
        json.dump({'node_dim': node_dim}, f)

    model = BioGuardGAT(node_dim=node_dim, edge_dim=6).to(device)

    # Weighted Loss Setup
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

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
            batch = batch.to(device)

            # Construct Batch objects for the Siamese network
            # batch.batch is automatically created for 'x'
            # batch.x_b_batch is created because of follow_batch=['x_b']
            batch_a = Data(x=batch.x, edge_index=batch.edge_index, edge_attr=batch.edge_attr, batch=batch.batch)
            batch_b = Data(x=batch.x_b, edge_index=batch.edge_index_b, edge_attr=batch.edge_attr_b,
                           batch=batch.x_b_batch)
            batch_y = batch.y.unsqueeze(1)

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
            for batch in val_loader:
                batch = batch.to(device)

                batch_a = Data(x=batch.x, edge_index=batch.edge_index, edge_attr=batch.edge_attr, batch=batch.batch)
                batch_b = Data(x=batch.x_b, edge_index=batch.edge_index_b, edge_attr=batch.edge_attr_b,
                               batch=batch.x_b_batch)
                batch_y = batch.y.unsqueeze(1)

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