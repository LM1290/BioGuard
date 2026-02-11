import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import json
import logging
import argparse
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.data import Dataset

# Internal Imports
from .model import BioGuardGAT
from .data_loader import load_twosides_data
from .enzyme import EnzymeManager
from .config import NODE_DIM, EDGE_DIM

# --- CONFIG ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACT_DIR = os.path.join(BASE_DIR, 'artifacts')
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_PATH = os.path.join(ARTIFACT_DIR, 'model.pt')
META_PATH = os.path.join(ARTIFACT_DIR, 'metadata.json')

os.makedirs(ARTIFACT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# Logger
logger = logging.getLogger(__name__)


class BioGuardDataset(Dataset):
    """
    In-Memory Dataset with RAM Caching for maximum speed.
    Stores data as tuples: (Graph_A, Graph_B, Label)
    """

    def __init__(self, root, df, split='train', transform=None, pre_transform=None):
        self.df = df.reset_index(drop=True)
        self.split = split
        # Enzyme Manager handles the biological context lookup
        self.enzyme_manager = EnzymeManager(allow_degraded=True)
        self.cached_data = []
        super().__init__(root, transform, pre_transform)

        self._process_in_ram()

    @property
    def processed_file_names(self):
        return ['not_used.pt']

    def _process_in_ram(self):
        """
        Converts the dataframe into a list of PyG Data objects stored in self.cached_data
        """
        # Lazy import to avoid circular dependencies
        from .featurizer import drug_to_graph

        print(f"[{self.split}] Processing {len(self.df)} graphs (Parallelized)...")

        data_list = []

        # We iterate and create graph objects once, keeping them in RAM.
        for idx, row in tqdm(self.df.iterrows(), total=len(self.df)):
            # Drug A
            g_a = drug_to_graph(row['smiles_a'], row['drug_a'])
            g_a.enzyme_a = torch.tensor([self.enzyme_manager.get_vector(row['drug_a'])], dtype=torch.float)

            # Drug B
            g_b = drug_to_graph(row['smiles_b'], row['drug_b'])
            g_b.enzyme_b = torch.tensor([self.enzyme_manager.get_vector(row['drug_b'])], dtype=torch.float)

            label = torch.tensor([float(row['label'])], dtype=torch.float)

            # Store as tuple (Lightweight)
            data_list.append((g_a, g_b, label))

        self.cached_data = data_list
        print(f"[{self.split}] Successfully processed {len(data_list)} pairs.")

    def len(self):
        return len(self.df)

    def get(self, idx):
        return self.cached_data[idx]


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for batch_data in loader:
        # Unpack tuple from DataLoader
        batch_a = batch_data[0].to(device)
        batch_b = batch_data[1].to(device)
        batch_y = batch_data[2].to(device)  # Shape [Batch, 1]

        optimizer.zero_grad()

        # Forward Pass
        logits = model(batch_a, batch_b)
        loss = criterion(logits, batch_y)

        # Backward Pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    y_true = []
    y_probs = []

    with torch.no_grad():
        for batch_data in loader:
            batch_a = batch_data[0].to(device)
            batch_b = batch_data[1].to(device)
            batch_y = batch_data[2].to(device)

            logits = model(batch_a, batch_b)
            loss = criterion(logits, batch_y)
            total_loss += loss.item()

            probs = torch.sigmoid(logits)
            y_true.extend(batch_y.cpu().numpy().flatten())
            y_probs.extend(probs.cpu().numpy().flatten())

    y_true = np.array(y_true)
    y_probs = np.array(y_probs)

    # METRICS
    # AUPRC is the primary metric for imbalanced classification
    auprc = average_precision_score(y_true, y_probs)
    roc_auc = roc_auc_score(y_true, y_probs)

    # Helper F1 for logging (using 0.5 threshold)
    y_pred = (y_probs >= 0.5).astype(int)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    return total_loss / len(loader), auprc, roc_auc, f1


def run_training(args):
    # 1. Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 2. Load Data
    df = load_twosides_data(split_method=args.split)

    train_df = df[df['split'] == 'train']
    val_df = df[df['split'] == 'val']

    # 3. Create Datasets
    train_dataset = BioGuardDataset(root=DATA_DIR, df=train_df, split='train')
    val_dataset = BioGuardDataset(root=DATA_DIR, df=val_df, split='val')

    # 4. DataLoaders
    # num_workers=0 is crucial for in-memory datasets to avoid fork overhead/copying
    train_loader = PyGDataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = PyGDataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # 5. Model Initialization
    enzyme_manager = EnzymeManager(allow_degraded=True)
    enzyme_dim = enzyme_manager.vector_dim

    print(f"Initializing BioGuardGAT with Enzyme Dim: {enzyme_dim}")

    model = BioGuardGAT(
        node_dim=NODE_DIM,
        edge_dim=EDGE_DIM,
        enzyme_dim=enzyme_dim
    ).to(device)

    # --- PRE-TRAINING INTEGRATION ---
    pretrain_path = os.path.join(ARTIFACT_DIR, 'gat_encoder_weights.pt')
    if os.path.exists(pretrain_path):
        print(f"Loading pretrained encoder from {pretrain_path}...")
        try:
            encoder_weights = torch.load(pretrain_path, map_location=device)
            # Load weights into the first GAT layer (encoder)
            model.conv1.load_state_dict(encoder_weights, strict=True)
            print("Pretrained weights loaded successfully! (Transfer Learning Activated)")
        except Exception as e:
            print(f"WARNING: Could not load pretrained weights: {e}")
            print("Continuing with random initialization...")
    else:
        print("No pretrained weights found. Training from scratch.")
    # --------------------------------

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    # 6. Training Loop
    print("\nStarting training loop (Optimizing for AUPRC)...")

    best_val_auprc = -1.0
    patience = 8
    counter = 0

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_auprc, val_roc, val_f1 = validate(model, val_loader, criterion, device)

        print(f"Epoch {epoch}: Train Loss={train_loss:.4f} | Val Loss={val_loss:.4f}")
        print(f"    -> Val AUPRC: {val_auprc:.4f} | ROC-AUC: {val_roc:.4f} | F1: {val_f1:.4f}")

        # TARGET: Maximize AUPRC
        if val_auprc > best_val_auprc:
            best_val_auprc = val_auprc
            counter = 0
            torch.save(model.state_dict(), MODEL_PATH)

            # Save Metadata
            meta = {
                'node_dim': NODE_DIM,
                'edge_dim': EDGE_DIM,
                'enzyme_dim': enzyme_dim,
                'split_type': args.split,
                'best_epoch': epoch,
                'val_auprc': val_auprc,
                'val_roc': val_roc,
                'val_loss': val_loss
            }
            with open(META_PATH, 'w') as f:
                json.dump(meta, f)

            print("    -> Model saved (New Best AUPRC).")
        else:
            counter += 1
            print(f"    -> Patience {counter}/{patience}")
            if counter >= patience:
                print("Early stopping triggered.")
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BioGuard Training")
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--split', type=str, default='random', choices=['random', 'cold_drug'], help='Split method')

    args = parser.parse_args()
    run_training(args)