import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import json
import logging
import argparse
import lmdb
import io
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.data import Dataset

# Internal Imports
from .model import BioGuardGAT
from .data_loader import load_twosides_data
from .enzyme import EnzymeManager
from .config import NODE_DIM, EDGE_DIM, LMDB_DIR, NUM_WORKERS

# --- CONFIG ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACT_DIR = os.path.join(BASE_DIR, 'artifacts')
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_PATH = os.path.join(ARTIFACT_DIR, 'model.pt')
META_PATH = os.path.join(ARTIFACT_DIR, 'metadata.json')

os.makedirs(ARTIFACT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

logger = logging.getLogger(__name__)


class BioFocalLoss(nn.Module):
    def __init__(self, alpha=0.95, gamma=2.0, reduction='mean'):
        """
        Numerically Stabilized Focal Loss for BioGuard V3.0.
        Fixes NaN issues by clamping logits and adding epsilon stability.
        """
        super(BioFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.eps = 1e-8

    def forward(self, inputs, targets):
        inputs = torch.clamp(inputs, min=-10.0, max=10.0)

        # 2. STABLE BCE: Use the built-in Log-Sum-Exp version
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

        # 3. PROBABILITY CALC: Get pt (probability of correct class)
        # Using sigmoid(inputs) for the probability estimation
        p = torch.sigmoid(inputs)
        pt = targets * p + (1 - targets) * (1 - p)

        # 4. FOCAL TERM: Apply alpha and the (1-pt)^gamma modulation
        # Add epsilon to prevent 0^gamma if pt is exactly 1.0
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        focal_loss = alpha_t * (1 - pt + self.eps) ** self.gamma * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class BioGuardDataset(Dataset):
    """
    Production LMDB-Backed Dataset.
    Strictly utilizes the OS page cache via LMDB and native PyTorch
    tensor decoding. Safe for massive datasets and multiprocess workers.
    """

    def __init__(self, root, df, split='train', transform=None, pre_transform=None):
        self.df = df.reset_index(drop=True)
        self.split = split
        self.lmdb_path = LMDB_DIR
        self.enzyme_manager = EnzymeManager(allow_degraded=True)

        # Must be initialized lazily to prevent deadlocks across DataLoader workers
        self.env = None

        super().__init__(root, transform, pre_transform)

    @property
    def processed_file_names(self):
        return ['not_used.pt']

    def _init_db(self):
        """
        Initializes the LMDB environment inside the worker process.
        readahead=False is critical for random access performance in ML.
        """
        self.env = lmdb.open(
            self.lmdb_path,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False
        )

    def len(self):
        return len(self.df)

    def get(self, idx):
        # Lazy DB initialization for num_workers > 0
        if self.env is None:
            self._init_db()

        row = self.df.iloc[idx]
        smi_a = row['smiles_a']
        smi_b = row['smiles_b']

        # 1. True LMDB Access (Hitting the OS Page Cache)
        with self.env.begin(write=False) as txn:
            raw_a = txn.get(smi_a.encode('utf-8'))
            raw_b = txn.get(smi_b.encode('utf-8'))

            if raw_a is None or raw_b is None:
                raise KeyError(f"Missing SMILES in LMDB: {smi_a if raw_a is None else smi_b}")

            # 2. Native PyTorch Deserialization
            # weights_only=False is required for loading custom PyG Data objects
            data_a = torch.load(io.BytesIO(raw_a), weights_only=False)
            data_b = torch.load(io.BytesIO(raw_b), weights_only=False)

        # 3. Dynamic Enzyme Inference (100% flat RAM usage)
        # We use the LightGBM predictor via get_by_smiles to ensure NCEs are scored correctly
        data_a.enzyme_a = torch.tensor([self.enzyme_manager.get_by_smiles(smi_a)], dtype=torch.float)
        data_b.enzyme_b = torch.tensor([self.enzyme_manager.get_by_smiles(smi_b)], dtype=torch.float)

        label = torch.tensor([float(row['label'])], dtype=torch.float)

        return data_a, data_b, label

def train_epoch(model, loader, optimizer, criterion, device, epoch_num, run_profiler=False):
    model.train()
    total_loss = 0

    # TQDM Progress Bar
    pbar = tqdm(loader, desc=f"Epoch {epoch_num} [Train]")

    if run_profiler:
        # Profiler Setup: Waits 1 step, warms up 1 step, records 3 steps
        prof = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(os.path.join(ARTIFACT_DIR, 'profiler_logs')),
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        )
        prof.start()

    for batch_data in pbar:
        batch_a = batch_data[0].to(device)
        batch_b = batch_data[1].to(device)
        batch_y = batch_data[2].to(device)

        optimizer.zero_grad()
        logits = model(batch_a, batch_b)
        loss = criterion(logits, batch_y)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

        # Update progress bar with the current loss
        pbar.set_postfix(loss=f"{loss.item():.4f}")

        if run_profiler:
            prof.step()

    if run_profiler:
        prof.stop()
        print(f"\n[Profiler Output Saved to {ARTIFACT_DIR}/profiler_logs]")

    return total_loss / len(loader)


def validate(model, loader, criterion, device, epoch_num):
    model.eval()
    total_loss = 0
    y_true = []
    y_probs = []

    with torch.no_grad():
        for batch_data in tqdm(loader, desc=f"Epoch {epoch_num} [Val  ]"):
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

    auprc = average_precision_score(y_true, y_probs)
    roc_auc = roc_auc_score(y_true, y_probs)

    y_pred = (y_probs >= 0.5).astype(int)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    return total_loss / len(loader), auprc, roc_auc, f1


def run_training(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    df = load_twosides_data(split_method=args.split)
    train_df = df[df['split'] == 'train']
    val_df = df[df['split'] == 'val']

    train_dataset = BioGuardDataset(root=DATA_DIR, df=train_df, split='train')
    val_dataset = BioGuardDataset(root=DATA_DIR, df=val_df, split='val')

    train_loader = PyGDataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=NUM_WORKERS,drop_last=True)
    val_loader = PyGDataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=NUM_WORKERS)

    enzyme_manager = EnzymeManager(allow_degraded=True)
    enzyme_dim = enzyme_manager.vector_dim

    print(f"Initializing BioGuardGAT with Enzyme Dim: {enzyme_dim}")
    print(f"Arch: Node={NODE_DIM}, Emb={args.embedding_dim}, Heads={args.heads}")

    model = BioGuardGAT(
        node_dim=NODE_DIM,
        edge_dim=EDGE_DIM,
        embedding_dim=args.embedding_dim,
        heads=args.heads,
        enzyme_dim=enzyme_dim
    ).to(device)

    pretrain_path = os.path.join(ARTIFACT_DIR, 'gat_encoder_weights.pt')
    if os.path.exists(pretrain_path):
        print(f"Loading pretrained encoder from {pretrain_path}...")
        try:
            checkpoint = torch.load(pretrain_path, map_location=device)
            if isinstance(checkpoint, dict) and 'atom_encoder' in checkpoint:
                print("Detected Metabolic Injection Checkpoint (v2). Loading AtomEncoder + GAT.")
                model.atom_encoder.load_state_dict(checkpoint['atom_encoder'])
                model.conv1.load_state_dict(checkpoint['conv1'])
            elif isinstance(checkpoint, dict) and 'conv1' not in checkpoint:
                print("WARNING: Detected Legacy Checkpoint (v1). Weights may mismatch new architecture.")
                try:
                    model.conv1.load_state_dict(checkpoint, strict=False)
                except:
                    pass
            print("Pretrained weights loaded successfully! (Transfer Learning Activated)")
        except Exception as e:
            print(f"WARNING: Could not load pretrained weights: {e}")
    else:
        print("No pretrained weights found. Training from scratch.")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = BioFocalLoss(alpha=0.95, gamma=2.0)

    print("\nStarting training loop (Optimizing for AUPRC)...")
    best_val_auprc = -1.0
    counter = 0

    for epoch in range(1, args.epochs + 1):
        # Run profiler ONLY on the first epoch to diagnose bottlenecks
        run_prof = (epoch == 1)

        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, epoch, run_profiler=run_prof)
        val_loss, val_auprc, val_roc, val_f1 = validate(model, val_loader, criterion, device, epoch)

        print(f"Epoch {epoch}: Train Loss={train_loss:.4f} | Val Loss={val_loss:.4f}")
        print(f"    -> Val AUPRC: {val_auprc:.4f} | ROC-AUC: {val_roc:.4f} | F1: {val_f1:.4f}")

        if val_auprc > best_val_auprc:
            best_val_auprc = val_auprc
            counter = 0
            torch.save(model.state_dict(), MODEL_PATH)

            meta = {
                'node_dim': NODE_DIM,
                'edge_dim': EDGE_DIM,
                'enzyme_dim': enzyme_dim,
                'embedding_dim': args.embedding_dim,
                'heads': args.heads,
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
            print(f"    -> Patience {counter}/{args.patience}")
            if counter >= args.patience:
                print("Early stopping triggered.")
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BioGuard Training")
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--patience', type=int, default=8, help='Early stopping patience')
    parser.add_argument('--embedding_dim', type=int, default=128, help='Latent dimension for atoms')
    parser.add_argument('--heads', type=int, default=4, help='Number of GAT attention heads')
    parser.add_argument('--split', type=str, default='random', choices=['random', 'scaffold', 'cold_drug'],
                        help='Split method')

    args = parser.parse_args()
    run_training(args)