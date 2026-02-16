import torch
import torch.nn as nn
import torch.optim as optim
import os
import argparse
from tqdm import tqdm
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.nn import GATv2Conv

# Internal Imports
from .data_loader import load_twosides_data
from .train import BioGuardDataset
from .config import NODE_DIM, EDGE_DIM

# --- CONFIG ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACT_DIR = os.path.join(BASE_DIR, 'artifacts')
PRETRAIN_PATH = os.path.join(ARTIFACT_DIR, 'gat_encoder_weights.pt')
os.makedirs(ARTIFACT_DIR, exist_ok=True)


class BioGuardPretrain(nn.Module):
    """
    Wrapper model that matches the NEW BioGuardGAT structure.
    Structure: AtomEncoder -> GATv2Conv -> Linear (Predict Masked Atom)
    """

    def __init__(self, node_dim=NODE_DIM, edge_dim=EDGE_DIM, embedding_dim=128, heads=4):
        super().__init__()

        # 1. Atom Encoder (MUST MATCH BioGuardGAT)
        # R^41 -> R^128
        self.atom_encoder = nn.Sequential(
            nn.Linear(node_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU()
        )

        # 2. GAT Layer
        # Input is now embedding_dim (128), NOT node_dim
        self.conv1 = GATv2Conv(
            embedding_dim,
            embedding_dim,
            heads=heads,
            dropout=0.4,
            edge_dim=edge_dim
        )

        # 3. Decoder: Predicts Atom Type from the Graph Embedding
        # Input: Output of Conv1 (embedding_dim * heads)
        # Output: Original Atom Types (24 classes for C, N, O, etc.)
        self.lin_pred = nn.Linear(embedding_dim * heads, 24)

    def forward(self, x, edge_index, edge_attr):
        # 1. Project Raw Features
        x = self.atom_encoder(x)

        # 2. Convolve
        x = self.conv1(x, edge_index, edge_attr=edge_attr)

        # 3. Predict Identity
        return self.lin_pred(x)


def mask_nodes(batch, mask_ratio=0.15):
    """
    Masks node features for self-supervised learning.
    """
    # 1. Identify True Atom Type (Label)
    # The first 24 columns are the One-Hot Atom Type
    atom_types = batch.x[:, :24].argmax(dim=1)

    # 2. Select Nodes to Mask
    num_nodes = batch.num_nodes
    mask = torch.rand(num_nodes) < mask_ratio

    # 3. Create Targets
    y_target = atom_types[mask]

    # 4. Mask Features
    # We zero out the features for the masked nodes so the model relies on neighbors
    batch.x[mask] = 0.0

    return batch, y_target, mask


def run_pretraining(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("--- 1. Loading Data for Pretraining ---")
    df = load_twosides_data(split_method='random')
    train_df = df[df['split'] == 'train']
    dataset = BioGuardDataset(root='data', df=train_df, split='pretrain')

    # Num workers 0 is safer for in-memory datasets
    loader = PyGDataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    print("--- 2. Initializing Model ---")
    model = BioGuardPretrain(node_dim=NODE_DIM, edge_dim=EDGE_DIM).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    print("--- 3. Starting Pre-training Loop ---")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0
        total_correct = 0
        total_masked = 0

        pbar = tqdm(loader, desc=f"Epoch {epoch}")
        for batch_data in pbar:
            # batch_data is (g_a, g_b, label)
            graphs = [batch_data[0], batch_data[1]]
            batch_loss = 0

            for g in graphs:
                g = g.to(device)
                g_masked, y_target, mask_idx = mask_nodes(g)

                if y_target.numel() == 0: continue

                optimizer.zero_grad()
                pred_logits = model(g_masked.x, g_masked.edge_index, g_masked.edge_attr)

                pred_masked = pred_logits[mask_idx]
                loss = criterion(pred_masked, y_target)

                loss.backward()
                optimizer.step()

                batch_loss += loss.item()

                preds = pred_masked.argmax(dim=1)
                total_correct += (preds == y_target).sum().item()
                total_masked += y_target.size(0)

            total_loss += batch_loss
            pbar.set_postfix({'loss': batch_loss})

        avg_loss = total_loss / len(loader)
        acc = total_correct / total_masked if total_masked > 0 else 0
        print(f"Epoch {epoch}: Loss={avg_loss:.4f} | Atom Prediction Acc={acc:.4f}")

    # --- SAVE LOGIC UPDATED ---
    print(f"Saving encoder weights to {PRETRAIN_PATH}...")
    # We now save a dictionary containing BOTH the atom_encoder and conv1
    state_dict = {
        'atom_encoder': model.atom_encoder.state_dict(),
        'conv1': model.conv1.state_dict()
    }
    torch.save(state_dict, PRETRAIN_PATH)
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args()
    run_pretraining(args)
