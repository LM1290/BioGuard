import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool


class BioGuardGAT(nn.Module):
    # UPDATED: node_dim bumped to 41 to account for expanded atoms + chirality
    def __init__(self, node_dim=41, edge_dim=6, embedding_dim=128, heads=4):
        super().__init__()

        # --- 1. Graph Encoder ---
        # Layer 1: Takes node features + edge features
        self.conv1 = GATv2Conv(
            node_dim,
            embedding_dim,
            heads=heads,
            dropout=0.2,
            edge_dim=edge_dim
        )

        # Layer 2: Refines embeddings
        self.conv2 = GATv2Conv(
            embedding_dim * heads,
            embedding_dim,
            heads=1,
            dropout=0.2,
            edge_dim=edge_dim
        )

        # --- 2. Symmetric Interaction Head ---
        self.classifier_input_dim = embedding_dim * 3

        self.fc1 = nn.Linear(self.classifier_input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 64)
        self.out = nn.Linear(64, 1)

    def forward_one_arm(self, data):
        """Encodes a single drug graph into a vector."""
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        x = self.conv1(x, edge_index, edge_attr=edge_attr)
        x = F.elu(x)

        x = self.conv2(x, edge_index, edge_attr=edge_attr)
        x = F.elu(x)

        x = global_mean_pool(x, batch)
        return x

    def forward(self, data_a, data_b):
        """
        Args:
            data_a: Batch of Drug A graphs
            data_b: Batch of Drug B graphs
        """
        # 1. Siamese Encoding
        vec_a = self.forward_one_arm(data_a)
        vec_b = self.forward_one_arm(data_b)

        # 2. Symmetric Combination
        diff = torch.abs(vec_a - vec_b)
        combined = torch.cat([vec_a + vec_b, diff, vec_a * vec_b], dim=1)

        # 3. Classification
        x = self.fc1(combined)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)

        x = self.fc2(x)
        x = F.relu(x)

        return self.out(x)