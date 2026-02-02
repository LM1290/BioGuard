import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool
from bioguard.config import NODE_DIM, EDGE_DIM


class BioGuardGAT(nn.Module):
    def __init__(self, node_dim=NODE_DIM, edge_dim=EDGE_DIM, embedding_dim=128, heads=4, enzyme_dim=0):
        super().__init__()

        # --- 1. Graph Encoder ---
        self.conv1 = GATv2Conv(
            node_dim,
            embedding_dim,
            heads=heads,
            dropout=0.4,
            edge_dim=edge_dim
        )

        self.conv2 = GATv2Conv(
            embedding_dim * heads,
            embedding_dim,
            heads=1,
            dropout=0.3,
            edge_dim=edge_dim
        )

        # --- 2. Dimension Calculation ---
        # Graph Branch: (Mean Pool + Max Pool) = 128 * 2 = 256
        self.graph_out_dim = embedding_dim * 2

        # Total Arm Dimension = Graph + Enzyme
        self.arm_dim = self.graph_out_dim + enzyme_dim

        print(f"[BioGuardGAT] Graph Dim: {self.graph_out_dim} | Enzyme Dim: {enzyme_dim} | Total Arm: {self.arm_dim}")

        # --- 3. Symmetric Interaction Head ---
        # Input: (Arm_A + Arm_B), |Arm_A - Arm_B|, (Arm_A * Arm_B) -> 3 * arm_dim
        self.classifier_input_dim = self.arm_dim * 3

        self.fc1 = nn.Linear(self.classifier_input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 64)
        self.out = nn.Linear(64, 1)

    @property
    def device(self):
        """Dynamic device property to check where the model weights are."""
        return next(self.parameters()).device

    def forward_one_arm(self, x, edge_index, edge_attr, batch, enzyme_feat):
        """
        Encodes a single drug graph + enzyme vector into a unified embedding.
        """
        # A. Graph Convolution
        x = self.conv1(x, edge_index, edge_attr=edge_attr)
        x = F.elu(x)

        x = self.conv2(x, edge_index, edge_attr=edge_attr)
        x = F.elu(x)

        # B. Pooling (Graph Representation)
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x_graph = torch.cat([x_mean, x_max], dim=1)

        # C. Late Fusion (Concatenate Biological Context)
        # enzyme_feat is [Batch, enzyme_dim]
        # x_graph is [Batch, 256]
        return torch.cat([x_graph, enzyme_feat], dim=1)

    def forward(self, data_a, data_b):
        """
        Args:
            data_a: Batch object for Drug A (includes enzyme_a)
            data_b: Batch object for Drug B (includes enzyme_b)
        """
        # 1. Siamese Encoding
        # Unpack explicitly for clarity
        vec_a = self.forward_one_arm(
            data_a.x,
            data_a.edge_index,
            data_a.edge_attr,
            data_a.batch,
            data_a.enzyme_a
        )

        vec_b = self.forward_one_arm(
            data_b.x,
            data_b.edge_index,
            data_b.edge_attr,
            data_b.batch,
            data_b.enzyme_b
        )

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