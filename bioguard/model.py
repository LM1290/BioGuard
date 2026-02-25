import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool
from bioguard.config import NODE_DIM, EDGE_DIM


class BioGuardGAT(nn.Module):
    def __init__(self, node_dim=NODE_DIM, edge_dim=EDGE_DIM, embedding_dim=128, heads=4, enzyme_dim=0):
        super().__init__()

        # Atom Encoder
        self.atom_encoder = nn.Sequential(
            nn.Linear(node_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU()
        )

        # --- 2. Metabolic Injection Layer ---
        # Projects biological context to the SAME latent space as the atoms.
        # R^Enzyme -> R^128
        if enzyme_dim > 0:
            self.enzyme_projector = nn.Sequential(
                nn.Linear(enzyme_dim, embedding_dim),
                nn.BatchNorm1d(embedding_dim),
                nn.ReLU(),
                nn.Linear(embedding_dim, embedding_dim)  # Match atom embedding size
            )
        else:
            self.enzyme_projector = None

        # --- 3. Graph Encoder (Context-Aware) ---
        # Input is now embedding_dim (128), NOT node_dim (41).
        self.conv1 = GATv2Conv(
            embedding_dim,  # <--- UPDATED INPUT DIM
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

        # --- 4. Dimension Calculation ---
        self.graph_out_dim = embedding_dim * 2
        self.arm_dim = self.graph_out_dim

        print(f"[BioGuardGAT] Latent Injection Enabled. Graph Dim: {self.graph_out_dim}")

        # --- 5. Symmetric Interaction Head ---
        self.classifier_input_dim = self.arm_dim * 3

        self.fc1 = nn.Linear(self.classifier_input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 64)
        self.out = nn.Linear(64, 1)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward_one_arm(self, x, edge_index, edge_attr, batch, enzyme_feat):
        """
        Args:
            x: Raw Atom features [Total_Nodes, 41] (Discrete/One-Hot)
            batch: Batch index [Total_Nodes]
            enzyme_feat: Enzyme vectors [Batch_Size, Enzyme_Dim]
        """

        # --- A. ENCODE ATOMS ---
        # Project sparse identity to dense meaning.
        # x becomes [Total_Nodes, 128]
        x = self.atom_encoder(x)

        # --- B. INJECT METABOLISM ---
        if self.enzyme_projector is not None:
            # 1. Project Enzyme to Latent Space
            # Shape: [Batch_Size, 128]
            enzyme_ctx = self.enzyme_projector(enzyme_feat)

            # 2. Broadcast to Nodes
            # Shape: [Total_Nodes, 128]
            enzyme_ctx_expanded = enzyme_ctx[batch]

            # 3. Inject (Residual Addition in Latent Space)
            # We are now shifting the *embedding* of the atom, not its identity.
            x = x + enzyme_ctx_expanded

        # --- C. Context-Aware Graph Attention ---
        x = self.conv1(x, edge_index, edge_attr=edge_attr)
        x = F.elu(x)

        x = self.conv2(x, edge_index, edge_attr=edge_attr)
        x = F.elu(x)

        # --- D. Pooling ---
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)

        return torch.cat([x_mean, x_max], dim=1)

    def forward(self, data_a, data_b):
        vec_a = self.forward_one_arm(
            data_a.x, data_a.edge_index, data_a.edge_attr,
            data_a.batch, data_a.enzyme_a
        )

        vec_b = self.forward_one_arm(
            data_b.x, data_b.edge_index, data_b.edge_attr,
            data_b.batch, data_b.enzyme_b
        )

        diff = torch.abs(vec_a - vec_b)
        combined = torch.cat([vec_a + vec_b, diff, vec_a * vec_b], dim=1)

        x = self.fc1(combined)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)

        x = self.fc2(x)
        x = F.relu(x)

        return self.out(x)
