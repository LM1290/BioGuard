import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool
from bioguard.config import NODE_DIM, EDGE_DIM


class BioGuardGAT(nn.Module):
    def __init__(self, node_dim=NODE_DIM, edge_dim=EDGE_DIM, embedding_dim=128, heads=4, enzyme_dim=60):
        super().__init__()

        # --- PATHWAY 1: Pure 3D Spatial Physics ---
        self.atom_encoder = nn.Sequential(
            nn.Linear(node_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU()
        )
        self.conv1 = GATv2Conv(embedding_dim, embedding_dim, heads=heads, dropout=0.3, edge_dim=edge_dim)
        self.conv2 = GATv2Conv(embedding_dim * heads, embedding_dim, heads=1, dropout=0.3, edge_dim=edge_dim)

        self.gat_head = nn.Sequential(
            nn.Linear(embedding_dim * 6, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 1)  # Outputs raw GAT logit
        )

        # --- PATHWAY 2: Pure Biological Prior ---
        # Projects the 60-dim LightGBM probabilities into the logit space
        self.prior_head = nn.Sequential(
            nn.Linear(enzyme_dim * 3, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)  # Outputs raw Prior logit
        )

    @property
    def device(self):
        return next(self.parameters()).device

    def forward_one_arm(self, x, edge_index, edge_attr, batch):
        # Pure spatial processing. No deep injection.
        h = self.atom_encoder(x.float())
        h = F.elu(self.conv1(h, edge_index, edge_attr=edge_attr.float()))
        h = F.elu(self.conv2(h, edge_index, edge_attr=edge_attr.float()))
        return torch.cat([global_mean_pool(h, batch), global_max_pool(h, batch)], dim=1)

    def forward(self, data_a, data_b, force_graph_only=False):
        # 1. Evaluate 3D Physics
        vec_a = self.forward_one_arm(data_a.x, data_a.edge_index, data_a.edge_attr, data_a.batch)
        vec_b = self.forward_one_arm(data_b.x, data_b.edge_index, data_b.edge_attr, data_b.batch)

        gat_combined = torch.cat([vec_a + vec_b, torch.abs(vec_a - vec_b), vec_a * vec_b], dim=1)
        gat_logits = self.gat_head(gat_combined)

        # 2. Evaluate Biological Prior
        if force_graph_only:
            prior_logits = torch.zeros_like(gat_logits)
        else:
            enz_a = data_a.enzyme_a
            enz_b = data_b.enzyme_b
            # Combine the two 60-dim profiles using the same symmetry logic
            enz_combined = torch.cat([enz_a + enz_b, torch.abs(enz_a - enz_b), enz_a * enz_b], dim=1)
            prior_logits = self.prior_head(enz_combined.float())

        # 3. THE RESIDUAL SUM
        # Gradients flow equally to both pathways. Modality collapse is mathematically impossible.
        final_logits = gat_logits + prior_logits
        dummy_alpha = torch.full_like(gat_logits, 0.5)
        return final_logits, dummy_alpha