import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool
from bioguard.config import NODE_DIM, EDGE_DIM


class BioGuardGAT(nn.Module):
    def __init__(self, node_dim=NODE_DIM, edge_dim=EDGE_DIM, embedding_dim=128, heads=4, enzyme_dim=15):
        super().__init__()

        # --- 1. The Pure Graph Pathway ---
        self.atom_encoder = nn.Sequential(
            nn.Linear(node_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU()
        )

        self.conv1 = GATv2Conv(embedding_dim, embedding_dim, heads=heads, dropout=0.4, edge_dim=edge_dim)
        self.conv2 = GATv2Conv(embedding_dim * heads, embedding_dim, heads=1, dropout=0.3, edge_dim=edge_dim,
                               fill_value=0.0)

        self.graph_out_dim = embedding_dim * 2

        # The Graph-Only Prediction Head
        self.gat_head = nn.Sequential(
            nn.Linear(self.graph_out_dim * 3, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        # --- 2. The Pure Prior Pathway ---
        # The Metabolic-Only Prediction Head
        self.prior_head = nn.Sequential(
            nn.Linear(enzyme_dim * 3, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        # --- 3. The Adaptive Alpha Gate ---
        # Decides how much to trust the Graph vs the Prior
        self.alpha_gate = nn.Sequential(
            nn.Linear((self.graph_out_dim * 3) + (enzyme_dim * 3), 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

        print(f"[BioGuardGAT] Adaptive Ensemble Gate Enabled. Enzyme Dim: {enzyme_dim}")

    @property
    def device(self):
        return next(self.parameters()).device

    def forward_one_arm(self, x, edge_index, edge_attr, batch):
        """Processes the 3D Graph structure without injecting the prior."""
        x = self.atom_encoder(x.float())
        x = F.elu(self.conv1(x, edge_index, edge_attr=edge_attr.float()))
        x = F.elu(self.conv2(x, edge_index, edge_attr=edge_attr.float()))

        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)

        return torch.cat([x_mean, x_max], dim=1)

    def forward(self, data_a, data_b):
        # 1. Graph Embeddings
        vec_a = self.forward_one_arm(data_a.x, data_a.edge_index, data_a.edge_attr, data_a.batch)
        vec_b = self.forward_one_arm(data_b.x, data_b.edge_index, data_b.edge_attr, data_b.batch)

        # Symmetric Graph Combination
        diff_g = torch.abs(vec_a - vec_b)
        gat_combined = torch.cat([vec_a + vec_b, diff_g, vec_a * vec_b], dim=1)

        # 2. Metabolic Embeddings
        enz_a = data_a.enzyme_a.float()
        enz_b = data_b.enzyme_b.float()

        # Symmetric Metabolic Combination
        diff_e = torch.abs(enz_a - enz_b)
        enz_combined = torch.cat([enz_a + enz_b, diff_e, enz_a * enz_b], dim=1)

        # 3. Independent Logits
        gat_logits = self.gat_head(gat_combined)
        prior_logits = self.prior_head(enz_combined)

        # 4. The Ensemble Gate
        gate_input = torch.cat([gat_combined, enz_combined], dim=1)
        alpha = self.alpha_gate(gate_input)

        # Final Logit = (Graph * Alpha) + (Prior * (1 - Alpha))
        final_logits = (alpha * gat_logits) + ((1 - alpha) * prior_logits)

        return final_logits