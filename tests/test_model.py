import unittest

import torch
from torch_geometric.data import Batch, Data

from bioguard.model import BioGuardGAT


def make_arm(node_features, edge_features, enzyme_features, enzyme_field):
    data = Data(
        x=torch.tensor(node_features, dtype=torch.float),
        edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        edge_attr=torch.tensor(edge_features, dtype=torch.float),
    )
    setattr(
        data,
        enzyme_field,
        torch.tensor([enzyme_features], dtype=torch.float),
    )
    return Batch.from_data_list([data])


class PairSymmetryTests(unittest.TestCase):
    def test_prediction_is_invariant_to_drug_order(self):
        torch.manual_seed(7)
        model = BioGuardGAT(
            node_dim=3,
            edge_dim=2,
            enzyme_dim=2,
            embedding_dim=8,
            heads=2,
        ).eval()

        graph_a = [[1.0, 0.0, 0.2], [0.0, 1.0, -0.1]]
        graph_b = [[0.1, 0.4, 1.0], [0.5, -0.2, 0.3]]
        edges_a = [[1.0, 0.2], [1.0, 0.2]]
        edges_b = [[0.5, 0.7], [0.5, 0.7]]
        enzyme_a = [1.0, 0.0]
        enzyme_b = [0.0, 0.5]

        batch_a = make_arm(graph_a, edges_a, enzyme_a, "enzyme_a")
        batch_b = make_arm(graph_b, edges_b, enzyme_b, "enzyme_b")
        swapped_a = make_arm(graph_b, edges_b, enzyme_b, "enzyme_a")
        swapped_b = make_arm(graph_a, edges_a, enzyme_a, "enzyme_b")

        with torch.no_grad():
            logits_ab, alpha_ab = model(batch_a, batch_b)
            logits_ba, alpha_ba = model(swapped_a, swapped_b)

        torch.testing.assert_close(logits_ab, logits_ba)
        torch.testing.assert_close(alpha_ab, alpha_ba)


if __name__ == "__main__":
    unittest.main()
