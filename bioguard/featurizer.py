"""
Molecular featurization module.
Contains BOTH:
1. BioFeaturizer (Fingerprints) -> For Baselines
2. GraphFeaturizer (Graphs) -> For BioGuardGAT
"""

import numpy as np
import torch
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator, Descriptors, rdchem
from rdkit.Chem.MolStandardize import rdMolStandardize
from torch_geometric.data import Data


# --- 1. LEGACY FEATURIZER (FOR BASELINES) ---
class BioFeaturizer:
    def __init__(self):
        self.fp_size = 2048
        self.morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

    def featurize_single_drug(self, smiles):
        if not smiles: return np.zeros(2048 + 5)
        mol = Chem.MolFromSmiles(smiles)
        if not mol: return np.zeros(2048 + 5)

        fp = self.morgan_gen.GetFingerprint(mol)
        arr = np.zeros((0,), dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, arr)

        phys = np.array([
            Descriptors.MolWt(mol), Descriptors.MolLogP(mol),
            Descriptors.TPSA(mol), Descriptors.NumHDonors(mol),
            Descriptors.NumHAcceptors(mol)
        ])
        return np.concatenate([arr, phys])

    def featurize_pair(self, s1, s2):
        v1 = self.featurize_single_drug(s1)
        v2 = self.featurize_single_drug(s2)
        return np.concatenate([v1 + v2, np.abs(v1 - v2), v1 * v2])


# --- 2. NEW GRAPH FEATURIZER (FOR GAT) ---
class GraphFeaturizer:
    def __init__(self):
        # Atoms: C N O S F Cl Br I P
        self.atom_types = ['C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'I', 'P']
        self.hybridization = [
            rdchem.HybridizationType.SP,
            rdchem.HybridizationType.SP2,
            rdchem.HybridizationType.SP3,
            rdchem.HybridizationType.SP3D,
            rdchem.HybridizationType.SP3D2
        ]

        # Node dim: 9 (types) + 6 (degrees) + 5 (hybrid) + 1 (aromatic) + 1 (formal charge) = 22
        self.node_dim = 22
        # Edge dim: 4 (Bond types) + 1 (Conjugated) + 1 (InRing) = 6
        self.edge_dim = 6

    def _one_hot(self, x, allowable_set):
        if x not in allowable_set: x = allowable_set[-1]
        return list(map(lambda s: x == s, allowable_set))

    def smiles_to_graph(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            return Data(
                x=torch.zeros((1, self.node_dim)),
                edge_index=torch.empty((2, 0), dtype=torch.long),
                edge_attr=torch.empty((0, self.edge_dim))
            )

        # 1. Node Features (Atom properties)
        x = []
        for atom in mol.GetAtoms():
            features = (
                    self._one_hot(atom.GetSymbol(), self.atom_types) +
                    self._one_hot(atom.GetDegree(), [0, 1, 2, 3, 4, 5]) +
                    self._one_hot(atom.GetHybridization(), self.hybridization) +
                    [1 if atom.GetIsAromatic() else 0] +
                    [atom.GetFormalCharge()]
            )
            x.append(features)
        x = torch.tensor(x, dtype=torch.float)

        # 2. Edge Features (Bond properties)
        edge_indices = []
        edge_attrs = []

        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            # Feature vector
            bond_type = bond.GetBondType()
            feats = [
                1 if bond_type == rdchem.BondType.SINGLE else 0,
                1 if bond_type == rdchem.BondType.DOUBLE else 0,
                1 if bond_type == rdchem.BondType.TRIPLE else 0,
                1 if bond_type == rdchem.BondType.AROMATIC else 0,
                1 if bond.GetIsConjugated() else 0,
                1 if bond.IsInRing() else 0
            ]

            # Add bidirectional edges
            edge_indices.append((i, j))
            edge_attrs.append(feats)

            edge_indices.append((j, i))
            edge_attrs.append(feats)

        if not edge_indices:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, self.edge_dim))
        else:
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attrs, dtype=torch.float)

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)