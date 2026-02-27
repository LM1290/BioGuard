import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import Descriptors, rdchem, AllChem, rdFreeSASA, Descriptors3D
from torch_geometric.data import Data


class GraphFeaturizer:
    def __init__(self):
        self.atom_types = ['C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'I', 'P', 'B', 'Si', 'Se', 'Na', 'K', 'Mg', 'Ca', 'Fe',
                           'Zn', 'Mn', 'Cu', 'Co', 'Ni', 'As', 'UNK']
        self.hybridization = [rdchem.HybridizationType.SP, rdchem.HybridizationType.SP2, rdchem.HybridizationType.SP3,
                              rdchem.HybridizationType.SP3D, rdchem.HybridizationType.SP3D2]
        self.chiral_types = [rdchem.ChiralType.CHI_UNSPECIFIED, rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
                             rdchem.ChiralType.CHI_TETRAHEDRAL_CCW, rdchem.ChiralType.CHI_OTHER]

        # Topological (41) + Spatial (5) = 46
        self.node_dim = len(self.atom_types) + 6 + len(self.hybridization) + 1 + 1 + len(self.chiral_types) + 4 + 1
        self.edge_dim = 8

    def _one_hot(self, x, allowable_set):
        if x not in allowable_set: x = allowable_set[-1]
        return list(map(lambda s: x == s, allowable_set))

    def _get_3d_features(self, mol, num_confs=5):
        mol_3d = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        ids = list(AllChem.EmbedMultipleConfs(mol_3d, numConfs=num_confs, params=params))
        num_atoms = mol.GetNumAtoms()

        if not ids:
            # NO GHOST CONFORMERS. Fail loudly.
            raise ValueError("RDKit ETKDGv3 failed to embed 3D coordinates.")

        # 1. Gasteiger Stats
        conf_charges = []
        for cid in ids:
            AllChem.ComputeGasteigerCharges(mol_3d, confId=cid)
            conf_charges.append([float(mol_3d.GetAtomWithIdx(i).GetProp('_GasteigerCharge')) for i in range(num_atoms)])

        conf_charges = np.array(conf_charges)
        charge_stats = [[np.mean(conf_charges[:, i]), np.std(conf_charges[:, i]), np.min(conf_charges[:, i]),
                         np.max(conf_charges[:, i])] for i in range(num_atoms)]

        # 2. SASA
        sasa_vals = []
        radii = rdFreeSASA.classifyAtoms(mol_3d)
        for cid in ids:
            rdFreeSASA.CalcSASA(mol_3d, radii, confId=cid)
            sasa_vals.append([float(mol_3d.GetAtomWithIdx(i).GetProp("SASA")) for i in range(num_atoms)])
        sasa_mean = np.mean(sasa_vals, axis=0).tolist()

        # 3. Global Shape
        vols, gyrs, asphs = [], [], []
        for cid in ids:
            vols.append(AllChem.ComputeMolVolume(mol_3d, confId=cid))
            gyrs.append(Descriptors3D.RadiusOfGyration(mol_3d, confId=cid))
            asphs.append(Descriptors3D.Asphericity(mol_3d, confId=cid))
        global_3d = [np.mean(gyrs), np.mean(asphs), np.mean(vols)]

        # 4. Distance Matrix (Use Conformer 0)
        dist_matrix = AllChem.Get3DDistanceMatrix(mol_3d, confId=ids[0])
        return charge_stats, sasa_mean, global_3d, dist_matrix

    def smiles_to_graph(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            # NO BLANK GRAPHS. Fail loudly.
            raise ValueError(f"Invalid SMILES string cannot be parsed by RDKit: {smiles}")

        charge_stats, sasa, global_3d, dist_matrix = self._get_3d_features(mol)

        x = []
        for i, atom in enumerate(mol.GetAtoms()):
            feats = (
                    self._one_hot(atom.GetSymbol(), self.atom_types) +
                    self._one_hot(atom.GetDegree(), [0, 1, 2, 3, 4, 5]) +
                    self._one_hot(atom.GetHybridization(), self.hybridization) +
                    [1 if atom.GetIsAromatic() else 0] +
                    [atom.GetFormalCharge()] +
                    self._one_hot(atom.GetChiralTag(), self.chiral_types) +
                    charge_stats[i] + [sasa[i]]
            )
            x.append(feats)

        edge_indices, edge_attrs = [], []

        # A. Chemical Bonds
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            dist = dist_matrix[i, j] if dist_matrix is not None else 1.5
            inv_dist = min(10.0, 1.0 / (dist + 1e-5))

            bt = bond.GetBondType()
            feats = [
                1 if bt == rdchem.BondType.SINGLE else 0,
                1 if bt == rdchem.BondType.DOUBLE else 0,
                1 if bt == rdchem.BondType.TRIPLE else 0,
                1 if bt == rdchem.BondType.AROMATIC else 0,
                1 if bond.GetIsConjugated() else 0,
                1 if bond.IsInRing() else 0,
                0.0,
                inv_dist
            ]
            edge_indices += [(i, j), (j, i)];
            edge_attrs += [feats, feats]

        # B. Virtual Spatial Edges
        if dist_matrix is not None:
            for i in range(mol.GetNumAtoms()):
                for j in range(i + 1, mol.GetNumAtoms()):
                    dist = dist_matrix[i, j]
                    if dist < 4.5 and mol.GetBondBetweenAtoms(i, j) is None:
                        inv_dist = min(10.0, 1.0 / (dist + 1e-5))
                        feats = [0, 0, 0, 0, 0, 0, 1.0 * inv_dist, inv_dist]
                        edge_indices += [(i, j), (j, i)];
                        edge_attrs += [feats, feats]

        if not edge_indices:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, self.edge_dim))
        else:
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attrs, dtype=torch.float)

        data = Data(x=torch.tensor(x, dtype=torch.float), edge_index=edge_index, edge_attr=edge_attr)
        data.global_3d = torch.tensor([global_3d], dtype=torch.float)
        return data


_global_featurizer = GraphFeaturizer()


def drug_to_graph(smiles, drug_name=None):
    return _global_featurizer.smiles_to_graph(smiles)