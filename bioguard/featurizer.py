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
        params.numThreads=1

        # Use all CPU cores for embedding
        ids = list(AllChem.EmbedMultipleConfs(mol_3d, numConfs=num_confs, params=params))
        num_atoms = mol.GetNumAtoms()

        if not ids:
            # NO GHOST CONFORMERS. Fail loudly.
            raise ValueError("RDKit ETKDGv3 failed to embed 3D coordinates.")

        # 1. Gasteiger Stats (Topological - Compute once safely)
        AllChem.ComputeGasteigerCharges(mol_3d)
        charge_stats = []
        for i in range(num_atoms):
            try:
                c = float(mol_3d.GetAtomWithIdx(i).GetProp('_GasteigerCharge'))
                if np.isnan(c) or np.isinf(c):
                    c = 0.0
            except:
                c = 0.0
            # [mean, std, min, max] -> std is 0 for topological charges
            charge_stats.append([c, 0.0, c, c])

        # 2. SASA (Spatial - Averaged over conformers)
        sasa_vals = []
        try:
            radii = rdFreeSASA.classifyAtoms(mol_3d)
            for cid in ids:
                rdFreeSASA.CalcSASA(mol_3d, radii, confIdx=cid)
                vals = []
                for i in range(num_atoms):
                    try:
                        v = float(mol_3d.GetAtomWithIdx(i).GetProp("SASA"))
                        vals.append(v if not np.isnan(v) else 0.0)
                    except:
                        vals.append(0.0)
                sasa_vals.append(vals)
            sasa_mean = np.mean(sasa_vals, axis=0).tolist()
        except:
            # Fallback if SASA fails entirely
            sasa_mean = [0.0] * num_atoms

        # 3. Global Shape (Spatial - Averaged over conformers)
        vols, gyrs, asphs = [], [], []
        for cid in ids:
            # Volume
            try:
                v = AllChem.ComputeMolVolume(mol_3d, confId=cid)
                vols.append(v if not np.isnan(v) else 0.0)
            except:
                vols.append(0.0)

            # Radius of Gyration
            try:
                g = Descriptors3D.RadiusOfGyration(mol_3d, confId=cid)
                gyrs.append(g if not np.isnan(g) else 0.0)
            except:
                gyrs.append(0.0)

            # Asphericity (Handles divide-by-zero for single atoms/linear molecules)
            try:
                a = Descriptors3D.Asphericity(mol_3d, confId=cid)
                asphs.append(a if not np.isnan(a) else 0.0)
            except:
                asphs.append(0.0)

        # Safely compute means
        g_mean = float(np.mean(gyrs)) if gyrs else 0.0
        a_mean = float(np.mean(asphs)) if asphs else 0.0
        v_mean = float(np.mean(vols)) if vols else 0.0

        global_3d = [g_mean, a_mean, v_mean]

        # 4. Distance Matrix (Use Conformer 0)
        dist_matrix = AllChem.Get3DDistanceMatrix(mol_3d, confId=ids[0])
        return charge_stats, sasa_mean, global_3d, dist_matrix

    def smiles_to_graph(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
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
            edge_indices += [(i, j), (j, i)]
            edge_attrs += [feats, feats]

        # B. Virtual Spatial Edges
        if dist_matrix is not None:
            num_atoms = mol.GetNumAtoms()
            for i in range(num_atoms):
                for j in range(i + 1, num_atoms):
                    dist = dist_matrix[i, j]
                    if dist < 4.5 and mol.GetBondBetweenAtoms(i, j) is None:
                        inv_dist = min(10.0, 1.0 / (dist + 1e-5))
                        feats = [0, 0, 0, 0, 0, 0, 1.0 * inv_dist, inv_dist]
                        edge_indices += [(i, j), (j, i)]
                        edge_attrs += [feats, feats]
        # C. Biologically Accurate Self-Loops
        num_atoms = mol.GetNumAtoms()
        for i in range(num_atoms):
            # Features:[Single, Double, Triple, Arom, Conj, Ring, Virtual, InvDist]
            # InvDist is clamped to max (10.0) for self-distance of 0.0
            self_feats =[0, 0, 0, 0, 0, 0, 1.0, 10.0]
            edge_indices.append((i, i))
            edge_attrs.append(self_feats)
        # -----
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