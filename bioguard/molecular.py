"""Versioned molecular identity shared by ingestion and inference."""

from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize

STANDARDIZATION_POLICY = "charge_parent_isomeric_v1"
LEGACY_POLICY = "legacy_raw_v1"


def normalize_smiles(value, policy=STANDARDIZATION_POLICY):
    if policy not in (STANDARDIZATION_POLICY, LEGACY_POLICY):
        raise ValueError(f"Unsupported molecular standardization policy: {policy}")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("SMILES must be a non-empty string")
    mol = Chem.MolFromSmiles(value.strip())
    if mol is None or mol.GetNumAtoms() == 0:
        raise ValueError("Invalid SMILES")
    if policy == LEGACY_POLICY:
        return value.strip()
    parent = rdMolStandardize.ChargeParent(mol)
    # Preserve the historical ingestion policy for single-atom parents.
    if parent.GetNumAtoms() < 2:
        parent = mol
    return Chem.MolToSmiles(parent, isomericSmiles=True, canonical=True)


def policy_for_metadata(metadata):
    policy = metadata.get("standardization_policy", LEGACY_POLICY)
    if policy not in (STANDARDIZATION_POLICY, LEGACY_POLICY):
        raise ValueError(f"Unsupported molecular standardization policy: {policy}")
    return policy


def prepare_molecule(value, policy=STANDARDIZATION_POLICY,
                     max_length=4096, max_heavy_atoms=256):
    """Bound original input size before normalizing and building conformers."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("SMILES must be a non-empty string")
    if len(value) > max_length:
        raise ValueError(f"SMILES exceeds the {max_length}-character limit")
    mol = Chem.MolFromSmiles(value.strip())
    if mol is None or mol.GetNumAtoms() == 0:
        raise ValueError("Invalid SMILES")
    if mol.GetNumHeavyAtoms() > max_heavy_atoms:
        raise ValueError(f"SMILES exceeds the {max_heavy_atoms}-heavy-atom limit")
    normalized = normalize_smiles(value, policy)
    # The legacy enzyme table intentionally has a separate non-isomeric key.
    enzyme_key = Chem.MolToSmiles(
        Chem.MolFromSmiles(normalized), isomericSmiles=False,
    )
    return normalized, enzyme_key
