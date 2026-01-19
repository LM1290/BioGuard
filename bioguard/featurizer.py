"""
Molecular featurization module for DDI prediction.

Combines structural fingerprints with biophysical properties
to create comprehensive drug pair representations.
"""

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator, Descriptors
from rdkit.Chem.MolStandardize import rdMolStandardize

FINGERPRINT_SIZE = 2048
MAX_SMILES_LENGTH = 5000
BIO_PHYS_DIM = 5  # MW, LogP, TPSA, H-donors, H-acceptors
ENZYME_DIM = 11   # Enzyme features (for compatibility with trained model)


class BioFeaturizer:
    """
    Featurizer for drug-drug interaction prediction.
    
    Features per drug:
    - Morgan fingerprints (2048-bit, radius 2)
    - Biophysical properties (5 features)
    - Enzyme features (11 features, zeros for deployment)
    
    Pair features: [sum, diff, product] of individual drug features
    """

    def __init__(self):
        self.single_drug_dim = FINGERPRINT_SIZE + BIO_PHYS_DIM + ENZYME_DIM
        self.total_dim = self.single_drug_dim * 3  # sum, diff, product

        self.morgan_gen = rdFingerprintGenerator.GetMorganGenerator(
            radius=2,
            fpSize=FINGERPRINT_SIZE
        )
        self.tautomer_enumerator = rdMolStandardize.TautomerEnumerator()

        # SAFETY LIMITS: Prevent hangs on molecules with thousands of tautomers
        self.tautomer_enumerator.SetMaxTautomers(50)
        self.tautomer_enumerator.SetMaxTransforms(50)
    def _get_bio_physical_features(self, mol):
        """Calculate normalized biophysical properties."""
        if mol is None:
            return np.zeros(BIO_PHYS_DIM, dtype=np.float32)
        
        try:
            return np.array([
                Descriptors.MolWt(mol) / 500.0,
                Descriptors.MolLogP(mol) / 5.0,
                Descriptors.TPSA(mol) / 100.0,
                Descriptors.NumHDonors(mol) / 5.0,
                Descriptors.NumHAcceptors(mol) / 10.0
            ], dtype=np.float32)
        except:
            return np.zeros(BIO_PHYS_DIM, dtype=np.float32)

    def _get_enzyme_features(self, mol):
        """
        Enzyme features placeholder.
        Returns zeros since enzyme data not available in deployment.
        Model was trained with these features, so we maintain compatibility.
        """
        return np.zeros(ENZYME_DIM, dtype=np.float32)

    def featurize_single_drug(self, smiles, drug_name=None):
        """Featurize a single drug with safety-first standardization."""
        if pd.isna(smiles) or not smiles:
            return np.zeros(self.single_drug_dim, dtype=np.float32)

        mol = Chem.MolFromSmiles(smiles)

        if mol:
            try:
                # 1. ALWAYS strip salts and uncharge (very robust)
                mol = rdMolStandardize.ChargeParent(mol)

                try:
                    # 2. TRY to canonicalize tautomers
                    mol = self.tautomer_enumerator.Canonicalize(mol)
                except:
                    # 3. FALLBACK: If tautomers fail, keep the salt-stripped version
                    # This prevents returning zeros for valid but complex drugs
                    pass
            except:
                # Only return zeros if the molecule is completely unreadable
                return np.zeros(self.single_drug_dim, dtype=np.float32)

        if mol is None:
            return np.zeros(self.single_drug_dim, dtype=np.float32)

        # Morgan fingerprint
        fp_arr = np.zeros(FINGERPRINT_SIZE, dtype=np.float32)
        fp = self.morgan_gen.GetFingerprint(mol)
        DataStructs.ConvertToNumpyArray(fp, fp_arr)

        # Biophysical properties
        phys_arr = self._get_bio_physical_features(mol)

        # Enzyme features (zeros for deployment)
        enzyme_arr = self._get_enzyme_features(mol)

        return np.concatenate([fp_arr, phys_arr, enzyme_arr])
    def featurize_pair(self, smiles_a, smiles_b, name_a=None, name_b=None):
        """Featurize a drug pair using symmetric operations."""
        vec_a = self.featurize_single_drug(smiles_a, name_a)
        vec_b = self.featurize_single_drug(smiles_b, name_b)
        
        return np.concatenate([
            vec_a + vec_b,
            np.abs(vec_a - vec_b),
            vec_a * vec_b
        ])
