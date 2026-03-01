import pandas as pd
import numpy as np
import os
import logging
from rdkit import Chem
from bioguard.cyp_predictor import CYPPredictor  # <--- IMPORT NEW MODULE

logger = logging.getLogger(__name__)


class EnzymeManager:
    def __init__(self, csv_path=None, allow_degraded=False):
        self.feature_map = {}
        self.smiles_map = {}
        self.vector_dim = 0

        # Initialize the Predictor
        self.predictor = CYPPredictor()

        if csv_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            csv_path = os.path.join(base_dir, 'data', 'enzyme_features_full.csv')

        self._load_features(csv_path, allow_degraded)

    # ... [Keep _canonicalize method unchanged] ...
    def _canonicalize(self, smiles):
        if pd.isna(smiles) or not smiles: return None
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                return Chem.MolToSmiles(mol, isomericSmiles=False)
        except:
            return None
        return None

    # ... [Keep _load_features method unchanged] ...
    def _load_features(self, path, allow_degraded):
        if not os.path.exists(path):
            if allow_degraded:
                logger.warning("Enzyme data missing. Running in DEGRADED mode (Zero Vectors).")
                self.vector_dim = 1
                return
            raise FileNotFoundError(f"Enzyme data not found at {path}")

        df = pd.read_csv(path)
        meta_cols = ['drug_name', 'smiles', 'drug_id']
        self.feature_names = [c for c in df.columns if c not in meta_cols]
        self.vector_dim = len(self.feature_names)
        logger.info(f"EnzymeManager: Loaded {self.vector_dim} metabolic features.")

        if 'drug_name' in df.columns:
            keys = df['drug_name'].astype(str).str.lower().str.strip().values
            values = df[self.feature_names].values.astype(np.float32)
            self.feature_map = dict(zip(keys, values))

        if 'smiles' in df.columns:
            logger.info("Indexing SMILES for OOD/In-Distribution detection...")
            count = 0
            for idx, row in df.iterrows():
                can_smi = self._canonicalize(row['smiles'])
                if can_smi:
                    vec = row[self.feature_names].values.astype(np.float32)
                    self.smiles_map[can_smi] = vec
                    count += 1
            logger.info(f"EnzymeManager: Indexed {count} SMILES for lookup.")
        else:
            logger.warning("No 'smiles' column in enzyme CSV.")

    def get_vector(self, drug_name):
        """Legacy lookup by Name/ID"""
        if not drug_name: return np.zeros(self.vector_dim, dtype=np.float32)
        key = str(drug_name).lower().strip()
        return self.feature_map.get(key, np.zeros(self.vector_dim, dtype=np.float32))

    def get_by_smiles(self, smiles):
        """Robust lookup: Hits CSV ground truth first, falls back to Predictor."""
        if not smiles: return np.zeros(self.vector_dim, dtype=np.float32)

        # 1. Try Direct Hit
        if smiles in self.smiles_map:
            return self.smiles_map[smiles]

        # 2. Try Canonical Hit
        can_smi = self._canonicalize(smiles)
        if can_smi and can_smi in self.smiles_map:
            return self.smiles_map[can_smi]

        # 3. Fallback to Prediction
        # Pass the 60-dim feature list so the predictor slots the 15 targets correctly!
        return self.predictor.predict(smiles, target_feature_list=self.feature_names)