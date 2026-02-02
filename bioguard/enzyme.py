import pandas as pd
import numpy as np
import os
import warnings
import json


class EnzymeManager:
    def __init__(self, csv_path=None, allow_degraded=False):
        self.feature_map = {}
        self.vector_dim = 0

        if csv_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            # POINTS TO THE NEW FILE
            csv_path = os.path.join(base_dir, 'data', 'enzyme_features_full.csv')

        self._load_features(csv_path, allow_degraded)

    def _load_features(self, path, allow_degraded):
        if not os.path.exists(path):
            if allow_degraded:
                print("WARNING: Enzyme data missing. Running in DEGRADED mode.")
                self.vector_dim = 1
                return
            raise FileNotFoundError(f"Enzyme data not found at {path}")

        df = pd.read_csv(path)
        # Dynamically grab all columns except drug_name
        self.feature_names = [c for c in df.columns if c != 'drug_name']
        self.vector_dim = len(self.feature_names)

        print(f"EnzymeManager: Loaded {len(self.feature_names)} metabolic features.")

        keys = df['drug_name'].astype(str).str.lower().str.strip().values
        values = df[self.feature_names].values.astype(np.float32)
        self.feature_map = dict(zip(keys, values))

    def get_vector(self, drug_name):
        if not drug_name: return np.zeros(self.vector_dim, dtype=np.float32)
        key = str(drug_name).lower().strip()
        return self.feature_map.get(key, np.zeros(self.vector_dim, dtype=np.float32))