import pandas as pd
import numpy as np
import os
import warnings
import json


class EnzymeManager:
    def __init__(self, csv_path=None, allow_degraded=False):
        self.feature_map = {}
        self.column_indices = {}
        self.feature_names = []
        self.vector_dim = 0
        self.degraded_mode = False

        # Allow path injection for worker processes
        if csv_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            csv_path = os.path.join(base_dir, 'data', 'enzyme_features.csv')

        self._load_features(csv_path, allow_degraded)

    def _load_features(self, path, allow_degraded=False):
        """
        CRITICAL FIX: Order-independent column loading with schema validation.
        Supports degraded mode if enzyme data is missing.
        """
        if not os.path.exists(path):
            if allow_degraded or os.getenv('BG_ALLOW_DEGRADED_MODE', '').lower() == 'true':
                warnings.warn(
                    f"WARNING: Enzyme data missing at {path}. "
                    f"Running in DEGRADED MODE (all unknown vectors). "
                    f"Model will have no enzyme signal."
                )
                self.degraded_mode = True
                # Set minimal dimensions for degraded mode
                self.feature_names = []
                self.vector_dim = 1  # Just the unknown flag
                return
            else:
                raise FileNotFoundError(
                    f"CRITICAL: Enzyme artifact missing at {path}. "
                    f"System cannot start without enzyme feature data. "
                    f"Set BG_ALLOW_DEGRADED_MODE=true to run without enzyme features (not recommended)."
                )

        try:
            df = pd.read_csv(path)
        except Exception as e:
            raise RuntimeError(f"Unreadable Enzyme CSV at {path}: {e}")

        if 'drug_name' not in df.columns:
            raise KeyError("Enzyme CSV missing required 'drug_name' column.")

        # CRITICAL FIX: Order-independent column selection
        self.feature_names = [c for c in df.columns if c != 'drug_name']

        if not self.feature_names:
            raise ValueError("Enzyme CSV has no feature columns.")

        self.vector_dim = len(self.feature_names) + 1  # +1 for unknown flag
        self.column_indices = {col: idx for idx, col in enumerate(self.feature_names)}

        # Save feature schema for validation
        schema_path = os.path.join(os.path.dirname(path), 'enzyme_schema.json')
        try:
            schema = {
                'feature_names': self.feature_names,
                'vector_dim': self.vector_dim,
                'num_features': len(self.feature_names)
            }
            with open(schema_path, 'w') as f:
                json.dump(schema, f, indent=2)
        except Exception as e:
            warnings.warn(f"Could not save enzyme schema: {e}")

        # Check for duplicates
        if df['drug_name'].duplicated().any():
            warnings.warn(f"Duplicate drugs in {path}. Keeping last occurrence.")

        # Vectorized loading with explicit column order
        df_feats = df[self.feature_names].apply(pd.to_numeric, errors='coerce')

        if df_feats.isna().any().any():
            n_corrupted = df_feats.isna().any(axis=1).sum()
            warnings.warn(f"Warning: {n_corrupted} rows with NaNs in enzyme data. Dropping corrupted rows.")
            valid_mask = ~df_feats.isna().any(axis=1)
            df = df[valid_mask]
            df_feats = df_feats[valid_mask]

        # Case-insensitive keying
        keys = df['drug_name'].astype(str).str.lower().str.strip().values
        values = df_feats.values.astype(float)

        self.feature_map = dict(zip(keys, values))
        
        print(f"EnzymeManager loaded {len(self.feature_map)} drugs with {len(self.feature_names)} features")

    def get_vector(self, drug_name):
        """Get enzyme feature vector for a drug (or unknown vector)."""
        if self.degraded_mode:
            return self._get_unknown_vector()
            
        if pd.isna(drug_name): 
            return self._get_unknown_vector()
        
        key = str(drug_name).lower().strip()

        if key in self.feature_map:
            # Known drug: features + 0 for unknown flag
            return np.concatenate([self.feature_map[key], [0.0]])
        
        # Unknown drug
        return self._get_unknown_vector()

    def _get_unknown_vector(self):
        """Return vector for unknown drugs: zeros + 1 for unknown flag."""
        vec = np.zeros(self.vector_dim, dtype=np.float32)
        vec[-1] = 1.0  # Set unknown flag
        return vec

    def get_column_index(self, col_name):
        """Get the index of a feature column."""
        return self.column_indices.get(col_name)

    def get_feature_list(self):
        """Get list of feature names."""
        return self.feature_names
    
    def get_coverage(self, drug_list):
        """
        Calculate what percentage of drugs have enzyme features.
        Useful for training validation.
        """
        if not drug_list:
            return 0.0
        
        normalized = [str(d).lower().strip() for d in drug_list if not pd.isna(d)]
        matched = sum(1 for d in normalized if d in self.feature_map)
        
        return matched / len(normalized) if normalized else 0.0
