import json
import joblib
import numpy as np
import os
from rdkit import Chem
from rdkit.Chem import AllChem

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACT_DIR = os.path.join(BASE_DIR, 'artifacts')
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_PATH = os.path.join(ARTIFACT_DIR, 'cyp_predictor.joblib')
SCHEMA_PATH = os.path.join(DATA_DIR, 'enzyme_schema.json')

class CYPPredictor:
    def __init__(self, model_path=MODEL_PATH, schema_path=SCHEMA_PATH):
        if os.path.exists(schema_path):
            with open(schema_path, 'r') as f:
                schema = json.load(f)
            self.feature_names = schema['feature_names']
            self.vector_dim = schema['vector_dim']
        else:
            raise FileNotFoundError(f"Schema not found at {schema_path}")

        if os.path.exists(model_path):
            self.models = joblib.load(model_path)
        else:
            raise FileNotFoundError(f"Models not found at {model_path}")

    @staticmethod
    def _get_features(smiles):
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            raise ValueError(f"Invalid SMILES: {smiles}")
        # PURE TOPOLOGY: 2048 bits. No PhysChem.
        return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048), dtype=np.float32)

    def predict(self, smiles):
        if not self.models: raise ValueError("No models loaded.")
        features = self._get_features(smiles).reshape(1, -1)
        probs = []

        for target in self.feature_names:
            if target in self.models:
                clf = self.models[target]
                probs.append(clf.predict_proba(features)[0, 1])
            else:
                raise ValueError(f"Missing LightGBM model for '{target}'")

        return np.array(probs, dtype=np.float32)