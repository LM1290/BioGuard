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
        # We still load the schema to know which 15 models we actually have
        if os.path.exists(schema_path):
            with open(schema_path, 'r') as f:
                schema = json.load(f)
            self.trained_features = schema['feature_names']
        else:
            self.trained_features = []

        if os.path.exists(model_path):
            self.models = joblib.load(model_path)
        else:
            self.models = {}

    @staticmethod
    def _get_features(smiles):
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            raise ValueError(f"Invalid SMILES: {smiles}")
        return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048), dtype=np.float32)

    def predict(self, smiles, target_feature_list=None):
        """
        Maps the trained predictions into the exact dimension expected by EnzymeManager.
        target_feature_list: The 60-dim list of feature names from the CSV.
        """
        if not self.models:
            if target_feature_list: return np.zeros(len(target_feature_list), dtype=np.float32)
            raise ValueError("No models loaded.")

        features = self._get_features(smiles).reshape(1, -1)

        # If no target list is provided, just return what we have (fallback)
        if target_feature_list is None:
            target_feature_list = self.trained_features

        out_vec = np.zeros(len(target_feature_list), dtype=np.float32)

        for i, target in enumerate(target_feature_list):
            if target in self.models:
                out_vec[i] = self.models[target].predict_proba(features)[0, 1]
            else:
                # Sparse target (e.g., P-gp or UGT without enough LightGBM training data)
                # Defaults to 0.0 (Inert) for unknown cold drugs
                out_vec[i] = 0.0

        return out_vec