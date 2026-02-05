import pandas as pd
import numpy as np
import os
import joblib
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from rdkit import Chem
from rdkit.Chem import AllChem

# Configure Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("BioGuard.CYP")

ARTIFACT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'artifacts')
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
MODEL_PATH = os.path.join(ARTIFACT_DIR, 'cyp_predictor_rf.joblib')


class CYPPredictor:
    def __init__(self, model_path=MODEL_PATH):
        self.model = None
        self.feature_names = []
        if os.path.exists(model_path):
            logger.info(f"Loading CYP Predictor from {model_path}")
            data = joblib.load(model_path)
            self.model = data['model']
            self.feature_names = data['feature_names']
        else:
            logger.warning(f"CYP Predictor model not found at {model_path}. Inference will fail until trained.")

    @staticmethod
    def _get_ecfp4(smiles):
        """
        Generates 1024-bit Morgan Fingerprint (Radius 2).
        Static method to allow usage without instantiation.
        """
        if not smiles: return np.zeros((1024,), dtype=np.float32)
        mol = Chem.MolFromSmiles(smiles)
        if not mol: return np.zeros((1024,), dtype=np.float32)
        # Use fixed 1024 bits to match the training configuration
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
        return np.array(fp, dtype=np.float32)

    def predict(self, smiles):
        """
        Predicts the enzyme interaction PROBABILITIES for a given SMILES string.
        Returns: np.array (float32) of probabilities [0.0 - 1.0].
        """
        if self.model is None:
            return np.zeros(len(self.feature_names) if self.feature_names else 30, dtype=np.float32)

        fp = self._get_ecfp4(smiles).reshape(1, -1)

        # RandomForest with Multi-Output returns a LIST of arrays (one per target).
        # Each array has shape (n_samples, n_classes).
        # We need to extract the positive class probability (index 1) for each target.
        all_preds = self.model.predict_proba(fp)

        probs = []
        for target_pred in all_preds:
            # target_pred shape is (1, n_classes)
            # Check if class 1 exists (handling edge case of single-class targets)
            if target_pred.shape[1] > 1:
                probs.append(target_pred[0, 1])
            else:
                # If only one class exists, it is almost certainly the negative class (0).
                # (Assuming reasonable sparsity in biological data)
                probs.append(0.0)

        return np.array(probs, dtype=np.float32)


def train_cyp_predictor(data_path=None):
    """Trains the Random Forest model on the Enzyme CSV."""
    if data_path is None:
        data_path = os.path.join(DATA_DIR, 'enzyme_features_full.csv')

    logger.info(f"Loading training data from {data_path}...")
    df = pd.read_csv(data_path)

    # 1. Prepare Features (X) and Targets (y)
    df = df.dropna(subset=['smiles'])

    meta_cols = ['drug_name', 'smiles', 'drug_id']
    target_cols = [c for c in df.columns if c not in meta_cols]

    logger.info(f"Training on {len(df)} compounds for {len(target_cols)} enzyme targets.")

    # Generate Fingerprints using the static method
    logger.info("Generating ECFP4 fingerprints...")
    # Clean call to static method
    X = np.stack(df['smiles'].apply(CYPPredictor._get_ecfp4).values)
    y = df[target_cols].values

    # 2. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    # 3. Model Training
    logger.info("Fitting Random Forest (n_estimators=100)...")
    rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42, max_depth=20)
    rf.fit(X_train, y_train)

    # 4. Evaluation (Using Predict Proba logic for accuracy check)
    # Note: For simple accuracy logging, we can just use the built-in predict which uses 0.5 threshold
    y_pred_class = rf.predict(X_test)
    acc = np.mean(y_pred_class == y_test)
    logger.info(f"Global Subset Accuracy (Hard Classes): {acc:.4f}")

    # 5. Save Artifacts
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    joblib.dump({'model': rf, 'feature_names': target_cols}, MODEL_PATH)
    logger.info(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    train_cyp_predictor()