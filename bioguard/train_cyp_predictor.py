import os
import json
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score

# --- CONFIG ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
ARTIFACT_DIR = os.path.join(BASE_DIR, 'artifacts')
ENZYME_CSV_PATH = os.path.join(DATA_DIR, 'enzyme_features_full.csv')
SCHEMA_PATH = os.path.join(DATA_DIR, 'enzyme_schema.json')
MODEL_SAVE_PATH = os.path.join(ARTIFACT_DIR, 'cyp_predictor.joblib')

os.makedirs(ARTIFACT_DIR, exist_ok=True)


def get_physchem_and_fp(smiles, radius=2, n_bits=2048):
    """Generates Morgan FP + 5 Key PhysChem properties for generalization."""
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return np.zeros(n_bits + 5, dtype=np.float32)

    # 1. Morgan Fingerprint (Structural specifics)
    fp = np.array(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits), dtype=np.float32)

    # 2. PhysChem Descriptors (Overall physical feel for active site fitting)
    physchem = np.array([
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        Descriptors.TPSA(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.NumHAcceptors(mol)
    ], dtype=np.float32)

    # Handle potential NaNs in weird RDKit failures
    physchem = np.nan_to_num(physchem, nan=0.0)

    return np.concatenate([fp, physchem])


def generate_scaffold(smiles):
    """Extracts Murcko Scaffold for strict data splitting."""
    mol = Chem.MolFromSmiles(smiles)
    if not mol: return "INVALID"
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    except:
        return "ERROR"


def main():
    print("--- Training NCE CYP/Transporter Predictor ---")

    # 1. Load Data and Schema
    if not os.path.exists(ENZYME_CSV_PATH):
        raise FileNotFoundError(f"Missing ChEMBL/DrugBank CSV at {ENZYME_CSV_PATH}")

    df = pd.read_csv(ENZYME_CSV_PATH)

    with open(SCHEMA_PATH, 'r') as f:
        schema = json.load(f)
    target_cols = schema['feature_names']

    # Drop missing SMILES or rows where ALL targets are null
    df = df.dropna(subset=['smiles'])
    df = df.dropna(subset=target_cols, how='all').reset_index(drop=True)

    print(f"Loaded {len(df)} molecules.")

    # 2. Featurization
    print("Extracting Fingerprints and PhysChem features...")
    X = np.stack(df['smiles'].apply(get_physchem_and_fp).values)

    # Fill any remaining NaNs in targets with 0 (assuming lack of evidence = negative)
    Y = df[target_cols].fillna(0).values

    # 3. Scaffold Split
    print("Performing strict scaffold split...")
    df['scaffold'] = df['smiles'].apply(generate_scaffold)
    unique_scaffolds = df['scaffold'].unique()

    # Split scaffolds, not molecules
    train_scaffolds, test_scaffolds = train_test_split(unique_scaffolds, test_size=0.2, random_state=42)

    train_idx = df[df['scaffold'].isin(train_scaffolds)].index
    test_idx = df[df['scaffold'].isin(test_scaffolds)].index

    X_train, X_test = X[train_idx], X[test_idx]
    Y_train, Y_test = Y[train_idx], Y[test_idx]

    print(f"Train size: {len(X_train)} | Test size (NCE simulation): {len(X_test)}")

    # 4. Initialize Regularized LightGBM
    # Heavy regularization is critical here to avoid memorizing exact ChEMBL analogs
    base_lgbm = lgb.LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,  # Limit tree depth
        min_child_samples=30,  # Equivalent to min_data_in_leaf
        colsample_bytree=0.2,  # feature_fraction: Forces diversity, relies on PhysChem more
        subsample=0.8,  # bagging_fraction
        random_state=42,
        class_weight='balanced',  # Crucial for rare metabolic pathways
        n_jobs=-1
    )

    clf = MultiOutputClassifier(base_lgbm)

    # 5. Train
    print("Training 10-Task MultiOutput LightGBM...")
    clf.fit(X_train, Y_train)

    # 6. Evaluate
    print("\n--- Model Evaluation on Unseen Scaffolds (NCE Simulation) ---")
    y_pred_proba = clf.predict_proba(X_test)

    # predict_proba with MultiOutputClassifier returns a list of arrays (one for each target)
    # We need to extract the probability of the positive class ([:, 1]) for each
    preds_positive = np.array([prob[:, 1] for prob in y_pred_proba]).T

    avg_roc = []
    avg_pr = []

    for i, col in enumerate(target_cols):
        try:
            auc = roc_auc_score(Y_test[:, i], preds_positive[:, i])
            pr = average_precision_score(Y_test[:, i], preds_positive[:, i])
            avg_roc.append(auc)
            avg_pr.append(pr)
            print(f"{col:>15}: ROC-AUC = {auc:.4f} | PR-AUC = {pr:.4f}")
        except ValueError:
            print(f"{col:>15}: Only one class present in test set. Skipping metrics.")

    print("-" * 50)
    print(f"Average ROC-AUC: {np.mean(avg_roc):.4f}")
    print(f"Average PR-AUC:  {np.mean(avg_pr):.4f}")

    # 7. Save the model
    joblib.dump(clf, MODEL_SAVE_PATH)
    print(f"\nSaved CYP Predictor to {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    main()