import os
import json
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
from joblib import Parallel, delayed
from concurrent.futures import ProcessPoolExecutor, as_completed

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
ARTIFACT_DIR = os.path.join(BASE_DIR, 'artifacts')
ENZYME_CSV_PATH = os.path.join(DATA_DIR, 'enzyme_features_full.csv')
SCHEMA_PATH = os.path.join(DATA_DIR, 'enzyme_schema.json')
MODEL_SAVE_PATH = os.path.join(ARTIFACT_DIR, 'cyp_predictor.joblib')

os.makedirs(ARTIFACT_DIR, exist_ok=True)


def get_physchem_and_fp(smiles, radius=2, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if not mol: raise ValueError(f"Invalid SMILES: {smiles}")

    fp = np.array(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits), dtype=np.float32)
    physchem = np.array([
        Descriptors.MolWt(mol), Descriptors.MolLogP(mol),
        Descriptors.TPSA(mol), Descriptors.NumHDonors(mol), Descriptors.NumHAcceptors(mol)
    ], dtype=np.float32)

    return np.concatenate([fp, np.nan_to_num(physchem, nan=0.0)])


def generate_scaffold(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if not mol: raise ValueError(f"Invalid SMILES: {smiles}")
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)


def process_molecule(smiles):
    try:
        return True, get_physchem_and_fp(smiles), generate_scaffold(smiles)
    except ValueError:
        return False, None, None


def train_target_model(target, valid_df):
    unique_scaffolds = valid_df['scaffold'].unique()
    train_scaffolds, test_scaffolds = train_test_split(unique_scaffolds, test_size=0.2, random_state=42)

    train_df = valid_df[valid_df['scaffold'].isin(train_scaffolds)]
    test_df = valid_df[valid_df['scaffold'].isin(test_scaffolds)]

    X_train = np.stack(train_df['features'].values)
    Y_train = train_df[target].values
    X_test = np.stack(test_df['features'].values)
    Y_test = test_df[target].values

    clf = lgb.LGBMClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=6, min_child_samples=30,
        colsample_bytree=0.2, subsample=0.8, random_state=42, class_weight='balanced',
        n_jobs=1, verbose=-1
    )
    clf.fit(X_train, Y_train)

    try:
        preds_positive = clf.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(Y_test, preds_positive)
        pr = average_precision_score(Y_test, preds_positive)
    except Exception:
        auc, pr = None, None

    return target, clf, auc, pr, len(valid_df)


def main():
    print("--- Training NCE CYP/Transporter Predictor (Production Scale) ---")

    if not os.path.exists(ENZYME_CSV_PATH):
        raise FileNotFoundError(f"Missing ChEMBL/DrugBank CSV at {ENZYME_CSV_PATH}")

    df = pd.read_csv(ENZYME_CSV_PATH).dropna(subset=['smiles']).reset_index(drop=True)
    meta_cols = ['drug_name', 'smiles', 'drug_id']
    target_cols = [c for c in df.columns if c not in meta_cols]

    core_count = os.cpu_count() or 4
    print(f"\nExtracting Fingerprints & Scaffolds across {core_count} CPU cores...")

    results = Parallel(n_jobs=-1, batch_size=100)(delayed(process_molecule)(smi) for smi in df['smiles'])

    # Filter out molecules that failed parsing
    valid_mask = [r[0] for r in results]
    df = df[valid_mask].copy()
    df['features'] = [r[1] for r, v in zip(results, valid_mask) if v]
    df['scaffold'] = [r[2] for r, v in zip(results, valid_mask) if v]

    trained_models = {}
    avg_roc, avg_pr = [], []

    print(f"\nTraining Target Models in Parallel...")
    tasks = []
    for target in target_cols:
        valid_df = df.dropna(subset=[target]).copy()
        if len(valid_df) >= 50:
            tasks.append((target, valid_df))
        else:
            print(f"  -> Skipping {target}: Not enough data ({len(valid_df)} samples)")

    max_workers = max(1, core_count - 1)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(train_target_model, t[0], t[1]): t[0] for t in tasks}
        for future in as_completed(futures):
            try:
                res_target, clf, auc, pr, n_samples = future.result()
                trained_models[res_target] = clf
                if auc is not None:
                    avg_roc.append(auc)
                    avg_pr.append(pr)
                    print(f"  -> [DONE] {res_target:<15} (N={n_samples:<4}) | ROC: {auc:.4f} | PR: {pr:.4f}")
                else:
                    print(f"  -> [DONE] {res_target:<15} (N={n_samples:<4}) | Evaluation skipped (mono-class)")
            except Exception as e:
                print(f"  -> [FAIL] {futures[future]}: {e}")

    # THE FIX: Only write successful targets to the schema
    successful_targets = list(trained_models.keys())
    schema = {
        "feature_names": successful_targets,
        "vector_dim": len(successful_targets),
        "num_features": len(successful_targets)
    }
    with open(SCHEMA_PATH, 'w') as f:
        json.dump(schema, f, indent=2)

    print(f"\nSynced schema.json with {len(successful_targets)} verified LightGBM models.")
    joblib.dump(trained_models, MODEL_SAVE_PATH)
    print(f"Saved Parallel Sparse CYP Predictor Ensemble to {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    main()