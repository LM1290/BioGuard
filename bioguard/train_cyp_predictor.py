import os
import json
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.calibration import CalibratedClassifierCV
from joblib import Parallel, delayed
from concurrent.futures import ProcessPoolExecutor, as_completed

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
ARTIFACT_DIR = os.path.join(BASE_DIR, 'artifacts')
ENZYME_CSV_PATH = os.path.join(DATA_DIR, 'enzyme_features_full.csv')
SCHEMA_PATH = os.path.join(DATA_DIR, 'enzyme_schema.json')
MODEL_SAVE_PATH = os.path.join(ARTIFACT_DIR, 'cyp_predictor.joblib')

os.makedirs(ARTIFACT_DIR, exist_ok=True)


# 1. PURE TOPOLOGY: Rip out PhysChem descriptors
def get_fp_only(smiles, radius=2, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if not mol: raise ValueError(f"Invalid SMILES: {smiles}")
    return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits), dtype=np.float32)


def generate_scaffold(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if not mol: raise ValueError(f"Invalid SMILES: {smiles}")
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)


def process_molecule(smiles):
    try:
        return True, get_fp_only(smiles), generate_scaffold(smiles)
    except ValueError:
        return False, None, None


def train_target_model(target, valid_df):
    unique_scaffolds = valid_df['scaffold'].unique()

    # 2. STRICT 3-WAY SCAFFOLD SPLIT (Train/Calibrate/Test)
    train_scaffolds, temp_scaffolds = train_test_split(unique_scaffolds, test_size=0.3, random_state=42)
    calib_scaffolds, test_scaffolds = train_test_split(temp_scaffolds, test_size=0.5, random_state=42)

    train_df = valid_df[valid_df['scaffold'].isin(train_scaffolds)]
    calib_df = valid_df[valid_df['scaffold'].isin(calib_scaffolds)]
    test_df = valid_df[valid_df['scaffold'].isin(test_scaffolds)]

    X_train, Y_train = np.stack(train_df['features'].values), train_df[target].values
    X_calib, Y_calib = np.stack(calib_df['features'].values), calib_df[target].values
    X_test, Y_test = np.stack(test_df['features'].values), test_df[target].values

    clf = lgb.LGBMClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=6, min_child_samples=30,
        colsample_bytree=0.2, subsample=0.8, random_state=42, class_weight='balanced',
        n_jobs=1, verbose=-1
    )
    clf.fit(X_train, Y_train)

    # 3. PLATT SCALING (Sigmoid Calibration)
    try:
        # Only calibrate if the calibration holdout has both classes
        if len(np.unique(Y_calib)) > 1:
            calibrated_clf = CalibratedClassifierCV(estimator=clf, method='sigmoid', cv='prefit')
            calibrated_clf.fit(X_calib, Y_calib)
        else:
            calibrated_clf = clf
    except Exception:
        calibrated_clf = clf

    try:
        preds_positive = calibrated_clf.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(Y_test, preds_positive)
        pr = average_precision_score(Y_test, preds_positive)
    except Exception:
        auc, pr = None, None

    return target, calibrated_clf, auc, pr, len(valid_df)


def main():
    print("--- Training Pure-Topology CYP Predictor ---")
    df = pd.read_csv(ENZYME_CSV_PATH).dropna(subset=['smiles']).reset_index(drop=True)
    meta_cols = ['drug_name', 'smiles', 'drug_id']
    target_cols = [c for c in df.columns if c not in meta_cols]

    core_count = os.cpu_count() or 4
    results = Parallel(n_jobs=-1, batch_size=100)(delayed(process_molecule)(smi) for smi in df['smiles'])

    valid_mask = [r[0] for r in results]
    df = df[valid_mask].copy()
    df['features'] = [r[1] for r, v in zip(results, valid_mask) if v]
    df['scaffold'] = [r[2] for r, v in zip(results, valid_mask) if v]

    trained_models = {}
    tasks = [(t, df.dropna(subset=[t]).copy()) for t in target_cols if len(df.dropna(subset=[t])) >= 50]

    with ProcessPoolExecutor(max_workers=max(1, core_count - 1)) as executor:
        futures = {executor.submit(train_target_model, t[0], t[1]): t[0] for t in tasks}
        for future in as_completed(futures):
            try:
                res_target, clf, auc, pr, n_samples = future.result()
                if auc is not None:
                    trained_models[res_target] = clf
                    print(f"  -> [DONE] {res_target:<15} | ROC: {auc:.4f} | PR: {pr:.4f}")
            except Exception as e:
                pass

    # The Ruthless Filter
    successful_targets = []
    ruthless_trained_models = {}

    for target in target_cols:
        if target in trained_models:
            valid_df = df.dropna(subset=[target])
            if valid_df[target].nunique() > 1:
                ruthless_trained_models[target] = trained_models[target]
                successful_targets.append(target)

    schema = {"feature_names": successful_targets, "vector_dim": len(successful_targets),
              "num_features": len(successful_targets)}
    with open(SCHEMA_PATH, 'w') as f:
        json.dump(schema, f, indent=2)

    joblib.dump(ruthless_trained_models, MODEL_SAVE_PATH)
    print(f"\nSaved Calibrated Ensemble to {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    main()