import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
from rdkit import Chem
from rdkit.Chem import AllChem
from bioguard.data_loader import load_twosides_data
import os
import joblib

# --- CONFIG ---
ARTIFACT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'artifacts')
os.makedirs(ARTIFACT_DIR, exist_ok=True)


def get_fingerprint(smiles, radius=2, n_bits=1024):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros((n_bits,), dtype=np.float32)
    return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits))


def main():
    print("--- Training LightGBM Baseline (Scaffold Split) ---")


    df = load_twosides_data(split_method='Scaffold')

    # 2. Vectorize (This takes a minute)
    print("Generating Morgan Fingerprints...")

    # Pre-compute unique fingerprints to save time
    unique_smiles = pd.concat([df['smiles_a'], df['smiles_b']]).unique()
    fp_map = {s: get_fingerprint(s) for s in unique_smiles}

    def vectorize(row):
        fp_a = fp_map.get(row['smiles_a'])
        fp_b = fp_map.get(row['smiles_b'])
        return np.concatenate([fp_a, fp_b])

    # 3. Prepare Splits
    train_df = df[df['split'] == 'train']
    val_df = df[df['split'] == 'val']
    test_df = df[df['split'] == 'test']

    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    X_train = np.stack(train_df.apply(vectorize, axis=1).values)
    y_train = train_df['label'].values

    X_val = np.stack(val_df.apply(vectorize, axis=1).values)
    y_val = val_df['label'].values

    X_test = np.stack(test_df.apply(vectorize, axis=1).values)
    y_test = test_df['label'].values

    # 4. Train
    print("Fitting LightGBM...")
    clf = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
    clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric='auc',
            callbacks=[lgb.early_stopping(stopping_rounds=10)])

    # 5. Evaluate
    y_pred = clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred)
    pr = average_precision_score(y_test, y_pred)
    acc = accuracy_score(y_test, (y_pred >= 0.5).astype(int))

    print("\n" + "=" * 30)
    print("LGBM BASELINE RESULTS (SCAFFOLD SPLIT)")
    print("=" * 30)
    print(f"ROC-AUC: {auc:.4f}")
    print(f"PR-AUC:  {pr:.4f}")
    print(f"Accuracy: {acc:.4f}")
    print("=" * 30)

    # Save
    joblib.dump(clf, os.path.join(ARTIFACT_DIR, 'lgbm_baseline.pkl'))


if __name__ == "__main__":
    main()