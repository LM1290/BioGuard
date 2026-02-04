import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, recall_score
from rdkit import Chem
from rdkit.Chem import AllChem
from bioguard.data_loader import load_twosides_data
# --- NEW IMPORT ---
from bioguard.enzyme import EnzymeManager
import os
import joblib
import argparse

# --- CONFIG ---
ARTIFACT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'artifacts')
os.makedirs(ARTIFACT_DIR, exist_ok=True)


def get_fingerprint(smiles, radius=2, n_bits=1024):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros((n_bits,), dtype=np.float32)
    return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, default='cold', choices=['random', 'cold', 'scaffold'])
    args = parser.parse_args()

    print(f"--- Training LightGBM Baseline ({args.split.upper()} Split) ---")

    # 1. Load Data
    df = load_twosides_data(split_method=args.split)

    # 2. Initialize Enzyme Manager
    print("Loading Enzyme Features...")
    enzyme_mgr = EnzymeManager(allow_degraded=True)
    print(f"Enzyme Vector Dim: {enzyme_mgr.vector_dim}")

    # 3. Vectorize
    print("Generating Features (Fingerprints + Enzymes)...")

    # Pre-compute unique fingerprints to save time
    unique_smiles = pd.concat([df['smiles_a'], df['smiles_b']]).unique()
    fp_map = {s: get_fingerprint(s) for s in unique_smiles}

    def vectorize(row):
        # 1. Chemical Features (Morgan FP)
        fp_a = fp_map.get(row['smiles_a'])
        fp_b = fp_map.get(row['smiles_b'])

        # 2. Biological Features (Enzymes)
        enz_a = enzyme_mgr.get_vector(row['drug_a'])
        enz_b = enzyme_mgr.get_vector(row['drug_b'])

        # Concatenate: [FP_A, Enz_A, FP_B, Enz_B]
        return np.concatenate([fp_a, enz_a, fp_b, enz_b])

    # 4. Prepare Splits
    train_df = df[df['split'] == 'train']
    val_df = df[df['split'] == 'val']
    test_df = df[df['split'] == 'test']

    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    # Note: apply() is a bit slow, but fine for baselines
    print("Building Training Matrix...")
    X_train = np.stack(train_df.apply(vectorize, axis=1).values)
    y_train = train_df['label'].values

    print("Building Validation Matrix...")
    X_val = np.stack(val_df.apply(vectorize, axis=1).values)
    y_val = val_df['label'].values

    print("Building Test Matrix...")
    X_test = np.stack(test_df.apply(vectorize, axis=1).values)
    y_test = test_df['label'].values

    # 5. Train
    print("Fitting LightGBM...")
    clf = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.1, random_state=42)

    clf.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='auc',
        callbacks=[lgb.early_stopping(stopping_rounds=10)]
    )

    # 6. Evaluate
    y_pred = clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred)
    pr = average_precision_score(y_test, y_pred)
    acc = accuracy_score(y_test, (y_pred >= 0.5).astype(int))
    rec = recall_score(y_test, y_pred >= 0.5)

    print("\n" + "=" * 40)
    print(f"LGBM + ENZYMES RESULTS ({args.split.upper()})")
    print("=" * 40)
    print(f"ROC-AUC: {auc:.4f}")
    print(f"PR-AUC:  {pr:.4f}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Recall:   {rec:.4f}")
    print("=" * 40)

    # Save
    model_name = f'lgbm_baseline_{args.split}.pkl'
    joblib.dump(clf, os.path.join(ARTIFACT_DIR, model_name))
    print(f"Model saved to {model_name}")


if __name__ == "__main__":
    main()
