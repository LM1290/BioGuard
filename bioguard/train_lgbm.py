import argparse
import os
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score

# Import your existing pipeline components
from bioguard.data_loader import load_twosides_data
from bioguard.train import get_pair_disjoint_split, get_cold_drug_split, get_scaffold_split
from bioguard.featurizer import BioFeaturizer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACT_DIR = os.path.join(BASE_DIR, 'artifacts')


def prepare_features(df, featurizer, desc="Featurizing"):
    """
    Flattens SMILES pairs into vectors using the existing BioFeaturizer.
    """
    X, y = [], []
    # Drop rows with missing SMILES just in case
    df = df.dropna(subset=['smiles_a', 'smiles_b'])

    for _, row in tqdm(df.iterrows(), total=len(df), desc=desc):
        try:
            # Reusing the featurizer logic from baselines.py
            vec = featurizer.featurize_pair(row['smiles_a'], row['smiles_b'])
            X.append(vec)
            y.append(row['label'])
        except Exception:
            continue

    return np.array(X), np.array(y)


def run_lgbm(args):
    print(f"--- BioGuard LightGBM Training ({args.split.upper()} Split) ---")

    # 1. Load Data
    # Utilizing the caching logic in load_twosides_data
    df = load_twosides_data()

    # 2. Apply the requested split strategy
    # This ensures apples-to-apples comparison with the GAT model
    if args.split == 'random':
        train_df, val_df, test_df = get_pair_disjoint_split(df)
    elif args.split == 'cold':
        train_df, val_df, test_df = get_cold_drug_split(df)
    elif args.split == 'scaffold':
        train_df, val_df, test_df = get_scaffold_split(df)
    else:
        raise ValueError(f"Unknown split: {args.split}")

    print(f"Split Sizes -> Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # 3. Featurize
    # LightGBM requires flat vectors (unlike GAT which takes graphs)
    featurizer = BioFeaturizer()

    print("\n[1/3] Featurizing Train Set...")
    X_train, y_train = prepare_features(train_df, featurizer)

    print("\n[2/3] Featurizing Val Set...")
    X_val, y_val = prepare_features(val_df, featurizer)

    print("\n[3/3] Featurizing Test Set...")
    X_test, y_test = prepare_features(test_df, featurizer)

    # 4. Train LightGBM
    print(f"\nTraining LightGBM (n_estimators={args.n_estimators})...")

    clf = lgb.LGBMClassifier(
        n_estimators=args.n_estimators,
        learning_rate=0.05,
        num_leaves=32,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'  # Important for the 1:10 test imbalance
    )

    # Train with early stopping on validation set
    clf.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='auc',
        callbacks=[
            lgb.early_stopping(stopping_rounds=10),
            lgb.log_evaluation(period=50)
        ]
    )

    # 5. Evaluate
    print("\nEvaluating on Test Set...")
    probs = clf.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)

    results = {
        "model": "LightGBM",
        "split": args.split,
        "roc_auc": float(roc_auc_score(y_test, probs)),
        "pr_auc": float(average_precision_score(y_test, probs)),
        "f1": float(f1_score(y_test, preds)),
        "accuracy": float(accuracy_score(y_test, preds))
    }

    print("-" * 30)
    print(f"ROC-AUC: {results['roc_auc']:.4f}")
    print(f"PR-AUC : {results['pr_auc']:.4f}")
    print(f"F1     : {results['f1']:.4f}")
    print("-" * 30)

    # 6. Save Artifacts
    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    # Save results JSON
    out_file = os.path.join(ARTIFACT_DIR, f'lgbm_results_{args.split}.json')
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2)

    # Save the model
    model_file = os.path.join(ARTIFACT_DIR, f'lgbm_model_{args.split}.txt')
    clf.booster_.save_model(model_file)

    print(f"Saved results to {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, default='cold',
                        choices=['random', 'cold', 'scaffold'],
                        help="Split strategy (must match what you used for GAT)")
    parser.add_argument('--n_estimators', type=int, default=500,
                        help="Number of boosting rounds")

    args = parser.parse_args()
    run_lgbm(args)