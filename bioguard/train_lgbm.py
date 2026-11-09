"""LightGBM ranking baseline for observed-positive versus unlabeled pairs."""

import argparse
import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from sklearn.metrics import average_precision_score, roc_auc_score

from bioguard.config import ARTIFACT_DIR, RANDOM_SEED
from bioguard.data_loader import load_twosides_binary_pu
from bioguard.enzyme import EnzymeManager


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split",
        default="scaffold_disjoint",
        choices=["pair_random", "drug_disjoint", "scaffold_disjoint"],
    )
    parser.add_argument("--unlabeled-ratio", type=float, default=1.0)
    args = parser.parse_args()

    bundle = load_twosides_binary_pu(
        split_method=args.split,
        train_unlabeled_ratio=args.unlabeled_ratio,
        val_unlabeled_ratio=args.unlabeled_ratio,
        test_unlabeled_ratio=args.unlabeled_ratio,
        random_seed=RANDOM_SEED,
    )
    frame = bundle.records
    enzyme_manager = EnzymeManager(allow_degraded=False)
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=2,
        fpSize=1024,
        includeChirality=True,
    )

    unique_smiles = pd.unique(
        pd.concat([frame["smiles_a"], frame["smiles_b"]])
    )
    fingerprint_map = {}
    for smiles in unique_smiles:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid standardized SMILES: {smiles}")
        fingerprint_map[smiles] = np.asarray(
            generator.GetFingerprintAsNumPy(mol),
            dtype=np.float32,
        )

    def vectorize(row):
        fp_a = fingerprint_map[row.smiles_a]
        fp_b = fingerprint_map[row.smiles_b]
        enzyme_a = enzyme_manager.get_by_smiles(row.smiles_a)
        enzyme_b = enzyme_manager.get_by_smiles(row.smiles_b)
        return np.concatenate(
            [
                fp_a + fp_b,
                np.abs(fp_a - fp_b),
                fp_a * fp_b,
                enzyme_a + enzyme_b,
                np.abs(enzyme_a - enzyme_b),
                enzyme_a * enzyme_b,
            ]
        )

    matrices = {}
    targets = {}
    for split in ("train", "val", "test"):
        subset = frame[frame["split"] == split]
        matrices[split] = np.stack(
            [vectorize(row) for row in subset.itertuples(index=False)]
        )
        targets[split] = subset["target"].to_numpy()

    classifier = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    classifier.fit(
        matrices["train"],
        targets["train"],
        eval_set=[(matrices["val"], targets["val"])],
        eval_metric="average_precision",
        callbacks=[lgb.early_stopping(stopping_rounds=30)],
    )
    scores = classifier.predict_proba(matrices["test"])[:, 1]

    artifact_dir = Path(ARTIFACT_DIR)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    model_path = artifact_dir / f"lgbm_pu_{args.split}.joblib"
    result_path = artifact_dir / f"lgbm_pu_{args.split}.json"
    joblib.dump(classifier, model_path)
    results = {
        "schema_version": 2,
        "task": "binary_any_reported_event_positive_unlabeled",
        "model": "LightGBM symmetric ECFP and legacy enzyme features",
        "split": args.split,
        "data_records_sha256": bundle.manifest["records_sha256"],
        "pu_roc_auc": float(roc_auc_score(targets["test"], scores)),
        "pu_pr_auc": float(average_precision_score(targets["test"], scores)),
        "warning": (
            "This supervised baseline distinguishes observed positives from "
            "sampled unlabeled pairs; it does not learn from confirmed negatives."
        ),
    }
    result_path.write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
