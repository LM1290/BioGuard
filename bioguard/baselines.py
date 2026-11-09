"""Reproducible structure baselines for the binary PU compatibility task."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from scipy import sparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

from .config import ARTIFACT_DIR, RANDOM_SEED, LMDB_DIR
from .cohorts import require_binary_splits
from .graph_cache import filter_cached_pairs
from .data_loader import load_twosides_binary_pu

BASELINE_RESULTS = Path(ARTIFACT_DIR) / "baseline_results.json"
FINGERPRINT_BITS = 1024


class SymmetricFingerprintFeaturizer:
    """Sparse ECFP pair operators invariant to drug order."""

    def __init__(self, n_bits: int = FINGERPRINT_BITS):
        self.generator = rdFingerprintGenerator.GetMorganGenerator(
            radius=2,
            fpSize=n_bits,
            includeChirality=True,
        )
        self.n_bits = n_bits
        self.cache: dict[str, np.ndarray] = {}

    def fingerprint(self, smiles: str) -> np.ndarray:
        if smiles not in self.cache:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise ValueError(f"Invalid standardized SMILES: {smiles}")
            self.cache[smiles] = np.asarray(
                self.generator.GetFingerprintAsNumPy(mol),
                dtype=np.uint8,
            )
        return self.cache[smiles]

    def transform(self, frame) -> sparse.csr_matrix:
        rows = []
        for row in frame.itertuples(index=False):
            fp_a = self.fingerprint(row.smiles_a)
            fp_b = self.fingerprint(row.smiles_b)
            features = np.concatenate(
                [
                    fp_a + fp_b,
                    np.bitwise_xor(fp_a, fp_b),
                    np.bitwise_and(fp_a, fp_b),
                ]
            )
            rows.append(sparse.csr_matrix(features.reshape(1, -1)))
        if not rows:
            return sparse.csr_matrix((0, self.n_bits * 3))
        return sparse.vstack(rows, format="csr")


def _ranking_metrics(targets, scores):
    return {
        "pu_roc_auc": float(roc_auc_score(targets, scores)),
        "pu_pr_auc": float(average_precision_score(targets, scores)),
    }


def run_baselines(args):
    """Fit baselines on exactly the same PU records and split manifest."""
    bundle = load_twosides_binary_pu(
        split_method=args.split,
        train_unlabeled_ratio=args.unlabeled_ratio,
        val_unlabeled_ratio=args.unlabeled_ratio,
        test_unlabeled_ratio=args.unlabeled_ratio,
        random_seed=RANDOM_SEED,
    )
    records = bundle.records
    if getattr(args, "cohort", "graph") == "graph":
        records = filter_cached_pairs(records, LMDB_DIR)
    effective_cohorts = require_binary_splits(records)
    train = records[records["split"] == "train"]
    val = records[records["split"] == "val"]
    test = records[records["split"] == "test"]

    featurizer = SymmetricFingerprintFeaturizer()
    x_train = featurizer.transform(train)
    x_val = featurizer.transform(val)
    x_test = featurizer.transform(test)
    y_train = train["target"].to_numpy()
    y_val = val["target"].to_numpy()
    y_test = test["target"].to_numpy()

    candidates = []
    for c_value in (0.01, 0.1, 1.0, 10.0):
        model = LogisticRegression(
            C=c_value,
            max_iter=2000,
            random_state=RANDOM_SEED,
        )
        model.fit(x_train, y_train)
        val_scores = model.predict_proba(x_val)[:, 1]
        candidates.append(
            (
                average_precision_score(y_val, val_scores),
                c_value,
                model,
            )
        )
    _, selected_c, logistic = max(candidates, key=lambda item: item[0])
    results = {
        "schema_version": 3,
        "task": "binary_any_reported_event_positive_unlabeled",
        "evaluation_semantics": (
            "Observed associations versus sampled unlabeled pairs; "
            "unlabeled pairs are not confirmed negatives."
        ),
        "split": args.split,
        "data_records_sha256": bundle.manifest["records_sha256"],
        "effective_cohorts": effective_cohorts,
        "coverage": len(records) / len(bundle.records),
        "models": {
            "logistic_regression_ecfp": {
                "name": "Logistic Regression on symmetric ECFP operators",
                "selected_c": selected_c,
                **_ranking_metrics(
                    y_test,
                    logistic.predict_proba(x_test)[:, 1],
                ),
            }
        },
    }

    if not args.quick:
        forest = RandomForestClassifier(
            n_estimators=300,
            max_depth=20,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=RANDOM_SEED,
        )
        forest.fit(x_train, y_train)
        results["models"]["random_forest_ecfp"] = {
            "name": "Random Forest on symmetric ECFP operators",
            **_ranking_metrics(
                y_test,
                forest.predict_proba(x_test)[:, 1],
            ),
        }

    BASELINE_RESULTS.parent.mkdir(parents=True, exist_ok=True)
    BASELINE_RESULTS.write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Baseline results written to {BASELINE_RESULTS}")
    return results
