import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import lmdb
import pandas as pd
import torch

from bioguard.artifacts import load_model_bundle, save_model_bundle
from bioguard.cohorts import cohort_manifest, require_matching_cohorts
from bioguard.data_loader import _standardize_smiles
from bioguard.featurizer import GraphFeaturizer
from bioguard.graph_cache import CONTRACT_KEY, check_cache_contract, close_readers, read_cache
from bioguard.model import BioGuardGAT
from bioguard.molecular import LEGACY_POLICY, STANDARDIZATION_POLICY, normalize_smiles, prepare_molecule
from tests.test_model import make_arm
from worker.celery_app import _validate_smiles


class PreprocessingTests(unittest.TestCase):
    def test_salt_is_identical_across_training_worker_and_graph_paths(self):
        salt = "CC(=O)[O-].[Na+]"
        training = _standardize_smiles(salt)
        worker, _ = _validate_smiles(salt, "a", STANDARDIZATION_POLICY)
        dashboard, _ = prepare_molecule(salt, STANDARDIZATION_POLICY)
        self.assertEqual(training, "CC(=O)O")
        self.assertEqual(worker, training)
        self.assertEqual(dashboard, training)
        featurizer = GraphFeaturizer()
        first = featurizer.smiles_to_graph(salt)
        second = featurizer.smiles_to_graph(training)
        torch.testing.assert_close(first.x, second.x)
        torch.testing.assert_close(first.edge_attr, second.edge_attr)
        torch.testing.assert_close(first.edge_index, second.edge_index)

    def test_stereoisomers_stay_distinct_and_policy_is_idempotent(self):
        left, right = "N[C@H](C)C(=O)O", "N[C@@H](C)C(=O)O"
        self.assertNotEqual(normalize_smiles(left), normalize_smiles(right))
        self.assertEqual(normalize_smiles(left), normalize_smiles(normalize_smiles(left)))

    def test_legacy_policy_and_input_limits(self):
        salt = "CC(=O)[O-].[Na+]"
        self.assertEqual(normalize_smiles(salt, LEGACY_POLICY), salt)
        for bad in ("", "not a molecule", None):
            with self.assertRaises(ValueError):
                prepare_molecule(bad)
        with self.assertRaisesRegex(ValueError, "heavy-atom"):
            prepare_molecule(salt, max_heavy_atoms=4)
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            normalize_smiles("CC", "future_policy")


class CacheTests(unittest.TestCase):
    def tearDown(self):
        close_readers()

    def test_old_or_modified_cache_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            with lmdb.open(root, map_size=1024**2) as env:
                with env.begin(write=True) as txn:
                    txn.put(b"CC", b"legacy graph")
                with self.assertRaisesRegex(ValueError, "contract mismatch"):
                    check_cache_contract(env, initialize=True)
        with tempfile.TemporaryDirectory() as root:
            with lmdb.open(root, map_size=1024**2) as env:
                contract = check_cache_contract(env, initialize=True)
                contract["rdkit"] = "different"
                with env.begin(write=True) as txn:
                    txn.put(CONTRACT_KEY, json.dumps(contract).encode())
                with self.assertRaisesRegex(ValueError, "contract mismatch"):
                    check_cache_contract(env)

    def test_multiple_datasets_share_a_validated_reader(self):
        with tempfile.TemporaryDirectory() as root:
            with lmdb.open(root, map_size=1024**2) as env:
                check_cache_contract(env, initialize=True)
            self.assertIs(read_cache(root), read_cache(Path(root)))
            close_readers()


class CohortTests(unittest.TestCase):
    def frame(self):
        return pd.DataFrame([
            {"sample_id": f"{split}_{i}", "observation": obs, "target": float(i),
             "smiles_a": "CC", "smiles_b": "CO", "split": split}
            for split in ("train", "val", "test")
            for i, obs in enumerate(("unlabeled", "positive"))
        ])

    def test_order_does_not_change_identity_but_exclusions_do(self):
        frame = self.frame()
        original = cohort_manifest(frame)
        require_matching_cohorts(original, cohort_manifest(frame.sample(frac=1, random_state=9)))
        with self.assertRaisesRegex(ValueError, "test cohorts"):
            require_matching_cohorts(original, cohort_manifest(frame.iloc[:-1]))
        changed = frame.copy()
        changed.loc[5, "smiles_b"] = "CCO"
        with self.assertRaises(ValueError):
            require_matching_cohorts(original, cohort_manifest(changed))
        with self.assertRaises(ValueError):
            require_matching_cohorts(None, None)

    def test_contradictory_labels_and_duplicate_ids_fail(self):
        frame = self.frame()
        frame.loc[0, "target"] = 1
        with self.assertRaisesRegex(ValueError, "disagree"):
            cohort_manifest(frame)
        with self.assertRaisesRegex(ValueError, "unique"):
            cohort_manifest(pd.concat([self.frame(), self.frame()]))

    def test_comparison_rejects_filtered_holdout_despite_same_source_hash(self):
        from bioguard.compare import load_comparable_results
        from bioguard.artifacts import ArtifactContractError
        evaluation = {
            "schema_version": 3,
            "task": "binary_any_reported_event_positive_unlabeled",
            "checkpoint_sha256": "model", "data_records_sha256": "same-source",
            "split": "scaffold_disjoint", "effective_cohorts": cohort_manifest(self.frame()),
        }
        baselines = copy.deepcopy(evaluation)
        baselines["effective_cohorts"] = cohort_manifest(self.frame().iloc[:-1])
        with patch("bioguard.compare.read_metadata", return_value={"checkpoint_sha256": "model"}), \
                patch("bioguard.compare._read_json", side_effect=[evaluation, baselines]):
            with self.assertRaisesRegex(ArtifactContractError, "test cohorts"):
                load_comparable_results()


class AblationTests(unittest.TestCase):
    def test_branch_is_preserved_on_reload_and_ignores_disabled_features(self):
        a = make_arm([[1., 0., 0.], [0., 1., 0.]], [[1., 0.], [1., 0.]], [1., 0.], "enzyme_a")
        b = make_arm([[0., 0., 1.], [0., 1., 0.]], [[1., 0.], [1., 0.]], [0., 1.], "enzyme_b")
        for branch in ("graph", "enzyme"):
            model = BioGuardGAT(node_dim=3, edge_dim=2, embedding_dim=8, heads=2,
                                enzyme_dim=2, branch=branch).eval()
            with tempfile.TemporaryDirectory() as root:
                save_model_bundle(model, root, {})
                loaded = load_model_bundle(root).model
            self.assertEqual(loaded.branch, branch)
            altered = a.clone()
            if branch == "graph":
                altered.enzyme_a += 50
            else:
                altered.x += 50
            with torch.no_grad():
                expected, _ = model(a, b)
                actual, _ = loaded(altered, b)
            torch.testing.assert_close(expected, actual)


if __name__ == "__main__":
    unittest.main()
