import copy
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

from bioguard.artifacts import (
    ArtifactContractError,
    ModelSpec,
    load_model_bundle,
    read_metadata,
    save_model_bundle,
    validate_metadata,
)
from bioguard.model import BioGuardGAT

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
ARTIFACT_DIR = REPOSITORY_ROOT / "artifacts"


class ArtifactContractTests(unittest.TestCase):
    def test_package_import_is_side_effect_free(self):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import sys; import bioguard; "
                    "assert 'bioguard.api' not in sys.modules; "
                    "assert 'celery' not in sys.modules"
                ),
            ],
            cwd=REPOSITORY_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_legacy_checkpoint_matches_declared_contract(self):
        bundle = load_model_bundle(ARTIFACT_DIR)
        self.assertEqual(
            bundle.spec,
            ModelSpec(
                node_dim=46,
                edge_dim=8,
                enzyme_dim=60,
                embedding_dim=128,
                heads=4,
            ),
        )
        self.assertFalse(bundle.metadata["threshold_validated"])
        self.assertFalse(bundle.metadata["calibrator_enabled"])

    def test_metadata_mismatch_is_rejected(self):
        metadata = read_metadata(ARTIFACT_DIR)
        altered = copy.deepcopy(metadata)
        altered["node_dim"] = metadata["node_dim"] + 1
        with self.assertRaisesRegex(
            ArtifactContractError,
            "architecture mismatch",
        ):
            validate_metadata(
                altered,
                ModelSpec(
                    node_dim=46,
                    edge_dim=8,
                    enzyme_dim=60,
                    embedding_dim=128,
                    heads=4,
                ),
            )

    def test_boolean_dimension_is_rejected(self):
        metadata = read_metadata(ARTIFACT_DIR)
        altered = copy.deepcopy(metadata)
        altered["heads"] = True
        with self.assertRaisesRegex(
            ArtifactContractError,
            "positive integers",
        ):
            validate_metadata(
                altered,
                ModelSpec(
                    node_dim=46,
                    edge_dim=8,
                    enzyme_dim=60,
                    embedding_dim=128,
                    heads=4,
                ),
            )

    def test_saved_bundle_round_trips(self):
        model = BioGuardGAT(
            node_dim=5,
            edge_dim=2,
            enzyme_dim=3,
            embedding_dim=8,
            heads=2,
        )
        with tempfile.TemporaryDirectory() as directory:
            metadata = save_model_bundle(
                model,
                directory,
                {
                    "threshold": 0.5,
                    "threshold_validated": False,
                    "calibrator_enabled": False,
                    "split_type": "random",
                },
            )
            loaded = load_model_bundle(directory)

        self.assertEqual(loaded.spec.node_dim, 5)
        self.assertEqual(loaded.spec.enzyme_dim, 3)
        self.assertEqual(
            loaded.checkpoint_sha256,
            metadata["checkpoint_sha256"],
        )
        for expected, actual in zip(
            model.state_dict().values(),
            loaded.model.state_dict().values(),
        ):
            self.assertTrue(torch.equal(expected.cpu(), actual.cpu()))


if __name__ == "__main__":
    unittest.main()
