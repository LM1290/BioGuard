import copy
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

from bioguard.api import app
from bioguard.featurizer import GraphFeaturizer
from bioguard.molecular import STANDARDIZATION_POLICY
from worker.celery_app import run_cpu_processing


class PredictionPipelineTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        self.client.__enter__()
        # Exercise the new input policy using a compatible loaded architecture;
        # this test does not validate the legacy weights on normalized inputs.
        app.state.metadata = {**app.state.metadata, "standardization_policy": STANDARDIZATION_POLICY}
        self.worker = patch("worker.celery_app.worker_featurizer", GraphFeaturizer())
        self.worker.start()

    def tearDown(self):
        self.worker.stop()
        self.client.__exit__(None, None, None)

    def test_new_policy_crosses_api_worker_boundary_and_reports_coverage(self):
        def dispatch(name, args):
            self.assertEqual(args[2], STANDARDIZATION_POLICY)
            return SimpleNamespace(get=lambda timeout: run_cpu_processing.run(*args))

        with patch("bioguard.api.celery_app.send_task", side_effect=dispatch):
            result = self.client.post("/predict", json={
                "drug_a_smiles": "CC(=O)[O-].[Na+]", "drug_b_smiles": "CCO",
            })
        self.assertEqual(result.status_code, 200, result.text)
        self.assertEqual(result.json()["normalized_smiles"]["drug_a"], "CC(=O)O")
        self.assertEqual(set(result.json()["enzyme_coverage"]), {"drug_a", "drug_b"})
        self.assertIsNone(result.json()["threshold_decision"])

    def test_timeout_revokes_and_returns_504(self):
        task = Mock()
        task.get.side_effect = TimeoutError("worker unavailable")
        with patch("bioguard.api.celery_app.send_task", return_value=task):
            result = self.client.post("/predict", json={"drug_a_smiles": "CC", "drug_b_smiles": "CO"})
        self.assertEqual(result.status_code, 504)
        task.revoke.assert_called_once_with(terminate=False)

    def test_mismatched_policy_and_malformed_graph_fail_closed(self):
        valid = run_cpu_processing.run("CC", "CO", STANDARDIZATION_POLICY)
        mismatched = {**valid, "standardization_policy": "legacy_raw_v1"}
        malformed = copy.deepcopy(valid)
        malformed["graph_a"]["edge_index"] = [[0], [9999]]
        app.state.metadata["graph_contract"] = valid["graph_contract"]
        wrong_contract = {**valid, "graph_contract": {}}
        for payload in (mismatched, malformed, wrong_contract):
            task = SimpleNamespace(get=lambda timeout: payload)
            with patch("bioguard.api.celery_app.send_task", return_value=task):
                result = self.client.post("/predict", json={"drug_a_smiles": "CC", "drug_b_smiles": "CO"})
            self.assertEqual(result.status_code, 502, result.text)

    def test_empty_input_rejected_before_queue(self):
        with patch("bioguard.api.celery_app.send_task") as dispatch:
            result = self.client.post("/predict", json={"drug_a_smiles": " ", "drug_b_smiles": "CC"})
        self.assertEqual(result.status_code, 422)
        dispatch.assert_not_called()


if __name__ == "__main__":
    unittest.main()
