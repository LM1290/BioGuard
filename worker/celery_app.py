"""CPU worker for bounded RDKit graph featurization."""

import logging
import os

from celery import Celery
from celery.signals import worker_process_init

from bioguard.featurizer import GraphFeaturizer
from bioguard.molecular import LEGACY_POLICY, prepare_molecule
from bioguard.graph_cache import graph_contract

logger = logging.getLogger("bioguard.worker")

CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/0")
MAX_SMILES_LENGTH = int(os.getenv("BG_MAX_SMILES_LENGTH", "4096"))
MAX_HEAVY_ATOMS = int(os.getenv("BG_MAX_HEAVY_ATOMS", "256"))
TASK_SOFT_TIME_LIMIT = int(os.getenv("BG_TASK_SOFT_TIME_LIMIT", "45"))
TASK_TIME_LIMIT = int(os.getenv("BG_TASK_TIME_LIMIT", "60"))

celery_app = Celery(
    "bioguard_worker",
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND,
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_soft_time_limit=TASK_SOFT_TIME_LIMIT,
    task_time_limit=TASK_TIME_LIMIT,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    result_expires=300,
)

worker_featurizer = None


@worker_process_init.connect
def init_worker(**_kwargs):
    """
    Runs once per worker process on startup.
    This ensures we only initialize the GraphFeaturizer (and any heavy
    models/dictionaries it relies on) once per core, rather than per task.
    """
    global worker_featurizer
    logger.info("Initializing GraphFeaturizer in Celery worker process...")
    worker_featurizer = GraphFeaturizer()


def _graph_to_dict(data):
    """Converts a PyG Data object to a JSON-serializable dictionary."""
    return {
        "x": data.x.tolist(),
        "edge_index": data.edge_index.tolist(),
        "edge_attr": data.edge_attr.tolist(),
    }


def _validate_smiles(value, label, policy=LEGACY_POLICY):
    try:
        return prepare_molecule(value, policy, MAX_SMILES_LENGTH, MAX_HEAVY_ATOMS)
    except ValueError as exc:
        raise ValueError(f"{label}: {exc}") from exc


@celery_app.task(name="bioguard.worker.run_cpu_processing")
def run_cpu_processing(smiles_a, smiles_b, policy=LEGACY_POLICY):
    """
    The main Celery task triggered by the FastAPI gateway.
    It takes two SMILES strings, builds their graphs, canonicalizes them,
    and returns a JSON-friendly dictionary back to the API.
    """
    global worker_featurizer

    if worker_featurizer is None:
        return {"success": False, "error": "Worker featurizer not initialized."}

    try:
        graph_input_a, can_a = _validate_smiles(smiles_a, "drug_a_smiles", policy)
        graph_input_b, can_b = _validate_smiles(smiles_b, "drug_b_smiles", policy)
        g_a = worker_featurizer.smiles_to_graph(graph_input_a, policy=policy)
        g_b = worker_featurizer.smiles_to_graph(graph_input_b, policy=policy)

        return {
            "success": True,
            "standardization_policy": policy,
            "graph_contract": graph_contract(),
            "normalized_a": graph_input_a,
            "normalized_b": graph_input_b,
            "graph_a": _graph_to_dict(g_a),
            "graph_b": _graph_to_dict(g_b),
            "can_a": can_a,
            "can_b": can_b,
        }
    except ValueError as exc:
        logger.info("Rejected invalid prediction input: %s", exc)
        return {"success": False, "error": str(exc)}
    except Exception:
        logger.exception("Unexpected graph-featurization failure")
        return {
            "success": False,
            "error": "Molecular graph generation failed",
        }
