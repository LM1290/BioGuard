"""
BioGuard Celery Worker Application
Handles CPU-intensive RDKit conformer generation and PyG graph featurization.
"""

import os
import logging
from celery import Celery
from celery.signals import worker_process_init
from rdkit import Chem

# Import the GraphFeaturizer from your local package
from bioguard.featurizer import GraphFeaturizer

logger = logging.getLogger("bioguard.worker")

# --- CELERY CONFIGURATION ---
CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/0")

celery_app = Celery(
    "bioguard_worker",
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND
)

# Optional: Configure Celery serialization to handle standard JSON safely
celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
)

# Global worker state
worker_featurizer = None


@worker_process_init.connect
def init_worker(**kwargs):
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
        "edge_attr": data.edge_attr.tolist()
    }


def standardize_smiles(smiles):
    """Helper to canonicalize SMILES strings safely."""
    if not smiles:
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            return Chem.MolToSmiles(mol, isomericSmiles=False)
    except Exception as e:
        logger.warning(f"Failed to standardize SMILES {smiles}: {e}")
        return None
    return None


@celery_app.task(name="bioguard.worker.run_cpu_processing")
def run_cpu_processing(smiles_a, smiles_b):
    """
    The main Celery task triggered by the FastAPI gateway.
    It takes two SMILES strings, builds their graphs, canonicalizes them,
    and returns a JSON-friendly dictionary back to the API.
    """
    global worker_featurizer

    if not worker_featurizer:
        return {"success": False, "error": "Worker featurizer not initialized."}

    try:
        # 1. Generate 3D Conformers and Extract Features
        g_a = worker_featurizer.smiles_to_graph(smiles_a)
        g_b = worker_featurizer.smiles_to_graph(smiles_b)

        # 2. Canonicalize SMILES for Enzyme Manager lookups
        can_a = standardize_smiles(smiles_a)
        can_b = standardize_smiles(smiles_b)

        # 3. Serialize and Return Result to Redis backend
        return {
            "success": True,
            "graph_a": _graph_to_dict(g_a),
            "graph_b": _graph_to_dict(g_b),
            "can_a": can_a,
            "can_b": can_b
        }
    except Exception as e:
        logger.error(f"Error processing SMILES pair ({smiles_a}, {smiles_b}): {e}", exc_info=True)
        return {"success": False, "error": str(e)}