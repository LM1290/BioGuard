"""
BioGuard DDI Prediction API
UPDATED: v2.1 (Dynamic Device Support)
- Auto-detects CUDA/CPU
- Moves input tensors to correct device automatically
"""

import torch
import os
import asyncio
import logging
import json
import joblib
from concurrent.futures import ProcessPoolExecutor
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, constr
from typing import List, Optional

# PyG Imports
from torch_geometric.data import Data, Batch

from .model import BioGuardGAT
from .featurizer import GraphFeaturizer

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
ARTIFACT_DIR = os.getenv("BG_ARTIFACT_DIR", os.path.join(BASE_DIR, 'artifacts'))

# Paths
MODEL_PATH = os.path.join(ARTIFACT_DIR, 'model.pt')
CALIBRATOR_PATH = os.path.join(ARTIFACT_DIR, 'calibrator.joblib')
META_PATH = os.path.join(ARTIFACT_DIR, 'metadata.json')

logger = logging.getLogger("bioguard")

# Global worker state
worker_featurizer: Optional[GraphFeaturizer] = None


def worker_init():
    """Initialize worker-local featurizer."""
    global worker_featurizer
    worker_featurizer = GraphFeaturizer()


def _graph_to_dict(data):
    """Serialize PyG Data object to dict for process transfer."""
    return {
        "x": data.x.tolist(),
        "edge_index": data.edge_index.tolist(),
        "edge_attr": data.edge_attr.tolist()
    }


def run_cpu_featurization(smiles_a, smiles_b):
    """
    Worker function: Converts SMILES -> Graph Dicts.
    """
    global worker_featurizer
    if not worker_featurizer:
        return {"success": False, "error": "Worker not initialized"}

    try:
        # Featurize separately
        g_a = worker_featurizer.smiles_to_graph(smiles_a)
        g_b = worker_featurizer.smiles_to_graph(smiles_b)

        return {
            "success": True,
            "graph_a": _graph_to_dict(g_a),
            "graph_b": _graph_to_dict(g_b)
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing BioGuard API v2.1 (Dynamic Device)...")

    # 1. Setup Workers
    app.state.executor = ProcessPoolExecutor(max_workers=2, initializer=worker_init)

    # 2. Load Model Configuration
    node_dim = 41
    threshold = 0.5

    if os.path.exists(META_PATH):
        try:
            with open(META_PATH, 'r') as f:
                meta = json.load(f)
                node_dim = meta.get('node_dim', 41)
                threshold = meta.get('threshold', 0.5)
                logger.info(f"Config loaded: node_dim={node_dim}, threshold={threshold}")
        except:
            logger.warning("Metadata load failed, using defaults.")

    # 3. Initialize Model with DYNAMIC DEVICE
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    logger.info(f"Inference Device: {device_str.upper()}")

    app.state.device = device # Store device in state for request handling

    model = BioGuardGAT(node_dim=node_dim, embedding_dim=128, heads=4).to(device)

    if os.path.exists(MODEL_PATH):
        try:
            # map_location ensures weights load correctly even if trained on different device
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
            model.eval()
            app.state.model = model
            app.state.model_ready = True
            logger.info("BioGuardGAT loaded successfully.")
        except Exception as e:
            logger.critical(f"Model weight mismatch: {e}")
            app.state.model_ready = False
    else:
        logger.error("Model file not found!")
        app.state.model_ready = False

    app.state.threshold = threshold

    # 4. Load Calibrator
    app.state.calibrator = None
    if os.path.exists(CALIBRATOR_PATH):
        app.state.calibrator = joblib.load(CALIBRATOR_PATH)

    yield
    app.state.executor.shutdown()


app = FastAPI(title="BioGuard API", version="2.1", lifespan=lifespan)


class PredictionRequest(BaseModel):
    drug_a_smiles: constr(min_length=1)
    drug_b_smiles: constr(min_length=1)


class PredictionResponse(BaseModel):
    raw_score: float
    calibrated_probability: float
    risk_level: str


@app.post("/predict", response_model=PredictionResponse)
async def predict(req: PredictionRequest, request: Request):
    st = request.app.state
    if not st.model_ready:
        raise HTTPException(status_code=503, detail="Model not initialized")

    # 1. Offload Featurization (CPU Bound)
    loop = asyncio.get_running_loop()
    res = await loop.run_in_executor(
        st.executor,
        run_cpu_featurization,
        req.drug_a_smiles,
        req.drug_b_smiles
    )

    if not res['success']:
        raise HTTPException(status_code=400, detail=res['error'])

    # 2. Reconstruct Graphs & MOVE TO DEVICE
    try:
        def dict_to_data(d):
            return Data(
                x=torch.tensor(d['x'], dtype=torch.float),
                edge_index=torch.tensor(d['edge_index'], dtype=torch.long),
                edge_attr=torch.tensor(d['edge_attr'], dtype=torch.float)
            )

        data_a = dict_to_data(res['graph_a'])
        data_b = dict_to_data(res['graph_b'])

        # Create Batch (Size 1)
        # IMPORTANT: .to(st.device) moves the batch to GPU if available
        batch_a = Batch.from_data_list([data_a]).to(st.device)
        batch_b = Batch.from_data_list([data_b]).to(st.device)

        # 3. Inference
        with torch.no_grad():
            logits = st.model(batch_a, batch_b)
            raw_prob = torch.sigmoid(logits).item()

        # 4. Calibration
        calib_prob = raw_prob
        if st.calibrator:
            calib_prob = float(st.calibrator.transform([raw_prob])[0])
            calib_prob = max(0.0, min(1.0, calib_prob))

        return {
            "raw_score": raw_prob,
            "calibrated_probability": calib_prob,
            "risk_level": "High" if calib_prob >= st.threshold else "Low"
        }

    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail="Inference failed")
