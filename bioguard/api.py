"""
BioGuard DDI Prediction API
UPDATED: v3.2 (Celery/Redis Integration & Alpha Telemetry)
"""

import torch
import os
import asyncio
import logging
import json
import joblib
import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, constr
from celery import Celery

# PyG Imports
from torch_geometric.data import Data, Batch

from .model import BioGuardGAT
from .enzyme import EnzymeManager

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
ARTIFACT_DIR = os.getenv("BG_ARTIFACT_DIR", os.path.join(BASE_DIR, 'artifacts'))
META_PATH = os.path.join(ARTIFACT_DIR, 'metadata.json')
MODEL_PATH = os.path.join(ARTIFACT_DIR, 'model.pt')
CALIBRATOR_PATH = os.path.join(ARTIFACT_DIR, 'calibrator.joblib')

logger = logging.getLogger("bioguard")

# --- CELERY CONFIGURATION ---
CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/0")

celery_app = Celery(
    "bioguard_api",
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing BioGuard API v3.2 (Celery Mode)...")

    # 1. Load Enzyme Manager (Strict Mode: No degrading allowed)
    manager = EnzymeManager(allow_degraded=False)
    app.state.enzyme_manager = manager

    # 2. Load Config & Verify Dimensions
    node_dim = 46  # Defaulting to 46 based on updated featurizer
    enzyme_dim = manager.vector_dim
    embedding_dim = 128
    heads = 4
    threshold = 0.5

    if os.path.exists(META_PATH):
        try:
            with open(META_PATH, 'r') as f:
                meta = json.load(f)
                node_dim = meta.get('node_dim', node_dim)
                embedding_dim = meta.get('embedding_dim', embedding_dim)
                heads = meta.get('heads', heads)

                model_enz_dim = meta.get('enzyme_dim', 0)

                # STRICT FAILURE: Do not patch or fake dimensions.
                if model_enz_dim > 0 and model_enz_dim != enzyme_dim:
                    logger.error(
                        f"CRITICAL MISMATCH: Model expects {model_enz_dim} enzyme features, "
                        f"but EnzymeManager provides {enzyme_dim}. Predictions will 422."
                    )
                    enzyme_dim = model_enz_dim

                threshold = meta.get('threshold', 0.5)
        except Exception as e:
            logger.error(f"Metadata load failed: {e}")

    # 3. Initialize Model
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    app.state.device = device

    logger.info(f"Model Config: Node={node_dim}, Enzyme={enzyme_dim}, Device={device_str}")

    model = BioGuardGAT(
        node_dim=node_dim,
        embedding_dim=embedding_dim,
        heads=heads,
        enzyme_dim=enzyme_dim
    ).to(device)

    if os.path.exists(MODEL_PATH):
        try:
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
            model.eval()
            app.state.model = model
            app.state.model_ready = True
            logger.info("Model weights loaded successfully.")
        except Exception as e:
            logger.critical(f"Model load failed: {e}")
            app.state.model_ready = False
    else:
        logger.warning(f"Model not found at {MODEL_PATH}")
        app.state.model_ready = False

    app.state.threshold = threshold
    app.state.calibrator = joblib.load(CALIBRATOR_PATH) if os.path.exists(CALIBRATOR_PATH) else None
    app.state.final_enzyme_dim = enzyme_dim

    yield
    # No ProcessPoolExecutor to shut down anymore


app = FastAPI(title="BioGuard API", version="3.2", lifespan=lifespan)


class PredictionRequest(BaseModel):
    drug_a_smiles: constr(min_length=1)
    drug_b_smiles: constr(min_length=1)


class PredictionResponse(BaseModel):
    raw_score: float
    calibrated_probability: float
    risk_level: str
    alpha_telemetry: float
    enzyme_status: str


@app.post("/predict", response_model=PredictionResponse)
async def predict(req: PredictionRequest, request: Request):
    st = request.app.state
    if not st.model_ready:
        raise HTTPException(status_code=503, detail="Model not initialized")

    # 1. Offload Heavy Lifting to Celery Worker
    try:
        # Submit the task to the queue mapped in docker-compose
        task = celery_app.send_task(
            "bioguard.worker.run_cpu_processing",
            args=[req.drug_a_smiles, req.drug_b_smiles]
        )

        # Await the result in a thread so we don't block the async event loop
        res = await asyncio.to_thread(task.get, timeout=30.0)
    except Exception as e:
        raise HTTPException(status_code=504, detail=f"Celery worker timeout or error: {str(e)}")

    if not res.get('success'):
        raise HTTPException(status_code=400, detail=res.get('error', 'Unknown worker error'))

    # 2. Look up Enzymes (Metabolic Prior)
    vec_a = st.enzyme_manager.get_by_smiles(res['can_a'])
    vec_b = st.enzyme_manager.get_by_smiles(res['can_b'])
    required_dim = st.final_enzyme_dim

    # --- STRICT DIMENSION GUARD ---
    if len(vec_a) != required_dim:
        raise HTTPException(
            status_code=422,
            detail=f"Metabolic extraction failed for Drug A. Expected {required_dim} dimensions, got {len(vec_a)}."
        )
    if len(vec_b) != required_dim:
        raise HTTPException(
            status_code=422,
            detail=f"Metabolic extraction failed for Drug B. Expected {required_dim} dimensions, got {len(vec_b)}."
        )

    has_enzymes = np.any(vec_a) or np.any(vec_b)

    # 3. Construct Batch
    def dict_to_data(d, enz_vec):
        data = Data(
            x=torch.tensor(d['x'], dtype=torch.float),
            edge_index=torch.tensor(d['edge_index'], dtype=torch.long),
            edge_attr=torch.tensor(d['edge_attr'], dtype=torch.float)
        )
        data.enzyme_a = torch.tensor(enz_vec, dtype=torch.float).unsqueeze(0)
        return data

    data_a = dict_to_data(res['graph_a'], vec_a)
    data_b = dict_to_data(res['graph_b'], vec_b)

    data_b.enzyme_b = data_b.enzyme_a
    del data_b.enzyme_a

    batch_a = Batch.from_data_list([data_a]).to(st.device)
    batch_b = Batch.from_data_list([data_b]).to(st.device)

    # 4. Inference (Unpacking Tuple from Hybrid Architecture)
    with torch.no_grad():
        logits, alpha = st.model(batch_a, batch_b)
        raw_prob = torch.sigmoid(logits).item()
        alpha_val = alpha.item()

    # 5. Calibration
    calib_prob = raw_prob
    if st.calibrator:
        calib_prob = float(st.calibrator.transform([raw_prob])[0])
        calib_prob = max(0.0, min(1.0, calib_prob))

    return {
        "raw_score": raw_prob,
        "calibrated_probability": calib_prob,
        "risk_level": "High" if calib_prob >= st.threshold else "Low",
        "alpha_telemetry": alpha_val,
        "enzyme_status": "Active" if has_enzymes else "Silent (No Features Found)"
    }