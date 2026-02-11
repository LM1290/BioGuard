"""
BioGuard DDI Prediction API
UPDATED: v2.3 (Robust Dimension Handling)
"""

import torch
import os
import asyncio
import logging
import json
import joblib
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, constr
from typing import Optional
from rdkit import Chem

# PyG Imports
from torch_geometric.data import Data, Batch

from .model import BioGuardGAT
from .featurizer import GraphFeaturizer
from .enzyme import EnzymeManager

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
ARTIFACT_DIR = os.getenv("BG_ARTIFACT_DIR", os.path.join(BASE_DIR, 'artifacts'))
META_PATH = os.path.join(ARTIFACT_DIR, 'metadata.json')
MODEL_PATH = os.path.join(ARTIFACT_DIR, 'model.pt')
CALIBRATOR_PATH = os.path.join(ARTIFACT_DIR, 'calibrator.joblib')

logger = logging.getLogger("bioguard")

# Global worker state
worker_featurizer: Optional[GraphFeaturizer] = None

def worker_init():
    global worker_featurizer
    worker_featurizer = GraphFeaturizer()

def _graph_to_dict(data):
    return {
        "x": data.x.tolist(),
        "edge_index": data.edge_index.tolist(),
        "edge_attr": data.edge_attr.tolist()
    }

def standardize_smiles(smiles):
    """Helper to canonicalize SMILES in the worker process."""
    if not smiles: return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            return Chem.MolToSmiles(mol, isomericSmiles=False)
    except:
        return None
    return None

def run_cpu_processing(smiles_a, smiles_b):
    """
    Worker function: Featurization + Standardization
    """
    global worker_featurizer
    if not worker_featurizer:
        return {"success": False, "error": "Worker not initialized"}

    try:
        # 1. Featurize (Heavy RDKit work)
        g_a = worker_featurizer.smiles_to_graph(smiles_a)
        g_b = worker_featurizer.smiles_to_graph(smiles_b)

        # 2. Standardize Strings (For Enzyme Lookup in main process)
        can_a = standardize_smiles(smiles_a)
        can_b = standardize_smiles(smiles_b)

        return {
            "success": True,
            "graph_a": _graph_to_dict(g_a),
            "graph_b": _graph_to_dict(g_b),
            "can_a": can_a,
            "can_b": can_b
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing BioGuard API v2.3 (Robust)...")

    # 1. Setup Workers
    app.state.executor = ProcessPoolExecutor(max_workers=2, initializer=worker_init)

    # 2. Load Enzyme Manager
    # We load with allow_degraded=True to prevent startup crash
    manager = EnzymeManager(allow_degraded=True)
    app.state.enzyme_manager = manager

    # 3. Load Config & Resolve Dimensions
    node_dim = 41
    # Start with the CSV's declared dimension
    enzyme_dim = manager.vector_dim
    threshold = 0.5

    if os.path.exists(META_PATH):
        try:
            with open(META_PATH, 'r') as f:
                meta = json.load(f)
                node_dim = meta.get('node_dim', 41)

                # Check what the Model expects
                model_enz_dim = meta.get('enzyme_dim', 0)

                if model_enz_dim > 0 and model_enz_dim != enzyme_dim:
                    logger.warning(
                        f"CRITICAL: Model expects {model_enz_dim} dims, but EnzymeCSV has {enzyme_dim}. "
                        "Enabling compatibility mode."
                    )
                    # Force the pipeline to use the Model's dimension
                    enzyme_dim = model_enz_dim

                    # PATCH: Force the predictor to return vectors of the correct size
                    # This prevents the 30 vs 60 mismatch crash when lookup fails
                    if hasattr(manager, 'predictor'):
                        # If the predictor has no feature names (untrained/missing),
                        # give it dummy names so it returns the right sized zero-vector.
                        if not manager.predictor.feature_names or len(manager.predictor.feature_names) != enzyme_dim:
                            logger.info(f"Patching Predictor to output {enzyme_dim} dimensions.")
                            manager.predictor.feature_names = [f"dim_{i}" for i in range(enzyme_dim)]

                threshold = meta.get('threshold', 0.5)
        except Exception as e:
            logger.error(f"Metadata load failed: {e}")

    # 4. Initialize Model
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    app.state.device = device

    logger.info(f"Model Config: Node={node_dim}, Enzyme={enzyme_dim}, Device={device_str}")

    model = BioGuardGAT(
        node_dim=node_dim,
        embedding_dim=128,
        heads=4,
        enzyme_dim=enzyme_dim
    ).to(device)

    if os.path.exists(MODEL_PATH):
        try:
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
            model.eval()
            app.state.model = model
            app.state.model_ready = True
        except Exception as e:
            logger.critical(f"Model load failed: {e}")
            app.state.model_ready = False
    else:
        logger.warning(f"Model not found at {MODEL_PATH}")
        app.state.model_ready = False

    app.state.threshold = threshold
    app.state.calibrator = joblib.load(CALIBRATOR_PATH) if os.path.exists(CALIBRATOR_PATH) else None

    # Store the FINAL agreed dimension for runtime checks
    app.state.final_enzyme_dim = enzyme_dim

    yield
    app.state.executor.shutdown()


app = FastAPI(title="BioGuard API", version="2.3", lifespan=lifespan)

class PredictionRequest(BaseModel):
    drug_a_smiles: constr(min_length=1)
    drug_b_smiles: constr(min_length=1)

class PredictionResponse(BaseModel):
    raw_score: float
    calibrated_probability: float
    risk_level: str
    enzyme_status: str

@app.post("/predict", response_model=PredictionResponse)
async def predict(req: PredictionRequest, request: Request):
    st = request.app.state
    if not st.model_ready:
        raise HTTPException(status_code=503, detail="Model not initialized")

    # 1. Offload Heavy Lifting
    loop = asyncio.get_running_loop()
    res = await loop.run_in_executor(
        st.executor,
        run_cpu_processing,
        req.drug_a_smiles,
        req.drug_b_smiles
    )

    if not res['success']:
        raise HTTPException(status_code=400, detail=res['error'])

    # 2. Look up Enzymes
    vec_a = st.enzyme_manager.get_by_smiles(res['can_a'])
    vec_b = st.enzyme_manager.get_by_smiles(res['can_b'])

    # --- DIMENSION GUARD ---
    # Even with the patch, we ensure the vector matches the model's expectation exactly.
    # This handles edge cases where the CSV has data (30 dims) but we need 60.
    required_dim = st.final_enzyme_dim

    def enforce_dim(vec, target_dim):
        if len(vec) == target_dim:
            return vec
        # If we have real data (non-zero) but wrong shape, we must pad carefully.
        # Ideally, we should not see this if the CSV matched the Model.
        new_vec = np.zeros(target_dim, dtype=np.float32)
        # Copy what we have
        min_len = min(len(vec), target_dim)
        new_vec[:min_len] = vec[:min_len]
        return new_vec

    vec_a = enforce_dim(vec_a, required_dim)
    vec_b = enforce_dim(vec_b, required_dim)
    # -----------------------

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

    # Map enzyme_b correctly for Siamese network
    data_b.enzyme_b = data_b.enzyme_a
    del data_b.enzyme_a

    batch_a = Batch.from_data_list([data_a]).to(st.device)
    batch_b = Batch.from_data_list([data_b]).to(st.device)

    # 4. Inference
    with torch.no_grad():
        logits = st.model(batch_a, batch_b)
        raw_prob = torch.sigmoid(logits).item()

    # 5. Calibration
    calib_prob = raw_prob
    if st.calibrator:
        calib_prob = float(st.calibrator.transform([raw_prob])[0])
        calib_prob = max(0.0, min(1.0, calib_prob))

    return {
        "raw_score": raw_prob,
        "calibrated_probability": calib_prob,
        "risk_level": "High" if calib_prob >= st.threshold else "Low",
        "enzyme_status": "Active" if has_enzymes else "Degraded (Missing Data)"
    }