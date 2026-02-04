"""
BioGuard DDI Prediction API
UPDATED: v2.2 (Smart Enzyme Lookup)
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
from typing import List, Optional
from rdkit import Chem

# PyG Imports
from torch_geometric.data import Data, Batch

from .model import BioGuardGAT
from .featurizer import GraphFeaturizer
from .enzyme import EnzymeManager  # <--- NEW IMPORT

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
    logger.info("Initializing BioGuard API v2.2 (Smart Lookup)...")

    # 1. Setup Workers
    app.state.executor = ProcessPoolExecutor(max_workers=2, initializer=worker_init)

    # 2. Load Enzyme Manager (The Source of Truth)
    # Ensure allow_degraded=True so API doesn't crash if CSV is missing
    app.state.enzyme_manager = EnzymeManager(allow_degraded=True)

    # 3. Load Config
    node_dim = 41
    enzyme_dim = app.state.enzyme_manager.vector_dim # <--- Dynamic from CSV
    threshold = 0.5

    if os.path.exists(META_PATH):
        try:
            with open(META_PATH, 'r') as f:
                meta = json.load(f)
                node_dim = meta.get('node_dim', 41)
                # We prefer the CSV dimension, but check consistency
                saved_enz_dim = meta.get('enzyme_dim', 0)
                if saved_enz_dim != enzyme_dim and saved_enz_dim != 0:
                    logger.warning(f"Dim Mismatch: Model expects {saved_enz_dim}, CSV has {enzyme_dim}")
                    enzyme_dim = saved_enz_dim # Trust the model config

                threshold = meta.get('threshold', 0.5)
        except:
            pass

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
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
        model.eval()
        app.state.model = model
        app.state.model_ready = True
    else:
        app.state.model_ready = False

    app.state.threshold = threshold
    app.state.calibrator = joblib.load(CALIBRATOR_PATH) if os.path.exists(CALIBRATOR_PATH) else None

    yield
    app.state.executor.shutdown()


app = FastAPI(title="BioGuard API", version="2.2", lifespan=lifespan)

class PredictionRequest(BaseModel):
    drug_a_smiles: constr(min_length=1)
    drug_b_smiles: constr(min_length=1)

class PredictionResponse(BaseModel):
    raw_score: float
    calibrated_probability: float
    risk_level: str
    enzyme_status: str  # <--- New debug field

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

    # 2. Look up Enzymes (Main Thread - Fast Dictionary Lookups)
    # Using the standardized SMILES returned by the worker
    vec_a = st.enzyme_manager.get_by_smiles(res['can_a'])
    vec_b = st.enzyme_manager.get_by_smiles(res['can_b'])

    # Debug: Check if we found anything (non-zero)
    has_enzymes = np.any(vec_a) or np.any(vec_b)

    # 3. Construct Batch
    def dict_to_data(d, enz_vec):
        data = Data(
            x=torch.tensor(d['x'], dtype=torch.float),
            edge_index=torch.tensor(d['edge_index'], dtype=torch.long),
            edge_attr=torch.tensor(d['edge_attr'], dtype=torch.float)
        )
        # Reshape [Dim] -> [1, Dim] for batching
        data.enzyme_a = torch.tensor(enz_vec, dtype=torch.float).unsqueeze(0)
        # Note: Model expects enzyme_a on one, enzyme_b on other, or handled via Siamese logic
        # Based on BioGuardGAT.forward, we need enzyme_a on graph A and enzyme_b on graph B.
        return data

    data_a = dict_to_data(res['graph_a'], vec_a)
    # For data_b, we attach the B vector. The model code (BioGuardGAT.forward)
    # reads 'enzyme_a' from the first argument and 'enzyme_b' from the second.
    # Wait, in model.py:
    # vec_a = self.forward_one_arm(..., data_a.enzyme_a)
    # vec_b = self.forward_one_arm(..., data_b.enzyme_b)
    # So we must name them correctly!

    data_b = dict_to_data(res['graph_b'], vec_b)
    # Rename attribute for B to match model expectation
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
        "enzyme_status": "Active" if has_enzymes else "NCE-Mode"
    }