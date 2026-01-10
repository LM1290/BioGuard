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

from .model import BioGuardNet
from .featurizer import BioFeaturizer

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
ARTIFACT_DIR = os.getenv("BG_ARTIFACT_DIR", os.path.join(BASE_DIR, 'artifacts'))
DATA_DIR = os.getenv("BG_DATA_DIR", os.path.join(BASE_DIR, 'data'))

# Paths
MODEL_PATH = os.path.join(ARTIFACT_DIR, 'model.pt')
CALIBRATOR_PATH = os.path.join(ARTIFACT_DIR, 'calibrator.joblib')
META_PATH = os.path.join(ARTIFACT_DIR, 'metadata.json')
CATALOG_PATH = os.path.join(DATA_DIR, 'drug_catalog.csv')

logger = logging.getLogger("bioguard")

# Module-level worker state (initialized once per worker process)
worker_featurizer: Optional[BioFeaturizer] = None


def worker_init():
    """Initialize worker-local state. Called once per process."""
    global worker_featurizer
    try:
        worker_featurizer = BioFeaturizer()
        logger.info("Worker initialized successfully")
    except Exception as e:
        logger.error(f"Worker init failed: {e}")
        raise


def run_cpu_inference(smiles_a, smiles_b, name_a, name_b):
    """
    Worker function for CPU-bound featurization.
    Returns dict with type markers, not exception objects.
    """
    global worker_featurizer
    if worker_featurizer is None:
        return {"success": False, "error": "Worker not initialized", "error_type": "InitializationError"}

    try:
        vec = worker_featurizer.featurize_pair(smiles_a, smiles_b, name_a, name_b)
        return {"success": True, "vector": vec.tolist()}
    except ValueError as e:
        return {"success": False, "error": str(e), "error_type": "ValueError"}
    except Exception as e:
        return {"success": False, "error": f"Internal featurization error: {type(e).__name__}",
                "error_type": "InternalError"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Proper lifecycle management with cleanup."""
    logger.info("Initializing BioGuard API v1.2...")

    # Process pool configuration
    cpu_count = os.cpu_count() or 1
    max_workers_env = os.getenv("BG_MAX_WORKERS")
    if max_workers_env:
        max_workers = min(int(max_workers_env), cpu_count)
    else:
        max_workers = min(2, cpu_count)

    logger.info(f"Starting ProcessPool with {max_workers} workers (CPU count: {cpu_count})")
    logger.warning("Note: Use ONLY 1 uvicorn worker when using internal process pool")

    app.state.executor = ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=worker_init
    )

    # Load Model
    device = torch.device("cpu")
    temp_feat = BioFeaturizer()
    model = BioGuardNet(input_dim=temp_feat.total_dim).to(device)

    app.state.model_ready = False
    app.state.calibrator = None
    app.state.threshold = 0.5

    if os.path.exists(MODEL_PATH):
        try:
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
            model.eval()
            app.state.model_ready = True
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.critical(f"Model load failed: {e}")
            model = None
            app.state.model_ready = False
    else:
        logger.warning(f"Model file not found at {MODEL_PATH}")
        model = None
        app.state.model_ready = False

    # Load Calibrator
    if os.path.exists(CALIBRATOR_PATH):
        try:
            app.state.calibrator = joblib.load(CALIBRATOR_PATH)
            logger.info("Calibrator loaded successfully")
        except Exception as e:
            logger.error(f"Calibrator load failed: {e}")
    else:
        logger.warning(f"Calibrator file not found at {CALIBRATOR_PATH}")

    # Load Metadata (Threshold)
    if os.path.exists(META_PATH):
        try:
            with open(META_PATH, 'r') as f:
                meta = json.load(f)
                app.state.threshold = meta.get('threshold', 0.5)
                logger.info(f"Loaded dynamic threshold: {app.state.threshold:.4f}")
        except Exception as e:
            logger.warning(f"Metadata load failed, using default threshold 0.5: {e}")
    else:
        logger.warning(f"Metadata file not found at {META_PATH}, using default threshold 0.5")

    app.state.model = model
    logger.info(f"BioGuard API ready (model_ready={app.state.model_ready})")

    yield

    logger.info("Shutting down BioGuard API...")
    app.state.executor.shutdown(wait=True, cancel_futures=False)
    logger.info("Shutdown complete")


app = FastAPI(
    title="BioGuard DDI Prediction API",
    version="1.2.0",
    lifespan=lifespan
)


class PredictionRequest(BaseModel):
    drug_a_name: constr(max_length=200)
    drug_b_name: constr(max_length=200)
    drug_a_smiles: constr(max_length=5000)
    drug_b_smiles: constr(max_length=5000)


class PredictionResponse(BaseModel):
    drug_a: str
    drug_b: str
    raw_score: float
    calibrated_probability: float
    risk_level: str
    threshold_used: float
    mechanism_hints: List[str]


@app.post("/predict", response_model=PredictionResponse)
async def predict(req: PredictionRequest, request: Request):
    """Predict drug-drug interaction probability."""
    st = request.app.state
    if not st.model_ready or st.model is None:
        raise HTTPException(status_code=503, detail="Model unavailable - service not fully initialized")

    loop = asyncio.get_running_loop()

    try:
        res = await loop.run_in_executor(
            st.executor,
            run_cpu_inference,
            req.drug_a_smiles,
            req.drug_b_smiles,
            req.drug_a_name,
            req.drug_b_name
        )
    except Exception as e:
        logger.error(f"Executor failure: {e}")
        raise HTTPException(status_code=500, detail="Inference service error")

    if not res["success"]:
        error_type = res.get("error_type", "Unknown")
        error_msg = res.get("error", "Unknown error")

        if error_type == "ValueError":
            raise HTTPException(status_code=400, detail=f"Invalid input: {error_msg}")
        elif error_type == "InitializationError":
            raise HTTPException(status_code=503, detail=error_msg)
        else:
            logger.error(f"Internal featurization error: {error_msg}")
            raise HTTPException(status_code=500, detail="Internal processing error")

    # Inference
    try:
        vec = torch.FloatTensor(res["vector"]).unsqueeze(0)
        with torch.no_grad():
            logits = st.model(vec)
            raw_prob = torch.sigmoid(logits).item()
    except Exception as e:
        logger.error(f"Model inference error: {e}")
        raise HTTPException(status_code=500, detail="Model inference error")

    # Calibration
    calibrated_prob = raw_prob
    if st.calibrator:
        try:
            calibrated_prob = float(st.calibrator.transform([raw_prob])[0])
            calibrated_prob = max(0.0, min(1.0, calibrated_prob))
        except Exception as e:
            logger.warning(f"Calibration failed, using raw probability: {e}")
            calibrated_prob = raw_prob

    is_risky = calibrated_prob >= st.threshold
    
    # Generic mechanism hints (no enzyme-specific features)
    hints = ["Based on structural similarity and molecular properties."]

    return {
        "drug_a": req.drug_a_name,
        "drug_b": req.drug_b_name,
        "raw_score": round(raw_prob, 4),
        "calibrated_probability": round(calibrated_prob, 4),
        "risk_level": "High" if is_risky else "Low",
        "threshold_used": st.threshold,
        "mechanism_hints": hints
    }


@app.get("/health")
async def health_check(request: Request):
    """Health check endpoint."""
    st = request.app.state
    model_loaded = getattr(st, 'model_ready', False) and getattr(st, 'model', None) is not None

    if model_loaded:
        return {
            "status": "healthy",
            "model_loaded": True
        }
    else:
        raise HTTPException(
            status_code=503,
            detail={
                "status": "degraded",
                "model_loaded": False,
                "message": "Model not loaded - service not ready"
            }
        )
