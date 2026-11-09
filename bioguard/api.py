"""
BioGuard DDI Prediction API
Research-use API with strict artifact compatibility checks.
"""

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import torch
from celery import Celery
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, constr
from torch_geometric.data import Batch, Data

from .artifacts import (
    ArtifactContractError,
    load_calibrator,
    load_model_bundle,
)
from .enzyme import EnzymeManager
from .molecular import policy_for_metadata

BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACT_DIR = Path(os.getenv("BG_ARTIFACT_DIR", BASE_DIR / "artifacts"))

logger = logging.getLogger("bioguard")

CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/0")
PREDICTION_TIMEOUT_SECONDS = float(os.getenv("BG_PREDICTION_TIMEOUT_SECONDS", "30"))
MAX_SMILES_LENGTH = int(os.getenv("BG_MAX_SMILES_LENGTH", "4096"))

celery_app = Celery(
    "bioguard_api",
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND,
)
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing BioGuard research API v4 preview...")

    app.state.model_ready = False
    app.state.model_error = None
    app.state.model = None
    app.state.calibrator = None
    app.state.calibration_applied = False
    app.state.metadata = {}
    app.state.model_version = "unavailable"
    app.state.scientific_task = "unknown"
    app.state.threshold = 0.5
    app.state.threshold_validated = False
    app.state.final_enzyme_dim = None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    app.state.device = device

    try:
        manager = EnzymeManager(allow_degraded=False)
        bundle = load_model_bundle(
            ARTIFACT_DIR,
            device=device,
            supporting_file_root=BASE_DIR,
        )
        if manager.vector_dim != bundle.spec.enzyme_dim:
            raise ArtifactContractError(
                "Enzyme feature schema does not match checkpoint: "
                f"manager={manager.vector_dim}, checkpoint={bundle.spec.enzyme_dim}"
            )

        calibrator = load_calibrator(ARTIFACT_DIR, bundle.metadata)
        threshold = float(bundle.metadata.get("threshold", 0.5))
        if not 0.0 <= threshold <= 1.0:
            raise ArtifactContractError(f"Threshold must be within [0, 1], got {threshold}")

        app.state.enzyme_manager = manager
        app.state.model = bundle.model
        app.state.model_ready = True
        app.state.metadata = bundle.metadata
        app.state.threshold = threshold
        app.state.threshold_validated = bool(
            bundle.metadata.get("threshold_validated", False)
        )
        app.state.calibrator = calibrator
        app.state.calibration_applied = calibrator is not None
        app.state.final_enzyme_dim = bundle.spec.enzyme_dim
        app.state.model_version = bundle.checkpoint_sha256[:12]
        app.state.scientific_task = bundle.metadata.get(
            "scientific_task",
            "unknown_legacy",
        )
        logger.info(
            "Model ready: version=%s node=%d edge=%d enzyme=%d device=%s",
            app.state.model_version,
            bundle.spec.node_dim,
            bundle.spec.edge_dim,
            bundle.spec.enzyme_dim,
            device,
        )
    except (ArtifactContractError, FileNotFoundError, ValueError) as exc:
        app.state.model_error = str(exc)
        logger.critical("Model initialization failed: %s", exc)
    except Exception:
        app.state.model_error = "Unexpected model initialization failure"
        logger.exception("Unexpected model initialization failure")

    yield


app = FastAPI(
    title="BioGuard Research API",
    version="4.0.0a1",
    lifespan=lifespan,
)


class PredictionRequest(BaseModel):
    drug_a_smiles: constr(
        strip_whitespace=True, min_length=1, max_length=MAX_SMILES_LENGTH
    )
    drug_b_smiles: constr(
        strip_whitespace=True, min_length=1, max_length=MAX_SMILES_LENGTH
    )


class PredictionResponse(BaseModel):
    raw_model_score: float
    screening_score: float
    threshold_decision: str | None
    gate_value: float
    enzyme_status: str
    enzyme_coverage: dict[str, str]
    normalized_smiles: dict[str, str]
    standardization_policy: str
    calibration_applied: bool
    threshold_validated: bool
    model_version: str
    scientific_task: str
    warning: str


@app.get("/health")
async def health():
    """Liveness: the API process is accepting requests."""
    return {"status": "ok"}


@app.get("/ready")
async def ready(request: Request):
    """Readiness: the model and its feature schema passed contract checks."""
    state = request.app.state
    if not state.model_ready:
        raise HTTPException(
            status_code=503,
            detail=state.model_error or "Model is not initialized",
        )
    return {
        "status": "ready",
        "model_version": state.model_version,
        "threshold_validated": state.threshold_validated,
        "calibration_applied": state.calibration_applied,
        "scientific_task": state.scientific_task,
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(req: PredictionRequest, request: Request):
    st = request.app.state
    if not st.model_ready:
        raise HTTPException(
            status_code=503,
            detail=st.model_error or "Model not initialized",
        )

    task = None
    try:
        task = celery_app.send_task(
            "bioguard.worker.run_cpu_processing",
            args=[req.drug_a_smiles, req.drug_b_smiles, policy_for_metadata(st.metadata)],
        )
        res = await asyncio.to_thread(
            task.get,
            timeout=PREDICTION_TIMEOUT_SECONDS,
        )
    except Exception as exc:
        if task is not None:
            try:
                await asyncio.to_thread(task.revoke, terminate=False)
            except Exception:
                logger.warning("Could not revoke failed prediction task", exc_info=True)
        logger.warning("Prediction worker failed: %s", exc)
        raise HTTPException(
            status_code=504,
            detail="Prediction worker timed out or failed",
        ) from exc

    if not isinstance(res, dict) or not res.get("success"):
        detail = (
            res.get("error", "Unknown worker error")
            if isinstance(res, dict)
            else "Invalid worker response"
        )
        raise HTTPException(status_code=400, detail=detail)
    if res.get("standardization_policy") != policy_for_metadata(st.metadata):
        raise HTTPException(status_code=502, detail="Worker preprocessing policy mismatch")
    if "graph_contract" in st.metadata and res.get("graph_contract") != st.metadata["graph_contract"]:
        raise HTTPException(status_code=502, detail="Worker graph contract mismatch")
    if not res.get("normalized_a") or not res.get("normalized_b"):
        raise HTTPException(status_code=502, detail="Worker normalized structures are missing")
    if not res.get("can_a") or not res.get("can_b"):
        raise HTTPException(
            status_code=400,
            detail="Worker could not canonicalize both SMILES",
        )

    vec_a = st.enzyme_manager.get_by_smiles(res["can_a"])
    vec_b = st.enzyme_manager.get_by_smiles(res["can_b"])
    required_dim = st.final_enzyme_dim

    if len(vec_a) != required_dim:
        raise HTTPException(
            status_code=422,
            detail=(
                "Metabolic extraction failed for Drug A. "
                f"Expected {required_dim} dimensions, got {len(vec_a)}."
            ),
        )
    if len(vec_b) != required_dim:
        raise HTTPException(
            status_code=422,
            detail=(
                "Metabolic extraction failed for Drug B. "
                f"Expected {required_dim} dimensions, got {len(vec_b)}."
            ),
        )
    if not np.isfinite(vec_a).all() or not np.isfinite(vec_b).all():
        raise HTTPException(
            status_code=422,
            detail="Metabolic extraction produced non-finite features",
        )

    enzyme_coverage = {
        "drug_a": st.enzyme_manager.coverage_status(res["can_a"]),
        "drug_b": st.enzyme_manager.coverage_status(res["can_b"]),
    }

    def dict_to_data(d, enz_vec):
        if not isinstance(d, dict):
            raise HTTPException(status_code=502, detail="Invalid graph payload")
        try:
            x = torch.tensor(d["x"], dtype=torch.float)
            edge_index = torch.tensor(d["edge_index"], dtype=torch.long)
            edge_attr = torch.tensor(d["edge_attr"], dtype=torch.float)
        except (KeyError, TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=502,
                detail="Invalid graph payload",
            ) from exc
        if (
            x.ndim != 2
            or x.shape[0] == 0
            or x.shape[1] != st.model.atom_encoder[0].in_features
            or edge_index.ndim != 2
            or edge_index.shape[0] != 2
            or edge_attr.ndim != 2
            or edge_attr.shape[1] != st.model.conv1.edge_dim
            or edge_attr.shape[0] != edge_index.shape[1]
        ):
            raise HTTPException(
                status_code=502,
                detail="Worker graph schema does not match the loaded model",
            )
        if not torch.isfinite(x).all() or not torch.isfinite(edge_attr).all():
            raise HTTPException(
                status_code=502,
                detail="Worker graph contains non-finite features",
            )
        if edge_index.numel() and (
            edge_index.min() < 0 or edge_index.max() >= x.shape[0]
        ):
            raise HTTPException(
                status_code=502,
                detail="Worker graph contains invalid atom indices",
            )
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
        )
        data.enzyme_a = torch.tensor(enz_vec, dtype=torch.float).unsqueeze(0)
        return data

    data_a = dict_to_data(res.get("graph_a"), vec_a)
    data_b = dict_to_data(res.get("graph_b"), vec_b)

    data_b.enzyme_b = data_b.enzyme_a
    del data_b.enzyme_a

    batch_a = Batch.from_data_list([data_a]).to(st.device)
    batch_b = Batch.from_data_list([data_b]).to(st.device)

    with torch.no_grad():
        logits, alpha = st.model(batch_a, batch_b)
        raw_prob = torch.sigmoid(logits).item()
        alpha_val = alpha.item()
    if not np.isfinite(raw_prob) or not np.isfinite(alpha_val):
        raise HTTPException(
            status_code=500,
            detail="Model produced a non-finite output",
        )

    calib_prob = raw_prob
    if st.calibrator is not None:
        calib_prob = float(st.calibrator.transform([raw_prob])[0])
        if not np.isfinite(calib_prob):
            raise HTTPException(
                status_code=500,
                detail="Calibrator produced a non-finite output",
            )
        calib_prob = max(0.0, min(1.0, calib_prob))

    if st.threshold_validated:
        threshold_decision = "High" if calib_prob >= st.threshold else "Low"
        warning = "Research-use prediction; not medical advice."
    else:
        threshold_decision = None
        warning = (
            "No validated operating threshold is attached to this model. "
            "The score is for research use only and is not a clinical risk estimate."
        )

    return {
        "raw_model_score": raw_prob,
        "screening_score": calib_prob,
        "threshold_decision": threshold_decision,
        "gate_value": alpha_val,
        "enzyme_status": (
            "Legacy table coverage for both drugs; zero entries do not establish inactivity"
            if all(value == "legacy_table" for value in enzyme_coverage.values())
            else "Incomplete enzyme coverage; inspect per-drug provenance"
        ),
        "enzyme_coverage": enzyme_coverage,
        "normalized_smiles": {"drug_a": res["normalized_a"], "drug_b": res["normalized_b"]},
        "standardization_policy": res["standardization_policy"],
        "calibration_applied": st.calibration_applied,
        "threshold_validated": st.threshold_validated,
        "model_version": st.model_version,
        "scientific_task": st.scientific_task,
        "warning": warning,
    }
