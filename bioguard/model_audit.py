"""Artifact and feature-coverage audit without mechanistic overclaims."""

import json
from pathlib import Path

from .artifacts import load_model_bundle
from .config import ARTIFACT_DIR, BASE_DIR
from .enzyme import EnzymeManager

OUTPUT_PATH = Path(ARTIFACT_DIR) / "model_audit.json"


def run_audit():
    bundle = load_model_bundle(
        ARTIFACT_DIR,
        supporting_file_root=BASE_DIR,
    )
    enzyme_manager = EnzymeManager(allow_degraded=False)
    metadata = bundle.metadata
    parameter_count = sum(
        parameter.numel()
        for parameter in bundle.model.parameters()
    )

    report = {
        "schema_version": 1,
        "checkpoint_sha256": bundle.checkpoint_sha256,
        "architecture": bundle.spec.as_dict(),
        "parameter_count": int(parameter_count),
        "scientific_task": metadata.get("scientific_task"),
        "training_provenance_status": metadata.get(
            "training_provenance_status"
        ),
        "legacy_unverified": bool(metadata.get("legacy_unverified", False)),
        "threshold_validated": bool(
            metadata.get("threshold_validated", False)
        ),
        "calibrator_enabled": bool(
            metadata.get("calibrator_enabled", False)
        ),
        "enzyme_feature_dimension": enzyme_manager.vector_dim,
        "enzyme_table_structure_count": len(enzyme_manager.smiles_map),
        "enzyme_fallback_model_available": bool(
            enzyme_manager.predictor.models
        ),
        "limitations": [
            "Alpha is a learned gate value, not a causal attribution.",
            "The enzyme table is closed-world and transductive.",
            (
                "Unseen molecules receive zero enzyme vectors when the "
                "fallback model is unavailable."
            ),
            (
                "A binary PU checkpoint does not predict event identity, "
                "direction, severity, incidence, or clinical causality."
            ),
        ],
    }
    OUTPUT_PATH.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


if __name__ == "__main__":
    run_audit()
