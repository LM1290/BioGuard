"""Versioned model-artifact loading and compatibility validation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch


class ArtifactContractError(RuntimeError):
    """Raised when serialized artifacts cannot be used together safely."""


@dataclass(frozen=True)
class ModelSpec:
    node_dim: int
    edge_dim: int
    enzyme_dim: int
    embedding_dim: int
    heads: int

    def as_dict(self) -> dict[str, int]:
        return {
            "node_dim": self.node_dim,
            "edge_dim": self.edge_dim,
            "enzyme_dim": self.enzyme_dim,
            "embedding_dim": self.embedding_dim,
            "heads": self.heads,
        }


@dataclass(frozen=True)
class ModelBundle:
    model: torch.nn.Module
    spec: ModelSpec
    metadata: dict[str, Any]
    checkpoint_path: Path
    checkpoint_sha256: str


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_metadata(artifact_dir: str | Path) -> dict[str, Any]:
    path = Path(artifact_dir) / "metadata.json"
    if not path.is_file():
        raise ArtifactContractError(f"Required model metadata is missing: {path}")

    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactContractError(f"Could not read model metadata {path}: {exc}") from exc

    if not isinstance(metadata, dict):
        raise ArtifactContractError(f"Model metadata must be a JSON object: {path}")
    if metadata.get("schema_version") != 1:
        raise ArtifactContractError(
            f"Unsupported metadata schema_version={metadata.get('schema_version')!r}; expected 1"
        )
    if metadata.get("model_class") != "BioGuardGAT":
        raise ArtifactContractError(
            f"Unsupported model_class={metadata.get('model_class')!r}; expected 'BioGuardGAT'"
        )
    return metadata


def _required_tensor(
    state_dict: Mapping[str, torch.Tensor], key: str
) -> torch.Tensor:
    value = state_dict.get(key)
    if not isinstance(value, torch.Tensor):
        raise ArtifactContractError(f"Checkpoint is missing required tensor {key!r}")
    return value


def infer_model_spec(state_dict: Mapping[str, torch.Tensor]) -> ModelSpec:
    """Infer the BioGuardGAT constructor contract from tensor shapes."""
    atom_weight = _required_tensor(state_dict, "atom_encoder.0.weight")
    conv_attention = _required_tensor(state_dict, "conv1.att")
    edge_weight = _required_tensor(state_dict, "conv1.lin_edge.weight")
    prior_weight = _required_tensor(state_dict, "prior_head.0.weight")
    graph_head_weight = _required_tensor(state_dict, "gat_head.0.weight")
    gate_weight = _required_tensor(state_dict, "alpha_gate.0.weight")

    if atom_weight.ndim != 2:
        raise ArtifactContractError("atom_encoder.0.weight must be rank 2")
    embedding_dim, node_dim = map(int, atom_weight.shape)

    if conv_attention.ndim != 3 or int(conv_attention.shape[2]) != embedding_dim:
        raise ArtifactContractError("conv1.att is incompatible with the atom embedding dimension")
    heads = int(conv_attention.shape[1])

    if edge_weight.ndim != 2:
        raise ArtifactContractError("conv1.lin_edge.weight must be rank 2")
    edge_dim = int(edge_weight.shape[1])

    if prior_weight.ndim != 2 or int(prior_weight.shape[1]) % 3:
        raise ArtifactContractError("prior_head.0.weight does not encode three pair operators")
    enzyme_dim = int(prior_weight.shape[1]) // 3

    expected_graph_pair_dim = embedding_dim * 2 * 3
    if graph_head_weight.ndim != 2 or int(graph_head_weight.shape[1]) != expected_graph_pair_dim:
        raise ArtifactContractError("gat_head.0.weight is incompatible with graph pair pooling")

    expected_gate_dim = expected_graph_pair_dim + enzyme_dim * 3
    if gate_weight.ndim != 2 or int(gate_weight.shape[1]) != expected_gate_dim:
        raise ArtifactContractError("alpha_gate.0.weight is incompatible with graph/enzyme inputs")

    return ModelSpec(
        node_dim=node_dim,
        edge_dim=edge_dim,
        enzyme_dim=enzyme_dim,
        embedding_dim=embedding_dim,
        heads=heads,
    )


def validate_metadata(metadata: Mapping[str, Any], spec: ModelSpec) -> None:
    missing = [field for field in spec.as_dict() if field not in metadata]
    if missing:
        raise ArtifactContractError(
            f"Metadata is missing architecture fields: {', '.join(sorted(missing))}"
        )

    invalid_types = [
        field
        for field in spec.as_dict()
        if isinstance(metadata[field], bool)
        or not isinstance(metadata[field], int)
        or metadata[field] <= 0
    ]
    if invalid_types:
        raise ArtifactContractError(
            "Metadata architecture fields must be positive integers: "
            f"{', '.join(sorted(invalid_types))}"
        )

    mismatches = {
        field: (metadata[field], expected)
        for field, expected in spec.as_dict().items()
        if metadata[field] != expected
    }
    if mismatches:
        details = ", ".join(
            f"{field}=metadata:{actual!r}/checkpoint:{expected!r}"
            for field, (actual, expected) in sorted(mismatches.items())
        )
        raise ArtifactContractError(f"Metadata/checkpoint architecture mismatch: {details}")


def validate_supporting_files(
    metadata: Mapping[str, Any],
    project_root: str | Path,
) -> None:
    """Verify versioned feature files used by a checkpoint."""
    records = metadata.get("supporting_files", [])
    if not isinstance(records, list):
        raise ArtifactContractError("supporting_files must be a list")

    root = Path(project_root).resolve()
    for record in records:
        if (
            not isinstance(record, dict)
            or not isinstance(record.get("path"), str)
            or not isinstance(record.get("sha256"), str)
        ):
            raise ArtifactContractError(
                "Each supporting file requires string path and sha256 fields"
            )
        candidate = (root / record["path"]).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise ArtifactContractError(
                f"Supporting file escapes the project root: {record['path']}"
            ) from exc
        if not candidate.is_file():
            raise ArtifactContractError(
                f"Required supporting file is missing: {candidate}"
            )
        actual_hash = file_sha256(candidate)
        if actual_hash != record["sha256"]:
            raise ArtifactContractError(
                f"Supporting file checksum mismatch for {candidate}: "
                f"metadata={record['sha256']}, actual={actual_hash}"
            )


def load_model_bundle(
    artifact_dir: str | Path,
    device: str | torch.device = "cpu",
    supporting_file_root: str | Path | None = None,
) -> ModelBundle:
    """Load a model only when metadata, checksum, and tensor schemas agree."""
    artifact_dir = Path(artifact_dir)
    metadata = read_metadata(artifact_dir)
    from .molecular import policy_for_metadata
    policy_for_metadata(metadata)
    if "graph_contract" in metadata:
        from .graph_cache import graph_contract
        if metadata["graph_contract"] != graph_contract():
            raise ArtifactContractError("Checkpoint graph contract differs from this runtime")
    validate_supporting_files(
        metadata,
        supporting_file_root or artifact_dir.parent,
    )

    checkpoint_name = metadata.get("checkpoint", "model.pt")
    if Path(checkpoint_name).name != checkpoint_name:
        raise ArtifactContractError("Metadata checkpoint must be a filename, not a path")
    checkpoint_path = artifact_dir / checkpoint_name
    if not checkpoint_path.is_file():
        raise ArtifactContractError(f"Model checkpoint is missing: {checkpoint_path}")

    checkpoint_hash = file_sha256(checkpoint_path)
    expected_hash = metadata.get("checkpoint_sha256")
    if not expected_hash:
        raise ArtifactContractError("Metadata is missing checkpoint_sha256")
    if checkpoint_hash != expected_hash:
        raise ArtifactContractError(
            f"Checkpoint checksum mismatch for {checkpoint_path}: "
            f"metadata={expected_hash}, actual={checkpoint_hash}"
        )

    try:
        state_dict = torch.load(
            checkpoint_path,
            map_location=torch.device(device),
            weights_only=True,
        )
    except Exception as exc:
        raise ArtifactContractError(f"Could not load checkpoint {checkpoint_path}: {exc}") from exc
    if not isinstance(state_dict, Mapping):
        raise ArtifactContractError("Checkpoint must contain a state-dict mapping")

    spec = infer_model_spec(state_dict)
    validate_metadata(metadata, spec)

    from .model import BioGuardGAT

    model = BioGuardGAT(**spec.as_dict(), branch=metadata.get("branch", "fused")).to(device)
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise ArtifactContractError(f"Checkpoint state-dict is incompatible: {exc}") from exc
    model.eval()

    return ModelBundle(
        model=model,
        spec=spec,
        metadata=metadata,
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_hash,
    )


def save_model_bundle(
    model: torch.nn.Module,
    artifact_dir: str | Path,
    metadata: Mapping[str, Any],
    checkpoint_name: str = "model.pt",
    supporting_file_root: str | Path | None = None,
) -> dict[str, Any]:
    """Write a portable checkpoint and matching metadata with atomic replaces."""
    if Path(checkpoint_name).name != checkpoint_name:
        raise ArtifactContractError("Checkpoint name must be a filename, not a path")

    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = artifact_dir / checkpoint_name
    checkpoint_tmp = artifact_dir / f".{checkpoint_name}.tmp"
    metadata_path = artifact_dir / "metadata.json"
    metadata_tmp = artifact_dir / ".metadata.json.tmp"

    state_dict = {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
    }
    spec = infer_model_spec(state_dict)

    try:
        torch.save(state_dict, checkpoint_tmp)
        checkpoint_hash = file_sha256(checkpoint_tmp)
        complete_metadata = {
            **dict(metadata),
            "schema_version": 1,
            "model_class": "BioGuardGAT",
            "checkpoint": checkpoint_name,
            "checkpoint_sha256": checkpoint_hash,
            "branch": getattr(model, "branch", "fused"),
            **spec.as_dict(),
        }
        validate_metadata(complete_metadata, spec)
        validate_supporting_files(complete_metadata, supporting_file_root or artifact_dir.parent)
        metadata_tmp.write_text(
            json.dumps(complete_metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        checkpoint_tmp.replace(checkpoint_path)
        metadata_tmp.replace(metadata_path)
    except Exception:
        checkpoint_tmp.unlink(missing_ok=True)
        metadata_tmp.unlink(missing_ok=True)
        raise

    return complete_metadata


def load_calibrator(
    artifact_dir: str | Path,
    metadata: Mapping[str, Any],
) -> Any | None:
    """Load the calibrator only when explicitly enabled and checksummed."""
    if not metadata.get("calibrator_enabled", False):
        return None

    filename = metadata.get("calibrator")
    expected_hash = metadata.get("calibrator_sha256")
    if not filename or not expected_hash or Path(filename).name != filename:
        raise ArtifactContractError("Enabled calibrator lacks a safe filename/checksum contract")

    path = Path(artifact_dir) / filename
    if not path.is_file():
        raise ArtifactContractError(f"Enabled calibrator is missing: {path}")
    actual_hash = file_sha256(path)
    if actual_hash != expected_hash:
        raise ArtifactContractError(
            f"Calibrator checksum mismatch: metadata={expected_hash}, actual={actual_hash}"
        )

    import joblib

    try:
        return joblib.load(path)
    except Exception as exc:
        raise ArtifactContractError(f"Could not load calibrator {path}: {exc}") from exc
