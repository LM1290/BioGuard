"""Compare PU ranking results only when their provenance contracts match."""

import json
from pathlib import Path

import numpy as np

from .artifacts import ArtifactContractError, read_metadata
from .config import ARTIFACT_DIR
from .cohorts import require_matching_cohorts

ARTIFACT_PATH = Path(ARTIFACT_DIR)
EVAL_PATH = ARTIFACT_PATH / "eval_results.json"
BASELINE_PATH = ARTIFACT_PATH / "baseline_results.json"
OUTPUT_PATH = ARTIFACT_PATH / "pu_comparison.png"


def _read_json(path):
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactContractError(f"Could not read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ArtifactContractError(f"{path} must contain a JSON object")
    return value


def load_comparable_results():
    metadata = read_metadata(ARTIFACT_PATH)
    evaluation = _read_json(EVAL_PATH)
    baselines = _read_json(BASELINE_PATH)

    if evaluation.get("schema_version") != 3:
        raise ArtifactContractError("Evaluation results use a legacy schema")
    if baselines.get("schema_version") != 3:
        raise ArtifactContractError("Baseline results use a legacy schema")
    expected_task = "binary_any_reported_event_positive_unlabeled"
    if evaluation.get("task") != expected_task:
        raise ArtifactContractError("Evaluation task is not the PU benchmark")
    if baselines.get("task") != expected_task:
        raise ArtifactContractError("Baseline task is not the PU benchmark")
    if evaluation.get("checkpoint_sha256") != metadata.get("checkpoint_sha256"):
        raise ArtifactContractError(
            "Evaluation results belong to a different checkpoint"
        )
    if (
        evaluation.get("data_records_sha256")
        != baselines.get("data_records_sha256")
    ):
        raise ArtifactContractError(
            "Model and baseline results use different data records"
        )
    if evaluation.get("split") != baselines.get("split"):
        raise ArtifactContractError(
            "Model and baseline results use different split contracts"
        )
    try:
        require_matching_cohorts(
            evaluation.get("effective_cohorts"), baselines.get("effective_cohorts"),
        )
    except ValueError as exc:
        raise ArtifactContractError(str(exc)) from exc
    return evaluation, baselines


def main():
    import matplotlib.pyplot as plt

    evaluation, baselines = load_comparable_results()
    model_records = [
        (
            "BioGuardGAT",
            evaluation["pu_roc_auc"],
            evaluation["pu_pr_auc"],
        )
    ]
    model_records.extend(
        (
            values.get("name", name),
            values["pu_roc_auc"],
            values["pu_pr_auc"],
        )
        for name, values in baselines["models"].items()
    )

    names = [record[0] for record in model_records]
    roc_values = [record[1] for record in model_records]
    pr_values = [record[2] for record in model_records]
    positions = np.arange(len(names))
    width = 0.38

    figure, axis = plt.subplots(figsize=(10, 5))
    axis.bar(positions - width / 2, roc_values, width, label="PU ROC-AUC")
    axis.bar(positions + width / 2, pr_values, width, label="PU PR-AUC")
    axis.set_xticks(positions)
    axis.set_xticklabels(names, rotation=20, ha="right")
    axis.set_ylim(0, 1)
    axis.set_ylabel("Observed-positive vs unlabeled ranking metric")
    axis.set_title(
        f"PU benchmark comparison: {evaluation['split']}"
    )
    axis.legend()
    figure.tight_layout()
    figure.savefig(OUTPUT_PATH, dpi=200)
    plt.close(figure)

    print(
        "Comparison is limited to observed-positive versus sampled-unlabeled "
        "ranking and is not a clinical validation."
    )
    print(f"Figure written to {OUTPUT_PATH}")
    return OUTPUT_PATH


if __name__ == "__main__":
    main()
