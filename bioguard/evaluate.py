import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch_geometric.loader import DataLoader as PyGDataLoader
from tqdm import tqdm

from .artifacts import ArtifactContractError, load_model_bundle
from .config import ARTIFACT_DIR, DATA_DIR
from .data_loader import load_twosides_binary_pu
from .cohorts import cohort_manifest, require_matching_cohorts
from .graph_cache import filter_cached_pairs
from .enzyme import EnzymeManager
from .train import BioGuardDataset, sample_membership_sha256

EVAL_RESULTS_PATH = Path(ARTIFACT_DIR) / "eval_results.json"


def evaluate_model(override_split=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    bundle = load_model_bundle(
        ARTIFACT_DIR,
        device=device,
        supporting_file_root=Path(DATA_DIR).parent,
    )
    meta = bundle.metadata
    split_type = meta.get("split_type")
    if not split_type:
        raise ArtifactContractError(
            "The checkpoint has no verified split_type and cannot be evaluated "
            "under the new data contract."
        )
    if override_split and override_split != split_type:
        raise ArtifactContractError(
            "Evaluation split overrides are forbidden because they break the "
            f"training contract: checkpoint={split_type}, requested={override_split}"
        )
    if (
        meta.get("scientific_task")
        != "binary_any_reported_event_positive_unlabeled"
    ):
        raise ArtifactContractError(
            "This evaluator only accepts checkpoints trained under the explicit "
            "positive-unlabeled data contract."
        )

    print("--- BioGuard Positive-Unlabeled Ranking Evaluation ---")
    print(
        f"Architecture: Dim={bundle.spec.embedding_dim}, "
        f"Heads={bundle.spec.heads}"
    )
    print(f"Eval Split:   {split_type.upper()}")
    print(f"Class Prior:  {meta['class_prior']:.4f}")

    print("\nLoading holdout data...")
    data_bundle = load_twosides_binary_pu(
        split_method=split_type,
        train_unlabeled_ratio=float(meta["unlabeled_ratio"]),
        val_unlabeled_ratio=float(meta["unlabeled_ratio"]),
        test_unlabeled_ratio=float(meta["unlabeled_ratio"]),
        random_seed=int(meta["random_seed"]),
    )
    expected_data_hash = (
        meta.get("data_manifest", {}).get("records_sha256")
    )
    if data_bundle.manifest["records_sha256"] != expected_data_hash:
        raise ArtifactContractError(
            "Evaluation data hash does not match the checkpoint training manifest"
        )
    df = data_bundle.records
    test_df = df[df["split"] == "test"].reset_index(drop=True)
    if test_df["target"].nunique() != 2:
        raise ValueError(
            "PU ranking evaluation requires positive and unlabeled test samples"
        )

    enzyme_manager = EnzymeManager(allow_degraded=False)
    if enzyme_manager.vector_dim != bundle.spec.enzyme_dim:
        raise ArtifactContractError(
            "Enzyme feature schema does not match checkpoint: "
            f"manager={enzyme_manager.vector_dim}, "
            f"checkpoint={bundle.spec.enzyme_dim}"
        )
    model = bundle.model

    test_dataset = BioGuardDataset(
        root=DATA_DIR,
        df=test_df,
        split="test",
        enzyme_manager=enzyme_manager,
    )
    if len(test_dataset) == 0:
        raise ValueError("The requested test split has no LMDB-backed samples")
    expected_test_membership = (
        meta.get("lmdb_effective_samples", {})
        .get("test", {})
        .get("sample_membership_sha256")
    )
    if sample_membership_sha256(test_dataset.df) != expected_test_membership:
        raise ArtifactContractError(
            "LMDB-filtered test membership does not match the checkpoint manifest"
        )
    effective = filter_cached_pairs(df, test_dataset.lmdb_path)
    effective_cohorts = cohort_manifest(effective)
    require_matching_cohorts(meta.get("effective_cohorts"), effective_cohorts)
    test_loader = PyGDataLoader(
        test_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=0,
    )

    print(f"[Test] Running inference on {len(test_dataset)} pairs...")

    y_true = []
    y_raw_probs = []
    alpha_vals = []

    with torch.no_grad():
        for batch_data in tqdm(test_loader, desc="Testing"):
            batch_a = batch_data[0].to(device)
            batch_b = batch_data[1].to(device)
            batch_y = batch_data[2].to(device)

            logits, alpha = model(batch_a, batch_b)
            probs = torch.sigmoid(logits)

            y_true.extend(batch_y.cpu().numpy().flatten())
            y_raw_probs.extend(probs.cpu().numpy().flatten())
            alpha_vals.extend(alpha.cpu().numpy().flatten())

    y_true = np.array(y_true)
    y_raw_probs = np.array(y_raw_probs)
    mean_alpha = np.mean(alpha_vals)

    roc_auc = roc_auc_score(y_true, y_raw_probs)
    pr_auc = average_precision_score(y_true, y_raw_probs)

    print("\n" + "=" * 60)
    print("OBSERVED-POSITIVE VS UNLABELED RANKING SUMMARY")
    print("=" * 60)
    print(f"  PU ROC-AUC  : {roc_auc:.4f}")
    print(f"  PU PR-AUC   : {pr_auc:.4f}")
    print(f"  Mean Alpha  : {mean_alpha:.4f}")
    print("=" * 60)

    results = {
        "schema_version": 3,
        "name": "BioGuardGAT",
        "checkpoint_sha256": bundle.checkpoint_sha256,
        "split": split_type,
        "task": "binary_any_reported_event_positive_unlabeled",
        "evaluation_semantics": (
            "Observed TWOSIDES associations versus sampled unlabeled pairs; "
            "unlabeled pairs are not confirmed negatives."
        ),
        "data_records_sha256": data_bundle.manifest["records_sha256"],
        "class_prior": float(meta["class_prior"]),
        "calibrated": False,
        "effective_cohorts": effective_cohorts,
        "test_coverage": len(test_dataset) / len(test_df),
        "n_test": len(test_dataset),
        "observed_positive_fraction": float(np.mean(y_true)),
        "pu_roc_auc": float(roc_auc),
        "pu_pr_auc": float(pr_auc),
        "mean_alpha": float(mean_alpha),
        "provenance_warning": (
            "Metrics measure ranking of reported associations above sampled "
            "unlabeled pairs. They do not estimate clinical discrimination, "
            "incidence, causality, sensitivity, or specificity."
        ),
    }
    predictions = test_dataset.df.copy()
    predictions["score"] = y_raw_probs
    predictions["gate_value"] = alpha_vals
    predictions.to_csv(EVAL_RESULTS_PATH.with_name("eval_predictions.csv"), index=False)
    EVAL_RESULTS_PATH.write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Results written to {EVAL_RESULTS_PATH}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split",
        choices=["pair_random", "drug_disjoint", "scaffold_disjoint"],
        default=None,
    )
    args = parser.parse_args()
    evaluate_model(args.split)
