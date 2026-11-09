"""Controlled, local benchmark of observed associations versus unlabeled pairs.

This deliberately estimates observation ranking, not latent clinical risk.
Test examples are shared by every model; only validation selects parameters.
"""

import argparse
import io
import json
import random
import subprocess
import sys
from pathlib import Path

import joblib
import lmdb
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from torch_geometric.loader import DataLoader

from .artifacts import file_sha256, load_model_bundle, save_model_bundle
from .baselines import SymmetricFingerprintFeaturizer, _ranking_metrics
from .cohorts import require_binary_splits
from .config import DATA_DIR
from .data_loader import (
    PUSamplingConfig, SplitConfig, build_binary_pu_dataset,
    load_twosides_relations, prepare_twosides_relations,
)
from .enzyme import EnzymeManager
from .featurizer import GraphFeaturizer
from .graph_cache import (
    check_cache_contract, close_readers, filter_cached_pairs, graph_contract,
)
from .model import BioGuardGAT
from .molecular import STANDARDIZATION_POLICY
from .pu import PUBalancedBatchSampler
from .train import BioGuardDataset

ROOT = Path(__file__).resolve().parent.parent
TASK = "observed_association_ranking"
SEMANTICS = (
    "Ranks reported associations above sampled unobserved pairs. Unobserved "
    "pairs are unlabeled, not confirmed negatives. Scores are not probabilities "
    "of interaction, causality, incidence, or patient harm."
)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def capture_environment(output):
    """Capture actual source bytes, including uncommitted/new implementation."""
    source_dir = output / "source"
    files = [*ROOT.glob("*.py"), *ROOT.glob("requirements*.txt")]
    for name in ("bioguard", "worker", "tests"):
        files.extend((ROOT / name).rglob("*.py"))
    hashes = {}
    for path in sorted(files):
        relative = path.relative_to(ROOT)
        dest = source_dir / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(path.read_bytes())
        hashes[str(relative)] = file_sha256(dest)
    freeze = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=True,
    )
    (output / "environment.txt").write_text(freeze.stdout)
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True,
    )
    return {
        "git_revision": revision.stdout.strip() if revision.returncode == 0 else None,
        "source_sha256": hashes,
        "python": sys.version,
        "environment_sha256": file_sha256(output / "environment.txt"),
    }


def build_run_cache(records, path):
    """A new per-run cache avoids changing any deployed or legacy cache."""
    path.mkdir()
    failures = []
    featurizer = GraphFeaturizer()
    molecules = sorted(set(records.smiles_a) | set(records.smiles_b))
    with lmdb.open(str(path), map_size=4 * 1024**3) as env:
        check_cache_contract(env, initialize=True)
        for index, smiles in enumerate(molecules):
            try:
                graph = featurizer.smiles_to_graph(smiles)
                if not torch.isfinite(graph.x).all() or not torch.isfinite(graph.edge_attr).all():
                    raise ValueError("Non-finite graph features")
                buffer = io.BytesIO()
                torch.save(graph, buffer)
                with env.begin(write=True) as txn:
                    txn.put(smiles.encode(), buffer.getvalue())
            except (ValueError, RuntimeError) as exc:
                failures.append({"smiles": smiles, "error": str(exc)})
            if (index + 1) % 25 == 0:
                print(f"Graphs: {index + 1}/{len(molecules)}", flush=True)
    return failures


def score_model(model, loader):
    scores = []
    model.eval()
    with torch.no_grad():
        for a, b, _ in loader:
            logits, _ = model(a, b)
            scores.extend(torch.sigmoid(logits).flatten().tolist())
    scores = np.asarray(scores)
    if not np.isfinite(scores).all():
        raise ValueError("Non-finite model scores")
    return scores


def write_predictions(directory, frame, scores):
    predictions = frame.copy()
    predictions["score"] = scores
    path = directory / "predictions.csv"
    predictions.to_csv(path, index=False)
    return {
        **_ranking_metrics(frame.target.to_numpy(), scores),
        "n_test": len(frame),
        "observed_positive_fraction": float(frame.target.mean()),
        "predictions_sha256": file_sha256(path),
    }


def fit_fingerprint(seed, splits, matrices, output):
    directory = output / f"seed_{seed}" / "fingerprint"
    directory.mkdir(parents=True)
    candidates = []
    for c in (0.01, 0.1, 1.0, 10.0):
        model = LogisticRegression(C=c, max_iter=2000, random_state=seed)
        model.fit(matrices["train"], splits["train"].target)
        score = _ranking_metrics(
            splits["val"].target, model.predict_proba(matrices["val"])[:, 1],
        )["pu_pr_auc"]
        candidates.append((score, c, model))
    val_ap, selected_c, model = max(candidates, key=lambda item: item[0])
    joblib.dump(model, directory / "model.joblib")
    metrics = write_predictions(
        directory, splits["test"], model.predict_proba(matrices["test"])[:, 1],
    )
    return {"model": "fingerprint", "seed": seed, "selected_c": selected_c,
            "validation_ap": val_ap, **metrics}


def fit_neural(branch, seed, datasets, args, output, metadata):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = BioGuardGAT(
        enzyme_dim=datasets["train"].enzyme_manager.vector_dim,
        embedding_dim=args.embedding_dim, heads=args.heads, branch=branch,
    )
    # All neural ablations use identical balanced observation batches.
    sampler = PUBalancedBatchSampler(
        datasets["train"].df.target, args.batch_size, seed,
    )
    train_loader = DataLoader(datasets["train"], batch_sampler=sampler, num_workers=0)
    val_loader = DataLoader(datasets["val"], batch_size=args.batch_size, num_workers=0)
    test_loader = DataLoader(datasets["test"], batch_size=args.batch_size, num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = torch.nn.BCEWithLogitsLoss()
    best_ap, stale = -1.0, 0
    directory = output / f"seed_{seed}" / branch
    directory.mkdir(parents=True)
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for a, b, targets in train_loader:
            optimizer.zero_grad()
            logits, _ = model(a, b)
            loss = criterion(logits, targets.reshape_as(logits))
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())
        val_ap = _ranking_metrics(
            datasets["val"].df.target, score_model(model, val_loader),
        )["pu_pr_auc"]
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), "validation_ap": val_ap})
        print(f"seed={seed} branch={branch} epoch={epoch} val_AP={val_ap:.4f}", flush=True)
        if val_ap > best_ap:
            best_ap, stale = val_ap, 0
            save_model_bundle(
                model, directory,
                {**metadata, "seed": seed, "best_epoch": epoch,
                 "validation_ap": val_ap, "loss": "balanced_observation_bce"},
                supporting_file_root=ROOT,
            )
        else:
            stale += 1
            if stale >= args.patience:
                break
    write_json(directory / "history.json", history)
    bundle = load_model_bundle(directory, supporting_file_root=ROOT)
    metrics = write_predictions(
        directory, datasets["test"].df, score_model(bundle.model, test_loader),
    )
    return {"model": branch, "seed": seed, "validation_ap": best_ap,
            "checkpoint_sha256": bundle.checkpoint_sha256, **metrics}


def run_benchmark(args):
    if (args.epochs < 1 or args.batch_size < 2 or args.patience < 1
            or args.threads < 1 or args.lr <= 0 or not np.isfinite(args.lr)
            or args.embedding_dim < 1 or args.heads < 1):
        raise ValueError("Invalid training configuration")
    if args.unlabeled_ratio <= 0 or not np.isfinite(args.unlabeled_ratio):
        raise ValueError("unlabeled_ratio must be finite and positive")
    if len(set(args.seeds)) != len(args.seeds) or not args.seeds:
        raise ValueError("Training seeds must be present and unique")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "status.json", {"status": "running"})
    old_threads = torch.get_num_threads()
    old_determinism = torch.are_deterministic_algorithms_enabled()
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    try:
        provenance = capture_environment(output)
        write_json(output / "configuration.json", {k: v for k, v in vars(args).items() if k != "func"})
        if args.source_csv:
            raw = pd.read_csv(args.source_csv, dtype={"Drug1_ID": str, "Drug2_ID": str, "Y": str})
            relation_bundle = prepare_twosides_relations(raw)
            source_hash = file_sha256(args.source_csv)
        else:
            relation_bundle = load_twosides_relations()
            source_hash = relation_bundle.manifest["records_sha256"]
        bundle = build_binary_pu_dataset(
            relation_bundle.records, SplitConfig(method=args.split, seed=args.split_seed),
            PUSamplingConfig(args.unlabeled_ratio, args.unlabeled_ratio,
                             args.unlabeled_ratio, args.split_seed),
        )
        records = bundle.records
        require_binary_splits(records)
        records.to_csv(output / "candidate_cohort.csv", index=False)
        failures = build_run_cache(records, output / "graphs")
        write_json(output / "graph_failures.json", failures)
        effective = filter_cached_pairs(records, output / "graphs")
        cohorts = require_binary_splits(effective)
        effective.to_csv(output / "effective_cohort.csv", index=False)
        records.loc[~records.sample_id.isin(effective.sample_id)].to_csv(
            output / "excluded_pairs.csv", index=False,
        )
        manager = EnzymeManager(allow_degraded=False)
        # Freeze auxiliary features once; no undeclared learned CYP fallback.
        if manager.predictor.models:
            raise ValueError("Benchmark requires the frozen enzyme table; undeclared CYP predictor found")
        datasets, splits, coverage = {}, {}, {}
        for split in ("train", "val", "test"):
            splits[split] = effective.loc[effective.split == split].reset_index(drop=True)
            datasets[split] = BioGuardDataset(
                root=str(output), df=splits[split], split=split,
                enzyme_manager=manager, lmdb_path=str(output / "graphs"),
            )
            original = records.loc[records.split == split]
            present = [
                (manager._canonicalize(row.smiles_a) in manager.smiles_map,
                 manager._canonicalize(row.smiles_b) in manager.smiles_map)
                for row in splits[split].itertuples(index=False)
            ]
            coverage[split] = {
                "candidate_count": len(original), "effective_count": len(splits[split]),
                "graph_coverage": len(splits[split]) / len(original),
                "both_in_enzyme_table": sum(a and b for a, b in present),
                "one_in_enzyme_table": sum(a != b for a, b in present),
                "neither_in_enzyme_table": sum(not a and not b for a, b in present),
            }
        metadata = {
            "scientific_task": TASK, "evaluation_semantics": SEMANTICS,
            "standardization_policy": STANDARDIZATION_POLICY,
            "graph_contract": graph_contract(), "effective_cohorts": cohorts,
            "split_type": args.split, "split_seed": args.split_seed,
            "data_manifest": bundle.manifest,
            "threshold_validated": False, "calibrator_enabled": False,
            "training_provenance_status": "benchmark_run_manifest",
            "enzyme_fallback_status": "zero_vector_for_unseen_smiles",
            "supporting_files": [{"role": "enzyme_features", "path": "data/enzyme_features_full.csv",
                                  "sha256": file_sha256(Path(DATA_DIR) / "enzyme_features_full.csv")}],
        }
        manifest = {
            "schema_version": 1, "task": TASK, "semantics": SEMANTICS,
            "source_sha256": source_hash, "source_manifest": relation_bundle.manifest,
            "data_manifest": bundle.manifest, "effective_cohorts": cohorts,
            "coverage": coverage, "provenance": provenance,
            "graph_contract": graph_contract(),
            "effective_cohort_sha256": file_sha256(output / "effective_cohort.csv"),
        }
        write_json(output / "manifest.json", manifest)
        featurizer = SymmetricFingerprintFeaturizer()
        matrices = {split: featurizer.transform(frame) for split, frame in splits.items()}
        results = []
        for seed in args.seeds:
            results.append(fit_fingerprint(seed, splits, matrices, output))
            for branch in ("graph", "enzyme", "fused"):
                results.append(fit_neural(branch, seed, datasets, args, output, metadata))
            write_json(output / "results.json", results)
        summary = {}
        for model in ("fingerprint", "graph", "enzyme", "fused"):
            rows = [row for row in results if row["model"] == model]
            summary[model] = {
                metric: {"mean": float(np.mean([r[metric] for r in rows])),
                         "seed_std": float(np.std([r[metric] for r in rows]))}
                for metric in ("pu_pr_auc", "pu_roc_auc")
            }
        write_json(output / "summary.json", {
            "task": TASK, "models": summary, "training_seeds": args.seeds,
            "split_seed": args.split_seed,
            "uncertainty_note": "Seed spread on one frozen holdout is not a confidence interval.",
        })
        write_json(output / "status.json", {"status": "complete", "models_evaluated": len(results)})
        print(f"Benchmark written to {output}", flush=True)
        return manifest, results
    except Exception as exc:
        write_json(output / "status.json", {"status": "failed", "error": str(exc)})
        raise
    finally:
        close_readers()
        torch.set_num_threads(old_threads)
        torch.use_deterministic_algorithms(old_determinism)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    from .main import add_benchmark_arguments
    add_benchmark_arguments(parser)
    run_benchmark(parser.parse_args())
