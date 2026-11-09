"""Machine-readable validation for the TWOSIDES relation and PU contracts."""

import argparse
import json
from itertools import combinations
from pathlib import Path

import pandas as pd

from bioguard.config import DATA_DIR
from bioguard.data_loader import (
    ACTIVE_SPLITS,
    _scaffold_key,
    load_twosides_binary_pu,
    load_twosides_relations,
)


def _split_sets(frame, column):
    return {
        split: set(frame.loc[frame["split"] == split, column])
        for split in ACTIVE_SPLITS
    }


def _overlap_counts(values):
    return {
        f"{left}_{right}": len(values[left] & values[right])
        for left, right in combinations(ACTIVE_SPLITS, 2)
    }


def validate_dataset(split_method="scaffold_disjoint"):
    relations = load_twosides_relations()
    binary = load_twosides_binary_pu(split_method=split_method)
    frame = binary.records

    pair_overlaps = _overlap_counts(_split_sets(frame, "pair_id"))
    drug_sets = {
        split: set(
            pd.concat(
                [
                    frame.loc[frame["split"] == split, "drug_a"],
                    frame.loc[frame["split"] == split, "drug_b"],
                ]
            )
        )
        for split in ACTIVE_SPLITS
    }
    drug_overlaps = _overlap_counts(drug_sets)

    scaffold_sets = {
        split: {
            _scaffold_key(smiles)
            for smiles in pd.concat(
                [
                    frame.loc[frame["split"] == split, "smiles_a"],
                    frame.loc[frame["split"] == split, "smiles_b"],
                ]
            )
        }
        for split in ACTIVE_SPLITS
    }
    scaffold_overlaps = _overlap_counts(scaffold_sets)

    unlabeled = frame[frame["observation"] == "unlabeled"]
    positive_pair_ids = set(relations.records["pair_id"])
    checks = {
        "relation_events_preserved": (
            "event_id" in relations.records
            and relations.records["event_id"].notna().all()
        ),
        "relation_triples_unique": not relations.records["triple_id"].duplicated().any(),
        "binary_observations_explicit": set(frame["observation"]) == {
            "positive",
            "unlabeled",
        },
        "unlabeled_not_named_negative": (
            "label" not in frame
            and not binary.manifest["unlabeled_are_confirmed_negatives"]
        ),
        "unlabeled_excludes_all_source_positives": not bool(
            set(unlabeled["pair_id"]) & positive_pair_ids
        ),
        "pair_disjoint": not any(pair_overlaps.values()),
        "drug_disjoint_when_required": (
            split_method == "pair_random"
            or not any(drug_overlaps.values())
        ),
        "scaffold_disjoint_when_required": (
            split_method != "scaffold_disjoint"
            or not any(scaffold_overlaps.values())
        ),
        "all_splits_nonempty": all(
            (frame["split"] == split).any()
            for split in ACTIVE_SPLITS
        ),
        "all_splits_have_both_observations": all(
            frame.loc[frame["split"] == split, "observation"].nunique() == 2
            for split in ACTIVE_SPLITS
        ),
    }
    report = {
        "schema_version": 1,
        "split_method": split_method,
        "passed": all(checks.values()),
        "checks": checks,
        "pair_overlap": pair_overlaps,
        "drug_overlap": drug_overlaps,
        "scaffold_overlap": scaffold_overlaps,
        "relation_manifest": relations.manifest,
        "binary_manifest": binary.manifest,
    }

    output = Path(DATA_DIR) / f"validation_{split_method}.json"
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"passed": report["passed"], "checks": checks}, indent=2))
    print(f"Validation report written to {output}")
    return report["passed"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split",
        default="scaffold_disjoint",
        choices=["pair_random", "drug_disjoint", "scaffold_disjoint"],
    )
    arguments = parser.parse_args()
    raise SystemExit(0 if validate_dataset(arguments.split) else 1)
