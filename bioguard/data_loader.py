"""Versioned TWOSIDES relation and positive-unlabeled data contracts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rdkit
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from .config import DATA_DIR, DATA_VERSION
from .molecular import STANDARDIZATION_POLICY, normalize_smiles

RELATION_COLUMNS = (
    "triple_id",
    "pair_id",
    "drug_a",
    "smiles_a",
    "event_id",
    "drug_b",
    "smiles_b",
    "source_row_count",
)
ACTIVE_SPLITS = ("train", "val", "test")
SPLIT_METHODS = ("pair_random", "drug_disjoint", "scaffold_disjoint")


class DataContractError(ValueError):
    """Raised when source data or a derived split violates its contract."""


@dataclass(frozen=True)
class DatasetBundle:
    records: pd.DataFrame
    manifest: dict[str, Any]


@dataclass(frozen=True)
class SplitConfig:
    method: str = "scaffold_disjoint"
    train_fraction: float = 0.8
    val_fraction: float = 0.1
    test_fraction: float = 0.1
    seed: int = 42

    def validate(self) -> None:
        if self.method not in SPLIT_METHODS:
            raise DataContractError(
                f"Unsupported split method {self.method!r}; "
                f"choose from {', '.join(SPLIT_METHODS)}"
            )
        fractions = (
            self.train_fraction,
            self.val_fraction,
            self.test_fraction,
        )
        if any(value <= 0 or value >= 1 for value in fractions):
            raise DataContractError("Split fractions must each be within (0, 1)")
        if not np.isclose(sum(fractions), 1.0):
            raise DataContractError("Split fractions must sum to 1")


@dataclass(frozen=True)
class PUSamplingConfig:
    train_unlabeled_ratio: float = 1.0
    val_unlabeled_ratio: float = 1.0
    test_unlabeled_ratio: float = 1.0
    seed: int = 42

    def ratio_for(self, split: str) -> float:
        return {
            "train": self.train_unlabeled_ratio,
            "val": self.val_unlabeled_ratio,
            "test": self.test_unlabeled_ratio,
        }[split]

    def validate(self) -> None:
        ratios = (
            self.train_unlabeled_ratio,
            self.val_unlabeled_ratio,
            self.test_unlabeled_ratio,
        )
        if any(value < 0 for value in ratios):
            raise DataContractError("Unlabeled sampling ratios cannot be negative")


def _stable_id(prefix: str, *parts: Any) -> str:
    payload = "\x1f".join(str(part) for part in parts)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"{prefix}_{digest[:24]}"


def _frame_sha256(frame: pd.DataFrame, sort_by: list[str]) -> str:
    ordered = frame.sort_values(sort_by, kind="stable").reset_index(drop=True)
    payload = ordered.to_csv(index=False, lineterminator="\n")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_event_id(value: Any) -> str:
    if pd.isna(value):
        raise DataContractError("TWOSIDES contains a missing event identifier")
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    result = str(value).strip()
    if not result:
        raise DataContractError("TWOSIDES contains an empty event identifier")
    return result


def _standardize_smiles(value: Any) -> str | None:
    try:
        return normalize_smiles(value)
    except (RuntimeError, TypeError, ValueError):
        return None


def _scaffold_key(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise DataContractError(f"Cannot scaffold invalid standardized SMILES: {smiles}")
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(
        mol=mol,
        includeChirality=True,
    )
    # Murcko scaffolds are empty for acyclic structures. Keeping their complete
    # standardized structure prevents unrelated acyclic drugs becoming one group.
    return scaffold or f"ACYCLIC::{smiles}"


def prepare_twosides_relations(raw: pd.DataFrame) -> DatasetBundle:
    """Preserve TWOSIDES as drug-event-drug positive association triples."""
    required = {
        "Drug1_ID",
        "Drug2_ID",
        "Drug1",
        "Drug2",
        "Y",
    }
    missing = sorted(required.difference(raw.columns))
    if missing:
        raise DataContractError(
            f"TWOSIDES source is missing columns: {', '.join(missing)}"
        )

    source = raw.loc[:, sorted(required)].copy()
    source = source.rename(
        columns={
            "Drug1_ID": "drug_a",
            "Drug2_ID": "drug_b",
            "Drug1": "smiles_a",
            "Drug2": "smiles_b",
            "Y": "event_id",
        }
    )
    source["drug_a"] = source["drug_a"].astype(str).str.strip()
    source["drug_b"] = source["drug_b"].astype(str).str.strip()
    if (source["drug_a"] == "").any() or (source["drug_b"] == "").any():
        raise DataContractError("TWOSIDES contains an empty drug identifier")
    source["event_id"] = source["event_id"].map(_normalize_event_id)

    unique_smiles = pd.unique(
        pd.concat([source["smiles_a"], source["smiles_b"]], ignore_index=True)
    )
    smiles_map = {
        value: _standardize_smiles(value)
        for value in unique_smiles
    }
    source["smiles_a"] = source["smiles_a"].map(smiles_map)
    source["smiles_b"] = source["smiles_b"].map(smiles_map)
    invalid_structure_rows = int(
        source[["smiles_a", "smiles_b"]].isna().any(axis=1).sum()
    )
    source = source.dropna(subset=["smiles_a", "smiles_b"]).copy()

    swap = source["drug_a"] > source["drug_b"]
    drug_a = source["drug_a"].copy()
    smiles_a = source["smiles_a"].copy()
    source.loc[swap, "drug_a"] = source.loc[swap, "drug_b"]
    source.loc[swap, "smiles_a"] = source.loc[swap, "smiles_b"]
    source.loc[swap, "drug_b"] = drug_a.loc[swap]
    source.loc[swap, "smiles_b"] = smiles_a.loc[swap]

    self_relations = int((source["drug_a"] == source["drug_b"]).sum())
    source = source[source["drug_a"] != source["drug_b"]].copy()

    drug_structures = pd.concat(
        [
            source[["drug_a", "smiles_a"]].rename(
                columns={"drug_a": "drug_id", "smiles_a": "smiles"}
            ),
            source[["drug_b", "smiles_b"]].rename(
                columns={"drug_b": "drug_id", "smiles_b": "smiles"}
            ),
        ],
        ignore_index=True,
    ).drop_duplicates()
    conflicts = drug_structures.groupby("drug_id")["smiles"].nunique()
    conflicts = conflicts[conflicts > 1]
    if not conflicts.empty:
        examples = ", ".join(conflicts.index.astype(str)[:5])
        raise DataContractError(
            "A drug identifier maps to multiple standardized structures; "
            f"examples: {examples}"
        )

    source["pair_id"] = [
        _stable_id("pair", drug_a, drug_b)
        for drug_a, drug_b in zip(source["drug_a"], source["drug_b"])
    ]
    source["triple_id"] = [
        _stable_id("triple", pair_id, event_id)
        for pair_id, event_id in zip(source["pair_id"], source["event_id"])
    ]

    grouped = (
        source.groupby(
            [
                "triple_id",
                "pair_id",
                "drug_a",
                "smiles_a",
                "event_id",
                "drug_b",
                "smiles_b",
            ],
            as_index=False,
            sort=True,
        )
        .size()
        .rename(columns={"size": "source_row_count"})
    )
    grouped = grouped.loc[:, RELATION_COLUMNS].sort_values(
        ["pair_id", "event_id"],
        kind="stable",
    )
    grouped = grouped.reset_index(drop=True)

    manifest = {
        "schema_version": 1,
        "dataset": "TWOSIDES",
        "data_version": DATA_VERSION,
        "task": "multi_relation_positive_associations",
        "label_semantics": (
            "Each record is a reported drug-event-drug association. "
            "Absence is not a negative label."
        ),
        "standardization": {
            "policy": STANDARDIZATION_POLICY,
            "rdkit_version": rdkit.__version__,
            "charge_parent": True,
            "canonical_smiles": True,
            "preserve_stereochemistry": True,
            "tautomer_canonicalization": False,
        },
        "raw_rows": int(len(raw)),
        "invalid_structure_rows_excluded": invalid_structure_rows,
        "self_relations_excluded": self_relations,
        "unique_relations": int(len(grouped)),
        "unique_pairs": int(grouped["pair_id"].nunique()),
        "unique_events": int(grouped["event_id"].nunique()),
        "unique_drugs": int(
            pd.unique(
                pd.concat([grouped["drug_a"], grouped["drug_b"]])
            ).size
        ),
        "records_sha256": _frame_sha256(
            grouped,
            ["triple_id"],
        ),
    }
    return DatasetBundle(records=grouped, manifest=manifest)


def _load_tdc_twosides() -> pd.DataFrame:
    try:
        from tdc.multi_pred import DDI
    except ImportError as exc:
        raise DataContractError(
            "PyTDC is required to download TWOSIDES"
        ) from exc
    return DDI(name="TWOSIDES").get_data()


def load_twosides_relations(force_rebuild: bool = False) -> DatasetBundle:
    """Load or build the versioned event-preserving TWOSIDES relation table."""
    cache_dir = Path(DATA_DIR)
    records_path = cache_dir / f"twosides_relations_{DATA_VERSION}.parquet"
    manifest_path = cache_dir / f"twosides_relations_{DATA_VERSION}.manifest.json"

    if not force_rebuild and records_path.is_file() and manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise DataContractError(
                f"Could not read relation manifest: {exc}"
            ) from exc
        if manifest.get("records_file_sha256") != _file_sha256(records_path):
            raise DataContractError("Cached relation table checksum mismatch")
        standardization = manifest.get("standardization", {})
        if (standardization.get("policy") != STANDARDIZATION_POLICY
                or standardization.get("rdkit_version") != rdkit.__version__):
            raise DataContractError(
                "Cached relation standardization differs from this runtime; "
                "rebuild with load_twosides_relations(force_rebuild=True)"
            )
        records = pd.read_parquet(records_path)
        if tuple(records.columns) != RELATION_COLUMNS:
            raise DataContractError("Cached relation table has an invalid schema")
        return DatasetBundle(records=records, manifest=manifest)

    bundle = prepare_twosides_relations(_load_tdc_twosides())
    cache_dir.mkdir(parents=True, exist_ok=True)
    bundle.records.to_parquet(records_path, index=False)
    manifest = {
        **bundle.manifest,
        "source_provider": "Therapeutics Data Commons",
        "source_release": "not_exposed_by_provider_api",
        "records_file": records_path.name,
        "records_file_sha256": _file_sha256(records_path),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return DatasetBundle(records=bundle.records, manifest=manifest)


def _drug_table(relations: pd.DataFrame) -> pd.DataFrame:
    drugs = pd.concat(
        [
            relations[["drug_a", "smiles_a"]].rename(
                columns={"drug_a": "drug_id", "smiles_a": "smiles"}
            ),
            relations[["drug_b", "smiles_b"]].rename(
                columns={"drug_b": "drug_id", "smiles_b": "smiles"}
            ),
        ],
        ignore_index=True,
    ).drop_duplicates()
    conflicts = drugs.groupby("drug_id")["smiles"].nunique()
    if (conflicts > 1).any():
        raise DataContractError("Relation table has conflicting drug structures")
    return drugs.drop_duplicates("drug_id").sort_values("drug_id").reset_index(drop=True)


def _assign_groups(
    group_weights: dict[str, int],
    config: SplitConfig,
) -> dict[str, str]:
    targets = {
        "train": config.train_fraction * sum(group_weights.values()),
        "val": config.val_fraction * sum(group_weights.values()),
        "test": config.test_fraction * sum(group_weights.values()),
    }
    assigned = {split: 0 for split in ACTIVE_SPLITS}
    rng = np.random.default_rng(config.seed)
    tie_order = {
        group: rank
        for rank, group in enumerate(rng.permutation(sorted(group_weights)))
    }
    ordered_groups = sorted(
        group_weights,
        key=lambda group: (-group_weights[group], tie_order[group], group),
    )

    result: dict[str, str] = {}
    for group in ordered_groups:
        split = min(
            ACTIVE_SPLITS,
            key=lambda candidate: (
                assigned[candidate] / targets[candidate],
                ACTIVE_SPLITS.index(candidate),
            ),
        )
        result[group] = split
        assigned[split] += group_weights[group]
    return result


def _pair_split_map(
    relations: pd.DataFrame,
    config: SplitConfig,
) -> tuple[dict[str, str], dict[str, str] | None]:
    pairs = relations[
        ["pair_id", "drug_a", "drug_b"]
    ].drop_duplicates("pair_id")
    if config.method == "pair_random":
        result = {
            row.pair_id: _candidate_split(
                row.drug_a,
                row.drug_b,
                row.pair_id,
                config,
                None,
            )
            for row in pairs.itertuples(index=False)
        }
        return result, None

    drugs = _drug_table(relations)
    if config.method == "drug_disjoint":
        drugs["group"] = drugs["drug_id"]
    else:
        drugs["group"] = drugs["smiles"].map(_scaffold_key)

    group_weights = drugs.groupby("group")["drug_id"].nunique().to_dict()
    group_assignments = _assign_groups(group_weights, config)
    drug_assignments = {
        row.drug_id: group_assignments[row.group]
        for row in drugs.itertuples(index=False)
    }

    result = {}
    for row in pairs.itertuples(index=False):
        split_a = drug_assignments[row.drug_a]
        split_b = drug_assignments[row.drug_b]
        result[row.pair_id] = (
            split_a if split_a == split_b else "excluded_cross_partition"
        )
    return result, drug_assignments


def assign_relation_splits(
    relations: pd.DataFrame,
    config: SplitConfig,
) -> DatasetBundle:
    """Assign every event for a pair to one auditable split."""
    config.validate()
    missing = sorted(set(RELATION_COLUMNS).difference(relations.columns))
    if missing:
        raise DataContractError(
            f"Relation table is missing columns: {', '.join(missing)}"
        )

    records = relations.copy()
    pair_splits, drug_assignments = _pair_split_map(records, config)
    records["split"] = records["pair_id"].map(pair_splits)
    if records["split"].isna().any():
        raise DataContractError("At least one relation pair was not assigned")

    active = records[records["split"].isin(ACTIVE_SPLITS)]
    split_counts = {
        split: {
            "relations": int((active["split"] == split).sum()),
            "pairs": int(
                active.loc[active["split"] == split, "pair_id"].nunique()
            ),
        }
        for split in ACTIVE_SPLITS
    }
    if any(values["pairs"] == 0 for values in split_counts.values()):
        raise DataContractError(
            "Split construction produced an empty train, validation, or test set"
        )

    drug_sets = {
        split: set(
            pd.concat(
                [
                    active.loc[active["split"] == split, "drug_a"],
                    active.loc[active["split"] == split, "drug_b"],
                ]
            )
        )
        for split in ACTIVE_SPLITS
    }
    overlaps = {
        f"{left}_{right}": len(drug_sets[left] & drug_sets[right])
        for left, right in combinations(ACTIVE_SPLITS, 2)
    }
    if config.method != "pair_random" and any(overlaps.values()):
        raise DataContractError(
            f"{config.method} unexpectedly produced cross-split drug overlap"
        )

    manifest = {
        "schema_version": 1,
        "task": "multi_relation_positive_associations",
        "split_config": asdict(config),
        "split_semantics": (
            "Pair-disjoint random split."
            if config.method == "pair_random"
            else (
                "Both drugs are assigned to disjoint entity partitions; "
                "cross-partition pairs are explicitly excluded."
            )
        ),
        "split_counts": split_counts,
        "excluded_cross_partition_relations": int(
            (records["split"] == "excluded_cross_partition").sum()
        ),
        "drug_overlap": overlaps,
        "drug_assignment_sha256": (
            hashlib.sha256(
                json.dumps(
                    drug_assignments,
                    sort_keys=True,
                ).encode("utf-8")
            ).hexdigest()
            if drug_assignments is not None
            else None
        ),
        "drug_assignments": drug_assignments,
        "records_sha256": _frame_sha256(
            records,
            ["triple_id"],
        ),
    }
    return DatasetBundle(records=records, manifest=manifest)


def _candidate_split(
    drug_a: str,
    drug_b: str,
    pair_id: str,
    split_config: SplitConfig,
    drug_assignments: dict[str, str] | None,
) -> str:
    if drug_assignments is not None:
        split_a = drug_assignments[drug_a]
        split_b = drug_assignments[drug_b]
        return split_a if split_a == split_b else "excluded_cross_partition"

    digest = hashlib.sha256(
        f"{split_config.seed}\x1f{pair_id}".encode("utf-8")
    ).digest()
    value = int.from_bytes(digest[:8], "big") / 2**64
    if value < split_config.train_fraction:
        return "train"
    if value < split_config.train_fraction + split_config.val_fraction:
        return "val"
    return "test"


def build_binary_pu_dataset(
    relations: pd.DataFrame,
    split_config: SplitConfig,
    sampling_config: PUSamplingConfig | None = None,
) -> DatasetBundle:
    """Build an explicit binary positive-unlabeled compatibility benchmark."""
    split_config.validate()
    sampling_config = sampling_config or PUSamplingConfig(
        seed=split_config.seed
    )
    sampling_config.validate()

    split_bundle = assign_relation_splits(relations, split_config)
    split_relations = split_bundle.records
    active_relations = split_relations[
        split_relations["split"].isin(ACTIVE_SPLITS)
    ].copy()

    positives = (
        active_relations.groupby(
            [
                "pair_id",
                "drug_a",
                "smiles_a",
                "drug_b",
                "smiles_b",
                "split",
            ],
            as_index=False,
            sort=True,
        )
        .agg(
            event_count=("event_id", "nunique"),
            event_ids=("event_id", lambda values: json.dumps(sorted(set(values)))),
        )
    )
    positives["observation"] = "positive"
    positives["target"] = 1.0
    positives["negative_evidence"] = "not_applicable"

    drugs = _drug_table(relations)
    id_to_smiles = dict(zip(drugs["drug_id"], drugs["smiles"]))
    all_drug_ids = sorted(id_to_smiles)
    positive_pair_ids = set(relations["pair_id"])

    _, drug_assignments = _pair_split_map(relations, split_config)
    candidates: list[dict[str, Any]] = []
    for drug_a, drug_b in combinations(all_drug_ids, 2):
        pair_id = _stable_id("pair", drug_a, drug_b)
        if pair_id in positive_pair_ids:
            continue
        split = _candidate_split(
            drug_a,
            drug_b,
            pair_id,
            split_config,
            drug_assignments,
        )
        if split not in ACTIVE_SPLITS:
            continue
        candidates.append(
            {
                "pair_id": pair_id,
                "drug_a": drug_a,
                "smiles_a": id_to_smiles[drug_a],
                "drug_b": drug_b,
                "smiles_b": id_to_smiles[drug_b],
                "split": split,
            }
        )
    candidate_frame = pd.DataFrame(
        candidates,
        columns=[
            "pair_id",
            "drug_a",
            "smiles_a",
            "drug_b",
            "smiles_b",
            "split",
        ],
    )

    sampled_frames = []
    rng = np.random.default_rng(sampling_config.seed)
    for split in ACTIVE_SPLITS:
        available = candidate_frame[candidate_frame["split"] == split]
        n_positive = int((positives["split"] == split).sum())
        n_requested = int(
            np.ceil(n_positive * sampling_config.ratio_for(split))
        )
        if n_requested > len(available):
            raise DataContractError(
                f"Requested {n_requested} unlabeled {split} pairs but only "
                f"{len(available)} globally non-positive candidates exist"
            )
        if n_requested == 0:
            continue
        chosen = rng.choice(
            available.index.to_numpy(),
            size=n_requested,
            replace=False,
        )
        sampled_frames.append(available.loc[chosen].copy())

    if sampled_frames:
        unlabeled = pd.concat(sampled_frames, ignore_index=True)
    else:
        unlabeled = candidate_frame.iloc[0:0].copy()
    unlabeled["event_count"] = 0
    unlabeled["event_ids"] = "[]"
    unlabeled["observation"] = "unlabeled"
    unlabeled["target"] = 0.0
    unlabeled["negative_evidence"] = "none_missing_from_twosides"

    records = pd.concat([positives, unlabeled], ignore_index=True)
    records["sample_id"] = records["pair_id"]
    records = records[
        [
            "sample_id",
            "pair_id",
            "drug_a",
            "smiles_a",
            "drug_b",
            "smiles_b",
            "event_count",
            "event_ids",
            "observation",
            "target",
            "negative_evidence",
            "split",
        ]
    ]
    records = records.sort_values(
        ["split", "sample_id", "observation"],
        kind="stable",
    ).reset_index(drop=True)

    pair_overlap = sum(
        len(
            set(records.loc[records["split"] == left, "pair_id"])
            & set(records.loc[records["split"] == right, "pair_id"])
        )
        for left, right in combinations(ACTIVE_SPLITS, 2)
    )
    if pair_overlap:
        raise DataContractError("Binary PU construction produced pair leakage")

    manifest = {
        "schema_version": 1,
        "task": "binary_any_reported_event_positive_unlabeled",
        "source_task": "multi_relation_positive_associations",
        "split_manifest": split_bundle.manifest,
        "sampling_config": asdict(sampling_config),
        "target_semantics": {
            "1": "pair has at least one reported TWOSIDES event",
            "0": (
                "pair is sampled from associations absent in TWOSIDES; "
                "this is unlabeled, not a confirmed negative"
            ),
        },
        "unlabeled_are_confirmed_negatives": False,
        "counts": {
            split: {
                observation: int(
                    (
                        (records["split"] == split)
                        & (records["observation"] == observation)
                    ).sum()
                )
                for observation in ("positive", "unlabeled")
            }
            for split in ACTIVE_SPLITS
        },
        "records_sha256": _frame_sha256(records, ["sample_id", "observation"]),
    }
    return DatasetBundle(records=records, manifest=manifest)


def load_twosides_binary_pu(
    split_method: str = "scaffold_disjoint",
    train_unlabeled_ratio: float = 1.0,
    val_unlabeled_ratio: float = 1.0,
    test_unlabeled_ratio: float = 1.0,
    random_seed: int = 42,
    force_rebuild_relations: bool = False,
) -> DatasetBundle:
    """Load relations and build the explicitly derived binary PU benchmark."""
    relation_bundle = load_twosides_relations(
        force_rebuild=force_rebuild_relations
    )
    split_config = SplitConfig(method=split_method, seed=random_seed)
    sampling_config = PUSamplingConfig(
        train_unlabeled_ratio=train_unlabeled_ratio,
        val_unlabeled_ratio=val_unlabeled_ratio,
        test_unlabeled_ratio=test_unlabeled_ratio,
        seed=random_seed,
    )
    bundle = build_binary_pu_dataset(
        relation_bundle.records,
        split_config,
        sampling_config,
    )
    return DatasetBundle(
        records=bundle.records,
        manifest={
            **bundle.manifest,
            "relation_manifest": relation_bundle.manifest,
        },
    )
