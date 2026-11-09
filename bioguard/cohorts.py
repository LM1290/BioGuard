"""Effective sample identities, independent of model feature representation."""

import hashlib

SPLITS = ("train", "val", "test")


def cohort_manifest(frame):
    columns = ["sample_id", "observation", "target", "smiles_a", "smiles_b", "split"]
    if frame["sample_id"].isna().any() or frame["sample_id"].duplicated().any():
        raise ValueError("Cohort sample IDs must be present and unique")
    if not frame["split"].isin(SPLITS).all():
        raise ValueError("Cohort contains an unsupported split")
    expected = frame["observation"].map({"positive": 1.0, "unlabeled": 0.0})
    if expected.isna().any() or not (expected == frame["target"]).all():
        raise ValueError("Observation and optimization target disagree")
    result = {}
    for split in SPLITS:
        subset = frame.loc[frame["split"] == split, columns].copy()
        subset["target"] = subset["target"].astype(float)
        payload = subset.sort_values("sample_id").to_csv(index=False, lineterminator="\n")
        result[split] = {
            "count": len(subset),
            "sha256": hashlib.sha256(payload.encode()).hexdigest(),
            "positive_count": int((subset["observation"] == "positive").sum()),
        }
    return result


def require_binary_splits(frame):
    manifest = cohort_manifest(frame)
    for split, value in manifest.items():
        if not 0 < value["positive_count"] < value["count"]:
            raise ValueError(f"Effective {split} cohort requires positive and unlabeled pairs")
    return manifest


def require_matching_cohorts(left, right):
    if not isinstance(left, dict) or not isinstance(right, dict):
        raise ValueError("Effective cohort manifests are required; regenerate results")
    for split in SPLITS:
        a, b = left.get(split, {}), right.get(split, {})
        if not a.get("sha256") or not a.get("count") or a != b:
            raise ValueError(f"Effective {split} cohorts differ or are missing")
