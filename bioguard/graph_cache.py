"""Fail closed when graph generation code or runtime differs from a cache."""

import hashlib
import json
import os
from pathlib import Path

import lmdb
import rdkit
import torch
import torch_geometric

from .molecular import STANDARDIZATION_POLICY

CONTRACT_KEY = b"__bioguard_graph_contract__"
_READERS = {}


def graph_contract():
    root = Path(__file__).parent
    return {
        "schema_version": 1,
        "standardization_policy": STANDARDIZATION_POLICY,
        "rdkit": rdkit.__version__,
        "torch": torch.__version__,
        "torch_geometric": torch_geometric.__version__,
        "sources": {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in ("featurizer.py", "molecular.py")
        },
    }


def check_cache_contract(env, initialize=False):
    expected = graph_contract()
    with env.begin(write=initialize) as txn:
        raw = txn.get(CONTRACT_KEY)
        if raw is None and initialize and txn.stat()["entries"] == 0:
            txn.put(CONTRACT_KEY, json.dumps(expected, sort_keys=True).encode())
            return expected
        try:
            actual = json.loads(raw) if raw is not None else None
        except (ValueError, TypeError):
            actual = None
        if actual != expected:
            raise ValueError(
                "Graph cache contract mismatch or missing manifest. "
                "Build a new cache directory with the current featurizer."
            )
    return expected


def read_cache(path):
    key = (os.getpid(), str(Path(path).resolve()))
    if key not in _READERS:
        env = lmdb.open(key[1], readonly=True, lock=False, readahead=False)
        try:
            check_cache_contract(env)
        except Exception:
            env.close()
            raise
        _READERS[key] = env
    return _READERS[key]


def close_readers():
    for env in _READERS.values():
        env.close()
    _READERS.clear()


def filter_cached_pairs(frame, path):
    # Do not retain a parent-process reader while DataLoader workers start.
    key = (os.getpid(), str(Path(path).resolve()))
    existing = _READERS.get(key)
    env = existing or lmdb.open(key[1], readonly=True, lock=False, readahead=False)
    try:
        check_cache_contract(env)
        molecules = set(frame["smiles_a"]) | set(frame["smiles_b"])
        with env.begin() as txn:
            available = {s for s in molecules if txn.get(s.encode()) is not None}
    finally:
        if existing is None:
            env.close()
    mask = frame["smiles_a"].isin(available) & frame["smiles_b"].isin(available)
    return frame.loc[mask].reset_index(drop=True)
