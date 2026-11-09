import io
import json
import lmdb
import multiprocessing as mp
import os
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from bioguard.config import LMDB_DIR, NUM_WORKERS
from bioguard.data_loader import load_twosides_relations
from bioguard.featurizer import drug_to_graph
from bioguard.graph_cache import check_cache_contract

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')


def init_worker():
    """Reserve an initializer for future per-process featurizer state."""


def process_molecule(smiles):
    try:
        graph_data = drug_to_graph(smiles)
        buffer = io.BytesIO()
        torch.save(graph_data, buffer)
        return smiles, buffer.getvalue(), None
    except Exception as exc:
        return smiles, None, str(exc)


def build_graph_lmdb(df: pd.DataFrame, lmdb_path: str, map_size: int = 50 * 1024 * 1024 * 1024):
    os.makedirs(lmdb_path, exist_ok=True)

    unique_smiles = pd.concat([df['smiles_a'], df['smiles_b']]).unique()

    env = lmdb.open(lmdb_path, map_size=map_size, readonly=False, lock=True)

    try:
        check_cache_contract(env, initialize=True)
    except Exception:
        env.close()
        raise

    # 1. Quick pass to figure out which smiles haven't been processed yet
    with env.begin() as txn:
        smiles_to_process = [s for s in unique_smiles if txn.get(s.encode('utf-8')) is None]

    if not smiles_to_process:
        print("All molecules already exist in LMDB!")
        env.close()
        return

    processes = min(NUM_WORKERS, len(smiles_to_process))
    print(
        f"Featurizing {len(smiles_to_process)} unique molecules "
        f"with {processes} worker processes..."
    )

    failed_smiles = set()

    with mp.Pool(processes=processes, initializer=init_worker) as pool:

        # Open write transaction on the main thread only
        with env.begin(write=True) as txn:

            # imap_unordered yields results as soon as any worker finishes
            iterator = pool.imap_unordered(process_molecule, smiles_to_process)

            for smiles, byte_data, error in tqdm(iterator, total=len(smiles_to_process)):
                if error is not None:
                    # Catch the loud failures and blacklist the molecule
                    failed_smiles.add(smiles)
                else:
                    # Write to DB securely on main thread
                    txn.put(smiles.encode('utf-8'), byte_data)

    env.close()

    report = {
        "schema_version": 1,
        "input_molecules": int(len(unique_smiles)),
        "attempted_molecules": int(len(smiles_to_process)),
        "failed_molecules": sorted(failed_smiles),
        "failed_molecule_count": int(len(failed_smiles)),
    }
    report_path = Path(DATA_DIR) / "lmdb_featurization_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    if failed_smiles:
        print(
            f"\n[WARNING] {len(failed_smiles)} molecules failed graph "
            f"featurization. Source labels were not modified; see {report_path}."
        )
    else:
        print(f"\nAll molecules passed. LMDB Cache built successfully at: {lmdb_path}")
    return report


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)

    relation_bundle = load_twosides_relations()
    build_graph_lmdb(relation_bundle.records, LMDB_DIR)
