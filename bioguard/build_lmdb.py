import os
import io
import lmdb
import torch
import pandas as pd
from tqdm import tqdm
import multiprocessing as mp
from bioguard.config import LMDB_DIR
from bioguard.featurizer import drug_to_graph
from bioguard.cyp_predictor import CYPPredictor
from bioguard.data_loader import load_twosides_data

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')

# Global variable for the worker processes so we don't reload the
# CYPPredictor for every single molecule.
worker_predictor = None


def init_worker():
    global worker_predictor
    # Initialize CYPPredictor on CPU to avoid CUDA multiprocessing locks
    worker_predictor = CYPPredictor()


def process_molecule(smiles):
    try:
        # 1. 3D PHYSICS GAUNTLET
        graph_data = drug_to_graph(smiles)

        # 2. METABOLIC GAUNTLET
        _ = worker_predictor.predict(smiles)

        # Serialize to bytes
        buffer = io.BytesIO()
        torch.save(graph_data, buffer)

        # Return success
        return smiles, buffer.getvalue(), None
    except Exception as e:
        # Return failure
        return smiles, None, str(e)


def build_graph_lmdb(df: pd.DataFrame, lmdb_path: str, map_size: int = 50 * 1024 * 1024 * 1024):
    os.makedirs(lmdb_path, exist_ok=True)

    unique_smiles = pd.concat([df['smiles_a'], df['smiles_b']]).unique()

    env = lmdb.open(lmdb_path, map_size=map_size, readonly=False, lock=True)

    # 1. Quick pass to figure out which smiles haven't been processed yet
    with env.begin() as txn:
        smiles_to_process = [s for s in unique_smiles if txn.get(s.encode('utf-8')) is None]

    if not smiles_to_process:
        print("All molecules already exist in LMDB!")
        env.close()
        return

    print(
        f"Running Validation Gauntlet on {len(smiles_to_process)} unique molecules using {mp.cpu_count()} CPU cores...")

    failed_smiles = set()

    # 2. Start multiprocessing pool
    # Use spawn if you are using CUDA inside CYPPredictor, otherwise default is fine.
    with mp.Pool(processes=mp.cpu_count(), initializer=init_worker) as pool:

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

    # --- THE PURGE ---
    if failed_smiles:
        print(f"\n[WARNING] {len(failed_smiles)} molecules failed the physics/inference gauntlet.")
        original_len = len(df)

        # Drop ANY interaction pair that contains a blacklisted molecule
        df_clean = df[~df['smiles_a'].isin(failed_smiles) & ~df['smiles_b'].isin(failed_smiles)]
        dropped_pairs = original_len - len(df_clean)

        print(f"Purged {dropped_pairs} invalid DDI pairs from the dataset.")
        print(f"Final Cleaned Dataset Size: {len(df_clean)} pairs.")

        # Overwrite the canonical dataset so train.py never encounters them
        parquet_path = os.path.join(DATA_DIR, 'twosides.parquet')
        df_clean.to_parquet(parquet_path, index=False)
        print(f"Overwrote canonical dataset at {parquet_path}")
    else:
        print(f"\nAll molecules passed. LMDB Cache built successfully at: {lmdb_path}")


if __name__ == "__main__":
    # If using PyTorch/CUDA inside the predictor, this ensures smooth multiprocessing
    mp.set_start_method('spawn', force=True)

    df_final = load_twosides_data(split_method='cold_drug')
    build_graph_lmdb(df_final, LMDB_DIR)