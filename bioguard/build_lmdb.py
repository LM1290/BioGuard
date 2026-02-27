import os
import io
import lmdb
import torch
import pandas as pd
from tqdm import tqdm
from bioguard.config import LMDB_DIR
from bioguard.featurizer import drug_to_graph
from bioguard.cyp_predictor import CYPPredictor
from bioguard.data_loader import load_twosides_data

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')


def build_graph_lmdb(df: pd.DataFrame, lmdb_path: str, map_size: int = 50 * 1024 * 1024 * 1024):
    os.makedirs(lmdb_path, exist_ok=True)

    predictor = CYPPredictor()
    unique_smiles = pd.concat([df['smiles_a'], df['smiles_b']]).unique()
    print(f"Running Validation Gauntlet on {len(unique_smiles)} unique molecules...")

    env = lmdb.open(lmdb_path, map_size=map_size, readonly=False, lock=True)
    failed_smiles = set()

    with env.begin(write=True) as txn:
        for smiles in tqdm(unique_smiles):
            if txn.get(smiles.encode('utf-8')) is not None:
                continue

            try:
                # 1. 3D PHYSICS GAUNTLET
                graph_data = drug_to_graph(smiles)

                # 2. METABOLIC GAUNTLET
                _ = predictor.predict(smiles)

                # If it survives, commit to the database
                buffer = io.BytesIO()
                torch.save(graph_data, buffer)
                txn.put(smiles.encode('utf-8'), buffer.getvalue())

            except ValueError as e:
                # Catch the loud failures and blacklist the molecule
                failed_smiles.add(smiles)

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
    df_final = load_twosides_data(split_method='cold_drug')
    build_graph_lmdb(df_final, LMDB_DIR)