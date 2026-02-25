import os
import lmdb
import pickle
import pandas as pd
from tqdm import tqdm
from bioguard.config import LMDB_DIR
from bioguard.featurizer import drug_to_graph
from bioguard.data_loader import load_twosides_data


def build_graph_lmdb(df: pd.DataFrame, lmdb_path: str, map_size: int = 20 * 1024 * 1024 * 1024):
    """
    Extracts unique SMILES, generates 3D conformer graphs, and caches to LMDB.
    """
    os.makedirs(lmdb_path, exist_ok=True)

    unique_smiles = pd.concat([df['smiles_a'], df['smiles_b']]).unique()
    print(f"Building LMDB Cache for {len(unique_smiles)} molecules (including 3D features)...")

    env = lmdb.open(lmdb_path, map_size=map_size, readonly=False, lock=True)

    with env.begin(write=True) as txn:
        for smiles in tqdm(unique_smiles):
            if txn.get(smiles.encode('utf-8')) is not None:
                continue

            graph_data = drug_to_graph(smiles)
            txn.put(smiles.encode('utf-8'), pickle.dumps(graph_data))

    env.close()
    print(f"LMDB Cache built successfully at: {lmdb_path}")


if __name__ == "__main__":
    df_final = load_twosides_data(split_method='scaffold')
    build_graph_lmdb(df_final, LMDB_DIR)