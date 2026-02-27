# bioguard/build_lmdb.py

import os
import io
import lmdb
import torch
import pandas as pd
from tqdm import tqdm
from bioguard.config import LMDB_DIR
from bioguard.featurizer import drug_to_graph
from bioguard.data_loader import load_twosides_data


def build_graph_lmdb(df: pd.DataFrame, lmdb_path: str, map_size: int = 50 * 1024 * 1024 * 1024):
    """
    Extracts unique SMILES, generates 3D conformer graphs, and caches to LMDB
    using native PyTorch serialization for zero-copy loading.
    Map size increased to 50GB to handle production-scale tensor storage.
    """
    os.makedirs(lmdb_path, exist_ok=True)

    unique_smiles = pd.concat([df['smiles_a'], df['smiles_b']]).unique()
    print(f"Building Production LMDB Cache for {len(unique_smiles)} molecules...")

    env = lmdb.open(lmdb_path, map_size=map_size, readonly=False, lock=True)

    with env.begin(write=True) as txn:
        for smiles in tqdm(unique_smiles):
            if txn.get(smiles.encode('utf-8')) is not None:
                continue

            # 1. Generate the PyG Data Object
            graph_data = drug_to_graph(smiles)

            # 2. Serialize natively via PyTorch into a byte buffer
            buffer = io.BytesIO()
            torch.save(graph_data, buffer)

            # 3. Store the raw tensor bytes in the database
            txn.put(smiles.encode('utf-8'), buffer.getvalue())

    env.close()
    print(f"Production LMDB Cache built successfully at: {lmdb_path}")


if __name__ == "__main__":
    # Ensure you are using the scaffold split as the source of truth
    df_final = load_twosides_data(split_method='scaffold')
    build_graph_lmdb(df_final, LMDB_DIR)