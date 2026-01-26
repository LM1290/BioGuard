"""
Data loading and preprocessing module for TWOSIDES DDI dataset.
UPDATED: v4.0 (Cache Busting & Strict Deduplication)
"""

import pandas as pd
import numpy as np
import os
import warnings
from tdc.multi_pred import DDI
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem.MolStandardize import rdMolStandardize
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
from collections import defaultdict

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')

# FIX 1: BUMP VERSION TO v16 TO FORCE REBUILD (Ignore stale v15 files)
def get_cache_path(split_method):
    return os.path.join(CACHE_DIR, f'twosides_{split_method}_v16_clean.parquet')

def _generate_scaffold(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if not mol: return None
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    except:
        return None

def _is_valid_smiles(smiles):
    if not smiles or pd.isna(smiles): return None
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        try:
            clean_mol = rdMolStandardize.ChargeParent(mol)
            if clean_mol.GetNumAtoms() < 2: clean_mol = mol
            return Chem.MolToSmiles(clean_mol)
        except:
            return None
    return None

def load_twosides_data(train_negative_ratio=1.0, test_negative_ratio=9.0, random_seed=42, split_method='scaffold'):
    """
    Load TWOSIDES with selectable split method.
    """
    cache_file = get_cache_path(split_method)

    if os.path.exists(cache_file):
        print(f"Loading cached {split_method} data from {cache_file}")
        return pd.read_parquet(cache_file, engine='pyarrow')

    print(f"Processing TWOSIDES ({split_method.upper()} SPLIT)...")
    data = DDI(name='TWOSIDES')
    df_pos = data.get_data()

    print(f"Raw TDC Rows: {len(df_pos)}")

    df_pos = df_pos.rename(columns={
        'Drug1_ID': 'drug_a', 'Drug2_ID': 'drug_b',
        'Drug1': 'smiles_a', 'Drug2': 'smiles_b'
    })

    # 1. Standardize
    print("Standardizing SMILES...")
    unique_smiles = pd.concat([df_pos['smiles_a'], df_pos['smiles_b']]).unique()
    results = Parallel(n_jobs=-1)(delayed(_is_valid_smiles)(s) for s in unique_smiles)
    smiles_map = dict(zip(unique_smiles, results))

    df_pos['smiles_a'] = df_pos['smiles_a'].map(smiles_map)
    df_pos['smiles_b'] = df_pos['smiles_b'].map(smiles_map)
    df_pos = df_pos.dropna(subset=['smiles_a', 'smiles_b'])

    # 2. Canonicalize
    print("Canonicalizing pairs...")
    mask = df_pos['drug_a'] > df_pos['drug_b']
    df_pos['drug_a'], df_pos['drug_b'] = np.where(mask, df_pos['drug_b'], df_pos['drug_a']), \
                                         np.where(mask, df_pos['drug_a'], df_pos['drug_b'])
    df_pos['smiles_a'], df_pos['smiles_b'] = np.where(mask, df_pos['smiles_b'], df_pos['smiles_a']), \
                                             np.where(mask, df_pos['smiles_a'], df_pos['smiles_b'])

    df_pos['label'] = 1.0

    # FIX 2: Explicit Deduplication Report
    before_dedup = len(df_pos)
    df_pos = df_pos.drop_duplicates(subset=['drug_a', 'drug_b'])
    after_dedup = len(df_pos)
    print(f"Collapsed Polypharmacy: {before_dedup} -> {after_dedup} unique pairs")

    unique_drugs = pd.concat([
        df_pos[['drug_a', 'smiles_a']].rename(columns={'drug_a': 'id', 'smiles_a': 'smiles'}),
        df_pos[['drug_b', 'smiles_b']].rename(columns={'drug_b': 'id', 'smiles_b': 'smiles'})
    ]).drop_duplicates(subset='id')

    # --- SPLIT LOGIC ---
    if split_method == 'random':
        print("Performing RANDOM Split (Baseline)...")
        train_pos, temp_pos = train_test_split(df_pos, test_size=0.2, random_state=random_seed)
        val_pos, test_pos = train_test_split(temp_pos, test_size=0.5, random_state=random_seed)

        train_pos['split'] = 'train'
        val_pos['split'] = 'val'
        test_pos['split'] = 'test'

        all_ids = list(unique_drugs['id'])
        train_anchors = val_anchors = test_anchors = all_ids
        train_bg = val_bg = test_bg = all_ids

    else: # SCAFFOLD
        print("Performing SCAFFOLD Split (Strict)...")
        unique_drugs['scaffold'] = Parallel(n_jobs=-1)(delayed(_generate_scaffold)(s) for s in unique_drugs['smiles'])

        scaffold_to_drugs = defaultdict(list)
        for _, row in unique_drugs.iterrows():
            if row['scaffold']:
                scaffold_to_drugs[row['scaffold']].append(row['id'])

        scaffolds = list(scaffold_to_drugs.keys())
        train_scaff, temp_scaff = train_test_split(scaffolds, test_size=0.2, random_state=random_seed)
        val_scaff, test_scaff = train_test_split(temp_scaff, test_size=0.5, random_state=random_seed)

        train_drugs = set([d for s in train_scaff for d in scaffold_to_drugs[s]])
        val_drugs = set([d for s in val_scaff for d in scaffold_to_drugs[s]])
        test_drugs = set([d for s in test_scaff for d in scaffold_to_drugs[s]])

        # Priority Assignment
        is_a_test = df_pos['drug_a'].isin(test_drugs)
        is_b_test = df_pos['drug_b'].isin(test_drugs)
        test_mask = is_a_test | is_b_test

        is_a_val = df_pos['drug_a'].isin(val_drugs)
        is_b_val = df_pos['drug_b'].isin(val_drugs)
        val_mask = (is_a_val | is_b_val) & (~test_mask)

        train_mask = ~(test_mask | val_mask)

        train_pos = df_pos[train_mask].copy()
        val_pos = df_pos[val_mask].copy()
        test_pos = df_pos[test_mask].copy()

        train_pos['split'] = 'train'
        val_pos['split'] = 'val'
        test_pos['split'] = 'test'

        train_anchors = train_bg = list(train_drugs)
        val_anchors = list(val_drugs)
        val_bg = list(train_drugs | val_drugs)
        test_anchors = list(test_drugs)
        test_bg = list(unique_drugs['id'])

    # --- NEGATIVE GENERATION ---
    print("Generating Negatives...")

    train_neg = _generate_partitioned_negatives(
        train_pos, train_anchors, train_bg, unique_drugs,
        int(len(train_pos) * train_negative_ratio), random_seed, mode='train'
    )
    train_neg['split'] = 'train'

    val_neg = _generate_partitioned_negatives(
        val_pos, val_anchors, val_bg, unique_drugs,
        int(len(val_pos) * train_negative_ratio), random_seed, mode='val'
    )
    val_neg['split'] = 'val'

    test_neg = _generate_partitioned_negatives(
        test_pos, test_anchors, test_bg, unique_drugs,
        int(len(test_pos) * test_negative_ratio), random_seed, mode='test'
    )
    test_neg['split'] = 'test'

    df_final = pd.concat([
        train_pos, train_neg,
        val_pos, val_neg,
        test_pos, test_neg
    ], axis=0).sample(frac=1, random_state=random_seed).reset_index(drop=True)

    # FIX 3: Safety Net - One final deduplication before save
    final_len_before = len(df_final)
    df_final = df_final.drop_duplicates(subset=['drug_a', 'drug_b'])
    if len(df_final) < final_len_before:
        print(f"[WARNING] Removed {final_len_before - len(df_final)} duplicates found during merge.")

    os.makedirs(CACHE_DIR, exist_ok=True)
    df_final.to_parquet(cache_file, engine='pyarrow')

    return df_final


def _generate_partitioned_negatives(df_pos_subset, anchors, background, all_drugs_df, n_needed, seed, mode='debug'):
    """
    Rejection Sampling Strategy (O(K) complexity) using Python sets for O(1) lookups.
    """
    if n_needed == 0: return pd.DataFrame()

    # 1. Build O(1) lookup set for existing positive pairs
    # Canonicalize pairs (min, max) to handle undirected edges
    pos_pairs_set = set(zip(
        np.minimum(df_pos_subset['drug_a'], df_pos_subset['drug_b']),
        np.maximum(df_pos_subset['drug_a'], df_pos_subset['drug_b'])
    ))

    # Ensure unique pools
    anchors = np.array(list(set(anchors)))
    background = np.array(list(set(background)))

    negatives = set()
    rng = np.random.default_rng(seed)

    # Safety: Max attempts to prevent infinite loops if saturation is high
    max_attempts = n_needed * 50
    attempts = 0

    print(f"[{mode}] Generating {n_needed} negatives via Rejection Sampling...")

    while len(negatives) < n_needed and attempts < max_attempts:
        # Sample in batches for vectorization speedup (Python loop overhead reduction)
        remaining = n_needed - len(negatives)
        batch_size = max(remaining * 2, 500)

        a_samples = rng.choice(anchors, size=batch_size)
        b_samples = rng.choice(background, size=batch_size)

        # Canonicalize samples
        mins = np.minimum(a_samples, b_samples)
        maxs = np.maximum(a_samples, b_samples)

        # Filter valid pairs
        for d1, d2 in zip(mins, maxs):
            if d1 == d2: continue  # No self-loops

            pair = (d1, d2)
            # O(1) Lookup
            if pair not in pos_pairs_set and pair not in negatives:
                negatives.add(pair)
                if len(negatives) == n_needed:
                    break

        attempts += batch_size

    if len(negatives) < n_needed:
        print(f"[{mode}] WARNING: Could only generate {len(negatives)}/{n_needed} negatives after {attempts} attempts.")

    # Convert to DataFrame
    res_df = pd.DataFrame(list(negatives), columns=['drug_a', 'drug_b'])
    res_df['label'] = 0.0

    # Map SMILES
    id_to_smiles = all_drugs_df.set_index('id')['smiles'].to_dict()
    res_df['smiles_a'] = res_df['drug_a'].map(id_to_smiles)
    res_df['smiles_b'] = res_df['drug_b'].map(id_to_smiles)

    return res_df