"""
Data loading and preprocessing module for TWOSIDES DDI dataset.
UPDATED: v5.0 (Enzyme-Ready & Strict Deduplication)
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

# Bump version to v17 to ensure fresh clean build with names preserved
def get_cache_path(split_method):
    return os.path.join(CACHE_DIR, f'twosides_{split_method}_v17_clean.parquet')

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

    # 2. Canonicalize Pair Order (A < B) to prevent duplicates like (A,B) and (B,A)
    print("Canonicalizing pairs...")
    mask = df_pos['drug_a'] > df_pos['drug_b']
    df_pos['drug_a'], df_pos['drug_b'] = np.where(mask, df_pos['drug_b'], df_pos['drug_a']), \
                                         np.where(mask, df_pos['drug_a'], df_pos['drug_b'])
    df_pos['smiles_a'], df_pos['smiles_b'] = np.where(mask, df_pos['smiles_b'], df_pos['smiles_a']), \
                                             np.where(mask, df_pos['smiles_a'], df_pos['smiles_b'])

    df_pos['label'] = 1.0

    # Explicit Deduplication Report
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

    # Safety Net - One final deduplication before save
    final_len_before = len(df_final)
    df_final = df_final.drop_duplicates(subset=['drug_a', 'drug_b'])
    if len(df_final) < final_len_before:
        print(f"[WARNING] Removed {final_len_before - len(df_final)} duplicates found during merge.")

    os.makedirs(CACHE_DIR, exist_ok=True)
    df_final.to_parquet(cache_file, engine='pyarrow')

    return df_final

def _generate_partitioned_negatives(df_pos_subset, anchors, background, all_drugs_df, n_needed, seed, mode='debug'):
    """
    Finite Pool Strategy with Deduplication.
    """
    if n_needed == 0: return pd.DataFrame()

    anchors = np.array(list(set(anchors)))
    background = np.array(list(set(background)))

    # Create meshgrid for candidates
    idx_a, idx_b = np.meshgrid(np.arange(len(anchors)), np.arange(len(background)))
    idx_a = idx_a.flatten()
    idx_b = idx_b.flatten()

    # Remove self-loops
    mask_diff = anchors[idx_a] != background[idx_b]
    idx_a = idx_a[mask_diff]
    idx_b = idx_b[mask_diff]

    candidates_a = anchors[idx_a]
    candidates_b = background[idx_b]

    # Canonicalize to ensure dedup works (A < B)
    swap_mask = candidates_a > candidates_b
    candidates_a[swap_mask], candidates_b[swap_mask] = candidates_b[swap_mask], candidates_a[swap_mask]

    # Unique string keys for fast filtering
    cand_keys = pd.Series(candidates_a).astype(str) + "|" + pd.Series(candidates_b).astype(str)
    unique_mask = ~cand_keys.duplicated()
    candidates_a = candidates_a[unique_mask]
    candidates_b = candidates_b[unique_mask]
    cand_keys = cand_keys[unique_mask]

    # Filter out existing positives
    pos_keys = set(df_pos_subset['drug_a'].astype(str) + "|" + df_pos_subset['drug_b'].astype(str))
    valid_mask = ~cand_keys.isin(pos_keys)

    valid_a = candidates_a[valid_mask]
    valid_b = candidates_b[valid_mask]

    num_available = len(valid_a)
    print(f"[{mode}] Available Negatives: {num_available} (Needed: {n_needed})")

    rng = np.random.default_rng(seed)

    if num_available <= n_needed:
        if num_available < n_needed:
            print(f"[{mode}] WARNING: Insufficient negatives. Returning ALL available.")
        final_a = valid_a
        final_b = valid_b
    else:
        chosen_idx = rng.choice(num_available, size=n_needed, replace=False)
        final_a = valid_a[chosen_idx]
        final_b = valid_b[chosen_idx]

    id_to_smiles = all_drugs_df.set_index('id')['smiles'].to_dict()

    res_df = pd.DataFrame({
        'drug_a': final_a,
        'drug_b': final_b,
        'label': 0.0
    })

    res_df['smiles_a'] = res_df['drug_a'].map(id_to_smiles)
    res_df['smiles_b'] = res_df['drug_b'].map(id_to_smiles)

    return res_df