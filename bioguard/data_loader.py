"""
Data loading and preprocessing module for TWOSIDES DDI dataset.

Implements canonical pair representation and stratified negative sampling
to prevent data leakage and provide realistic class imbalance in test sets.
"""

import pandas as pd
import numpy as np
import os
import warnings
from tdc.multi_pred import DDI
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem.MolStandardize import rdMolStandardize
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
CACHE_VERSION = "v10_global_negative_tracking"
PROCESSED_FILE = os.path.join(CACHE_DIR, f'twosides_processed_{CACHE_VERSION}.parquet')


def _is_valid_smiles(smiles):
    """Validate and standardize SMILES with strict performance limits."""
    if not smiles or pd.isna(smiles):
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        try:
            # 1. Robust Salt Stripping
            # ChargeParent removes salts and ions; it rarely fails.
            clean_mol = rdMolStandardize.ChargeParent(mol)

            # 2. Safe Tautomer Enumeration
            # Set strict limits to avoid "Max transforms reached" errors.
            te = rdMolStandardize.TautomerEnumerator()
            te.SetMaxTautomers(20)
            te.SetMaxTransforms(20)

            try:
                # Attempt to find the canonical tautomer
                clean_mol = te.Canonicalize(clean_mol)
            except:
                # FALLBACK: If tautomers fail, we keep the salt-stripped molecule
                # rather than returning None and deleting the data.
                pass

            return Chem.MolToSmiles(clean_mol)
        except Exception:
            # Only return None if the molecule is fundamentally unreadable
            return None
    return None


def _get_fp(smiles):
    """Generate Morgan fingerprint with safety-first standardization."""
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        try:
            # 1. Always strip salts first
            clean_mol = rdMolStandardize.ChargeParent(mol)

            # 2. Setup enumerator with performance caps
            te = rdMolStandardize.TautomerEnumerator()
            te.SetMaxTautomers(20)
            te.SetMaxTransforms(20)

            try:
                # 3. Try tautomer fixing
                clean_mol = te.Canonicalize(clean_mol)
            except:
                pass

            return AllChem.GetMorganFingerprintAsBitVect(clean_mol, 2, nBits=1024)
        except:
            return None
    return None


def _canonicalize_pair(drug_a, drug_b, smiles_a, smiles_b):
    """
    Canonicalize drug pairs by alphabetical ordering.
    Ensures (A,B) and (B,A) are treated as identical pairs.
    Critical for preventing data leakage.
    """
    if drug_a <= drug_b:
        return drug_a, drug_b, smiles_a, smiles_b
    else:
        return drug_b, drug_a, smiles_b, smiles_a


def load_twosides_data(train_negative_ratio=1.0, test_negative_ratio=9.0, random_seed=42):
    """
    Load and process TWOSIDES DDI dataset with realistic evaluation setup.
    
    CRITICAL: Generates ALL negatives first, then splits them to prevent duplicates.
    
    Key features:
    - Canonicalizes pairs to prevent (A,B)/(B,A) duplicates
    - Balanced training set (50/50) for effective learning
    - Realistic test set (10% positive) matching real-world DDI prevalence
    - Hard negative mining for challenging evaluation
    - Global negative tracking prevents duplicates across splits
    
    Args:
        train_negative_ratio: Ratio of negatives to positives for train/val (1.0 = balanced)
        test_negative_ratio: Ratio of negatives to positives for test (9.0 = 10% positive)
        random_seed: Random seed for reproducibility
        
    Returns:
        DataFrame with columns: drug_a, drug_b, smiles_a, smiles_b, label, split
    """
    if os.path.exists(PROCESSED_FILE):
        print(f"Loading cached data from {PROCESSED_FILE}")
        return pd.read_parquet(PROCESSED_FILE, engine='pyarrow')

    print("Processing TWOSIDES dataset...")
    data = DDI(name='TWOSIDES')
    df_pos = data.get_data()

    df_pos = df_pos.rename(columns={
        'Drug1_ID': 'drug_a', 'Drug2_ID': 'drug_b',
        'Drug1': 'smiles_a', 'Drug2': 'smiles_b'
    })

    # Validate and standardize SMILES in parallel
    print("Standardizing SMILES structures...")
    unique_smiles = pd.concat([df_pos['smiles_a'], df_pos['smiles_b']]).unique()

    # results now contains clean SMILES strings or None
    results = Parallel(n_jobs=-1)(delayed(_is_valid_smiles)(s) for s in unique_smiles)
    smiles_map = dict(zip(unique_smiles, results))

    # Apply the clean SMILES to the dataframe
    df_pos['smiles_a'] = df_pos['smiles_a'].map(smiles_map)
    df_pos['smiles_b'] = df_pos['smiles_b'].map(smiles_map)

    # Drop rows where standardization failed
    invalid_count = df_pos[['smiles_a', 'smiles_b']].isna().any(axis=1).sum()
    if invalid_count > 0:
        print(f"Removed {invalid_count} pairs with invalid or unstandardizable SMILES")

    df_pos = df_pos.dropna(subset=['smiles_a', 'smiles_b'])
    # Canonicalize pairs
    print("Canonicalizing drug pairs...")
    original_count = len(df_pos)

    # Identify rows where Drug A > Drug B (alphabetically)
    mask = df_pos['drug_a'] > df_pos['drug_b']

    drug_a_vals = df_pos['drug_a'].values
    drug_b_vals = df_pos['drug_b'].values

    df_pos['drug_a'] = np.where(mask, drug_b_vals, drug_a_vals)
    df_pos['drug_b'] = np.where(mask, drug_a_vals, drug_b_vals)

    # Swap SMILES where necessary
    smiles_a_vals = df_pos['smiles_a'].values
    smiles_b_vals = df_pos['smiles_b'].values

    df_pos['smiles_a'] = np.where(mask, smiles_b_vals, smiles_a_vals)
    df_pos['smiles_b'] = np.where(mask, smiles_a_vals, smiles_b_vals)

    df_pos['label'] = 1.0
    df_pos = df_pos.drop_duplicates(subset=['drug_a', 'drug_b'], keep='first')
    
    duplicates_removed = original_count - len(df_pos)
    print(f"Removed {duplicates_removed} duplicate pairs ({duplicates_removed/original_count*100:.1f}%)")
    print(f"Unique positive interactions: {len(df_pos)}")

    # Split positives into train/val/test
    print("Creating stratified splits...")
    train_val_pos, test_pos = train_test_split(
        df_pos, test_size=0.2, random_state=random_seed
    )
    train_pos, val_pos = train_test_split(
        train_val_pos, test_size=0.125, random_state=random_seed
    )
    
    print(f"Positive samples - Train: {len(train_pos)}, Val: {len(val_pos)}, Test: {len(test_pos)}")

    # CRITICAL FIX: Generate ALL negatives first, then split
    # This prevents duplicates across splits
    print("Generating negative samples (global pool)...")
    
    total_negatives_needed = (
        int(len(train_pos) * train_negative_ratio) +
        int(len(val_pos) * train_negative_ratio) +
        int(len(test_pos) * test_negative_ratio)
    )
    
    print(f"  Total negatives needed: {total_negatives_needed}")
    print(f"    Train: {int(len(train_pos) * train_negative_ratio)}")
    print(f"    Val:   {int(len(val_pos) * train_negative_ratio)}")
    print(f"    Test:  {int(len(test_pos) * test_negative_ratio)}")
    
    # Generate all negatives in one call
    all_negatives = _generate_negatives_hybrid(
        df_pos, 
        n_negatives=total_negatives_needed, 
        seed=random_seed
    )
    
    # Shuffle and split negatives
    all_negatives = all_negatives.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    
    n_train_neg = int(len(train_pos) * train_negative_ratio)
    n_val_neg = int(len(val_pos) * train_negative_ratio)
    
    train_neg = all_negatives.iloc[:n_train_neg].copy()
    val_neg = all_negatives.iloc[n_train_neg:n_train_neg+n_val_neg].copy()
    test_neg = all_negatives.iloc[n_train_neg+n_val_neg:].copy()
    
    # Add split labels
    train_pos['split'] = 'train'
    val_pos['split'] = 'val'
    test_pos['split'] = 'test'
    train_neg['split'] = 'train'
    val_neg['split'] = 'val'
    test_neg['split'] = 'test'
    
    # Combine and shuffle within splits
    train_df = pd.concat([train_pos, train_neg], axis=0).sample(frac=1, random_state=random_seed)
    val_df = pd.concat([val_pos, val_neg], axis=0).sample(frac=1, random_state=random_seed)
    test_df = pd.concat([test_pos, test_neg], axis=0).sample(frac=1, random_state=random_seed)
    
    df_final = pd.concat([train_df, val_df, test_df], axis=0).reset_index(drop=True)
    
    # Report statistics
    print(f"\nDataset statistics:")
    print(f"  Train: {len(train_df)} pairs ({train_df['label'].mean()*100:.1f}% positive)")
    print(f"  Val:   {len(val_df)} pairs ({val_df['label'].mean()*100:.1f}% positive)")
    print(f"  Test:  {len(test_df)} pairs ({test_df['label'].mean()*100:.1f}% positive)")
    print(f"  Note: Test set uses realistic {test_df['label'].mean()*100:.1f}% positive rate")

    os.makedirs(CACHE_DIR, exist_ok=True)
    df_final.to_parquet(PROCESSED_FILE, engine='pyarrow')
    print(f"Cached to {PROCESSED_FILE}")
    
    return df_final


def _generate_negatives_hybrid(df_pos, n_negatives, seed):
    """
    Generate negative samples using hybrid strategy.
    Ensures all generated negatives are unique (no duplicates).
    
    Args:
        df_pos: Positive interactions dataframe
        n_negatives: Number of negatives to generate
        seed: Random seed
        
    Returns:
        DataFrame of negative samples (guaranteed unique)
    """
    np.random.seed(seed)

    # Prepare drug mappings
    unique_drugs = pd.concat([
        df_pos[['drug_a', 'smiles_a']].rename(columns={'drug_a': 'id', 'smiles_a': 'smiles'}),
        df_pos[['drug_b', 'smiles_b']].rename(columns={'drug_b': 'id', 'smiles_b': 'smiles'})
    ]).drop_duplicates(subset='id')

    drug_ids = unique_drugs['id'].values
    id_to_smiles = unique_drugs.set_index('id')['smiles'].to_dict()

    # Precompute fingerprints for hard negative mining
    max_jobs = min(os.cpu_count() or 1, 8)
    fps = Parallel(n_jobs=max_jobs)(delayed(_get_fp)(s) for s in unique_drugs['smiles'])
    id_to_fp = dict(zip(unique_drugs['id'], fps))
    
    valid_ids = [i for i, fp in id_to_fp.items() if fp is not None]

    # Robust canonicalization (handles cases where input df might not be sorted)
    mask = df_pos['drug_a'] > df_pos['drug_b']

    canon_a = np.where(mask, df_pos['drug_b'], df_pos['drug_a'])
    canon_b = np.where(mask, df_pos['drug_a'], df_pos['drug_b'])

    # Create set directly from zipped arrays
    existing_pairs = set(zip(canon_a, canon_b))
    n_hard = n_negatives // 2
    n_random = n_negatives - n_hard

    neg_data = []

    # Random negative sampling
    print(f"  Generating {n_random} random negatives...")
    count = 0
    attempts = 0
    max_attempts = n_random * 10
    
    while count < n_random and attempts < max_attempts:
        cand_a = np.random.choice(drug_ids, size=1000)
        cand_b = np.random.choice(drug_ids, size=1000)
        for da, db in zip(cand_a, cand_b):
            if da != db:
                ca, cb, sa, sb = _canonicalize_pair(da, db, id_to_smiles[da], id_to_smiles[db])
                if (ca, cb) not in existing_pairs:
                    neg_data.append({
                        'drug_a': ca, 
                        'drug_b': cb, 
                        'smiles_a': sa, 
                        'smiles_b': sb,
                        'label': 0.0
                    })
                    existing_pairs.add((ca, cb))
                    count += 1
                    if count >= n_random: 
                        break
        attempts += 1000

    # Hard negative sampling
    print(f"  Generating {n_hard} hard negatives...")
    sampled_anchors = np.random.choice(valid_ids, size=min(len(valid_ids), 5000), replace=False)

    for anchor in sampled_anchors:
        if len(neg_data) >= n_negatives: 
            break

        anchor_fp = id_to_fp[anchor]
        if anchor_fp is None:
            continue
            
        candidates = np.random.choice(valid_ids, size=100, replace=True)
        cand_fps = [id_to_fp[c] for c in candidates if id_to_fp[c] is not None]
        
        if not cand_fps:
            continue

        sims = DataStructs.BulkTanimotoSimilarity(anchor_fp, cand_fps)
        sorted_indices = np.argsort(sims)[::-1]

        for idx in sorted_indices:
            target = candidates[idx]
            if anchor == target: 
                continue

            ca, cb, sa, sb = _canonicalize_pair(
                anchor, target, 
                id_to_smiles[anchor], id_to_smiles[target]
            )

            if sims[idx] > 0.6 and (ca, cb) not in existing_pairs:
                neg_data.append({
                    'drug_a': ca, 
                    'drug_b': cb, 
                    'smiles_a': sa,
                    'smiles_b': sb, 
                    'label': 0.0
                })
                existing_pairs.add((ca, cb))
                if len(neg_data) >= n_negatives: 
                    break

    # Fill remaining with random sampling
    max_fill_attempts = min(n_negatives * 10, len(drug_ids) * len(drug_ids) // 2)
    fill_attempts = 0
    
    while len(neg_data) < n_negatives and fill_attempts < max_fill_attempts:
        da, db = np.random.choice(drug_ids, 2, replace=False)
        if da != db:
            ca, cb, sa, sb = _canonicalize_pair(da, db, id_to_smiles[da], id_to_smiles[db])
            if (ca, cb) not in existing_pairs:
                neg_data.append({
                    'drug_a': ca, 
                    'drug_b': cb, 
                    'smiles_a': sa, 
                    'smiles_b': sb, 
                    'label': 0.0
                })
                existing_pairs.add((ca, cb))
        fill_attempts += 1
    
    if len(neg_data) < n_negatives:
        warnings.warn(
            f"Generated {len(neg_data)}/{n_negatives} negatives. "
            f"Dense interaction graph limits negative sampling."
        )

    print(f"  Generated {len(neg_data)} unique negatives (no duplicates)")
    return pd.DataFrame(neg_data)
