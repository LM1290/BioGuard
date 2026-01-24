"""
Data loading and preprocessing module for TWOSIDES DDI dataset.
UPDATED: v3.0 (Strict Scaffold Split + Partitioned Negatives)

- Implements Bemis-Murcko Scaffold Splitting directly in the loader.
- REPLACED "Global Negative Pool" with "Partitioned Negative Generation".
- GUARANTEE: No drug scaffold in Test has ever been seen in Train.
"""

import pandas as pd
import numpy as np
import os
import warnings
from tdc.multi_pred import DDI
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem.MolStandardize import rdMolStandardize
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
from collections import defaultdict

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
CACHE_VERSION = "v12_scaffold_partitioned"
PROCESSED_FILE = os.path.join(CACHE_DIR, f'twosides_processed_{CACHE_VERSION}.parquet')


def _generate_scaffold(smiles):
    """Compute the Bemis-Murcko scaffold for a SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    if not mol: return None
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    except:
        return None

def _is_valid_smiles(smiles):
    """Validate and standardize SMILES."""
    if not smiles or pd.isna(smiles): return None
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        try:
            # Series B: Safe salt stripping
            clean_mol = rdMolStandardize.ChargeParent(mol)
            if clean_mol.GetNumAtoms() < 2: clean_mol = mol

            # Canonicalize
            return Chem.MolToSmiles(clean_mol)
        except:
            return None
    return None

def _get_fp(smiles):
    """Generate fingerprint (cached helper)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
    return None

def _canonicalize_pair(drug_a, drug_b, smiles_a, smiles_b):
    if drug_a <= drug_b:
        return drug_a, drug_b, smiles_a, smiles_b
    else:
        return drug_b, drug_a, smiles_b, smiles_a


def load_twosides_data(train_negative_ratio=1.0, test_negative_ratio=9.0, random_seed=42):
    """
    Load TWOSIDES with STRICT SCAFFOLD SPLIT.
    """
    if os.path.exists(PROCESSED_FILE):
        print(f"Loading cached data from {PROCESSED_FILE}")
        return pd.read_parquet(PROCESSED_FILE, engine='pyarrow')

    print("Processing TWOSIDES dataset (SCAFFOLD SPLIT MODE)...")
    data = DDI(name='TWOSIDES')
    df_pos = data.get_data()

    df_pos = df_pos.rename(columns={
        'Drug1_ID': 'drug_a', 'Drug2_ID': 'drug_b',
        'Drug1': 'smiles_a', 'Drug2': 'smiles_b'
    })

    # 1. Standardize SMILES
    print("Standardizing SMILES...")
    unique_smiles = pd.concat([df_pos['smiles_a'], df_pos['smiles_b']]).unique()
    results = Parallel(n_jobs=-1)(delayed(_is_valid_smiles)(s) for s in unique_smiles)
    smiles_map = dict(zip(unique_smiles, results))

    df_pos['smiles_a'] = df_pos['smiles_a'].map(smiles_map)
    df_pos['smiles_b'] = df_pos['smiles_b'].map(smiles_map)
    df_pos = df_pos.dropna(subset=['smiles_a', 'smiles_b'])

    # 2. Canonicalize Pairs
    print("Canonicalizing pairs...")
    mask = df_pos['drug_a'] > df_pos['drug_b']
    df_pos['drug_a'], df_pos['drug_b'] = np.where(mask, df_pos['drug_b'], df_pos['drug_a']), \
                                         np.where(mask, df_pos['drug_a'], df_pos['drug_b'])
    df_pos['smiles_a'], df_pos['smiles_b'] = np.where(mask, df_pos['smiles_b'], df_pos['smiles_a']), \
                                             np.where(mask, df_pos['smiles_a'], df_pos['smiles_b'])

    df_pos['label'] = 1.0
    df_pos = df_pos.drop_duplicates(subset=['drug_a', 'drug_b'])

    # --- 3. SCAFFOLD SPLIT LOGIC ---
    print("Computing Scaffolds (This determines the split)...")
    unique_drugs = pd.concat([
        df_pos[['drug_a', 'smiles_a']].rename(columns={'drug_a': 'id', 'smiles_a': 'smiles'}),
        df_pos[['drug_b', 'smiles_b']].rename(columns={'drug_b': 'id', 'smiles_b': 'smiles'})
    ]).drop_duplicates(subset='id')

    # Generate scaffolds
    unique_drugs['scaffold'] = Parallel(n_jobs=-1)(delayed(_generate_scaffold)(s) for s in unique_drugs['smiles'])

    # Group drugs by scaffold
    scaffold_to_drugs = defaultdict(list)
    for _, row in unique_drugs.iterrows():
        if row['scaffold']:
            scaffold_to_drugs[row['scaffold']].append(row['id'])

    scaffolds = list(scaffold_to_drugs.keys())
    print(f"Found {len(scaffolds)} unique scaffolds for {len(unique_drugs)} drugs.")

    # Split Scaffolds (80/10/10)
    train_scaff, temp_scaff = train_test_split(scaffolds, test_size=0.2, random_state=random_seed)
    val_scaff, test_scaff = train_test_split(temp_scaff, test_size=0.5, random_state=random_seed)

    # Map back to Drug IDs
    train_drugs = set([d for s in train_scaff for d in scaffold_to_drugs[s]])
    val_drugs = set([d for s in val_scaff for d in scaffold_to_drugs[s]])
    test_drugs = set([d for s in test_scaff for d in scaffold_to_drugs[s]])

    print(f"Drugs -> Train: {len(train_drugs)}, Val: {len(val_drugs)}, Test: {len(test_drugs)}")

    # Assign Positives to Sets (Strict: Both drugs must belong to the set)
    train_pos = df_pos[df_pos['drug_a'].isin(train_drugs) & df_pos['drug_b'].isin(train_drugs)].copy()
    val_pos = df_pos[df_pos['drug_a'].isin(val_drugs) & df_pos['drug_b'].isin(val_drugs)].copy()
    test_pos = df_pos[df_pos['drug_a'].isin(test_drugs) & df_pos['drug_b'].isin(test_drugs)].copy()

    train_pos['split'] = 'train'
    val_pos['split'] = 'val'
    test_pos['split'] = 'test'

    print(f"Positives -> Train: {len(train_pos)}, Val: {len(val_pos)}, Test: {len(test_pos)}")

    # --- 4. PARTITIONED NEGATIVE SAMPLING ---
    # We generate negatives specifically for each set to maintain the Scaffold/Cold isolation.

    # Train Negatives (1:1 ratio)
    print("Generating Train Negatives...")
    train_neg = _generate_partitioned_negatives(
        train_pos, train_drugs, unique_drugs,
        n_needed=int(len(train_pos) * train_negative_ratio),
        seed=random_seed, mode='train'
    )
    train_neg['split'] = 'train'

    # Val Negatives (1:1 ratio)
    print("Generating Val Negatives...")
    val_neg = _generate_partitioned_negatives(
        val_pos, val_drugs, unique_drugs,
        n_needed=int(len(val_pos) * train_negative_ratio),
        seed=random_seed, mode='val'
    )
    val_neg['split'] = 'val'

    # Test Negatives (1:9 ratio - Realistic Imbalance)
    print("Generating Test Negatives...")
    test_neg = _generate_partitioned_negatives(
        test_pos, test_drugs, unique_drugs,
        n_needed=int(len(test_pos) * test_negative_ratio),
        seed=random_seed, mode='test'
    )
    test_neg['split'] = 'test'

    # Combine
    df_final = pd.concat([
        train_pos, train_neg,
        val_pos, val_neg,
        test_pos, test_neg
    ], axis=0).sample(frac=1, random_state=random_seed).reset_index(drop=True)

    # Stats
    print(f"\nFinal Statistics (SCAFFOLD SPLIT):")
    print(f"  Train: {len(train_pos)+len(train_neg)} | Pos Rate: {(len(train_pos)/(len(train_pos)+len(train_neg))):.2%}")
    print(f"  Test:  {len(test_pos)+len(test_neg)} | Pos Rate: {(len(test_pos)/(len(test_pos)+len(test_neg))):.2%}")

    os.makedirs(CACHE_DIR, exist_ok=True)
    df_final.to_parquet(PROCESSED_FILE, engine='pyarrow')

    return df_final


def _generate_partitioned_negatives(df_pos_subset, allowed_drug_ids, all_drugs_df, n_needed, seed, mode='train'):
    """
    Generate negatives using ONLY drugs from the allowed set.
    """
    if n_needed == 0: return pd.DataFrame()

    # Filter drugs to allowed set
    candidate_drugs = all_drugs_df[all_drugs_df['id'].isin(allowed_drug_ids)].copy()
    drug_ids = candidate_drugs['id'].values
    id_to_smiles = candidate_drugs.set_index('id')['smiles'].to_dict()

    # Pre-compute existing pairs in this subset
    existing_pairs = set(zip(df_pos_subset['drug_a'], df_pos_subset['drug_b']))

    neg_data = []
    attempts = 0
    max_attempts = n_needed * 50

    # If set is too small, fallback to random sampling without hardness
    use_hard = len(drug_ids) > 100

    while len(neg_data) < n_needed and attempts < max_attempts:
        # Simple Random Sampling for speed and safety
        # (Hard mining is risky on small scaffold partitions)
        da, db = np.random.choice(drug_ids, 2, replace=False)

        # Canonicalize
        if da <= db: ca, cb, sa, sb = da, db, id_to_smiles[da], id_to_smiles[db]
        else: ca, cb, sa, sb = db, da, id_to_smiles[db], id_to_smiles[da]

        if (ca, cb) not in existing_pairs:
            # SOFT LABELING:
            # Even for random, if we are in 'test' mode, we might want to check similarity.
            # For now, we stick to 0.0 for pure random to be safe.
            neg_data.append({
                'drug_a': ca, 'drug_b': cb,
                'smiles_a': sa, 'smiles_b': sb,
                'label': 0.0
            })
            existing_pairs.add((ca, cb))

        attempts += 1

    if len(neg_data) < n_needed:
        warnings.warn(f"[{mode}] Could only generate {len(neg_data)}/{n_needed} negatives. Partition too dense/small.")

    return pd.DataFrame(neg_data)