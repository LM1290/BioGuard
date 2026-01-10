"""
Drug-disjoint split for evaluating generalization to completely new drugs.

This is a research feature that tests model performance on drugs never seen
during training - significantly harder than pair-disjoint evaluation.
"""

import pandas as pd
import numpy as np


def get_drug_disjoint_split(df, test_size=0.2, val_size=0.1, seed=42):
    """
    Create drug-disjoint split for evaluating COMPLETELY NEW drugs.
    
    Significantly harder than pair-disjoint. Tests generalization to
    novel molecular entities not seen during training.
    
    Args:
        df: DataFrame with drug pairs
        test_size: Fraction for test set
        val_size: Fraction for validation set
        seed: Random seed
        
    Returns:
        train_df, val_df, test_df
    """
    np.random.seed(seed)
    
    # Get unique drugs and shuffle
    all_drugs = list(set(df['drug_a']).union(set(df['drug_b'])))
    np.random.shuffle(all_drugs)
    
    # Split drugs
    n_drugs = len(all_drugs)
    n_test = int(n_drugs * test_size)
    n_val = int(n_drugs * val_size)
    
    test_drugs = set(all_drugs[:n_test])
    val_drugs = set(all_drugs[n_test:n_test + n_val])
    train_drugs = set(all_drugs[n_test + n_val:])
    
    print(f"Drug-disjoint split:")
    print(f"  Train: {len(train_drugs)} drugs")
    print(f"  Val:   {len(val_drugs)} drugs")
    print(f"  Test:  {len(test_drugs)} drugs")
    
    # Assign pairs where both drugs belong to same split
    train_pairs = []
    val_pairs = []
    test_pairs = []
    
    for _, row in df.iterrows():
        drug_a = row['drug_a']
        drug_b = row['drug_b']
        
        if drug_a in train_drugs and drug_b in train_drugs:
            train_pairs.append(row)
        elif drug_a in val_drugs and drug_b in val_drugs:
            val_pairs.append(row)
        elif drug_a in test_drugs and drug_b in test_drugs:
            test_pairs.append(row)
    
    train_df = pd.DataFrame(train_pairs)
    val_df = pd.DataFrame(val_pairs)
    test_df = pd.DataFrame(test_pairs)
    
    print(f"  Train: {len(train_df)} pairs ({train_df['label'].mean():.1%} positive)")
    print(f"  Val:   {len(val_df)} pairs ({val_df['label'].mean():.1%} positive)")
    print(f"  Test:  {len(test_df)} pairs ({test_df['label'].mean():.1%} positive)")
    
    # Verify no drug overlap
    train_drugs_check = set(train_df['drug_a']).union(set(train_df['drug_b']))
    val_drugs_check = set(val_df['drug_a']).union(set(val_df['drug_b']))
    test_drugs_check = set(test_df['drug_a']).union(set(test_df['drug_b']))
    
    overlap_tv = len(train_drugs_check & val_drugs_check)
    overlap_tt = len(train_drugs_check & test_drugs_check)
    overlap_vt = len(val_drugs_check & test_drugs_check)
    
    if overlap_tv > 0 or overlap_tt > 0 or overlap_vt > 0:
        raise ValueError(f"Drug leakage detected: T-V={overlap_tv}, T-T={overlap_tt}, V-T={overlap_vt}")
    
    return train_df, val_df, test_df
