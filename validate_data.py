"""
Data Validation Script

Verifies that the dataset has no data leakage issues:
1. No duplicate pairs in the dataset
2. No pair overlap between train/val/test splits
3. Reports statistics on drug overlap (expected for pair-disjoint)
"""

import pandas as pd
import numpy as np
from collections import Counter

from bioguard.data_loader import load_twosides_data



def canonicalize_pair(drug_a, drug_b):
    """Canonicalize a drug pair for comparison."""
    if drug_a <= drug_b:
        return (drug_a, drug_b)
    else:
        return (drug_b, drug_a)


def validate_dataset():
    """Run comprehensive validation checks on the dataset."""
    
    print("="*80)
    print("BIOGUARD DATA VALIDATION")
    print("="*80)
    
    # Load data
    print("\n1. Loading dataset...")
    df = load_twosides_data()
    print(f"   Total samples: {len(df)}")
    print(f"   Positive samples: {df['label'].sum():.0f} ({df['label'].mean()*100:.1f}%)")
    print(f"   Negative samples: {(1-df['label']).sum():.0f} ({(1-df['label']).mean()*100:.1f}%)")
    
    # Check for duplicates in full dataset
    print("\n2. Checking for duplicate pairs in dataset...")
    pairs = [canonicalize_pair(row['drug_a'], row['drug_b']) for _, row in df.iterrows()]
    pair_counts = Counter(pairs)
    duplicates = {pair: count for pair, count in pair_counts.items() if count > 1}
    
    if duplicates:
        print(f"   [WARNING]  WARNING: Found {len(duplicates)} duplicate pairs!")
        print(f"   Total duplicate entries: {sum(count - 1 for count in duplicates.values())}")
        print("\n   Top 10 duplicates:")
        for i, (pair, count) in enumerate(sorted(duplicates.items(), key=lambda x: x[1], reverse=True)[:10], 1):
            print(f"      {i}. {pair}: {count} occurrences")
        print("\n   [ERROR] CRITICAL: Data leakage risk - pairs appear multiple times!")
        print("   → This will inflate model performance metrics")
        print("   → Use the canonicalization fix in data_loader.py")
    else:
        print("   [OK] PASS: No duplicate pairs found")
        print("   → Each drug combination appears exactly once")
    
    # Check for reverse pairs
    print("\n3. Checking for reverse pairs (A,B) vs (B,A)...")
    forward_pairs = [(row['drug_a'], row['drug_b']) for _, row in df.iterrows()]
    reverse_pairs = [(row['drug_b'], row['drug_a']) for _, row in df.iterrows()]
    
    reverse_found = set(forward_pairs) & set(reverse_pairs)
    if reverse_found:
        print(f"   [WARNING]  WARNING: Found {len(reverse_found)} reverse pair duplicates!")
        print("   Examples:")
        for pair in list(reverse_found)[:5]:
            print(f"      {pair} and {(pair[1], pair[0])}")
        print("\n   [ERROR] CRITICAL: This causes data leakage in train/test splits!")
    else:
        print("   [OK] PASS: No reverse pair duplicates")
    
    # Create splits and validate
    print("\n4. ")
    train_df = df[df['split'] == 'train']
    val_df = df[df['split'] == 'val']
    test_df = df[df['split'] == 'test']
    
    print(f"   Train: {len(train_df)} pairs")
    print(f"   Val:   {len(val_df)} pairs")
    print(f"   Test:  {len(test_df)} pairs")
    
    # Check for pair overlap between splits
    print("\n5. Checking for pair overlap between splits...")
    
    train_pairs = set(canonicalize_pair(row['drug_a'], row['drug_b']) for _, row in train_df.iterrows())
    val_pairs = set(canonicalize_pair(row['drug_a'], row['drug_b']) for _, row in val_df.iterrows())
    test_pairs = set(canonicalize_pair(row['drug_a'], row['drug_b']) for _, row in test_df.iterrows())
    
    overlap_train_val = train_pairs & val_pairs
    overlap_train_test = train_pairs & test_pairs
    overlap_val_test = val_pairs & test_pairs
    
    total_overlap = len(overlap_train_val) + len(overlap_train_test) + len(overlap_val_test)
    
    if total_overlap > 0:
        print(f"   [ERROR] FAIL: Found pair overlap between splits!")
        print(f"      Train-Val overlap:   {len(overlap_train_val)} pairs")
        print(f"      Train-Test overlap:  {len(overlap_train_test)} pairs")
        print(f"      Val-Test overlap:    {len(overlap_val_test)} pairs")
        
        if overlap_train_test:
            print("\n   CRITICAL: Train-Test overlap causes severe data leakage!")
            print("   Examples:")
            for pair in list(overlap_train_test)[:5]:
                print(f"      {pair}")
    else:
        print("   [OK] PASS: No pair overlap between splits")
        print("   → Train/val/test are properly disjoint")
    
    # Check drug overlap (expected for pair-disjoint)
    print("\n6. Analyzing drug overlap ...")
    
    train_drugs = set(train_df['drug_a']).union(set(train_df['drug_b']))
    val_drugs = set(val_df['drug_a']).union(set(val_df['drug_b']))
    test_drugs = set(test_df['drug_a']).union(set(test_df['drug_b']))
    
    all_drugs = train_drugs | val_drugs | test_drugs
    shared_train_val = train_drugs & val_drugs
    shared_train_test = train_drugs & test_drugs
    shared_val_test = val_drugs & test_drugs
    shared_all = train_drugs & val_drugs & test_drugs
    
    print(f"   Total unique drugs: {len(all_drugs)}")
    print(f"   Train drugs: {len(train_drugs)}")
    print(f"   Val drugs:   {len(val_drugs)}")
    print(f"   Test drugs:  {len(test_drugs)}")
    print(f"\n   Drug overlap:")
    print(f"      Train-Val:   {len(shared_train_val)} drugs ({len(shared_train_val)/len(all_drugs)*100:.1f}%)")
    print(f"      Train-Test:  {len(shared_train_test)} drugs ({len(shared_train_test)/len(all_drugs)*100:.1f}%)")
    print(f"      Val-Test:    {len(shared_val_test)} drugs ({len(shared_val_test)/len(all_drugs)*100:.1f}%)")
    print(f"      All splits:  {len(shared_all)} drugs ({len(shared_all)/len(all_drugs)*100:.1f}%)")
    

    # Check label distribution
    print("\n7. Checking label distribution across splits...")
    
    train_pos_rate = train_df['label'].mean()
    val_pos_rate = val_df['label'].mean()
    test_pos_rate = test_df['label'].mean()
    
    print(f"   Train positive rate: {train_pos_rate*100:.2f}%")
    print(f"   Val positive rate:   {val_pos_rate*100:.2f}%")
    print(f"   Test positive rate:  {test_pos_rate*100:.2f}%")
    
    # Check if rates are similar (should be due to stratification)
    max_diff = max(abs(train_pos_rate - val_pos_rate), 
                   abs(train_pos_rate - test_pos_rate),
                   abs(val_pos_rate - test_pos_rate))
    
    if max_diff > 0.05:  # More than 5% difference
        print(f"   [WARNING]  WARNING: Label distribution varies by {max_diff*100:.1f}%")
        print("   → May indicate stratification issues")
    else:
        print(f"   [OK] PASS: Label distribution is balanced across splits")
    
    # Summary
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)
    
    checks_passed = 0
    checks_total = 5
    
    if not duplicates:
        print("[OK] No duplicate pairs in dataset")
        checks_passed += 1
    else:
        print("[ERROR] Duplicate pairs found in dataset")
    
    if not reverse_found:
        print("[OK] No reverse pair duplicates")
        checks_passed += 1
    else:
        print("[ERROR] Reverse pair duplicates found")
    
    if total_overlap == 0:
        print("[OK] No pair overlap between splits")
        checks_passed += 1
    else:
        print("[ERROR] Pair overlap between splits")
    
    if max_diff <= 0.05:
        print("[OK] Balanced label distribution")
        checks_passed += 1
    else:
        print("[ERROR] Imbalanced label distribution")
    
    # Drug overlap is expected, not a failure
    print("[INFO]  Drug overlap present")
    
    print(f"\nPassed {checks_passed}/{checks_total} critical checks")
    
    if checks_passed == checks_total:
        print("\n[OK] VALIDATION PASSED")
        print("   Dataset is ready for training with no data leakage concerns")
    elif checks_passed >= checks_total - 1:
        print("\n[WARNING] VALIDATION MOSTLY PASSED")
        print("   Minor issues detected - review warnings above")
    else:
        print("\n[ERROR] VALIDATION FAILED")
        print("   Critical data leakage issues detected!")
        print("   → DO NOT train model until issues are fixed")
        print("   → See warnings above for details")
    
    print("="*80)
    
    return checks_passed == checks_total


if __name__ == "__main__":
    success = validate_dataset()
    exit(0 if success else 1)
