"""
Training module for BioGuardNet DDI prediction model.

Supports pair-disjoint evaluation strategy with early stopping,
probability calibration, and comprehensive metrics logging.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import os
import json
import random
import joblib
from sklearn.model_selection import train_test_split
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import precision_recall_curve
from joblib import dump
from tqdm import tqdm

from .data_loader import load_twosides_data
from .featurizer import BioFeaturizer
from .model import BioGuardNet

CONFIG = {
    'BATCH_SIZE': 128,
    'LEARNING_RATE': 5e-5,
    'EPOCHS': 40,
    'PATIENCE': 8,
    'NUM_WORKERS': 0,
    'SEED': 42
}

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACT_DIR = os.path.join(BASE_DIR, 'artifacts')


def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_pair_disjoint_split(df, test_size=0.2, val_size=0.1, seed=42):
    """
    Create pair-disjoint split for evaluating NEW COMBINATIONS of KNOWN drugs.
    
    If data_loader has already created splits (with realistic test imbalance),
    respects those boundaries. Otherwise creates balanced splits.
    
    Args:
        df: DataFrame with drug pairs
        test_size: Fraction for test set
        val_size: Fraction for validation set
        seed: Random seed
        
    Returns:
        train_df, val_df, test_df
    """
    # Check for pre-existing split
    if 'split' in df.columns:
        print("Using pre-defined splits from data_loader")
        
        train_df = df[df['split'] == 'train'].drop(columns=['split'], errors='ignore').copy()
        val_df = df[df['split'] == 'val'].drop(columns=['split'], errors='ignore').copy()
        test_df = df[df['split'] == 'test'].drop(columns=['split'], errors='ignore').copy()
        
        print(f"  Train: {len(train_df)} pairs ({train_df['label'].mean():.1%} positive)")
        print(f"  Val:   {len(val_df)} pairs ({val_df['label'].mean():.1%} positive)")
        print(f"  Test:  {len(test_df)} pairs ({test_df['label'].mean():.1%} positive)")
        
        return train_df, val_df, test_df
    
    # Create new split
    print("Creating pair-disjoint split...")
    
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=df['label']
    )
    
    val_fraction = val_size / (1 - test_size)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_fraction,
        random_state=seed,
        stratify=train_val_df['label']
    )
    
    print(f"  Train: {len(train_df)} pairs ({train_df['label'].mean():.1%} positive)")
    print(f"  Val:   {len(val_df)} pairs ({val_df['label'].mean():.1%} positive)")
    print(f"  Test:  {len(test_df)} pairs ({test_df['label'].mean():.1%} positive)")
    
    # Verify no pair overlap
    train_pairs = set(zip(train_df['drug_a'], train_df['drug_b']))
    val_pairs = set(zip(val_df['drug_a'], val_df['drug_b']))
    test_pairs = set(zip(test_df['drug_a'], test_df['drug_b']))
    
    overlap_tv = len(train_pairs & val_pairs)
    overlap_tt = len(train_pairs & test_pairs)
    overlap_vt = len(val_pairs & test_pairs)
    
    if overlap_tv > 0 or overlap_tt > 0 or overlap_vt > 0:
        raise ValueError(f"Pair leakage detected: T-V={overlap_tv}, T-T={overlap_tt}, V-T={overlap_vt}")
    
    return train_df, val_df, test_df


class BioDataset(Dataset):
    """
    PyTorch dataset for DDI prediction.
    Pre-computes drug features for efficiency.
    """
    
    def __init__(self, df):
        self.df = df.reset_index(drop=True)
        self.featurizer = BioFeaturizer()
        
        # Pre-compute all drug features
        print("  Pre-computing drug features...")
        self.drug_cache = {}
        
        all_drugs = pd.concat([
            df[['drug_a', 'smiles_a']].rename(columns={'drug_a': 'id', 'smiles_a': 'smiles'}),
            df[['drug_b', 'smiles_b']].rename(columns={'drug_b': 'id', 'smiles_b': 'smiles'})
        ]).drop_duplicates(subset='id')
        
        for _, row in tqdm(all_drugs.iterrows(), total=len(all_drugs), desc="  Featurizing", leave=False):
            vec = self.featurizer.featurize_single_drug(row['smiles'], row['id'])
            self.drug_cache[row['id']] = vec
        
        print(f"  Cached {len(self.drug_cache)} drug feature vectors")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        vec_a = self.drug_cache.get(row['drug_a'])
        vec_b = self.drug_cache.get(row['drug_b'])
        
        if vec_a is None or vec_b is None:
            return torch.zeros(self.featurizer.total_dim), torch.FloatTensor([row['label']])
        
        # Compute pair features
        pair_vec = np.concatenate([
            vec_a + vec_b,
            np.abs(vec_a - vec_b),
            vec_a * vec_b
        ])
        
        return torch.FloatTensor(pair_vec), torch.FloatTensor([row['label']])


def run_training(split_type='pair_disjoint'):
    """
    Train BioGuardNet model with pair-disjoint evaluation.
    
    Args:
        split_type: Only 'pair_disjoint' is supported in production
    """
    if split_type != 'pair_disjoint':
        raise ValueError(f"Only 'pair_disjoint' split is supported in production. Got: {split_type}")
    
    set_seed(CONFIG['SEED'])
    
    artifact_dir = ARTIFACT_DIR
    os.makedirs(artifact_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")
    print(f"Split type: {split_type.upper()}")

    # Load data
    print("\nLoading data...")
    df = load_twosides_data()
    
    # Create splits
    train_df, val_df, test_df = get_pair_disjoint_split(df, seed=CONFIG['SEED'])
    
    # Create datasets
    print("\nPreparing training data...")
    train_dataset = BioDataset(train_df)
    print("Preparing validation data...")
    val_dataset = BioDataset(val_df)
    
    input_dim = train_dataset.featurizer.total_dim
    print(f"Feature dimension: {input_dim}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['BATCH_SIZE'],
        shuffle=True,
        num_workers=CONFIG['NUM_WORKERS']
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG['BATCH_SIZE'],
        shuffle=False,
        num_workers=CONFIG['NUM_WORKERS']
    )

    # Initialize model
    model = BioGuardNet(input_dim=input_dim).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['LEARNING_RATE'])
    criterion = nn.BCEWithLogitsLoss()
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    
    print(f"\nTraining for {CONFIG['EPOCHS']} epochs (patience={CONFIG['PATIENCE']})")
    print("-" * 60)
    
    for epoch in range(CONFIG['EPOCHS']):
        # Training
        model.train()
        train_loss = 0.0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                logits = model(batch_X)
                loss = criterion(logits, batch_y)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1:3d}/{CONFIG['EPOCHS']} | "
              f"Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f}", end="")
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(artifact_dir, 'model.pt'))
            print(" [BEST]")
        else:
            patience_counter += 1
            print(f" [patience: {patience_counter}/{CONFIG['PATIENCE']}]")
            
            if patience_counter >= CONFIG['PATIENCE']:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
    
    # Load best model for calibration
    print("\nCalibrating probabilities...")
    model.load_state_dict(torch.load(os.path.join(artifact_dir, 'model.pt')))
    model.eval()
    
    # Get validation predictions
    y_true_val = []
    y_logits_val = []
    
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X = batch_X.to(device)
            logits = model(batch_X)
            
            y_logits_val.extend(logits.cpu().numpy().flatten())
            y_true_val.extend(batch_y.numpy().flatten())
    
    y_true_val = np.array(y_true_val)
    y_probs_val = 1 / (1 + np.exp(-np.array(y_logits_val)))
    
    # Fit calibrator
    iso = IsotonicRegression(out_of_bounds='clip', y_min=0, y_max=1)
    iso.fit(y_probs_val, y_true_val)
    dump(iso, os.path.join(artifact_dir, 'calibrator.joblib'))
    
    # Find optimal threshold
    y_calibrated = iso.transform(y_probs_val)
    precision, recall, thresholds = precision_recall_curve(y_true_val, y_calibrated)
    
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
    
    print(f"Optimal threshold: {optimal_threshold:.4f} (F1={f1_scores[optimal_idx]:.4f})")
    
    # Save metadata
    metadata = {
        'version': f'2.0-{split_type}',
        'split_type': split_type,
        'config': CONFIG,
        'threshold': float(optimal_threshold),
        'input_dim': input_dim,
        'best_val_loss': float(best_val_loss),
        'split_info': {
            'train_size': len(train_df),
            'val_size': len(val_df),
            'test_size': len(test_df),
            'train_positive_rate': float(train_df['label'].mean()),
            'val_positive_rate': float(val_df['label'].mean()),
            'test_positive_rate': float(test_df['label'].mean())
        }
    }
    
    with open(os.path.join(artifact_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("\nTraining complete")
    print(f"Artifacts saved to: {artifact_dir}/")
    print(f"  - model.pt")
    print(f"  - calibrator.joblib")
    print(f"  - metadata.json")
    print(f"\nNext: python -m bioguard.main eval")
