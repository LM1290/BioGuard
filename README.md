# BioGuard DDI Prediction System (Production Version)

Drug-Drug Interaction (DDI) prediction using deep learning with structural molecular features.
Interact with the model at https://lmbioguard.streamlit.app/ !

## Features

- **Neural Network Model**: BioGuardNet with calibrated probability predictions
- **Baseline Comparisons**: Tanimoto similarity, Logistic Regression, Random Forest
- **Streamlit Interface**: User-friendly web app for predictions

## Quick Start

### Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

### Run Streamlit App

```bash
cd streamlit
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`


### Streamlit Web Interface (Recommended)

```bash
cd streamlit
streamlit run app.py
```

Enter two SMILES strings and get predictions from:
- Neural Network (BioGuardNet)
- Tanimoto Similarity baseline
- Logistic Regression baseline

### Command Line Interface

```bash
# Train model
python -m bioguard.main train

# Evaluate model
python -m bioguard.main eval

# Run baselines
python -m bioguard.main baselines

# Compare all methods
python -m bioguard.main compare

```

## Model Details

### BioGuardNet Architecture

- Input: Molecular fingerprints + biophysical properties
- Architecture: 512 → 256 → 1 with BatchNorm and Dropout
- Training: AdamW optimizer with early stopping
- Calibration: Isotonic regression for probability calibration

### Features Used

**Per Drug:**
- Morgan fingerprints (2048-bit, radius 2)
- Biophysical properties (5 features):
  - Molecular weight
  - LogP (lipophilicity)
  - TPSA (topological polar surface area)
  - H-bond donors
  - H-bond acceptors

**Pair Features:**
- Sum, difference, and product of individual drug features

### Evaluation Split

- **Pair-Disjoint**: Tests on new drug combinations
- Training: 70% | Validation: 10% | Test: 20%
- Class imbalance is handled by using PR-AUC as the primary metric

## Research Features (Not in Production)

The `researchdev/` directory contains experimental features:

1. **Enzyme Features**: CYP450 metabolism and transporter interactions
2. **Drug-Disjoint Split**: Testing on completely new drugs

These are excluded from production to minimize dependencies and complexity.

## Performance

Results on TWOSIDES dataset (pair-disjoint split):

| Method                  | ROC-AUC | PR-AUC |
|-------------------------|---------|--------|
| Neural Network          | 0.947   | 0.819  |
| Logistic Regression     | 0.940   | 0.816  |
| Tanimoto Similarity     | 0.530   | 0.169  |
| Random Forest           | 0.874   | 0.588  |

*Note: Actual performance depends on training and data version*

## Dependencies

- Python 3.10 or 3.11
- PyTorch 2.8.0
- RDKit 2023.9.6
- NumPy 1.26.4 (compatible with RDKit)
- scikit-learn 1.3.2
- Streamlit 1.52.2

See `requirements.txt` for complete list.

## Troubleshooting

### RDKit/NumPy Compatibility

If you encounter version conflicts:
```bash
pip install --force-reinstall numpy==1.26.4 rdkit==2023.9.6
```

### Model Not Found

Ensure artifacts are present:
```bash
ls artifacts/pair_disjoint/
# Should show: model.pt, calibrator.joblib, metadata.json
```

If missing, train the model:
```bash
python -m bioguard.main train
```

## Citation

If you use this code, please cite:

```bibtex
@software{bioguard2025,
  title={BioGuard: Drug-Drug Interaction Prediction},
  year={2025},
  author={Maheswaran, Lalit},
  note={Structure-based DDI prediction using deep learning}
}
```

## License

MIT License - see LICENSE file for details

## Contact

For questions or issues, please open a GitHub issue.
