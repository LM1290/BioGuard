# BioGuard: Symmetric Deep Learning for Drug-Drug Interaction Risk Scoring

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://bioguardlm.streamlit.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

**Live Inference Engine:** [https://bioguardlm.streamlit.app/](https://bioguardlm.streamlit.app/)

## Executive Summary

BioGuard is a deep learning inference engine designed to predict non-linear drug-drug interactions (DDIs) by overcoming the limitations of traditional structural similarity metrics. Unlike standard baselines, BioGuard utilizes a Symmetric MLP architecture on ECFP4 fingerprints to capture latent interference mechanisms while strictly enforcing permutational invariance.

### Key Capabilities
*   **Permutational Invariance:** Implemented a symmetric pair-encoding strategy `(A+B) ⊕ |A-B| ⊕ (A*B)` to ensure `Interaction(Drug A, Drug B) == Interaction(Drug B, Drug A)`, eliminating directional bias.
*   **Zero Data Leakage:** Utilizes Pair-Disjoint Splitting to ensure drug pairs present in the test set are strictly unseen during training.
*   **High-Sensitivity Screening:** Optimized for Recall (0.89) to function as a safety filter in early-stage discovery pipelines, prioritizing the detection of potential adverse events.

---

## Performance Benchmarks

*Evaluation performed on the TWOSIDES dataset using a strict pair-disjoint split.*

| Model | ROC-AUC | PR-AUC | Recall (Sensitivity) |
| :--- | :--- | :--- | :--- |
| **BioGuard (Neural Net)** | **0.947** | **0.819** | **0.89** |
| Logistic Regression | 0.940 | 0.816 | 0.72 |
| Random Forest | 0.874 | 0.588 | 0.64 |
| Tanimoto Similarity | 0.530 | 0.169 | 0.92 |

*Note: BioGuard outperforms the non-linear Random Forest baseline by >7% in ROC-AUC and demonstrates superior recall compared to linear baselines.*
Open-World Negative Sampling: Addresses the positive-unlabeled nature of DDI datasets by treating unrecorded pairs as non-interactions, with plans to implement hard-negative mining in v2.0 to reduce false positives.

---

## System Architecture

### 1. Featurization Pipeline
*   **Structural:** 2048-bit Morgan Fingerprints (Radius 2) generated via RDKit.
*   **Biophysical:** 5-dimensional scalar vector (Molecular Weight, LogP, TPSA, H-Donors, H-Acceptors).
*   **Preprocessing:** Automated SMILES canonicalization and chiral tag handling.

### 2. Network Topology
*   **Input Layer:** 6192-dim symmetric vector.
*   **Hidden Layers:** Dense(512) → BatchNorm → ReLU → Dropout(0.3) → Dense(256) → ReLU.
*   **Output:** Sigmoid activation with Isotonic Regression calibration for probability scoring.

Note: A symmetric MLP was chosen over Graph Neural Networks (GNNs) for v1.0 to establish a high-speed inference baseline and prioritize interpretability via feature importance analysis, before migrating to computationally heavier graph topologies.

### 3. Deployment Stack
*   **Inference:** PyTorch / FastAPI backend.
*   **Frontend:** Streamlit for real-time computational chemistry visualization.

---

## V2 Roadmap & Technical Trajectory

*Current development focused on the `researchdev/` branch to address public dataset limitations.*

1.  **Hard Negative Mining:** Integrating Tanimoto-thresholded negative sampling to correct for the "soft-negative" bias inherent in random sampling.
2.  **Enzyme Profiling:** Integration of CYP450 metabolism vectors to move beyond pure structural features and model metabolic interference directly.
3.  **Explainability:** Implementation of Integrated Gradients to map prediction weights back to specific substructural motifs.
4.  Current validation utilizes a Random Cold-Start split (Drug Disjoint). Future iterations will implement Butina Clustering to ensure strict Scaffold Disjointness.

---

## Quick Start & Installation

For developers wishing to reproduce the training loop or run the CLI.

### Environment Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies (Order matters for PyTDC)
pip install -r requirements_training.txt
# Validate Data Pipeline
python -m validate_data

# Train model from scratch
python -m bioguard.main train

# Run baseline comparisons
python -m bioguard.main baselines

```
## Citation & Contact

If you utilize this pipeline or methodology in your research, please cite:

```bibtex
@software{bioguard2025,
  title = {BioGuard: Symmetric Deep Learning for Pharmacokinetic Interaction Prediction},
  author = {Maheswaran, Lalit},
  year = {2026},
  institution = {Georgia Institute of Technology},
  url = {https://github.com/LM1290/BioGuard},
  note = {Deployed Inference Engine for DDI Screening}
}
