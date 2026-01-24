# BioGuard-GAT: Graph Attention Networks for Pharmacokinetic Interaction Prediction

**Branch:** feature/mpnn-integration
**Status:** Research Prototype / Experimental

## Executive Summary

This branch implements a Graph Neural Network (GNN) architecture for the BioGuard inference engine. Unlike the production branch which utilizes fixed-length ECFP4 fingerprints, this implementation models molecules as undirected graphs using `torch_geometric`.

The primary objective of this architecture is to address the **Generalization Gap** observed in scaffold-disjoint splits. By explicitly capturing topological dependencies and local chemical environments via message passing, this model aims to improve predictive performance on out-of-distribution (OOD) molecular scaffolds.

## System Architecture

### 1. Graph Featurization
Input SMILES are converted into `torch_geometric.data.Data` objects with the following feature set:

*   **Node Features (Atoms):** One-hot encoding of Atomic Number, Chirality Tag, Hybridization, Aromaticity, and Formal Charge.
*   **Edge Features (Bonds):** One-hot encoding of Bond Type (Single, Double, Triple, Aromatic), Conjugation, and Ring Membership.
*   **Topology:** Adjacency matrices generated via RDKit for tautomer consistency.

### 2. Network Topology (GAT)
*   **Backbone:** Multi-Head Graph Attention Layers (`GATv2Conv`) to compute learned weighting coefficients for neighboring nodes.
*   **Readout:** Global Mean Pooling to aggregate node embeddings into a graph-level representation.
*   **Regularization:** DropEdge (random edge removal during training) and standard Dropout applied to dense layers to mitigate overfitting on small datasets.

### 3. Training Configuration
*   **Optimizer:** `AdamW` with decoupled weight decay.
*   **Validation:** Scaffold-Disjoint splitting (Bemis-Murcko) to enforce structural separation between training and evaluation sets.

## Installation & Usage

This branch requires `torch-geometric` and specific CUDA toolkits if running on GPU.

### Local Development

```bash
# Clone the feature branch
git clone -b feature/mpnn-integration https://github.com/LM1290/BioGuard.git
cd BioGuard

# Install dependencies
pip install -r requirements_training.txt

# Execute training
python -m bioguard.main train --split scaffold
```
## Preliminary Benchmarks

Evaluation performed on the TWOSIDES dataset using Scaffold-Disjoint Splitting.

| Model Architecture | Split Type | ROC-AUC | Analysis |
| :--- | :--- | :--- | :--- |
| **GAT (Graph Attention)** | Scaffold | **0.650** | Demonstrates superior generalization to novel scaffolds compared to fingerprints. |
| **MLP (ECFP4 Baseline)** | Scaffold | 0.570 | Fingerprints degrade significantly when testing on unseen structural classes. |
Technical Note: The GAT architecture exhibits higher variance during training due to data scarcity (~60k pairs). Future development will focus on integrating pre-trained molecular embeddings (e.g., ChemBERTa or Grover) to stabilize weights during fine-tuning.

## Citation & Contact

If you utilize this pipeline or methodology in your research, please cite:

```bibtex
@software{bioguard_gat2026,
  title = {BioGuard-GAT: Topological Deep Learning for DDI},
  author = {Maheswaran, Lalit},
  year = {2026},
  institution = {Georgia Institute of Technology},
  url = {https://github.com/LM1290/BioGuard},
  note = {Experimental Graph Architecture}
}