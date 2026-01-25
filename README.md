# BioGuard: Comparative Analysis of Graph vs. Fingerprint Architectures for OOD DDI Prediction

**Status:** Architecture Validation / Negative Result Analysis

---

## Abstract

This repository implements a production-grade inference engine to benchmark **Graph Attention Networks (GATv2)** against tabular baselines (**LightGBM/ECFP4**) on the TWOSIDES dataset. Crucially, this project enforces **Strict Bemis-Murcko Scaffold Splitting** to evaluate Out-of-Distribution (OOD) generalization.

**Key Finding:** Contrary to prevailing literature, structure-aware Graph Neural Networks (**0.58 AUC**) significantly underperformed fingerprint-based Gradient Boosting (**0.72 AUC**) in strict OOD settings. This suggests that for pure small-molecule input, 2D topological embeddings suffer from signal dilution compared to explicit substructure fingerprints when data is sparse.

---
## Reproducibility
* Docker
```
git clone https://github.com/LM1290/BioGuard.git
cd bioguard
docker build -t bioguard_app .
docker run -p 8000:8000 bioguard_app
```

* Run Model/train
```
pip install -r requirements_training.txt
python -m validate_data
python -m bioguard.main train
python -m bioguard.main --mode evaluate
python -m bioguard.train_lgbm 
```
## Benchmarks (Strict Scaffold Split)

Evaluation performed on a held-out test set comprising entirely unseen molecular scaffolds. The test set maintains a realistic 1:3 positive-to-negative ratio to penalize false positives.

| Model | Input | ROC-AUC  | PR-AUC   | Observation |
| :--- | :--- |:---------|:---------| :--- |
| **LightGBM (Baseline)** | ECFP4 (1024-bit) | **0.72** | **0.44** | **Current SOTA.** Explicit substructure hashing retains signal even on novel scaffolds. |
| **BioGuard GATv2** | Graph Topology | 0.639    | 0.37     | **Negative Result.** Attention mechanism struggled to generalize SAR to unseen topologies, likely due to data starvation or over-smoothing. |
| **MLP** | ECFP4 (1024-bit) | 0.58     | 0.32     | **Non-linear baseline.** Fails to capture interactions as effectively as tree-based logic. |

---

## Methodology

### 1. Data Split Strategy
To simulate real-world drug discovery (Novel Entity vs. Known Target), we reject random splitting.
* **Split Method:** Bemis-Murcko Scaffold Split.
* **Constraint:** No overlapping scaffolds between Train, Validation, and Test.
* **Leakage Prevention:** Target encoding or global feature aggregation is strictly calculated on the Train fold only.

### 2. Architecture: GATv2
While underperforming in this specific benchmark, the GATv2 infrastructure is implemented to support future multi-modal integration (Sequence + Graph).
* **Mechanism:** Dynamic Graph Attention (Brody et al., 2021).
* **Node Features:** 41-dim (Atomic Num, Chirality, Charge, Hybridization).
* **Readout:** Global Mean/Max Pooling $\rightarrow$ Concatenation $\rightarrow$ MLP Head.
* **Symmetry:** Interaction embedding enforced via commutative operation:
    $$h_{interaction} = (h_a + h_b) \oplus |h_a - h_b|$$

### 3. Engineering Stack
* **Inference:** `FastAPI` (Async) with `ProcessPoolExecutor` for non-blocking RDKit featurization.
* **Reproducibility:** Dockerized environment (`torch-geometric`, `rdkit`, `lightgbm`).
* **Calibration:** Post-hoc Isotonic Regression applied to logits to align confidence probabilities with empirical risk.

---

## Conclusion & Future Work

The "Structural Ceiling" at 0.72 AUC indicates that 2D molecular representation is insufficient for high-fidelity DDI prediction. The failure of the GATv2 to outperform the baseline suggests that the problem is not topological extraction, but **biological context**.

**Next Steps (Phase 2):**
* **Injecting protein target information** (ESM-2 embeddings) into the GAT readout.
* **Hypothesis:** GATv2 will outperform LightGBM only when the input space includes target-specific context, which fingerprints cannot represent.

---

## Citation

```bibtex
@misc {bioguard_2026,
  author = {Maheswaran, Lalit},
  title = {Benchmarking GNNs under Strict Scaffold Splitting: A Negative Result},
  year = {2026},
  url = {[https://github.com/LM1290/BioGuard](https://github.com/LM1290/BioGuard)}
}
