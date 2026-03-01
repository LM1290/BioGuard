# BioGuard V3: Adaptive Graph-Metabolic DDI Prediction

**Status:** V3.0 (Strict Mode / LMDB-Backed)

**License:** MIT

**Docker:** `bioguard_app:latest`

## Summary

BioGuard is a production-grade inference engine designed to predict Drug-Drug Interactions (DDIs) on **New Chemical Entities (NCEs)**. Unlike academic baselines that report inflated metrics using random data splits, BioGuard is benchmarked on strict **Bemis-Murcko Scaffold-Disjoint splits** to enforce true Out-of-Distribution (OOD) generalization.

Previous versions relied on static concatenation of biological features. BioGuard V3 introduces an **Adaptive Alpha Gate ($\alpha$)**, a learnable architectural component that dynamically weights the contribution of the **Graph Neural Network (GAT)** against the **Metabolic Prior**. This allows the model to autonomously balance 3D structural physics against metabolic heuristics based on signal confidence.

**Key Findings:**
1.  **Adaptive Telemetry:** The model converged to a **Mean Alpha of 0.81**, indicating a learned preference for Structural Physics (80%) over Metabolic Priors (20%) in the OOD regime.
2.  **BioGuard GATv3:** Sacrifices precision for **High Recall (0.81)**, effectively functioning as a safety net that catches 81% of toxic interactions in strictly novel compounds.
3.  **End-to-End NCE Support:** We replaced the static ChEMBL injection layer with an embedded **LightGBM Ensemble**, allowing immediate metabolic prediction for drugs not present in public databases.

---

## 1. The "SOTA" Integrity Gap

Current literature (e.g., *DeepDDS*, *CASTER*) reports ROC-AUCs > 0.90. However, our reproduction analysis identifies a critical methodological flaw: **Random Splitting.**

| Methodology | Data Split | ROC-AUC | Verdict                                                                                                               |
| :--- | :--- | :--- |:----------------------------------------------------------------------------------------------------------------------|
| **Academic Baseline (DeepDDS)** | Random / 5-Fold | **0.93** | **Structural Leakage.** The model memorizes the graph topology of drugs seen in training. Fails on NCEs.              |
| **BioGuard (Ours)** | **Scaffold-Disjoint** | **0.73** | **Novel Generalization.** The model never sees the test scaffolds during training. Represents realistic risk scoring. |

*Note: A 0.93 AUC on a random split translates to near-zero utility in a Lead Optimization pipeline where the molecule is novel.*

---

## 2. Methodology & Architecture

### A. Data Engineering (LMDB)
To resolve memory bottlenecks and ensure data integrity:
*   **Storage:** Switched from CSV/Parquet to **LMDB (Lightning Memory-Mapped Database)** for zero-copy tensor loading.
*   **Featurization:** 3D Conformer generation via ETKDGv3, extracting **Gasteiger Charges** and **SASA (Solvent Accessible Surface Area)**.
*   **Validation:** Strict multiprocessing filter that purges molecules failing physical embedding generation.

### B. The Metabolic Prior (CYP-Ensemble)
*   **Input:** 2048-bit Morgan Fingerprints.
*   **Mechanism:** An internal ensemble of 15 LightGBM Classifiers trained to predict activity for key enzymes (CYP3A4, CYP2D6, P-gp).
*   **Role:** Provides a dense 15-dimensional metabolic vector for *every* input, ensuring NCEs have a biological baseline before graph convolution.

### C. Model B: BioGuard GATv3 (Adaptive)
*   **Input:** Molecular Graph + Predicted CYP Vector.
*   **Gate Mechanism:** $\text{Logits} = \sigma(\text{Gate}) \cdot f_{GAT} + (1 - \sigma(\text{Gate})) \cdot f_{Prior}$.
*   **Loss Function:** **BioFocalLoss** ($\gamma=2.0, \alpha=0.70$) implemented to handle extreme class imbalance (3:10) and focus learning on hard negatives.

---

## 3. Benchmarks (Scaffold-Disjoint Test Set)

Evaluation performed on a held-out test set of **entirely unseen molecular scaffolds**.

| Model | Architecture | ROC-AUC  | PR-AUC   | Recall (Sensitivity) | Role                                                                                    |
| :--- | :--- |:---------|:---------| :--- |:----------------------------------------------------------------------------------------|
| **BioGuard V3** | Adaptive GAT + Prior | **0.73** | **0.45** | **0.81** | **High-Sensitivity Safety Net.**                                                        |
|**LightGBM**|Gradient-Boosted Decision Tree|0.71|0.44|0.68|*Represents Limit of ECFP4 for generalizing to NCEs.*
| **BioGuard V2** | Static Concatenation | 0.71     | 0.43     | 0.70 | *No Ensemble Architecture, Unable to infer metabolic liability for NCEs not in ChEMBL.* |
| *Naive GAT* | Graph Neural Net | 0.64     | 0.37     | 0.27 | *Failed Control (No Biological Context).*                                               |

**Analysis:**
*   **Recall Improvement:** V3 improves recall by **13%** over LightGBM Baseline (0.68 -> 0.81) via Focal Loss Implementation.
*   **Specificty Trade-off:** Precision remains moderate (0.42), yielding ~1.3 false positives per true positive. This is an intentional design choice to prioritize safety (Recall) over specificity.

---

## 4. Reproducibility

### Setup
BioGuard is fully dockerized for reproducibility.

```bash
# 1. Clone
git clone https://github.com/LM1290/BioGuard.git
cd BioGuard

# 2. Build Environment
docker build -t bioguard_app .
```

### Data Pipeline & LMDB Generation
To reproduce the 3D physics generation and LMDB caching:

```bash
# 1. Download and Clean TWOSIDES (Scaffold Split)
python -c "from bioguard.data_loader import load_twosides_data; load_twosides_data(split_method='cold_drug')"

# 2. Patch Smiles to the enzyme dataset so LightGBM can train
python -m tools.patch_smiles

# 3. Train the internal CYP Predictor (LightGBM Ensemble)
python -m bioguard.train_cyp_predictor

# 4. Build LMDB Cache (High CPU Usage - Multiprocessed 3D Embedding)
python -m bioguard.build_lmdb
```

### Inference & Training
Reproduce the benchmarks reported above:

```bash
# 1. Train BioGuard GATv3 (Adaptive Mode)
python -m bioguard.main train --split cold_drug --epochs 50

# 2. Evaluate on Holdout
python -m bioguard.main eval

# 3. Run Production API
python -m bioguard.main serve
```

---

## 5. Model Audit & Interpretability

We include an audit suite (`model_audit.py`) to verify biological grounding and prevent bit-collision artifacts.

*   **Consensus Decoding:** Maps high-activation bits in the Morgan Fingerprint back to visual pharmacophores to ensure the CYP Predictor is learning chemical motifs.
*   **Specificity-Recall Curves:** Generated on strict holdout sets to verify the clinical utility range.

---

## 6. Resolved Technical Debt

*   **GPU Utilization:** Previous versions suffered from IO bottlenecks. The implementation of `BioGuardDataset` backed by LMDB now allows for full GPU saturation during training.
*   **Memory Management:** Dataset is now memory-mapped; RAM usage is decoupled from dataset size, enabling scaling to millions of pairs.
*   **Strict Dimension Guard:** The API now enforces strict vector dimension checks between the Enzyme Manager and the GNN to prevent silent tensor broadcasting errors.

---

## 7. Production Features (Implemented)

*   **NCE Generalization:** The `EnzymeManager` now defaults to the trained `CYPPredictor` when a standard InChIKey lookup fails, enabling support for novel molecules.
*   **Class Imbalance:** Implemented `BioFocalLoss` to aggressively penalize easy negatives, stabilizing training on the 10:1 negative-to-positive ratio in the cleaning pipeline.
*   **API Telemetry:** The `/predict` endpoint now returns `alpha_telemetry` and `risk_level` alongside raw probabilities to provide transparency into the model's decision-making logic (Structure vs. Prior).