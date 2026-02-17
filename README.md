
# BioGuard V2: Breaking the Scaffold Gap in DDI Prediction via Biological Priors

**Status:** V2.2 RELEASED (Hybrid Architecture)

**License:** MIT

**Docker:** `bioguard_app:latest`

## Summary

BioGuard is a production-grade inference engine designed to predict Drug-Drug Interactions (DDIs) on **New Chemical Entities (NCEs)**. Unlike academic baselines that report inflated metrics using random data splits, BioGuard is benchmarked on strict **Bemis-Murcko Scaffold-Disjoint splits** to enforce true Out-of-Distribution (OOD) generalization.

**The Breakthrough:**
Pure structural embeddings (GAT/ECFP4) hit a Structural Ceiling on unseen scaffolds. By engineering a **CYP450 Metabolic Injection Layer** (mapping 280+ active drug hubs to their enzyme substrate/inhibitor profiles via ChEMBL), we recovered significant signal in the OOD regime.

**Key Findings:**
1.  **LightGBM:** Achieves SOTA Precision (**0.43 PR-AUC**) on the Scaffold split via ECFP4 Fingerprints.
2.  **BioGuard GATv2:** Sacrifices precision for **Maximal Recall (0.70)**, catching toxic interactions. Modular Engineering designed to intake Multi-modal input.
3.  We demonstrate that SOTA papers reporting 0.90+ AUC are relying on structural data leakage.

---

## 1. The "SOTA" Integrity Gap

Current literature (e.g., *DeepDDS*, *CASTER*) reports ROC-AUCs > 0.90. However, our reproduction analysis identifies a critical methodological flaw: **Random Splitting.**

| Methodology | Data Split | ROC-AUC | Verdict                                                                                                               |
| :--- | :--- | :--- |:----------------------------------------------------------------------------------------------------------------------|
| **Academic Baseline (DeepDDS)** | Random / 5-Fold | **0.93** | **Structural Leakage.** The model memorizes the graph topology of drugs seen in training. Fails on NCEs.              |
| **BioGuard (Ours)** | **Scaffold-Disjoint** | **0.71** | **Novel Generalization.** The model never sees the test scaffolds during training. Represents realistic risk scoring. |

*Note: A 0.93 AUC on a random split translates to near-zero utility in a Lead Optimization pipeline where the molecule is novel.*

---

## 2. Methodology & Architecture

### A. Data Engineering 
Standard datasets (TWOSIDES) suffer from **Hub Bias** (e.g., Warfarin interacts with everything). To fix this, we built a custom ETL pipeline:
*   **Source:** ChEMBL API / PubChem.
*   **Resolution:** InChIKey-based ID mapping to resolve mismatched datasets.
*   **Enrichment:** Explicit feature injection for certain Metabolic Enzymes (CYP3A4, CYP2D6, CYP2C9, CYP2C19, CYP1A2).
*   **Extrapolation:** Uses metabolic imputation to predict CYP substrate/inhibitor status for NCEs via trained RandomForest Classifier.

### B. Model A: LightGBM 
*   **Input:** 1024-bit ECFP4 Fingerprints + 30-dim CYP Vector.
*   **Optimization:** Gradient-based One-Side Sampling (GOSS).
*   **Role:** High-fidelity discriminator for filtering Easy Negatives.

### C. Model B: BioGuard GATv2
*   **Input:** Molecular Graph + CYP Node Features.
*   **Pre-training:** In-Domain Self-Supervised Learning (Masked Atom Prediction) to initialize weights with chemical valency intuition (Accuracy: 81%).
*   **Role:** High-sensitivity filter. The GAT propagates metabolic risk through the graph.

---

## 3. Benchmarks (Scaffold-Disjoint Test Set)

Evaluation performed on a held-out test set of **entirely unseen molecular scaffolds**.

| Model | Architecture | ROC-AUC  | PR-AUC   | Recall (Sensitivity) | Role |
| :--- | :--- |:---------|:---------| :--- | :--- |
| **LightGBM (Hybrid)** | Gradient Boosting | **0.71** | **0.43** | 0.66 | **Best Baseline.** Precision-optimized. |
| **BioGuard GATv2** | Pre-trained GNN | 0.67     | 0.40     | **0.70** | **Safety Filter.** Catching 70% of toxic events in OOD space. |
| *Naive GAT* | Graph Neural Net | 0.64     | 0.37     | 0.27 | *Failed Control (No Biological Context).* |

**Analysis:**
*   The **LightGBM Hybrid** provides a **50% Signal Lift** over random screening (Baseline 0.28).
*   The **BioGuard GAT** catches ~4% more lethal interactions than the tree-based model. In a production setting, we recommend an ensemble approach: flag if *either* model predicts toxicity.

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

### Data Pipeline & Enzyme ETL
To reproduce the ChEMBL scraping and feature engineering:

```bash
# Downloads TWOSIDES, performs Scaffold Split, and maps InChIKeys
python -m validate_data

# Train model to generate required parquet file.
python -m bioguard.main train

# (Optional) Re-run the ChEMBL scraper (Cached file included in /data)
python tools/map_human_cyps.py
python tools/fetch_full_enzyme_profile.py
python bioguard.cyp_predictor
#Unit Test for NCEs
python tools.patch_smiles

python bioguard.NCEcypTest
```

### Inference & Training
Reproduce the benchmarks reported above:

```bash
# 1. Train the BioGuard GATv2 (Includes SSL Pre-training)
python -m bioguard.main train

# 2. Evaluate GAT on Test Set
python -m bioguard.main eval

# 3. Train and Benchmark LightGBM Hybrid
python -m bioguard.train_lgbm
```

---

## 5. Future Work & Roadmap

*   **Latent Space Disjointness (UMAP):** While Murcko scaffolds are the industry standard, recent literature suggests UMAP-based splitting prevents latent space leakage.
*   **Transporter Features:** Integration of P-gp and OATP transporter data to address the current false-negative rate on non-metabolic interactions.

---
## 6. Technical Debt

* **GPU Utilization:** Currently, torch.profiler shows GPU underutilization. Moving to LMDB for data retrieval to maximize training velocity.
* **Memory Management:** Currently loads all graphs into RAM, which is incompatible with large-scale datasets. Engineering move to LMDB.

---
## 7. Production Roadmap
* **Status:** BioGuard v2.2 utilizes a ChEMBL-anchored lookup for metabolic enzyme features (CYP450 profiles).
* **Generalization:** While this proved that injection of biological priors recovers signal for OOD molecules, it struggles with Novel Chemical Entities for which enzymatic/metabolic activity is presently unknown.
* **Fix:** Architecting end-to-end pipeline that intakes SMILES, predicts enzyme activity from graph topology, and establishes baseline for metabolic activity prior to risk prediction. Alpha prototype available in bioguard/cyp_predictor.