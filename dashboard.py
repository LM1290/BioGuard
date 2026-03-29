"""
BioGuard V3 -- Drug-Drug Interaction Dashboard
Streamlit interface for DDI prediction, model analytics, and enzyme profiling.
"""

import streamlit as st
import numpy as np
import pandas as pd
import torch
import joblib
import json
import os
import signal
import plotly.graph_objects as go
import plotly.express as px
from rdkit import Chem
from rdkit.Chem import Draw, AllChem, Descriptors
from torch_geometric.data import Batch

from bioguard.config import BASE_DIR, ARTIFACT_DIR, DATA_DIR, NODE_DIM, EDGE_DIM
from bioguard.model import BioGuardGAT
from bioguard.featurizer import GraphFeaturizer
from bioguard.enzyme import EnzymeManager

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MODEL_PATH = os.path.join(ARTIFACT_DIR, "model.pt")
META_PATH = os.path.join(ARTIFACT_DIR, "metadata.json")
CALIBRATOR_PATH = os.path.join(ARTIFACT_DIR, "calibrator.joblib")
CALIBRATION_DATA_PATH = os.path.join(ARTIFACT_DIR, "calibration_data.npz")
ENZYME_CSV = os.path.join(DATA_DIR, "enzyme_features_full.csv")

# ---------------------------------------------------------------------------
# Limits for community cloud
# ---------------------------------------------------------------------------
MAX_HEAVY_ATOMS = 80
CONFORMER_TIMEOUT_SEC = 45

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="BioGuard DDI Dashboard",
    page_icon="B",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    .main .block-container { padding-top: 1.5rem; }
    .metric-card {
        background: linear-gradient(135deg, #1e1e2f 0%, #2d2d44 100%);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        border: 1px solid #3d3d5c;
    }
    .metric-card h3 { color: #a0a0c0; font-size: 0.85rem; margin-bottom: 0.3rem; }
    .metric-card p { color: #ffffff; font-size: 1.8rem; font-weight: 700; margin: 0; }
    .risk-high { color: #ff4b4b; }
    .risk-low { color: #00cc96; }
    div[data-testid="stMetric"] { background: #1e1e2f; border-radius: 10px; padding: 12px; border: 1px solid #3d3d5c; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Molecule validation
# ---------------------------------------------------------------------------
def validate_smiles(smiles):
    if not smiles or not smiles.strip():
        return False
    return Chem.MolFromSmiles(smiles.strip()) is not None


def check_molecule_complexity(smiles):
    """Returns (ok, message). Rejects molecules that will choke on free compute."""
    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None:
        return False, "Invalid SMILES string."
    n_heavy = mol.GetNumHeavyAtoms()
    if n_heavy > MAX_HEAVY_ATOMS:
        return False, (
            f"Molecule has {n_heavy} heavy atoms (limit: {MAX_HEAVY_ATOMS}). "
            "3D conformer generation would exceed available compute. "
            "Try a smaller molecule."
        )
    return True, ""


def smiles_to_image(smiles, size=(280, 200)):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Draw.MolToImage(mol, size=size)


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model():
    meta = {}
    if os.path.exists(META_PATH):
        with open(META_PATH) as f:
            meta = json.load(f)

    enzyme_manager = EnzymeManager(allow_degraded=True)

    node_dim = meta.get("node_dim", NODE_DIM)
    embedding_dim = meta.get("embedding_dim", 128)
    heads = meta.get("heads", 4)
    enzyme_dim = meta.get("enzyme_dim", enzyme_manager.vector_dim)
    threshold = meta.get("threshold", 0.5)

    device = torch.device("cpu")
    model = BioGuardGAT(
        node_dim=node_dim,
        embedding_dim=embedding_dim,
        heads=heads,
        enzyme_dim=enzyme_dim,
    ).to(device)

    model_ready = False
    if os.path.exists(MODEL_PATH):
        try:
            model.load_state_dict(
                torch.load(MODEL_PATH, map_location=device, weights_only=True)
            )
            model.eval()
            model_ready = True
        except Exception as e:
            st.error(f"Model load failed: {e}")

    calibrator = None
    if os.path.exists(CALIBRATOR_PATH):
        calibrator = joblib.load(CALIBRATOR_PATH)

    return model, model_ready, enzyme_manager, calibrator, threshold, enzyme_dim, meta


@st.cache_resource
def load_featurizer():
    return GraphFeaturizer()


@st.cache_data
def load_enzyme_df():
    if os.path.exists(ENZYME_CSV):
        return pd.read_csv(ENZYME_CSV)
    return None


@st.cache_data
def load_calibration_data():
    if os.path.exists(CALIBRATION_DATA_PATH):
        return dict(np.load(CALIBRATION_DATA_PATH, allow_pickle=True))
    return None


@st.cache_data(show_spinner=False)
def featurize_molecule(_featurizer, smiles):
    """Cache 3D graph generation per SMILES so repeat predictions skip ETKDGv3."""
    return _featurizer.smiles_to_graph(smiles.strip())


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("BioGuard V3")
st.sidebar.caption("Drug-Drug Interaction Prediction")

page = st.sidebar.radio(
    "Navigation",
    ["Predict DDI", "Model Performance", "Enzyme Explorer", "About"],
    index=0,
)

model, model_ready, enzyme_manager, calibrator, threshold, enzyme_dim, meta = load_model()
featurizer = load_featurizer()

st.sidebar.divider()
st.sidebar.markdown(f"**Model status:** {'Ready' if model_ready else 'Not loaded'}")
st.sidebar.markdown(f"**Device:** CPU")
if meta:
    st.sidebar.markdown(f"**Split:** {meta.get('split_type', 'N/A')}")
    st.sidebar.markdown(f"**Threshold:** {meta.get('threshold', 0.5):.2f}")


# ===================================================================
# PAGE: Predict DDI
# ===================================================================
if page == "Predict DDI":
    st.title("Drug-Drug Interaction Prediction")
    st.markdown("Enter two drug SMILES strings to predict potential interactions.")

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Drug A")
        smiles_a = st.text_input(
            "SMILES",
            value="CC(=O)Oc1ccccc1C(=O)O",
            key="smiles_a",
            help="Aspirin: CC(=O)Oc1ccccc1C(=O)O",
        )
        if smiles_a and validate_smiles(smiles_a):
            img_a = smiles_to_image(smiles_a)
            if img_a:
                st.image(img_a, caption="Drug A Structure")
            mol_a = Chem.MolFromSmiles(smiles_a)
            st.caption(
                f"MW: {Descriptors.MolWt(mol_a):.1f} | "
                f"LogP: {Descriptors.MolLogP(mol_a):.2f} | "
                f"HBD: {Descriptors.NumHDonors(mol_a)} | "
                f"HBA: {Descriptors.NumHAcceptors(mol_a)} | "
                f"Atoms: {mol_a.GetNumHeavyAtoms()}"
            )
        elif smiles_a:
            st.error("Invalid SMILES")

    with col_b:
        st.subheader("Drug B")
        smiles_b = st.text_input(
            "SMILES",
            value="CC12CCC3C(CCC4=CC(=O)CCC43C)C1CCC2O",
            key="smiles_b",
            help="Testosterone: CC12CCC3C(CCC4=CC(=O)CCC43C)C1CCC2O",
        )
        if smiles_b and validate_smiles(smiles_b):
            img_b = smiles_to_image(smiles_b)
            if img_b:
                st.image(img_b, caption="Drug B Structure")
            mol_b = Chem.MolFromSmiles(smiles_b)
            st.caption(
                f"MW: {Descriptors.MolWt(mol_b):.1f} | "
                f"LogP: {Descriptors.MolLogP(mol_b):.2f} | "
                f"HBD: {Descriptors.NumHDonors(mol_b)} | "
                f"HBA: {Descriptors.NumHAcceptors(mol_b)} | "
                f"Atoms: {mol_b.GetNumHeavyAtoms()}"
            )
        elif smiles_b:
            st.error("Invalid SMILES")

    st.divider()

    if st.button("Predict Interaction", type="primary", use_container_width=True):
        if not model_ready:
            st.error("Model not loaded. Ensure model.pt exists in artifacts/.")
        elif not validate_smiles(smiles_a) or not validate_smiles(smiles_b):
            st.error("Both SMILES must be valid.")
        else:
            # Pre-flight complexity check
            ok_a, msg_a = check_molecule_complexity(smiles_a)
            ok_b, msg_b = check_molecule_complexity(smiles_b)
            if not ok_a:
                st.error(f"Drug A: {msg_a}")
            elif not ok_b:
                st.error(f"Drug B: {msg_b}")
            else:
                with st.spinner("Generating 3D conformers and running inference..."):
                    try:
                        # Featurize (cached per SMILES, with timeout)
                        data_a = featurize_molecule(featurizer, smiles_a)
                        data_b = featurize_molecule(featurizer, smiles_b)

                        # Canonicalize for enzyme lookup
                        can_a = Chem.MolToSmiles(
                            Chem.MolFromSmiles(smiles_a.strip()), isomericSmiles=False
                        )
                        can_b = Chem.MolToSmiles(
                            Chem.MolFromSmiles(smiles_b.strip()), isomericSmiles=False
                        )

                        # Enzyme vectors
                        vec_a = enzyme_manager.get_by_smiles(can_a)
                        vec_b = enzyme_manager.get_by_smiles(can_b)

                        # Pad/truncate to expected enzyme_dim
                        def fit_enzyme_vec(vec, dim):
                            if len(vec) == dim:
                                return vec
                            out = np.zeros(dim, dtype=np.float32)
                            n = min(len(vec), dim)
                            out[:n] = vec[:n]
                            return out

                        vec_a = fit_enzyme_vec(vec_a, enzyme_dim)
                        vec_b = fit_enzyme_vec(vec_b, enzyme_dim)

                        has_enzymes = np.any(vec_a) or np.any(vec_b)

                        # Attach enzyme vectors
                        data_a.enzyme_a = torch.tensor(
                            vec_a, dtype=torch.float
                        ).unsqueeze(0)
                        data_b.enzyme_b = torch.tensor(
                            vec_b, dtype=torch.float
                        ).unsqueeze(0)

                        batch_a = Batch.from_data_list([data_a])
                        batch_b = Batch.from_data_list([data_b])

                        # Inference
                        with torch.no_grad():
                            logits, alpha = model(batch_a, batch_b)
                            raw_prob = torch.sigmoid(logits).item()
                            alpha_val = alpha.item()

                        # Calibration
                        calib_prob = raw_prob
                        if calibrator:
                            calib_prob = float(
                                calibrator.transform([raw_prob])[0]
                            )
                            calib_prob = max(0.0, min(1.0, calib_prob))

                        risk = "High" if calib_prob >= threshold else "Low"

                        # Display results
                        st.divider()
                        st.subheader("Prediction Results")

                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("Risk Level", risk)
                        c2.metric("Calibrated Probability", f"{calib_prob:.3f}")
                        c3.metric("Raw Score", f"{raw_prob:.3f}")
                        c4.metric("Alpha (Graph Weight)", f"{alpha_val:.3f}")

                        # Risk gauge
                        fig_gauge = go.Figure(
                            go.Indicator(
                                mode="gauge+number",
                                value=calib_prob * 100,
                                title={"text": "Interaction Risk (%)"},
                                gauge={
                                    "axis": {"range": [0, 100]},
                                    "bar": {
                                        "color": "#ff4b4b"
                                        if risk == "High"
                                        else "#00cc96"
                                    },
                                    "steps": [
                                        {"range": [0, 30], "color": "#e8f5e9"},
                                        {"range": [30, 50], "color": "#fff9c4"},
                                        {"range": [50, 70], "color": "#ffe0b2"},
                                        {"range": [70, 100], "color": "#ffcdd2"},
                                    ],
                                    "threshold": {
                                        "line": {"color": "black", "width": 3},
                                        "thickness": 0.8,
                                        "value": threshold * 100,
                                    },
                                },
                            )
                        )
                        fig_gauge.update_layout(
                            height=300, margin=dict(t=60, b=20)
                        )
                        st.plotly_chart(fig_gauge, use_container_width=True)

                        # Telemetry
                        with st.expander("Telemetry Details"):
                            tel_c1, tel_c2 = st.columns(2)
                            with tel_c1:
                                st.markdown(
                                    f"**Enzyme Status:** "
                                    f"{'Active' if has_enzymes else 'Silent (No Features)'}"
                                )
                                st.markdown(f"**Alpha Gate:** {alpha_val:.4f}")
                                st.caption(
                                    "Alpha = 1.0 means pure graph pathway; "
                                    "0.0 means pure metabolic prior"
                                )

                                fig_alpha = go.Figure(
                                    go.Bar(
                                        x=["Graph (GAT)", "Prior (Metabolic)"],
                                        y=[alpha_val, 1 - alpha_val],
                                        marker_color=["#636efa", "#ef553b"],
                                    )
                                )
                                fig_alpha.update_layout(
                                    title="Pathway Contribution",
                                    yaxis_title="Weight",
                                    height=280,
                                    margin=dict(t=40, b=20),
                                )
                                st.plotly_chart(
                                    fig_alpha, use_container_width=True
                                )

                            with tel_c2:
                                feature_names = getattr(
                                    enzyme_manager, "feature_names", None
                                )
                                if feature_names and has_enzymes:
                                    n_show = min(
                                        len(feature_names), len(vec_a)
                                    )
                                    enz_df = pd.DataFrame(
                                        {
                                            "Feature": feature_names[:n_show],
                                            "Drug A": vec_a[:n_show],
                                            "Drug B": vec_b[:n_show],
                                        }
                                    )
                                    active = enz_df[
                                        (enz_df["Drug A"] > 0)
                                        | (enz_df["Drug B"] > 0)
                                    ]
                                    if not active.empty:
                                        st.markdown(
                                            "**Active Metabolic Features**"
                                        )
                                        st.dataframe(
                                            active.reset_index(drop=True),
                                            use_container_width=True,
                                            height=250,
                                        )
                                    else:
                                        st.info(
                                            "No active metabolic features "
                                            "detected."
                                        )
                                else:
                                    st.info(
                                        "Enzyme profile not available for "
                                        "these compounds."
                                    )


                    except Exception as e:
                        st.error(f"Prediction failed: {e}")
                        with st.expander("Error details"):
                            st.exception(e)

# ===================================================================
# PAGE: Model Performance
# ===================================================================
elif page == "Model Performance":
    st.title("Model Performance")

    eval_path = os.path.join(ARTIFACT_DIR, "eval_results.json")
    eval_data = None
    if os.path.exists(eval_path):
        with open(eval_path) as f:
            eval_data = json.load(f)

    if meta:
        st.subheader("Training Configuration")
        config_col1, config_col2, config_col3 = st.columns(3)
        with config_col1:
            st.metric("Node Dim", meta.get("node_dim", NODE_DIM))
            st.metric("Edge Dim", meta.get("edge_dim", EDGE_DIM))
        with config_col2:
            st.metric("Embedding Dim", meta.get("embedding_dim", 128))
            st.metric("Attention Heads", meta.get("heads", 4))
        with config_col3:
            st.metric("Enzyme Dim", meta.get("enzyme_dim", "N/A"))
            st.metric("Split Type", meta.get("split_type", "N/A"))

    if eval_data:
        st.divider()
        st.subheader("Evaluation Metrics (Holdout Test Set)")

        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("ROC-AUC", f"{eval_data.get('roc_auc', 0):.4f}")
        m2.metric("PR-AUC", f"{eval_data.get('pr_auc', 0):.4f}")
        m3.metric("Recall", f"{eval_data.get('recall', 0):.4f}")
        m4.metric("Precision", f"{eval_data.get('precision', 0):.4f}")
        m5.metric("F1 Score", f"{eval_data.get('f1', 0):.4f}")
        m6.metric("Accuracy", f"{eval_data.get('accuracy', 0):.4f}")

    # Calibration plot
    cal_data = load_calibration_data()
    if cal_data is not None:
        st.divider()
        st.subheader("Calibration Analysis")

        cal_col1, cal_col2 = st.columns(2)

        with cal_col1:
            if "y_prob" in cal_data and "y_true" in cal_data:
                y_prob = cal_data["y_prob"]
                y_true = cal_data["y_true"]

                from sklearn.calibration import calibration_curve

                fraction_pos, mean_predicted = calibration_curve(
                    y_true, y_prob, n_bins=10
                )

                fig_cal = go.Figure()
                fig_cal.add_trace(
                    go.Scatter(
                        x=mean_predicted,
                        y=fraction_pos,
                        mode="lines+markers",
                        name="BioGuard",
                        line=dict(color="#636efa", width=2),
                    )
                )
                fig_cal.add_trace(
                    go.Scatter(
                        x=[0, 1],
                        y=[0, 1],
                        mode="lines",
                        name="Perfect",
                        line=dict(color="gray", dash="dash"),
                    )
                )
                fig_cal.update_layout(
                    title="Calibration Curve",
                    xaxis_title="Mean Predicted Probability",
                    yaxis_title="Fraction of Positives",
                    height=400,
                )
                st.plotly_chart(fig_cal, use_container_width=True)

        with cal_col2:
            if "y_prob" in cal_data:
                y_prob = cal_data["y_prob"]
                fig_hist = go.Figure()
                fig_hist.add_trace(
                    go.Histogram(
                        x=y_prob,
                        nbinsx=50,
                        marker_color="#636efa",
                        opacity=0.7,
                        name="Predicted Probabilities",
                    )
                )
                fig_hist.update_layout(
                    title="Prediction Score Distribution",
                    xaxis_title="Predicted Probability",
                    yaxis_title="Count",
                    height=400,
                )
                st.plotly_chart(fig_hist, use_container_width=True)

    if not eval_data and cal_data is None:
        st.info(
            "No evaluation results or calibration data found. "
            "Run the evaluation pipeline to populate this page."
        )

    # Architecture diagram
    st.divider()
    st.subheader("Model Architecture")
    st.markdown("""
    ```
    Drug A SMILES -> GraphFeaturizer (ETKDGv3, 3D) -> GATv2 (2 layers) -> Pool --+
                                                                                  +-- Alpha Gate -> Final Logits
    Drug B SMILES -> GraphFeaturizer (ETKDGv3, 3D) -> GATv2 (2 layers) -> Pool --+        |
                                                                                            |
    Drug A SMILES -> EnzymeManager (CSV / LightGBM) -> Prior Head ---------------------+
    Drug B SMILES -> EnzymeManager (CSV / LightGBM) -> Prior Head
    ```
    """)

    st.markdown("""
    | Component | Details |
    |-----------|---------|
    | **Graph Pathway** | GATv2, 2 layers, 4 heads, 128-dim embeddings, mean+max pool |
    | **Prior Pathway** | Metabolic enzyme vectors (60-dim from ChEMBL, 15-dim LightGBM fallback) |
    | **Alpha Gate** | Learned sigmoid gate weighting graph vs. prior (converges ~0.81) |
    | **Loss** | BioFocalLoss (gamma=2.0, alpha=0.70) for class imbalance |
    | **Evaluation** | Bemis-Murcko scaffold-disjoint split (true OOD) |
    """)


# ===================================================================
# PAGE: Enzyme Explorer
# ===================================================================
elif page == "Enzyme Explorer":
    st.title("Metabolic Enzyme Profile Explorer")
    st.markdown(
        "Explore CYP450 enzyme substrate/inhibitor profiles "
        "for drugs in the database."
    )

    enzyme_df = load_enzyme_df()

    if enzyme_df is None:
        st.error("Enzyme features CSV not found.")
    else:
        meta_cols = ["drug_name", "smiles", "drug_id"]
        feature_cols = [c for c in enzyme_df.columns if c not in meta_cols]

        search = st.text_input("Search drug by name or SMILES", "")
        if search:
            mask = enzyme_df["drug_name"].str.contains(
                search, case=False, na=False
            )
            if "smiles" in enzyme_df.columns:
                mask = mask | enzyme_df["smiles"].str.contains(
                    search, case=False, na=False
                )
            filtered = enzyme_df[mask]
        else:
            filtered = enzyme_df.head(50)

        st.caption(f"Showing {len(filtered)} / {len(enzyme_df)} drugs")

        if not filtered.empty:
            selected_drug = st.selectbox(
                "Select a drug to visualize",
                filtered["drug_name"].tolist(),
            )

            row = filtered[filtered["drug_name"] == selected_drug].iloc[0]

            smi = row.get("smiles", "")
            if smi and pd.notna(smi):
                scol1, scol2 = st.columns([1, 2])
                with scol1:
                    img = smiles_to_image(str(smi), size=(350, 250))
                    if img:
                        st.image(img, caption=selected_drug)
                with scol2:
                    vals = row[feature_cols].values.astype(float)
                    active_mask = vals > 0
                    active_features = [
                        feature_cols[i]
                        for i in range(len(feature_cols))
                        if active_mask[i]
                    ]
                    active_values = vals[active_mask]

                    if len(active_features) > 0:
                        fig_bar = go.Figure(
                            go.Bar(
                                x=active_features,
                                y=active_values,
                                marker_color="#636efa",
                            )
                        )
                        fig_bar.update_layout(
                            title=f"Active Enzyme Interactions -- {selected_drug}",
                            xaxis_title="CYP Enzyme",
                            yaxis_title="Activity Score",
                            xaxis_tickangle=-45,
                            height=350,
                            margin=dict(b=100),
                        )
                        st.plotly_chart(fig_bar, use_container_width=True)
                    else:
                        st.info(
                            "No active enzyme interactions recorded "
                            "for this drug."
                        )

            with st.expander("Full Enzyme Profile"):
                profile = pd.DataFrame(
                    {
                        "Feature": feature_cols,
                        "Value": row[feature_cols].values,
                    }
                )
                st.dataframe(profile, use_container_width=True, height=400)

        st.divider()
        st.subheader("Dataset Statistics")

        stat_c1, stat_c2 = st.columns(2)
        with stat_c1:
            st.metric("Total Drugs", len(enzyme_df))
            st.metric("Enzyme Features", len(feature_cols))

            activity_counts = (
                (enzyme_df[feature_cols] > 0)
                .sum()
                .sort_values(ascending=False)
                .head(15)
            )
            fig_top = go.Figure(
                go.Bar(
                    x=activity_counts.values,
                    y=activity_counts.index,
                    orientation="h",
                    marker_color="#ef553b",
                )
            )
            fig_top.update_layout(
                title="Most Common Active Enzymes",
                xaxis_title="Number of Active Drugs",
                height=400,
                margin=dict(l=150),
            )
            st.plotly_chart(fig_top, use_container_width=True)

        with stat_c2:
            activity_per_drug = (enzyme_df[feature_cols] > 0).sum(axis=1)
            fig_dist = go.Figure(
                go.Histogram(
                    x=activity_per_drug,
                    nbinsx=30,
                    marker_color="#00cc96",
                    opacity=0.7,
                )
            )
            fig_dist.update_layout(
                title="Enzyme Activities per Drug",
                xaxis_title="Number of Active Enzyme Features",
                yaxis_title="Drug Count",
                height=350,
            )
            st.plotly_chart(fig_dist, use_container_width=True)

            sub_cols = [c for c in feature_cols if c.endswith("_sub")]
            inh_cols = [c for c in feature_cols if c.endswith("_inh")]
            total_sub = (enzyme_df[sub_cols] > 0).sum().sum()
            total_inh = (enzyme_df[inh_cols] > 0).sum().sum()

            fig_pie = go.Figure(
                go.Pie(
                    labels=["Substrate", "Inhibitor"],
                    values=[total_sub, total_inh],
                    marker_colors=["#636efa", "#ef553b"],
                    hole=0.4,
                )
            )
            fig_pie.update_layout(
                title="Substrate vs Inhibitor Annotations", height=300
            )
            st.plotly_chart(fig_pie, use_container_width=True)


# ===================================================================
# PAGE: About
# ===================================================================
elif page == "About":
    st.title("About BioGuard V3")

    st.markdown("""
    ### Drug-Drug Interaction Prediction with Adaptive Graph Attention Networks

    BioGuard V3 is a production-grade DDI prediction system designed for
    **New Chemical Entities (NCEs)** that uses an adaptive Graph Attention
    Network with a metabolic prior ensemble.

    #### Key Features

    - **Adaptive Alpha Gate**: Learns to weight structural (graph) vs. metabolic (prior) evidence
    - **3D Molecular Graphs**: ETKDGv3 conformer generation with spatial edge features
    - **Scaffold-Disjoint Evaluation**: True out-of-distribution testing (no inflated metrics)
    - **CYP450 Metabolic Priors**: 60-dimensional enzyme profiles from ChEMBL with LightGBM fallback
    - **Isotonic Calibration**: Calibrated probabilities for clinical decision support
    - **High-Recall Safety Net**: Prioritizes catching DDIs (recall ~0.81) over precision

    #### Architecture

    The model combines two independent pathways:
    1. **Graph Pathway**: GATv2 operating on 3D molecular graphs (46-dim node features, 8-dim edge features)
    2. **Prior Pathway**: Metabolic enzyme vectors processed through a learned MLP

    A sigmoid alpha gate adaptively weights these pathways
    (converges to ~0.81, preferring structural evidence).

    #### Citation

    Built on the TWOSIDES dataset via the Therapeutic Data Commons (TDC).
    """)

    st.divider()
    st.subheader("System Information")

    info_c1, info_c2 = st.columns(2)
    with info_c1:
        st.markdown(f"- **PyTorch:** {torch.__version__}")
        st.markdown(f"- **CUDA Available:** {torch.cuda.is_available()}")
        st.markdown(f"- **Model Path:** `{MODEL_PATH}`")
        st.markdown(f"- **Model Loaded:** {model_ready}")

    with info_c2:
        st.markdown(f"- **Artifacts Dir:** `{ARTIFACT_DIR}`")
        st.markdown(f"- **Data Dir:** `{DATA_DIR}`")
        if meta:
            st.markdown(
                f"- **Data Version:** {meta.get('data_version', 'N/A')}"
            )
            st.markdown(
                f"- **Training Epochs:** {meta.get('epochs', 'N/A')}"
            )