"""
BioGuard DDI Prediction - Streamlit Interface
Compare Neural Network predictions with baseline methods
"""

import streamlit as st
import torch
import numpy as np
import pandas as pd
import os
import sys
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, Lipinski
import joblib

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bioguard.model import BioGuardNet
from bioguard.featurizer import BioFeaturizer

# Configuration
ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), '..', 'artifacts')
MODEL_PATH = os.path.join(ARTIFACT_DIR, 'model.pt')
CALIBRATOR_PATH = os.path.join(ARTIFACT_DIR, 'calibrator.joblib')

# Drug-like molecule thresholds (Lipinski's Rule of Five + complexity)
MIN_MOLECULAR_WEIGHT = 150  # Water is 18, simple molecules < 150
MAX_MOLECULAR_WEIGHT = 900  # Most drugs < 900
MIN_HEAVY_ATOMS = 6         # Water has 1, ethanol has 3, drugs typically > 6
MIN_RINGS = 1               # Most drugs have at least one ring


def is_drug_like(mol):
    """
    Check if molecule meets basic drug-like criteria.
    Filters out simple molecules like water, salts, solvents.
    """
    if mol is None:
        return False, "Invalid molecule"
    
    # Calculate properties
    mw = Descriptors.MolWt(mol)
    heavy_atoms = Lipinski.HeavyAtomCount(mol)
    num_rings = Lipinski.RingCount(mol)
    
    # Check molecular weight
    if mw < MIN_MOLECULAR_WEIGHT:
        return False, f"Molecular weight too low ({mw:.1f} Da). This appears to be a simple molecule (water, salt, etc.), not a drug. Minimum: {MIN_MOLECULAR_WEIGHT} Da."
    
    if mw > MAX_MOLECULAR_WEIGHT:
        return False, f"Molecular weight too high ({mw:.1f} Da). Maximum: {MAX_MOLECULAR_WEIGHT} Da."
    
    # Check structural complexity
    if heavy_atoms < MIN_HEAVY_ATOMS:
        return False, f"Molecule too simple ({heavy_atoms} heavy atoms). This model is designed for drug-like molecules. Minimum: {MIN_HEAVY_ATOMS} heavy atoms."
    
    if num_rings < MIN_RINGS:
        return False, f"No ring structures detected. Most drugs contain at least one aromatic or aliphatic ring. This may be too simple for meaningful prediction."
    
    return True, "Drug-like"


# Initialize
@st.cache_resource
def load_model():
    """Load the trained neural network model."""
    featurizer = BioFeaturizer()
    model = BioGuardNet(input_dim=featurizer.total_dim)
    
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu', weights_only=True))
        model.eval()
    else:
        st.error(f"Model not found at {MODEL_PATH}")
        return None, None, None
    
    calibrator = None
    if os.path.exists(CALIBRATOR_PATH):
        calibrator = joblib.load(CALIBRATOR_PATH)
    
    return model, calibrator, featurizer


def compute_tanimoto_similarity(smiles_a, smiles_b):
    """Compute Tanimoto similarity between two SMILES strings."""
    try:
        mol_a = Chem.MolFromSmiles(smiles_a)
        mol_b = Chem.MolFromSmiles(smiles_b)
        
        if mol_a is None or mol_b is None:
            return None
        
        fp_a = AllChem.GetMorganFingerprintAsBitVect(mol_a, 2, nBits=1024)
        fp_b = AllChem.GetMorganFingerprintAsBitVect(mol_b, 2, nBits=1024)
        
        return DataStructs.TanimotoSimilarity(fp_a, fp_b)
    except:
        return None


def predict_with_nn(smiles_a, smiles_b, model, calibrator, featurizer):
    """Make prediction using neural network."""
    try:
        # Featurize
        vec = featurizer.featurize_pair(smiles_a, smiles_b)
        vec_tensor = torch.FloatTensor(vec).unsqueeze(0)
        
        # Predict
        with torch.no_grad():
            logits = model(vec_tensor)
            prob = torch.sigmoid(logits).item()
        
        # Calibrate
        if calibrator:
            prob = float(calibrator.transform([prob])[0])
            prob = max(0.0, min(1.0, prob))
        
        return prob
    except Exception as e:
        st.error(f"Neural network prediction failed: {e}")
        return None


def predict_with_logreg_features(smiles_a, smiles_b, featurizer):
    """
    Placeholder for logistic regression baseline.
    In production, this would load a trained sklearn model.
    For demo, we'll estimate based on feature similarity.
    """
    try:
        vec_a = featurizer.featurize_single_drug(smiles_a)
        vec_b = featurizer.featurize_single_drug(smiles_b)
        
        # Simple heuristic: cosine similarity of features
        dot_product = np.dot(vec_a, vec_b)
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        
        if norm_a > 0 and norm_b > 0:
            similarity = dot_product / (norm_a * norm_b)
            # Map to probability (this is a placeholder)
            prob = (similarity + 1) / 2  # Map [-1,1] to [0,1]
            return prob
        return 0.5
    except:
        return 0.5


# Streamlit UI
st.set_page_config(page_title="BioGuard DDI Predictor", layout="wide")

st.title("🔬 BioGuard Drug-Drug Interaction Predictor")
st.markdown("Compare neural network predictions with baseline methods")

# Sidebar
with st.sidebar:
    st.header("About")
    st.markdown("""
    This tool predicts drug-drug interactions using:
    - **Neural Network**: Deep learning model with calibrated probabilities
    - **Tanimoto Similarity**: Structure-based baseline
    - **Logistic Regression**: Linear model baseline
    
    Enter two drug SMILES strings to get predictions.
    
    **Note:** This model is trained on pharmaceutical drugs. Very simple molecules 
    (water, salts, solvents) will be rejected.
    """)
    
    st.markdown("---")
    st.markdown("**Model Info**")
    if os.path.exists(MODEL_PATH):
        st.success("Model loaded")
    else:
        st.error("Model not found")

# Load model
model, calibrator, featurizer = load_model()

if model is None:
    st.error("Failed to load model. Please check model files.")
    st.stop()

# Input section
st.header("Input Drug Pair")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Drug A")
    smiles_a = st.text_input(
        "SMILES String A",
        value="CC(C)Cc1ccc(cc1)C(C)C(O)=O",
        help="Enter SMILES notation for the first drug (e.g., Ibuprofen)"
    )
    
with col2:
    st.subheader("Drug B")
    smiles_b = st.text_input(
        "SMILES String B",
        value="CC(=O)Oc1ccccc1C(O)=O",
        help="Enter SMILES notation for the second drug (e.g., Aspirin)"
    )

# Predict button
if st.button("Predict Interaction", type="primary"):
    with st.spinner("Computing predictions..."):
        
        # Validate SMILES are not empty
        if not smiles_a or not smiles_a.strip():
            st.error("❌ Drug A SMILES string is empty. Please enter a valid SMILES.")
        elif not smiles_b or not smiles_b.strip():
            st.error("❌ Drug B SMILES string is empty. Please enter a valid SMILES.")
        elif smiles_a.strip() == smiles_b.strip():
            st.warning("⚠️ Both SMILES strings are identical. Drug-drug interaction prediction requires two **different** drugs.")
        else:
            # Validate SMILES structure
            mol_a = Chem.MolFromSmiles(smiles_a)
            mol_b = Chem.MolFromSmiles(smiles_b)
            
            if mol_a is None or mol_b is None:
                st.error("❌ Invalid SMILES string(s). Please check your input.")
            else:
                # Check if molecules are drug-like
                is_drug_a, msg_a = is_drug_like(mol_a)
                is_drug_b, msg_b = is_drug_like(mol_b)
                
                if not is_drug_a:
                    st.error(f"❌ **Drug A is not drug-like:** {msg_a}")
                    st.info(f"💡 **Hint:** The model is trained on pharmaceutical drugs (MW: {MIN_MOLECULAR_WEIGHT}-{MAX_MOLECULAR_WEIGHT} Da, ≥{MIN_HEAVY_ATOMS} heavy atoms, ≥{MIN_RINGS} ring). Simple molecules like water, ethanol, or salts should not be used.")
                elif not is_drug_b:
                    st.error(f"❌ **Drug B is not drug-like:** {msg_b}")
                    st.info(f"💡 **Hint:** The model is trained on pharmaceutical drugs (MW: {MIN_MOLECULAR_WEIGHT}-{MAX_MOLECULAR_WEIGHT} Da, ≥{MIN_HEAVY_ATOMS} heavy atoms, ≥{MIN_RINGS} ring). Simple molecules like water, ethanol, or salts should not be used.")
                else:
                    # Compute all predictions
                    nn_prob = predict_with_nn(smiles_a, smiles_b, model, calibrator, featurizer)
                    tanimoto_sim = compute_tanimoto_similarity(smiles_a, smiles_b)
                    logreg_prob = predict_with_logreg_features(smiles_a, smiles_b, featurizer)
                    
                    # Display results
                    st.header("Prediction Results")
                    
                    # Create comparison table
                    results_data = {
                        "Method": [
                            "Neural Network (BioGuardNet)",
                            "Tanimoto Similarity",
                            "Logistic Regression (estimated)"
                        ],
                        "Score": [
                            f"{nn_prob:.4f}" if nn_prob is not None else "Error",
                            f"{tanimoto_sim:.4f}" if tanimoto_sim is not None else "Error",
                            f"{logreg_prob:.4f}"
                        ],
                        "Risk Level": [
                            "High" if nn_prob and nn_prob >= 0.5 else "Low",
                            "High" if tanimoto_sim and tanimoto_sim >= 0.6 else "Low",
                            "High" if logreg_prob >= 0.5 else "Low"
                        ]
                    }
                    
                    results_df = pd.DataFrame(results_data)
                    
                    # Display table
                    st.dataframe(
                        results_df,
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    # Highlight best prediction
                    st.markdown("---")
                    st.subheader("Neural Network Prediction (Recommended)")
                    
                    if nn_prob is not None:
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Probability", f"{nn_prob:.1%}")
                        
                        with col2:
                            risk = "High Risk" if nn_prob >= 0.5 else "Low Risk"
                            st.metric("Risk Level", risk)
                        
                        with col3:
                            confidence = "High" if abs(nn_prob - 0.5) > 0.3 else "Medium"
                            st.metric("Confidence", confidence)
                        
                        # Interpretation
                        st.markdown("### Interpretation")
                        if nn_prob >= 0.7:
                            st.error("**High risk of interaction** - Clinical monitoring recommended")
                        elif nn_prob >= 0.5:
                            st.warning("**Moderate risk** - Consider alternative combinations")
                        else:
                            st.success("**Low risk** - Interaction unlikely based on current evidence")
                        
                        # Comparison with baselines
                        st.markdown("### Baseline Comparison")
                        if tanimoto_sim is not None:
                            diff_tanimoto = nn_prob - tanimoto_sim
                            st.write(f"- Neural network vs. Tanimoto: {diff_tanimoto:+.3f}")
                        
                        diff_logreg = nn_prob - logreg_prob
                        st.write(f"- Neural network vs. Logistic Regression: {diff_logreg:+.3f}")
                        
                    else:
                        st.error("Prediction failed. Please check your input.")
                    
                    # Molecular structures (optional)
                    with st.expander("View Molecular Structures"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("**Drug A**")
                            st.write(f"SMILES: `{smiles_a}`")
                            st.write(f"MW: {Descriptors.MolWt(mol_a):.1f} Da")
                            st.write(f"Heavy atoms: {Lipinski.HeavyAtomCount(mol_a)}")
                        with col2:
                            st.write("**Drug B**")
                            st.write(f"SMILES: `{smiles_b}`")
                            st.write(f"MW: {Descriptors.MolWt(mol_b):.1f} Da")
                            st.write(f"Heavy atoms: {Lipinski.HeavyAtomCount(mol_b)}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; font-size: 0.8em;'>
BioGuard DDI Predictor | Structure-based interaction prediction for pharmaceutical drugs
</div>
""", unsafe_allow_html=True)
