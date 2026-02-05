import numpy as np
import logging
import pandas as pd
from bioguard.enzyme import EnzymeManager
from bioguard.cyp_predictor import CYPPredictor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TestCYP")


def test_integration():
    print("\n" + "=" * 50)
    print("BIO-GUARD CYP450 INTEGRATION TEST")
    print("=" * 50)

    # 1. Setup
    mgr = EnzymeManager(allow_degraded=True)
    predictor = CYPPredictor()

    # ---------------------------------------------------------
    # TEST CASE A: KNOWN DRUG (Warfarin)
    # ---------------------------------------------------------
    # Warfarin is the 'Hello World' of DDI. It MUST be in your CSV.
    # If this fails, your CSV loading is broken.
    warfarin_smiles = "CC(=O)C(c1ccccc1)c1c(O)c2ccccc2oc1=O"

    print(f"\n[TEST A] Known Drug: Warfarin")
    vec_known = mgr.get_by_smiles(warfarin_smiles)

    # Validation: ChEMBL vectors are usually binary (0.0 or 1.0) or close to it.
    # We check if the vector is 'hard' (ground truth) vs 'soft' (predicted probability).
    is_hard_vector = np.all(np.isin(vec_known, [0.0, 1.0]))

    if np.sum(vec_known) == 0:
        print("  [WARNING] Warfarin vector is all zeros. Is it missing from your CSV?")
    elif is_hard_vector:
        print("  [PASS] Returned Discrete Vector (0.0/1.0). Retrieved from CSV Lookup.")
    else:
        print("  [WARN] Returned Probability Vector. Fallback triggered unexpectedly (Drug missing from CSV?).")

    # ---------------------------------------------------------
    # TEST CASE B: TRUE NCE ("BioGuard-X")
    # ---------------------------------------------------------
    # This is a nonsense molecule (Fluorinated-Phenyl-Sulfonamide-Pyridine derivative)
    # Guaranteed to NOT be in TWOSIDES/ChEMBL.
    nce_smiles = "Fc1ccc(S(=O)(=O)N(C)CC2=CN=C(C)N=C2CC(=O)O)cc1"

    print(f"\n[TEST B] True NCE: 'BioGuard-X'")
    print(f"  SMILES: {nce_smiles}")

    # 1. Run direct prediction to see what the model thinks
    raw_pred = predictor.predict(nce_smiles)
    print(f"  Predictor Raw Output (First 5): {raw_pred[:5]}")

    # 2. Run Manager Lookup (Should trigger fallback)
    vec_nce = mgr.get_by_smiles(nce_smiles)

    # Validation
    if np.array_equal(vec_known, vec_nce):
        print("  [FAIL] NCE Vector is identical to Warfarin. Caching/Hashing collision?")
    elif np.sum(vec_nce) == 0:
        print("  [FAIL] NCE Vector is all zeros. Predictor failed to fire.")
    else:
        # Check for probabilities (e.g. 0.73, 0.12) which indicate the RF model ran
        is_probabilistic = np.any((vec_nce > 0.0) & (vec_nce < 1.0))
        if is_probabilistic:
            print("  [PASS] Returned Probability Vector (e.g., 0.42, 0.88). Fallback logic SUCCESS.")
        else:
            print("  [UNK] Returned Hard Vector. Did the RF model predict 100% certainty or is this a hash collision?")


if __name__ == "__main__":
    # Ensure model exists
    try:
        test_integration()
    except Exception as e:
        print(f"\n[CRITICAL FAILURE] {e}")
        # If model missing, offer fix
        if "model" in str(e).lower() or "found" in str(e).lower():
            print("\n!!! Model missing. Run: python -m bioguard.cyp_predictor")
