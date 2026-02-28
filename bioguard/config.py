import os

# --- PATHS ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
ARTIFACT_DIR = os.path.join(BASE_DIR, 'artifacts')
LMDB_DIR = os.path.join(DATA_DIR, 'lmdb_cache')

# --- DATA HYPERPARAMETERS ---
DATA_VERSION = "v18"
RANDOM_SEED = 42
TRAIN_NEG_RATIO = 1.0
TEST_NEG_RATIO = 9.0

# --- MODEL DIMENSIONS (Must match Featurizer) ---
# Node Dim: Atom(24) + Degree(6) + Hybrid(5) + Aromatic(1) + Charge(1) + Chiral(4) = 41
NODE_DIM = 46
# Edge Dim: BondType(4) + Conjugated(1) + InRing(1) = 6
EDGE_DIM = 8

# --- TRAINING HYPERPARAMETERS ---
BATCH_SIZE = 128
LEARNING_RATE = 1e-4
EPOCHS = 50
NUM_WORKERS = min(4, os.cpu_count() or 1)