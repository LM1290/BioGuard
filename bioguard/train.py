import argparse
import hashlib
import io
import logging
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from sklearn.metrics import average_precision_score, roc_auc_score
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader as PyGDataLoader
from tqdm import tqdm

from bioguard.artifacts import file_sha256, save_model_bundle
from bioguard.config import (
    DATA_VERSION,
    EDGE_DIM,
    LMDB_DIR,
    NODE_DIM,
    NUM_WORKERS,
    RANDOM_SEED,
)
from bioguard.data_loader import load_twosides_binary_pu
from bioguard.cohorts import cohort_manifest
from bioguard.graph_cache import filter_cached_pairs, read_cache, graph_contract
from bioguard.molecular import STANDARDIZATION_POLICY
from bioguard.enzyme import EnzymeManager
from bioguard.model import BioGuardGAT
from bioguard.pu import NonNegativePULoss, PUBalancedBatchSampler

# --- CONFIG ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACT_DIR = os.path.join(BASE_DIR, 'artifacts')
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_PATH = os.path.join(ARTIFACT_DIR, 'model.pt')

logger = logging.getLogger(__name__)


def sample_membership_sha256(frame):
    values = sorted(
        f"{row.sample_id}\t{row.observation}"
        for row in frame.itertuples(index=False)
    )
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


class BioGuardDataset(Dataset):
    """
    LMDB-backed pair dataset for the legacy binary DDI task.
    """

    def __init__(
        self,
        root,
        df,
        split="train",
        transform=None,
        pre_transform=None,
        enzyme_manager=None,
        lmdb_path=None,
    ):
        self.lmdb_path = lmdb_path or LMDB_DIR
        required_columns = {
            "smiles_a",
            "smiles_b",
            "target",
            "observation",
        }
        missing = sorted(required_columns.difference(df.columns))
        if missing:
            raise ValueError(
                f"PU dataset is missing columns: {', '.join(missing)}"
            )
        if not set(df["observation"]).issubset({"positive", "unlabeled"}):
            raise ValueError("Unsupported PU observation value")

        # --- PRE-FILTER: Synchronize DataFrame with LMDB Cache ---
        df_clean = filter_cached_pairs(df, self.lmdb_path)
        dropped = len(df) - len(df_clean)
        if dropped > 0:
            print(f"[{split.upper()}] Dropped {dropped} invalid pairs to sync with LMDB cache.")

        self.df = df_clean.reset_index(drop=True)
        self.split = split
        self.enzyme_manager = enzyme_manager or EnzymeManager(
            allow_degraded=False
        )
        self.env = None
        super().__init__(root, transform, pre_transform)

    @property
    def processed_file_names(self):
        return ['not_used.pt']

    def _init_db(self):
        self.env = read_cache(self.lmdb_path)

    def len(self):
        return len(self.df)

    def get(self, idx):
        if self.env is None:
            self._init_db()

        row = self.df.iloc[idx]
        smi_a = row['smiles_a']
        smi_b = row['smiles_b']

        with self.env.begin(write=False) as txn:
            raw_a = txn.get(smi_a.encode('utf-8'))
            raw_b = txn.get(smi_b.encode('utf-8'))

            if raw_a is None or raw_b is None:
                raise KeyError(f"Missing SMILES in LMDB: {smi_a if raw_a is None else smi_b}")

            data_a = torch.load(io.BytesIO(raw_a), weights_only=False)
            data_b = torch.load(io.BytesIO(raw_b), weights_only=False)

        enzyme_a = np.asarray(
            self.enzyme_manager.get_by_smiles(smi_a),
            dtype=np.float32,
        )
        enzyme_b = np.asarray(
            self.enzyme_manager.get_by_smiles(smi_b),
            dtype=np.float32,
        )
        data_a.enzyme_a = torch.from_numpy(enzyme_a).unsqueeze(0)
        data_b.enzyme_b = torch.from_numpy(enzyme_b).unsqueeze(0)
        target = torch.tensor([float(row["target"])], dtype=torch.float)

        return data_a, data_b, target


def train_epoch(model, loader, optimizer, criterion, device, epoch_num, run_profiler=False, force_graph_only=False):
    model.train()
    total_loss = 0
    pbar = tqdm(loader, desc=f"Epoch {epoch_num} [Train | Phase {'1' if force_graph_only else '2'}]")

    if run_profiler:
        activities = [torch.profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        prof = torch.profiler.profile(
            activities=activities,
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(os.path.join(ARTIFACT_DIR, 'profiler_logs')),
            record_shapes=True, profile_memory=True, with_stack=True
        )
        prof.start()

    for batch_data in pbar:
        batch_a = batch_data[0].to(device)
        batch_b = batch_data[1].to(device)
        batch_y = batch_data[2].to(device)

        optimizer.zero_grad()
        # Enforce two-phase routing
        logits, _ = model(batch_a, batch_b, force_graph_only=force_graph_only)

        loss = criterion(logits, batch_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")

        if run_profiler:
            prof.step()

    if run_profiler:
        prof.stop()
        print(f"\n[Profiler Output Saved to {ARTIFACT_DIR}/profiler_logs]")

    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, criterion, device, epoch_num, force_graph_only=False):
    model.eval()
    y_true, y_logits, y_probs = [], [], []
    alpha_telemetry = []

    for batch_data in tqdm(loader, desc=f"Epoch {epoch_num} [Val  ]"):
        batch_a = batch_data[0].to(device)
        batch_b = batch_data[1].to(device)
        batch_y = batch_data[2].to(device)

        logits, alpha = model(batch_a, batch_b, force_graph_only=force_graph_only)
        probs = torch.sigmoid(logits)
        y_true.extend(batch_y.cpu().numpy().flatten())
        y_logits.extend(logits.cpu().numpy().flatten())
        y_probs.extend(probs.cpu().numpy().flatten())

        alpha_telemetry.extend(alpha.cpu().numpy().flatten())

    y_true = np.array(y_true)
    y_logits = np.array(y_logits)
    y_probs = np.array(y_probs)
    mean_alpha = float(np.mean(alpha_telemetry)) if len(alpha_telemetry) > 0 else 0.0

    auprc = average_precision_score(y_true, y_probs)
    roc_auc = roc_auc_score(y_true, y_probs)
    pu_risk = criterion(
        torch.from_numpy(y_logits),
        torch.from_numpy(y_true),
    ).item()

    return pu_risk, auprc, roc_auc, mean_alpha


def run_training(args):
    if args.epochs < 1:
        raise ValueError("epochs must be at least 1")
    if args.warmup_epochs < 0 or args.warmup_epochs >= args.epochs:
        raise ValueError("warmup_epochs must be non-negative and less than epochs")
    if args.batch_size < 2:
        raise ValueError("batch_size must be at least 2 because the model uses BatchNorm")
    if args.class_prior is None:
        raise ValueError(
            "--class-prior is required for positive-unlabeled training; "
            "it must come from a documented prevalence estimate or sensitivity analysis"
        )
    if not 0.0 < args.class_prior < 1.0:
        raise ValueError("class_prior must be within (0, 1)")
    if args.unlabeled_ratio <= 0:
        raise ValueError("unlabeled_ratio must be greater than zero")

    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data_bundle = load_twosides_binary_pu(
        split_method=args.split,
        train_unlabeled_ratio=args.unlabeled_ratio,
        val_unlabeled_ratio=args.unlabeled_ratio,
        test_unlabeled_ratio=args.unlabeled_ratio,
        random_seed=RANDOM_SEED,
    )
    df = data_bundle.records
    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    test_df = df[df["split"] == "test"]
    for split_name, split_df in (
        ("train", train_df),
        ("validation", val_df),
        ("test", test_df),
    ):
        if split_df.empty:
            raise ValueError(f"The {split_name} split is empty")
        if split_df["target"].nunique() != 2:
            raise ValueError(
                f"The {split_name} split must contain positive and unlabeled samples"
            )

    enzyme_manager = EnzymeManager(allow_degraded=False)
    train_dataset = BioGuardDataset(
        root=DATA_DIR,
        df=train_df,
        split="train",
        enzyme_manager=enzyme_manager,
    )
    val_dataset = BioGuardDataset(
        root=DATA_DIR,
        df=val_df,
        split="val",
        enzyme_manager=enzyme_manager,
    )
    test_dataset = BioGuardDataset(
        root=DATA_DIR,
        df=test_df,
        split="test",
        enzyme_manager=enzyme_manager,
    )
    if (
        len(train_dataset) < 2
        or len(val_dataset) == 0
        or len(test_dataset) == 0
    ):
        raise ValueError(
            "LMDB filtering left fewer than two training pairs "
            "or an empty validation/test set"
        )
    for split_name, dataset in (
        ("train", train_dataset),
        ("validation", val_dataset),
        ("test", test_dataset),
    ):
        if dataset.df["target"].nunique() != 2:
            raise ValueError(
                f"LMDB filtering removed one PU class from the {split_name} split"
            )

    effective_samples = {
        split_name: {
            "count": int(len(dataset)),
            "sample_membership_sha256": sample_membership_sha256(dataset.df),
        }
        for split_name, dataset in (
            ("train", train_dataset),
            ("val", val_dataset),
            ("test", test_dataset),
        )
    }

    train_sampler = PUBalancedBatchSampler(
        train_dataset.df["target"].to_numpy(),
        batch_size=args.batch_size,
        seed=RANDOM_SEED,
    )
    train_loader = PyGDataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=NUM_WORKERS,
    )
    val_loader = PyGDataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    enzyme_dim = enzyme_manager.vector_dim

    print(f"Initializing BioGuardGAT with Enzyme Dim: {enzyme_dim}")
    model = BioGuardGAT(
        node_dim=NODE_DIM,
        edge_dim=EDGE_DIM,
        embedding_dim=args.embedding_dim,
        heads=args.heads,
        enzyme_dim=enzyme_dim
    ).to(device)

    pretrain_path = os.path.join(ARTIFACT_DIR, 'gat_encoder_weights.pt')
    if os.path.exists(pretrain_path):
        print(f"Loading pretrained encoder from {pretrain_path}...")
        try:
            checkpoint = torch.load(pretrain_path, map_location=device, weights_only=True)
            if isinstance(checkpoint, dict) and 'atom_encoder' in checkpoint:
                model.atom_encoder.load_state_dict(checkpoint['atom_encoder'])
                model.conv1.load_state_dict(checkpoint['conv1'])
                print("Pretrained weights loaded successfully! (Transfer Learning Activated)")
        except Exception as e:
            print(f"WARNING: Could not load pretrained weights: {e}")

    criterion = NonNegativePULoss(class_prior=args.class_prior)
    best_val_auprc = -1.0
    counter = 0

    # ==========================================
    # PHASE 1: GAT PHYSICS WARMUP
    # ==========================================
    if args.warmup_epochs > 0:
        print(f"\n--- PHASE 1: GAT Physics Warmup ({args.warmup_epochs} Epochs) ---")
        # Freeze Prior Head and Gate
        for param in model.prior_head.parameters():
            param.requires_grad = False
        for param in model.alpha_gate.parameters():
            param.requires_grad = False

        optimizer_phase1 = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

        for epoch in range(1, args.warmup_epochs + 1):
            run_prof = getattr(args, "profile", False) and epoch == 1
            train_loss = train_epoch(model, train_loader, optimizer_phase1, criterion, device, epoch,
                                     run_profiler=run_prof, force_graph_only=True)
            val_loss, val_auprc, val_roc, mean_alpha = validate(
                model,
                val_loader,
                criterion,
                device,
                epoch,
                force_graph_only=True,
            )
            print(
                f"Phase 1 Epoch {epoch}: Train Loss={train_loss:.4f} | Val AUPRC={val_auprc:.4f} | Alpha Telemetry: {mean_alpha:.4f}")

    # ==========================================
    # PHASE 2: ADAPTIVE ENSEMBLE CALIBRATION
    # ==========================================
    print("\n--- PHASE 2: Adaptive Ensemble Calibration ---")
    # Unfreeze everything
    for param in model.parameters():
        param.requires_grad = True

    # Drop LR for fine-tuning the fusion (10x smaller)
    optimizer_phase2 = optim.AdamW(model.parameters(), lr=args.lr * 0.5, weight_decay=1e-4)

    start_epoch = args.warmup_epochs + 1
    for epoch in range(start_epoch, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer_phase2, criterion, device, epoch, run_profiler=False,
                                 force_graph_only=False)
        val_loss, val_auprc, val_roc, mean_alpha = validate(
            model,
            val_loader,
            criterion,
            device,
            epoch,
        )

        print(f"Phase 2 Epoch {epoch}: Train Loss={train_loss:.4f} | Val Loss={val_loss:.4f}")
        print(f"    -> Val AUPRC: {val_auprc:.4f} | ROC-AUC: {val_roc:.4f} | Mean Alpha: {mean_alpha:.4f}")

        if val_auprc > best_val_auprc:
            best_val_auprc = val_auprc
            counter = 0
            meta = {
                "threshold": 0.5,
                "threshold_validated": False,
                "calibrator_enabled": False,
                "split_type": args.split,
                "data_version": DATA_VERSION,
                "random_seed": RANDOM_SEED,
                "scientific_task": (
                    "binary_any_reported_event_positive_unlabeled"
                ),
                "source_task": "multi_relation_positive_associations",
                "training_provenance_status": "manifest_embedded",
                "class_prior": float(args.class_prior),
                "loss": "non_negative_positive_unlabeled",
                "unlabeled_ratio": float(args.unlabeled_ratio),
                "data_manifest": data_bundle.manifest,
                "lmdb_effective_samples": effective_samples,
                "effective_cohorts": cohort_manifest(pd.concat([
                    train_dataset.df, val_dataset.df, test_dataset.df,
                ], ignore_index=True)),
                "standardization_policy": STANDARDIZATION_POLICY,
                "graph_contract": graph_contract(),
                "enzyme_fallback_status": "zero_vector_for_unseen_smiles",
                "supporting_files": [
                    {
                        "role": "enzyme_features",
                        "path": "data/enzyme_features_full.csv",
                        "sha256": file_sha256(
                            os.path.join(
                                DATA_DIR,
                                "enzyme_features_full.csv",
                            )
                        ),
                    }
                ],
                "best_epoch": epoch,
                "val_auprc": float(val_auprc),
                "val_roc": float(val_roc),
                "val_loss": float(val_loss),
                "mean_alpha_gate": float(mean_alpha),
            }
            save_model_bundle(
                model,
                ARTIFACT_DIR,
                metadata=meta,
                checkpoint_name=os.path.basename(MODEL_PATH),
            )

            print("    -> Model saved (New Best AUPRC).")
        else:
            counter += 1
            print(f"    -> Patience {counter}/{args.patience}")
            if counter >= args.patience:
                print("Early stopping triggered.")
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BioGuard Training")
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--warmup_epochs', type=int, default=5, help='Epochs to train GAT exclusively')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--patience', type=int, default=8, help='Early stopping patience')
    parser.add_argument('--embedding_dim', type=int, default=128, help='Latent dimension for atoms')
    parser.add_argument('--heads', type=int, default=4, help='Number of GAT attention heads')
    parser.add_argument(
        "--split",
        default="scaffold_disjoint",
        choices=["pair_random", "drug_disjoint", "scaffold_disjoint"],
    )
    parser.add_argument(
        "--class-prior",
        type=float,
        default=None,
        help="Required prevalence assumption for nnPU risk estimation",
    )
    parser.add_argument(
        "--unlabeled-ratio",
        type=float,
        default=1.0,
        help="Sampled-unlabeled pairs per observed-positive pair",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Capture an expensive PyTorch trace during the first warmup epoch",
    )

    args = parser.parse_args()
    run_training(args)
