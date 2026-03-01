import argparse
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# --- COMMAND HANDLERS (Lazy Imports) ---

def run_serve(args):
    """Start the API server."""
    # Lazy import to prevent slowdowns if just training
    import uvicorn
    print("=" * 60)
    print(f"Starting BioGuard API on {args.host}:{args.port}")
    print("Mode: Production (GNN Inference)")
    print("=" * 60)
    # Note: We reference the module path string for reload support if needed
    uvicorn.run(
        "bioguard.api:app",
        host=args.host,
        port=args.port,
        workers=1,
        log_level="info"
    )


def run_train(args):
    """Run the training pipeline."""
    from bioguard.train import run_training

    # Handle quick mode overrides
    if args.quick:
        print("--- QUICK MODE: 1 Epoch, Tiny Batch ---")
        args.epochs = 1
        args.batch_size = 4
        args.split = 'random'

    run_training(args)


def run_eval(args):
    """Run model evaluation."""
    from bioguard.evaluate import evaluate_model
    evaluate_model(override_split=args.split)


def run_baselines(args):
    """Run baseline models (Tanimoto, RF, LR)."""
    from bioguard.baselines import run_baselines
    run_baselines(args)


def run_compare(args):
    """Generate comparison charts and report."""
    from bioguard.compare import main as compare_main
    compare_main()


# --- MAIN ENTRYPOINT ---

def main():
    parser = argparse.ArgumentParser(description="BioGuard CLI - DDI Prediction System")
    subparsers = parser.add_subparsers(dest="command", required=True, help="Command to run")

    # 1. SERVE
    p_serve = subparsers.add_parser("serve", help="Start API Server")
    p_serve.add_argument("--host", default="0.0.0.0", help="Host IP")
    p_serve.add_argument("--port", type=int, default=8000, help="Port")
    p_serve.set_defaults(func=run_serve)

    # 2. TRAIN
    p_train = subparsers.add_parser("train", help="Train GNN Model")
    # Training Params
    p_train.add_argument("--epochs", type=int, default=50)
    p_train.add_argument("--batch_size", type=int, default=128)
    p_train.add_argument("--lr", type=float, default=1e-4)
    p_train.add_argument("--patience", type=int, default=8, help="Early stopping patience")

    # Model Architecture Params (MISSING IN YOUR VERSION)
    p_train.add_argument("--embedding_dim", type=int, default=128, help="Latent dimension for atoms")
    p_train.add_argument("--heads", type=int, default=4, help="Number of GAT attention heads")

    # Data Params
    p_train.add_argument("--split", type=str, default='cold_drug', choices=['random', 'cold_drug', 'scaffold'])
    p_train.add_argument("--quick", action='store_true', help="Sanity check run")
    p_train.add_argument("--warmup_epochs", type=int, default=5, help="Epochs to train GAT exclusively")
    p_train.set_defaults(func=run_train)

    # 3. EVAL
    p_eval = subparsers.add_parser("eval", help="Evaluate Trained Model")
    p_eval.add_argument("--split", type=str, default=None, help="Override split type")
    p_eval.set_defaults(func=run_eval)

    # 4. BASELINES
    p_base = subparsers.add_parser("baselines", help="Run Baseline Models")
    p_base.add_argument("--split", type=str, default='random', choices=['random', 'cold_drug', 'scaffold'])
    p_base.add_argument("--quick", action='store_true', help="Skip slow ML baselines")
    p_base.set_defaults(func=run_baselines)

    # 5. COMPARE
    p_comp = subparsers.add_parser("compare", help="Generate Comparison Report")
    p_comp.set_defaults(func=run_compare)

    # Parse & Execute
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
