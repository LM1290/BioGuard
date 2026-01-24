"""
BioGuard - Drug-Drug Interaction Prediction System
Main entry point for training, evaluation, and serving.

Usage:
    python -m bioguard.main [command] --split [random|cold|scaffold]
"""

import sys
import argparse
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    parser = argparse.ArgumentParser(
        description='BioGuard - Drug-Drug Interaction Prediction System',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        'command',
        choices=['train', 'eval', 'baselines', 'compare', 'serve'],
        help='Command to run'
    )

    # Series B: Added split strategy control
    parser.add_argument(
        '--split',
        type=str,
        default='cold',
        choices=['random', 'cold', 'scaffold'],
        help="Data split strategy: 'random' (easy), 'cold' (strict), 'scaffold' (hardest)"
    )

    args = parser.parse_args()

    if args.command == "train":
        from .train import run_training
        # Pass the whole args object so train.py can see flags
        run_training(args)

    elif args.command == "eval":
        from .evaluate import evaluate_model
        evaluate_model(override_split=args.split)

    elif args.command == "baselines":
        from .baselines import run_baselines
        run_baselines(args)

    elif args.command == "compare":
        from .compare import main as compare_main
        compare_main()

    elif args.command == "serve":
        import uvicorn
        print("="*60)
        print("Starting BioGuard API Server")
        print("="*60)
        print("IMPORTANT: Graph Neural Network Inference Mode")
        print("Running with workers=1 (ProcessPoolExecutor handles parallelism)")
        print("="*60 + "\n")

        uvicorn.run(
            "bioguard.api:app",
            host="0.0.0.0",
            port=8000,
            workers=1,
            log_level="info"
        )
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()