"""
BioGuard - Drug-Drug Interaction Prediction System
Main entry point for training, evaluation, and serving.

Usage:
    python -m bioguard.main [command] [options]

Examples:
    python -m bioguard.main train
    python -m bioguard.main eval
    python -m bioguard.main baselines
    python -m bioguard.main compare
    python -m bioguard.main serve
"""

import sys
import argparse
import uvicorn
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
        choices=['train', 'eval', 'baselines', 'compare', 'serve', 'validate'],
        help='Command to run'
    )

    args = parser.parse_args()

    if args.command == "train":
        from .train import run_training
        run_training(split_type='pair_disjoint')

    elif args.command == "eval":
        from .evaluate import evaluate_model
        evaluate_model(split_type='pair_disjoint')

    elif args.command == "baselines":
        from .baselines import run_all_baselines
        run_all_baselines(split_type='pair_disjoint')

    elif args.command == "compare":
        from .compare import main as compare_main
        compare_main(split_type='pair_disjoint')

    elif args.command == "validate":
        import os
        validate_data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'validate_data.py')
        if os.path.exists(validate_data_path):
            import subprocess
            result = subprocess.run([sys.executable, validate_data_path], capture_output=False)
            sys.exit(result.returncode)
        else:
            print("validate_data.py not found")
            sys.exit(1)

    elif args.command == "serve":
        # Use workers=1 because we use an internal ProcessPoolExecutor
        print("="*60)
        print("Starting BioGuard API Server")
        print("="*60)
        print("IMPORTANT: Running with workers=1 (internal process pool active)")
        print("For production, use external load balancer + multiple single-worker instances")
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
