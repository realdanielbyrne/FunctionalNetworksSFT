#!/usr/bin/env python3
"""
Fine-Tuning Experiment Runner

This script runs comprehensive fine-tuning experiments comparing two approaches:
- Experiment A: PEFT (LoRA) only
- Experiment B: PEFT (LoRA) + ICA masking

Both experiments use the meta-llama/Llama-3.2-1B-Instruct model and sarcasm dataset.

Usage:
    python experiments/run_experiments.py [--experiment {a,b,both}] [--verbose]

Examples:
    python experiments/run_experiments.py --experiment a      # Run only Experiment A
    python experiments/run_experiments.py --experiment b      # Run only Experiment B  
    python experiments/run_experiments.py --experiment both   # Run both experiments
    python experiments/run_experiments.py                     # Run both experiments (default)
"""

import argparse
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Add the src directory to the Python path
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

# Configure logging
def setup_logging(verbose=False):
    """Set up logging configuration"""
    log_level = logging.DEBUG if verbose else logging.INFO
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Create logs directory
    os.makedirs('experiments/logs', exist_ok=True)
    
    # Create log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f'experiments/logs/experiment_run_{timestamp}.log'
    
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def run_experiment_a():
    """Run Experiment A: PEFT-only fine-tuning"""
    logger = logging.getLogger(__name__)
    logger.info("Starting Experiment A: PEFT-only fine-tuning")
    
    try:
        # Import and run experiment A
        sys.path.insert(0, str(Path(__file__).parent / "experiment_a_peft_only" / "scripts"))
        from run_experiment_a import run_experiment_a as run_a
        
        start_time = time.time()
        success = run_a()
        end_time = time.time()
        
        duration = end_time - start_time
        logger.info(f"Experiment A completed in {duration:.2f} seconds")
        
        return success
    except Exception as e:
        logger.error(f"Experiment A failed: {str(e)}")
        return False


def run_experiment_b():
    """Run Experiment B: PEFT + ICA masking fine-tuning"""
    logger = logging.getLogger(__name__)
    logger.info("Starting Experiment B: PEFT + ICA masking fine-tuning")
    
    try:
        # Import and run experiment B
        sys.path.insert(0, str(Path(__file__).parent / "experiment_b_peft_ica" / "scripts"))
        from run_experiment_b import run_experiment_b as run_b
        
        start_time = time.time()
        success = run_b()
        end_time = time.time()
        
        duration = end_time - start_time
        logger.info(f"Experiment B completed in {duration:.2f} seconds")
        
        return success
    except Exception as e:
        logger.error(f"Experiment B failed: {str(e)}")
        return False


def print_experiment_summary():
    """Print a summary of the experiment setup"""
    print("=" * 80)
    print("FINE-TUNING EXPERIMENT COMPARISON")
    print("=" * 80)
    print("Model: meta-llama/Llama-3.2-1B-Instruct")
    print("Dataset: datasets/sarcasm.csv (200 Q&A pairs)")
    print("Training Epochs: 2")
    print()
    print("Experiment A: PEFT (LoRA) only")
    print("  - LoRA rank: 16")
    print("  - LoRA alpha: 32")
    print("  - LoRA dropout: 0.1")
    print("  - ICA masking: DISABLED")
    print()
    print("Experiment B: PEFT (LoRA) + ICA masking")
    print("  - LoRA rank: 16 (identical to A)")
    print("  - LoRA alpha: 32 (identical to A)")
    print("  - LoRA dropout: 0.1 (identical to A)")
    print("  - ICA masking: ENABLED (key mode)")
    print("  - ICA components: 20")
    print("  - ICA percentile: 98.0")
    print("=" * 80)


def main():
    """Main experiment runner"""
    parser = argparse.ArgumentParser(
        description="Run fine-tuning experiments comparing PEFT vs PEFT+ICA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--experiment",
        choices=["a", "b", "both"],
        default="both",
        help="Which experiment(s) to run (default: both)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Set up logging
    logger = setup_logging(args.verbose)
    
    # Print experiment summary
    print_experiment_summary()
    
    # Track results
    results = {}
    start_time = time.time()
    
    # Run experiments based on selection
    if args.experiment in ["a", "both"]:
        logger.info("Running Experiment A...")
        results["experiment_a"] = run_experiment_a()
    
    if args.experiment in ["b", "both"]:
        logger.info("Running Experiment B...")
        results["experiment_b"] = run_experiment_b()
    
    # Calculate total time
    total_time = time.time() - start_time
    
    # Print final results
    print("\n" + "=" * 80)
    print("EXPERIMENT RESULTS")
    print("=" * 80)
    
    for exp_name, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"{exp_name.replace('_', ' ').title()}: {status}")
    
    print(f"\nTotal execution time: {total_time:.2f} seconds")
    print("\nOutput locations:")
    if "experiment_a" in results:
        print("  Experiment A: experiments/experiment_a_peft_only/output/")
    if "experiment_b" in results:
        print("  Experiment B: experiments/experiment_b_peft_ica/output/")
    
    # Exit with appropriate code
    if all(results.values()):
        print("\n🎉 All experiments completed successfully!")
        sys.exit(0)
    else:
        print("\n⚠️  Some experiments failed. Check logs for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
