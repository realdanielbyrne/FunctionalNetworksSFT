#!/usr/bin/env python3
"""
Experiment A: PEFT-Only Fine-Tuning Script

This script runs Experiment A, which fine-tunes meta-llama/Llama-3.2-1B-Instruct
using PEFT (LoRA) only, without ICA masking.

Usage:
    python experiments/experiment_a_peft_only/scripts/run_experiment_a.py

The script will:
1. Load the configuration from experiment_a_config.yaml
2. Run the fnsft_trainer with PEFT-only settings
3. Save results to experiments/experiment_a_peft_only/output/
"""

import os
import sys
import logging
from pathlib import Path

# Add the src directory to the Python path
project_root = Path(__file__).parent.parent.parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

from functionalnetworkssft.fnsft_trainer import main as fnsft_main

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(
            "experiments/experiment_a_peft_only/output/experiment_a.log"
        ),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def run_experiment_a():
    """
    Run Experiment A: PEFT-only fine-tuning
    """
    logger.info("=" * 80)
    logger.info("EXPERIMENT A: PEFT-ONLY FINE-TUNING")
    logger.info("=" * 80)
    logger.info("Model: meta-llama/Llama-3.2-1B-Instruct")
    logger.info("Dataset: datasets/sarcasm.csv")
    logger.info("Method: PEFT (LoRA) only")
    logger.info("Epochs: 2")
    logger.info("ICA Masking: DISABLED")
    logger.info("=" * 80)

    # Set up the configuration file path
    config_path = "experiments/experiment_a_peft_only/config/experiment_a_config.yaml"

    # Verify configuration file exists
    if not os.path.exists(config_path):
        logger.error(f"Configuration file not found: {config_path}")
        return False

    # Verify dataset exists
    dataset_path = "datasets/sarcasm.csv"
    if not os.path.exists(dataset_path):
        logger.error(f"Dataset file not found: {dataset_path}")
        return False

    # Create output directory if it doesn't exist
    output_dir = "experiments/experiment_a_peft_only/output"
    os.makedirs(output_dir, exist_ok=True)

    # Simulate command line arguments for the fnsft_trainer
    original_argv = sys.argv.copy()
    try:
        sys.argv = ["fnsft_trainer.py", "--config", config_path]

        logger.info(f"Starting training with config: {config_path}")
        logger.info("This may take several minutes...")

        # Run the training
        fnsft_main()

        logger.info("Experiment A completed successfully!")
        logger.info(f"Results saved to: {output_dir}")
        return True

    except Exception as e:
        logger.error(f"Experiment A failed with error: {str(e)}")
        return False
    finally:
        # Restore original argv
        sys.argv = original_argv


if __name__ == "__main__":
    success = run_experiment_a()
    if success:
        print("\n✅ Experiment A completed successfully!")
        print("📁 Check experiments/experiment_a_peft_only/output/ for results")
        print("\n💡 To run evaluation comparing both models, use:")
        print("   python experiments/peft_vs_peft-ica/evaluate_models.py")
        print("   (Note: Both experiments A and B must be completed first)")
    else:
        print("\n❌ Experiment A failed!")
        print(
            "📋 Check experiments/experiment_a_peft_only/output/experiment_a.log for details"
        )
        sys.exit(1)
