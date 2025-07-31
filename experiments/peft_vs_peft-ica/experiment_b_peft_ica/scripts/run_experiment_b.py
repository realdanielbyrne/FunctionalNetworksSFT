#!/usr/bin/env python3
"""
Experiment B: PEFT + ICA Masking Fine-Tuning Script

This script runs Experiment B, which fine-tunes meta-llama/Llama-3.2-1B-Instruct
using PEFT (LoRA) with ICA masking enabled.

Usage:
    python experiment_b_peft_ica/scripts/run_experiment_b.py

The script will:
1. Load the configuration from experiment_b_config.yaml
2. Run the fnsft_trainer with PEFT+ICA settings
3. Save results to experiment_b_peft_ica/output/
"""

import os
import sys
import logging

from functionalnetworkssft.fnsft_trainer import main as fnsft_main

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("experiment_b_peft_ica/output/experiment_b.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def run_experiment_b():
    """
    Run Experiment B: PEFT + ICA masking fine-tuning
    """
    logger.info("=" * 80)
    logger.info("EXPERIMENT B: PEFT + ICA MASKING FINE-TUNING")
    logger.info("=" * 80)
    logger.info("Model: meta-llama/Llama-3.2-1B-Instruct")
    logger.info("Dataset: ../../datasets/sarcasm.csv")
    logger.info("Method: PEFT (LoRA) + ICA masking")
    logger.info("Epochs: 2")
    logger.info("ICA Masking: ENABLED (key mode)")
    logger.info("ICA Components: 20")
    logger.info("ICA Percentile: 98.0")
    logger.info("=" * 80)

    # Set up the configuration file path
    config_path = "experiment_b_peft_ica/config/experiment_b_config.yaml"

    # Verify configuration file exists
    if not os.path.exists(config_path):
        logger.error(f"Configuration file not found: {config_path}")
        return False

    # Verify dataset exists
    dataset_path = "../../datasets/sarcasm.csv"
    if not os.path.exists(dataset_path):
        logger.error(f"Dataset file not found: {dataset_path}")
        return False

    # Create output directory if it doesn't exist
    output_dir = "experiment_b_peft_ica/output"
    os.makedirs(output_dir, exist_ok=True)

    # Simulate command line arguments for the fnsft_trainer
    original_argv = sys.argv.copy()
    try:
        sys.argv = ["fnsft_trainer.py", "--config", config_path]

        logger.info(f"Starting training with config: {config_path}")
        logger.info("This may take several minutes...")
        logger.info("Note: ICA computation will add extra time at the beginning")

        # Run the training
        fnsft_main()

        logger.info("Experiment B completed successfully!")
        logger.info(f"Results saved to: {output_dir}")
        return True

    except Exception as e:
        logger.error(f"Experiment B failed with error: {str(e)}")
        return False
    finally:
        # Restore original argv
        sys.argv = original_argv


if __name__ == "__main__":
    success = run_experiment_b()
    if success:
        print("\n✅ Experiment B completed successfully!")
        print("📁 Check experiment_b_peft_ica/output/ for results")
        print("\n💡 To run evaluation comparing both models, use:")
        print("   python evaluate_models.py")
        print("   (Note: Both experiments A and B must be completed first)")
    else:
        print("\n❌ Experiment B failed!")
        print("📋 Check experiment_b_peft_ica/output/experiment_b.log for details")
        sys.exit(1)
