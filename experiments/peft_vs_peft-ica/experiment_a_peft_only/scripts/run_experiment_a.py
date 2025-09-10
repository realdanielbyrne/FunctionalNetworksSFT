#!/usr/bin/env python3
"""
Experiment A: PEFT-Only Fine-Tuning Script

This script runs Experiment A, which fine-tunes meta-llama/Llama-3.2-1B-Instruct
using PEFT (LoRA) only, without ICA masking.

Usage:
    python experiment_a_peft_only/scripts/run_experiment_a.py

The script will:
1. Load the configuration from experiment_a_config.yaml
2. Run the fnsft_trainer with PEFT-only settings
3. Save results to experiment_a_peft_only/output/
"""

import os
import sys
import logging

from functionalnetworkssft.fnsft_trainer import main as fnsft_main

# Ensure output directory exists before configuring logging
os.makedirs("experiments/peft_vs_peft-ica/experiment_a_peft_only/output", exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(
            "experiments/peft_vs_peft-ica/experiment_a_peft_only/output/experiment_a.log"
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
    logger.info("Dataset: Amod/mental_health_counseling_conversations")
    logger.info("Method: PEFT (LoRA) only")
    logger.info("Epochs: 2")
    logger.info("ICA Masking: DISABLED")
    logger.info("Data Preprocessing: Context <= 1500 chars, Response <= 4000 chars")
    logger.info("=" * 80)

    # Set up the configuration file path (relative to project root)
    config_path = "experiments/peft_vs_peft-ica/experiment_a_peft_only/config/experiment_a_config.yaml"

    # Verify configuration file exists
    if not os.path.exists(config_path):
        logger.error(f"Configuration file not found: {config_path}")
        return False

    # Note: Using HuggingFace dataset, no local file validation needed

    # Set output directory
    output_dir = "experiments/peft_vs_peft-ica/experiment_a_peft_only/output"

    # Simulate command line arguments for the fnsft_trainer
    original_argv = sys.argv.copy()
    try:
        sys.argv = ["fnsft_trainer.py", "--config", config_path]

        logger.info(f"Starting training with config: {config_path}")

        # Run the training with experiment-specific log file
        log_file_path = "experiments/peft_vs_peft-ica/experiment_a_peft_only/output/experiment_a.log"
        fnsft_main(log_file=log_file_path)

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
        print("\nExperiment A completed successfully!")
    else:
        print("\n!!! Experiment A failed!")
        sys.exit(1)
