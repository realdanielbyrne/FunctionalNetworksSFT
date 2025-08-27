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
import yaml

from functionalnetworkssft.fnsft_trainer import main as fnsft_main

# Ensure output directory exists before configuring logging
os.makedirs("experiments/peft_vs_peft-ica/experiment_b_peft_ica/output", exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(
            "experiments/peft_vs_peft-ica/experiment_b_peft_ica/output/experiment_b.log"
        ),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def run_experiment_b():
    """
    Run Experiment B: PEFT + ICA masking fine-tuning
    """
    # Set up the configuration file path
    config_path = "experiments/peft_vs_peft-ica/experiment_b_peft_ica/config/experiment_b_config.yaml"

    # Load config to get actual values
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    logger.info("=" * 80)
    logger.info("EXPERIMENT B: PEFT + ICA MASKING FINE-TUNING")
    logger.info("=" * 80)
    logger.info(
        f"Model: {config.get('model_name_or_path', 'meta-llama/Llama-3.2-1B-Instruct')}"
    )
    logger.info(
        f"Dataset: {config.get('dataset_name_or_path', 'Amod/mental_health_counseling_conversations')}"
    )
    logger.info("Method: PEFT (LoRA) + ICA masking")
    logger.info(f"Epochs: {config.get('num_train_epochs', 2)}")
    logger.info(f"ICA Masking: ENABLED ({config.get('mask_mode', 'lesion')} mode)")
    logger.info(f"ICA Components: {config.get('ica_components', 20)}")
    logger.info(f"ICA Percentile: {config.get('ica_percentile', 98.0)}")
    logger.info("Data Preprocessing: Context <= 1500 chars, Response <= 4000 chars")
    logger.info("=" * 80)

    # Verify configuration file exists
    if not os.path.exists(config_path):
        logger.error(f"Configuration file not found: {config_path}")
        return False

    # Note: Using HuggingFace dataset, no local file validation needed

    # Set output directory
    output_dir = "experiments/peft_vs_peft-ica/experiment_b_peft_ica/output"

    # Simulate command line arguments for the fnsft_trainer
    original_argv = sys.argv.copy()
    try:
        sys.argv = ["fnsft_trainer.py", "--config", config_path]

        logger.info(f"Starting training with config: {config_path}")

        # Run the training with experiment-specific log file
        log_file_path = (
            "experiments/peft_vs_peft-ica/experiment_b_peft_ica/output/experiment_b.log"
        )
        fnsft_main(log_file=log_file_path)

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
        print("\nExperiment B completed successfully!")
        print(
            "Check experiments/peft_vs_peft-ica/experiment_b_peft_ica/output/ for results"
        )
        print("\nTo run evaluation comparing both models, use:")
        print("   python experiments/peft_vs_peft-ica/evaluate_models.py")
        print("   (Note: Both experiments A and B must be completed first)")
    else:
        print("\nExperiment B failed!")
        print(
            "Check experiments/peft_vs_peft-ica/experiment_b_peft_ica/output/experiment_b.log for details"
        )
        sys.exit(1)
