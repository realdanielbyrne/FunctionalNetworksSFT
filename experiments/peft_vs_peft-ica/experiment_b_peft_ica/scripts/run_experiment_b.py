#!/usr/bin/env python3
"""
Experiment B: PEFT + ICA Masking Fine-Tuning Script

This script runs Experiment B once with the ICA component configuration
specified in the experiment_b_config.yaml file in lesion mask mode.

Usage:
    python experiment_b_peft_ica/scripts/run_experiment_b.py

The script will:
1. Load the configuration from experiment_b_config.yaml
2. Run the fnsft_trainer with PEFT+ICA settings using the configured component IDs
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


def update_config_with_dynamic_names(config_path):
    """
    Update config file with dynamic names based on the configured component IDs.

    Args:
        config_path: Path to the configuration file

    Returns:
        Updated config dictionary
    """
    # Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Get component IDs from config
    component_ids = config.get("ica_component_ids", [0, 1])

    # Update hub commit message
    components_str = ",".join(map(str, component_ids))
    config["hub_commit_message"] = (
        f"Experiment B - Components [{components_str}] masked"
    )

    # Update wandb run name to reflect component configuration
    wandb_components_str = "_".join(map(str, component_ids))
    config["wandb_run_name"] = f"exp_b_components_{wandb_components_str}"

    return config


def run_single_experiment(config):
    """
    Run a single experiment with the configured component IDs.

    Args:
        config: Configuration dictionary

    Returns:
        bool: True if successful, False otherwise
    """
    component_ids = config.get("ica_component_ids", [0, 1])
    components_str = ",".join(map(str, component_ids))

    logger.info("=" * 80)
    logger.info(f"EXPERIMENT B: COMPONENTS [{components_str}] MASKED")
    logger.info("=" * 80)

    # Create temporary config file with updated names
    temp_config_path = (
        "experiments/peft_vs_peft-ica/experiment_b_peft_ica/config/temp_config.yaml"
    )
    with open(temp_config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    # Simulate command line arguments for the fnsft_trainer
    original_argv = sys.argv.copy()
    try:
        sys.argv = ["fnsft_trainer.py", "--config", temp_config_path]

        logger.info(f"Starting training with config: {temp_config_path}")

        # Run the training
        log_file_path = (
            "experiments/peft_vs_peft-ica/experiment_b_peft_ica/output/experiment_b.log"
        )
        fnsft_main(log_file=log_file_path)

        logger.info("Experiment B completed successfully!")
        logger.info(f"Results saved to: {config.get('output_dir')}")
        return True

    except Exception as e:
        logger.error(f"Experiment B failed with error: {str(e)}")
        return False
    finally:
        # Restore original argv
        sys.argv = original_argv
        # Clean up temporary config file
        try:
            if os.path.exists(temp_config_path):
                os.remove(temp_config_path)
        except Exception as e:
            logger.warning(
                f"Failed to clean up temporary config {temp_config_path}: {e}"
            )


def run_experiment_b():
    """
    Run Experiment B: PEFT + ICA masking fine-tuning with configured component IDs
    """
    # Set up the configuration file path
    config_path = "experiments/peft_vs_peft-ica/experiment_b_peft_ica/config/experiment_b_config.yaml"

    # Verify configuration file exists
    if not os.path.exists(config_path):
        logger.error(f"Configuration file not found: {config_path}")
        return False

    # Load and update config with dynamic names
    config = update_config_with_dynamic_names(config_path)
    component_ids = config.get("ica_component_ids", [0, 1])
    components_str = ",".join(map(str, component_ids))

    logger.info(f"Components to mask: [{components_str}]")

    # Run the experiment
    success = run_single_experiment(config)

    # Final summary
    logger.info("\n" + "=" * 80)
    logger.info("EXPERIMENT B SUMMARY")
    logger.info("=" * 80)
    if success:
        logger.info("Experiment completed successfully")
    else:
        logger.error("Experiment failed")
    logger.info("=" * 80)

    return success


if __name__ == "__main__":
    success = run_experiment_b()
    if success:
        print("\n" + "=" * 80)
        print("EXPERIMENT B COMPLETED SUCCESSFULLY!")

        # Load config to show which components were used
        config_path = "experiments/peft_vs_peft-ica/experiment_b_peft_ica/config/experiment_b_config.yaml"
        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
            component_ids = config.get("ica_component_ids", [0, 1])
            components_str = ",".join(map(str, component_ids))
            print(f"\nConfiguration completed:")
            print(f"  Components [{components_str}] masked in lesion mode")
        except Exception as e:
            print(f"\nNote: Could not read component configuration: {e}")

        print("\nTo run evaluation comparing models, use:")
        print("   python experiments/peft_vs_peft-ica/evaluate_models.py")
        print("   (Note: Both experiments A and B must be completed first)")
        print("=" * 80)
    else:
        print("\n" + "=" * 80)
        print("!!! EXPERIMENT B FAILED!")
        print("=" * 80)
        sys.exit(1)
