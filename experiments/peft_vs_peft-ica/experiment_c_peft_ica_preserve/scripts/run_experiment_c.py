#!/usr/bin/env python3
"""
Experiment C: PEFT + ICA Masking Fine-Tuning Script (Preserve Mode)

This script runs Experiment C once with the ICA component configuration
specified in the experiment_c_config.yaml file in preserve mask mode.

Usage:
    python experiment_c_peft_ica_preserve/scripts/run_experiment_c.py

The script will:
1. Load the configuration from experiment_c_config.yaml
2. Run the fnsft_trainer with PEFT+ICA settings using the configured component IDs
3. Save results to experiment_c_peft_ica_preserve/output/
"""

import os
import sys
import logging
import yaml

from functionalnetworkssft.fnsft_trainer import main as fnsft_main

# Ensure output directory exists before configuring logging
os.makedirs(
    "experiments/peft_vs_peft-ica/experiment_c_peft_ica_preserve/output", exist_ok=True
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(
            "experiments/peft_vs_peft-ica/experiment_c_peft_ica_preserve/output/experiment_c_preserve.log"
        ),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def update_config_with_dynamic_names(config_path):
    """
    Update config file with dynamic names based on the configured component IDs for preserve mode.

    Args:
        config_path: Path to the configuration file

    Returns:
        Updated config dictionary
    """
    # Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Ensure preserve mode
    config["mask_mode"] = "preserve"

    # Get component IDs from config
    component_ids = config.get("ica_component_ids", [0, 1])

    # Update hub commit message (explicitly reference preserve mode)
    components_str = ",".join(map(str, component_ids))
    config["hub_commit_message"] = (
        f"Experiment C (Preserve Mode) - Components [{components_str}] preserved"
    )

    # Update wandb run name to reflect component configuration
    wandb_components_str = "_".join(map(str, component_ids))
    config["wandb_run_name"] = f"exp_c_preserve_components_{wandb_components_str}"

    return config


def run_single_experiment(config):
    """
    Run a single experiment with the configured component IDs (preserve mode).

    Args:
        config: Configuration dictionary

    Returns:
        bool: True if successful, False otherwise
    """
    component_ids = config.get("ica_component_ids", [0, 1])
    components_str = ",".join(map(str, component_ids))

    logger.info("=" * 80)
    logger.info(f"EXPERIMENT C (PRESERVE): COMPONENTS [{components_str}] PRESERVED")
    logger.info("=" * 80)

    # Create temporary config file with updated names
    temp_config_path = "experiments/peft_vs_peft-ica/experiment_c_peft_ica_preserve/config/temp_config.yaml"
    with open(temp_config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    # Simulate command line arguments for the fnsft_trainer
    original_argv = sys.argv.copy()
    try:
        sys.argv = ["fnsft_trainer.py", "--config", temp_config_path]

        logger.info(f"Starting training with config: {temp_config_path}")

        # Run the training
        log_file_path = "experiments/peft_vs_peft-ica/experiment_c_peft_ica_preserve/output/experiment_c_preserve.log"
        fnsft_main(log_file=log_file_path)

        logger.info("Experiment C Preserve completed successfully!")
        logger.info(f"Results saved to: {config.get('output_dir')}")
        return True

    except Exception as e:
        logger.error(f"Experiment C Preserve failed with error: {str(e)}")
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


def run_experiment_c_preserve():
    """
    Run Experiment C (Preserve Mode): PEFT + ICA masking fine-tuning with configured component IDs
    """
    # Set up the configuration file path
    config_path = "experiments/peft_vs_peft-ica/experiment_c_peft_ica_preserve/config/experiment_c_config.yaml"

    # Verify configuration file exists
    if not os.path.exists(config_path):
        logger.error(f"Configuration file not found: {config_path}")
        return False

    # Load and update config with dynamic names
    config = update_config_with_dynamic_names(config_path)
    component_ids = config.get("ica_component_ids", [0, 1])
    components_str = ",".join(map(str, component_ids))

    logger.info(f"Components to preserve: [{components_str}]")

    # Run the experiment
    success = run_single_experiment(config)

    # Final summary
    logger.info("\n" + "=" * 80)
    logger.info("EXPERIMENT C (PRESERVE) SUMMARY")
    logger.info("=" * 80)
    if success:
        logger.info("Experiment completed successfully")
    else:
        logger.error("!!! Experiment failed")
    logger.info("=" * 80)

    return success


if __name__ == "__main__":
    success = run_experiment_c_preserve()
    if success:
        print("\n" + "=" * 80)
        print("EXPERIMENT C (PRESERVE) COMPLETED SUCCESSFULLY!")
        print("=" * 80)

        # Load config to show which components were used
        config_path = "experiments/peft_vs_peft-ica/experiment_c_peft_ica_preserve/config/experiment_c_config.yaml"
        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
            component_ids = config.get("ica_component_ids", [0, 1])
            components_str = ",".join(map(str, component_ids))
            print(f"\nConfiguration completed:")
            print(f"  Components [{components_str}] preserved in preserve mode")
        except Exception as e:
            print(f"\nNote: Could not read component configuration: {e}")

        print("\nTo run evaluation comparing models, use:")
        print("   python experiments/peft_vs_peft-ica/evaluate_models.py")
        print("   (Note: Both experiments A, B and C must be completed first)")
        print("=" * 80)
    else:
        print("\n" + "=" * 80)
        print("!!EXPERIMENT C (PRESERVE) FAILED!")

        sys.exit(1)
