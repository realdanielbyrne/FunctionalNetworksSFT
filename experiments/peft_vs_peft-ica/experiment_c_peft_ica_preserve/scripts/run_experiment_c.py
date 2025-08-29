#!/usr/bin/env python3
"""
Experiment C: PEFT + ICA Masking Fine-Tuning Script (Preserve Mode)

This script runs Experiment C three times sequentially with different ICA component
masking configurations in preserve mask mode:
- Run 1: [0] (preserve only component 0)
- Run 2: [0, 1] (preserve components 0 and 1)
- Run 3: [0, 1, 2] (preserve components 0, 1, and 2)

Usage:
    python experiment_c_peft_ica_preserve/scripts/run_experiment_c.py

The script will:
1. Load the base configuration from experiment_c_config.yaml
2. Run the fnsft_trainer with PEFT+ICA settings for each component configuration
3. Save results to experiment_c_peft_ica_preserve/output/
"""

import os
import sys
import logging
import yaml

from functionalnetworkssft.fnsft_trainer import main as fnsft_main

# Ensure output directory exists before configuring logging
os.makedirs("experiments/peft_vs_peft-ica/experiment_c_peft_ica_preserve/output", exist_ok=True)

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


def create_config_for_components(base_config_path, component_ids, run_number):
    """
    Create a temporary config file with specific component IDs and updated names for preserve mode.

    Args:
        base_config_path: Path to the base configuration file
        component_ids: List of component IDs to preserve
        run_number: Run number for naming

    Returns:
        Path to the temporary config file
    """
    # Load base config
    with open(base_config_path, "r") as f:
        config = yaml.safe_load(f)

    # Ensure preserve mode
    config["mask_mode"] = "preserve"

    # Update component IDs
    config["ica_component_ids"] = component_ids

    # Update hub commit message (explicitly reference preserve mode)
    components_str = ",".join(map(str, component_ids))
    config["hub_commit_message"] = (
        f"Experiment C (Preserve Mode) - Components [{components_str}] preserved"
    )

    # Update wandb run name to reflect component configuration
    wandb_components_str = "_".join(map(str, component_ids))
    config["wandb_run_name"] = f"exp_c_preserve_components_{wandb_components_str}"

    # Update output directory to be run-specific
    base_output = config.get(
        "output_dir", "experiments/peft_vs_peft-ica/experiment_c_peft_ica_preserve/output"
    )
    config["output_dir"] = (
        f"{base_output}/run_{run_number}_components_{'_'.join(map(str, component_ids))}"
    )

    # Create temporary config file
    temp_config_path = f"experiments/peft_vs_peft-ica/experiment_c_peft_ica_preserve/config/temp_config_run_{run_number}.yaml"
    with open(temp_config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    return temp_config_path


def run_single_experiment(config_path, component_ids, run_number):
    """
    Run a single experiment with specific component configuration (preserve mode).

    Args:
        config_path: Path to the configuration file
        component_ids: List of component IDs to preserve
        run_number: Run number for logging

    Returns:
        bool: True if successful, False otherwise
    """
    components_str = ",".join(map(str, component_ids))

    logger.info("=" * 80)
    logger.info(
        f"EXPERIMENT C (PRESERVE) - RUN {run_number}: COMPONENTS [{components_str}] PRESERVED"
    )
    logger.info("=" * 80)

    # Load config to get actual values for logging
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    logger.info(
        f"Model: {config.get('model_name_or_path', 'meta-llama/Llama-3.2-1B-Instruct')}"
    )
    logger.info(
        f"Dataset: {config.get('dataset_name_or_path', 'databricks/databricks-dolly-15k')}"
    )
    logger.info("Method: PEFT (LoRA) + ICA masking")
    logger.info(f"Epochs: {config.get('num_train_epochs', 2)}")
    logger.info(f"ICA Masking: ENABLED ({config.get('mask_mode', 'preserve')} mode)")
    logger.info(f"ICA Components: {config.get('ica_components', 10)}")
    logger.info(f"ICA Percentile: {config.get('ica_percentile', 98.0)}")
    logger.info(f"Preserved Component IDs: {component_ids}")
    logger.info(f"Output Directory: {config.get('output_dir')}")
    logger.info("=" * 80)

    # Simulate command line arguments for the fnsft_trainer
    original_argv = sys.argv.copy()
    try:
        sys.argv = ["fnsft_trainer.py", "--config", config_path]

        logger.info(f"Starting training run {run_number} with config: {config_path}")

        # Run the training with run-specific log file
        log_file_path = (
            f"experiments/peft_vs_peft-ica/experiment_c_peft_ica_preserve/output/experiment_c_preserve_run_{run_number}.log"
        )
        fnsft_main(log_file=log_file_path)

        logger.info(f"Experiment C Preserve Run {run_number} completed successfully!")
        logger.info(f"Results saved to: {config.get('output_dir')}")
        return True

    except Exception as e:
        logger.error(f"Experiment C Preserve Run {run_number} failed with error: {str(e)}")
        return False
    finally:
        # Restore original argv
        sys.argv = original_argv


def run_experiment_c_preserve():
    """
    Run Experiment C (Preserve Mode): PEFT + ICA masking fine-tuning with multiple component configurations
    """
    # Define the component configurations to test (exactly 3)
    component_configurations = [
        [0],  # Run 1: preserve only component 0
        [0, 1],  # Run 2: preserve components 0 and 1
        [0, 1, 2],  # Run 3: preserve components 0, 1, and 2
    ]

    # Set up the base configuration file path
    base_config_path = "experiments/peft_vs_peft-ica/experiment_c_peft_ica_preserve/config/experiment_c_config.yaml"

    # Verify base configuration file exists
    if not os.path.exists(base_config_path):
        logger.error(f"Base configuration file not found: {base_config_path}")
        return False

    logger.info("=" * 80)
    logger.info("EXPERIMENT C (PRESERVE): MULTI-RUN PEFT + ICA MASKING FINE-TUNING")
    logger.info("=" * 80)
    logger.info(f"Total runs planned: {len(component_configurations)}")
    for i, components in enumerate(component_configurations, 1):
        components_str = ",".join(map(str, components))
        logger.info(f"  Run {i}: Components [{components_str}] preserved")
    logger.info("=" * 80)

    successful_runs = 0
    failed_runs = 0
    temp_config_files = []

    try:
        for run_number, component_ids in enumerate(component_configurations, 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"PREPARING RUN {run_number}/{len(component_configurations)}")
            logger.info(f"{'='*60}")

            # Create temporary config for this run
            temp_config_path = create_config_for_components(
                base_config_path, component_ids, run_number
            )
            temp_config_files.append(temp_config_path)

            # Run the experiment
            success = run_single_experiment(temp_config_path, component_ids, run_number)

            if success:
                successful_runs += 1
                logger.info(f"✓ Run {run_number} completed successfully")
            else:
                failed_runs += 1
                logger.error(f"✗ Run {run_number} failed")
                # Continue with next run even if this one failed

        # Final summary
        logger.info("\n" + "=" * 80)
        logger.info("EXPERIMENT C (PRESERVE) MULTI-RUN SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Total runs: {len(component_configurations)}")
        logger.info(f"Successful runs: {successful_runs}")
        logger.info(f"Failed runs: {failed_runs}")
        logger.info("=" * 80)

        return failed_runs == 0  # Return True only if all runs succeeded

    finally:
        # Clean up temporary config files
        for temp_file in temp_config_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                    logger.debug(f"Cleaned up temporary config: {temp_file}")
            except Exception as e:
                logger.warning(f"Failed to clean up temporary config {temp_file}: {e}")


if __name__ == "__main__":
    success = run_experiment_c_preserve()
    if success:
        print("\n" + "=" * 80)
        print("🎉 ALL EXPERIMENT C (PRESERVE) RUNS COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print("Results saved to:")
        print("  • experiments/peft_vs_peft-ica/experiment_c_peft_ica_preserve/output/")
        print("  • Individual run directories for each component configuration")
        print("\nRun configurations completed:")
        component_configurations = [
            [0],
            [0, 1],
            [0, 1, 2],
        ]
        for i, components in enumerate(component_configurations, 1):
            components_str = ",".join(map(str, components))
            print(f"  ✓ Run {i}: Components [{components_str}] preserved")
        print("\nTo run evaluation comparing models, use:")
        print("   python experiments/peft_vs_peft-ica/evaluate_models.py")
        print("   (Note: Both experiments A, B and C must be completed first)")
        print("=" * 80)
    else:
        print("\n" + "=" * 80)
        print("❌ EXPERIMENT C (PRESERVE) FAILED!")
        print("=" * 80)
        print("Check the following log files for details:")
        print(
            "  • experiments/peft_vs_peft-ica/experiment_c_peft_ica_preserve/output/experiment_c_preserve_run_*.log"
        )
        print("=" * 80)
        sys.exit(1)

