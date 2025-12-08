#!/usr/bin/env python3
"""
Fine-Tuning Experiment Runner

This script runs comprehensive fine-tuning experiments comparing three approaches:
- Experiment A: PEFT (LoRA) only
- Experiment B: PEFT (LoRA) + ICA masking (lesion mode)
- Experiment C: PEFT (LoRA) + ICA masking (preserve mode)

Configuration Architecture:
- All shared parameters are defined in common_config.yaml (single source of truth)
- Experiment-specific differences (ICA settings) are defined in Python code
- Merged configs are created dynamically for each experiment run

Usage:
    python experiments/peft_vs_peft-ica/run_experiments.py [--experiment {a,b,c,all}] [--verbose]

Examples:
python experiments/peft_vs_peft-ica/run_experiments.py --experiment a
python experiments/peft_vs_peft-ica/run_experiments.py --experiment b
python experiments/peft_vs_peft-ica/run_experiments.py --experiment c
python experiments/peft_vs_peft-ica/run_experiments.py --experiment all
python experiments/peft_vs_peft-ica/run_experiments.py
"""

import argparse
import logging
import os
import sys
import time
import yaml
from datetime import datetime
from pathlib import Path
from copy import deepcopy

try:
    import torch
except ImportError:
    torch = None


# =============================================================================
# EXPERIMENT DEFINITIONS
# =============================================================================
# Only the parameters that DIFFER between experiments are defined here.
# All shared parameters are in common_config.yaml (single source of truth).
# =============================================================================

# Experiment A: PEFT only (no ICA masking)
EXPERIMENT_A_OVERRIDES = {
    "output_dir": "experiments/peft_vs_peft-ica/experiment_a_peft_only/output",
    "wandb_project": "VibeThinker-1.5B-Physics-PEFT-Only",
    "hub_repo_id": "realdanielbyrne/VibeThinker-1.5B-Physics",
    "hub_commit_message": "1 Epoch Physics, PEFT Only",
    "mask_mode": None,  # No ICA masking
}

# Experiment B: PEFT + ICA masking (lesion mode)
EXPERIMENT_B_OVERRIDES = {
    "output_dir": "experiments/peft_vs_peft-ica/experiment_b_peft_ica/output",
    "wandb_project": "VibeThinker-1.5B-Physics-Fnsft-PEFT+ICA-Lesion",
    "hub_repo_id": "realdanielbyrne/VibeThinker-1.5B-Instruct-Physics-PEFT+ICA-Lesion",
    "hub_commit_message": "1 Epoch Physics, Components [0,1], Lesion",
    "mask_mode": "lesion",
    "lora_target_modules": ["down_proj"],
    "ica_template_path": "ica_templates/vibethinker-1.5B/camel-ai_physics/global_templates.json",
    "ica_components": 5,
    "ica_percentile": 98.0,
    "ica_component_ids": [0, 1],
}

# Experiment C: PEFT + ICA masking (preserve mode)
EXPERIMENT_C_OVERRIDES = {
    "output_dir": "experiments/peft_vs_peft-ica/experiment_c_peft_ica_preserve/output",
    "wandb_project": "VibeThinker1.5B-Physics-PEFT+ICA-Preserve",
    "hub_repo_id": "realdanielbyrne/VibeThinker-1.5B-Physics-Fnsft-PEFT+ICA-Preserve",
    "hub_commit_message": "1 Epoch, Physics, Components [0,1] preserved",
    "mask_mode": "preserve",
    "lora_target_modules": ["down_proj"],
    "ica_template_path": "ica_templates/vibethinker-1.5B/camel-ai_physics/global_templates.json",
    "ica_components": 5,
    "ica_percentile": 98.0,
    "ica_component_ids": [0, 1],
}

# Map experiment names to their overrides
EXPERIMENT_OVERRIDES = {
    "a": EXPERIMENT_A_OVERRIDES,
    "b": EXPERIMENT_B_OVERRIDES,
    "c": EXPERIMENT_C_OVERRIDES,
}

# Base directory for experiments
BASE_OUTPUT_DIR = "experiments/peft_vs_peft-ica"


def find_last_checkpoint(output_dir):
    """Return path to the latest checkpoint directory in output_dir, or None."""
    if not os.path.isdir(output_dir):
        return None

    checkpoints = []
    for name in os.listdir(output_dir):
        path = os.path.join(output_dir, name)
        if not os.path.isdir(path):
            continue
        if not name.startswith("checkpoint-"):
            continue
        try:
            step = int(name.split("-")[-1])
        except ValueError:
            continue
        checkpoints.append((step, path))

    if not checkpoints:
        return None

    checkpoints.sort(key=lambda item: item[0])
    return checkpoints[-1][1]


def create_experiment_config(experiment_name):
    """
    Create experiment-specific configuration by merging common_config.yaml with experiment overrides.

    Configuration Loading Order:
    1. common_config.yaml - All shared parameters (single source of truth)
    2. Experiment-specific overrides - Only ICA/masking differences defined in Python

    Args:
        experiment_name: Name of experiment ('a', 'b', or 'c')

    Returns:
        Tuple of (merged_config_dict, temp_config_path)
    """
    logger = logging.getLogger(__name__)

    if experiment_name not in EXPERIMENT_OVERRIDES:
        raise ValueError(
            f"Unknown experiment: {experiment_name}. Must be 'a', 'b', or 'c'"
        )

    # Path to common configuration (single source of truth)
    common_config_path = f"{BASE_OUTPUT_DIR}/common_config.yaml"

    # Temp config path for this experiment
    temp_config_path = f"{BASE_OUTPUT_DIR}/temp_config_{experiment_name}.yaml"

    # Step 1: Load common configuration
    if not os.path.exists(common_config_path):
        logger.error(f"Common config not found: {common_config_path}")
        raise FileNotFoundError(
            f"Required common config file not found: {common_config_path}"
        )

    logger.info(f"Loading common config from {common_config_path}")
    with open(common_config_path, "r") as f:
        merged_config = yaml.safe_load(f) or {}

    # Step 2: Apply experiment-specific overrides
    experiment_overrides = EXPERIMENT_OVERRIDES[experiment_name]
    logger.info(
        f"Applying experiment {experiment_name.upper()} overrides: mask_mode={experiment_overrides.get('mask_mode')}"
    )
    merged_config.update(experiment_overrides)

    return merged_config, temp_config_path


def clear_cuda_memory():
    """Clear CUDA memory cache to free up GPU memory between experiments."""
    if torch is not None and torch.cuda.is_available():
        logger = logging.getLogger(__name__)
        torch.cuda.empty_cache()
        logger.info(f"CUDA memory cleared:")

    else:
        logger = logging.getLogger(__name__)
        logger.debug("CUDA not available or torch not imported - skipping memory clear")


def setup_logging(verbose=False):
    """Set up logging configuration"""
    log_level = logging.DEBUG if verbose else logging.INFO
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Create logs directory
    os.makedirs("../logs", exist_ok=True)

    # Create log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"../logs/experiment_run_{timestamp}.log"

    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    return logging.getLogger(__name__)


def run_experiment(experiment_name, resume_from_last_checkpoint=False):
    """
    Run a single experiment by merging common config with experiment-specific overrides.

    Args:
        experiment_name: Name of experiment ('a', 'b', or 'c')

    Returns:
        bool: True if experiment succeeded, False otherwise
    """
    from functionalnetworkssft.fnsft_trainer import main as fnsft_main

    logger = logging.getLogger(__name__)
    experiment_labels = {
        "a": "Experiment A: PEFT-only fine-tuning",
        "b": "Experiment B: PEFT + ICA masking (lesion mode)",
        "c": "Experiment C: PEFT + ICA masking (preserve mode)",
    }

    logger.info(
        f"Starting {experiment_labels.get(experiment_name, f'Experiment {experiment_name}')}"
    )

    # Save current directory and change to project root
    original_cwd = os.getcwd()
    project_root = Path(__file__).parent.parent.parent
    os.chdir(project_root)

    temp_config_path = None
    try:
        # Create merged configuration
        merged_config, temp_config_path = create_experiment_config(experiment_name)

        if resume_from_last_checkpoint:
            output_dir = merged_config.get("output_dir", BASE_OUTPUT_DIR)
            last_checkpoint = find_last_checkpoint(output_dir)
            if last_checkpoint:
                merged_config["resume_from_checkpoint"] = last_checkpoint
                logger.info(
                    f"Resuming {experiment_name.upper()} from last checkpoint: {last_checkpoint}"
                )
            else:
                logger.warning(
                    f"No checkpoints found in {output_dir}; starting {experiment_name.upper()} from scratch"
                )

        # Write merged config to temporary file
        os.makedirs(os.path.dirname(temp_config_path), exist_ok=True)
        with open(temp_config_path, "w") as f:
            yaml.dump(merged_config, f, default_flow_style=False, sort_keys=False)
        logger.info(f"Created merged config at {temp_config_path}")

        # Set up sys.argv for fnsft_main
        original_argv = sys.argv.copy()
        sys.argv = ["fnsft_trainer.py", "--config", temp_config_path]

        # Experiment-specific log file
        output_dir = merged_config.get("output_dir", BASE_OUTPUT_DIR)
        os.makedirs(output_dir, exist_ok=True)
        log_file_path = os.path.join(output_dir, f"experiment_{experiment_name}.log")

        start_time = time.time()

        # Run training
        fnsft_main(log_file=log_file_path)

        end_time = time.time()
        duration = end_time - start_time
        logger.info(
            f"Experiment {experiment_name.upper()} completed in {duration:.2f} seconds"
        )

        # Restore sys.argv
        sys.argv = original_argv

        # Clear CUDA memory after experiment
        clear_cuda_memory()

        return True

    except Exception as e:
        logger.error(f"Experiment {experiment_name.upper()} failed: {str(e)}")
        import traceback

        logger.error(traceback.format_exc())
        # Clear CUDA memory even on failure
        clear_cuda_memory()
        return False

    finally:
        # Clean up temporary config
        if temp_config_path:
            try:
                if os.path.exists(temp_config_path):
                    os.remove(temp_config_path)
            except Exception as e:
                logger.warning(
                    f"Failed to clean up temp config {temp_config_path}: {e}"
                )
        # Restore original directory
        os.chdir(original_cwd)


def run_experiment_a(resume_from_last_checkpoint=False):
    """Run Experiment A: PEFT-only fine-tuning"""
    return run_experiment("a", resume_from_last_checkpoint=resume_from_last_checkpoint)


def run_experiment_b(resume_from_last_checkpoint=False):
    """Run Experiment B: PEFT + ICA masking fine-tuning (lesion mode)"""
    return run_experiment("b", resume_from_last_checkpoint=resume_from_last_checkpoint)


def run_experiment_c(resume_from_last_checkpoint=False):
    """Run Experiment C: PEFT + ICA masking fine-tuning (preserve mode)"""
    return run_experiment("c", resume_from_last_checkpoint=resume_from_last_checkpoint)


def run_evaluation():
    """Run model evaluation after experiments complete."""
    logger = logging.getLogger(__name__)
    logger.info("Starting model evaluation...")

    try:
        # Import and run evaluation
        sys.path.insert(0, str(Path(__file__).parent))
        from evaluate_models import main as evaluate_main

        # Temporarily modify sys.argv for evaluation
        original_argv = sys.argv.copy()
        sys.argv = [
            "evaluate_models.py",
            "--test-size",
            "0.2",
            "--output-dir",
            "experiments/peft_vs_peft-ica/evaluation_results",
        ]

        start_time = time.time()
        evaluate_main()
        end_time = time.time()

        duration = end_time - start_time
        logger.info(f"Model evaluation completed in {duration:.2f} seconds")

        # Clear CUDA memory after evaluation
        clear_cuda_memory()

        # Restore original argv
        sys.argv = original_argv

        return True
    except Exception as e:
        logger.error(f"Model evaluation failed: {str(e)}")
        # Clear CUDA memory even on failure
        clear_cuda_memory()
        return False


def print_experiment_summary():
    """Print a summary of the experiment setup using the centralized configuration."""
    # Load common config (single source of truth for shared parameters)
    common_config_path = f"{BASE_OUTPUT_DIR}/common_config.yaml"

    common_config = {}
    try:
        with open(common_config_path, "r") as f:
            common_config = yaml.safe_load(f) or {}
    except (FileNotFoundError, yaml.YAMLError) as e:
        print(f"Warning: Could not load common config ({common_config_path}): {e}")

    # Get experiment-specific overrides
    exp_a = EXPERIMENT_A_OVERRIDES
    exp_b = EXPERIMENT_B_OVERRIDES
    exp_c = EXPERIMENT_C_OVERRIDES

    print("=" * 80)
    print("FINE-TUNING EXPERIMENT COMPARISON")
    print("=" * 80)
    print(f"Model: {common_config.get('model_name_or_path', '[CONFIG NOT LOADED]')}")
    print(
        f"Dataset: {common_config.get('dataset_name_or_path', '[CONFIG NOT LOADED]')}"
    )
    print(
        f"Training Epochs: {common_config.get('num_train_epochs', '[CONFIG NOT LOADED]')}"
    )
    print(f"LoRA rank: {common_config.get('lora_r', '[CONFIG NOT LOADED]')}")
    print(f"LoRA alpha: {common_config.get('lora_alpha', '[CONFIG NOT LOADED]')}\n")

    print("Data Preprocessing (identical across all experiments):")
    print(f"  - Response max length: {common_config.get('response_max_length', 'N/A')}")
    print(
        f"  - Instruction max length: {common_config.get('instruction_max_length', 'N/A')}\n"
    )

    print("Experiment A: PEFT (LoRA) only")
    print(f"  - ICA masking: DISABLED")
    print(f"  - Output: {exp_a.get('output_dir')}\n")

    print("Experiment B: PEFT (LoRA) + ICA masking (lesion)")
    print(f"  - ICA masking: ENABLED ({exp_b.get('mask_mode')} mode)")
    print(f"  - ICA components: {exp_b.get('ica_components')}")
    print(f"  - ICA component IDs: {exp_b.get('ica_component_ids')}")
    print(f"  - ICA percentile: {exp_b.get('ica_percentile')}")
    print(f"  - LoRA target modules: {exp_b.get('lora_target_modules')}")
    print(f"  - Output: {exp_b.get('output_dir')}\n")

    print("Experiment C: PEFT (LoRA) + ICA masking (preserve)")
    print(f"  - ICA masking: ENABLED ({exp_c.get('mask_mode')} mode)")
    print(f"  - ICA components: {exp_c.get('ica_components')}")
    print(f"  - ICA component IDs: {exp_c.get('ica_component_ids')}")
    print(f"  - ICA percentile: {exp_c.get('ica_percentile')}")
    print(f"  - LoRA target modules: {exp_c.get('lora_target_modules')}")
    print(f"  - Output: {exp_c.get('output_dir')}")
    print("=" * 80)


def main():
    """Main experiment runner"""
    parser = argparse.ArgumentParser(
        description="Run fine-tuning experiments comparing PEFT vs PEFT+ICA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--experiment",
        choices=["a", "b", "c", "all"],
        default="all",
        help="Which experiment(s) to run (default: all)",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument(
        "--resume_from_last_checkpoint",
        action="store_true",
        help="Resume each selected experiment from the latest checkpoint in its output_dir",
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
    if args.experiment in ["a", "all"]:
        logger.info("Running Experiment A...")
        results["experiment_a"] = run_experiment_a(
            resume_from_last_checkpoint=args.resume_from_last_checkpoint
        )

    if args.experiment in ["b", "all"]:
        logger.info("Running Experiment B...")
        results["experiment_b"] = run_experiment_b(
            resume_from_last_checkpoint=args.resume_from_last_checkpoint
        )

    if args.experiment in ["c", "all"]:
        logger.info("Running Experiment C...")
        results["experiment_c"] = run_experiment_c(
            resume_from_last_checkpoint=args.resume_from_last_checkpoint
        )

    # Run evaluation if selected set completed successfully
    if (
        args.experiment == "all"
        and results.get("experiment_a", False)
        and results.get("experiment_b", False)
        and results.get("experiment_c", False)
    ):
        logger.info("All experiments completed successfully. Running evaluation...")
        results["evaluation"] = run_evaluation()

    # Calculate total time
    total_time = time.time() - start_time

    # Final CUDA memory cleanup
    clear_cuda_memory()

    # Print final results
    print("\n" + "=" * 80)
    print("EXPERIMENT RESULTS")
    print("=" * 80)

    for exp_name, success in results.items():
        status = "SUCCESS" if success else "FAILED"
        print(f"{exp_name.replace('_', ' ').title()}: {status}")

    print(f"\nTotal execution time: {total_time:.2f} seconds")
    print("\nOutput locations:")
    if "experiment_a" in results:
        print("  Experiment A: experiment_a_peft_only/output/")
    if "experiment_b" in results:
        print("  Experiment B: experiment_b_peft_ica/output/")
    if "experiment_c" in results:
        print("  Experiment C: experiment_c_peft_ica_preserve/output/")
    if "evaluation" in results and results["evaluation"]:
        print("  Evaluation Results: evaluation_results/")
        print("  Summary Report: evaluation_results/evaluation_summary.md")

    # Exit with appropriate code
    if all(results.values()):
        print("\nAll experiments and evaluation completed successfully!")
        if "evaluation" in results and results["evaluation"]:
            print("Check the evaluation summary for detailed performance comparison!")
        sys.exit(0)
    else:
        print("\nSome experiments failed. Check logs for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
