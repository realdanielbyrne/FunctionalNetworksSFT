#!/usr/bin/env python3
"""
Fine-Tuning Experiment Runner

This script runs comprehensive fine-tuning experiments comparing three approaches:
- Experiment A: PEFT (LoRA) only
- Experiment B: PEFT (LoRA) + ICA masking (lesion mode)
- Experiment C: PEFT (LoRA) + ICA masking (preserve mode)

Both experiments use the meta-llama/Llama-3.2-1B-Instruct model and mental health dataset.

Data Preprocessing:
Both experiments apply consistent data preprocessing to ensure fair comparison:
- Filter Context field: Remove rows where Context length > 1500 characters
- Filter Response field: Remove rows where Response length > 4000 characters
- Applied sequentially to maintain data integrity
- Preprocessing statistics are logged for transparency

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

try:
    import torch
except ImportError:
    torch = None


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


def run_experiment_a():
    """Run Experiment A: PEFT-only fine-tuning"""
    logger = logging.getLogger(__name__)
    logger.info("Starting Experiment A: PEFT-only fine-tuning")

    # Save current directory and change to project root
    original_cwd = os.getcwd()
    project_root = Path(__file__).parent.parent.parent
    os.chdir(project_root)

    try:
        # Import and run experiment A
        sys.path.insert(
            0, str(Path(__file__).parent / "experiment_a_peft_only" / "scripts")
        )
        from run_experiment_a import run_experiment_a as run_a

        start_time = time.time()
        success = run_a()
        end_time = time.time()

        duration = end_time - start_time
        logger.info(f"Experiment A completed in {duration:.2f} seconds")

        # Clear CUDA memory after experiment
        clear_cuda_memory()

        return success
    except Exception as e:
        logger.error(f"Experiment A failed: {str(e)}")
        # Clear CUDA memory even on failure
        clear_cuda_memory()
        return False
    finally:
        # Restore original directory
        os.chdir(original_cwd)


def run_experiment_b():
    """Run Experiment B: PEFT + ICA masking fine-tuning"""
    logger = logging.getLogger(__name__)
    logger.info("Starting Experiment B: PEFT + ICA masking fine-tuning")

    try:
        # Import and run experiment B
        sys.path.insert(
            0, str(Path(__file__).parent / "experiment_b_peft_ica" / "scripts")
        )
        from run_experiment_b import run_experiment_b as run_b

        start_time = time.time()
        success = run_b()
        end_time = time.time()

        duration = end_time - start_time
        logger.info(f"Experiment B completed in {duration:.2f} seconds")

        # Clear CUDA memory after experiment
        clear_cuda_memory()

        return success
    except Exception as e:
        logger.error(f"Experiment B failed: {str(e)}")
        # Clear CUDA memory even on failure
        clear_cuda_memory()
        return False


def run_experiment_c():
    """Run Experiment C: PEFT + ICA masking fine-tuning (preserve mode)"""
    logger = logging.getLogger(__name__)
    logger.info("Starting Experiment C: PEFT + ICA masking fine-tuning (preserve mode)")

    try:
        # Import and run experiment C
        sys.path.insert(
            0,
            str(Path(__file__).parent / "experiment_c_peft_ica_preserve" / "scripts"),
        )
        from run_experiment_c import run_experiment_c_preserve as run_c

        start_time = time.time()
        success = run_c()
        end_time = time.time()

        duration = end_time - start_time
        logger.info(f"Experiment C completed in {duration:.2f} seconds")

        # Clear CUDA memory after experiment
        clear_cuda_memory()

        return success
    except Exception as e:
        logger.error(f"Experiment C failed: {str(e)}")
        # Clear CUDA memory even on failure
        clear_cuda_memory()
        return False


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
    """Print a summary of the experiment setup"""
    # Load configs to get actual values
    config_a_path = "experiments/peft_vs_peft-ica/experiment_a_peft_only/config/experiment_a_config.yaml"
    config_b_path = "experiments/peft_vs_peft-ica/experiment_b_peft_ica/config/experiment_b_config.yaml"
    config_c_path = "experiments/peft_vs_peft-ica/experiment_c_peft_ica_preserve/config/experiment_c_config.yaml"

    config_a = {}
    config_b = {}
    config_c = {}

    # Load config A
    try:
        with open(config_a_path, "r") as f:
            config_a = yaml.safe_load(f) or {}
    except (FileNotFoundError, yaml.YAMLError) as e:
        print(f"Warning: Could not load config A ({config_a_path}): {e}")
        print("Using fallback values for Experiment A")

    # Load config B
    try:
        with open(config_b_path, "r") as f:
            config_b = yaml.safe_load(f) or {}
    except (FileNotFoundError, yaml.YAMLError) as e:
        print(f"Warning: Could not load config B ({config_b_path}): {e}")
        print("Using fallback values for Experiment B")

    # Load config C
    try:
        with open(config_c_path, "r") as f:
            config_c = yaml.safe_load(f) or {}
    except (FileNotFoundError, yaml.YAMLError) as e:
        print(f"Warning: Could not load config C ({config_c_path}): {e}")
        print("Using fallback values for Experiment C")

    print("=" * 80)
    print("FINE-TUNING EXPERIMENT COMPARISON")
    print("=" * 80)
    print(f"Model: {config_b.get('model_name_or_path', '[CONFIG NOT LOADED]')}")
    print(f"Dataset: {config_b.get('dataset_name_or_path', '[CONFIG NOT LOADED]')}")
    print(
        f"Training Epochs: {config_b.get('num_train_epochs', '[CONFIG NOT LOADED]')}\n"
    )
    print("Data Preprocessing:")
    print("  - Context field length filter: ≤ 1500 characters")
    print("  - Response field length filter: ≤ 4000 characters")
    print("  - Applied consistently to both experiments for fair comparison\n")
    print("Experiment A: PEFT (LoRA) only")
    print(f"  - LoRA rank: {config_a.get('lora_r', '[CONFIG NOT LOADED]')}")
    print(f"  - LoRA alpha: {config_a.get('lora_alpha', '[CONFIG NOT LOADED]')}")
    print(
        f"  - ICA masking:{'DISABLED' if config_a.get('mask_mode') is None else 'ENABLED'}\n"
    )
    print("Experiment B: PEFT (LoRA) + ICA masking")
    print(
        f"  - LoRA rank: {config_b.get('lora_r', '[CONFIG NOT LOADED]')} (identical to A)"
    )
    print(
        f"  - LoRA alpha: {config_b.get('lora_alpha', '[CONFIG NOT LOADED]')} (identical to A)"
    )

    mask_mode = config_b.get("mask_mode", "[CONFIG NOT LOADED]")
    if mask_mode and mask_mode != "[CONFIG NOT LOADED]":
        print(f"  - ICA masking: ENABLED ({mask_mode} mode)")
    else:
        print(f"  - ICA masking: {mask_mode}")

    print(
        f"  - ICA components: {config_b.get('ica_components', '[CONFIG NOT LOADED]')}"
    )
    print(
        f"  - ICA percentile: {config_b.get('ica_percentile', '[CONFIG NOT LOADED]')}"
    )

    print("Experiment C: PEFT (LoRA) + ICA masking (preserve)")
    print(
        f"  - LoRA rank: {config_c.get('lora_r', '[CONFIG NOT LOADED]')} (identical to A/B)"
    )
    print(
        f"  - LoRA alpha: {config_c.get('lora_alpha', '[CONFIG NOT LOADED]')} (identical to A/B)"
    )

    mask_mode_c = config_c.get("mask_mode", "[CONFIG NOT LOADED]")
    if mask_mode_c and mask_mode_c != "[CONFIG NOT LOADED]":
        print(f"  - ICA masking: ENABLED ({mask_mode_c} mode)")
    else:
        print(f"  - ICA masking: {mask_mode_c}")

    print(
        f"  - ICA components: {config_c.get('ica_components', '[CONFIG NOT LOADED]')}"
    )
    print(
        f"  - ICA percentile: {config_c.get('ica_percentile', '[CONFIG NOT LOADED]')}"
    )
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
        results["experiment_a"] = run_experiment_a()

    if args.experiment in ["b", "all"]:
        logger.info("Running Experiment B...")
        results["experiment_b"] = run_experiment_b()

    if args.experiment in ["c", "all"]:
        logger.info("Running Experiment C...")
        results["experiment_c"] = run_experiment_c()

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
