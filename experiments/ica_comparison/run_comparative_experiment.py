#!/usr/bin/env python3
"""
Comparative Training Experiment: ICA Masking vs Baseline SFT

This script runs a controlled A/B comparison between:
1. Baseline: Standard SFT without ICA masking
2. Experimental: SFT with ICA masking enabled

Both experiments use identical hyperparameters, random seeds, and training conditions.
"""

import os
import sys
import json
import time
import logging
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("experiments/ica_comparison/experiment.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


class ComparativeExperiment:
    """Manages the comparative training experiment between baseline and ICA masking."""

    def __init__(self, experiment_dir: str = "experiments/ica_comparison"):
        self.experiment_dir = Path(experiment_dir)
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        # Configuration paths
        self.baseline_config = self.experiment_dir / "baseline_training_config.yaml"
        self.experimental_config = (
            self.experiment_dir / "experimental_training_config.yaml"
        )

        # Results tracking
        self.results = {
            "experiment_start": datetime.now().isoformat(),
            "baseline": {},
            "experimental": {},
            "comparison": {},
        }

        # Ensure configs exist
        if not self.baseline_config.exists():
            raise FileNotFoundError(
                f"Baseline config not found: {self.baseline_config}"
            )
        if not self.experimental_config.exists():
            raise FileNotFoundError(
                f"Experimental config not found: {self.experimental_config}"
            )

    def run_training(self, config_path: Path, experiment_name: str) -> Dict[str, Any]:
        """
        Run a single training experiment.

        Args:
            config_path: Path to the training configuration YAML
            experiment_name: Name of the experiment (baseline/experimental)

        Returns:
            Dictionary with training results and metadata
        """
        logger.info(f"Starting {experiment_name} training...")
        start_time = time.time()

        # Construct training command
        cmd = [
            sys.executable,
            "-m",
            "functionalnetworkssft.fnsft_trainer",
            "--config",
            str(config_path),
        ]

        try:
            # Run training
            result = subprocess.run(
                cmd, capture_output=True, text=True, cwd=Path.cwd(), check=True
            )

            end_time = time.time()
            training_time = end_time - start_time

            logger.info(
                f"{experiment_name} training completed successfully in {training_time:.2f} seconds"
            )

            return {
                "status": "success",
                "training_time": training_time,
                "start_time": start_time,
                "end_time": end_time,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "command": " ".join(cmd),
            }

        except subprocess.CalledProcessError as e:
            logger.error(f"{experiment_name} training failed: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "stdout": e.stdout,
                "stderr": e.stderr,
                "command": " ".join(cmd),
            }

    def run_evaluation(self, model_path: str, experiment_name: str) -> Dict[str, Any]:
        """
        Run evaluation on a trained model.

        Args:
            model_path: Path to the trained model
            experiment_name: Name of the experiment

        Returns:
            Dictionary with evaluation results
        """
        logger.info(f"Starting {experiment_name} evaluation...")

        # Create evaluation config for this model
        eval_config_path = (
            self.experiment_dir / f"{experiment_name}_evaluation_config.yaml"
        )
        self._create_evaluation_config(model_path, eval_config_path)

        cmd = [
            sys.executable,
            "-m",
            "functionalnetworkssft.evaluation.run_evaluation",
            "--config",
            str(eval_config_path),
        ]

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, cwd=Path.cwd(), check=True
            )

            logger.info(f"{experiment_name} evaluation completed successfully")

            # Try to load evaluation results
            eval_results_path = (
                Path(model_path) / "evaluation_results" / "evaluation_report.json"
            )
            eval_results = {}
            if eval_results_path.exists():
                with open(eval_results_path, "r") as f:
                    eval_results = json.load(f)

            return {
                "status": "success",
                "stdout": result.stdout,
                "stderr": result.stderr,
                "results": eval_results,
                "command": " ".join(cmd),
            }

        except subprocess.CalledProcessError as e:
            logger.error(f"{experiment_name} evaluation failed: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "stdout": e.stdout,
                "stderr": e.stderr,
                "command": " ".join(cmd),
            }

    def _create_evaluation_config(self, model_path: str, config_path: Path):
        """Create evaluation configuration for a specific model."""
        eval_config = {
            "model_name_or_path": model_path,
            "use_auth_token": True,
            "trust_remote_code": True,
            "torch_dtype": "auto",
            "output_dir": f"{model_path}/evaluation_results",
            "run_name": f"evaluation_{Path(model_path).name}",
            "batch_size": 16,
            "max_length": 512,
            "num_workers": 4,
            "generate_report": True,
            "include_visualizations": False,
            "save_predictions": False,
            "log_level": "INFO",
            "use_wandb": False,
            "benchmarks": [
                {
                    "name": "language_understanding",
                    "enabled": True,
                    "max_samples": 100,
                    "batch_size": 16,
                    "metrics": [
                        {"name": "perplexity", "enabled": True},
                        {"name": "bleu", "enabled": True},
                    ],
                },
                {
                    "name": "mmlu",
                    "enabled": True,
                    "dataset_name": "cais/mmlu",
                    "max_samples": 200,
                    "batch_size": 8,
                    "metrics": [{"name": "mmlu_accuracy", "enabled": True}],
                },
                {
                    "name": "performance",
                    "enabled": True,
                    "max_samples": 20,
                    "batch_size": 8,
                    "metrics": [
                        {"name": "inference_speed", "enabled": True},
                        {"name": "memory_usage", "enabled": True},
                        {"name": "model_size", "enabled": True},
                    ],
                    "parameters": {"max_length": 256, "max_new_tokens": 25},
                },
            ],
        }

        # Save evaluation config as YAML
        try:
            import yaml

            with open(config_path, "w") as f:
                yaml.dump(eval_config, f, default_flow_style=False)
        except ImportError:
            # Fallback to JSON if PyYAML not available
            import json

            with open(config_path.with_suffix(".json"), "w") as f:
                json.dump(eval_config, f, indent=2)

    def run_full_experiment(self) -> Dict[str, Any]:
        """
        Run the complete comparative experiment.

        Returns:
            Complete results dictionary
        """
        logger.info("Starting comparative ICA masking experiment")

        # Run baseline training
        logger.info("=" * 60)
        logger.info("PHASE 1: BASELINE TRAINING (No ICA Masking)")
        logger.info("=" * 60)
        baseline_results = self.run_training(self.baseline_config, "baseline")
        self.results["baseline"]["training"] = baseline_results

        # Run experimental training
        logger.info("=" * 60)
        logger.info("PHASE 2: EXPERIMENTAL TRAINING (With ICA Masking)")
        logger.info("=" * 60)
        experimental_results = self.run_training(
            self.experimental_config, "experimental"
        )
        self.results["experimental"]["training"] = experimental_results

        # Run evaluations if training succeeded
        if baseline_results["status"] == "success":
            logger.info("=" * 60)
            logger.info("PHASE 3: BASELINE EVALUATION")
            logger.info("=" * 60)
            baseline_eval = self.run_evaluation(
                "./experiments/ica_comparison/baseline_model", "baseline"
            )
            self.results["baseline"]["evaluation"] = baseline_eval

        if experimental_results["status"] == "success":
            logger.info("=" * 60)
            logger.info("PHASE 4: EXPERIMENTAL EVALUATION")
            logger.info("=" * 60)
            experimental_eval = self.run_evaluation(
                "./experiments/ica_comparison/experimental_model", "experimental"
            )
            self.results["experimental"]["evaluation"] = experimental_eval

        # Generate comparison
        self._generate_comparison()

        # Save results
        self.results["experiment_end"] = datetime.now().isoformat()
        results_path = self.experiment_dir / "experiment_results.json"
        with open(results_path, "w") as f:
            json.dump(self.results, f, indent=2)

        logger.info(f"Experiment completed. Results saved to {results_path}")
        return self.results

    def _generate_comparison(self):
        """Generate comparison between baseline and experimental results."""
        comparison = {
            "training_comparison": {},
            "evaluation_comparison": {},
            "summary": {},
        }

        # Training comparison
        baseline_training = self.results["baseline"].get("training", {})
        experimental_training = self.results["experimental"].get("training", {})

        if (
            baseline_training.get("status") == "success"
            and experimental_training.get("status") == "success"
        ):
            comparison["training_comparison"] = {
                "baseline_time": baseline_training.get("training_time", 0),
                "experimental_time": experimental_training.get("training_time", 0),
                "time_difference": experimental_training.get("training_time", 0)
                - baseline_training.get("training_time", 0),
            }

        # Evaluation comparison
        baseline_eval = self.results["baseline"].get("evaluation", {})
        experimental_eval = self.results["experimental"].get("evaluation", {})

        if (
            baseline_eval.get("status") == "success"
            and experimental_eval.get("status") == "success"
        ):
            # Extract key metrics for comparison
            baseline_metrics = baseline_eval.get("results", {})
            experimental_metrics = experimental_eval.get("results", {})

            comparison["evaluation_comparison"] = {
                "baseline_metrics": baseline_metrics,
                "experimental_metrics": experimental_metrics,
            }

        # Summary
        comparison["summary"] = {
            "baseline_success": baseline_training.get("status") == "success",
            "experimental_success": experimental_training.get("status") == "success",
            "both_completed": (
                baseline_training.get("status") == "success"
                and experimental_training.get("status") == "success"
            ),
        }

        self.results["comparison"] = comparison


def main():
    """Main entry point for the comparative experiment."""
    experiment = ComparativeExperiment()

    try:
        results = experiment.run_full_experiment()

        # Print summary
        print("\n" + "=" * 80)
        print("EXPERIMENT SUMMARY")
        print("=" * 80)

        comparison = results.get("comparison", {})
        summary = comparison.get("summary", {})

        print(
            f"Baseline Training: {'✓ Success' if summary.get('baseline_success') else '✗ Failed'}"
        )
        print(
            f"Experimental Training: {'✓ Success' if summary.get('experimental_success') else '✗ Failed'}"
        )
        print(f"Both Completed: {'✓ Yes' if summary.get('both_completed') else '✗ No'}")

        if summary.get("both_completed"):
            training_comp = comparison.get("training_comparison", {})
            print(f"\nTraining Time Comparison:")
            print(f"  Baseline: {training_comp.get('baseline_time', 0):.2f}s")
            print(f"  Experimental: {training_comp.get('experimental_time', 0):.2f}s")
            print(f"  Difference: {training_comp.get('time_difference', 0):.2f}s")

        print(
            f"\nDetailed results saved to: experiments/ica_comparison/experiment_results.json"
        )

    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
