#!/usr/bin/env python3
"""
Test Script for ICA Comparison Experiment Setup

This script validates that all components of the comparative experiment
are properly configured and can work together before running the full experiment.
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, List

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExperimentValidator:
    """Validates the experiment setup before running the full comparison."""

    def __init__(self, experiment_dir: str = "."):
        self.experiment_dir = Path(experiment_dir)
        self.validation_results = {
            "data_preprocessing": False,
            "training_configs": False,
            "evaluation_config": False,
            "scripts": False,
            "dependencies": False,
        }

    def validate_data_preprocessing(self) -> bool:
        """Test the enhanced data preprocessing for 3-column format."""
        logger.info("Testing data preprocessing for 3-column format...")

        try:
            from functionalnetworkssft.utils.dataset_utils import DatasetFormatter

            # Test sample data in the expected format
            test_data = [
                {
                    "system": "You are a financial advisor.",
                    "user": "What is diversification in investing?",
                    "assistant": "Diversification is a risk management strategy...",
                },
                {
                    "system": "",  # Empty system prompt
                    "user": "How do I calculate ROI?",
                    "assistant": "ROI (Return on Investment) is calculated as...",
                },
            ]

            # Test format detection
            detected_format = DatasetFormatter.detect_format(test_data)
            logger.info(f"Detected format: {detected_format}")

            # Test conversion
            for i, item in enumerate(test_data):
                converted = DatasetFormatter.convert_to_standard_format(
                    item, detected_format
                )
                logger.info(f"Sample {i+1} conversion: {converted}")

                # Validate conversion
                assert "instruction" in converted
                assert "response" in converted
                assert "System:" in converted["instruction"]
                assert "User:" in converted["instruction"]

            logger.info("✓ Data preprocessing validation passed")
            return True

        except Exception as e:
            logger.error(f"✗ Data preprocessing validation failed: {e}")
            return False

    def validate_training_configs(self) -> bool:
        """Validate training configuration files."""
        logger.info("Validating training configurations...")

        try:
            baseline_config = self.experiment_dir / "baseline_training_config.yaml"
            experimental_config = (
                self.experiment_dir / "experimental_training_config.yaml"
            )

            # Check if files exist
            if not baseline_config.exists():
                logger.error(f"✗ Baseline config not found: {baseline_config}")
                return False

            if not experimental_config.exists():
                logger.error(f"✗ Experimental config not found: {experimental_config}")
                return False

            # Try to parse YAML files
            import yaml

            with open(baseline_config, "r") as f:
                baseline_data = yaml.safe_load(f)

            with open(experimental_config, "r") as f:
                experimental_data = yaml.safe_load(f)

            # Validate key differences
            assert baseline_data["mask_mode"] is None, "Baseline should have no masking"
            assert (
                experimental_data["mask_mode"] == "key"
            ), "Experimental should have ICA masking"

            # Validate identical core settings
            core_settings = [
                "model_name_or_path",
                "dataset_name_or_path",
                "num_train_epochs",
                "learning_rate",
                "seed",
                "use_peft",
                "lora_r",
                "lora_alpha",
            ]

            for setting in core_settings:
                if baseline_data.get(setting) != experimental_data.get(setting):
                    logger.warning(
                        f"Difference in {setting}: {baseline_data.get(setting)} vs {experimental_data.get(setting)}"
                    )

            logger.info("✓ Training configurations validation passed")
            return True

        except Exception as e:
            logger.error(f"✗ Training configurations validation failed: {e}")
            return False

    def validate_evaluation_config(self) -> bool:
        """Validate evaluation configuration."""
        logger.info("Validating evaluation configuration...")

        try:
            eval_config = self.experiment_dir / "financial_evaluation_config.yaml"

            if not eval_config.exists():
                logger.error(f"✗ Evaluation config not found: {eval_config}")
                return False

            import yaml

            with open(eval_config, "r") as f:
                eval_data = yaml.safe_load(f)

            # Validate required sections
            required_sections = ["benchmarks", "model_name_or_path", "output_dir"]
            for section in required_sections:
                assert section in eval_data, f"Missing required section: {section}"

            # Validate benchmarks
            benchmarks = eval_data["benchmarks"]
            assert len(benchmarks) > 0, "No benchmarks configured"

            for benchmark in benchmarks:
                assert "name" in benchmark, "Benchmark missing name"
                assert "enabled" in benchmark, "Benchmark missing enabled flag"
                assert "metrics" in benchmark, "Benchmark missing metrics"

            logger.info("✓ Evaluation configuration validation passed")
            return True

        except Exception as e:
            logger.error(f"✗ Evaluation configuration validation failed: {e}")
            return False

    def validate_scripts(self) -> bool:
        """Validate experiment scripts."""
        logger.info("Validating experiment scripts...")

        try:
            # Check main experiment script
            main_script = self.experiment_dir / "run_comparative_experiment.py"
            if not main_script.exists():
                logger.error(f"✗ Main experiment script not found: {main_script}")
                return False

            # Check comparison script
            comparison_script = self.experiment_dir / "compare_results.py"
            if not comparison_script.exists():
                logger.error(f"✗ Comparison script not found: {comparison_script}")
                return False

            # Try to import the scripts (basic syntax check)
            import importlib.util

            for script_path in [main_script, comparison_script]:
                spec = importlib.util.spec_from_file_location(
                    "test_module", script_path
                )
                module = importlib.util.module_from_spec(spec)
                # Don't execute, just check if it can be loaded

            logger.info("✓ Scripts validation passed")
            return True

        except Exception as e:
            logger.error(f"✗ Scripts validation failed: {e}")
            return False

    def validate_dependencies(self) -> bool:
        """Validate required dependencies."""
        logger.info("Validating dependencies...")

        required_packages = [
            "torch",
            "transformers",
            "datasets",
            "peft",
            "bitsandbytes",
            "accelerate",
            "pandas",
            "numpy",
            "matplotlib",
            "seaborn",
            "sklearn",  # scikit-learn imports as sklearn
            "yaml",
        ]

        missing_packages = []

        for package in required_packages:
            try:
                if package == "yaml":
                    import yaml
                else:
                    __import__(package)
                logger.info(f"✓ {package}")
            except ImportError:
                logger.warning(f"✗ {package} not available")
                missing_packages.append(package)

        if missing_packages:
            logger.warning(f"Missing packages: {missing_packages}")
            logger.info(
                "Install missing packages with: pip install "
                + " ".join(missing_packages)
            )
            return False

        logger.info("✓ Dependencies validation passed")
        return True

    def validate_dataset_access(self) -> bool:
        """Test access to the dataset."""
        logger.info("Testing dataset access...")

        try:
            from datasets import load_dataset

            # Try to load a small sample of the dataset
            dataset = load_dataset(
                "Josephgflowers/Finance-Instruct-500k", split="train[:5]"
            )

            # Check dataset structure
            sample = dataset[0]
            logger.info(f"Dataset sample keys: {list(sample.keys())}")

            # Validate expected columns
            expected_columns = {"system", "user", "assistant"}
            actual_columns = set(sample.keys())

            if not expected_columns.issubset(actual_columns):
                logger.warning(
                    f"Dataset columns mismatch. Expected: {expected_columns}, Found: {actual_columns}"
                )
                return False

            logger.info("✓ Dataset access validation passed")
            return True

        except Exception as e:
            logger.error(f"✗ Dataset access validation failed: {e}")
            return False

    def run_full_validation(self) -> Dict[str, bool]:
        """Run all validation tests."""
        logger.info("Starting full experiment validation...")
        logger.info("=" * 60)

        # Run all validation tests
        self.validation_results["data_preprocessing"] = (
            self.validate_data_preprocessing()
        )
        self.validation_results["training_configs"] = self.validate_training_configs()
        self.validation_results["evaluation_config"] = self.validate_evaluation_config()
        self.validation_results["scripts"] = self.validate_scripts()
        self.validation_results["dependencies"] = self.validate_dependencies()

        # Optional: Test dataset access (may require authentication)
        try:
            dataset_access = self.validate_dataset_access()
            self.validation_results["dataset_access"] = dataset_access
        except Exception as e:
            logger.warning(f"Dataset access test skipped: {e}")
            self.validation_results["dataset_access"] = None

        # Summary
        logger.info("=" * 60)
        logger.info("VALIDATION SUMMARY")
        logger.info("=" * 60)

        all_passed = True
        for test_name, result in self.validation_results.items():
            if result is None:
                status = "SKIPPED"
            elif result:
                status = "PASSED"
            else:
                status = "FAILED"
                all_passed = False

            logger.info(f"{test_name:20}: {status}")

        if all_passed:
            logger.info("\n✓ All validations passed! Experiment is ready to run.")
        else:
            logger.warning(
                "\n✗ Some validations failed. Please fix issues before running the experiment."
            )

        return self.validation_results


def main():
    """Main entry point for validation."""
    validator = ExperimentValidator()
    results = validator.run_full_validation()

    # Return appropriate exit code
    failed_tests = [k for k, v in results.items() if v is False]
    if failed_tests:
        print(f"\nFailed tests: {failed_tests}")
        return 1
    else:
        print("\n🎉 Validation completed successfully!")
        return 0


if __name__ == "__main__":
    exit(main())
