#!/usr/bin/env python3
"""
Unit tests for build_ica_templates.py

This script creates mock datasets and tests the ICA template building functionality.
"""

import csv
import json
import os
import tempfile
import unittest
from unittest.mock import patch, MagicMock
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


class TestBuildICATemplates(unittest.TestCase):
    """Unit tests for ICA template building functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.addCleanup(self.cleanup_temp_dir)

    def cleanup_temp_dir(self):
        """Clean up temporary directory."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def create_mock_csv_dataset(
        self, output_path: str, num_samples: int = 20, dataset_name: str = "mock"
    ):
        """Create a mock CSV dataset for testing."""

        with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
            fieldnames = ["instruction", "response"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for i in range(num_samples):
                writer.writerow(
                    {
                        "instruction": f"What is the capital of country {i} in {dataset_name}?",
                        "response": f"The capital of country {i} is City{i}. This information comes from {dataset_name} dataset.",
                    }
                )

        print(f"Created mock CSV dataset: {output_path} with {num_samples} samples")
        return output_path

    def create_mock_json_dataset(
        self, output_path: str, num_samples: int = 20, dataset_name: str = "mock"
    ):
        """Create a mock JSON dataset for testing."""

        sample_data = []
        for i in range(num_samples):
            sample_data.append(
                {
                    "instruction": f"Explain concept {i} from {dataset_name} domain.",
                    "response": f"Concept {i} from {dataset_name} is an important topic that involves understanding key principles.",
                }
            )

        with open(output_path, "w") as f:
            json.dump(sample_data, f, indent=2)

        print(f"Created mock JSON dataset: {output_path} with {num_samples} samples")
        return output_path

    def test_dataset_loader(self):
        """Test the DatasetLoader functionality."""
        from functionalnetworkssft.build_ica_templates import DatasetLoader

        # Create mock datasets
        csv_path = os.path.join(self.temp_dir, "test_dataset.csv")
        json_path = os.path.join(self.temp_dir, "test_dataset.json")

        self.create_mock_csv_dataset(csv_path, num_samples=15)
        self.create_mock_json_dataset(json_path, num_samples=25)

        # Mock tokenizer
        mock_tokenizer = MagicMock()

        # Test loading and sampling
        combined_data = DatasetLoader.load_and_sample_datasets(
            dataset_paths=[csv_path, json_path],
            samples_per_dataset=10,
            tokenizer=mock_tokenizer,
            max_seq_length=256,
            template_format="auto",
        )

        # Should have 20 samples total (10 from each dataset)
        self.assertEqual(len(combined_data), 20)

        # Check that data has expected structure
        for item in combined_data:
            self.assertIn("instruction", item)
            self.assertIn("response", item)

    @patch("functionalnetworkssft.build_ica_templates.AutoModelForCausalLM")
    @patch("functionalnetworkssft.build_ica_templates.AutoTokenizer")
    def test_build_ica_templates_mocked(self, mock_tokenizer_class, mock_model_class):
        """Test the main build_ica_templates function with mocked model and tokenizer."""

        # Setup mocks
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "<eos>"
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        mock_model = MagicMock()
        mock_model_class.from_pretrained.return_value = mock_model

        # Create mock datasets
        csv_path = self.create_mock_csv_dataset(
            os.path.join(self.temp_dir, "dataset1.csv"), num_samples=30
        )
        json_path = self.create_mock_json_dataset(
            os.path.join(self.temp_dir, "dataset2.json"), num_samples=40
        )

        output_dir = os.path.join(self.temp_dir, "templates")

        # Mock ICAMask to avoid actual ICA computation
        with patch(
            "functionalnetworkssft.build_ica_templates.ICAMask"
        ) as mock_ica_class:
            mock_ica = MagicMock()
            mock_ica_class.return_value = mock_ica

            # Mock the compute_global_networks to return some dummy component masks
            mock_ica.compute_global_networks.return_value = {
                0: {"0": [1, 2, 3], "1": [4, 5, 6]},
                1: {"0": [7, 8, 9], "1": [10, 11, 12]},
            }

            # Mock build_templates_from_current_components
            mock_templates = {
                "name": "test_templates",
                "layout": {"captured_layers_sorted": [0, 1], "hidden_size": 768},
                "templates": {
                    0: {"0": [1, 2, 3], "1": [4, 5, 6]},
                    1: {"0": [7, 8, 9], "1": [10, 11, 12]},
                },
            }
            mock_ica.build_templates_from_current_components.return_value = (
                mock_templates
            )

            # Import and test the function
            from functionalnetworkssft.build_ica_templates import build_ica_templates

            # Run the function
            build_ica_templates(
                dataset_paths=[csv_path, json_path],
                samples_per_dataset=15,
                output_path=output_dir,
                model_name_or_path="microsoft/DialoGPT-medium",
                ica_components=2,
                ica_percentile=95.0,
                ica_dtype="float32",
                max_seq_length=256,
                template_format="auto",
            )

            # Verify mocks were called
            mock_tokenizer_class.from_pretrained.assert_called_once()
            mock_model_class.from_pretrained.assert_called_once()
            mock_ica.compute_global_networks.assert_called_once()
            mock_ica.build_templates_from_current_components.assert_called_once()
            mock_ica.save_templates.assert_called_once()

            # Check that output directory was created
            self.assertTrue(os.path.exists(output_dir))

    def test_cli_argument_parsing(self):
        """Test command-line argument parsing."""
        import subprocess
        import sys

        # Test help message
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "functionalnetworkssft.build_ica_templates",
                "--help",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )

        self.assertEqual(result.returncode, 0)
        self.assertIn("Build ICA templates", result.stdout)
        self.assertIn("--ica_build_templates_from", result.stdout)
        self.assertIn("--ica_template_samples_per_ds", result.stdout)
        self.assertIn("--ica_template_output", result.stdout)

    def test_invalid_arguments(self):
        """Test handling of invalid arguments."""
        import subprocess
        import sys

        # Test missing required arguments
        result = subprocess.run(
            [sys.executable, "-m", "functionalnetworkssft.build_ica_templates"],
            capture_output=True,
            text=True,
            timeout=10,
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("required", result.stderr.lower())

    def test_dataset_format_detection(self):
        """Test dataset format detection with different file types."""
        from functionalnetworkssft.build_ica_templates import DatasetLoader

        # Create datasets with different formats
        csv_path = self.create_mock_csv_dataset(
            os.path.join(self.temp_dir, "test.csv"), num_samples=10
        )

        json_path = self.create_mock_json_dataset(
            os.path.join(self.temp_dir, "test.json"), num_samples=10
        )

        # Mock tokenizer
        mock_tokenizer = MagicMock()

        # Test CSV loading
        csv_data = DatasetLoader.load_and_sample_datasets(
            dataset_paths=[csv_path], samples_per_dataset=5, tokenizer=mock_tokenizer
        )
        self.assertEqual(len(csv_data), 5)

        # Test JSON loading
        json_data = DatasetLoader.load_and_sample_datasets(
            dataset_paths=[json_path], samples_per_dataset=5, tokenizer=mock_tokenizer
        )
        self.assertEqual(len(json_data), 5)

        # Test mixed loading
        mixed_data = DatasetLoader.load_and_sample_datasets(
            dataset_paths=[csv_path, json_path],
            samples_per_dataset=5,
            tokenizer=mock_tokenizer,
        )
        self.assertEqual(len(mixed_data), 10)


class TestIntegration(unittest.TestCase):
    """Integration tests that can be run optionally."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.addCleanup(self.cleanup_temp_dir)

    def cleanup_temp_dir(self):
        """Clean up temporary directory."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def create_realistic_csv_dataset(self, output_path: str, num_samples: int = 50):
        """Create a more realistic CSV dataset for integration testing."""

        instructions = [
            "Explain the concept of machine learning",
            "What are the benefits of renewable energy?",
            "How does photosynthesis work?",
            "Describe the process of DNA replication",
            "What is the theory of relativity?",
            "Explain how neural networks function",
            "What causes climate change?",
            "How do vaccines work?",
            "Describe the water cycle",
            "What is quantum computing?",
        ]

        responses = [
            "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed.",
            "Renewable energy sources like solar and wind power provide clean, sustainable alternatives to fossil fuels, reducing greenhouse gas emissions.",
            "Photosynthesis is the process by which plants convert sunlight, carbon dioxide, and water into glucose and oxygen using chlorophyll.",
            "DNA replication involves unwinding the double helix and synthesizing complementary strands using DNA polymerase enzymes.",
            "Einstein's theory of relativity describes the relationship between space, time, gravity, and the speed of light in the universe.",
            "Neural networks are computing systems inspired by biological neural networks, using interconnected nodes to process information.",
            "Climate change is primarily caused by increased greenhouse gas emissions from human activities like burning fossil fuels.",
            "Vaccines work by training the immune system to recognize and fight specific pathogens without causing the disease.",
            "The water cycle involves evaporation, condensation, precipitation, and collection, continuously moving water through Earth's systems.",
            "Quantum computing uses quantum mechanical phenomena like superposition and entanglement to process information differently than classical computers.",
        ]

        with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
            fieldnames = ["instruction", "response"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for i in range(num_samples):
                inst_idx = i % len(instructions)
                resp_idx = i % len(responses)
                writer.writerow(
                    {
                        "instruction": f"{instructions[inst_idx]} (Sample {i})",
                        "response": f"{responses[resp_idx]} (Response {i})",
                    }
                )

        return output_path

    @unittest.skip("Integration test - run manually with --integration flag")
    def test_full_integration_with_small_model(self):
        """Full integration test with a small model (run manually)."""

        # Create realistic test datasets
        csv_path1 = self.create_realistic_csv_dataset(
            os.path.join(self.temp_dir, "science_qa.csv"), num_samples=30
        )
        csv_path2 = self.create_realistic_csv_dataset(
            os.path.join(self.temp_dir, "general_qa.csv"), num_samples=25
        )

        output_dir = os.path.join(self.temp_dir, "integration_templates")

        from functionalnetworkssft.build_ica_templates import build_ica_templates

        # Run with a very small model for faster testing
        build_ica_templates(
            dataset_paths=[csv_path1, csv_path2],
            samples_per_dataset=10,
            output_path=output_dir,
            model_name_or_path="microsoft/DialoGPT-small",  # Smaller model
            ica_components=3,  # Fewer components
            ica_percentile=90.0,
            ica_dtype="float32",
            max_seq_length=128,  # Shorter sequences
            template_format="auto",
        )

        # Verify output
        template_file = os.path.join(output_dir, "global_templates.json")
        self.assertTrue(os.path.exists(template_file))

        with open(template_file, "r") as f:
            templates = json.load(f)

        self.assertIn("name", templates)
        self.assertIn("layout", templates)
        self.assertIn("templates", templates)
        self.assertGreater(len(templates["templates"]), 0)


def run_integration_tests():
    """Run integration tests manually."""
    print("Running integration tests...")
    suite = unittest.TestLoader().loadTestsFromTestCase(TestIntegration)

    # Remove the skip decorator temporarily
    for test_case in suite:
        if hasattr(test_case, "test_full_integration_with_small_model"):
            test_case.test_full_integration_with_small_model = (
                test_case.test_full_integration_with_small_model.__func__
            )

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test ICA template building script")
    parser.add_argument(
        "--integration",
        action="store_true",
        help="Run integration tests (requires model download)",
    )
    args = parser.parse_args()

    if args.integration:
        print("Running integration tests (this may take several minutes)...")
        success = run_integration_tests()
        if success:
            print("✅ Integration tests passed!")
        else:
            print("❌ Integration tests failed!")
    else:
        print("Running unit tests...")
        unittest.main(verbosity=2, argv=[""])
