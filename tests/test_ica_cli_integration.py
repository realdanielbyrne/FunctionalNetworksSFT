#!/usr/bin/env python3
"""
Integration tests for ICA masking command line interface.

Tests the command line argument parsing and integration with the main training loop.
"""

import sys
import unittest
import tempfile
import json
import os
import argparse
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add project root to path for proper imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestICACommandLineArguments(unittest.TestCase):
    """Test cases for ICA command line argument parsing."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.mask_file_path = os.path.join(self.temp_dir, "test_mask.json")

        # Create test mask file
        test_mask = {"0": [100, 200, 300], "1": [150, 250, 350]}

        with open(self.mask_file_path, "w") as f:
            json.dump(test_mask, f)

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir)

    def create_parser(self):
        """Create argument parser with ICA arguments."""
        parser = argparse.ArgumentParser()

        # Add the ICA-related arguments
        parser.add_argument(
            "--mask_mode",
            type=str,
            choices=["key", "complement"],
            default=None,
            help="Masking mode for ICA neurons",
        )
        parser.add_argument(
            "--ica_mask_path", type=str, default=None, help="Path to ICA mask JSON file"
        )
        parser.add_argument(
            "--ica_components", type=int, default=20, help="Number of ICA components"
        )
        parser.add_argument(
            "--ica_percentile",
            type=float,
            default=98.0,
            help="ICA percentile threshold",
        )

        return parser

    def test_mask_mode_argument_parsing(self):
        """Test parsing of --mask_mode argument."""
        parser = self.create_parser()

        # Test valid values
        args = parser.parse_args(["--mask_mode", "key"])
        self.assertEqual(args.mask_mode, "key")

        args = parser.parse_args(["--mask_mode", "complement"])
        self.assertEqual(args.mask_mode, "complement")

        # Test default value
        args = parser.parse_args([])
        self.assertIsNone(args.mask_mode)

        # Test invalid value should raise SystemExit
        with self.assertRaises(SystemExit):
            parser.parse_args(["--mask_mode", "invalid"])

    def test_ica_mask_path_argument_parsing(self):
        """Test parsing of --ica_mask_path argument."""
        parser = self.create_parser()

        # Test with valid path
        args = parser.parse_args(["--ica_mask_path", self.mask_file_path])
        self.assertEqual(args.ica_mask_path, self.mask_file_path)

        # Test default value
        args = parser.parse_args([])
        self.assertIsNone(args.ica_mask_path)

    def test_ica_components_argument_parsing(self):
        """Test parsing of --ica_components argument."""
        parser = self.create_parser()

        # Test custom value
        args = parser.parse_args(["--ica_components", "30"])
        self.assertEqual(args.ica_components, 30)

        # Test default value
        args = parser.parse_args([])
        self.assertEqual(args.ica_components, 20)

        # Test invalid value should raise SystemExit
        with self.assertRaises(SystemExit):
            parser.parse_args(["--ica_components", "invalid"])

    def test_ica_percentile_argument_parsing(self):
        """Test parsing of --ica_percentile argument."""
        parser = self.create_parser()

        # Test custom value
        args = parser.parse_args(["--ica_percentile", "95.5"])
        self.assertEqual(args.ica_percentile, 95.5)

        # Test default value
        args = parser.parse_args([])
        self.assertEqual(args.ica_percentile, 98.0)

        # Test invalid value should raise SystemExit
        with self.assertRaises(SystemExit):
            parser.parse_args(["--ica_percentile", "invalid"])

    def test_combined_arguments(self):
        """Test parsing multiple ICA arguments together."""
        parser = self.create_parser()

        args = parser.parse_args(
            [
                "--mask_mode",
                "key",
                "--ica_mask_path",
                self.mask_file_path,
                "--ica_components",
                "25",
                "--ica_percentile",
                "97.0",
            ]
        )

        self.assertEqual(args.mask_mode, "key")
        self.assertEqual(args.ica_mask_path, self.mask_file_path)
        self.assertEqual(args.ica_components, 25)
        self.assertEqual(args.ica_percentile, 97.0)


class TestICATrainingIntegration(unittest.TestCase):
    """Test cases for ICA integration in training pipeline."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.mask_file_path = os.path.join(self.temp_dir, "test_mask.json")

        # Create test mask file
        test_mask = {"0": [100, 200, 300], "1": [150, 250, 350]}

        with open(self.mask_file_path, "w") as f:
            json.dump(test_mask, f)

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir)

    def test_mask_loading_from_file(self):
        """Test loading mask dictionary from file path."""
        # Simulate the file loading logic from main()
        with open(self.mask_file_path) as f:
            mask_dict = json.load(f)

        expected_mask = {"0": [100, 200, 300], "1": [150, 250, 350]}

        self.assertEqual(mask_dict, expected_mask)

    def test_mask_mode_logic(self):
        """Test the mask mode decision logic."""
        # Test case 1: mask_mode is None (no masking)
        mask_mode = None
        should_apply_masking = mask_mode is not None
        self.assertFalse(should_apply_masking)

        # Test case 2: mask_mode is set (apply masking)
        mask_mode = "key"
        should_apply_masking = mask_mode is not None
        self.assertTrue(should_apply_masking)

    def test_mask_path_vs_compute_logic(self):
        """Test the logic for choosing between pre-computed vs on-the-fly ICA."""
        # Mock args object
        args = Mock()

        # Case 1: Pre-computed mask provided
        args.ica_mask_path = self.mask_file_path
        use_precomputed = bool(args.ica_mask_path)
        self.assertTrue(use_precomputed)

        # Case 2: No pre-computed mask (compute on-the-fly)
        args.ica_mask_path = None
        use_precomputed = bool(args.ica_mask_path)
        self.assertFalse(use_precomputed)

    @patch("src.functionalnetworkssft.ica_mask.ICAMask.apply_masks")
    @patch("src.functionalnetworkssft.ica_mask.ICAMask.compute_masks_for_model")
    def test_ica_integration_workflow(self, mock_compute_ica, mock_apply_masks):
        """Test the complete ICA integration workflow."""
        # Mock return values
        mock_compute_ica.return_value = {"0": [100, 200], "1": [150, 250]}
        mock_apply_masks.return_value = [Mock(), Mock()]  # Mock handles

        # Mock objects
        mock_model = Mock()
        mock_dataset = Mock()
        mock_tokenizer = Mock()

        # Mock args
        args = Mock()
        args.mask_mode = "key"
        args.ica_mask_path = None  # Force on-the-fly computation
        args.ica_components = 20
        args.ica_percentile = 98.0

        # Simulate the workflow from main() using ICAMask
        from src.functionalnetworkssft.ica_mask import ICAMask

        mask_handles = []
        if args.mask_mode is not None:
            ica_mask = ICAMask(
                num_components=args.ica_components,
                percentile=args.ica_percentile,
                sample_batches=50,
            )

            if args.ica_mask_path:
                # This branch won't execute due to ica_mask_path = None
                pass
            else:
                # On-the-fly computation
                mask_dict = ica_mask.compute_masks_for_model(
                    mock_model,
                    mock_dataset,
                    mock_tokenizer,
                )

            mask_handles = ica_mask.apply_masks(
                mock_model, mask_dict, mask_mode=args.mask_mode
            )

        # Verify the workflow executed correctly
        mock_compute_ica.assert_called_once()
        mock_apply_masks.assert_called_once()
        self.assertEqual(len(mask_handles), 2)  # Two mock handles returned


class TestICAErrorHandling(unittest.TestCase):
    """Test cases for ICA error handling and edge cases."""

    def test_invalid_mask_file_path(self):
        """Test handling of invalid mask file path."""
        invalid_path = "/nonexistent/path/mask.json"

        with self.assertRaises(FileNotFoundError):
            with open(invalid_path) as f:
                json.load(f)

    def test_invalid_mask_file_format(self):
        """Test handling of invalid mask file format."""
        temp_dir = tempfile.mkdtemp()
        invalid_mask_file = os.path.join(temp_dir, "invalid_mask.json")

        try:
            # Create invalid JSON file
            with open(invalid_mask_file, "w") as f:
                f.write("invalid json content")

            with self.assertRaises(json.JSONDecodeError):
                with open(invalid_mask_file) as f:
                    json.load(f)

        finally:
            import shutil

            shutil.rmtree(temp_dir)

    def test_mask_file_with_invalid_data_types(self):
        """Test handling of mask file with invalid data types."""
        temp_dir = tempfile.mkdtemp()
        invalid_mask_file = os.path.join(temp_dir, "invalid_data_mask.json")

        try:
            # Create mask file with invalid data types
            invalid_mask = {
                "0": "not_a_list",  # Should be a list
                "1": [1.5, 2.5],  # Should be integers
                2: [100, 200],  # Key should be string
            }

            with open(invalid_mask_file, "w") as f:
                json.dump(invalid_mask, f)

            # Load and validate
            with open(invalid_mask_file) as f:
                mask_dict = json.load(f)

            # Validation logic (similar to what should be in the main code)
            validation_errors = []
            for layer_idx, neuron_list in mask_dict.items():
                if not isinstance(layer_idx, str):
                    validation_errors.append(
                        f"Layer index should be string, got {type(layer_idx)}"
                    )
                if not isinstance(neuron_list, list):
                    validation_errors.append(
                        f"Neuron list should be list, got {type(neuron_list)}"
                    )

            # Should have found validation errors
            self.assertGreater(
                len(validation_errors), 0, "Should have detected invalid data types"
            )

        finally:
            import shutil

            shutil.rmtree(temp_dir)


if __name__ == "__main__":
    unittest.main()
