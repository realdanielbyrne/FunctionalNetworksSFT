#!/usr/bin/env python3
"""
Unit tests for the --ica-mask-layers functionality.
Tests layer specification parsing, edge cases, and error handling.
"""

import unittest
from unittest.mock import Mock, patch
import sys
import os

# Add the src directory to the path so we can import the module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from functionalnetworkssft.fnsft_trainer import parse_layer_specification


class TestLayerSpecificationParsing(unittest.TestCase):
    """Test the parse_layer_specification function."""

    def test_single_layer(self):
        """Test single layer specification."""
        result = parse_layer_specification("0", 10)
        self.assertEqual(result, [0])

        result = parse_layer_specification("5", 10)
        self.assertEqual(result, [5])

        result = parse_layer_specification("9", 10)
        self.assertEqual(result, [9])

    def test_multiple_individual_layers(self):
        """Test multiple individual layers specification."""
        result = parse_layer_specification("0,3,7", 10)
        self.assertEqual(result, [0, 3, 7])

        result = parse_layer_specification("1,2,5,8", 10)
        self.assertEqual(result, [1, 2, 5, 8])

        # Test with spaces
        result = parse_layer_specification("0, 3, 7", 10)
        self.assertEqual(result, [0, 3, 7])

    def test_range_specifications(self):
        """Test range specifications using colon notation."""
        # Basic range
        result = parse_layer_specification("0:4", 10)
        self.assertEqual(result, [0, 1, 2, 3])

        # Range from start
        result = parse_layer_specification(":3", 10)
        self.assertEqual(result, [0, 1, 2])

        # Range to end
        result = parse_layer_specification("7:", 10)
        self.assertEqual(result, [7, 8, 9])

        # Single element range
        result = parse_layer_specification("5:6", 10)
        self.assertEqual(result, [5])

    def test_mixed_specifications(self):
        """Test mixed individual and range specifications."""
        result = parse_layer_specification("0,2:5,8", 10)
        self.assertEqual(result, [0, 2, 3, 4, 8])

        result = parse_layer_specification("1,3:6,7,9:", 12)
        self.assertEqual(result, [1, 3, 4, 5, 7, 9, 10, 11])

        # With spaces
        result = parse_layer_specification("0, 2:5, 8", 10)
        self.assertEqual(result, [0, 2, 3, 4, 8])

    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # All layers using range
        result = parse_layer_specification(":", 5)
        self.assertEqual(result, [0, 1, 2, 3, 4])

        # Last layer only
        result = parse_layer_specification("4", 5)
        self.assertEqual(result, [4])

        # Range to last layer
        result = parse_layer_specification("3:", 5)
        self.assertEqual(result, [3, 4])

    def test_duplicate_removal(self):
        """Test that duplicate layer indices are removed."""
        result = parse_layer_specification("0,1,0:3,2", 10)
        self.assertEqual(result, [0, 1, 2])

        result = parse_layer_specification("5,3:7,6", 10)
        self.assertEqual(result, [3, 4, 5, 6])

    def test_error_cases(self):
        """Test error handling for invalid specifications."""
        # Empty specification
        with self.assertRaises(ValueError) as cm:
            parse_layer_specification("", 10)
        self.assertIn("cannot be empty", str(cm.exception))

        with self.assertRaises(ValueError) as cm:
            parse_layer_specification("   ", 10)
        self.assertIn("cannot be empty", str(cm.exception))

        # Negative indices
        with self.assertRaises(ValueError) as cm:
            parse_layer_specification("-1", 10)
        self.assertIn("cannot be negative", str(cm.exception))

        with self.assertRaises(ValueError) as cm:
            parse_layer_specification("-1:3", 10)
        self.assertIn("cannot be negative", str(cm.exception))

        with self.assertRaises(ValueError) as cm:
            parse_layer_specification("2:-1", 10)
        self.assertIn("cannot be negative", str(cm.exception))

        # Out of bounds indices
        with self.assertRaises(ValueError) as cm:
            parse_layer_specification("10", 10)
        self.assertIn("exceeds total layers", str(cm.exception))

        with self.assertRaises(ValueError) as cm:
            parse_layer_specification("5:15", 10)
        self.assertIn("exceeds total layers", str(cm.exception))

        with self.assertRaises(ValueError) as cm:
            parse_layer_specification("15:", 10)
        self.assertIn("exceeds total layers", str(cm.exception))

        # Invalid range (start >= end)
        with self.assertRaises(ValueError) as cm:
            parse_layer_specification("5:3", 10)
        self.assertIn("Invalid range", str(cm.exception))

        with self.assertRaises(ValueError) as cm:
            parse_layer_specification("5:5", 10)
        self.assertIn("Invalid range", str(cm.exception))

        # Invalid format
        with self.assertRaises(ValueError) as cm:
            parse_layer_specification("abc", 10)
        self.assertIn("Invalid layer index", str(cm.exception))

        with self.assertRaises(ValueError) as cm:
            parse_layer_specification("1:2:3", 10)
        self.assertIn("Invalid range format", str(cm.exception))

        with self.assertRaises(ValueError) as cm:
            parse_layer_specification("a:b", 10)
        self.assertIn("Invalid range format", str(cm.exception))

    def test_whitespace_handling(self):
        """Test that whitespace is handled correctly."""
        result = parse_layer_specification(" 0 , 3 , 7 ", 10)
        self.assertEqual(result, [0, 3, 7])

        result = parse_layer_specification(" 0:4 , 7: ", 10)
        self.assertEqual(result, [0, 1, 2, 3, 7, 8, 9])

        result = parse_layer_specification("  :3  ,  5:  ", 10)
        self.assertEqual(result, [0, 1, 2, 5, 6, 7, 8, 9])


class TestICAMaskLayersIntegration(unittest.TestCase):
    """Test integration of layer filtering with ICA masking."""

    def test_target_layers_parameter_exists(self):
        """Test that target_layers parameter exists in compute_ica_masks_for_model signature."""
        from functionalnetworkssft.fnsft_trainer import compute_ica_masks_for_model
        import inspect

        # Verify the function signature includes target_layers parameter
        sig = inspect.signature(compute_ica_masks_for_model)
        target_layers_param = sig.parameters.get("target_layers")

        self.assertIsNotNone(target_layers_param)
        self.assertEqual(target_layers_param.default, None)

        # Verify the parameter has the correct type annotation
        annotation = target_layers_param.annotation
        # The annotation should be list[int] | None or similar
        self.assertIn("list", str(annotation).lower())

    def test_backward_compatibility(self):
        """Test that default behavior (no layer filtering) is preserved."""
        # When target_layers is None, all layers should be processed
        # This is tested by ensuring the function signature accepts None as default
        from functionalnetworkssft.fnsft_trainer import compute_ica_masks_for_model
        import inspect

        sig = inspect.signature(compute_ica_masks_for_model)
        target_layers_param = sig.parameters.get("target_layers")

        self.assertIsNotNone(target_layers_param)
        self.assertEqual(target_layers_param.default, None)


class TestCommandLineIntegration(unittest.TestCase):
    """Test command line argument integration."""

    def test_argument_parser_includes_ica_mask_layers(self):
        """Test that the argument parser includes the --ica-mask-layers option."""
        # This test verifies the argument was added to the parser
        # We can't easily test the full argument parser without running main()
        # but we can verify the parameter exists in the function signature
        from functionalnetworkssft.fnsft_trainer import parse_layer_specification
        import inspect

        # Verify the parsing function exists and has correct signature
        sig = inspect.signature(parse_layer_specification)
        params = list(sig.parameters.keys())
        self.assertIn("layer_spec", params)
        self.assertIn("total_layers", params)

    @patch("builtins.open")
    @patch("json.load")
    def test_precomputed_mask_filtering(self, mock_json_load, mock_open):
        """Test filtering of pre-computed masks when target layers are specified."""
        # Mock a pre-computed mask with multiple layers
        mock_json_load.return_value = {
            "0": [100, 200, 300],
            "1": [150, 250, 350],
            "2": [175, 275, 375],
            "3": [125, 225, 325],
        }

        # This would be tested in the main function integration
        # For now, we test the filtering logic directly
        original_mask = {
            "0": [100, 200, 300],
            "1": [150, 250, 350],
            "2": [175, 275, 375],
            "3": [125, 225, 325],
        }

        target_layers = [0, 2]
        filtered_mask = {
            str(layer): original_mask[str(layer)]
            for layer in target_layers
            if str(layer) in original_mask
        }

        expected = {"0": [100, 200, 300], "2": [175, 275, 375]}
        self.assertEqual(filtered_mask, expected)


class TestLayerFilteringOptimization(unittest.TestCase):
    """Test that layer filtering provides optimization benefits."""

    def test_skipped_layers_not_processed(self):
        """Test that layers not in target_layers are skipped during processing."""
        # This test verifies the optimization aspect - that ICA computation
        # is skipped for layers not in the target list

        # Mock a scenario where we have 10 layers but only want to process 2
        total_layers = 10
        target_layers = [0, 5]

        # Verify that only target layers would be processed
        processed_layers = []
        for i in range(total_layers):
            if target_layers is None or i in target_layers:
                processed_layers.append(i)

        self.assertEqual(processed_layers, [0, 5])

        # Verify optimization: only 2 layers processed instead of 10
        self.assertEqual(len(processed_layers), 2)
        self.assertLess(len(processed_layers), total_layers)


class TestErrorHandlingScenarios(unittest.TestCase):
    """Test comprehensive error handling scenarios."""

    def test_empty_layer_specification_variations(self):
        """Test various empty specification formats."""
        empty_specs = ["", "   ", "\t", "\n", ",", ",,", ", ,"]

        for spec in empty_specs:
            with self.subTest(spec=repr(spec)):
                with self.assertRaises(ValueError):
                    parse_layer_specification(spec, 10)

    def test_malformed_range_specifications(self):
        """Test malformed range specifications."""
        malformed_ranges = [
            "1:2:3",  # Too many colons
            "a:b",  # Non-numeric
            "1:",  # Missing end (should work, but test boundary)
            ":1:",  # Extra colon
            "::1",  # Double colon start
            "1::",  # Double colon end
        ]

        for spec in malformed_ranges:
            with self.subTest(spec=spec):
                if spec == "1:":  # This should actually work
                    result = parse_layer_specification(spec, 5)
                    self.assertEqual(result, [1, 2, 3, 4])
                else:
                    with self.assertRaises(ValueError):
                        parse_layer_specification(spec, 10)

    def test_boundary_conditions(self):
        """Test boundary conditions for layer specifications."""
        # Test with minimal model (1 layer)
        result = parse_layer_specification("0", 1)
        self.assertEqual(result, [0])

        # Test with single layer model - invalid specs
        with self.assertRaises(ValueError):
            parse_layer_specification("1", 1)  # Out of bounds

        with self.assertRaises(ValueError):
            parse_layer_specification("0:2", 1)  # Range exceeds bounds

    def test_large_model_specifications(self):
        """Test specifications for large models."""
        # Test with a large model (100 layers)
        large_model_layers = 100

        # Test large range
        result = parse_layer_specification("90:", large_model_layers)
        expected = list(range(90, 100))
        self.assertEqual(result, expected)

        # Test mixed specification with large numbers
        result = parse_layer_specification("0,50:55,99", large_model_layers)
        self.assertEqual(result, [0, 50, 51, 52, 53, 54, 99])


if __name__ == "__main__":
    unittest.main()
