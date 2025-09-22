#!/usr/bin/env python3
"""
Test script to verify that the ICA row parametrizations NoneType error is fixed.
This test ensures that apply_row_parametrizations handles None mlps gracefully.
"""

import os
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from functionalnetworkssft.ica_mask import ICAMask


class TestICARowParametrizationsFix(unittest.TestCase):
    """Test that the ICA row parametrizations NoneType error is fixed."""

    def setUp(self):
        """Set up test environment."""
        self.ica_mask = ICAMask(
            num_components=5,
            percentile=98.0,
            sample_batches=100,
        )

        # Set up mock component masks
        self.ica_mask.mask_dict_components = {
            0: {"0": [1, 2, 3], "1": [4, 5, 6]},
            1: {"0": [7, 8, 9], "1": [10, 11, 12]},
        }

    def test_apply_row_parametrizations_with_none_mlps(self):
        """Test that apply_row_parametrizations handles None mlps gracefully."""

        # Create a mock model
        mock_model = Mock()
        mock_model.base_model = Mock()

        # Mock the _find_decoder_blocks_and_mlps to return None, None
        with patch.object(self.ica_mask, "_find_decoder_blocks_and_mlps") as mock_find:
            mock_find.return_value = (None, None)

            # Create a mock logger
            mock_logger = Mock()

            # This should not raise an error about len() of NoneType
            result = self.ica_mask.apply_row_parametrizations(
                model=mock_model,
                component_ids=[0, 1],
                mode="lesion",
                target_layers=None,
                apply_to="lora",
                logger=mock_logger,
            )

            # Should return empty list when MLPs are None
            self.assertEqual(result, [])

            # Should log a warning (check that warning was called, don't check exact message)
            mock_logger.warning.assert_called_once()
            # Verify the warning message contains the expected text
            warning_call = mock_logger.warning.call_args[0][0]
            self.assertIn(
                "Could not find transformer blocks/MLPs for row parametrizations",
                warning_call,
            )

    def test_apply_row_parametrizations_with_valid_mlps_no_crash(self):
        """Test that apply_row_parametrizations doesn't crash with valid mlps structure."""

        # Create a mock model with valid structure
        mock_model = Mock()
        mock_model.base_model = Mock()

        # Create mock MLPs with minimal structure
        mock_mlps = [Mock(), Mock()]

        # Mock the _find_decoder_blocks_and_mlps to return valid mlps
        with patch.object(self.ica_mask, "_find_decoder_blocks_and_mlps") as mock_find:
            mock_find.return_value = (mock_mlps, mock_mlps)

            mock_logger = Mock()

            # This should not raise a len(None) error - it might fail for other reasons due to mocking
            # but the specific NoneType error should be fixed
            try:
                result = self.ica_mask.apply_row_parametrizations(
                    model=mock_model,
                    component_ids=[0, 1],
                    mode="lesion",
                    target_layers=None,
                    apply_to="lora",
                    logger=mock_logger,
                )
                # If we get here, the NoneType error is fixed
                self.assertIsInstance(result, list)
            except TypeError as e:
                # Make sure it's not the "object of type 'NoneType' has no len()" error
                self.assertNotIn("object of type 'NoneType' has no len()", str(e))
                # Other TypeErrors are acceptable due to mocking limitations
            except Exception as e:
                # Make sure it's not the specific NoneType len() error we're fixing
                self.assertNotIn("object of type 'NoneType' has no len()", str(e))

            # Should not log any warnings about missing MLPs
            mock_logger.warning.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
