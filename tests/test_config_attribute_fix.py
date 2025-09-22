#!/usr/bin/env python3
"""
Test script to verify that the config attribute access fix works.
This test ensures that the code handles config objects (like LlamaConfig) correctly.
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


class MockConfig:
    """Mock config object that mimics LlamaConfig behavior."""
    def __init__(self, model_type="llama"):
        self.model_type = model_type
        self.hidden_size = 4096


class TestConfigAttributeFix(unittest.TestCase):
    """Test that the config attribute access fix works."""

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

    def test_apply_row_parametrizations_with_config_object(self):
        """Test that apply_row_parametrizations handles config objects correctly."""
        
        # Create a mock model with a config object (not a dict)
        mock_model = Mock()
        mock_model.config = MockConfig(model_type="llama")
        mock_model.base_model = Mock()
        
        # Mock the _find_decoder_blocks_and_mlps to return None, None
        with patch.object(self.ica_mask, '_find_decoder_blocks_and_mlps') as mock_find:
            mock_find.return_value = (None, None)
            
            # Create a mock logger
            mock_logger = Mock()
            
            # This should not raise an error about 'LlamaConfig' object has no attribute 'get'
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
            
            # Should log a warning with correct model type
            mock_logger.warning.assert_called_once()
            warning_call = mock_logger.warning.call_args[0][0]
            self.assertIn("Could not find transformer blocks/MLPs for row parametrizations", warning_call)
            self.assertIn("Model type: llama", warning_call)

    def test_apply_row_parametrizations_with_no_config(self):
        """Test that apply_row_parametrizations handles models without config."""
        
        # Create a mock model without a config
        mock_model = Mock()
        del mock_model.config  # Remove config attribute
        mock_model.base_model = Mock()
        
        # Mock the _find_decoder_blocks_and_mlps to return None, None
        with patch.object(self.ica_mask, '_find_decoder_blocks_and_mlps') as mock_find:
            mock_find.return_value = (None, None)
            
            # Create a mock logger
            mock_logger = Mock()
            
            # This should not raise an error
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
            
            # Should log a warning with "unknown" model type
            mock_logger.warning.assert_called_once()
            warning_call = mock_logger.warning.call_args[0][0]
            self.assertIn("Could not find transformer blocks/MLPs for row parametrizations", warning_call)
            self.assertIn("Model type: unknown", warning_call)


if __name__ == "__main__":
    unittest.main(verbosity=2)
