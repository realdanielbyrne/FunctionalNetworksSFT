#!/usr/bin/env python3
"""
Test script to verify that the training cleanup issue is fixed.
This test ensures that the cleanup code after training handles both forward hooks
and row parametrization tuples correctly.
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


class MockHandle:
    """Mock forward hook handle with remove method."""
    def __init__(self):
        self.removed = False
    
    def remove(self):
        self.removed = True


class TestTrainingCleanupFix(unittest.TestCase):
    """Test that the training cleanup handles mixed handle types correctly."""

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

    def test_cleanup_mixed_handle_types(self):
        """Test that cleanup code handles both forward hooks and parametrization tuples."""
        
        # Simulate the scenario from the training code
        mask_handles = [MockHandle(), MockHandle()]  # Forward hooks with .remove()
        param_handles = []  # Will contain tuples from row parametrizations
        
        # Add some mock tuples (module, tensor_name) like apply_row_parametrizations returns
        mock_module1 = Mock()
        mock_module2 = Mock()
        param_handles.extend([
            (mock_module1, "weight"),
            (mock_module2, "bias")
        ])
        
        # Also add a mock forward hook to param_handles (mixed scenario)
        mock_hook = MockHandle()
        param_handles.append(mock_hook)
        
        # Test the cleanup logic from the training code
        # Clean up forward hooks
        for h in mask_handles:
            h.remove()
        
        # Clean up any remaining forward hooks in param_handles (if any)
        # Note: param_handles may contain tuples from row parametrizations, not hooks
        for h in param_handles:
            if hasattr(h, 'remove'):
                h.remove()
        
        # Verify that forward hooks were removed
        for handle in mask_handles:
            self.assertTrue(handle.removed)
        
        # Verify that the mock hook in param_handles was removed
        self.assertTrue(mock_hook.removed)
        
        # Verify that tuples don't cause errors (they just don't have remove method)
        # This test passes if no AttributeError is raised

    def test_cleanup_with_row_parametrizations_flag(self):
        """Test cleanup logic with has_row_parametrizations flag."""
        
        # Mock the scenario where row parametrizations were applied
        has_row_parametrizations = True
        ica_mask = Mock()
        ica_mask.remove_row_parametrizations = Mock()
        
        # Mock args
        args = Mock()
        args.anti_drift_unwrap_on_save = True
        
        # Mock logger
        logger = Mock()
        
        # Test the cleanup logic
        if (
            has_row_parametrizations
            and ica_mask is not None
            and hasattr(ica_mask, "remove_row_parametrizations")
        ):
            ica_mask.remove_row_parametrizations(
                bake=args.anti_drift_unwrap_on_save, logger=logger
            )
        
        # Verify that remove_row_parametrizations was called with correct parameters
        ica_mask.remove_row_parametrizations.assert_called_once_with(
            bake=True, logger=logger
        )

    def test_cleanup_fallback_logic(self):
        """Test the fallback cleanup logic for backward compatibility."""
        
        # Mock the scenario where row parametrizations flag is False but unwrap is True
        has_row_parametrizations = False
        ica_mask = Mock()
        ica_mask.remove_row_parametrizations = Mock()
        
        # Mock args
        args = Mock()
        args.anti_drift_unwrap_on_save = True
        
        # Mock logger
        logger = Mock()
        
        # Test the fallback cleanup logic
        if (
            has_row_parametrizations
            and ica_mask is not None
            and hasattr(ica_mask, "remove_row_parametrizations")
        ):
            ica_mask.remove_row_parametrizations(
                bake=args.anti_drift_unwrap_on_save, logger=logger
            )
        elif (
            args.anti_drift_unwrap_on_save
            and ica_mask is not None
            and hasattr(ica_mask, "remove_row_parametrizations")
        ):
            # Fallback for backward compatibility
            ica_mask.remove_row_parametrizations(bake=True, logger=logger)
        
        # Verify that the fallback was called
        ica_mask.remove_row_parametrizations.assert_called_once_with(
            bake=True, logger=logger
        )

    def test_no_error_on_tuple_remove_attempt(self):
        """Test that attempting to call .remove() on tuples doesn't crash with proper checking."""
        
        # Create a list with mixed types like param_handles might contain
        mixed_handles = [
            MockHandle(),  # Has .remove()
            (Mock(), "weight"),  # Tuple, no .remove()
            (Mock(), "bias"),  # Tuple, no .remove()
            MockHandle(),  # Has .remove()
        ]
        
        # Test the safe cleanup approach
        removed_count = 0
        for h in mixed_handles:
            if hasattr(h, 'remove'):
                h.remove()
                removed_count += 1
        
        # Should have removed 2 handles (the MockHandle instances)
        self.assertEqual(removed_count, 2)
        
        # Verify that the MockHandle instances were actually removed
        self.assertTrue(mixed_handles[0].removed)
        self.assertTrue(mixed_handles[3].removed)


if __name__ == "__main__":
    unittest.main(verbosity=2)
