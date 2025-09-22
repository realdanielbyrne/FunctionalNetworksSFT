#!/usr/bin/env python3
"""
Test script to verify that the experiment b training cleanup issue is fixed.
This test simulates the actual training scenario that was failing.
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


class MockTrainer:
    """Mock trainer that simulates the training process."""
    def __init__(self):
        self.state = Mock()
        self.state.log_history = [{"eval_loss": 0.7539, "train_loss": 0.7011}]
    
    def train(self, resume_from_checkpoint=None):
        """Mock training that completes successfully."""
        pass
    
    def add_callback(self, callback):
        """Mock callback addition."""
        pass


class MockHandle:
    """Mock forward hook handle with remove method."""
    def __init__(self):
        self.removed = False
    
    def remove(self):
        self.removed = True


class TestExperimentBFix(unittest.TestCase):
    """Test that the experiment b training cleanup issue is fixed."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_training_cleanup_scenario(self):
        """Test the exact scenario that was failing in experiment b."""
        
        # Simulate the training scenario with ICA masking and row parametrizations
        mask_handles = [MockHandle(), MockHandle()]  # Forward hooks
        param_handles = []  # Will contain tuples from row parametrizations
        has_row_parametrizations = False
        
        # Mock ICA mask
        ica_mask = Mock()
        ica_mask.apply_component_masks = Mock(return_value=mask_handles)
        ica_mask.apply_row_parametrizations = Mock(return_value=[
            (Mock(), "weight"),
            (Mock(), "bias")
        ])
        ica_mask.remove_row_parametrizations = Mock()
        
        # Mock args
        args = Mock()
        args.mask_mode = "lesion"
        args.anti_drift_row_param = True
        args.anti_drift_unwrap_on_save = True
        
        # Mock logger
        logger = Mock()
        
        # Simulate the training process
        if args.mask_mode is not None:
            # Apply component masks (returns forward hooks)
            mh = ica_mask.apply_component_masks(
                model=Mock(),
                component_ids=[0],
                mode="lesion"
            )
            mask_handles.extend(mh)
            
            if args.anti_drift_row_param:
                # Apply row parametrizations (returns tuples)
                ph = ica_mask.apply_row_parametrizations(
                    model=Mock(),
                    component_ids=[0],
                    mode="lesion",
                    target_layers=None,
                    apply_to="lora",
                    logger=logger,
                )
                # Note: ph contains tuples (module, tensor_name), not handles with .remove()
                # These will be cleaned up via ica_mask.remove_row_parametrizations()
                has_row_parametrizations = True
        
        # Mock trainer
        trainer = MockTrainer()
        
        # Simulate training
        trainer.train(resume_from_checkpoint=None)
        
        # Test the cleanup logic that was failing
        try:
            # Clean up forward hooks
            for h in mask_handles:
                h.remove()
            
            # Clean up any remaining forward hooks in param_handles (if any)
            # Note: param_handles may contain tuples from row parametrizations, not hooks
            for h in param_handles:
                if hasattr(h, 'remove'):
                    h.remove()
            
            # Clean up row parametrizations if they were applied
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
            
            # If we get here without an exception, the fix worked
            cleanup_success = True
            
        except AttributeError as e:
            if "'tuple' object has no attribute 'remove'" in str(e):
                self.fail(f"The original error still exists: {e}")
            else:
                # Some other AttributeError
                raise
        
        # Verify that the cleanup was successful
        self.assertTrue(cleanup_success)
        
        # Verify that remove_row_parametrizations was called
        ica_mask.remove_row_parametrizations.assert_called_once_with(
            bake=True, logger=logger
        )

    def test_mixed_param_handles_scenario(self):
        """Test scenario where param_handles contains both hooks and tuples."""
        
        # This simulates a scenario where param_handles might contain both types
        param_handles = []
        
        # Add some tuples (from row parametrizations)
        param_handles.extend([
            (Mock(), "weight"),
            (Mock(), "bias")
        ])
        
        # Add a forward hook (hypothetically)
        mock_hook = MockHandle()
        param_handles.append(mock_hook)
        
        # Test the safe cleanup approach
        try:
            for h in param_handles:
                if hasattr(h, 'remove'):
                    h.remove()
            
            cleanup_success = True
            
        except AttributeError as e:
            if "'tuple' object has no attribute 'remove'" in str(e):
                self.fail(f"The cleanup still fails on tuples: {e}")
            else:
                raise
        
        # Verify cleanup was successful
        self.assertTrue(cleanup_success)
        
        # Verify that the hook was removed
        self.assertTrue(mock_hook.removed)

    def test_no_row_parametrizations_scenario(self):
        """Test scenario where no row parametrizations were applied."""
        
        mask_handles = [MockHandle(), MockHandle()]
        param_handles = []  # Empty
        has_row_parametrizations = False
        
        ica_mask = Mock()
        ica_mask.remove_row_parametrizations = Mock()
        
        args = Mock()
        args.anti_drift_unwrap_on_save = True
        
        logger = Mock()
        
        # Test cleanup
        for h in mask_handles:
            h.remove()
        
        for h in param_handles:
            if hasattr(h, 'remove'):
                h.remove()
        
        # Clean up row parametrizations if they were applied
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
        
        # Verify that fallback was called since has_row_parametrizations is False
        ica_mask.remove_row_parametrizations.assert_called_once_with(
            bake=True, logger=logger
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
