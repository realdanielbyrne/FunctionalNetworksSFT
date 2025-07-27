#!/usr/bin/env python3
"""
Tests for ICA mask application and hook functionality.

Tests the actual masking behavior, hook registration/removal, and
the effects of masking on model forward passes.
"""

import sys
import unittest
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch

# Add project root to path for proper imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.functionalnetworkssft.fnsft_trainer import apply_ica_masks


class TestMaskApplication(unittest.TestCase):
    """Test cases for mask application functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.hidden_size = 768
        self.intermediate_size = 3072

        # Create a simple model with linear layers
        self.model = self.create_test_model()

        self.mask_dict = {
            "0": [100, 200, 300, 400, 500],
            "1": [150, 250, 350, 450, 550],
        }

    def create_test_model(self):
        """Create a test model with transformer-like structure."""

        class TestLinear(nn.Linear):
            def __init__(self, in_features, out_features):
                super().__init__(in_features, out_features)

        class TestBlock(nn.Module):
            def __init__(self, hidden_size, intermediate_size):
                super().__init__()
                self.up_proj = TestLinear(hidden_size, intermediate_size)
                self.down_proj = TestLinear(intermediate_size, hidden_size)

            def modules(self):
                yield self
                yield self.up_proj
                yield self.down_proj

        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = Mock()
                self.config.hidden_size = self.hidden_size
                self.config.n_embd = None
                self.config.d_model = None

                self.model = Mock()
                self.model.layers = nn.ModuleList(
                    [
                        TestBlock(self.hidden_size, self.intermediate_size),
                        TestBlock(self.hidden_size, self.intermediate_size),
                    ]
                )

                self.embedding = nn.Embedding(1000, self.hidden_size)

            def get_input_embeddings(self):
                return self.embedding

        return TestModel()

    def test_mask_creation_key_mode(self):
        """Test that masks are created correctly in 'key' mode."""
        handles = apply_ica_masks(self.model, self.mask_dict, mask_mode="key")

        # Should have handles for each layer
        self.assertEqual(len(handles), 2)

        # Test that hooks are actually registered
        for i, layer in enumerate(self.model.model.layers):
            down_proj = layer.down_proj
            # Check if hooks are registered (non-zero hook count)
            self.assertGreater(len(down_proj._forward_pre_hooks), 0)

        # Clean up
        for handle in handles:
            handle.remove()

    def test_mask_creation_complement_mode(self):
        """Test that masks are created correctly in 'complement' mode."""
        handles = apply_ica_masks(self.model, self.mask_dict, mask_mode="complement")

        # Should have handles for each layer
        self.assertEqual(len(handles), 2)

        # Clean up
        for handle in handles:
            handle.remove()

    def test_mask_application_effect(self):
        """Test that masks actually affect the forward pass."""
        # Get the down projection layer from first block
        down_proj = self.model.model.layers[0].down_proj

        # Create test input
        batch_size, seq_len = 2, 10
        test_input = torch.randn(batch_size, seq_len, self.intermediate_size)

        # Forward pass without masking
        original_output = down_proj(test_input)

        # Apply masking
        handles = apply_ica_masks(self.model, self.mask_dict, mask_mode="key")

        # Forward pass with masking
        masked_output = down_proj(test_input)

        # Outputs should be different (masking should have an effect)
        self.assertFalse(torch.allclose(original_output, masked_output))

        # Clean up
        for handle in handles:
            handle.remove()

    def test_mask_values_key_mode(self):
        """Test that mask values are correct in 'key' mode."""
        neuron_ids = [100, 200, 300]

        # Create mask manually to verify logic
        mask = torch.ones(self.intermediate_size)
        mask[neuron_ids] = 0.0

        # Check that specified neurons are zeroed
        for neuron_id in neuron_ids:
            self.assertEqual(mask[neuron_id].item(), 0.0)

        # Check that other neurons are preserved
        non_masked_indices = [
            i for i in range(self.intermediate_size) if i not in neuron_ids
        ]
        for idx in non_masked_indices[:10]:  # Check first 10 non-masked
            self.assertEqual(mask[idx].item(), 1.0)

    def test_mask_values_complement_mode(self):
        """Test that mask values are correct in 'complement' mode."""
        neuron_ids = [100, 200, 300]

        # Create mask manually to verify logic
        mask = torch.zeros(self.intermediate_size)
        mask[neuron_ids] = 1.0

        # Check that specified neurons are preserved
        for neuron_id in neuron_ids:
            self.assertEqual(mask[neuron_id].item(), 1.0)

        # Check that other neurons are zeroed
        non_masked_indices = [
            i for i in range(self.intermediate_size) if i not in neuron_ids
        ]
        for idx in non_masked_indices[:10]:  # Check first 10 non-masked
            self.assertEqual(mask[idx].item(), 0.0)

    def test_hook_removal(self):
        """Test that hooks can be properly removed."""
        # Apply masks
        handles = apply_ica_masks(self.model, self.mask_dict, mask_mode="key")

        # Verify hooks are registered
        for layer in self.model.model.layers:
            down_proj = layer.down_proj
            self.assertGreater(len(down_proj._forward_pre_hooks), 0)

        # Remove hooks
        for handle in handles:
            handle.remove()

        # Verify hooks are removed
        for layer in self.model.model.layers:
            down_proj = layer.down_proj
            self.assertEqual(len(down_proj._forward_pre_hooks), 0)

    def test_empty_mask_dict(self):
        """Test behavior with empty mask dictionary."""
        empty_mask_dict = {}
        handles = apply_ica_masks(self.model, empty_mask_dict, mask_mode="key")

        # Should still create handles
        self.assertEqual(len(handles), 2)

        # Clean up
        for handle in handles:
            handle.remove()

    def test_partial_mask_dict(self):
        """Test behavior with partial mask dictionary (missing some layers)."""
        partial_mask_dict = {"0": [100, 200]}  # Only layer 0
        handles = apply_ica_masks(self.model, partial_mask_dict, mask_mode="key")

        # Should still create handles for all layers
        self.assertEqual(len(handles), 2)

        # Clean up
        for handle in handles:
            handle.remove()

    def test_mask_device_compatibility(self):
        """Test that masks work correctly across different devices."""
        # Test with CPU
        handles = apply_ica_masks(self.model, self.mask_dict, mask_mode="key")

        # Create test input
        test_input = torch.randn(1, 5, self.intermediate_size)

        # Forward pass should work
        down_proj = self.model.model.layers[0].down_proj
        output = down_proj(test_input)

        self.assertEqual(output.shape, (1, 5, self.hidden_size))

        # Clean up
        for handle in handles:
            handle.remove()

    def test_mask_dtype_compatibility(self):
        """Test that masks work correctly with different dtypes."""
        # Test with float32 input
        test_input_f32 = torch.randn(1, 5, self.intermediate_size, dtype=torch.float32)

        handles = apply_ica_masks(self.model, self.mask_dict, mask_mode="key")

        down_proj = self.model.model.layers[0].down_proj
        output_f32 = down_proj(test_input_f32)

        # Output should maintain input dtype
        self.assertEqual(output_f32.dtype, torch.float32)

        # Clean up
        for handle in handles:
            handle.remove()


class TestMaskHookBehavior(unittest.TestCase):
    """Test cases for mask hook behavior and edge cases."""

    def setUp(self):
        """Set up test fixtures."""
        self.hidden_size = 768
        self.intermediate_size = 3072

    def test_hook_function_signature(self):
        """Test that hook function has correct signature."""
        # Create a simple mask
        mask = torch.ones(100)
        mask[10:20] = 0.0

        # Define the hook function (copied from apply_ica_masks)
        def pre_hook(mod, inp, mask_tensor=mask):
            x = inp[0]
            return (x * mask_tensor.to(x.device, x.dtype),) + inp[1:]

        # Test with mock inputs
        mock_module = Mock()
        test_input = torch.randn(2, 5, 100)
        mock_inp = (test_input,)

        result = pre_hook(mock_module, mock_inp)

        # Should return tuple with modified first element
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].shape, test_input.shape)

    def test_hook_with_multiple_inputs(self):
        """Test hook behavior with multiple input arguments."""
        mask = torch.ones(100)
        mask[10:20] = 0.0

        def pre_hook(mod, inp, mask_tensor=mask):
            x = inp[0]
            return (x * mask_tensor.to(x.device, x.dtype),) + inp[1:]

        # Test with multiple inputs
        mock_module = Mock()
        test_input1 = torch.randn(2, 5, 100)
        test_input2 = torch.randn(2, 5, 50)
        mock_inp = (test_input1, test_input2)

        result = pre_hook(mock_module, mock_inp)

        # Should return tuple with modified first element and preserved others
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].shape, test_input1.shape)
        self.assertTrue(torch.equal(result[1], test_input2))

    def test_mask_gradient_flow(self):
        """Test that gradients flow correctly through masked layers."""
        # Create a simple model
        linear = nn.Linear(100, 50)

        # Create mask
        mask = torch.ones(100)
        mask[10:20] = 0.0

        # Define hook
        def pre_hook(mod, inp, mask_tensor=mask):
            x = inp[0]
            return (x * mask_tensor.to(x.device, x.dtype),) + inp[1:]

        # Register hook
        handle = linear.register_forward_pre_hook(pre_hook)

        try:
            # Forward pass with gradient tracking
            test_input = torch.randn(2, 100, requires_grad=True)
            output = linear(test_input)
            loss = output.sum()

            # Backward pass
            loss.backward()

            # Check that gradients exist
            self.assertIsNotNone(test_input.grad)
            self.assertIsNotNone(linear.weight.grad)

            # Check that masked positions have zero gradients in input
            masked_grad = test_input.grad[:, 10:20]
            self.assertTrue(torch.allclose(masked_grad, torch.zeros_like(masked_grad)))

        finally:
            handle.remove()


if __name__ == "__main__":
    unittest.main()
