#!/usr/bin/env python3
"""
Test cases for the estimate_masked_params_for_lora_down_proj method in ICAMask.
"""

import unittest
from unittest.mock import Mock, patch
import torch
import torch.nn as nn
from src.functionalnetworkssft.ica_mask import ICAMask


class MockDownProj(nn.Module):
    """Mock down projection layer for testing."""

    def __init__(self, hidden_size=768, intermediate_size=3072, has_lora=False):
        super().__init__()
        self.out_features = hidden_size
        self.weight = nn.Parameter(torch.randn(hidden_size, intermediate_size))
        self.weight.requires_grad = True

        if has_lora:
            # Mock LoRA structure
            self.lora_B = {"default": Mock()}
            # Mock LoRA weight with proper shape
            lora_weight = torch.randn(hidden_size, 16)  # rank=16
            self.lora_B["default"].weight = lora_weight


class MockMLP(nn.Module):
    """Mock MLP layer for testing."""

    def __init__(self, hidden_size=768, intermediate_size=3072, has_lora=False):
        super().__init__()
        self.down_proj = MockDownProj(hidden_size, intermediate_size, has_lora)


class MockLayer(nn.Module):
    """Mock transformer layer for testing."""

    def __init__(self, hidden_size=768, intermediate_size=3072, has_lora=False):
        super().__init__()
        self.mlp = MockMLP(hidden_size, intermediate_size, has_lora)


class MockModel(nn.Module):
    """Mock model for testing."""

    def __init__(
        self, num_layers=2, hidden_size=768, intermediate_size=3072, has_lora=False
    ):
        super().__init__()
        self.config = Mock()
        self.config.hidden_size = hidden_size
        self.config.n_embd = None
        self.config.d_model = None

        # Create mock transformer structure
        self.model = nn.Module()
        self.model.layers = nn.ModuleList(
            [
                MockLayer(hidden_size, intermediate_size, has_lora)
                for _ in range(num_layers)
            ]
        )


class TestEstimateMaskedParams(unittest.TestCase):
    """Test cases for estimate_masked_params_for_lora_down_proj method."""

    def setUp(self):
        """Set up test fixtures."""
        self.ica_mask = ICAMask()
        self.hidden_size = 768
        self.intermediate_size = 3072

    def test_estimate_with_no_model(self):
        """Test that method handles None model gracefully."""
        result = self.ica_mask.estimate_masked_params_for_lora_down_proj(
            None, {}, "lesion"
        )
        self.assertEqual(result, 0)

    def test_estimate_with_empty_mask_union(self):
        """Test that method handles empty mask_union."""
        model = MockModel()
        result = self.ica_mask.estimate_masked_params_for_lora_down_proj(
            model, {}, "lesion"
        )
        self.assertEqual(result, 0)

    def test_estimate_lesion_mode_no_lora(self):
        """Test estimation in lesion mode without LoRA."""
        model = MockModel(num_layers=2, has_lora=False)
        mask_union = {
            "0": [100, 200, 300],  # 3 channels masked in layer 0
            "1": [150, 250],  # 2 channels masked in layer 1
        }

        result = self.ica_mask.estimate_masked_params_for_lora_down_proj(
            model, mask_union, "lesion"
        )

        # For lesion mode: masked_rows = len(chans)
        # Layer 0: 3 masked rows * intermediate_size = 3 * 3072 = 9216
        # Layer 1: 2 masked rows * intermediate_size = 2 * 3072 = 6144
        # Total: 15360
        expected = 3 * self.intermediate_size + 2 * self.intermediate_size
        self.assertEqual(result, expected)

    def test_estimate_preserve_mode_no_lora(self):
        """Test estimation in preserve mode without LoRA."""
        model = MockModel(num_layers=2, has_lora=False)
        mask_union = {
            "0": [100, 200, 300],  # 3 channels preserved in layer 0
            "1": [150, 250],  # 2 channels preserved in layer 1
        }

        result = self.ica_mask.estimate_masked_params_for_lora_down_proj(
            model, mask_union, "preserve"
        )

        # For preserve mode: masked_rows = hidden_size - len(chans)
        # Layer 0: (768 - 3) masked rows * intermediate_size = 765 * 3072
        # Layer 1: (768 - 2) masked rows * intermediate_size = 766 * 3072
        expected = (self.hidden_size - 3) * self.intermediate_size + (
            self.hidden_size - 2
        ) * self.intermediate_size
        self.assertEqual(result, expected)

    def test_estimate_with_lora(self):
        """Test estimation with LoRA layers."""
        model = MockModel(num_layers=1, has_lora=True)
        mask_union = {"0": [100, 200]}  # 2 channels masked in layer 0

        result = self.ica_mask.estimate_masked_params_for_lora_down_proj(
            model, mask_union, "lesion"
        )

        # For LoRA: masked_rows * rank = 2 * 16 = 32
        expected = 2 * 16  # 2 masked rows * rank 16
        self.assertEqual(result, expected)

    def test_estimate_with_invalid_model_structure(self):
        """Test that method handles models without proper structure."""
        # Model without config
        invalid_model = nn.Linear(10, 10)

        result = self.ica_mask.estimate_masked_params_for_lora_down_proj(
            invalid_model, {"0": [1, 2]}, "lesion"
        )
        self.assertEqual(result, 0)

    def test_estimate_with_peft_wrapped_model(self):
        """Test estimation with PEFT-wrapped model."""
        # Create a mock PEFT model structure
        base_model = MockModel(num_layers=1, has_lora=True)
        peft_model = Mock()
        peft_model.base_model = Mock()
        peft_model.base_model.model = base_model
        peft_model.config = base_model.config

        mask_union = {"0": [100]}

        result = self.ica_mask.estimate_masked_params_for_lora_down_proj(
            peft_model, mask_union, "lesion"
        )

        # Should unwrap and process the base model
        expected = 1 * 16  # 1 masked row * rank 16
        self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()
