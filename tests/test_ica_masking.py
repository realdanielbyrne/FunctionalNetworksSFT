#!/usr/bin/env python3
"""
Unit tests for ICA masking functionality in FunctionalNetworksSFT.

Tests the ICA-based functional network masking features including:
- apply_ica_masks function
- compute_ica_masks_for_model function
- Command line argument integration
- Mask application and removal
"""

import sys
import unittest
import tempfile
import json
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import torch
import torch.nn as nn
import numpy as np

# Add project root to path for proper imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.functionalnetworkssft.sft_trainer import (
    apply_ica_masks,
    compute_ica_masks_for_model,
)


class MockLinearModule(nn.Module):
    """Mock linear module for testing."""

    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))

    def forward(self, x):
        return torch.nn.functional.linear(x, self.weight, self.bias)


class MockTransformerBlock(nn.Module):
    """Mock transformer block for testing."""

    def __init__(self, hidden_size=768, intermediate_size=3072):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        # MLP layers - up projection and down projection
        self.up_proj = nn.Linear(hidden_size, intermediate_size)
        self.down_proj = nn.Linear(intermediate_size, hidden_size)


class MockModel(nn.Module):
    """Mock model for testing ICA masking."""

    def __init__(self, num_layers=2, hidden_size=768, intermediate_size=3072):
        super().__init__()
        self.config = Mock()
        self.config.hidden_size = hidden_size
        self.config.n_embd = None
        self.config.d_model = None

        # Create mock transformer structure that matches what apply_ica_masks expects
        self.model = nn.Module()
        self.model.layers = nn.ModuleList(
            [
                MockTransformerBlock(hidden_size, intermediate_size)
                for _ in range(num_layers)
            ]
        )

        # Mock embedding layer
        self.embedding = nn.Embedding(1000, hidden_size)

    def get_input_embeddings(self):
        return self.embedding

    def parameters(self):
        """Return parameters for device detection."""
        return self.embedding.parameters()

    def eval(self):
        """Set model to eval mode."""
        return super().eval()

    def forward(self, input_ids, attention_mask=None):
        """Mock forward pass."""
        batch_size, seq_len = input_ids.shape
        return torch.randn(batch_size, seq_len, self.config.hidden_size)


class MockDataset:
    """Mock dataset for testing."""

    def __init__(self, size=100, seq_len=128, hidden_size=768):
        self.size = size
        self.seq_len = seq_len
        self.hidden_size = hidden_size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return {
            "input_ids": torch.randint(0, 1000, (self.seq_len,)),
            "attention_mask": torch.ones(self.seq_len),
            "labels": torch.randint(0, 1000, (self.seq_len,)),
        }


class TestICAMasking(unittest.TestCase):
    """Test cases for ICA masking functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.hidden_size = 768
        self.intermediate_size = 3072
        self.num_layers = 2

        self.model = MockModel(
            num_layers=self.num_layers,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
        )

        self.mask_dict = {"0": [100, 200, 300], "1": [150, 250, 350]}

        self.dataset = MockDataset(size=50)
        self.tokenizer = Mock()

    def test_apply_ica_masks_key_mode(self):
        """Test applying ICA masks in 'key' mode (ablate key neurons)."""
        handles = apply_ica_masks(self.model, self.mask_dict, mask_mode="key")

        # Should return handles for each layer
        self.assertEqual(len(handles), self.num_layers)

        # Verify hooks are attached
        for handle in handles:
            self.assertIsNotNone(handle)

        # Clean up
        for handle in handles:
            handle.remove()

    def test_apply_ica_masks_complement_mode(self):
        """Test applying ICA masks in 'complement' mode (keep only key neurons)."""
        handles = apply_ica_masks(self.model, self.mask_dict, mask_mode="complement")

        # Should return handles for each layer
        self.assertEqual(len(handles), self.num_layers)

        # Clean up
        for handle in handles:
            handle.remove()

    def test_apply_ica_masks_empty_dict(self):
        """Test applying ICA masks with empty mask dictionary."""
        empty_mask_dict = {}
        handles = apply_ica_masks(self.model, empty_mask_dict, mask_mode="key")

        # Should still return handles (but masks will be all ones/zeros)
        self.assertEqual(len(handles), self.num_layers)

        # Clean up
        for handle in handles:
            handle.remove()

    def test_apply_ica_masks_invalid_model(self):
        """Test applying ICA masks to model without transformer blocks."""
        invalid_model = nn.Linear(10, 10)

        # This should raise an AttributeError because Linear doesn't have config
        with self.assertRaises(AttributeError):
            apply_ica_masks(invalid_model, self.mask_dict, mask_mode="key")


class TestICAMaskComputation(unittest.TestCase):
    """Test cases for ICA mask computation."""

    def setUp(self):
        """Set up test fixtures."""
        self.model = MockModel(num_layers=2, hidden_size=768, intermediate_size=3072)
        self.dataset = MockDataset(size=20)
        self.tokenizer = Mock()

    @patch("src.functionalnetworkssft.sft_trainer.FastICA")
    @patch("torch.utils.data.DataLoader")
    def test_compute_ica_masks_for_model(self, mock_dataloader, mock_fastica):
        """Test computing ICA masks for model."""
        # Mock FastICA
        mock_ica_instance = Mock()
        mock_ica_instance.fit_transform.return_value = np.random.randn(100, 20)
        mock_ica_instance.mixing_ = np.random.randn(3072, 20)
        mock_fastica.return_value = mock_ica_instance

        # Mock DataLoader
        mock_samples = [
            {
                "input_ids": torch.randint(0, 1000, (128,)),
                "attention_mask": torch.ones(128),
            }
            for _ in range(10)
        ]
        mock_dataloader.return_value = iter(mock_samples)

        # Mock model forward pass to return activations
        with patch.object(self.model, "forward") as mock_forward:
            mock_forward.return_value = torch.randn(1, 128, 768)

            result = compute_ica_masks_for_model(
                self.model,
                self.dataset,
                self.tokenizer,
                num_components=20,
                percentile=98.0,
                sample_batches=10,
            )

        # Should return a dictionary with layer indices as keys
        self.assertIsInstance(result, dict)

        # Keys should be strings (layer indices)
        for key in result.keys():
            self.assertIsInstance(key, str)

        # Values should be lists of integers (neuron indices)
        for value in result.values():
            self.assertIsInstance(value, list)
            for neuron_idx in value:
                self.assertIsInstance(neuron_idx, int)


class TestICAMaskingIntegration(unittest.TestCase):
    """Integration tests for ICA masking with command line arguments."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.mask_file_path = os.path.join(self.temp_dir, "test_mask.json")

        # Create test mask file
        test_mask = {"0": [100, 200, 300], "1": [150, 250, 350], "2": [50, 150, 250]}

        with open(self.mask_file_path, "w") as f:
            json.dump(test_mask, f)

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir)

    def test_mask_file_loading(self):
        """Test loading mask dictionary from JSON file."""
        with open(self.mask_file_path, "r") as f:
            loaded_mask = json.load(f)

        expected_mask = {
            "0": [100, 200, 300],
            "1": [150, 250, 350],
            "2": [50, 150, 250],
        }

        self.assertEqual(loaded_mask, expected_mask)

    def test_mask_file_format_validation(self):
        """Test validation of mask file format."""
        # Test valid format
        with open(self.mask_file_path, "r") as f:
            mask_dict = json.load(f)

        # Validate format
        for layer_idx, neuron_list in mask_dict.items():
            self.assertIsInstance(layer_idx, str)
            self.assertIsInstance(neuron_list, list)
            for neuron_idx in neuron_list:
                self.assertIsInstance(neuron_idx, int)
                self.assertGreaterEqual(neuron_idx, 0)


if __name__ == "__main__":
    unittest.main()
