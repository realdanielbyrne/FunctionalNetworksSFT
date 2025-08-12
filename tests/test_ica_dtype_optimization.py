#!/usr/bin/env python3
"""
Test ICA dtype optimization functionality.

This module tests the new ica_dtype parameter that allows reducing precision
for ICA computation to improve performance while maintaining reasonable accuracy.
"""

import unittest
import torch
import torch.nn as nn
from unittest.mock import Mock, patch
import numpy as np

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from functionalnetworkssft.ica_mask import ICAMask


class TestICADtypeOptimization(unittest.TestCase):
    """Test cases for ICA dtype optimization."""

    def setUp(self):
        """Set up test fixtures."""
        self.hidden_size = 768
        self.intermediate_size = 3072

        # Create a simple mock model with different dtypes
        self.model_f32 = Mock()
        self.model_f32.parameters.return_value = [
            torch.randn(10, 10, dtype=torch.float32)
        ]

        self.model_f16 = Mock()
        self.model_f16.parameters.return_value = [
            torch.randn(10, 10, dtype=torch.float16)
        ]

        self.model_bf16 = Mock()
        self.model_bf16.parameters.return_value = [
            torch.randn(10, 10, dtype=torch.bfloat16)
        ]

    def test_ica_dtype_default(self):
        """Test default ICA dtype behavior (should use float32)."""
        ica_mask = ICAMask()

        # Test with float32 model
        dtype = ica_mask._get_ica_dtype(torch.float32)
        self.assertEqual(dtype, torch.float32)

        # Test with float16 model (should still use float32 for stability)
        dtype = ica_mask._get_ica_dtype(torch.float16)
        self.assertEqual(dtype, torch.float32)

    def test_ica_dtype_auto(self):
        """Test auto ICA dtype behavior."""
        ica_mask = ICAMask(ica_dtype="auto")

        # Test with float32 model (should match)
        dtype = ica_mask._get_ica_dtype(torch.float32)
        self.assertEqual(dtype, torch.float32)

        # Test with float16 model (should use float32 for stability)
        dtype = ica_mask._get_ica_dtype(torch.float16)
        self.assertEqual(dtype, torch.float32)

        # Test with bfloat16 model (should use float32 for stability)
        dtype = ica_mask._get_ica_dtype(torch.bfloat16)
        self.assertEqual(dtype, torch.float32)

    def test_ica_dtype_explicit(self):
        """Test explicit ICA dtype settings."""
        # Test float16
        ica_mask = ICAMask(ica_dtype="float16")
        dtype = ica_mask._get_ica_dtype(torch.float32)
        self.assertEqual(dtype, torch.float16)

        # Test bfloat16
        ica_mask = ICAMask(ica_dtype="bfloat16")
        dtype = ica_mask._get_ica_dtype(torch.float32)
        self.assertEqual(dtype, torch.bfloat16)

        # Test float32
        ica_mask = ICAMask(ica_dtype="float32")
        dtype = ica_mask._get_ica_dtype(torch.float16)
        self.assertEqual(dtype, torch.float32)

    def test_ica_dtype_invalid(self):
        """Test invalid ICA dtype handling."""
        ica_mask = ICAMask(ica_dtype="invalid_dtype")

        with patch("functionalnetworkssft.ica_mask.logger") as mock_logger:
            dtype = ica_mask._get_ica_dtype(torch.float32)
            self.assertEqual(dtype, torch.float32)
            mock_logger.warning.assert_called_once()

    def test_dtype_memory_efficiency(self):
        """Test that using lower precision reduces memory usage."""
        # Create test data
        test_data = torch.randn(100, 1000, dtype=torch.float32)

        # Convert to different dtypes and check memory usage
        f32_data = test_data.to(torch.float32)
        f16_data = test_data.to(torch.float16)

        # float16 should use half the memory
        f32_size = f32_data.element_size() * f32_data.nelement()
        f16_size = f16_data.element_size() * f16_data.nelement()

        self.assertEqual(f16_size, f32_size // 2)

    def test_dtype_numerical_stability(self):
        """Test that different dtypes maintain reasonable numerical stability."""
        # Create test data with known properties
        np.random.seed(42)
        test_data = np.random.randn(50, 100).astype(np.float32)

        # Convert to different dtypes
        f32_tensor = torch.from_numpy(test_data)
        f16_tensor = f32_tensor.to(torch.float16)
        bf16_tensor = f32_tensor.to(torch.bfloat16)

        # Check that basic statistics are preserved reasonably well
        f32_mean = torch.mean(f32_tensor)
        f16_mean = torch.mean(f16_tensor.float())
        bf16_mean = torch.mean(bf16_tensor.float())

        # Allow for some precision loss but ensure it's reasonable
        self.assertAlmostEqual(f32_mean.item(), f16_mean.item(), places=2)
        self.assertAlmostEqual(f32_mean.item(), bf16_mean.item(), places=3)

    def test_ica_dtype_parameter_passing(self):
        """Test that ica_dtype parameter is properly stored and used."""
        # Test None (default)
        ica_mask = ICAMask()
        self.assertIsNone(ica_mask.ica_dtype)

        # Test explicit values
        ica_mask = ICAMask(ica_dtype="float16")
        self.assertEqual(ica_mask.ica_dtype, "float16")

        ica_mask = ICAMask(ica_dtype="auto")
        self.assertEqual(ica_mask.ica_dtype, "auto")


if __name__ == "__main__":
    unittest.main()
