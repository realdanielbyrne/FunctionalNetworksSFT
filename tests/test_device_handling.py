#!/usr/bin/env python3
"""
Comprehensive unit tests for device handling and CUDA support.

Tests verify:
- CUDA availability detection
- Device selection (GPU when available, CPU as fallback)
- Model and tensor placement on correct device
- Training functionality on both GPU and CPU modes
"""

import logging
import pytest
import torch
import torch.nn as nn
from unittest.mock import patch, MagicMock
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from functionalnetworkssft.utils.model_utils import (
    get_optimal_device,
    get_recommended_dtype,
    is_quantization_supported,
)

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestDeviceDetection:
    """Test device detection functionality."""

    def test_cuda_available_detection(self):
        """Test CUDA availability detection."""
        device, device_name = get_optimal_device()

        if torch.cuda.is_available():
            assert device.type == "cuda"
            assert "CUDA" in device_name
            assert "NVIDIA" in device_name or "GeForce" in device_name
            logger.info(f"OK CUDA detected: {device_name}")
        else:
            assert device.type == "cpu"
            assert "CPU" in device_name
            logger.info(f"OK CPU fallback: {device_name}")

    def test_device_consistency(self):
        """Test that device detection is consistent across calls."""
        device1, name1 = get_optimal_device()
        device2, name2 = get_optimal_device()

        assert device1.type == device2.type
        assert name1 == name2
        logger.info(f"OK Device detection consistent: {device1}")

    @patch("torch.cuda.is_available")
    def test_cuda_unavailable_fallback(self, mock_cuda_available):
        """Test fallback to CPU when CUDA is unavailable."""
        mock_cuda_available.return_value = False

        device, device_name = get_optimal_device()

        assert device.type == "cpu"
        assert "CPU" in device_name
        logger.info("OK CPU fallback works when CUDA unavailable")

    @patch("torch.backends.mps.is_available")
    @patch("torch.cuda.is_available")
    def test_mps_detection(self, mock_cuda_available, mock_mps_available):
        """Test MPS detection on Apple Silicon."""
        mock_cuda_available.return_value = False
        mock_mps_available.return_value = True

        device, device_name = get_optimal_device()

        assert device.type == "mps"
        assert "MPS" in device_name
        logger.info("OK MPS detection works")


class TestDataTypeRecommendation:
    """Test data type recommendation functionality."""

    def test_recommended_dtype_cuda(self):
        """Test recommended dtype for CUDA devices."""
        if torch.cuda.is_available():
            dtype = get_recommended_dtype()

            # Should be bfloat16 or float16 for CUDA
            assert dtype in [torch.bfloat16, torch.float16]

            if torch.cuda.is_bf16_supported():
                assert dtype == torch.bfloat16
                logger.info("OK bfloat16 recommended for CUDA with bf16 support")
            else:
                assert dtype == torch.float16
                logger.info("OK float16 recommended for CUDA without bf16 support")

    @patch("torch.cuda.is_available")
    def test_recommended_dtype_cpu(self, mock_cuda_available):
        """Test recommended dtype for CPU."""
        mock_cuda_available.return_value = False

        dtype = get_recommended_dtype()

        assert dtype == torch.float32
        logger.info("OK float32 recommended for CPU")


class TestTensorOperations:
    """Test tensor operations on detected device."""

    def test_tensor_creation_and_operations(self):
        """Test tensor creation and basic operations on optimal device."""
        device, device_name = get_optimal_device()

        # Create tensors on device
        x = torch.randn(100, 100, device=device)
        y = torch.randn(100, 100, device=device)

        # Verify tensors are on correct device
        assert x.device == device
        assert y.device == device

        # Test basic operations
        z = torch.matmul(x, y)
        assert z.device == device
        assert z.shape == (100, 100)

        # Test reduction operations
        mean_val = torch.mean(z)
        assert mean_val.device == device

        logger.info(f"OK Tensor operations successful on {device_name}")

    def test_tensor_device_transfer(self):
        """Test moving tensors between devices."""
        device, _ = get_optimal_device()

        # Create tensor on CPU
        x_cpu = torch.randn(50, 50)
        assert x_cpu.device.type == "cpu"

        # Move to optimal device
        x_device = x_cpu.to(device)
        assert x_device.device == device

        # Move back to CPU
        x_back = x_device.cpu()
        assert x_back.device.type == "cpu"

        # Verify data integrity
        assert torch.allclose(x_cpu, x_back)

        logger.info("OK Tensor device transfer works correctly")


class TestModelPlacement:
    """Test model placement on devices."""

    def test_simple_model_placement(self):
        """Test placing a simple model on the optimal device."""
        device, device_name = get_optimal_device()

        # Create a simple model
        model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 1))

        # Move model to device
        model = model.to(device)

        # Verify all parameters are on correct device
        for param in model.parameters():
            assert param.device == device

        # Test forward pass
        x = torch.randn(5, 10, device=device)
        y = model(x)

        assert y.device == device
        assert y.shape == (5, 1)

        logger.info(f"OK Model placement and forward pass successful on {device_name}")

    def test_model_training_step(self):
        """Test a complete training step on the optimal device."""
        device, device_name = get_optimal_device()

        # Create model, loss, and optimizer
        model = nn.Linear(10, 1).to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        # Create sample data
        x = torch.randn(32, 10, device=device)
        y_true = torch.randn(32, 1, device=device)

        # Training step
        optimizer.zero_grad()
        y_pred = model(x)
        loss = criterion(y_pred, y_true)
        loss.backward()
        optimizer.step()

        # Verify loss is computed
        assert loss.item() > 0
        assert loss.device == device

        logger.info(f"OK Training step successful on {device_name}")


class TestQuantizationSupport:
    """Test quantization support detection."""

    def test_quantization_support_detection(self):
        """Test quantization support detection."""
        is_supported = is_quantization_supported()

        if torch.cuda.is_available():
            # Should be True if bitsandbytes is installed and CUDA is available
            logger.info(f"OK Quantization support: {is_supported}")
        else:
            # Should be False on CPU-only systems
            assert not is_supported
            logger.info("OK Quantization correctly unavailable on CPU")


class TestMemoryManagement:
    """Test GPU memory management."""

    def test_memory_allocation_and_cleanup(self):
        """Test GPU memory allocation and cleanup."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        device = torch.device("cuda")

        # Check initial memory
        initial_memory = torch.cuda.memory_allocated()

        # Allocate large tensor
        large_tensor = torch.randn(1000, 1000, device=device)
        allocated_memory = torch.cuda.memory_allocated()

        assert allocated_memory > initial_memory

        # Clean up
        del large_tensor
        torch.cuda.empty_cache()

        # Memory should be freed (or close to initial)
        final_memory = torch.cuda.memory_allocated()
        assert final_memory <= allocated_memory

        logger.info("OK GPU memory management works correctly")

    def test_memory_info_access(self):
        """Test accessing GPU memory information."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        # Test memory info functions
        total_memory = torch.cuda.get_device_properties(0).total_memory
        allocated_memory = torch.cuda.memory_allocated()
        cached_memory = torch.cuda.memory_reserved()

        assert total_memory > 0
        assert allocated_memory >= 0
        assert cached_memory >= 0

        logger.info(f"OK GPU memory info: {total_memory / 1024**3:.1f} GB total")


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])
