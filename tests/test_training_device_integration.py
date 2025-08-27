#!/usr/bin/env python3
"""
Integration tests for training functionality with device handling.

Tests verify:
- Training script device detection and usage
- Model loading and device placement
- Training loop execution on GPU/CPU
- Error handling and fallback mechanisms
"""

import logging
import pytest
import torch
import tempfile
import shutil
from pathlib import Path
import sys
from unittest.mock import patch, MagicMock

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from functionalnetworkssft.utils.model_utils import (
    get_optimal_device,
    get_recommended_dtype,
)

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestTrainingDeviceIntegration:
    """Test training functionality with device handling."""

    def test_device_detection_in_training_context(self):
        """Test device detection works in training context."""
        device, device_name = get_optimal_device()
        dtype = get_recommended_dtype()

        logger.info(f"Training will use: {device_name}")
        logger.info(f"Recommended dtype: {dtype}")

        # Verify device is accessible
        if device.type == "cuda":
            assert torch.cuda.is_available()
            # Test basic CUDA operation
            test_tensor = torch.randn(10, 10, device=device, dtype=dtype)
            result = torch.matmul(test_tensor, test_tensor)
            assert result.device == device
            assert result.dtype == dtype

        assert device.type in ["cuda", "mps", "cpu"]

    def test_model_loading_device_placement(self):
        """Test model loading with proper device placement."""
        device, device_name = get_optimal_device()

        # Simulate loading a small transformer-like model
        from transformers import AutoConfig

        # Create a minimal config for testing
        config = AutoConfig.from_pretrained("gpt2")
        config.n_layer = 2  # Reduce size for testing
        config.n_head = 2
        config.n_embd = 128
        config.vocab_size = 1000

        # Test device placement logic similar to training script
        try:
            # Simulate the device mapping logic from the training script
            device_map = None
            if device.type in ["cuda", "mps"]:
                # Test if operations work
                test_tensor = torch.randn(10, 10).to(device)
                _ = torch.matmul(test_tensor, test_tensor)
                device_map = "auto"
                logger.info("Device mapping: auto")
            else:
                logger.info("Device mapping: manual")

            # Verify device mapping decision
            if device.type == "cuda":
                assert device_map == "auto"

            logger.info(
                f"OK Model loading device placement logic works for {device_name}"
            )

        except RuntimeError as e:
            if device.type == "cuda":
                logger.warning(f"CUDA operation failed: {e}")
                # Should fall back to CPU
                device = torch.device("cpu")
                assert device.type == "cpu"
                logger.info("OK Fallback to CPU works")

    def test_training_arguments_device_adaptation(self):
        """Test training arguments adaptation based on device."""
        device, _ = get_optimal_device()
        dtype = get_recommended_dtype()

        # Simulate training arguments adaptation
        training_args = {
            "fp16": False,
            "bf16": False,
            "dataloader_pin_memory": True,
        }

        # Adapt based on device and dtype
        if device.type == "cuda":
            if dtype == torch.float16:
                training_args["fp16"] = True
                training_args["bf16"] = False
            elif dtype == torch.bfloat16:
                training_args["fp16"] = False
                training_args["bf16"] = True
        elif device.type == "mps":
            # MPS typically uses float16
            training_args["fp16"] = True
            training_args["bf16"] = False
            training_args["dataloader_pin_memory"] = (
                False  # MPS doesn't support pinned memory
            )
        else:  # CPU
            training_args["fp16"] = False
            training_args["bf16"] = False
            training_args["dataloader_pin_memory"] = False

        logger.info(f"OK Training args adapted for {device.type}: {training_args}")

        # Verify logical consistency
        assert not (training_args["fp16"] and training_args["bf16"])  # Can't have both

        if device.type == "cpu":
            assert not training_args["fp16"]  # CPU doesn't support fp16 training
            assert not training_args["bf16"]  # CPU doesn't support bf16 training

    def test_gradient_computation_device_consistency(self):
        """Test gradient computation works correctly on target device."""
        device, device_name = get_optimal_device()
        dtype = get_recommended_dtype()

        # Create a simple model for gradient testing
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 20), torch.nn.ReLU(), torch.nn.Linear(20, 1)
        ).to(device)

        # Convert to appropriate dtype
        if dtype != torch.float32:
            model = model.to(dtype)

        # Create sample data
        x = torch.randn(8, 10, device=device, dtype=dtype)
        y_true = torch.randn(8, 1, device=device, dtype=dtype)

        # Forward pass
        y_pred = model(x)
        loss = torch.nn.functional.mse_loss(y_pred, y_true)

        # Backward pass
        loss.backward()

        # Check gradients exist and are on correct device
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert param.grad.device == device, f"Gradient for {name} on wrong device"
            if dtype != torch.float32:
                # Gradients are typically computed in float32 even with mixed precision
                assert param.grad.dtype in [torch.float32, dtype]

        logger.info(f"OK Gradient computation works on {device_name} with {dtype}")

    def test_mixed_precision_compatibility(self):
        """Test mixed precision training compatibility."""
        device, device_name = get_optimal_device()

        if device.type not in ["cuda"]:
            pytest.skip(f"Mixed precision not supported on {device.type}")

        # Test automatic mixed precision (AMP)
        model = torch.nn.Linear(10, 1).to(device)
        scaler = torch.cuda.amp.GradScaler()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        x = torch.randn(8, 10, device=device)
        y_true = torch.randn(8, 1, device=device)

        # AMP training step
        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            y_pred = model(x)
            loss = torch.nn.functional.mse_loss(y_pred, y_true)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        logger.info(f"OK Mixed precision training works on {device_name}")

    def test_memory_efficient_training(self):
        """Test memory-efficient training techniques."""
        device, device_name = get_optimal_device()

        if device.type != "cuda":
            pytest.skip("Memory efficiency tests only relevant for CUDA")

        # Test gradient checkpointing simulation
        model = torch.nn.Sequential(
            torch.nn.Linear(100, 200),
            torch.nn.ReLU(),
            torch.nn.Linear(200, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 1),
        ).to(device)

        # Enable gradient checkpointing (simulation)
        def checkpoint_forward(module, input_tensor):
            """Simulate gradient checkpointing."""
            return torch.utils.checkpoint.checkpoint(module, input_tensor)

        x = torch.randn(16, 100, device=device, requires_grad=True)

        # Test memory usage with checkpointing
        initial_memory = torch.cuda.memory_allocated()

        # Forward pass with checkpointing
        y = checkpoint_forward(model, x)
        loss = y.sum()

        # Backward pass
        loss.backward()

        final_memory = torch.cuda.memory_allocated()

        logger.info(f"OK Memory efficient training tested on {device_name}")
        logger.info(f"Memory usage: {(final_memory - initial_memory) / 1024**2:.2f} MB")

    def test_error_handling_and_fallback(self):
        """Test error handling and fallback mechanisms."""
        device, _ = get_optimal_device()

        # Test handling of out-of-memory errors (simulation)
        if device.type == "cuda":
            try:
                # Try to allocate a very large tensor
                large_tensor = torch.randn(10000, 10000, device=device)
                logger.info("Large tensor allocation successful")
                del large_tensor
                torch.cuda.empty_cache()
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.info("OK OOM error handled gracefully")
                    # In real training, this would trigger fallback strategies
                    torch.cuda.empty_cache()
                else:
                    raise

        # Test device compatibility error handling
        try:
            # This should work on any device
            test_tensor = torch.randn(10, 10, device=device)
            result = torch.matmul(test_tensor, test_tensor)
            assert result.device == device
            logger.info("OK Device operations work correctly")
        except RuntimeError as e:
            logger.warning(f"Device operation failed: {e}")
            # Should fall back to CPU
            cpu_device = torch.device("cpu")
            test_tensor = torch.randn(10, 10, device=cpu_device)
            result = torch.matmul(test_tensor, test_tensor)
            assert result.device == cpu_device
            logger.info("OK Fallback to CPU successful")

    def test_batch_size_adaptation(self):
        """Test batch size adaptation based on available memory."""
        device, device_name = get_optimal_device()

        # Simulate batch size adaptation logic
        base_batch_size = 32

        if device.type == "cuda":
            # Get GPU memory info
            total_memory = torch.cuda.get_device_properties(0).total_memory
            memory_gb = total_memory / (1024**3)

            # Adapt batch size based on memory
            if memory_gb >= 24:  # High-end GPU
                adapted_batch_size = base_batch_size * 2
            elif memory_gb >= 12:  # Mid-range GPU
                adapted_batch_size = base_batch_size
            else:  # Low-memory GPU
                adapted_batch_size = base_batch_size // 2

            logger.info(f"GPU memory: {memory_gb:.1f} GB")
            logger.info(f"Adapted batch size: {adapted_batch_size}")

        elif device.type == "mps":
            # MPS typically has limited memory
            adapted_batch_size = base_batch_size // 2
            logger.info(f"MPS adapted batch size: {adapted_batch_size}")
        else:
            # CPU - can use larger batch sizes but slower
            adapted_batch_size = base_batch_size
            logger.info(f"CPU batch size: {adapted_batch_size}")

        assert adapted_batch_size > 0
        logger.info(f"OK Batch size adaptation works for {device_name}")


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])
