#!/usr/bin/env python3
"""
Installation verification script for FunctionalNetworksSFT.

This script tests that all dependencies are properly installed and
that cross-platform functionality is working correctly.
"""

import sys
import logging
from pathlib import Path

# Add project root to path for proper imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def test_basic_imports():
    """Test that all required libraries can be imported."""
    logger.info("Testing basic imports...")

    try:
        import torch

        logger.info(f"✓ PyTorch {torch.__version__}")
    except ImportError as e:
        logger.error(f"✗ PyTorch import failed: {e}")
        return False

    try:
        import transformers

        logger.info(f"✓ Transformers {transformers.__version__}")
    except ImportError as e:
        logger.error(f"✗ Transformers import failed: {e}")
        return False

    try:
        import peft

        logger.info(f"✓ PEFT {peft.__version__}")
    except ImportError as e:
        logger.error(f"✗ PEFT import failed: {e}")
        return False

    try:
        import datasets

        logger.info(f"✓ Datasets {datasets.__version__}")
    except ImportError as e:
        logger.error(f"✗ Datasets import failed: {e}")
        return False

    return True


def test_cross_platform_utilities():
    """Test cross-platform device detection and configuration."""
    logger.info("Testing cross-platform utilities...")

    try:
        from utils.model_utils import (
            get_optimal_device,
            get_recommended_dtype,
            is_quantization_supported,
        )

        device, device_name = get_optimal_device()
        logger.info(f"✓ Optimal device: {device} ({device_name})")

        dtype = get_recommended_dtype()
        logger.info(f"✓ Recommended dtype: {dtype}")

        quant_supported = is_quantization_supported()
        logger.info(f"✓ Quantization supported: {quant_supported}")

        return True

    except Exception as e:
        logger.error(f"✗ Cross-platform utilities failed: {e}")
        return False


def test_quantization_config():
    """Test quantization configuration loading."""
    logger.info("Testing quantization configuration...")

    try:
        from utils.model_utils import load_quantization_config

        # Test with quantization enabled
        config = load_quantization_config(use_4bit=True)
        if config is not None:
            logger.info("✓ 4-bit quantization config loaded")
        else:
            logger.info("✓ Quantization disabled (expected on non-CUDA platforms)")

        # Test with quantization disabled
        config = load_quantization_config(use_4bit=False, use_8bit=False)
        if config is None:
            logger.info("✓ Quantization properly disabled")

        return True

    except Exception as e:
        logger.error(f"✗ Quantization config test failed: {e}")
        return False


def test_platform_detection():
    """Test platform detection functionality."""
    logger.info("Testing platform detection...")

    try:
        from utils.platform_setup import (
            get_platform_info,
            check_cuda_availability,
            check_mps_availability,
        )

        platform_info = get_platform_info()
        logger.info(f"✓ Platform: {platform_info['system']} {platform_info['machine']}")

        cuda_available = check_cuda_availability()
        logger.info(f"✓ CUDA available: {cuda_available}")

        mps_available = check_mps_availability()
        logger.info(f"✓ MPS available: {mps_available}")

        return True

    except Exception as e:
        logger.error(f"✗ Platform detection failed: {e}")
        return False


def test_model_loading():
    """Test basic model loading functionality."""
    logger.info("Testing model loading (this may take a moment)...")

    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        # Use a very small model for testing
        model_name = "microsoft/DialoGPT-small"

        logger.info(f"Loading tokenizer for {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        logger.info(f"Loading model for {model_name}...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto" if torch.cuda.is_available() else None,
        )

        logger.info("✓ Model and tokenizer loaded successfully")

        # Clean up
        del model
        del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return True

    except Exception as e:
        logger.error(f"✗ Model loading failed: {e}")
        return False


def main():
    """Run all tests and report results."""
    logger.info("=" * 60)
    logger.info("FUNCTIONALNETWORKSSFT INSTALLATION TEST")
    logger.info("=" * 60)

    tests = [
        ("Basic Imports", test_basic_imports),
        ("Cross-Platform Utilities", test_cross_platform_utilities),
        ("Quantization Configuration", test_quantization_config),
        ("Platform Detection", test_platform_detection),
        ("Model Loading", test_model_loading),
    ]

    results = []

    for test_name, test_func in tests:
        logger.info(f"\n--- {test_name} ---")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            logger.error(f"Test {test_name} crashed: {e}")
            results.append((test_name, False))

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)

    passed = 0
    for test_name, success in results:
        status = "PASS" if success else "FAIL"
        logger.info(f"{test_name}: {status}")
        if success:
            passed += 1

    logger.info(f"\nPassed: {passed}/{len(results)} tests")

    if passed == len(results):
        logger.info("🎉 All tests passed! Installation is working correctly.")
        return 0
    else:
        logger.error("❌ Some tests failed. Please check the installation.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
