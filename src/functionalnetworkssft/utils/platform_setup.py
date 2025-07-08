#!/usr/bin/env python3
"""
Platform-specific setup utilities for cross-platform ML training.

This module handles platform detection and conditional dependency installation
for both CUDA-enabled systems and Apple Silicon Macs.
"""

import logging
import platform
import subprocess
import sys
from typing import List, Optional

logger = logging.getLogger(__name__)


def get_platform_info() -> dict:
    """
    Get detailed platform information.

    Returns:
        Dictionary containing platform details
    """
    info = {
        "system": platform.system(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "is_apple_silicon": False,
        "is_windows": False,
        "is_linux": False,
        "is_macos": False,
    }

    # Detect Apple Silicon
    if info["system"] == "Darwin":
        info["is_macos"] = True
        if info["machine"] in ["arm64", "aarch64"]:
            info["is_apple_silicon"] = True
    elif info["system"] == "Windows":
        info["is_windows"] = True
    elif info["system"] == "Linux":
        info["is_linux"] = True

    return info


def check_cuda_availability() -> bool:
    """
    Check if CUDA is available on the system.

    Returns:
        True if CUDA is available, False otherwise
    """
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        logger.warning("PyTorch not installed - cannot check CUDA availability")
        return False


def check_mps_availability() -> bool:
    """
    Check if MPS (Metal Performance Shaders) is available.

    Returns:
        True if MPS is available, False otherwise
    """
    try:
        import torch

        return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    except ImportError:
        logger.warning("PyTorch not installed - cannot check MPS availability")
        return False


def get_recommended_torch_index_url() -> Optional[str]:
    """
    Get the recommended PyTorch index URL based on platform.

    Returns:
        PyTorch index URL or None for default PyPI
    """
    platform_info = get_platform_info()

    if platform_info["is_windows"] or platform_info["is_linux"]:
        # For CUDA support on Windows/Linux
        return "https://download.pytorch.org/whl/cu121"
    elif platform_info["is_apple_silicon"]:
        # Apple Silicon uses default PyPI (includes MPS support)
        return None
    else:
        # Default PyPI for other platforms
        return None


def install_platform_dependencies() -> bool:
    """
    Install platform-specific dependencies using Poetry.

    Returns:
        True if installation successful, False otherwise
    """
    platform_info = get_platform_info()
    logger.info(f"Installing dependencies for platform: {platform_info}")

    try:
        # Install base dependencies (already done if this script is running)
        logger.info("Base dependencies should already be installed")

        # Install platform-specific optional dependencies
        if platform_info["is_apple_silicon"]:
            logger.info("Installing Apple Silicon optimizations...")
            try:
                subprocess.run(
                    ["poetry", "install", "--extras", "apple-silicon"], check=True
                )
            except subprocess.CalledProcessError:
                logger.warning("Failed to install Apple Silicon extras - continuing")
        else:
            # Try to install CUDA dependencies
            logger.info("Attempting to install CUDA dependencies...")
            try:
                subprocess.run(["poetry", "install", "--extras", "cuda"], check=True)
                logger.info("CUDA dependencies installed successfully")
            except subprocess.CalledProcessError:
                logger.warning(
                    "Failed to install CUDA dependencies - continuing without quantization"
                )

        return True

    except Exception as e:
        logger.error(f"Failed to install dependencies: {e}")
        return False


def verify_installation() -> dict:
    """
    Verify that the installation is working correctly.

    Returns:
        Dictionary with verification results
    """
    results = {
        "torch_available": False,
        "transformers_available": False,
        "peft_available": False,
        "datasets_available": False,
        "cuda_available": False,
        "mps_available": False,
        "bitsandbytes_available": False,
    }

    # Check PyTorch
    try:
        import torch

        results["torch_available"] = True
        results["cuda_available"] = torch.cuda.is_available()
        results["mps_available"] = (
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        )
        logger.info(f"PyTorch {torch.__version__} available")
    except ImportError:
        logger.error("PyTorch not available")

    # Check Transformers
    try:
        import transformers

        results["transformers_available"] = True
        logger.info(f"Transformers {transformers.__version__} available")
    except ImportError:
        logger.error("Transformers not available")

    # Check PEFT
    try:
        import peft

        results["peft_available"] = True
        logger.info(f"PEFT {peft.__version__} available")
    except ImportError:
        logger.error("PEFT not available")

    # Check Datasets
    try:
        import datasets

        results["datasets_available"] = True
        logger.info(f"Datasets {datasets.__version__} available")
    except ImportError:
        logger.error("Datasets not available")

    # Check BitsAndBytes (optional)
    try:
        import bitsandbytes

        results["bitsandbytes_available"] = True
        logger.info(f"BitsAndBytes {bitsandbytes.__version__} available")
    except ImportError:
        logger.info("BitsAndBytes not available (expected on Apple Silicon)")

    return results


def print_platform_summary():
    """Print a summary of the platform and available accelerators."""
    platform_info = get_platform_info()
    verification = verify_installation()

    print("\n" + "=" * 60)
    print("FUNCTIONAL NETWORKS SFT - PLATFORM SUMMARY")
    print("=" * 60)

    print(f"System: {platform_info['system']} {platform_info['machine']}")
    print(f"Python: {platform_info['python_version']}")

    if platform_info["is_apple_silicon"]:
        print("Platform: Apple Silicon Mac")
    elif platform_info["is_windows"]:
        print("Platform: Windows")
    elif platform_info["is_linux"]:
        print("Platform: Linux")

    print("\nAccelerator Support:")
    if verification["cuda_available"]:
        print("✓ CUDA GPU acceleration available")
    elif verification["mps_available"]:
        print("✓ Apple Silicon MPS acceleration available")
    else:
        print("⚠ CPU-only mode (no GPU acceleration)")

    print("\nLibrary Status:")
    for lib, available in verification.items():
        if lib.endswith("_available") and not lib.startswith(("cuda", "mps")):
            lib_name = lib.replace("_available", "").replace("_", " ").title()
            status = "✓" if available else "✗"
            print(f"{status} {lib_name}")

    print("\nQuantization Support:")
    if verification["bitsandbytes_available"]:
        print("✓ BitsAndBytes quantization available")
    else:
        print("⚠ Quantization not available (requires CUDA)")

    print("=" * 60)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Setting up platform-specific dependencies...")
    success = install_platform_dependencies()

    if success:
        print("Installation completed successfully!")
        print_platform_summary()
    else:
        print("Installation failed!")
        sys.exit(1)
