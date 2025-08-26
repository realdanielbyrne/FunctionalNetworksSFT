#!/usr/bin/env python3
"""
CUDA Setup Script for FunctionalNetworksSFT

This script properly installs PyTorch with CUDA support using Poetry
and handles the platform-specific installation requirements.
"""

import logging
import subprocess
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from functionalnetworkssft.utils.platform_setup import (
    get_platform_info,
    check_cuda_availability,
    verify_installation,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def check_nvidia_gpu():
    """Check if NVIDIA GPU is available on the system."""
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False


def install_cuda_pytorch():
    """Install PyTorch with CUDA support using Poetry and pip."""
    logger.info("Installing PyTorch with CUDA support...")

    try:
        # First, remove existing PyTorch installations to avoid conflicts
        logger.info("Removing existing PyTorch installations...")
        subprocess.run(
            [
                "poetry",
                "run",
                "pip",
                "uninstall",
                "torch",
                "torchvision",
                "torchaudio",
                "-y",
            ],
            check=False,
        )  # Don't fail if packages aren't installed

        # Install PyTorch with CUDA from the official PyTorch index (CUDA 12.8 wheels)
        logger.info("Installing PyTorch with CUDA support (cu128 wheels)...")
        subprocess.run(
            [
                "poetry",
                "run",
                "pip",
                "install",
                "torch",
                "torchvision",
                "torchaudio",
                "--index-url",
                "https://download.pytorch.org/whl/cu128",
            ],
            check=True,
        )

        # Install CUDA-specific optional dependencies
        logger.info("Installing CUDA-specific dependencies...")
        try:
            subprocess.run(["poetry", "install", "--extras", "cuda"], check=True)
        except subprocess.CalledProcessError:
            logger.warning(
                "Some CUDA dependencies failed to install (this is normal for flash-attn on Windows)"
            )

        return True

    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install CUDA PyTorch: {e}")
        return False


def install_cpu_pytorch():
    """Install CPU-only PyTorch using Poetry."""
    logger.info("Installing CPU-only PyTorch...")

    try:
        subprocess.run(["poetry", "install", "--extras", "cpu"], check=True)
        return True

    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install CPU PyTorch: {e}")
        return False


def install_apple_silicon_pytorch():
    """Install PyTorch with MPS support for Apple Silicon."""
    logger.info("Installing PyTorch with MPS support for Apple Silicon...")

    try:
        subprocess.run(["poetry", "install", "--extras", "apple-silicon"], check=True)
        return True

    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install Apple Silicon PyTorch: {e}")
        return False


def main():
    """Main setup function."""
    print("=" * 60)
    print("FUNCTIONALNETWORKSSFT CUDA SETUP")
    print("=" * 60)

    # Get platform information
    platform_info = get_platform_info()
    logger.info(f"Platform: {platform_info}")

    # Determine the best installation approach
    if platform_info["is_apple_silicon"]:
        logger.info("Detected Apple Silicon - installing MPS-enabled PyTorch")
        success = install_apple_silicon_pytorch()
    elif check_nvidia_gpu():
        logger.info("Detected NVIDIA GPU - installing CUDA-enabled PyTorch")
        success = install_cuda_pytorch()
    else:
        logger.info("No GPU detected - installing CPU-only PyTorch")
        success = install_cpu_pytorch()

    if not success:
        print("\n❌ Installation failed!")
        return False

    # Verify the installation
    print("\n" + "=" * 60)
    print("VERIFYING INSTALLATION")
    print("=" * 60)

    verification = verify_installation()

    # Test PyTorch CUDA availability
    try:
        import torch

        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"Device count: {torch.cuda.device_count()}")
            print(f"Device name: {torch.cuda.get_device_name()}")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            print("MPS (Apple Silicon) available: True")
        else:
            print("Using CPU-only mode")
    except ImportError:
        print("❌ PyTorch not available")
        return False

    # Print verification results
    print("\nLibrary Status:")
    for key, value in verification.items():
        status = "✓" if value else "✗"
        print(f"  {status} {key.replace('_', ' ').title()}")

    print("\n✅ Setup completed successfully!")
    print("\nNext steps:")
    print("1. Run the CUDA configuration test:")
    print("   python tests/test_cuda_configuration.py")
    print("2. Start training with GPU acceleration")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
