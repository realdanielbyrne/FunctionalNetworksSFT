#!/usr/bin/env python3
"""
CUDA Configuration Test Script

This script tests CUDA configuration and device assignment to verify
that GPU acceleration is properly set up for training.
"""

import logging
import sys
import torch
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from functionalnetworkssft.utils.model_utils import (
    get_optimal_device,
    get_recommended_dtype,
    is_quantization_supported,
)
from functionalnetworkssft.utils.platform_setup import (
    get_platform_info,
    check_cuda_availability,
    check_mps_availability,
    verify_installation,
    get_recommended_installation_command,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_basic_torch_cuda():
    """Test basic PyTorch CUDA functionality."""
    print("\n" + "=" * 60)
    print("BASIC PYTORCH CUDA TEST")
    print("=" * 60)
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Device count: {torch.cuda.device_count()}")
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name()}")
        print(f"Device capability: {torch.cuda.get_device_capability()}")
        print(f"Memory allocated: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
        print(f"Memory cached: {torch.cuda.memory_reserved() / 1024**2:.1f} MB")
        
        # Test basic tensor operations
        try:
            x = torch.randn(1000, 1000).cuda()
            y = torch.randn(1000, 1000).cuda()
            z = torch.matmul(x, y)
            print(f"✓ Basic CUDA tensor operations successful")
            print(f"  Result shape: {z.shape}")
            print(f"  Result device: {z.device}")
            del x, y, z
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"✗ Basic CUDA tensor operations failed: {e}")
            return False
    else:
        print("⚠ CUDA not available")
        
        # Check if NVIDIA GPU is present
        try:
            import subprocess
            result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
            if result.returncode == 0:
                print("⚠ NVIDIA GPU detected but PyTorch CUDA not available")
                print("  This suggests PyTorch was installed without CUDA support")
                return False
        except FileNotFoundError:
            print("  No NVIDIA GPU detected")
    
    return torch.cuda.is_available()


def test_device_utilities():
    """Test device detection utilities."""
    print("\n" + "=" * 60)
    print("DEVICE UTILITIES TEST")
    print("=" * 60)
    
    try:
        device, device_name = get_optimal_device()
        print(f"✓ Optimal device: {device} ({device_name})")
        
        dtype = get_recommended_dtype()
        print(f"✓ Recommended dtype: {dtype}")
        
        quant_supported = is_quantization_supported()
        print(f"✓ Quantization supported: {quant_supported}")
        
        return True
    except Exception as e:
        print(f"✗ Device utilities test failed: {e}")
        return False


def test_platform_detection():
    """Test platform detection and recommendations."""
    print("\n" + "=" * 60)
    print("PLATFORM DETECTION TEST")
    print("=" * 60)
    
    try:
        platform_info = get_platform_info()
        print(f"✓ Platform info: {platform_info}")
        
        cuda_available = check_cuda_availability()
        print(f"✓ CUDA availability check: {cuda_available}")
        
        mps_available = check_mps_availability()
        print(f"✓ MPS availability check: {mps_available}")
        
        recommended_cmd = get_recommended_installation_command()
        print(f"✓ Recommended installation: {recommended_cmd}")
        
        return True
    except Exception as e:
        print(f"✗ Platform detection test failed: {e}")
        return False


def test_model_device_assignment():
    """Test model device assignment."""
    print("\n" + "=" * 60)
    print("MODEL DEVICE ASSIGNMENT TEST")
    print("=" * 60)
    
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        # Use a small model for testing
        model_name = "microsoft/DialoGPT-small"
        
        print(f"Loading model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        device, device_name = get_optimal_device()
        print(f"Target device: {device_name}")
        
        # Load model with device mapping
        device_map = "auto" if device.type in ["cuda", "mps"] else None
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            torch_dtype=get_recommended_dtype()
        )
        
        # Move to device if device_map wasn't used
        if device_map is None:
            model = model.to(device)
        
        print(f"✓ Model loaded successfully")
        print(f"  Model device: {next(model.parameters()).device}")
        
        # Test inference
        inputs = tokenizer("Hello, how are you?", return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        print(f"✓ Model inference successful")
        print(f"  Output shape: {outputs.logits.shape}")
        print(f"  Output device: {outputs.logits.device}")
        
        # Clean up
        del model, tokenizer, inputs, outputs
        if device.type == "cuda":
            torch.cuda.empty_cache()
        
        return True
        
    except Exception as e:
        print(f"✗ Model device assignment test failed: {e}")
        return False


def main():
    """Run all CUDA configuration tests."""
    print("CUDA CONFIGURATION TEST SUITE")
    print("Testing device configuration and GPU acceleration setup...")
    
    tests = [
        ("Basic PyTorch CUDA", test_basic_torch_cuda),
        ("Device Utilities", test_device_utilities),
        ("Platform Detection", test_platform_detection),
        ("Model Device Assignment", test_model_device_assignment),
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            logger.error(f"Test '{test_name}' failed with exception: {e}")
            results[test_name] = False
    
    # Print summary
    print("\n" + "=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status} {test_name}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 All tests passed! CUDA configuration is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        print("\nTroubleshooting:")
        print("1. Ensure NVIDIA drivers are installed")
        print("2. Reinstall PyTorch with CUDA support:")
        print("   poetry install --extras cuda")
        print("3. Check CUDA compatibility with your GPU")
    
    # Print installation verification
    verification = verify_installation()
    print(f"\nInstallation Status:")
    for key, value in verification.items():
        status = "✓" if value else "✗"
        print(f"  {status} {key.replace('_', ' ').title()}")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
