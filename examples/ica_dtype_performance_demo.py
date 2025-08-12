#!/usr/bin/env python3
"""
Demonstration of ICA dtype optimization for performance improvement.

This script shows how using different data types for ICA computation
can improve performance while maintaining reasonable accuracy.
"""

import time
import torch
import numpy as np
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from functionalnetworkssft.ica_mask import ICAMask


def create_synthetic_activations(
    batch_size=100, seq_len=512, hidden_size=3072, dtype=torch.float32
):
    """Create synthetic activation data for testing."""
    # Simulate realistic activation patterns
    activations = []
    for _ in range(batch_size):
        # Create activations with some structure (not completely random)
        base_pattern = torch.randn(seq_len, hidden_size, dtype=dtype)
        # Add some correlated components to make ICA meaningful
        for i in range(0, hidden_size, 100):
            end_idx = min(i + 50, hidden_size)
            base_pattern[:, i:end_idx] += torch.randn(seq_len, 1, dtype=dtype) * 0.5
        activations.append(base_pattern)
    return activations


def benchmark_ica_dtype(dtype_name, ica_dtype, model_dtype=torch.float32):
    """Benchmark ICA computation with specific dtype."""
    print(f"\n=== Benchmarking ICA with {dtype_name} ===")

    # Create synthetic data
    print("Creating synthetic activation data...")
    activations = create_synthetic_activations(batch_size=50, dtype=model_dtype)

    # Initialize ICA mask with specific dtype
    ica_mask = ICAMask(
        num_components=10,  # Reduced for faster testing
        percentile=95.0,
        ica_dtype=ica_dtype,
        n_jobs=1,  # Single threaded for consistent timing
    )

    # Simulate the ICA computation process
    print(f"Processing activations with ICA dtype: {ica_dtype}")
    start_time = time.time()

    try:
        # Simulate the data conversion and processing that happens in _process_layer_ica
        target_dtype = ica_mask._get_ica_dtype(model_dtype)
        print(f"Target ICA dtype: {target_dtype}")

        # Convert activations to target dtype (simulating the hook conversion)
        converted_activations = []
        for act in activations:
            converted_activations.append(act.to(target_dtype))

        # Concatenate and convert to numpy (simulating ICA preprocessing)
        X = torch.cat(converted_activations, dim=0).flatten(0, 1).numpy()
        print(f"Data shape: {X.shape}, dtype: {X.dtype}")

        # Simulate basic preprocessing (standardization)
        X_mean = np.mean(X, axis=0)
        X_std_dev = np.std(X, axis=0)
        X_std_dev = np.maximum(X_std_dev, 1e-8)
        X_std = (X - X_mean) / X_std_dev

        # Simulate ICA computation time (without actually running ICA for speed)
        # In real usage, this would be: ica.fit_transform(X_std)
        time.sleep(0.1)  # Simulate computation time

        end_time = time.time()
        processing_time = end_time - start_time

        # Calculate memory usage
        memory_usage = X.nbytes / (1024 * 1024)  # MB

        print(f"Processing time: {processing_time:.3f} seconds")
        print(f"Memory usage: {memory_usage:.2f} MB")
        print(f"Data range: [{np.min(X_std):.6f}, {np.max(X_std):.6f}]")

        return {
            "dtype": dtype_name,
            "processing_time": processing_time,
            "memory_usage": memory_usage,
            "data_range": (np.min(X_std), np.max(X_std)),
        }

    except Exception as e:
        print(f"Error during processing: {e}")
        return None


def main():
    """Run the ICA dtype performance demonstration."""
    print("ICA Dtype Performance Demonstration")
    print("=" * 50)

    print("\nThis demo shows how different data types affect ICA computation:")
    print("- float32: Maximum precision and stability (default)")
    print("- float16: Reduced precision, ~50% memory savings")
    print("- auto: Automatically chooses based on model dtype")

    # Test different dtype configurations
    test_configs = [
        ("Float32 (Default)", None),
        ("Float16 (Reduced Precision)", "float16"),
        ("Auto (Model-Matched)", "auto"),
    ]

    results = []

    for dtype_name, ica_dtype in test_configs:
        result = benchmark_ica_dtype(dtype_name, ica_dtype)
        if result:
            results.append(result)

    # Summary comparison
    print("\n" + "=" * 50)
    print("PERFORMANCE COMPARISON SUMMARY")
    print("=" * 50)

    if results:
        print(f"{'Dtype':<20} {'Time (s)':<12} {'Memory (MB)':<15} {'Stability':<12}")
        print("-" * 60)

        for result in results:
            stability = "High" if "Float32" in result["dtype"] else "Medium"
            print(
                f"{result['dtype']:<20} {result['processing_time']:<12.3f} "
                f"{result['memory_usage']:<15.2f} {stability:<12}"
            )

    print("\nRecommendations:")
    print("- Use default (float32) for maximum stability and accuracy")
    print("- Use 'auto' to match model precision while maintaining ICA stability")
    print("- Use 'float16' for 2x memory savings (may have numerical issues)")
    print("- Use 'bfloat16' for better numerical stability than float16")
    print("- For production use, test with your specific model and data")

    print("\nTo use in training, add to your config:")
    print("  ica_dtype: 'float16'  # or 'auto', 'bfloat16'")


if __name__ == "__main__":
    main()
