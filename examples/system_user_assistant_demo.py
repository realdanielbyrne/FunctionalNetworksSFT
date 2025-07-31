#!/usr/bin/env python3
"""
Demonstration of the new system-user-assistant format support in DatasetFormatter.

This script shows how the DatasetFormatter can automatically detect and convert
the system-user-assistant format used by datasets like "Josephgflowers/Finance-Instruct-500k".
"""

import json
import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from functionalnetworkssft.utils.dataset_utils import DatasetFormatter


def demo_system_user_assistant_format():
    """Demonstrate the system-user-assistant format conversion."""
    print("System-User-Assistant Format Demo")
    print("=" * 50)

    # Sample data in the format used by Josephgflowers/Finance-Instruct-500k
    sample_data = [
        {
            "system": "You are a financial advisor with expertise in investment strategies.",
            "user": "What is the difference between stocks and bonds?",
            "assistant": "Stocks represent ownership shares in a company, while bonds are debt securities where you lend money to an entity. Stocks typically offer higher potential returns but with greater risk, whereas bonds generally provide more stable, predictable income.",
        },
        {
            "system": "You are a helpful assistant specializing in mathematics.",
            "user": "Can you explain compound interest?",
            "assistant": "Compound interest is interest calculated on the initial principal and also on the accumulated interest from previous periods. It's often called 'interest on interest' and can significantly increase your returns over time.",
        },
        {
            "system": "",  # Empty system field
            "user": "What is 2 + 2?",
            "assistant": "2 + 2 equals 4.",
        },
        {
            "system": None,  # Null system field
            "user": "What is the capital of France?",
            "assistant": "The capital of France is Paris.",
        },
        {
            "system": None,  # Another null system field to test
            "user": "What is machine learning?",
            "assistant": "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed.",
        },
    ]

    print("1. Format Detection")
    print("-" * 30)

    # Detect the format
    detected_format = DatasetFormatter.detect_format(sample_data)
    print(f"Detected format: {detected_format}")
    print()

    print("2. Sample Conversions")
    print("-" * 30)

    for i, item in enumerate(sample_data, 1):
        print(f"Sample {i}:")
        print(f"Original: {json.dumps(item, indent=2)}")

        # Convert to standard format
        converted = DatasetFormatter.convert_to_standard_format(item, detected_format)
        print(f"Converted:")
        print(f"  Instruction: {repr(converted['instruction'])}")
        print(f"  Response: {repr(converted['response'])}")
        print()

    print("3. Key Features Demonstrated")
    print("-" * 30)
    print("✓ Automatic detection of system-user-assistant format")
    print("✓ Proper handling of empty system fields (replaced with default)")
    print("✓ Proper handling of null system fields (replaced with default)")
    print("✓ Proper handling of missing system fields (replaced with default)")
    print("✓ Combination of system prompt with user input")
    print("✓ Conversion to standard instruction-response format")
    print()

    print("4. Default System Prompt")
    print("-" * 30)
    print("When system field is empty, null, or missing, it's replaced with:")
    print(
        "'You are a helpful assistant which thinks step by step when responding to a user.'"
    )
    print()

    print("5. Testing Missing System Field")
    print("-" * 30)

    # Test with data that has missing system field
    missing_system_data = [
        {
            "user": "What is machine learning?",
            "assistant": "Machine learning is a subset of artificial intelligence.",
        }
    ]

    # This will be detected as a different format, but we can still convert it
    print("Data with missing system field:")
    print(f"Original: {json.dumps(missing_system_data[0], indent=2)}")

    # Manually convert using the system-user-assistant format
    converted_missing = DatasetFormatter._convert_system_user_assistant_format(
        missing_system_data[0]
    )
    print(f"Converted:")
    print(f"  Instruction: {repr(converted_missing['instruction'])}")
    print(f"  Response: {repr(converted_missing['response'])}")
    print()

    print("6. Integration with Fine-tuning")
    print("-" * 30)
    print("This format can now be used directly with the fine-tuning trainer:")
    print("python -m src.functionalnetworkssft.fnsft_trainer \\")
    print("  --model_name_or_path gpt2 \\")
    print("  --dataset_name_or_path your_system_user_assistant_data.json \\")
    print("  --output_dir ./output \\")
    print("  --auto_detect_format")


if __name__ == "__main__":
    demo_system_user_assistant_format()
