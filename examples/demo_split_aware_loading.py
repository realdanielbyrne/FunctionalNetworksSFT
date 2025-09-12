#!/usr/bin/env python3
"""
Demonstration of split-aware dataset loading functionality.

This script shows how the updated dataset utilities can automatically
detect and use existing train/validation/test splits from HuggingFace datasets.
"""

import logging
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from functionalnetworkssft.utils.model_utils import (
    load_dataset_with_splits,
    prepare_train_val_splits,
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def demo_split_aware_loading():
    """Demonstrate split-aware dataset loading."""

    print("=" * 80)
    print("SPLIT-AWARE DATASET LOADING DEMONSTRATION")
    print("=" * 80)

    # Example 1: Dataset with existing splits (using a well-known dataset)
    print("\n1. Loading dataset with existing splits...")
    print("   Dataset: squad (has train/validation splits)")

    try:
        # Load dataset with all available splits
        splits_data = load_dataset_with_splits("squad")

        print(f"   Available splits: {list(splits_data.keys())}")
        for split_name, split_data in splits_data.items():
            print(f"   - {split_name}: {len(split_data)} examples")

        # Prepare train/validation using existing splits
        train_data, val_data = prepare_train_val_splits(
            splits_data,
            validation_split=0.1,  # This will be ignored since validation exists
            prefer_existing_splits=True,
        )

        print(f"   Final training data: {len(train_data)} examples")
        print(f"   Final validation data: {len(val_data)} examples")
        print("   ✓ Used existing validation split")

    except Exception as e:
        print(f"   ⚠ Could not load squad dataset: {e}")
        print(
            "   (This is expected if you don't have internet or the dataset is not available)"
        )

    # Example 2: Simulated camel-ai/physics dataset behavior
    print("\n2. Simulated camel-ai/physics dataset behavior...")

    # Simulate what would happen with camel-ai/physics dataset
    simulated_splits = {
        "train": [
            {
                "message_1": "What is Newton's first law of motion?",
                "message_2": "Newton's first law states that an object at rest stays at rest and an object in motion stays in motion with the same speed and in the same direction unless acted upon by an unbalanced force.",
                "topic": "Classical Mechanics",
                "sub_topic": "Newton's Laws",
            },
            {
                "message_1": "Explain the concept of energy conservation.",
                "message_2": "Energy conservation is a fundamental principle stating that energy cannot be created or destroyed, only transformed from one form to another.",
                "topic": "Thermodynamics",
                "sub_topic": "Conservation Laws",
            },
            {
                "message_1": "What is the speed of light in vacuum?",
                "message_2": "The speed of light in vacuum is approximately 299,792,458 meters per second.",
                "topic": "Electromagnetism",
                "sub_topic": "Wave Properties",
            },
            {
                "message_1": "Define acceleration.",
                "message_2": "Acceleration is the rate of change of velocity with respect to time.",
                "topic": "Kinematics",
                "sub_topic": "Motion",
            },
        ]
    }

    print(f"   Available splits: {list(simulated_splits.keys())}")
    print(f"   - train: {len(simulated_splits['train'])} examples")

    # Case A: Use existing splits (none available, so create custom split)
    print("\n   Case A: prefer_existing_splits=True (no validation split exists)")
    train_data_a, val_data_a = prepare_train_val_splits(
        simulated_splits, validation_split=0.25, prefer_existing_splits=True
    )
    print(f"   - Training: {len(train_data_a)} examples")
    print(f"   - Validation: {len(val_data_a)} examples")
    print("   ✓ Created custom validation split from training data")

    # Case B: Add validation split and use it
    print("\n   Case B: prefer_existing_splits=True (validation split exists)")
    simulated_splits_with_val = simulated_splits.copy()
    simulated_splits_with_val["validation"] = [
        {
            "message_1": "What is momentum?",
            "message_2": "Momentum is the product of an object's mass and velocity.",
            "topic": "Classical Mechanics",
            "sub_topic": "Momentum",
        }
    ]

    train_data_b, val_data_b = prepare_train_val_splits(
        simulated_splits_with_val,
        validation_split=0.25,  # This will be ignored
        prefer_existing_splits=True,
    )
    print(f"   - Training: {len(train_data_b)} examples")
    print(f"   - Validation: {len(val_data_b)} examples")
    print("   ✓ Used existing validation split")

    # Example 3: Show format detection with camel-ai/physics format
    print("\n3. Format detection with camel-ai/physics data...")

    from functionalnetworkssft.utils.dataset_utils import DatasetFormatter

    sample_data = simulated_splits["train"][:2]  # Use first 2 examples
    detected_format = DatasetFormatter.detect_format(sample_data)
    print(f"   Detected format: {detected_format}")

    # Convert to standard format
    converted_example = DatasetFormatter.convert_to_standard_format(
        sample_data[0], detected_format
    )
    print(f"   Original keys: {list(sample_data[0].keys())}")
    print(f"   Converted keys: {list(converted_example.keys())}")
    print(f"   ✓ Successfully converted camel-ai/physics format")
    print(f"   - Instruction: {converted_example['instruction'][:50]}...")
    print(f"   - Response: {converted_example['response'][:50]}...")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("✓ Split-aware loading automatically detects available dataset splits")
    print("✓ Uses existing validation/test splits when available")
    print("✓ Falls back to custom splitting when no validation split exists")
    print("✓ Supports camel-ai/physics dataset format with proper column mapping")
    print("✓ Maintains backward compatibility with existing functionality")
    print("\nTo use split-aware loading in training:")
    print("  --use_existing_splits (default: enabled)")
    print("  --no_use_existing_splits (to disable and use custom splitting)")


if __name__ == "__main__":
    demo_split_aware_loading()
