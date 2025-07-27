#!/usr/bin/env python3
"""
Demonstration of the intelligent chat template handling feature.

This script shows how the new --template_format option works with different
template formats and tokenizers.
"""

import json
import tempfile
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from transformers import AutoTokenizer
from src.functionalnetworkssft.sft_trainer import InstructionDataset


def create_sample_data():
    """Create sample instruction-response data."""
    return [
        {
            "instruction": "What is the capital of France?",
            "response": "The capital of France is Paris.",
        },
        {
            "instruction": "Explain photosynthesis briefly.",
            "response": "Photosynthesis is the process by which plants convert sunlight, carbon dioxide, and water into glucose and oxygen.",
        },
        {
            "instruction": "How do you make a paper airplane?",
            "response": "To make a paper airplane: 1) Fold a sheet of paper in half lengthwise, 2) Unfold and fold the top corners to the center crease, 3) Fold the angled edges to the center again, 4) Fold the plane in half, 5) Create wings by folding each side down to align with the bottom.",
        },
    ]


def demo_template_formats():
    """Demonstrate different template formats."""
    print("🚀 Intelligent Chat Template Handling Demo")
    print("=" * 50)

    # Create sample data
    data = create_sample_data()

    # Test with a tokenizer that has a chat template (e.g., Llama-2-Chat)
    print("\n1. Testing with a tokenizer that has a chat template:")
    print("-" * 50)

    try:
        # Try to load a tokenizer with chat template
        tokenizer_with_chat = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
        # Add a mock chat template for demonstration
        tokenizer_with_chat.chat_template = "{% for message in messages %}{{ message['role'] }}: {{ message['content'] }}\n{% endfor %}"

        # Test auto-detection (should use chat template)
        dataset_auto = InstructionDataset(
            data=data[:1],  # Just one sample for demo
            tokenizer=tokenizer_with_chat,
            template_format="auto",
            auto_detect_format=False,
        )
        print(f"Auto-detection result: {dataset_auto.actual_template_format}")

        # Show formatted text
        formatted_text = dataset_auto._format_text(
            data[0]["instruction"], data[0]["response"]
        )
        print(f"Formatted text:\n{formatted_text}\n")

    except Exception as e:
        print(f"Could not load tokenizer with chat template: {e}")

    # Test with a basic tokenizer (no chat template)
    print("2. Testing with a basic tokenizer (no chat template):")
    print("-" * 50)

    try:
        tokenizer_basic = AutoTokenizer.from_pretrained("gpt2")

        # Test auto-detection (should fall back to basic)
        dataset_auto_basic = InstructionDataset(
            data=data[:1],
            tokenizer=tokenizer_basic,
            template_format="auto",
            auto_detect_format=False,
        )
        print(f"Auto-detection result: {dataset_auto_basic.actual_template_format}")

        # Show formatted text
        formatted_text = dataset_auto_basic._format_text(
            data[0]["instruction"], data[0]["response"]
        )
        print(f"Formatted text:\n{formatted_text}\n")

    except Exception as e:
        print(f"Could not load basic tokenizer: {e}")

    # Demonstrate different explicit formats
    print("3. Testing explicit template formats:")
    print("-" * 50)

    # Use a simple mock tokenizer for format demonstrations
    class MockTokenizer:
        def __init__(self):
            self.pad_token = None
            self.eos_token = "</s>"

    mock_tokenizer = MockTokenizer()

    formats = ["alpaca", "chatml", "basic"]
    custom_template = "Question: {instruction}\nAnswer: {response}"

    for fmt in formats:
        print(f"\n{fmt.upper()} format:")
        dataset = InstructionDataset(
            data=data[:1],
            tokenizer=mock_tokenizer,
            template_format=fmt,
            instruction_template=custom_template,
            auto_detect_format=False,
        )
        formatted_text = dataset._format_text(
            data[0]["instruction"], data[0]["response"]
        )
        print(f"{formatted_text}")


def demo_cli_usage():
    """Show CLI usage examples."""
    print("\n4. CLI Usage Examples:")
    print("-" * 50)

    examples = [
        {
            "description": "Auto-detect template format",
            "command": "python -m src.functionalnetworkssft.fnsft_trainer --model_name_or_path microsoft/DialoGPT-medium --dataset_name_or_path data.json --output_dir ./output --template_format auto",
        },
        {
            "description": "Force chat template usage",
            "command": "python -m src.functionalnetworkssft.fnsft_trainer --model_name_or_path microsoft/DialoGPT-medium --dataset_name_or_path data.json --output_dir ./output --template_format chat",
        },
        {
            "description": "Use Alpaca format",
            "command": "python -m src.functionalnetworkssft.fnsft_trainer --model_name_or_path gpt2 --dataset_name_or_path data.json --output_dir ./output --template_format alpaca",
        },
        {
            "description": "Use ChatML format",
            "command": "python -m src.functionalnetworkssft.fnsft_trainer --model_name_or_path gpt2 --dataset_name_or_path data.json --output_dir ./output --template_format chatml",
        },
        {
            "description": "Use basic format with custom template",
            "command": "python -m src.functionalnetworkssft.fnsft_trainer --model_name_or_path gpt2 --dataset_name_or_path data.json --output_dir ./output --template_format basic --instruction_template 'Q: {instruction}\\nA: {response}'",
        },
    ]

    for i, example in enumerate(examples, 1):
        print(f"\n{i}. {example['description']}:")
        print(f"   {example['command']}")


def main():
    """Run the demonstration."""
    demo_template_formats()
    demo_cli_usage()

    print("\n" + "=" * 50)
    print("✅ Demo completed!")
    print("\nKey Benefits:")
    print("• Automatic detection of tokenizer chat templates")
    print("• Flexible override options for specific use cases")
    print("• Support for popular formats (Alpaca, ChatML, etc.)")
    print("• Optimal fine-tuning performance with correct templates")
    print("• Backward compatibility with existing workflows")


if __name__ == "__main__":
    main()
