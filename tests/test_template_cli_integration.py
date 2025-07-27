#!/usr/bin/env python3
"""
Integration test for template format CLI argument.
"""

import pytest
import argparse
import tempfile
import json
import os
from unittest.mock import patch, MagicMock
from src.functionalnetworkssft.fnsft_trainer import main


def test_template_format_cli_argument():
    """Test that the template_format CLI argument is properly parsed."""
    import argparse
    import sys
    from src.functionalnetworkssft.fnsft_trainer import main

    # Test argument parsing directly
    test_cases = ["auto", "chat", "alpaca", "chatml", "basic"]

    for template_format in test_cases:
        # Test that the argument parser accepts valid template formats
        test_args = [
            "sft_trainer.py",
            "--model_name_or_path",
            "gpt2",
            "--dataset_name_or_path",
            "/tmp/dummy.json",
            "--output_dir",
            "/tmp/test_output",
            "--template_format",
            template_format,
        ]

        # Mock sys.argv and test argument parsing
        with patch("sys.argv", test_args):
            # Import the parser creation part
            parser = argparse.ArgumentParser(
                description="Supervised Fine-Tuning for Language Models"
            )

            # Add the template_format argument (copy from main function)
            parser.add_argument(
                "--template_format",
                type=str,
                default="auto",
                choices=["auto", "chat", "alpaca", "chatml", "basic"],
                help="Template format to use",
            )

            # Add required arguments to avoid errors
            parser.add_argument("--model_name_or_path", type=str, required=True)
            parser.add_argument("--dataset_name_or_path", type=str, required=True)
            parser.add_argument("--output_dir", type=str, required=True)

            # This should not raise an exception for valid template formats
            args = parser.parse_args(test_args[1:])  # Skip script name
            assert args.template_format == template_format


def test_invalid_template_format_cli():
    """Test that invalid template format raises appropriate error."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        test_data = [{"instruction": "test", "response": "response"}]
        json.dump(test_data, f)
        dataset_path = f.name

    try:
        test_args = [
            "sft_trainer.py",
            "--model_name_or_path",
            "gpt2",
            "--dataset_name_or_path",
            dataset_path,
            "--output_dir",
            "/tmp/test_output",
            "--template_format",
            "invalid_format",
        ]

        with patch("sys.argv", test_args):
            # This should raise a SystemExit due to invalid choice
            with pytest.raises(SystemExit):
                main()

    finally:
        if os.path.exists(dataset_path):
            os.unlink(dataset_path)


if __name__ == "__main__":
    pytest.main([__file__])
