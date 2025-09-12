"""
Test suite for ConfigDefaults class functionality.

This module tests the intelligent default configuration values,
particularly the model-specific output directory generation.
"""

import os
import tempfile
import unittest
from unittest.mock import patch, MagicMock
import shutil
from pathlib import Path

# Add project root to path for imports
import sys

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.functionalnetworkssft.utils.config_defaults import ConfigDefaults


class TestConfigDefaults(unittest.TestCase):
    """Test cases for ConfigDefaults class."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_config = {
            "model_name_or_path": "microsoft/DialoGPT-medium",
            "stages": ["sft"],
            "stage_configs": {
                "sft": {"num_train_epochs": 3, "per_device_train_batch_size": 4}
            },
        }

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_extract_model_base_name_huggingface(self):
        """Test extracting base name from HuggingFace model names."""
        test_cases = [
            ("microsoft/DialoGPT-medium", "dialogpt-medium"),
            ("meta-llama/Llama-2-7b-hf", "llama-2-7b"),
            ("google/flan-t5-base", "flan-t5"),
            ("openai/gpt-3.5-turbo-instruct", "gpt-3.5-turbo"),
        ]

        for model_name, expected in test_cases:
            with self.subTest(model_name=model_name):
                result = ConfigDefaults.extract_model_base_name(model_name)
                self.assertEqual(result, expected)

    def test_extract_model_base_name_local_path(self):
        """Test extracting base name from local paths."""
        test_cases = [
            ("/path/to/my-model", "my-model"),
            ("./models/custom-model-chat", "custom-model"),
            ("C:\\models\\my-model-base", "my-model"),
        ]

        for model_path, expected in test_cases:
            with self.subTest(model_path=model_path):
                result = ConfigDefaults.extract_model_base_name(model_path)
                self.assertEqual(result, expected)

    @patch.dict(os.environ, {}, clear=True)
    def test_get_default_output_dir_no_env(self):
        """Test getting default output directory without environment variables."""
        result = ConfigDefaults.get_default_output_dir()
        self.assertEqual(result, "./models/output")

    @patch.dict(os.environ, {"LMPIPELINE_OUTPUT_DIR": "/custom/output"})
    def test_get_default_output_dir_with_env(self):
        """Test getting default output directory with environment variable."""
        result = ConfigDefaults.get_default_output_dir()
        self.assertEqual(result, "/custom/output")

    @patch(
        "src.functionalnetworkssft.utils.config_defaults.ConfigDefaults.ensure_directory_exists"
    )
    @patch(
        "src.functionalnetworkssft.utils.config_defaults.ConfigDefaults.get_default_output_dir"
    )
    @patch(
        "src.functionalnetworkssft.utils.config_defaults.ConfigDefaults.get_default_checkpoints_dir"
    )
    def test_apply_storage_defaults_model_specific(
        self, mock_get_checkpoints, mock_get_default, mock_ensure_dir
    ):
        """Test that apply_storage_defaults creates model-specific directories."""
        mock_get_default.return_value = "./models/output"
        mock_get_checkpoints.return_value = "./models/checkpoints"
        mock_ensure_dir.return_value = True

        config = {"model_name_or_path": "microsoft/DialoGPT-medium"}
        result = ConfigDefaults.apply_storage_defaults(config)

        expected_output_dir = os.path.join("./models/output", "dialogpt-medium")
        self.assertEqual(result["output_dir"], expected_output_dir)

        # Check that ensure_directory_exists was called with the model-specific path
        calls = mock_ensure_dir.call_args_list
        model_specific_call = any(call[0][0] == expected_output_dir for call in calls)
        self.assertTrue(
            model_specific_call,
            f"Expected call with {expected_output_dir} not found in {calls}",
        )

    @patch(
        "src.functionalnetworkssft.utils.config_defaults.ConfigDefaults.ensure_directory_exists"
    )
    @patch(
        "src.functionalnetworkssft.utils.config_defaults.ConfigDefaults.get_default_output_dir"
    )
    @patch(
        "src.functionalnetworkssft.utils.config_defaults.ConfigDefaults.get_default_checkpoints_dir"
    )
    def test_apply_storage_defaults_existing_output_dir(
        self, mock_get_checkpoints, mock_get_default, mock_ensure_dir
    ):
        """Test that existing output_dir is preserved."""
        mock_get_default.return_value = "./models/output"
        mock_get_checkpoints.return_value = "./models/checkpoints"
        mock_ensure_dir.return_value = True

        config = {
            "model_name_or_path": "microsoft/DialoGPT-medium",
            "output_dir": "/custom/existing/path",
        }
        result = ConfigDefaults.apply_storage_defaults(config)

        # Should preserve existing output_dir
        self.assertEqual(result["output_dir"], "/custom/existing/path")

        # Should not call ensure_directory_exists for model-specific path
        calls = mock_ensure_dir.call_args_list
        model_specific_path = os.path.join("./models/output", "dialogpt-medium")
        model_specific_call = any(call[0][0] == model_specific_path for call in calls)
        self.assertFalse(
            model_specific_call,
            f"Unexpected call with {model_specific_path} found in {calls}",
        )

    @patch(
        "src.functionalnetworkssft.utils.config_defaults.ConfigDefaults.ensure_directory_exists"
    )
    @patch(
        "src.functionalnetworkssft.utils.config_defaults.ConfigDefaults.get_default_output_dir"
    )
    @patch(
        "src.functionalnetworkssft.utils.config_defaults.ConfigDefaults.get_default_checkpoints_dir"
    )
    def test_apply_storage_defaults_fallback_model_name(
        self, mock_get_checkpoints, mock_get_default, mock_ensure_dir
    ):
        """Test fallback when no model_name_or_path is provided."""
        mock_get_default.return_value = "./models/output"
        mock_get_checkpoints.return_value = "./models/checkpoints"
        mock_ensure_dir.return_value = True

        config = {}  # No model_name_or_path
        result = ConfigDefaults.apply_storage_defaults(config)

        expected_output_dir = os.path.join("./models/output", "model")
        self.assertEqual(result["output_dir"], expected_output_dir)

        # Check that ensure_directory_exists was called with the fallback model path
        calls = mock_ensure_dir.call_args_list
        model_specific_call = any(call[0][0] == expected_output_dir for call in calls)
        self.assertTrue(
            model_specific_call,
            f"Expected call with {expected_output_dir} not found in {calls}",
        )

    def test_ensure_directory_exists_success(self):
        """Test successful directory creation."""
        test_dir = os.path.join(self.temp_dir, "test_directory")
        result = ConfigDefaults.ensure_directory_exists(test_dir)

        self.assertTrue(result)
        self.assertTrue(os.path.exists(test_dir))
        self.assertTrue(os.path.isdir(test_dir))

    def test_ensure_directory_exists_already_exists(self):
        """Test behavior when directory already exists."""
        test_dir = os.path.join(self.temp_dir, "existing_directory")
        os.makedirs(test_dir)

        result = ConfigDefaults.ensure_directory_exists(test_dir)

        self.assertTrue(result)
        self.assertTrue(os.path.exists(test_dir))

    def test_ensure_directory_exists_file_conflict(self):
        """Test behavior when path exists but is a file."""
        test_file = os.path.join(self.temp_dir, "test_file")
        with open(test_file, "w") as f:
            f.write("test")

        result = ConfigDefaults.ensure_directory_exists(test_file)

        self.assertFalse(result)

    def test_generate_model_name_basic(self):
        """Test basic model name generation."""
        result = ConfigDefaults.generate_model_name(
            base_model_name="microsoft/DialoGPT-medium", stages=["sft"]
        )

        self.assertEqual(result, "dialogpt-medium-finetuned")

    def test_generate_model_name_with_quantization(self):
        """Test model name generation with quantization."""
        result = ConfigDefaults.generate_model_name(
            base_model_name="microsoft/DialoGPT-medium",
            stages=["sft"],
            quantization_config={"use_4bit": True},
        )

        self.assertEqual(result, "dialogpt-medium-finetuned-4bit")

    def test_generate_model_name_with_dtype(self):
        """Test model name generation with specific dtype."""
        result = ConfigDefaults.generate_model_name(
            base_model_name="microsoft/DialoGPT-medium",
            stages=["sft"],
            torch_dtype="fp16",
        )

        self.assertEqual(result, "dialogpt-medium-finetuned-fp16")

    def test_generate_model_name_with_gguf(self):
        """Test model name generation with GGUF conversion."""
        result = ConfigDefaults.generate_model_name(
            base_model_name="microsoft/DialoGPT-medium",
            stages=["sft"],
            convert_to_gguf=True,
            gguf_quantization="q4_0",
        )

        self.assertEqual(result, "dialogpt-medium-finetuned-gguf-q4_0")


class TestConfigDefaultsIntegration(unittest.TestCase):
    """Integration tests for ConfigDefaults with real directory operations."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch(
        "src.functionalnetworkssft.utils.config_defaults.ConfigDefaults.get_default_output_dir"
    )
    def test_real_directory_creation(self, mock_get_default):
        """Test actual directory creation with model-specific paths."""
        mock_get_default.return_value = self.temp_dir

        config = {"model_name_or_path": "microsoft/DialoGPT-medium"}
        result = ConfigDefaults.apply_storage_defaults(config)

        expected_path = os.path.join(self.temp_dir, "dialogpt-medium")
        self.assertEqual(result["output_dir"], expected_path)
        self.assertTrue(os.path.exists(expected_path))
        self.assertTrue(os.path.isdir(expected_path))

    @patch(
        "src.functionalnetworkssft.utils.config_defaults.ConfigDefaults.get_default_output_dir"
    )
    @patch(
        "src.functionalnetworkssft.utils.config_defaults.ConfigDefaults.get_default_checkpoints_dir"
    )
    def test_apply_all_defaults_model_specific(
        self, mock_get_checkpoints, mock_get_default
    ):
        """Test that apply_all_defaults also creates model-specific directories."""
        mock_get_default.return_value = self.temp_dir
        mock_get_checkpoints.return_value = os.path.join(self.temp_dir, "checkpoints")

        config = {"model_name_or_path": "meta-llama/Llama-2-7b-hf"}
        result = ConfigDefaults.apply_all_defaults(config)

        expected_path = os.path.join(self.temp_dir, "llama-2-7b")
        self.assertEqual(result["output_dir"], expected_path)
        self.assertTrue(os.path.exists(expected_path))
        self.assertTrue(os.path.isdir(expected_path))


if __name__ == "__main__":
    unittest.main()
