#!/usr/bin/env python3
"""
Tests for PEFT vs Full Parameter Fine-Tuning functionality.

This test suite verifies that both PEFT and full parameter fine-tuning modes
work correctly, including model initialization, configuration, saving, and uploading.
"""

import os
import tempfile
import unittest
from unittest.mock import Mock, patch, MagicMock
import torch
import torch.nn as nn
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from functionalnetworkssft.fnsft_trainer import (
    LoRAArguments,
    ModelArguments,
    DataArguments,
    QuantizationArguments,
    setup_lora_from_args,
    adjust_training_args_for_mode,
    log_training_mode_details,
    upload_to_hub,
)
from functionalnetworkssft.utils.model_utils import setup_lora, save_model_and_tokenizer
from transformers import TrainingArguments


class MockModel(nn.Module):
    """Mock model for testing."""

    def __init__(self, has_peft=False):
        super().__init__()
        self.linear = nn.Linear(10, 10)
        self.config = Mock()
        self.config.quantization_config = None
        self.config.model_type = "test_model"  # Set a string value instead of Mock

        if has_peft:
            self.peft_config = {"default": Mock()}
            self.peft_config["default"].r = 16
            self.peft_config["default"].lora_alpha = 32
            self.peft_config["default"].lora_dropout = 0.1
            self.peft_config["default"].target_modules = ["linear"]
        # Don't set peft_config to None - let PEFT library handle missing attribute

    def parameters(self):
        return self.linear.parameters()

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        """Mock method required by PEFT."""
        return {"input_ids": input_ids}

    def save_pretrained(self, path):
        """Mock save_pretrained method."""
        os.makedirs(path, exist_ok=True)
        if hasattr(self, "peft_config") and self.peft_config:
            # Create adapter files for PEFT models
            with open(os.path.join(path, "adapter_config.json"), "w") as f:
                f.write('{"r": 16, "lora_alpha": 32}')
            with open(os.path.join(path, "adapter_model.safetensors"), "w") as f:
                f.write("mock adapter weights")
        else:
            # Create full model files
            with open(os.path.join(path, "config.json"), "w") as f:
                f.write('{"model_type": "test"}')
            with open(os.path.join(path, "pytorch_model.bin"), "w") as f:
                f.write("mock model weights")


class MockTokenizer:
    """Mock tokenizer for testing."""

    def save_pretrained(self, path):
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "tokenizer.json"), "w") as f:
            f.write('{"version": "1.0"}')

    def push_to_hub(self, repo_id, **kwargs):
        pass


class TestPEFTConfiguration(unittest.TestCase):
    """Test PEFT configuration and arguments."""

    def test_lora_arguments_default_peft_enabled(self):
        """Test that PEFT is enabled by default."""
        args = LoRAArguments()
        self.assertTrue(args.use_peft)

    def test_lora_arguments_peft_disabled(self):
        """Test PEFT can be disabled."""
        args = LoRAArguments(use_peft=False)
        self.assertFalse(args.use_peft)

    def test_lora_arguments_all_parameters(self):
        """Test all LoRA arguments can be set."""
        args = LoRAArguments(
            use_peft=True,
            lora_r=32,
            lora_alpha=64,
            lora_dropout=0.2,
            lora_target_modules=["q_proj", "v_proj"],
            lora_bias="all",
        )
        self.assertTrue(args.use_peft)
        self.assertEqual(args.lora_r, 32)
        self.assertEqual(args.lora_alpha, 64)
        self.assertEqual(args.lora_dropout, 0.2)
        self.assertEqual(args.lora_target_modules, ["q_proj", "v_proj"])
        self.assertEqual(args.lora_bias, "all")


class TestModelInitialization(unittest.TestCase):
    """Test model initialization for both PEFT and full fine-tuning."""

    def setUp(self):
        self.mock_model = MockModel()

    @patch("functionalnetworkssft.utils.model_utils.prepare_model_for_kbit_training")
    @patch("functionalnetworkssft.utils.model_utils.get_peft_model")
    def test_setup_lora_peft_enabled(self, mock_get_peft_model, mock_prepare_kbit):
        """Test LoRA setup when PEFT is enabled."""
        mock_peft_model = MockModel(has_peft=True)
        mock_get_peft_model.return_value = mock_peft_model

        result = setup_lora(self.mock_model, use_peft=True)

        mock_get_peft_model.assert_called_once()
        self.assertEqual(result, mock_peft_model)

    def test_setup_lora_peft_disabled(self):
        """Test model setup when PEFT is disabled."""
        result = setup_lora(self.mock_model, use_peft=False)

        # Should return the original model
        self.assertEqual(result, self.mock_model)

        # All parameters should require gradients
        for param in result.parameters():
            self.assertTrue(param.requires_grad)

    @patch("functionalnetworkssft.utils.model_utils.prepare_model_for_kbit_training")
    def test_setup_lora_quantized_model_full_training(self, mock_prepare_kbit):
        """Test full fine-tuning with quantized model."""
        # Mock quantized model
        self.mock_model.config.quantization_config = Mock()

        # Mock the prepare_model_for_kbit_training to return the original model
        mock_prepare_kbit.return_value = self.mock_model

        result = setup_lora(self.mock_model, use_peft=False)

        mock_prepare_kbit.assert_called_once_with(self.mock_model)
        self.assertEqual(result, self.mock_model)


class TestTrainingConfiguration(unittest.TestCase):
    """Test training configuration adjustments."""

    def test_adjust_training_args_peft_mode(self):
        """Test training arguments for PEFT mode."""
        training_args = TrainingArguments(
            output_dir="./test",
            learning_rate=2e-4,
            warmup_ratio=0.03,
            gradient_checkpointing=False,
        )

        result = adjust_training_args_for_mode(training_args, use_peft=True)

        # Should not modify PEFT settings
        self.assertEqual(result.learning_rate, 2e-4)
        self.assertEqual(result.warmup_ratio, 0.03)

    def test_adjust_training_args_full_mode(self):
        """Test training arguments for full fine-tuning mode."""
        training_args = TrainingArguments(
            output_dir="./test",
            learning_rate=2e-4,  # Default PEFT learning rate
            warmup_ratio=0.03,  # Default warmup
            gradient_checkpointing=False,
        )

        result = adjust_training_args_for_mode(training_args, use_peft=False)

        # Should adjust for full fine-tuning
        self.assertEqual(result.learning_rate, 5e-5)  # Lower LR
        self.assertEqual(result.warmup_ratio, 0.1)  # More warmup
        self.assertTrue(result.gradient_checkpointing)  # Enable checkpointing


class TestModelSaving(unittest.TestCase):
    """Test model saving for both modes."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.mock_tokenizer = MockTokenizer()

    def tearDown(self):
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_save_peft_model(self):
        """Test saving PEFT model."""
        mock_model = MockModel(has_peft=True)

        save_model_and_tokenizer(
            mock_model, self.mock_tokenizer, self.temp_dir, use_peft=True
        )

        # Check PEFT files were created
        self.assertTrue(
            os.path.exists(os.path.join(self.temp_dir, "adapter_config.json"))
        )
        self.assertTrue(
            os.path.exists(os.path.join(self.temp_dir, "adapter_model.safetensors"))
        )
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, "tokenizer.json")))

    def test_save_full_model(self):
        """Test saving full fine-tuned model."""
        mock_model = MockModel(has_peft=False)

        save_model_and_tokenizer(
            mock_model, self.mock_tokenizer, self.temp_dir, use_peft=False
        )

        # Check full model files were created
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, "config.json")))
        self.assertTrue(
            os.path.exists(os.path.join(self.temp_dir, "pytorch_model.bin"))
        )
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, "tokenizer.json")))

    def test_save_model_auto_detect_peft(self):
        """Test auto-detection of PEFT model."""
        mock_model = MockModel(has_peft=True)

        # Don't specify use_peft, should auto-detect
        save_model_and_tokenizer(mock_model, self.mock_tokenizer, self.temp_dir)

        # Should save as PEFT model
        self.assertTrue(
            os.path.exists(os.path.join(self.temp_dir, "adapter_config.json"))
        )


class TestHubUpload(unittest.TestCase):
    """Test Hub upload functionality."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.mock_tokenizer = MockTokenizer()

    def tearDown(self):
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch("functionalnetworkssft.utils.hf_utilities.HfApi")
    @patch("functionalnetworkssft.utils.hf_utilities.whoami")
    def test_upload_peft_model(self, mock_whoami, mock_hf_api):
        """Test uploading PEFT model."""
        # Create PEFT model files
        os.makedirs(self.temp_dir, exist_ok=True)
        with open(os.path.join(self.temp_dir, "adapter_config.json"), "w") as f:
            f.write('{"r": 16}')
        with open(os.path.join(self.temp_dir, "adapter_model.safetensors"), "w") as f:
            f.write("adapter weights")

        mock_api_instance = Mock()
        mock_hf_api.return_value = mock_api_instance
        mock_whoami.return_value = {"name": "test_user"}

        # Should not raise an exception
        upload_to_hub(
            model_path=self.temp_dir,
            tokenizer=self.mock_tokenizer,
            repo_id="test/repo",
            use_peft=True,
        )

        # Verify API calls were made
        mock_api_instance.repo_info.assert_called()

    @patch("functionalnetworkssft.utils.hf_utilities.HfApi")
    @patch("functionalnetworkssft.utils.hf_utilities.whoami")
    def test_upload_full_model(self, mock_whoami, mock_hf_api):
        """Test uploading full model."""
        # Create full model files
        os.makedirs(self.temp_dir, exist_ok=True)
        with open(os.path.join(self.temp_dir, "config.json"), "w") as f:
            f.write('{"model_type": "test"}')
        with open(os.path.join(self.temp_dir, "pytorch_model.bin"), "w") as f:
            f.write("model weights")

        mock_api_instance = Mock()
        mock_hf_api.return_value = mock_api_instance
        mock_whoami.return_value = {"name": "test_user"}

        # Should not raise an exception
        upload_to_hub(
            model_path=self.temp_dir,
            tokenizer=self.mock_tokenizer,
            repo_id="test/repo",
            use_peft=False,
        )

        # Verify API calls were made
        mock_api_instance.repo_info.assert_called()


class TestLogging(unittest.TestCase):
    """Test logging functionality."""

    @patch("functionalnetworkssft.fnsft_trainer.logger")
    def test_log_training_mode_peft(self, mock_logger):
        """Test logging for PEFT mode."""
        mock_model = MockModel(has_peft=True)

        log_training_mode_details(use_peft=True, model=mock_model)

        # Check that PEFT-specific messages were logged
        logged_messages = [call[0][0] for call in mock_logger.info.call_args_list]
        self.assertTrue(any("PEFT" in msg for msg in logged_messages))

    @patch("functionalnetworkssft.fnsft_trainer.logger")
    def test_log_training_mode_full(self, mock_logger):
        """Test logging for full fine-tuning mode."""
        mock_model = MockModel(has_peft=False)

        log_training_mode_details(use_peft=False, model=mock_model)

        # Check that full fine-tuning messages were logged
        logged_messages = [call[0][0] for call in mock_logger.info.call_args_list]
        self.assertTrue(any("Full Parameter" in msg for msg in logged_messages))


class TestIntegration(unittest.TestCase):
    """Integration tests for end-to-end functionality."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch("functionalnetworkssft.fnsft_trainer.load_model_and_tokenizer")
    @patch("functionalnetworkssft.fnsft_trainer.load_dataset_from_args")
    @patch("functionalnetworkssft.fnsft_trainer.Trainer")
    def test_peft_training_pipeline(
        self, mock_trainer, mock_load_dataset, mock_load_model
    ):
        """Test complete PEFT training pipeline."""
        # Mock model and tokenizer
        mock_model = MockModel(has_peft=False)
        mock_tokenizer = MockTokenizer()
        mock_load_model.return_value = (mock_model, mock_tokenizer)

        # Mock dataset
        mock_load_dataset.return_value = [
            {"instruction": "Test instruction", "response": "Test response"}
        ]

        # Mock trainer
        mock_trainer_instance = Mock()
        mock_trainer.return_value = mock_trainer_instance

        # Test arguments for PEFT
        lora_args = LoRAArguments(use_peft=True)

        # This would normally be called by the main function
        # We're testing the key components work together
        model_with_lora = setup_lora_from_args(mock_model, lora_args)

        # Verify PEFT was applied (in real scenario, this would be a PEFT model)
        self.assertIsNotNone(model_with_lora)

    @patch("functionalnetworkssft.fnsft_trainer.load_model_and_tokenizer")
    @patch("functionalnetworkssft.fnsft_trainer.load_dataset_from_args")
    @patch("functionalnetworkssft.fnsft_trainer.Trainer")
    def test_full_training_pipeline(
        self, mock_trainer, mock_load_dataset, mock_load_model
    ):
        """Test complete full fine-tuning pipeline."""
        # Mock model and tokenizer
        mock_model = MockModel(has_peft=False)
        mock_tokenizer = MockTokenizer()
        mock_load_model.return_value = (mock_model, mock_tokenizer)

        # Mock dataset
        mock_load_dataset.return_value = [
            {"instruction": "Test instruction", "response": "Test response"}
        ]

        # Mock trainer
        mock_trainer_instance = Mock()
        mock_trainer.return_value = mock_trainer_instance

        # Test arguments for full fine-tuning
        lora_args = LoRAArguments(use_peft=False)

        # This would normally be called by the main function
        model_prepared = setup_lora_from_args(mock_model, lora_args)

        # Verify all parameters require gradients for full fine-tuning
        for param in model_prepared.parameters():
            self.assertTrue(param.requires_grad)

    def test_configuration_consistency(self):
        """Test that configuration is consistent between modes."""
        # PEFT configuration (default)
        peft_args = LoRAArguments(lora_r=16, lora_alpha=32)
        self.assertTrue(peft_args.use_peft)  # Default is True
        self.assertEqual(peft_args.lora_r, 16)

        # Full fine-tuning configuration
        full_args = LoRAArguments(use_peft=False, lora_r=16, lora_alpha=32)
        self.assertFalse(full_args.use_peft)
        # LoRA parameters should still be accessible even if not used
        self.assertEqual(full_args.lora_r, 16)


class TestErrorHandling(unittest.TestCase):
    """Test error handling and edge cases."""

    def test_invalid_model_path_upload(self):
        """Test upload with invalid model path."""
        with self.assertRaises(ValueError):
            upload_to_hub(
                model_path="/nonexistent/path",
                tokenizer=MockTokenizer(),
                repo_id="test/repo",
            )

    @patch("functionalnetworkssft.utils.hf_utilities.HfApi")
    @patch("functionalnetworkssft.utils.hf_utilities.whoami")
    def test_missing_adapter_files_peft_upload(self, mock_whoami, mock_hf_api):
        """Test PEFT upload when adapter files are missing."""
        temp_dir = tempfile.mkdtemp()
        try:
            # Create directory but no adapter files
            os.makedirs(temp_dir, exist_ok=True)

            # Mock the API to avoid actual network calls
            mock_api_instance = Mock()
            mock_hf_api.return_value = mock_api_instance
            mock_whoami.return_value = {"name": "test_user"}

            with self.assertRaises(ValueError):
                upload_to_hub(
                    model_path=temp_dir,
                    tokenizer=MockTokenizer(),
                    repo_id="test/repo",
                    use_peft=True,
                    push_adapter_only=True,
                )
        finally:
            import shutil

            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    # Run tests with verbose output
    unittest.main(verbosity=2)
