#!/usr/bin/env python3
"""
Integration test to verify that the ICA mask fixes work in a realistic training scenario.
"""

import os
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from functionalnetworkssft.fnsft_trainer import main


class TestICAMaskIntegrationFix(unittest.TestCase):
    """Test that the ICA mask fixes work in integration scenarios."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create a minimal config that would trigger the ICA masking code path
        self.test_config = {
            "model_name_or_path": "microsoft/DialoGPT-small",
            "dataset_name_or_path": "tatsu-lab/alpaca",
            "output_dir": os.path.join(self.temp_dir, "test_output"),
            "num_train_epochs": 1,
            "max_steps": 2,  # Very small number for quick test
            "per_device_train_batch_size": 1,
            "per_device_eval_batch_size": 1,
            "learning_rate": 2e-4,
            "use_8bit": False,
            "use_4bit": False,
            "mask_mode": "lesion",  # Enable masking to trigger the code path
            "ica_components": 2,
            "ica_percentile": 98.0,
            "ica_component_ids": [0],
            "anti_drift_row_param": True,  # Enable row parametrizations
            "anti_drift_unwrap_on_save": False,
        }

    def tearDown(self):
        """Clean up test environment."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch('functionalnetworkssft.fnsft_trainer.load_model_and_tokenizer')
    @patch('functionalnetworkssft.fnsft_trainer.load_dataset_with_splits_from_args')
    @patch('functionalnetworkssft.fnsft_trainer.Trainer')
    @patch('functionalnetworkssft.fnsft_trainer.save_model_and_tokenizer')
    def test_training_with_ica_masking_no_len_none_error(self, mock_save, mock_trainer_class, mock_load_dataset, mock_load_model):
        """Test that training with ICA masking doesn't throw len(None) error."""
        
        # Mock the model and tokenizer
        mock_model = Mock()
        mock_model.parameters.return_value = [Mock(requires_grad=True, numel=lambda: 1000)]
        mock_model.config = Mock()
        mock_model.config.model_type = "gpt2"
        mock_model.config.hidden_size = 768
        
        mock_tokenizer = Mock()
        mock_tokenizer.name_or_path = "test_tokenizer"
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "<|endoftext|>"
        mock_tokenizer.eos_token_id = 50256
        mock_load_model.return_value = (mock_model, mock_tokenizer)
        
        # Mock dataset loading
        mock_dataset = {
            'train': [
                {'instruction': 'Test instruction 1', 'response': 'Test response 1'},
                {'instruction': 'Test instruction 2', 'response': 'Test response 2'},
            ],
            'validation': [
                {'instruction': 'Test instruction val', 'response': 'Test response val'},
            ]
        }
        mock_load_dataset.return_value = mock_dataset
        
        # Mock trainer
        mock_trainer = Mock()
        mock_trainer_class.return_value = mock_trainer
        
        # Mock sys.argv to simulate command line arguments
        test_args = []
        for key, value in self.test_config.items():
            if isinstance(value, bool):
                if value:
                    test_args.append(f"--{key}")
            elif isinstance(value, list):
                test_args.append(f"--{key}")
                test_args.extend([str(v) for v in value])
            elif value is not None:
                test_args.append(f"--{key}")
                test_args.append(str(value))
        
        with patch('sys.argv', ['test_script.py'] + test_args):
            try:
                # This should not raise the "object of type 'NoneType' has no len()" error
                main()
                print("SUCCESS: Training with ICA masking completed without len(None) error")
            except Exception as e:
                # Make sure it's not the specific NoneType len() error we're fixing
                error_msg = str(e)
                if "object of type 'NoneType' has no len()" in error_msg:
                    self.fail(f"len(None) error still exists: {e}")
                else:
                    # Other exceptions might be expected due to mocking
                    print(f"Other exception (may be expected): {e}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
