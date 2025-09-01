#!/usr/bin/env python3
"""
Test split-aware dataset loading functionality.
"""

import pytest
from unittest.mock import MagicMock, patch
from src.functionalnetworkssft.utils.model_utils import (
    load_dataset_with_splits,
    prepare_train_val_splits,
)


class TestSplitAwareLoading:
    """Test cases for split-aware dataset loading."""

    @patch('src.functionalnetworkssft.utils.model_utils.load_dataset')
    def test_load_dataset_with_splits_multiple_splits(self, mock_load_dataset):
        """Test loading dataset with multiple splits available."""
        # Mock dataset with train, validation, and test splits
        mock_dataset = {
            'train': [
                {'message_1': 'What is physics?', 'message_2': 'Physics is the study of matter and energy.', 'topic': 'Physics', 'sub_topic': 'General'},
                {'message_1': 'Define force.', 'message_2': 'Force is any interaction that changes motion.', 'topic': 'Physics', 'sub_topic': 'Mechanics'},
            ],
            'validation': [
                {'message_1': 'What is energy?', 'message_2': 'Energy is the capacity to do work.', 'topic': 'Physics', 'sub_topic': 'Energy'},
            ],
            'test': [
                {'message_1': 'What is momentum?', 'message_2': 'Momentum is mass times velocity.', 'topic': 'Physics', 'sub_topic': 'Mechanics'},
            ]
        }
        mock_load_dataset.return_value = mock_dataset
        
        # Test loading
        result = load_dataset_with_splits("camel-ai/physics")
        
        # Verify all splits are loaded
        assert "train" in result
        assert "validation" in result
        assert "test" in result
        assert len(result["train"]) == 2
        assert len(result["validation"]) == 1
        assert len(result["test"]) == 1

    @patch('src.functionalnetworkssft.utils.model_utils.load_dataset')
    def test_load_dataset_with_splits_train_only(self, mock_load_dataset):
        """Test loading dataset with only train split."""
        # Mock dataset with only train split
        mock_dataset = {
            'train': [
                {'message_1': 'What is physics?', 'message_2': 'Physics is the study of matter and energy.', 'topic': 'Physics', 'sub_topic': 'General'},
                {'message_1': 'Define force.', 'message_2': 'Force is any interaction that changes motion.', 'topic': 'Physics', 'sub_topic': 'Mechanics'},
            ]
        }
        mock_load_dataset.return_value = mock_dataset
        
        # Test loading
        result = load_dataset_with_splits("some-dataset")
        
        # Verify only train split is available
        assert "train" in result
        assert len(result) == 1
        assert len(result["train"]) == 2

    @patch('src.functionalnetworkssft.utils.model_utils.load_dataset')
    def test_load_dataset_with_splits_no_train_split(self, mock_load_dataset):
        """Test loading dataset with no train split (uses first available)."""
        # Mock dataset with only validation split
        mock_dataset = {
            'validation': [
                {'message_1': 'What is energy?', 'message_2': 'Energy is the capacity to do work.', 'topic': 'Physics', 'sub_topic': 'Energy'},
            ]
        }
        mock_load_dataset.return_value = mock_dataset
        
        # Test loading
        result = load_dataset_with_splits("some-dataset")
        
        # Verify validation split is used as train
        assert "train" in result
        assert "validation" in result
        assert result["train"] == result["validation"]  # Same data

    def test_prepare_train_val_splits_with_existing_validation(self):
        """Test preparing splits when validation split exists."""
        splits_data = {
            'train': [
                {'instruction': 'What is physics?', 'response': 'Physics is the study of matter and energy.'},
                {'instruction': 'Define force.', 'response': 'Force is any interaction that changes motion.'},
            ],
            'validation': [
                {'instruction': 'What is energy?', 'response': 'Energy is the capacity to do work.'},
            ]
        }
        
        train_data, val_data = prepare_train_val_splits(
            splits_data, 
            validation_split=0.5,  # This should be ignored since validation exists
            prefer_existing_splits=True
        )
        
        # Should use existing validation split
        assert len(train_data) == 2  # Original train data unchanged
        assert len(val_data) == 1   # Uses existing validation split
        assert val_data[0]['instruction'] == 'What is energy?'

    def test_prepare_train_val_splits_without_existing_validation(self):
        """Test preparing splits when no validation split exists."""
        splits_data = {
            'train': [
                {'instruction': 'What is physics?', 'response': 'Physics is the study of matter and energy.'},
                {'instruction': 'Define force.', 'response': 'Force is any interaction that changes motion.'},
                {'instruction': 'What is energy?', 'response': 'Energy is the capacity to do work.'},
                {'instruction': 'What is momentum?', 'response': 'Momentum is mass times velocity.'},
            ]
        }
        
        train_data, val_data = prepare_train_val_splits(
            splits_data, 
            validation_split=0.25,  # 25% for validation
            prefer_existing_splits=True
        )
        
        # Should create custom split
        assert len(train_data) == 3  # 75% of 4 = 3
        assert len(val_data) == 1    # 25% of 4 = 1

    def test_prepare_train_val_splits_prefer_existing_false(self):
        """Test preparing splits when prefer_existing_splits is False."""
        splits_data = {
            'train': [
                {'instruction': 'What is physics?', 'response': 'Physics is the study of matter and energy.'},
                {'instruction': 'Define force.', 'response': 'Force is any interaction that changes motion.'},
                {'instruction': 'What is energy?', 'response': 'Energy is the capacity to do work.'},
                {'instruction': 'What is momentum?', 'response': 'Momentum is mass times velocity.'},
            ],
            'validation': [
                {'instruction': 'Existing validation', 'response': 'This should be ignored.'},
            ]
        }
        
        train_data, val_data = prepare_train_val_splits(
            splits_data, 
            validation_split=0.25,
            prefer_existing_splits=False
        )
        
        # Should ignore existing validation and create custom split
        assert len(train_data) == 3  # 75% of 4 = 3
        assert len(val_data) == 1    # 25% of 4 = 1
        # Validation data should come from train split, not existing validation
        assert val_data[0]['instruction'] != 'Existing validation'

    def test_prepare_train_val_splits_validation_split_zero(self):
        """Test preparing splits with validation_split=0."""
        splits_data = {
            'train': [
                {'instruction': 'What is physics?', 'response': 'Physics is the study of matter and energy.'},
                {'instruction': 'Define force.', 'response': 'Force is any interaction that changes motion.'},
            ]
        }
        
        train_data, val_data = prepare_train_val_splits(
            splits_data, 
            validation_split=0.0,
            prefer_existing_splits=True
        )
        
        # Should use all data for training, no validation
        assert len(train_data) == 2
        assert len(val_data) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
