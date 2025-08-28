#!/usr/bin/env python3
"""
Test suite for DatasetFormatter class.
"""

import pytest
from src.functionalnetworkssft.utils.dataset_utils import DatasetFormatter
from src.functionalnetworkssft.fnsft_trainer import InstructionDataset


class MockTokenizer:
    """Mock tokenizer for testing."""

    def __init__(self):
        self.pad_token = None
        self.eos_token = "</s>"
        self.eos_token_id = 2
        self.pad_token_id = None

    def __call__(self, text, **kwargs):
        # Mock tokenization - just return some dummy tokens
        return {"input_ids": [0, 1, 2, 3, 4], "attention_mask": [1, 1, 1, 1, 1]}

    def apply_chat_template(self, messages, **kwargs):
        # Mock chat template application
        return "mocked chat template result"


class TestDatasetFormatter:
    """Test cases for DatasetFormatter class."""

    def test_detect_system_user_assistant_format(self):
        """Test detection of system-user-assistant format."""
        data = [
            {
                "system": "You are a helpful assistant.",
                "user": "What is 2+2?",
                "assistant": "2+2 equals 4.",
            },
            {
                "system": "You are a financial advisor.",
                "user": "Should I invest in stocks?",
                "assistant": "It depends on your risk tolerance and financial goals.",
            },
        ]

        detected_format = DatasetFormatter.detect_format(data)
        assert detected_format == ("assistant", "system", "user")

    def test_convert_system_user_assistant_format_with_system(self):
        """Test conversion of system-user-assistant format with system prompt."""
        item = {
            "system": "You are a helpful assistant.",
            "user": "What is the capital of France?",
            "assistant": "The capital of France is Paris.",
        }

        result = DatasetFormatter.convert_to_standard_format(
            item, ("assistant", "system", "user")
        )

        expected_instruction = (
            "You are a helpful assistant.\n\nWhat is the capital of France?"
        )
        assert result["instruction"] == expected_instruction
        assert result["response"] == "The capital of France is Paris."

    def test_convert_system_user_assistant_format_empty_system(self):
        """Test conversion with empty system field."""
        item = {"system": "", "user": "What is 2+2?", "assistant": "2+2 equals 4."}

        result = DatasetFormatter.convert_to_standard_format(
            item, ("assistant", "system", "user")
        )

        expected_instruction = "You are a helpful assistant which thinks step by step when responding to a user.\n\nWhat is 2+2?"
        assert result["instruction"] == expected_instruction
        assert result["response"] == "2+2 equals 4."

    def test_convert_system_user_assistant_format_null_system(self):
        """Test conversion with null system field."""
        item = {
            "system": None,
            "user": "Explain photosynthesis.",
            "assistant": "Photosynthesis is the process by which plants convert light energy into chemical energy.",
        }

        result = DatasetFormatter.convert_to_standard_format(
            item, ("assistant", "system", "user")
        )

        expected_instruction = "You are a helpful assistant which thinks step by step when responding to a user.\n\nExplain photosynthesis."
        assert result["instruction"] == expected_instruction
        assert (
            result["response"]
            == "Photosynthesis is the process by which plants convert light energy into chemical energy."
        )

    def test_convert_system_user_assistant_format_missing_system(self):
        """Test conversion with missing system field."""
        item = {
            "user": "What is machine learning?",
            "assistant": "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience.",
        }

        result = DatasetFormatter.convert_to_standard_format(
            item, ("assistant", "system", "user")
        )

        expected_instruction = "You are a helpful assistant which thinks step by step when responding to a user.\n\nWhat is machine learning?"
        assert result["instruction"] == expected_instruction
        assert (
            result["response"]
            == "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience."
        )

    def test_convert_system_user_assistant_format_whitespace_system(self):
        """Test conversion with whitespace-only system field."""
        item = {
            "system": "   ",
            "user": "How does gravity work?",
            "assistant": "Gravity is a fundamental force that attracts objects with mass toward each other.",
        }

        result = DatasetFormatter.convert_to_standard_format(
            item, ("assistant", "system", "user")
        )

        expected_instruction = "You are a helpful assistant which thinks step by step when responding to a user.\n\nHow does gravity work?"
        assert result["instruction"] == expected_instruction
        assert (
            result["response"]
            == "Gravity is a fundamental force that attracts objects with mass toward each other."
        )

    def test_format_priority_includes_system_user_assistant(self):
        """Test that system-user-assistant format is included in format priority."""
        data = [
            {
                "system": "You are a financial advisor.",
                "user": "What is compound interest?",
                "assistant": "Compound interest is interest calculated on the initial principal and accumulated interest.",
                "extra_field": "should be ignored",
            }
        ]

        detected_format = DatasetFormatter.detect_format(data)
        # Should detect the system-user-assistant format even with extra fields
        assert detected_format == ("assistant", "system", "user")

    def test_existing_formats_still_work(self):
        """Test that existing format detection still works."""
        # Test instruction-response format
        data1 = [{"instruction": "Hello", "response": "Hi there!"}]
        assert DatasetFormatter.detect_format(data1) == ("instruction", "response")

        # Test question-answer format
        data2 = [{"question": "What is AI?", "answer": "Artificial Intelligence"}]
        assert DatasetFormatter.detect_format(data2) == ("question", "answer")

        # Test prompt-completion format
        data3 = [{"prompt": "Complete this", "completion": "sentence"}]
        assert DatasetFormatter.detect_format(data3) == ("prompt", "completion")

        # Test context-response format (lowercase)
        data4 = [{"context": "Some context", "response": "Some response"}]
        assert DatasetFormatter.detect_format(data4) == ("context", "response")

        # Test Context-Response format (capitalized)
        data5 = [{"Context": "Some context", "Response": "Some response"}]
        assert DatasetFormatter.detect_format(data5) == ("Context", "Response")

    def test_integration_with_instruction_dataset(self):
        """Test integration with InstructionDataset using system-user-assistant format."""
        tokenizer = MockTokenizer()

        # Test data in system-user-assistant format
        data = [
            {
                "system": "You are a financial advisor.",
                "user": "What is compound interest?",
                "assistant": "Compound interest is interest calculated on the initial principal and accumulated interest.",
            },
            {
                "system": "",  # Empty system field
                "user": "What is 2+2?",
                "assistant": "2+2 equals 4.",
            },
        ]

        # Create dataset with auto-detection enabled
        dataset = InstructionDataset(
            data=data,
            tokenizer=tokenizer,
            template_format="basic",
            auto_detect_format=True,
            max_length=512,
        )

        # Verify format was detected correctly
        assert dataset.detected_format == ("assistant", "system", "user")

        # Test that the dataset can be accessed without errors
        assert len(dataset) == 2

        # Test conversion of first item (with system prompt)
        converted_item = DatasetFormatter.convert_to_standard_format(
            data[0], dataset.detected_format
        )
        expected_instruction = (
            "You are a financial advisor.\n\nWhat is compound interest?"
        )
        assert converted_item["instruction"] == expected_instruction
        assert (
            converted_item["response"]
            == "Compound interest is interest calculated on the initial principal and accumulated interest."
        )

        # Test conversion of second item (empty system prompt)
        converted_item2 = DatasetFormatter.convert_to_standard_format(
            data[1], dataset.detected_format
        )
        expected_instruction2 = "You are a helpful assistant which thinks step by step when responding to a user.\n\nWhat is 2+2?"
        assert converted_item2["instruction"] == expected_instruction2
        assert converted_item2["response"] == "2+2 equals 4."

    def test_detect_databricks_dolly_format(self):
        """Test detection of Databricks Dolly format (instruction, context, response)."""
        data = [
            {
                "instruction": "What is machine learning?",
                "context": "Machine learning is a subset of artificial intelligence.",
                "response": "Machine learning is a method of data analysis that automates analytical model building.",
            },
            {
                "instruction": "Explain photosynthesis",
                "context": "",
                "response": "Photosynthesis is the process by which plants convert light energy into chemical energy.",
            },
        ]

        detected_format = DatasetFormatter.detect_format(data)
        assert detected_format == ("instruction", "context", "response")

    def test_convert_databricks_dolly_format_with_context(self):
        """Test conversion of Databricks Dolly format with non-empty context."""
        item = {
            "instruction": "What is machine learning?",
            "context": "Machine learning is a subset of artificial intelligence.",
            "response": "Machine learning is a method of data analysis that automates analytical model building.",
        }

        converted_item = DatasetFormatter.convert_to_standard_format(
            item, ("instruction", "context", "response")
        )

        expected_instruction = "What is machine learning? Context for reference: Machine learning is a subset of artificial intelligence."
        assert converted_item["instruction"] == expected_instruction
        assert (
            converted_item["response"]
            == "Machine learning is a method of data analysis that automates analytical model building."
        )

    def test_convert_databricks_dolly_format_without_context(self):
        """Test conversion of Databricks Dolly format with empty context."""
        item = {
            "instruction": "Explain photosynthesis",
            "context": "",
            "response": "Photosynthesis is the process by which plants convert light energy into chemical energy.",
        }

        converted_item = DatasetFormatter.convert_to_standard_format(
            item, ("instruction", "context", "response")
        )

        # Should use instruction as-is when context is empty
        assert converted_item["instruction"] == "Explain photosynthesis"
        assert (
            converted_item["response"]
            == "Photosynthesis is the process by which plants convert light energy into chemical energy."
        )

    def test_convert_databricks_dolly_format_with_null_context(self):
        """Test conversion of Databricks Dolly format with null context."""
        item = {
            "instruction": "What is the capital of France?",
            "context": None,
            "response": "The capital of France is Paris.",
        }

        converted_item = DatasetFormatter.convert_to_standard_format(
            item, ("instruction", "context", "response")
        )

        # Should use instruction as-is when context is None
        assert converted_item["instruction"] == "What is the capital of France?"
        assert converted_item["response"] == "The capital of France is Paris."

    def test_preprocessing_with_combined_instruction_length(self):
        """Test that preprocessing correctly filters by combined instruction length."""
        from src.functionalnetworkssft.utils.model_utils import (
            preprocess_dataset_for_experiments,
        )

        # Create test data with Databricks Dolly format
        data = [
            {
                "instruction": "Short instruction",
                "context": "Short context",
                "response": "Short response",
            },
            {
                "instruction": "This is a very long instruction that exceeds the limit",
                "context": "This is also a very long context that when combined with the instruction will exceed the maximum allowed length for the combined instruction field",
                "response": "Response",
            },
            {
                "instruction": "Medium instruction",
                "context": "",  # Empty context
                "response": "Medium response",
            },
        ]

        # Test with strict limits
        filtered_data = preprocess_dataset_for_experiments(
            data,
            response_max_length=1000,
            instruction_max_length=100,  # Very strict limit
        )

        # Should only keep the first item (short instruction + context)
        # Second item should be filtered out due to combined length
        # Third item should pass (no context to combine)
        assert len(filtered_data) == 2
        assert filtered_data[0]["instruction"] == "Short instruction"
        assert filtered_data[1]["instruction"] == "Medium instruction"


if __name__ == "__main__":
    pytest.main([__file__])
