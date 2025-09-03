#!/usr/bin/env python3
"""
Test suite for DatasetFormatter class.
"""

import pytest
import torch
from src.functionalnetworkssft.utils.dataset_utils import DatasetFormatter
from src.functionalnetworkssft.fnsft_trainer import InstructionDataset


class MockTokenizer:
    """Mock tokenizer for testing."""

    def __init__(self):
        self.pad_token = None
        self.eos_token = "</s>"
        self.eos_token_id = 2
        self.pad_token_id = None
        self.last_text = None

    def __call__(self, text, **kwargs):
        # Capture the last text for assertions in tests
        self.last_text = text
        # Return tensors shaped like HF tokenizers when return_tensors='pt'
        return {
            "input_ids": torch.tensor([[0, 1, 2, 3, 4]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1]], dtype=torch.long),
        }

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

        # Test question-response format (e.g., 0xZee/dataset-CoT-Atomic-Nuclear-Physics-144)
        data2b = [{"question": "What is AI?", "response": "Artificial Intelligence"}]
        assert DatasetFormatter.detect_format(data2b) == ("question", "response")
        converted = DatasetFormatter.convert_to_standard_format(
            data2b[0], ("question", "response")
        )
        assert converted["instruction"] == "What is AI?"
        assert converted["response"] == "Artificial Intelligence"

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

    def test_detect_camel_ai_physics_format(self):
        """Test detection of camel-ai/physics dataset format (message_1, message_2, topic, sub_topic)."""
        data = [
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
        ]

        detected_format = DatasetFormatter.detect_format(data)
        assert detected_format == ("message_1", "message_2", "sub_topic", "topic")

    def test_convert_camel_ai_physics_format(self):
        """Test conversion of camel-ai/physics dataset format."""
        item = {
            "message_1": "What is the speed of light?",
            "message_2": "The speed of light in vacuum is approximately 299,792,458 meters per second.",
            "topic": "Electromagnetism",
            "sub_topic": "Wave Properties",
        }

        converted_item = DatasetFormatter.convert_to_standard_format(
            item, ("message_1", "message_2", "sub_topic", "topic")
        )

        # Should map message_1 to instruction and message_2 to response
        # topic and sub_topic should be ignored
        assert converted_item["instruction"] == "What is the speed of light?"
        assert (
            converted_item["response"]
            == "The speed of light in vacuum is approximately 299,792,458 meters per second."
        )
        # Verify that topic and sub_topic are not included in the converted item
        assert "topic" not in converted_item
        assert "sub_topic" not in converted_item

    def test_mixed_format_per_item_fallback(self):
        """Ensure InstructionDataset falls back to per-item detection for mixed formats.

        Simulate combined datasets where the first 5 items are camel-ai/physics format,
        so global detection picks that. Then include a sixth item with question/response
        fields (like 0xZee's dataset) and verify __getitem__ succeeds and uses the
        correct content.
        """
        tokenizer = MockTokenizer()

        # First 5 items: camel-ai/physics format
        camel_items = [
            {
                "message_1": f"Camel question {i}?",
                "message_2": f"Camel answer {i}.",
                "topic": "Physics",
                "sub_topic": "General",
            }
            for i in range(5)
        ]

        # Sixth item: question/response format with extra CoT field
        qr_item = {
            "question": "What is the atomic number of oxygen?",
            "response": "The atomic number of oxygen is 8.",
            "CoT": "Reason it out...",
        }

        data = camel_items + [qr_item]

        # Create dataset (auto_detect_format will detect based on first 5 items)
        dataset = InstructionDataset(
            data=data,
            tokenizer=tokenizer,
            template_format="basic",
            auto_detect_format=True,
            max_length=128,
        )

        # Access the last item (question/response) - should not raise and should format text
        item = dataset.__getitem__(5)
        assert "oxygen" in tokenizer.last_text
        assert "atomic number" in tokenizer.last_text
        # Ensure labels/ids are present
        assert "input_ids" in item and "attention_mask" in item and "labels" in item


if __name__ == "__main__":
    pytest.main([__file__])
