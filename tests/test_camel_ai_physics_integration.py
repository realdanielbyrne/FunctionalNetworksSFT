#!/usr/bin/env python3
"""
Integration test for camel-ai/physics dataset format support.
This test demonstrates how the camel-ai/physics dataset format works end-to-end.
"""

import pytest
from unittest.mock import MagicMock
from src.functionalnetworkssft.utils.dataset_utils import DatasetFormatter
from src.functionalnetworkssft.fnsft_trainer import InstructionDataset


class MockTokenizer:
    """Mock tokenizer for testing."""

    def __init__(self, has_chat_template=False):
        self.has_chat_template = has_chat_template
        self.pad_token = "[PAD]"
        self.pad_token_id = 0
        self.eos_token = "[EOS]"
        self.eos_token_id = 1
        self.vocab_size = 1000

    def __call__(self, text, **kwargs):
        # Simple mock tokenization
        import torch

        tokens = text.split()[:10]  # Limit to 10 tokens for testing
        input_ids = [hash(token) % 1000 for token in tokens]
        attention_mask = [1] * len(input_ids)

        return {
            "input_ids": torch.tensor([input_ids]),
            "attention_mask": torch.tensor([attention_mask]),
        }

    def decode(self, token_ids, **kwargs):
        return f"decoded_{len(token_ids)}_tokens"

    def apply_chat_template(self, messages, **kwargs):
        if not self.has_chat_template:
            raise AttributeError("No chat template available")
        return f"chat_template_applied_{len(messages)}_messages"


class TestCamelAiPhysicsIntegration:
    """Integration tests for camel-ai/physics dataset format."""

    def test_end_to_end_camel_ai_physics_processing(self):
        """Test complete processing pipeline for camel-ai/physics dataset."""
        # Sample data in camel-ai/physics format
        camel_ai_physics_data = [
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
        ]

        # Step 1: Test format detection
        detected_format = DatasetFormatter.detect_format(camel_ai_physics_data)
        assert detected_format == ("message_1", "message_2", "sub_topic", "topic")

        # Step 2: Test format conversion
        for item in camel_ai_physics_data:
            converted_item = DatasetFormatter.convert_to_standard_format(
                item, detected_format
            )

            # Verify conversion
            assert "instruction" in converted_item
            assert "response" in converted_item
            assert converted_item["instruction"] == item["message_1"]
            assert converted_item["response"] == item["message_2"]

            # Verify topic metadata is not included in converted format
            assert "topic" not in converted_item
            assert "sub_topic" not in converted_item

        # Step 3: Test integration with InstructionDataset
        tokenizer = MockTokenizer()

        dataset = InstructionDataset(
            data=camel_ai_physics_data,
            tokenizer=tokenizer,
            max_length=512,
            auto_detect_format=True,
            template_format="basic",
            detected_format=detected_format,
        )

        # Verify dataset properties
        assert len(dataset) == 3
        assert dataset.detected_format == detected_format

        # Test dataset item retrieval
        for i in range(len(dataset)):
            item = dataset[i]
            assert "input_ids" in item
            assert "attention_mask" in item
            assert "labels" in item

    def test_camel_ai_physics_format_priority(self):
        """Test that camel-ai/physics format is detected with correct priority."""
        # Data that could match multiple formats but should prioritize camel-ai/physics
        ambiguous_data = [
            {
                "message_1": "What is quantum mechanics?",
                "message_2": "Quantum mechanics is a fundamental theory in physics.",
                "topic": "Quantum Physics",
                "sub_topic": "Fundamentals",
                "instruction": "This could be an instruction field",  # Could match instruction format
                "response": "This could be a response field",  # Could match instruction-response format
            }
        ]

        detected_format = DatasetFormatter.detect_format(ambiguous_data)

        # Should detect camel-ai/physics format due to priority ordering
        assert detected_format == ("message_1", "message_2", "sub_topic", "topic")

    def test_camel_ai_physics_with_missing_metadata(self):
        """Test handling of camel-ai/physics format with missing topic metadata."""
        # Data with missing or empty topic fields
        data_with_missing_metadata = [
            {
                "message_1": "What is force?",
                "message_2": "Force is any interaction that changes the motion of an object.",
                "topic": "",  # Empty topic
                "sub_topic": "Basic Concepts",
            },
            {
                "message_1": "Define acceleration.",
                "message_2": "Acceleration is the rate of change of velocity with respect to time.",
                "topic": "Kinematics",
                "sub_topic": "",  # Empty sub_topic
            },
        ]

        # Should still detect the format correctly
        detected_format = DatasetFormatter.detect_format(data_with_missing_metadata)
        assert detected_format == ("message_1", "message_2", "sub_topic", "topic")

        # Should still convert correctly
        for item in data_with_missing_metadata:
            converted_item = DatasetFormatter.convert_to_standard_format(
                item, detected_format
            )
            assert converted_item["instruction"] == item["message_1"]
            assert converted_item["response"] == item["message_2"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
