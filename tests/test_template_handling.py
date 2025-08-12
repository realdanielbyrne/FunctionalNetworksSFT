#!/usr/bin/env python3
"""
Test suite for intelligent chat template handling in InstructionDataset.
"""

import pytest
import torch
from unittest.mock import Mock, MagicMock
from transformers import AutoTokenizer
from src.functionalnetworkssft.fnsft_trainer import InstructionDataset


class MockTokenizer:
    """Mock tokenizer for testing."""

    def __init__(self, has_chat_template=False, chat_template=None):
        self.pad_token = None
        self.eos_token = "</s>"
        self.eos_token_id = 2
        self.pad_token_id = None
        if has_chat_template:
            self.chat_template = (
                chat_template
                or "{% for message in messages %}{{ message['role'] }}: {{ message['content'] }}\n{% endfor %}"
            )

    def __call__(self, text, **kwargs):
        # Mock tokenization with padding simulation
        max_length = kwargs.get("max_length", 10)
        padding = kwargs.get("padding", False)

        tokens = text.split()[:max_length]
        token_ids = list(range(len(tokens)))

        # Simulate padding if requested
        if padding == "max_length" and len(token_ids) < max_length:
            pad_id = self.pad_token_id if self.pad_token_id is not None else 0
            token_ids.extend([pad_id] * (max_length - len(token_ids)))

        input_ids = torch.tensor([token_ids])
        attention_mask = torch.ones_like(input_ids)

        # Set attention mask to 0 for padding tokens
        if padding == "max_length" and self.pad_token_id is not None:
            attention_mask[input_ids == self.pad_token_id] = 0

        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def apply_chat_template(
        self, messages, tokenize=False, add_generation_prompt=False
    ):
        """Mock chat template application."""
        if not hasattr(self, "chat_template") or self.chat_template is None:
            raise ValueError("No chat template available")

        # Simple mock implementation
        result = ""
        for msg in messages:
            result += f"{msg['role']}: {msg['content']}\n"
        return result.strip()


class TestInstructionDatasetTemplateHandling:
    """Test cases for template handling in InstructionDataset."""

    def test_auto_detection_with_chat_template(self):
        """Test auto-detection when tokenizer has chat template."""
        tokenizer = MockTokenizer(has_chat_template=True)
        data = [{"instruction": "Hello", "response": "Hi there!"}]

        dataset = InstructionDataset(
            data=data, tokenizer=tokenizer, template_format="auto"
        )

        assert dataset.actual_template_format == "chat"

    def test_auto_detection_without_chat_template(self):
        """Test auto-detection when tokenizer has no chat template."""
        tokenizer = MockTokenizer(has_chat_template=False)
        data = [{"instruction": "Hello", "response": "Hi there!"}]

        dataset = InstructionDataset(
            data=data, tokenizer=tokenizer, template_format="auto"
        )

        assert dataset.actual_template_format == "basic"

    def test_force_chat_template_success(self):
        """Test forcing chat template when available."""
        tokenizer = MockTokenizer(has_chat_template=True)
        data = [{"instruction": "Hello", "response": "Hi there!"}]

        dataset = InstructionDataset(
            data=data, tokenizer=tokenizer, template_format="chat"
        )

        assert dataset.actual_template_format == "chat"

    def test_force_chat_template_failure(self):
        """Test forcing chat template when not available."""
        tokenizer = MockTokenizer(has_chat_template=False)
        data = [{"instruction": "Hello", "response": "Hi there!"}]

        with pytest.raises(ValueError, match="Chat template format requested"):
            InstructionDataset(data=data, tokenizer=tokenizer, template_format="chat")

    def test_alpaca_format(self):
        """Test Alpaca format."""
        tokenizer = MockTokenizer()
        data = [{"instruction": "Hello", "response": "Hi there!"}]

        dataset = InstructionDataset(
            data=data, tokenizer=tokenizer, template_format="alpaca"
        )

        assert dataset.actual_template_format == "alpaca"

        # Test formatting
        formatted = dataset._format_text("Hello", "Hi there!")
        expected = "### Instruction:\nHello\n\n### Response:\nHi there!"
        assert formatted == expected

    def test_chatml_format(self):
        """Test ChatML format."""
        tokenizer = MockTokenizer()
        data = [{"instruction": "Hello", "response": "Hi there!"}]

        dataset = InstructionDataset(
            data=data, tokenizer=tokenizer, template_format="chatml"
        )

        assert dataset.actual_template_format == "chatml"

        # Test formatting
        formatted = dataset._format_text("Hello", "Hi there!")
        expected = "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\nHi there!<|im_end|>"
        assert formatted == expected

    def test_basic_format(self):
        """Test basic format with custom template."""
        tokenizer = MockTokenizer()
        data = [{"instruction": "Hello", "response": "Hi there!"}]
        custom_template = "Q: {instruction}\nA: {response}"

        dataset = InstructionDataset(
            data=data,
            tokenizer=tokenizer,
            instruction_template=custom_template,
            template_format="basic",
        )

        assert dataset.actual_template_format == "basic"

        # Test formatting
        formatted = dataset._format_text("Hello", "Hi there!")
        expected = "Q: Hello\nA: Hi there!"
        assert formatted == expected

    def test_chat_template_formatting(self):
        """Test chat template formatting."""
        tokenizer = MockTokenizer(has_chat_template=True)
        data = [{"instruction": "Hello", "response": "Hi there!"}]

        dataset = InstructionDataset(
            data=data, tokenizer=tokenizer, template_format="chat"
        )

        # Test formatting
        formatted = dataset._format_text("Hello", "Hi there!")
        expected = "user: Hello\nassistant: Hi there!"
        assert formatted == expected

    def test_invalid_template_format(self):
        """Test invalid template format."""
        tokenizer = MockTokenizer()
        data = [{"instruction": "Hello", "response": "Hi there!"}]

        with pytest.raises(ValueError, match="Invalid template_format"):
            InstructionDataset(
                data=data, tokenizer=tokenizer, template_format="invalid"
            )

    def test_getitem_with_instruction_response(self):
        """Test __getitem__ with instruction/response data."""
        tokenizer = MockTokenizer()
        data = [{"instruction": "Hello", "response": "Hi there!"}]

        dataset = InstructionDataset(
            data=data,
            tokenizer=tokenizer,
            template_format="alpaca",
            auto_detect_format=False,
        )

        item = dataset[0]
        assert "input_ids" in item
        assert "attention_mask" in item
        assert "labels" in item

        # The InstructionDataset sets pad_token_id to eos_token_id when it's None
        # So padding tokens should be set to -100 in labels
        padding_positions = item["input_ids"] == tokenizer.pad_token_id
        if padding_positions.any():
            # Where input_ids has padding tokens, labels should have -100
            assert torch.all(item["labels"][padding_positions] == -100)
            # Where input_ids doesn't have padding tokens, labels should match input_ids
            non_padding_positions = ~padding_positions
            assert torch.all(
                item["input_ids"][non_padding_positions]
                == item["labels"][non_padding_positions]
            )

    def test_padding_token_handling(self):
        """Test that padding tokens are properly set to -100 in labels."""
        tokenizer = MockTokenizer()
        # Set up padding token
        tokenizer.pad_token = "<pad>"
        tokenizer.pad_token_id = 999

        data = [{"instruction": "Hello", "response": "Hi!"}]

        dataset = InstructionDataset(
            data=data,
            tokenizer=tokenizer,
            template_format="alpaca",
            auto_detect_format=False,
            max_length=20,  # Force padding
        )

        item = dataset[0]

        # Check that padding tokens in input_ids are set to -100 in labels
        padding_positions = item["input_ids"] == tokenizer.pad_token_id
        if padding_positions.any():
            # Where input_ids has padding tokens, labels should have -100
            assert torch.all(item["labels"][padding_positions] == -100)
            # Where input_ids doesn't have padding tokens, labels should match input_ids
            non_padding_positions = ~padding_positions
            assert torch.all(
                item["input_ids"][non_padding_positions]
                == item["labels"][non_padding_positions]
            )

    def test_pad_token_setup_from_eos(self):
        """Test that pad_token is set from eos_token when not available."""
        tokenizer = MockTokenizer()
        # Ensure no pad token initially
        tokenizer.pad_token = None
        tokenizer.pad_token_id = None

        data = [{"instruction": "Hello", "response": "Hi!"}]

        dataset = InstructionDataset(
            data=data,
            tokenizer=tokenizer,
            template_format="alpaca",
            auto_detect_format=False,
        )

        # Check that pad_token was set from eos_token
        assert tokenizer.pad_token == tokenizer.eos_token
        assert tokenizer.pad_token_id == tokenizer.eos_token_id


if __name__ == "__main__":
    pytest.main([__file__])
