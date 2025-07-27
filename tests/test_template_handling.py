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
        if has_chat_template:
            self.chat_template = (
                chat_template
                or "{% for message in messages %}{{ message['role'] }}: {{ message['content'] }}\n{% endfor %}"
            )

    def __call__(self, text, **kwargs):
        # Mock tokenization
        tokens = text.split()[: kwargs.get("max_length", 10)]
        input_ids = torch.tensor([list(range(len(tokens)))])
        attention_mask = torch.ones_like(input_ids)
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
        assert torch.equal(item["input_ids"], item["labels"])


if __name__ == "__main__":
    pytest.main([__file__])
