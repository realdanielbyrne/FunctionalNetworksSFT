"""
Shared dataset utilities for automatic format detection and conversion.
"""

import logging
import warnings
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import Dataset
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)
_TEMPLATE_DECISION_LOGGED = False


class DatasetFormatter:
    """Handles automatic detection and conversion of different dataset formats."""

    # Common dataset format mappings
    FORMAT_MAPPINGS = {
        # Standard instruction-response formats
        ("instruction", "response"): lambda item: {
            "instruction": item["instruction"],
            "response": item["response"],
        },
        ("instruction", "output"): lambda item: {
            "instruction": item["instruction"],
            "response": item["output"],
        },
        # Instruction with input context
        ("instruction", "input", "output"): lambda item: {
            "instruction": (
                f"{item['instruction']}\n\nInput: {item['input']}"
                if item.get("input", "").strip()
                else item["instruction"]
            ),
            "response": item["output"],
        },
        # Input-output only (no instruction field present)
        ("input", "output"): lambda item: {
            "instruction": item.get("instruction", item["input"]),
            "response": item["output"],
        },
        # Databricks Dolly format (instruction, context, response)
        ("instruction", "context", "response"): lambda item: {
            "instruction": (
                f"{item['instruction']} Context for reference: {item['context']}"
                if item.get("context") and str(item.get("context", "")).strip()
                else item["instruction"]
            ),
            "response": item["response"],
        },
        # System-user-assistant format (e.g., Josephgflowers/Finance-Instruct-500k)
        (
            "assistant",
            "system",
            "user",
        ): lambda item: DatasetFormatter._convert_system_user_assistant_format(item),
        # Prompt-completion formats
        ("prompt", "completion"): lambda item: {
            "instruction": item["prompt"],
            "response": item["completion"],
        },
        ("prompt", "response"): lambda item: {
            "instruction": item["prompt"],
            "response": item["response"],
        },
        # Question-answer formats
        ("question", "answer"): lambda item: {
            "instruction": item["question"],
            "response": item["answer"],
        },
        # Question-response format (e.g., 0xZee/dataset-CoT-Atomic-Nuclear-Physics-144)
        ("question", "response"): lambda item: {
            "instruction": item["question"],
            "response": item["response"],
        },
        # Context-response formats (common in some datasets)
        ("context", "response"): lambda item: {
            "instruction": item["context"],
            "response": item["response"],
        },
        # Context-Response formats (capitalized - common in some datasets)
        ("Context", "Response"): lambda item: {
            "instruction": item["Context"],
            "response": item["Response"],
        },
        # Context-based formats
        ("context", "question", "answer"): lambda item: {
            "instruction": f"Context: {item['context']}\n\nQuestion: {item['question']}",
            "response": item["answer"],
        },
        # Text-only format (already formatted)
        ("text",): lambda item: {"text": item["text"]},
        # camel-ai/physics dataset format (message_1, message_2, topic, sub_topic)
        ("message_1", "message_2", "sub_topic", "topic"): lambda item: {
            "instruction": item["message_1"],
            "response": item["message_2"],
        },
        # Relaxed camel-ai/physics format (only message_1 and message_2 required)
        ("message_1", "message_2"): lambda item: {
            "instruction": item["message_1"],
            "response": item["message_2"],
        },
    }

    @staticmethod
    def detect_format(data: List[Dict[str, Any]]) -> tuple:
        """
        Detect the format of the dataset by examining the first few samples.

        Args:
            data: List of dataset samples

        Returns:
            Tuple of column names representing the detected format
        """
        if not data:
            raise ValueError("Dataset is empty")

        # Check first few samples to determine format
        sample_size = min(5, len(data))
        common_keys = None

        for i in range(sample_size):
            item = data[i]
            if not isinstance(item, dict):
                raise ValueError(f"Dataset item {i} is not a dictionary")

            item_keys = set(item.keys())
            if common_keys is None:
                common_keys = item_keys
            else:
                common_keys = common_keys.intersection(item_keys)

        if not common_keys:
            raise ValueError("No common keys found across dataset samples")

        # Sort keys for consistent format detection
        sorted_keys = tuple(sorted(common_keys))

        # Check for known formats in order of preference
        format_priority = [
            ("instruction", "input", "output"),
            ("input", "output"),
            ("instruction", "context", "response"),  # Databricks Dolly format
            (
                "message_1",
                "message_2",
                "sub_topic",
                "topic",
            ),  # Full camel-ai/physics dataset format (prioritize over generic instruction/response)
            (
                "message_1",
                "message_2",
            ),  # Relaxed camel-ai/physics (prioritize over generic instruction/response)
            ("assistant", "system", "user"),  # System-user-assistant format
            ("prompt", "completion"),
            ("prompt", "response"),
            ("question", "answer"),
            ("question", "response"),
            ("context", "response"),  # Context-response format
            ("Context", "Response"),  # Context-Response format (capitalized)
            (
                "instruction",
                "response",
            ),  # Generic instruction/response after domain-specific formats
            ("instruction", "output"),
            ("context", "question", "answer"),
            ("text",),
        ]

        for format_keys in format_priority:
            if all(key in common_keys for key in format_keys):
                return format_keys

        # If no known format is detected, check for conversational format
        if "messages" in common_keys:
            return ("messages",)

        # Fallback: use all available keys
        logger.warning(
            f"Unknown dataset format detected. Available keys: {sorted_keys}"
        )
        return sorted_keys

    @staticmethod
    def convert_to_standard_format(
        item: Dict[str, Any], format_keys: tuple
    ) -> Dict[str, str]:
        """
        Convert a dataset item to standard instruction-response format.

        Args:
            item: Dataset item
            format_keys: Detected format keys

        Returns:
            Dictionary with 'instruction' and 'response' keys or 'text' key
        """
        if format_keys in DatasetFormatter.FORMAT_MAPPINGS:
            return DatasetFormatter.FORMAT_MAPPINGS[format_keys](item)
        elif format_keys == ("messages",):
            return DatasetFormatter._convert_conversational_format(item)
        else:
            # Fallback: try to infer the format
            return DatasetFormatter._infer_and_convert(item, format_keys)

    @staticmethod
    def _convert_conversational_format(item: Dict[str, Any]) -> Dict[str, str]:
        """Convert conversational format to instruction-response format."""
        messages = item.get("messages", [])
        if not messages:
            raise ValueError("Empty messages in conversational format")

        # Extract user messages as instruction and assistant messages as response
        user_messages = []
        assistant_messages = []

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")

            if role == "user":
                user_messages.append(content)
            elif role == "assistant":
                assistant_messages.append(content)

        if not user_messages:
            raise ValueError("No user messages found in conversational format")

        instruction = "\n".join(user_messages)
        response = "\n".join(assistant_messages) if assistant_messages else ""

        return {"instruction": instruction, "response": response}

    @staticmethod
    def _convert_system_user_assistant_format(item: Dict[str, Any]) -> Dict[str, str]:
        """Convert system-user-assistant format to instruction-response format."""
        system_content = item.get("system", "")
        user_content = item.get("user", "")
        assistant_content = item.get("assistant", "")

        # Handle empty, null, or missing system field
        if (
            not system_content
            or system_content is None
            or not str(system_content).strip()
        ):
            system_content = "You are a helpful assistant which thinks step by step when responding to a user."

        # Combine system prompt with user input to form the instruction
        if str(system_content).strip():
            instruction = f"{str(system_content).strip()}\n\n{user_content}"
        else:
            instruction = user_content

        return {
            "instruction": instruction,
            "response": assistant_content,
        }

    @staticmethod
    def _infer_and_convert(item: Dict[str, Any], format_keys: tuple) -> Dict[str, str]:
        """Infer format and convert to standard format."""
        # Try to identify instruction-like and response-like fields
        instruction_candidates = ["instruction", "prompt", "question", "input", "query"]
        response_candidates = [
            "response",
            "output",
            "answer",
            "completion",
            "target",
            "result",
        ]

        instruction_key = None
        response_key = None

        for key in format_keys:
            key_lower = key.lower()
            if any(candidate in key_lower for candidate in instruction_candidates):
                instruction_key = key
            elif any(candidate in key_lower for candidate in response_candidates):
                response_key = key

        if instruction_key and response_key:
            return {
                "instruction": str(item[instruction_key]),
                "response": str(item[response_key]),
            }
        elif len(format_keys) == 1 and "text" in format_keys:
            return {"text": str(item["text"])}
        else:
            # Last resort: concatenate all fields
            combined_text = " ".join(str(item[key]) for key in format_keys)
            return {"text": combined_text}


class InstructionDataset(Dataset):
    """Enhanced dataset class for instruction-following data with intelligent chat template handling."""

    def __init__(
        self,
        data: List[Dict[str, Any]],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 2048,
        instruction_template: str = "### Instruction:\n{instruction}\n\n### Response:\n{response}",
        auto_detect_format: bool = True,
        is_pretokenized: bool = False,
        template_format: str = "auto",
        detected_format: Optional[tuple] = None,
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_pretokenized = is_pretokenized
        self.instruction_template = instruction_template
        self.auto_detect_format = auto_detect_format
        self.template_format = template_format

        # Set pad token if not exists
        if getattr(tokenizer, "pad_token", None) is None:
            eos_token = getattr(tokenizer, "eos_token", None)
            if eos_token is not None:
                tokenizer.pad_token = eos_token
                # Also set pad_token_id to ensure consistency
                if (
                    not hasattr(tokenizer, "pad_token_id")
                    or tokenizer.pad_token_id is None
                ):
                    tokenizer.pad_token_id = getattr(tokenizer, "eos_token_id", None)

        # Determine the actual template format to use
        self.actual_template_format = self._determine_template_format()
        global _TEMPLATE_DECISION_LOGGED
        if not _TEMPLATE_DECISION_LOGGED:
            logger.info(f"Using template format: {self.actual_template_format}")
            _TEMPLATE_DECISION_LOGGED = True
        else:
            logger.debug(f"Using template format: {self.actual_template_format}")

        # Use provided detected format or detect it if not provided
        if detected_format is not None:
            # Use the pre-detected format to avoid duplicate detection/logging
            self.detected_format = detected_format
        elif self.auto_detect_format and data:
            # Only detect format if not already provided
            self.detected_format = DatasetFormatter.detect_format(data)
            logger.info(f"Detected dataset format: {self.detected_format}")
        else:
            self.detected_format = None

    def _determine_template_format(self) -> str:
        """Determine the actual template format to use based on the template_format setting."""
        if self.template_format == "auto":
            # Check if tokenizer has a chat template
            if (
                hasattr(self.tokenizer, "chat_template")
                and self.tokenizer.chat_template is not None
            ):
                logger.debug(
                    "Auto-detected tokenizer chat template; using 'chat' format"
                )
                return "chat"
            else:
                logger.debug(
                    "No tokenizer chat template; falling back to 'basic' format"
                )
                return "basic"
        elif self.template_format == "chat":
            # Force use of chat template
            if (
                not hasattr(self.tokenizer, "chat_template")
                or self.tokenizer.chat_template is None
            ):
                raise ValueError(
                    "Chat template format requested but tokenizer does not have a chat_template attribute"
                )
            return "chat"
        elif self.template_format in ["alpaca", "chatml", "basic"]:
            return self.template_format
        else:
            raise ValueError(
                f"Invalid template_format: {self.template_format}. "
                f"Must be one of: auto, chat, alpaca, chatml, basic"
            )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]

        # If pre-tokenized, return the tokenized data directly
        if self.is_pretokenized:
            return {
                "input_ids": torch.tensor(item["input_ids"], dtype=torch.long),
                "attention_mask": torch.tensor(
                    item["attention_mask"], dtype=torch.long
                ),
                "labels": torch.tensor(item["labels"], dtype=torch.long),
            }

        # Convert to standard format if auto-detection is enabled
        if self.auto_detect_format and self.detected_format:
            try:
                converted_item = DatasetFormatter.convert_to_standard_format(
                    item, self.detected_format
                )
            except Exception as e:
                logger.debug(
                    f"Failed to convert item {idx}: {e}. Attempting per-item format detection."
                )
                # Fallback: try to detect/convert format for this specific item
                try:
                    item_format = DatasetFormatter.detect_format([item])
                    converted_item = DatasetFormatter.convert_to_standard_format(
                        item, item_format
                    )
                except Exception as e2:
                    logger.debug(
                        f"Per-item detection failed for item {idx}: {e2}. Using original format."
                    )
                    converted_item = item
        else:
            converted_item = item

        # Extract instruction and response
        instruction = None
        response = None

        if "instruction" in converted_item and "response" in converted_item:
            instruction = converted_item["instruction"]
            response = converted_item["response"]
        elif "text" in converted_item:
            # If we have pre-formatted text, use it directly
            text = converted_item["text"]
        else:
            # Fallback for legacy behavior
            if "instruction" in item and "response" in item:
                instruction = item["instruction"]
                response = item["response"]
            elif "text" in item:
                text = item["text"]
            else:
                raise ValueError(
                    f"Dataset item {idx} must contain either 'instruction'+'response' or 'text' fields. "
                    f"Available keys: {list(item.keys())}"
                )

        # Format the text based on the template format
        if instruction is not None and response is not None:
            text = self._format_text(instruction, response)
        # If text is already set from above, use it as-is

        # Tokenize with dynamic padding (collator will pad to batch max length)
        encoding = self.tokenizer.__call__(
            text,
            truncation=True,
            padding=False,  # dynamic padding handled by data collator
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        # Create labels with padding tokens set to -100 to ignore them in loss calculation
        labels = input_ids.clone()
        if self.tokenizer.pad_token_id is not None:
            labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def _format_text(self, instruction: str, response: str) -> str:
        """Format instruction and response using the appropriate template."""
        if self.actual_template_format == "chat":
            # Use tokenizer's chat template
            messages = [
                {"role": "user", "content": instruction},
                {"role": "assistant", "content": response},
            ]
            formatted = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            # Ensure we return a string
            if isinstance(formatted, str):
                return formatted
            else:
                raise ValueError(
                    f"Chat template returned non-string type: {type(formatted)}"
                )
        elif self.actual_template_format == "alpaca":
            # Alpaca format
            return f"### Instruction:\n{instruction}\n\n### Response:\n{response}"
        elif self.actual_template_format == "chatml":
            # ChatML format
            return f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>"
        elif self.actual_template_format == "basic":
            # Use the provided instruction template
            return self.instruction_template.format(
                instruction=instruction, response=response
            )
        else:
            raise ValueError(f"Unknown template format: {self.actual_template_format}")
