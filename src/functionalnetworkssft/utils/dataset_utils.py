"""
Shared dataset utilities for automatic format detection and conversion.
"""

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


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
            ("instruction", "context", "response"),  # Databricks Dolly format
            ("instruction", "response"),
            ("instruction", "output"),
            (
                "message_1",
                "message_2",
                "sub_topic",
                "topic",
            ),  # camel-ai/physics dataset format
            ("assistant", "system", "user"),  # System-user-assistant format
            ("prompt", "completion"),
            ("prompt", "response"),
            ("question", "answer"),
            ("context", "response"),  # Context-response format
            ("Context", "Response"),  # Context-Response format (capitalized)
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
