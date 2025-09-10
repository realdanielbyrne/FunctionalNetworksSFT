#!/usr/bin/env python3
"""
Functional Network Supervised Fine-Tuning (SFT) Script for Language Models and Quantized Languager Models

This script provides a complete solution for fine-tuning language models
using LoRA/QLoRA techniques with support for various model architectures, datasets, and chat formats.

Researchers have found that neurons in LLMs form functional networks analogous to functional brain
networks. These are sets of neurons that consistently co-activate under certain conditions.
Crucially, only a small fraction of neurons may constitute key networks essential for performance: masking
these key networks (setting their outputs to zero) significantly degrades model performance, whereas
retaining only these networks (masking all others) can still maintain much of the model’s functionality.
Prior work even showed that manipulating important neurons’ outputs via amplification or masking can steer model behavior.

Our goal is to leverage these insights by introducing binary neuron masks during fine-tuning. This mask
will zero-out either a chosen functional network (to ablate it) or all but that network (to isolate it). The
masking is applied in the forward pass to the outputs of specific neurons, thereby affecting which neurons
contribute to model computations and which gradients are updated. This allows us to fine-tune the model
with or without certain functional subnetworks, potentially leading to fine-tuned models where only key neuron weights
are updated to accomodate new knowledge thus potentially mitigating the negative effects of full parameter fine tuning.

Author: Daniel Byrne
License: MIT
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Literal
import warnings

# Disable tokenizer parallelism to avoid forking warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from torch.utils.data import Dataset
import transformers
from transformers.utils.quantization_config import BitsAndBytesConfig
from transformers.training_args import TrainingArguments
from transformers.trainer import Trainer
from transformers.trainer_callback import EarlyStoppingCallback
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.modeling_utils import PreTrainedModel
from datasets import load_dataset, Dataset as HFDataset, load_from_disk
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
import wandb
from tqdm.auto import tqdm

# Import whoami at module level for testing
try:
    from huggingface_hub import whoami
except ImportError:
    whoami = None

# ===================== Legacy ICA helpers (for backward compatibility) =====================
import numpy as np
import json


# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)


# Configure logging only if not already configured
def setup_logging(log_file="sft_training.log"):
    """Set up logging configuration if not already configured."""
    root_logger = logging.getLogger()

    # Check if logging is already configured (has handlers)
    if root_logger.handlers:
        # Logging already configured, just get our logger
        return logging.getLogger(__name__)

    # Configure logging for the first time
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger(__name__)


logger = setup_logging()


def load_env_file():
    """Load environment variables from .env file if it exists."""
    env_file = Path(".env")
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    # Remove quotes if present
                    value = value.strip('"').strip("'")
                    os.environ[key] = value
    else:
        logger.debug("No .env file found")


@dataclass
class ModelArguments:
    """Arguments for model configuration."""

    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        }
    )
    use_auth_token: bool = field(
        default=True,
        metadata={"help": "Use HuggingFace auth token for private models"},
    )
    trust_remote_code: bool = field(
        default=True, metadata={"help": "Trust remote code when loading model"}
    )
    torch_dtype: str = field(
        default="auto",
        metadata={
            "help": "Torch dtype for model loading (auto, float16, bfloat16, float32)"
        },
    )
    attn_implementation: str = field(
        default="auto",
        metadata={
            "help": "Attention kernel implementation to request (auto, eager, sdpa, flash_attention_2)."
        },
    )


@dataclass
class DataArguments:
    """Arguments for data configuration."""

    dataset_name_or_path: str = field(
        metadata={"help": "Path to dataset file or HuggingFace dataset name"}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "Configuration name for HuggingFace dataset"}
    )
    max_seq_length: int = field(
        default=2048, metadata={"help": "Maximum sequence length for tokenization"}
    )
    instruction_template: str = field(
        default="### Instruction:\n{instruction}\n\n### Response:\n{response}",
        metadata={"help": "Template for formatting instruction-response pairs"},
    )
    validation_split: float = field(
        default=0.1, metadata={"help": "Fraction of data to use for validation"}
    )
    use_existing_splits: bool = field(
        default=True,
        metadata={
            "help": "Whether to use existing validation/test splits from HuggingFace datasets"
        },
    )
    auto_detect_format: bool = field(
        default=True,
        metadata={"help": "Automatically detect and convert dataset format"},
    )
    template_format: str = field(
        default="auto",
        metadata={"help": "Template format to use: auto, chat, alpaca, chatml, basic"},
    )
    # Dataset preprocessing parameters
    response_max_length: int = field(
        default=4000,
        metadata={
            "help": "Maximum allowed length for response field during preprocessing"
        },
    )
    instruction_max_length: int = field(
        default=2048,
        metadata={
            "help": "Maximum allowed length for combined instruction (including context) during preprocessing"
        },
    )


@dataclass
class QuantizationArguments:
    """Arguments for quantization configuration."""

    use_4bit: bool = field(default=True, metadata={"help": "Use 4-bit quantization"})
    use_8bit: bool = field(
        default=False,
        metadata={"help": "Use 8-bit quantization (overrides 4-bit if True)"},
    )
    bnb_4bit_compute_dtype: str = field(
        default="float16", metadata={"help": "Compute dtype for 4-bit quantization"}
    )
    bnb_4bit_quant_type: str = field(
        default="nf4", metadata={"help": "Quantization type for 4-bit (nf4, fp4)"}
    )
    bnb_4bit_use_double_quant: bool = field(
        default=True, metadata={"help": "Use double quantization for 4-bit"}
    )


@dataclass
class LoRAArguments:
    """Arguments for LoRA configuration."""

    use_peft: bool = field(
        default=True,
        metadata={
            "help": "Use Parameter-Efficient Fine-Tuning (LoRA/QLoRA). If False, performs full parameter fine-tuning."
        },
    )
    lora_r: int = field(default=16, metadata={"help": "LoRA rank"})
    lora_alpha: int = field(default=32, metadata={"help": "LoRA alpha parameter"})
    lora_dropout: float = field(default=0.1, metadata={"help": "LoRA dropout"})
    lora_target_modules: Optional[List[str]] = field(
        default=None,
        metadata={"help": "Target modules for LoRA (auto-detected if None)"},
    )
    lora_bias: Literal["none", "all", "lora_only"] = field(
        default="none", metadata={"help": "LoRA bias type (none, all, lora_only)"}
    )


# Import shared utilities
from .utils.dataset_utils import DatasetFormatter
from .utils.model_utils import (
    get_optimal_device,
    get_recommended_dtype,
    is_quantization_supported,
    load_quantization_config,
    setup_lora,
    load_dataset_from_path,
    load_dataset_with_splits,
    split_dataset,
    prepare_train_val_splits,
    save_model_and_tokenizer,
    convert_to_gguf,
)

# Import new modular utilities
from .utils.hf_utilities import upload_to_hub

# Lazy import ICAMask to avoid heavy dependencies during import (e.g., in tests)
try:
    from .ica_mask import ICAMask  # type: ignore
except Exception:  # pragma: no cover - only exercised when sklearn is unavailable
    ICAMask = None  # type: ignore


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
        logger.info(f"Using template format: {self.actual_template_format}")

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
                logger.info(
                    "Auto-detected tokenizer chat template, using 'chat' format"
                )
                return "chat"
            else:
                logger.info(
                    "No tokenizer chat template found, falling back to 'basic' format"
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


class DataCollatorForCausalLMWithPadding:
    """Pad batch dynamically and set labels to -100 on padded positions for causal LM."""

    def __init__(
        self, tokenizer: PreTrainedTokenizerBase, pad_to_multiple_of: Optional[int] = 8
    ):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Build features without labels for tokenizer.pad (it doesn't know how to pad 'labels')
        feats = []
        for f in features:
            feats.append(
                {
                    "input_ids": (
                        f["input_ids"].tolist()
                        if isinstance(f["input_ids"], torch.Tensor)
                        else f["input_ids"]
                    ),
                    "attention_mask": (
                        f["attention_mask"].tolist()
                        if isinstance(f["attention_mask"], torch.Tensor)
                        else f["attention_mask"]
                    ),
                }
            )

        # Pad input_ids and attention_mask
        # Suppress the fast tokenizer warning since we're intentionally using pad() for dynamic batching
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=".*using the `__call__` method is faster.*"
            )
            be = self.tokenizer.pad(
                feats,
                padding=True,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors="pt",
            )

        # Convert to plain dict
        batch = {k: v for k, v in be.items()}

        # Create labels from padded input_ids and mask out padding
        labels = batch["input_ids"].clone()
        if getattr(self.tokenizer, "pad_token_id", None) is not None:
            labels[batch["input_ids"] == self.tokenizer.pad_token_id] = -100
        else:
            # Fall back to attention mask when no pad token id is defined
            labels[batch["attention_mask"] == 0] = -100
        batch["labels"] = labels
        return batch


def create_pretokenization_function(
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
    instruction_template: str,
    template_format: str,
    detected_format: Optional[tuple],
    auto_detect_format: bool,
):
    """Create a function for pre-tokenizing dataset items via HuggingFace datasets.map()."""

    # Determine the actual template format to use
    if template_format == "auto":
        if hasattr(tokenizer, "chat_template") and tokenizer.chat_template is not None:
            actual_template_format = "chat"
        else:
            actual_template_format = "basic"
    else:
        actual_template_format = template_format

    def _format_text_for_pretokenization(instruction: str, response: str) -> str:
        """Format instruction and response using the appropriate template."""
        if actual_template_format == "chat":
            messages = [
                {"role": "user", "content": instruction},
                {"role": "assistant", "content": response},
            ]
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            return formatted if isinstance(formatted, str) else str(formatted)
        elif actual_template_format == "alpaca":
            return f"### Instruction:\n{instruction}\n\n### Response:\n{response}"
        elif actual_template_format == "chatml":
            return f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>"
        elif actual_template_format == "basic":
            return instruction_template.format(
                instruction=instruction, response=response
            )
        else:
            raise ValueError(f"Unknown template format: {actual_template_format}")

    def pretokenize_item(item):
        """Pre-tokenize a single dataset item."""
        # Convert to standard format if auto-detection is enabled
        if auto_detect_format and detected_format:
            try:
                converted_item = DatasetFormatter.convert_to_standard_format(
                    item, detected_format
                )
            except Exception:
                # Fallback: try per-item detection
                try:
                    item_format = DatasetFormatter.detect_format([item])
                    converted_item = DatasetFormatter.convert_to_standard_format(
                        item, item_format
                    )
                except Exception:
                    logger.debug(
                        "Pretokenize per-item detection failed; using original format."
                    )
                    converted_item = item
        else:
            converted_item = item

        # Extract instruction and response
        if "instruction" in converted_item and "response" in converted_item:
            instruction = converted_item["instruction"]
            response = converted_item["response"]
            text = _format_text_for_pretokenization(instruction, response)
        elif "text" in converted_item:
            text = converted_item["text"]
        elif "instruction" in item and "response" in item:
            instruction = item["instruction"]
            response = item["response"]
            text = _format_text_for_pretokenization(instruction, response)
        elif "text" in item:
            text = item["text"]
        else:
            raise ValueError(
                f"Dataset item must contain either 'instruction'+'response' or 'text' fields. Available keys: {list(item.keys())}"
            )

        # Tokenize
        encoding = tokenizer(
            text,
            truncation=True,
            padding=False,
            max_length=max_length,
            return_tensors=None,  # Return lists, not tensors
        )

        # Create labels (copy of input_ids)
        labels = encoding["input_ids"]

        return {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "labels": labels,
        }

    return pretokenize_item


def get_cache_key(
    dataset_path: str,
    tokenizer_name: str,
    template_format: str,
    max_seq_length: int,
    detected_format: Optional[tuple],
    instruction_template: str,
) -> str:
    """Generate a cache key for pre-tokenized datasets."""
    import hashlib

    key_parts = [
        dataset_path,
        tokenizer_name,
        template_format,
        str(max_seq_length),
        str(detected_format),
        instruction_template,
    ]
    key_string = "|".join(key_parts)
    return hashlib.md5(key_string.encode()).hexdigest()


def pretokenize_and_cache_dataset(
    data: List[Dict[str, Any]],
    tokenizer: PreTrainedTokenizerBase,
    data_args: DataArguments,
    detected_format: Optional[tuple],
    cache_dir: str,
    split_name: str,
) -> List[Dict[str, Any]]:
    """Pre-tokenize dataset and cache results to disk."""

    # Generate cache key
    cache_key = get_cache_key(
        dataset_path=data_args.dataset_name_or_path,
        tokenizer_name=tokenizer.name_or_path,
        template_format=data_args.template_format,
        max_seq_length=data_args.max_seq_length,
        detected_format=detected_format,
        instruction_template=data_args.instruction_template,
    )

    cache_path = os.path.join(cache_dir, f"{split_name}_{cache_key}")

    # Try to load from cache
    if os.path.exists(cache_path):
        try:
            logger.info(
                f"Loading pre-tokenized {split_name} dataset from cache: {cache_path}"
            )
            cached_dataset = load_from_disk(cache_path)
            return [item for item in cached_dataset]
        except Exception as e:
            logger.warning(f"Failed to load cached dataset: {e}. Re-tokenizing...")

    # Pre-tokenize the dataset
    logger.info(f"Pre-tokenizing {split_name} dataset ({len(data)} examples)...")

    # Convert to HF Dataset
    hf_dataset = HFDataset.from_list(data)

    # Create pre-tokenization function
    pretokenize_fn = create_pretokenization_function(
        tokenizer=tokenizer,
        max_length=data_args.max_seq_length,
        instruction_template=data_args.instruction_template,
        template_format=data_args.template_format,
        detected_format=detected_format,
        auto_detect_format=data_args.auto_detect_format,
    )

    # Apply pre-tokenization
    tokenized_dataset = hf_dataset.map(
        pretokenize_fn,
        remove_columns=hf_dataset.column_names,
        num_proc=min(4, os.cpu_count() or 1),
        desc=f"Pre-tokenizing {split_name}",
    )

    # Save to cache
    os.makedirs(cache_dir, exist_ok=True)
    tokenized_dataset.save_to_disk(cache_path)
    logger.info(f"Cached pre-tokenized {split_name} dataset to: {cache_path}")

    # Convert back to list
    return [item for item in tokenized_dataset]


def load_quantization_config_from_args(
    quant_args: QuantizationArguments,
) -> Optional[BitsAndBytesConfig]:
    """Load quantization configuration from arguments."""
    return load_quantization_config(
        use_4bit=quant_args.use_4bit,
        use_8bit=quant_args.use_8bit,
        bnb_4bit_compute_dtype=quant_args.bnb_4bit_compute_dtype,
        bnb_4bit_quant_type=quant_args.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=quant_args.bnb_4bit_use_double_quant,
    )


def load_model_and_tokenizer(
    model_args: ModelArguments, quant_config: Optional[BitsAndBytesConfig]
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load model and tokenizer with cross-platform support."""
    logger.info(f"Loading model: {model_args.model_name_or_path}")

    # Get optimal device and dtype for current platform
    device, device_name = get_optimal_device()
    logger.info(f"Target device: {device_name}")

    # Determine torch dtype - use platform-aware recommendation if not specified
    if model_args.torch_dtype == "auto":
        torch_dtype = get_recommended_dtype()
        logger.info(f"Auto-selected dtype: {torch_dtype}")
    else:
        # Parse the specified dtype string
        if model_args.torch_dtype == "float16":
            torch_dtype = torch.float16
        elif model_args.torch_dtype == "bfloat16":
            torch_dtype = torch.bfloat16
        elif model_args.torch_dtype == "float32":
            torch_dtype = torch.float32
        else:
            logger.warning(
                f"Unknown torch_dtype '{model_args.torch_dtype}', defaulting to float16"
            )
            torch_dtype = torch.float16
        logger.info(f"Using specified dtype: {torch_dtype}")

    # Get authentication token
    auth_token = None
    if model_args.use_auth_token:
        # First try environment variable
        auth_token = os.getenv("HF_TOKEN")
        if auth_token:
            logger.info("Using HF_TOKEN from environment")
        else:
            # Try to use cached credentials from huggingface-cli login
            try:
                if whoami is not None:
                    user_info = whoami()  # This will use cached token if available
                    logger.info(
                        f"Using cached HuggingFace credentials for user: {user_info['name']}"
                    )
                    auth_token = (
                        True  # Set to True to indicate we should use cached credentials
                    )
                else:
                    raise ImportError("whoami not available")
            except Exception as e:
                logger.warning(
                    f"No HF_TOKEN environment variable and no cached credentials found: {e}"
                )
                auth_token = None

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
        token=auth_token,
    )
    # Set pad token if not exists
    if getattr(tokenizer, "pad_token", None) is None:
        eos_token = getattr(tokenizer, "eos_token", None)
        if eos_token is not None:
            tokenizer.pad_token = eos_token
            tokenizer.pad_token_id = getattr(tokenizer, "eos_token_id", None)

    # Load model with cross-platform device mapping
    # Handle CUDA compatibility issues with newer GPUs
    device_map = None
    if device.type in ["cuda", "mps"]:
        try:
            # Test if GPU operations work with this device
            test_tensor = torch.randn(10, 10).to(device)
            _ = torch.matmul(test_tensor, test_tensor)
            device_map = "auto"
            logger.info("GPU operations verified - using automatic device mapping")
        except RuntimeError as e:
            logger.warning(
                f"GPU test failed: {e} - falling back to manual device placement"
            )

    # Prepare model loading kwargs
    model_kwargs = {
        "quantization_config": quant_config,
        "torch_dtype": torch_dtype,
        "trust_remote_code": model_args.trust_remote_code,
        "token": auth_token,
        "device_map": device_map,
    }

    # Add attention implementation if specified and not "auto"
    if model_args.attn_implementation != "auto":
        model_kwargs["attn_implementation"] = model_args.attn_implementation
        logger.info(f"Using attention implementation: {model_args.attn_implementation}")

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        **model_kwargs,
    )

    # Move to device if device_map wasn't used
    if device_map is None:
        try:
            model = model.to(device)
            logger.info(f"Model moved to {device}")
        except RuntimeError as e:
            if device.type == "cuda":
                logger.warning(f"!!!! Failed to move model to CUDA: {e}")
                device = torch.device("cpu")
                model = model.to(device)
            else:
                raise

    return model, tokenizer


def setup_lora_from_args(
    model: PreTrainedModel, lora_args: LoRAArguments
) -> PreTrainedModel:
    """Setup LoRA configuration from arguments."""
    return setup_lora(
        model=model,
        use_peft=lora_args.use_peft,
        lora_r=lora_args.lora_r,
        lora_alpha=lora_args.lora_alpha,
        lora_dropout=lora_args.lora_dropout,
        lora_target_modules=lora_args.lora_target_modules,
        lora_bias=lora_args.lora_bias,
    )


def load_dataset_from_args(data_args: DataArguments) -> List[Dict[str, Any]]:
    """Load dataset from arguments (legacy function - loads train split only)."""
    return load_dataset_from_path(
        dataset_name_or_path=data_args.dataset_name_or_path,
        dataset_config_name=data_args.dataset_config_name,
    )


def load_dataset_with_splits_from_args(
    data_args: DataArguments,
) -> Dict[str, List[Dict[str, Any]]]:
    """Load dataset with all available splits from arguments."""
    return load_dataset_with_splits(
        dataset_name_or_path=data_args.dataset_name_or_path,
        dataset_config_name=data_args.dataset_config_name,
    )


def log_training_mode_details(use_peft: bool, model: PreTrainedModel) -> None:
    """Log detailed information about the training mode and its implications."""
    logger.info("=" * 60)
    logger.info("TRAINING MODE CONFIGURATION")
    logger.info("=" * 60)

    if use_peft:
        logger.info("Training Mode: Parameter-Efficient Fine-Tuning (PEFT)")
        logger.info("Adapter Configuration:")
        if hasattr(model, "peft_config") and model.peft_config:
            logger.info("   - PEFT adapters configured and active")
        else:
            logger.info("   - No PEFT configuration detected")
    else:
        logger.info("Training Mode: Full Parameter Fine-Tuning")

    if torch.cuda.is_available():
        try:
            memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            memory_reserved = torch.cuda.memory_reserved() / 1024**3  # GB
            logger.info(f"GPU Memory Usage:")
            logger.info(f"   - Allocated: {memory_allocated:.2f} GB")
            logger.info(f"   - Reserved: {memory_reserved:.2f} GB")
        except Exception as e:
            logger.debug(f"Could not get GPU memory info: {e}")

    logger.info("=" * 60)


def adjust_training_args_for_mode(
    training_args: TrainingArguments, use_peft: bool
) -> TrainingArguments:
    """Adjust training arguments based on whether PEFT or full fine-tuning is used."""
    # Check if we're on MPS (Apple Silicon)
    device, _ = get_optimal_device()
    is_mps = device.type == "mps"

    if use_peft and is_mps:
        # PEFT on MPS has gradient computation issues with gradient checkpointing
        if training_args.gradient_checkpointing:
            logger.warning(
                "Disabling gradient checkpointing for PEFT on MPS due to gradient computation issues"
            )
            training_args.gradient_checkpointing = False

    if not use_peft:
        # For full fine-tuning, we might want to adjust some parameters
        logger.info("Adjusting training arguments for full parameter fine-tuning")

        # Full fine-tuning typically benefits from:
        # - Lower learning rates to avoid catastrophic forgetting
        # - More aggressive gradient checkpointing to save memory
        # - Potentially different warmup strategies

        # Adjust learning rate if it's the default PEFT learning rate
        if training_args.learning_rate == 2e-4:  # Default PEFT learning rate
            new_lr = 5e-5  # More conservative for full fine-tuning
            logger.info(
                f"Adjusting learning rate for full fine-tuning: {training_args.learning_rate} -> {new_lr}"
            )
            training_args.learning_rate = new_lr

        # Ensure gradient checkpointing is enabled for memory efficiency (except on MPS if problematic)
        if not training_args.gradient_checkpointing and not is_mps:
            logger.info(
                "Enabling gradient checkpointing for full fine-tuning memory efficiency"
            )
            training_args.gradient_checkpointing = True
        elif is_mps:
            logger.info(
                "Gradient checkpointing may cause issues on MPS - keeping current setting"
            )

        # Adjust warmup ratio for full fine-tuning
        if training_args.warmup_ratio == 0.03:  # Default value
            new_warmup = 0.1  # More warmup for full fine-tuning
            logger.info(
                f"Adjusting warmup ratio for full fine-tuning: {training_args.warmup_ratio} -> {new_warmup}"
            )
            training_args.warmup_ratio = new_warmup
    else:
        logger.info("Using training arguments optimized for PEFT")

    return training_args


def create_trainer(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    train_dataset: Dataset,
    eval_dataset: Optional[Dataset],
    training_args: TrainingArguments,
    data_collator,
) -> Trainer:
    """Create and configure the trainer."""

    # Add early stopping callback
    callbacks = [EarlyStoppingCallback(early_stopping_patience=3)]

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        callbacks=callbacks,
    )

    return trainer


def load_config_from_yaml(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def main(log_file=None):
    """Main training function."""
    # Set up logging with custom log file if provided
    global logger
    if log_file:
        logger = setup_logging(log_file)

    # Load environment variables from .env file if it exists
    load_env_file()

    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Supervised Fine-Tuning for Language Models"
    )

    # Configuration file option
    parser.add_argument(
        "--config", type=str, default=None, help="Path to YAML configuration file"
    )

    # Model arguments
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=False,
        help="Path to pretrained model or model identifier",
    )
    parser.add_argument(
        "--use_auth_token", action="store_true", help="Use HuggingFace auth token"
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Trust remote code when loading model",
    )
    parser.add_argument(
        "--torch_dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Torch dtype for model loading",
    )

    # Data arguments
    parser.add_argument(
        "--dataset_name_or_path",
        type=str,
        required=False,
        help="Path to dataset file or HuggingFace dataset name",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="Configuration name for HuggingFace dataset",
    )

    # Attention implementation flag
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="auto",
        choices=["auto", "eager", "sdpa", "flash_attention_2"],
        help=(
            "Attention implementation for models that support it. 'auto' uses model default, "
            "'sdpa' uses PyTorch SDPA (requires PyTorch 2.0+), 'flash_attention_2' uses Flash Attention v2 (if installed)."
        ),
    )
    parser.add_argument(
        "--max_seq_length", type=int, default=2048, help="Maximum sequence length"
    )
    parser.add_argument(
        "--instruction_template",
        type=str,
        default="### Instruction:\n{instruction}\n\n### Response:\n{response}",
        help="Template for formatting instruction-response pairs",
    )
    parser.add_argument(
        "--validation_split",
        type=float,
        default=0.1,
        help="Fraction of data for validation",
    )
    parser.add_argument(
        "--use_existing_splits",
        action="store_true",
        default=True,
        help="Use existing validation/test splits from HuggingFace datasets",
    )
    parser.add_argument(
        "--no_use_existing_splits",
        dest="use_existing_splits",
        action="store_false",
        help="Disable using existing splits and create custom validation split",
    )
    parser.add_argument(
        "--auto_detect_format",
        action="store_true",
        default=True,
        help="Automatically detect and convert dataset format",
    )
    parser.add_argument(
        "--no_auto_detect_format",
        dest="auto_detect_format",
        action="store_false",
        help="Disable automatic dataset format detection",
    )
    parser.add_argument(
        "--template_format",
        type=str,
        default="auto",
        choices=["auto", "chat", "alpaca", "chatml", "basic"],
        help="Template format to use: auto (use tokenizer's chat template if available, otherwise basic), "
        "chat (force tokenizer's chat template), alpaca (Alpaca format), "
        "chatml (ChatML format), basic (use instruction_template)",
    )

    parser.add_argument(
        "--response_max_length",
        type=int,
        default=4000,
        help="Maximum allowed length for response field during preprocessing",
    )
    parser.add_argument(
        "--instruction_max_length",
        type=int,
        default=2048,
        help="Maximum allowed length for combined instruction (including context) during preprocessing",
    )

    # Quantization arguments
    parser.add_argument(
        "--use_4bit", action="store_true", help="Use 4-bit quantization"
    )
    parser.add_argument(
        "--use_8bit", action="store_true", default=True, help="Use 8-bit quantization"
    )
    parser.add_argument(
        "--bnb_4bit_compute_dtype",
        type=str,
        default="float16",
        help="Compute dtype for 4-bit quantization",
    )
    parser.add_argument(
        "--bnb_4bit_quant_type",
        type=str,
        default="nf4",
        choices=["nf4", "fp4"],
        help="4-bit quantization type",
    )
    parser.add_argument(
        "--bnb_4bit_use_double_quant",
        action="store_true",
        default=True,
        help="Use double quantization for 4-bit",
    )

    # PEFT/LoRA arguments
    parser.add_argument(
        "--no_peft",
        action="store_true",
        default=False,
        help="Disable PEFT and use full parameter fine-tuning instead. By default, PEFT (LoRA/QLoRA) is used.",
    )
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout")
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        nargs="+",
        default=None,
        help="Target modules for LoRA",
    )
    parser.add_argument(
        "--lora_bias",
        type=str,
        default="none",
        choices=["none", "all", "lora_only"],
        help="LoRA bias type",
    )

    # Training arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        required=False,
        help="Output directory for model checkpoints",
    )
    parser.add_argument(
        "--num_train_epochs", type=int, default=3, help="Number of training epochs"
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=-1,
        help="Maximum number of training steps (overrides num_train_epochs if > 0)",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=4,
        help="Training batch size per device",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=4,
        help="Evaluation batch size per device",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=2e-4, help="Learning rate"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.001, help="Weight decay"
    )
    parser.add_argument("--warmup_ratio", type=float, default=0.03, help="Warmup ratio")
    parser.add_argument(
        "--lr_scheduler_type",
        type=str,
        default="cosine",
        help="Learning rate scheduler type",
    )
    parser.add_argument("--logging_steps", type=int, default=10, help="Logging steps")
    parser.add_argument(
        "--save_steps", type=int, default=100, help="Save checkpoint steps"
    )
    parser.add_argument("--eval_steps", type=int, default=500, help="Evaluation steps")
    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=3,
        help="Maximum number of checkpoints to keep",
    )
    parser.add_argument(
        "--load_best_model_at_end",
        action="store_true",
        default=True,
        help="Load best model at end of training",
    )
    parser.add_argument(
        "--metric_for_best_model",
        type=str,
        default="eval_loss",
        help="Metric for best model selection",
    )
    parser.add_argument(
        "--greater_is_better",
        action="store_true",
        help="Whether higher metric is better",
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Maximum gradient norm for gradient clipping",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        default=True,
        help="Enable gradient checkpointing for memory efficiency",
    )

    # Pre-tokenization options
    parser.add_argument(
        "--pretokenize",
        action="store_true",
        default=False,
        help="Enable pre-tokenization via HuggingFace datasets.map and cache to disk",
    )
    parser.add_argument(
        "--pretokenize_cache_dir",
        type=str,
        default=None,
        help="Directory to store/load tokenized dataset cache. Defaults to <output_dir>/tokenized_cache",
    )

    # Additional options
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Resume training from checkpoint",
    )
    parser.add_argument(
        "--use_wandb", action="store_true", help="Use Weights & Biases for logging"
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="sft-training",
        help="Weights & Biases project name",
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="Custom Weights & Biases run name",
    )
    parser.add_argument(
        "--convert_to_gguf",
        action="store_true",
        help="Convert final model to GGUF format",
    )
    parser.add_argument(
        "--gguf_quantization", type=str, default="q4_0", help="GGUF quantization type"
    )

    # Hugging Face Hub upload options
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Upload the fine-tuned model to Hugging Face Hub",
    )
    parser.add_argument(
        "--hub_repo_id",
        type=str,
        default=None,
        help="Repository ID for Hugging Face Hub (e.g., 'username/model-name')",
    )
    parser.add_argument(
        "--hub_commit_message",
        type=str,
        default=None,
        help="Commit message for Hub upload",
    )
    parser.add_argument(
        "--hub_private", action="store_true", help="Create private repository on Hub"
    )
    parser.add_argument(
        "--hub_token",
        type=str,
        default=None,
        help="Hugging Face authentication token (or set HF_TOKEN env var)",
    )
    parser.add_argument(
        "--merge_adapter_with_base",
        action="store_true",
        help="Merge trained LoRA adapter with base model after training completion. Saves both adapter and merged model in separate subdirectories.",
    )
    parser.add_argument(
        "--upload_merged_model",
        action="store_true",
        help="When both --merge_adapter_with_base and --push_to_hub are enabled, upload the merged model instead of the adapter to Hub.",
    )

    # Masking and ICA arguments
    parser.add_argument(
        "--mask_mode",
        type=str,
        choices=["lesion", "preserve", "key", "complement"],
        default=None,
        help="Masking polarity. Use 'lesion' (zero selected) or 'preserve' (keep selected). Legacy values 'key'/'complement' are still accepted but deprecated.",
    )
    parser.add_argument(
        "--ica_components",
        type=int,
        default=5,
        help="An integer argument that specifies the number of independent components to extract when ICA needs to be performed on-the-fly during training.",
    )
    parser.add_argument(
        "--ica_percentile",
        type=float,
        default=98.0,
        help="Percentile threshold (0-100). For selection_mode 'max_abs' and 'l2', selects neurons with scores >= this percentile. For 'topk', selects the top (100 - percentile) percent of neurons by score.",
    )
    parser.add_argument(
        "--ica_mask_layers",
        type=str,
        default=None,
        help="Specify which transformer layers should have ICA masking applied. Supports single layers ('0'), multiple layers ('0,3,7'), ranges ('0:4,5:6,9:'), and mixed formats ('0,2:5,8'). If not provided, ICA masking is applied to all layers (default behavior).",
    )
    parser.add_argument(
        "--ica_dtype",
        type=str,
        default=None,
        choices=[None, "auto", "float32", "float16", "bfloat16"],
        help="Data type for ICA computation. None/float32 (default) uses float32 for maximum numerical stability. 'auto' matches model dtype but uses float32 for half-precision models. 'float16'/'bfloat16' use reduced precision for better performance but may affect stability.",
    )

    # new ICA Global Path arguments
    parser.add_argument(
        "--ica_component_ids",
        type=int,
        nargs="+",
        default=None,
        help="For --ica_mode global: list of ICA component ids to target (e.g., 0 1 2). If omitted, defaults to [0].",
    )
    parser.add_argument(
        "--ica_template_path",
        type=str,
        default=None,
        help="Path to global ICA templates JSON (build with build_templates_from_current_components).",
    )

    args = parser.parse_args()
    print(args.hub_token)

    # Load configuration from YAML if provided
    if args.config:
        config = load_config_from_yaml(args.config)
        # Config file values override defaults, but CLI args override config
        # Store original CLI args that were explicitly provided
        original_argv = sys.argv[1:]  # Exclude script name
        explicitly_provided = set()

        # Parse which arguments were explicitly provided on command line
        i = 0
        while i < len(original_argv):
            arg = original_argv[i]
            if arg.startswith("--"):
                arg_name = arg[2:].replace("-", "_")
                explicitly_provided.add(arg_name)
                # Skip the value if this argument takes one
                for action in parser._actions:
                    if action.dest == arg_name and action.nargs != 0:
                        i += 1  # Skip the value
                        break
            i += 1

        # Apply config values, but don't override explicitly provided CLI args
        for key, value in config.items():
            if key not in explicitly_provided:
                # YAML already handles type conversion correctly, so use the value as-is
                # unless we need special handling
                converted_value = value

                # Handle special cases where YAML conversion might need adjustment
                if value is None:
                    # YAML null values are already None, keep them as None
                    converted_value = None
                elif isinstance(value, str) and value.lower() == "null":
                    # Handle string "null" as None
                    converted_value = None

                logger.debug(
                    f"Setting {key} from config: {converted_value} (type: {type(converted_value)})"
                )
                setattr(args, key, converted_value)

    # Validate required arguments
    required_args = ["model_name_or_path", "dataset_name_or_path", "output_dir"]
    missing_args = [arg for arg in required_args if not getattr(args, arg, None)]
    if missing_args:
        parser.error(
            f"The following required arguments are missing: {', '.join(missing_args)}"
        )

    # Create argument dataclasses
    model_args = ModelArguments(
        model_name_or_path=args.model_name_or_path,
        use_auth_token=args.use_auth_token,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=args.torch_dtype,
        attn_implementation=args.attn_implementation,
    )

    data_args = DataArguments(
        dataset_name_or_path=args.dataset_name_or_path,
        dataset_config_name=args.dataset_config_name,
        max_seq_length=args.max_seq_length,
        instruction_template=args.instruction_template,
        validation_split=args.validation_split,
        use_existing_splits=args.use_existing_splits,
        auto_detect_format=args.auto_detect_format,
        template_format=args.template_format,
        response_max_length=args.response_max_length,
        instruction_max_length=args.instruction_max_length,
    )

    quant_args = QuantizationArguments(
        use_4bit=args.use_4bit,
        use_8bit=args.use_8bit,
        bnb_4bit_compute_dtype=args.bnb_4bit_compute_dtype,
        bnb_4bit_quant_type=args.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=args.bnb_4bit_use_double_quant,
    )

    lora_args = LoRAArguments(
        use_peft=not args.no_peft,  # Invert the no_peft flag
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=args.lora_target_modules,
        lora_bias=args.lora_bias,
    )

    # Initialize Weights & Biases if requested
    if args.use_wandb:
        # Use custom run name if provided, otherwise use default format
        run_name = (
            args.wandb_run_name
            if args.wandb_run_name
            else f"sft-{Path(args.model_name_or_path).name}-{Path(args.dataset_name_or_path).name}"
        )
        wandb.init(
            project=args.wandb_project,
            config=vars(args),
            name=run_name,
        )

    # Determine device for data loader and enable TF32 on CUDA
    device_for_args, _ = get_optimal_device()
    if device_for_args.type == "cuda":
        try:
            # Enable TF32 on matmul and cuDNN for speed on Ampere+ GPUs
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            try:
                # Prefer high precision matmul policy when available (PyTorch 2.0+)
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass
            logger.info("Enabled TF32 (matmul/cudnn) for CUDA")
        except Exception as e:
            logger.debug(f"Could not enable TF32: {e}")
    pin_memory_flag = device_for_args.type == "cuda"

    # Determine optimal number of workers based on platform
    # Windows has issues with multiprocessing spawn and CUDA libraries
    import platform

    if platform.system() == "Windows":
        # Use fewer workers on Windows to avoid paging file issues with CUDA libraries
        num_workers = 2
        logger.info("Using 2 DataLoader workers on Windows to avoid paging file issues")
    else:
        # Use more workers on Unix-like systems
        num_workers = 8
        logger.info("Using 8 DataLoader workers on Unix-like systems")

    logger.info(
        f"DataLoader pin_memory: {pin_memory_flag} | num_workers: {num_workers} | TF32 matmul: "
        f"{getattr(torch.backends.cuda.matmul, 'allow_tf32', None) if device_for_args.type == 'cuda' else 'N/A'} | "
        f"TF32 cuDNN: {getattr(torch.backends.cudnn, 'allow_tf32', None) if device_for_args.type == 'cuda' else 'N/A'}"
    )

    # Select optimizer based on device and quantization support (avoid CUDA-only on MPS/CPU)
    optim_name = (
        "adamw_torch_fused" if device_for_args.type == "cuda" else "adamw_torch"
    )
    if args.use_4bit or args.use_8bit:
        if is_quantization_supported():
            optim_name = "paged_adamw_8bit"
        else:
            logger.warning(
                "4/8-bit optimization requested but quantization not supported; falling back to adamw_torch"
            )
            optim_name = "adamw_torch"

    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        eval_strategy="steps",
        save_strategy="steps",
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=args.load_best_model_at_end,
        metric_for_best_model=args.metric_for_best_model,
        greater_is_better=args.greater_is_better,
        report_to="wandb" if args.use_wandb else None,
        run_name=(
            args.wandb_run_name
            if args.wandb_run_name
            else f"sft-{Path(args.model_name_or_path).name}"
        ),
        remove_unused_columns=False,
        dataloader_pin_memory=pin_memory_flag,
        dataloader_num_workers=num_workers,
        dataloader_persistent_workers=True,
        dataloader_prefetch_factor=2,
        group_by_length=True,
        gradient_checkpointing=args.gradient_checkpointing,
        fp16=args.torch_dtype == "float16",
        bf16=args.torch_dtype == "bfloat16",
        optim=optim_name,
        max_grad_norm=args.max_grad_norm,
        label_names=["labels"],  # Quiet PEFT label warning
    )

    logger.info("Starting supervised fine-tuning...")
    logger.info(f"Model: {model_args.model_name_or_path}")
    logger.info(f"Dataset: {data_args.dataset_name_or_path}")
    logger.info(f"Output directory: {args.output_dir}")

    # Log training mode
    training_mode = (
        "PEFT (LoRA/QLoRA)" if lora_args.use_peft else "Full Parameter Fine-Tuning"
    )
    logger.info(f"Training Mode: {training_mode}")

    try:
        # Load quantization config
        quant_config = load_quantization_config_from_args(quant_args)

        # Load model and tokenizer
        model, tokenizer = load_model_and_tokenizer(model_args, quant_config)

        # Setup LoRA or prepare for full fine-tuning
        model = setup_lora_from_args(model, lora_args)

        # Log detailed training mode information
        log_training_mode_details(lora_args.use_peft, model)

        # Adjust training arguments based on training mode
        training_args = adjust_training_args_for_mode(training_args, lora_args.use_peft)

        # Load and prepare dataset with split awareness
        if data_args.use_existing_splits:
            logger.info("Loading dataset with split awareness enabled")
            splits_data = load_dataset_with_splits_from_args(data_args)

            # Apply preprocessing to all splits
            from .utils.model_utils import preprocess_dataset_for_experiments

            processed_splits = {}
            for split_name, split_data in splits_data.items():
                processed_data = preprocess_dataset_for_experiments(
                    split_data,
                    response_max_length=data_args.response_max_length,
                    instruction_max_length=data_args.instruction_max_length,
                )
                processed_splits[split_name] = processed_data
                logger.info(
                    f"Processed {split_name} split: {len(processed_data)} examples"
                )

            # Prepare train/validation splits using existing splits if available
            train_data, val_data = prepare_train_val_splits(
                processed_splits,
                data_args.validation_split,
                prefer_existing_splits=True,
            )
        else:
            logger.info("Loading dataset with legacy single-split mode")
            # Legacy behavior: load only train split and create custom validation split
            data = load_dataset_from_args(data_args)

            # Apply data preprocessing for experiments (filter by Context and Response lengths)
            from .utils.model_utils import preprocess_dataset_for_experiments

            data = preprocess_dataset_for_experiments(
                data,
                response_max_length=data_args.response_max_length,
                instruction_max_length=data_args.instruction_max_length,
            )

            train_data, val_data = split_dataset(data, data_args.validation_split)

        # Detect dataset format once to avoid duplicate logging
        detected_format = None
        if data_args.auto_detect_format and train_data:
            detected_format = DatasetFormatter.detect_format(train_data)
            logger.info(f"Detected dataset format: {detected_format}")

        # Pre-tokenization if enabled
        is_pretokenized = False
        if args.pretokenize:
            cache_dir = args.pretokenize_cache_dir or os.path.join(
                args.output_dir, "tokenized_cache"
            )

            # Pre-tokenize train data
            train_data = pretokenize_and_cache_dataset(
                data=train_data,
                tokenizer=tokenizer,
                data_args=data_args,
                detected_format=detected_format,
                cache_dir=cache_dir,
                split_name="train",
            )

            # Pre-tokenize validation data if it exists
            if val_data:
                val_data = pretokenize_and_cache_dataset(
                    data=val_data,
                    tokenizer=tokenizer,
                    data_args=data_args,
                    detected_format=detected_format,
                    cache_dir=cache_dir,
                    split_name="val",
                )

            is_pretokenized = True
            logger.info("Using pre-tokenized datasets")

        # Create datasets
        train_dataset = InstructionDataset(
            train_data,
            tokenizer,
            data_args.max_seq_length,
            data_args.instruction_template,
            data_args.auto_detect_format,
            is_pretokenized,
            data_args.template_format,
            detected_format=detected_format,
        )

        eval_dataset = None
        if val_data:
            eval_dataset = InstructionDataset(
                val_data,
                tokenizer,
                data_args.max_seq_length,
                data_args.instruction_template,
                data_args.auto_detect_format,
                is_pretokenized,
                data_args.template_format,
                detected_format=detected_format,
            )

        # Create data collator (dynamic padding + label masking on padding)
        data_collator = DataCollatorForCausalLMWithPadding(
            tokenizer=tokenizer, pad_to_multiple_of=8
        )

        mask_handles = []
        # Track union of selected channels per layer and applied mask mode for static estimates
        mask_union: Dict[str, List[int]] = {}
        applied_mask_mode: Optional[str] = None

        if args.mask_mode is not None:
            # Initialize ICA mask handler
            if ICAMask is None:
                raise ImportError(
                    "ICAMask is not available. Please install scikit-learn to enable mask features."
                )
            ica_mask = ICAMask(
                num_components=args.ica_components,
                percentile=args.ica_percentile,
                sample_batches=100,
                ica_dtype=args.ica_dtype,
                max_pca_components=1000,  # Default value for backward compatibility
            )

            # Parse layer specification if provided
            target_layers = None
            if args.ica_mask_layers is not None:
                # Determine total number of layers in the model
                total_layers = 0
                actual_model: Any = model
                if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
                    actual_model = model.base_model.model
                elif hasattr(model, "base_model"):
                    actual_model = model.base_model

                if hasattr(actual_model, "transformer"):
                    blocks = getattr(actual_model.transformer, "h", None) or getattr(
                        actual_model.transformer, "blocks", None
                    )
                elif hasattr(actual_model, "model"):
                    blocks = getattr(actual_model.model, "layers", None) or getattr(
                        actual_model.model, "decoder", None
                    )
                elif hasattr(actual_model, "layers"):
                    blocks = actual_model.layers
                else:
                    blocks = None

                if blocks is not None:
                    total_layers = len(blocks)
                    try:
                        target_layers = ica_mask.parse_layer_specification(
                            args.ica_mask_layers, total_layers
                        )
                        logger.info(
                            f"Parsed layer specification '{args.ica_mask_layers}' -> layers {target_layers}"
                        )
                    except ValueError as e:
                        logger.error(
                            f"Invalid layer specification '{args.ica_mask_layers}': {e}"
                        )
                        raise
                else:
                    logger.error(
                        "Could not determine model architecture for layer specification"
                    )
                    raise ValueError(
                        "Unable to parse layer specification: unknown model architecture"
                    )

            # ----------------- NEW: branch on ICA mode -----------------
            logger.info(
                "Using GLOBAL ICA mode (one ICA over concatenated final MLP outputs)."
            )

            # Load or compute component masks
            if args.ica_template_path:
                logger.info(
                    f"Loading global ICA templates from {args.ica_template_path}"
                )
                templates = ica_mask.load_templates(args.ica_template_path)
                # Materialize for immediate use
                ica_mask.mask_dict_components = templates["templates"]
                ica_mask.global_feature_layout = templates["layout"]
            else:
                # Use a small subset to estimate components, then build & save TEMPLATES
                sample_for_ica = torch.utils.data.Subset(
                    train_dataset, range(min(1024, len(train_dataset)))
                )
                component_masks = ica_mask.compute_global_networks(
                    model=model,
                    dataset=sample_for_ica,
                    tokenizer=tokenizer,
                    target_layers=target_layers,
                    n_components=args.ica_components,
                    top_percentile_per_component=args.ica_percentile,
                )
                if not component_masks:
                    logger.warning(
                        "Global ICA produced no component masks; proceeding without masking."
                    )
                else:
                    # Build & save templates (includes layout metadata for IoU later)
                    templates = ica_mask.build_templates_from_current_components(
                        name="groupwise_v1"
                    )
                    template_path = os.path.join(
                        args.output_dir, "global_templates.json"
                    )
                    ica_mask.save_templates(template_path, templates)
                    logger.info(f"Saved global ICA templates to {template_path}")

                    # Materialize for immediate use
                    ica_mask.mask_dict_components = templates["templates"]
                    ica_mask.global_feature_layout = templates["layout"]

            if getattr(ica_mask, "mask_dict_components", None):
                # Decide which components to target
                comp_ids = (
                    args.ica_component_ids
                    if args.ica_component_ids is not None
                    else [0]
                )
                existing_ids = sorted(ica_mask.mask_dict_components.keys())
                bad = [c for c in comp_ids if c not in existing_ids]
                if bad:
                    raise ValueError(
                        f"Requested component ids {bad} not in available {existing_ids}"
                    )

                # Map mask_mode to global mode with backward-compat for legacy values
                _mm = args.mask_mode
                if _mm in ("lesion", "key"):
                    global_mode = "lesion"
                else:
                    global_mode = "preserve"
                if _mm in ("key", "complement"):
                    logger.warning(
                        "Deprecated mask_mode '%s'. Use 'lesion'/'preserve' instead.",
                        _mm,
                    )

                # Log coverage summary
                logger.info(
                    "Global ICA Component Coverage Summary (per selected component):"
                )
                for cid in comp_ids:
                    layer_counts = {
                        lid: len(chs)
                        for lid, chs in ica_mask.mask_dict_components[cid].items()
                    }

                    logger.info(
                        f"  • comp {cid}: {sum(layer_counts.values())} channels across {len(layer_counts)} layers"
                    )

                # Build union of selected channels across components for static estimates
                from collections import defaultdict as _dd

                _union_sets = _dd(set)
                for cid in comp_ids:
                    comp = ica_mask.mask_dict_components.get(cid, {})
                    for layer, chans in comp.items():
                        for ch in chans:
                            _union_sets[layer].add(ch)
                mask_union = {k: sorted(v) for k, v in _union_sets.items()}
                applied_mask_mode = global_mode

                mask_handles = ica_mask.apply_component_masks(
                    model=model,
                    component_ids=comp_ids,
                    mode=global_mode,
                )
                logger.info(
                    f"Applied GLOBAL component masking: components={comp_ids} mode={global_mode}"
                )
            else:
                logger.warning("No component masks available - skipping masking.")

        # ------------------------------------------------------

        # Create trainer
        trainer = create_trainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            training_args=training_args,
            data_collator=data_collator,
        )

        # Resume from checkpoint if specified
        checkpoint = None
        if args.resume_from_checkpoint:
            checkpoint = args.resume_from_checkpoint
            logger.info(f"Resuming training from checkpoint: {checkpoint}")

        # Static estimate of masked trainable parameters (LoRA down_proj rows)
        def _estimate_masked_params_for_lora_down_proj(
            model, mask_union, mask_mode
        ) -> int:
            """Estimate count of trainable params that are effectively masked by output channel masking.
            - For PEFT: counts rows in lora_B of mlp.down_proj corresponding to masked output channels.
            - For full FT (no PEFT): counts rows in base down_proj weight if trainable.
            """
            total = 0
            try:
                # Unwrap PEFT to get base model for MLP discovery
                actual_model = model
                if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
                    actual_model = model.base_model.model
                elif hasattr(model, "base_model"):
                    actual_model = model.base_model

                # Use ICAMask helper to find MLPs if available
                mlps = None
                try:
                    blocks, mlps = ica_mask._find_decoder_blocks_and_mlps(actual_model)  # type: ignore[attr-defined]
                except Exception:
                    mlps = None

                hidden_size = (
                    getattr(model.config, "hidden_size", None)
                    or getattr(model.config, "n_embd", None)
                    or getattr(model.config, "d_model", None)
                )
                if hidden_size is None or not mlps:
                    return 0

                for i, mlp in enumerate(mlps):
                    if mlp is None:
                        continue
                    chans = (
                        mask_union.get(str(i), [])
                        if isinstance(mask_union, dict)
                        else []
                    )
                    masked_rows = (
                        (hidden_size - len(chans))
                        if mask_mode == "preserve"
                        else len(chans)
                    )
                    if masked_rows <= 0:
                        continue

                    dp = getattr(mlp, "down_proj", None)
                    if dp is None:
                        continue

                    # Prefer LoRA B matrices if present
                    lora_B = getattr(dp, "lora_B", None)
                    if lora_B:
                        for _, module in lora_B.items():
                            w = getattr(module, "weight", None)
                            if w is None or w.ndim != 2:
                                continue
                            dim0, dim1 = w.shape
                            out_features = getattr(dp, "out_features", hidden_size)
                            if dim0 == hidden_size or dim0 == out_features:
                                r = dim1
                                total += masked_rows * r
                            elif dim1 == hidden_size or dim1 == out_features:
                                r = dim0
                                total += masked_rows * r
                    else:
                        # If base layer is trainable, estimate masked rows on base weight
                        w = getattr(dp, "weight", None)
                        if (
                            w is not None
                            and getattr(w, "requires_grad", False)
                            and w.ndim == 2
                        ):
                            out_dim, in_dim = w.shape
                            if out_dim == hidden_size:
                                total += masked_rows * in_dim
                return int(total)
            except Exception:
                return 0

        # Log trainable parameters count
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Trainable parameters: {trainable_params:,}")

        # Also log a static estimate adjusted by masked output channels (LoRA down_proj rows)
        if applied_mask_mode in ("lesion", "preserve") and mask_union:
            masked_params_est = _estimate_masked_params_for_lora_down_proj(
                model, mask_union, applied_mask_mode
            )
            effective_trainable = max(0, trainable_params - masked_params_est)
            logger.info(
                f"Effective trainable parameters (static est.): {effective_trainable:,} "
                f"[= {trainable_params:,} - masked {masked_params_est:,}]"
            )

        # Start training
        logger.info("Starting training...")
        trainer.train(resume_from_checkpoint=checkpoint)

        # Remove hooks to save an unmasked model
        for h in mask_handles:
            h.remove()

        # Save final model
        final_output_dir = os.path.join(args.output_dir, "final_model")
        save_model_and_tokenizer(model, tokenizer, final_output_dir, lora_args.use_peft)

        # Merge adapter with base model if requested
        if args.merge_adapter_with_base and lora_args.use_peft:
            logger.info("Merging LoRA adapter with base model...")

            # Create separate directories for adapter and merged model
            adapter_output_dir = os.path.join(args.output_dir, "adapter")
            merged_output_dir = os.path.join(args.output_dir, "merged_model")

            # Save adapter separately (copy from final_model to adapter directory)
            import shutil

            if os.path.exists(adapter_output_dir):
                shutil.rmtree(adapter_output_dir)
            shutil.copytree(final_output_dir, adapter_output_dir)
            logger.info(f"Adapter saved to: {adapter_output_dir}")

            # Import merge function
            from .utils.model_utils import merge_adapter_with_base_model

            # Merge adapter with base model
            try:
                merge_adapter_with_base_model(
                    adapter_path=final_output_dir,
                    output_path=merged_output_dir,
                    base_model_name=model_args.model_name_or_path,
                )
                logger.info(f"Merged model saved to: {merged_output_dir}")

                # Also save tokenizer to merged model directory
                if hasattr(tokenizer, "save_pretrained"):
                    tokenizer.save_pretrained(merged_output_dir)
                    logger.info("Tokenizer saved to merged model directory")

            except Exception as e:
                logger.error(f"Failed to merge adapter with base model: {e}")
                logger.warning("Continuing with training completion...")
        elif args.merge_adapter_with_base and not lora_args.use_peft:
            logger.warning(
                "--merge_adapter_with_base was specified but PEFT is not enabled. Skipping merge operation."
            )

        # Convert to GGUF if requested
        if args.convert_to_gguf:
            gguf_output_path = os.path.join(args.output_dir, "model.gguf")
            # Prefer merged model for GGUF conversion when available
            model_dir_for_gguf = final_output_dir
            merged_dir_candidate = os.path.join(args.output_dir, "merged_model")
            if os.path.exists(merged_dir_candidate):
                logger.info("Using merged model directory for GGUF conversion")
                model_dir_for_gguf = merged_dir_candidate
            else:
                logger.info(
                    "Merged model directory not found; using final_model for GGUF conversion"
                )
            convert_to_gguf(
                model_dir_for_gguf, gguf_output_path, args.gguf_quantization
            )

        # Upload to Hugging Face Hub if requested
        if args.push_to_hub:
            if not args.hub_repo_id:
                logger.error("--hub_repo_id is required when using --push_to_hub")
                raise ValueError("hub_repo_id must be specified for Hub upload")

            # Determine which model to upload
            upload_model_path = final_output_dir
            upload_use_peft = lora_args.use_peft

            if (
                args.upload_merged_model
                and args.merge_adapter_with_base
                and lora_args.use_peft
            ):
                # Upload the merged model instead of the adapter
                merged_output_dir = os.path.join(args.output_dir, "merged_model")
                if os.path.exists(merged_output_dir):
                    upload_model_path = merged_output_dir
                    upload_use_peft = False  # Merged model is not a PEFT model
                    logger.info(f"Uploading merged model from: {merged_output_dir}")
                else:
                    logger.warning(
                        "Merged model directory not found, falling back to adapter upload"
                    )
            elif args.upload_merged_model and not args.merge_adapter_with_base:
                logger.warning(
                    "--upload_merged_model specified but --merge_adapter_with_base not enabled. Uploading adapter instead."
                )
            elif args.upload_merged_model and not lora_args.use_peft:
                logger.warning(
                    "--upload_merged_model specified but PEFT not enabled. Uploading full model instead."
                )

            try:
                upload_to_hub(
                    model_path=upload_model_path,
                    tokenizer=tokenizer,
                    repo_id=args.hub_repo_id,
                    commit_message=args.hub_commit_message,
                    private=args.hub_private,
                    token=args.hub_token,
                    use_peft=upload_use_peft,
                )
            except Exception as e:
                logger.error(f"Failed to upload to Hub: {e}")

        # Log final metrics
        if trainer.state.log_history:
            final_metrics = trainer.state.log_history[-1]
            logger.info("Training completed successfully!")
            logger.info(f"Final metrics: {final_metrics}")

        if args.use_wandb:
            wandb.finish()

    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        if args.use_wandb:
            wandb.finish()
        raise


if __name__ == "__main__":
    main()
