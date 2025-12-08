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

# Import whoami at module level for testing
try:
    from huggingface_hub import whoami
except ImportError:
    whoami = None

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)


# ======================== Logging ========================


def setup_logging(log_file: Optional[str] = "sft_training.log") -> logging.Logger:
    """Set up root logging (idempotent)."""
    root = logging.getLogger()
    if not root.handlers:
        level = os.getenv("SFT_LOG_LEVEL", "INFO").upper()
        handlers = [logging.StreamHandler(sys.stdout)]
        if log_file:
            handlers.append(logging.FileHandler(log_file))
        logging.basicConfig(
            level=getattr(logging, level, logging.INFO),
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=handlers,
        )
    return logging.getLogger(__name__)


logger = setup_logging()


# Log-once guard for template format messages
_TEMPLATE_DECISION_LOGGED = False


def load_env_file() -> None:
    """Load env vars from .env if present."""
    env_path = Path(".env")
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ[k] = v.strip().strip('"').strip("'")


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier"}
    )
    use_auth_token: bool = field(
        default=True, metadata={"help": "Use HF auth token for private models"}
    )
    trust_remote_code: bool = field(
        default=True, metadata={"help": "Trust remote code"}
    )
    torch_dtype: str = field(
        default="auto",
        metadata={"help": "auto, float16, bfloat16, float32"},
    )
    attn_implementation: str = field(
        default="auto",
        metadata={"help": "auto, eager, sdpa, flash_attention_2"},
    )
    attn_implementation: str = field(
        default="auto",
        metadata={
            "help": "Attention kernel implementation to request (auto, eager, sdpa, flash_attention_2)."
        },
    )


@dataclass
class DataArguments:
    dataset_name_or_path: str = field(
        metadata={"help": "Path to dataset file or HF dataset name"}
    )
    dataset_config_name: Optional[str] = field(default=None)
    max_seq_length: int = field(default=2048)
    instruction_template: str = field(
        default="### Instruction:\n{instruction}\n\n### Response:\n{response}"
    )
    validation_split: float = field(default=0.1)
    use_existing_splits: bool = field(default=True)
    auto_detect_format: bool = field(default=True)
    template_format: str = field(default="auto")  # auto, chat, alpaca, chatml, basic
    response_max_length: int = field(default=4000)
    instruction_max_length: int = field(default=2048)


@dataclass
class QuantizationArguments:
    use_4bit: bool = field(default=False)
    use_8bit: bool = field(default=True)
    bnb_4bit_compute_dtype: str = field(default="float16")
    bnb_4bit_quant_type: str = field(default="nf4")  # nf4, fp4
    bnb_4bit_use_double_quant: bool = field(default=True)


@dataclass
class LoRAArguments:
    use_peft: bool = field(default=True)
    lora_r: int = field(default=16)
    lora_alpha: int = field(default=32)
    lora_dropout: float = field(default=0.1)
    lora_target_modules: Optional[List[str]] = field(
        default_factory=lambda: ["down_proj", "dense_4h_to_h"]
    )
    lora_bias: Literal["none", "all", "lora_only"] = field(default="none")


# ======================== Utilities from local modules ========================
from .utils.dataset_utils import DatasetFormatter, InstructionDataset
from .utils.model_utils import (
    convert_to_gguf,
    get_optimal_device,
    get_recommended_dtype,
    is_quantization_supported,
    load_dataset_from_path,
    load_dataset_with_splits,
    load_quantization_config,
    merge_adapter_with_base_model,
    prepare_train_val_splits,
    preprocess_dataset_for_experiments,
    save_model_and_tokenizer,
    setup_lora,
    split_dataset,
)

from .utils.hf_utilities import upload_to_hub

# Lazy import ICAMask to avoid heavy dependencies during import (e.g., in tests)
try:
    from .ica_mask import ICAMask  # type: ignore
except Exception:  # pragma: no cover - only exercised when sklearn is unavailable
    ICAMask = None  # type: ignore


class DataCollatorForCausalLMWithPadding:
    """Dynamic padding + set padded labels to -100."""

    def __init__(
        self, tokenizer: PreTrainedTokenizerBase, pad_to_multiple_of: Optional[int] = 8
    ):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        feats = [
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
            for f in features
        ]
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
        # Convert to dict and ensure all values are tensors
        batch: Dict[str, torch.Tensor] = {}
        for key, value in be.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value
            else:
                batch[key] = torch.tensor(value)

        # Create labels from input_ids
        labels = batch["input_ids"].clone()
        if getattr(self.tokenizer, "pad_token_id", None) is not None:
            labels[batch["input_ids"] == self.tokenizer.pad_token_id] = -100
        else:
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
    if template_format == "auto":
        actual = "chat" if getattr(tokenizer, "chat_template", None) else "basic"
    else:
        actual = template_format

    def _format(instruction: str, response: str) -> str:
        if actual == "chat":
            messages = [
                {"role": "user", "content": instruction},
                {"role": "assistant", "content": response},
            ]
            out = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            return out if isinstance(out, str) else str(out)
        if actual == "alpaca":
            return f"### Instruction:\n{instruction}\n\n### Response:\n{response}"
        if actual == "chatml":
            return f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>"
        if actual == "basic":
            return instruction_template.format(
                instruction=instruction, response=response
            )
        raise ValueError(f"Unknown format: {actual}")

    def pretokenize_item(item):
        converted = item
        if auto_detect_format and detected_format:
            try:
                converted = DatasetFormatter.convert_to_standard_format(
                    item, detected_format
                )
            except Exception:
                try:
                    it_fmt = DatasetFormatter.detect_format([item])
                    converted = DatasetFormatter.convert_to_standard_format(
                        item, it_fmt
                    )
                except Exception:
                    logger.debug("Per-item detection failed; keeping original.")
                    converted = item

        if "instruction" in converted and "response" in converted:
            text = _format(converted["instruction"], converted["response"])
        elif "text" in converted:
            text = converted["text"]
        elif "instruction" in item and "response" in item:
            text = _format(item["instruction"], item["response"])
        elif "text" in item:
            text = item["text"]
        else:
            raise ValueError("Item must contain 'instruction'+'response' or 'text'.")

        enc = tokenizer(
            text,
            truncation=True,
            padding=False,
            max_length=max_length,
            return_tensors=None,
        )
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": enc["input_ids"],
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

    if os.path.exists(cache_path):
        try:
            logger.info(
                f"Loading pre-tokenized {split_name} dataset from cache: {cache_path}"
            )
            cached_dataset = load_from_disk(cache_path)
            return [item for item in cached_dataset]
        except Exception as e:
            logger.warning(f"Failed to load cached dataset: {e}. Re-tokenizing...")

    logger.info(f"Pre-tokenizing {split_name} dataset ({len(data)} examples)...")
    hf_dataset = HFDataset.from_list(data)

    pretokenize_fn = create_pretokenization_function(
        tokenizer=tokenizer,
        max_length=data_args.max_seq_length,
        instruction_template=data_args.instruction_template,
        template_format=data_args.template_format,
        detected_format=detected_format,
        auto_detect_format=data_args.auto_detect_format,
    )

    tokenized_dataset = hf_dataset.map(
        pretokenize_fn,
        remove_columns=hf_dataset.column_names,
        num_proc=min(4, os.cpu_count() or 1),
        desc=f"Pre-tokenizing {split_name}",
    )

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


def _parse_dtype(dtype_str: str) -> torch.dtype:
    if dtype_str == "auto":
        return get_recommended_dtype()
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }.get(dtype_str, torch.float16)


def load_model_and_tokenizer(
    model_args: ModelArguments, quant_config: Optional[BitsAndBytesConfig]
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    logger.info(f"Loading model: {model_args.model_name_or_path}")

    device, device_name = get_optimal_device()
    logger.info(f"Target device: {device_name}")

    torch_dtype = _parse_dtype(model_args.torch_dtype)
    if model_args.torch_dtype == "auto":
        logger.info(f"Auto-selected dtype: {torch_dtype}")

    # Auth token resolution
    auth_token: Optional[Union[str, bool]] = None
    if model_args.use_auth_token:
        auth_token = os.getenv("HF_TOKEN")
        if auth_token:
            logger.info("Using HF_TOKEN from environment")
        else:
            try:
                if whoami is not None:
                    _ = whoami()  # uses cached creds if available
                    auth_token = True
                    logger.info("Using cached HuggingFace credentials")
            except Exception:
                logger.warning("No HF token found; proceeding without auth")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
        token=auth_token,
    )
    if getattr(tokenizer, "pad_token", None) is None and getattr(
        tokenizer, "eos_token", None
    ):
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = getattr(tokenizer, "eos_token_id", None)

    # Device map probe
    device_map = None
    if device.type in {"cuda", "mps"}:
        try:
            test = torch.randn(10, 10).to(device)
            _ = test @ test
            device_map = "auto"
            logger.info("GPU ops verified - using automatic device mapping")
        except RuntimeError as e:
            logger.warning(f"GPU probe failed: {e} - falling back to manual placement")

    model_kwargs: Dict[str, Any] = {
        "quantization_config": quant_config,
        "torch_dtype": torch_dtype,
        "trust_remote_code": model_args.trust_remote_code,
        "token": auth_token,
        "device_map": device_map,
    }
    if model_args.attn_implementation != "auto":
        model_kwargs["attn_implementation"] = model_args.attn_implementation
        logger.info(f"Attention implementation: {model_args.attn_implementation}")

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path, **model_kwargs
    )

    if device_map is None:
        try:
            model = model.to(device)
            logger.info(f"Model moved to {device}")
        except RuntimeError as e:
            if device.type == "cuda":
                logger.warning(f"Failed to move model to CUDA: {e} -> CPU fallback")
                model = model.to(torch.device("cpu"))
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

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    return trainer


def load_config_from_yaml(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


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

    # --- Anti-drift controls ---
    parser.add_argument(
        "--anti_drift_row_param",
        action="store_true",
        default=True,
        help="Enable row-wise anti-drift parametrization (frozen + mask*delta). "
        "If disabled, optimizer-induced drift on masked rows is not compensated.",
    )
    parser.add_argument(
        "--anti_drift_apply_to",
        type=str,
        default="auto",
        choices=["auto", "lora", "base", "both"],
        help="Where to apply row parametrization. 'auto' -> 'lora' when PEFT is enabled, "
        "'base' otherwise. 'both' forces both (use only when full-FT + LoRA).",
    )
    parser.add_argument(
        "--anti_drift_unwrap_on_save",
        action="store_true",
        default=True,
        help="If True, bake parametrizations into weights before saving/merging "
        "(remove_parametrizations(..., leave_parametrized=True)).",
    )

    args = parser.parse_args()

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

    is_peft = lora_args.use_peft
    is_quant = quant_args.use_4bit or quant_args.use_8bit
    is_qlora = is_peft and is_quant  # training via LoRA on quantized base

    # Resolve target per user flag
    apply_to = args.anti_drift_apply_to
    if apply_to == "auto":
        apply_to = "lora" if is_peft else "base"

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
    else:
        # Use more workers on Unix-like systems
        num_workers = 8

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

    logger.info("Starting training...")
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

        # Adjust training arguments based on training mode
        training_args = adjust_training_args_for_mode(training_args, lora_args.use_peft)

        # Load and prepare dataset with split awareness
        if data_args.use_existing_splits:
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
            logger.info("Loading dataset with single-split mode")
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

        detected_format = None
        if data_args.auto_detect_format and train_data:
            detected_format = DatasetFormatter.detect_format(train_data)
            logger.info(f"Detected dataset format: {detected_format}")

        is_pretokenized = False
        if args.pretokenize:
            cache_dir = args.pretokenize_cache_dir or os.path.join(
                args.output_dir, "tokenized_cache"
            )

            train_data = pretokenize_and_cache_dataset(
                data=train_data,
                tokenizer=tokenizer,
                data_args=data_args,
                detected_format=detected_format,
                cache_dir=cache_dir,
                split_name="train",
            )

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

        data_collator = DataCollatorForCausalLMWithPadding(
            tokenizer=tokenizer, pad_to_multiple_of=8
        )

        mask_handles = []
        param_handles = []
        # Track union of selected channels per layer and applied mask mode for static estimates
        mask_union: Dict[str, List[int]] = {}
        applied_mask_mode: Optional[str] = None
        ica_mask: Optional[Any] = None  # Initialize ica_mask to None
        has_row_parametrizations = False  # Track if we applied row parametrizations

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

            # Load or compute component masks
            if args.ica_template_path:
                logger.info(
                    f"Loading global ICA templates from {args.ica_template_path}"
                )
                templates = ica_mask.load_templates(args.ica_template_path)
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
                    comp = (
                        ica_mask.mask_dict_components.get(cid, {})
                        if ica_mask.mask_dict_components
                        else {}
                    )
                    layer_counts = {lid: len(chs) for lid, chs in comp.items()}

                    logger.info(
                        f"  • comp {cid}: {sum(layer_counts.values())} channels across {len(layer_counts)} layers"
                    )

                # Build union of selected channels across components for static estimates
                from collections import defaultdict as _dd

                _union_sets = _dd(set)
                for cid in comp_ids:
                    comp = (
                        ica_mask.mask_dict_components.get(cid, {})
                        if ica_mask.mask_dict_components
                        else {}
                    )
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

                if args.mask_mode is not None and args.anti_drift_row_param:
                    # comp_ids already resolved; global_mode is 'lesion' or 'preserve'
                    ph = ica_mask.apply_row_parametrizations(
                        model=model,
                        component_ids=comp_ids,
                        mode=global_mode,  # 'lesion' or 'preserve'
                        target_layers=target_layers,  # may be None -> all
                        apply_to=apply_to,  # 'lora' | 'base' | 'both'
                        logger=logger,
                    )
                    # Note: ph contains tuples (module, tensor_name), not handles with .remove()
                    # These will be cleaned up via ica_mask.remove_row_parametrizations()
                    has_row_parametrizations = True

                logger.info(f"Applied GLOBAL component masking: components={comp_ids}.")

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

        if args.anti_drift_row_param and ica_mask is not None:
            trainer.add_callback(NoDriftCheckCallback(model, ica_mask))

        # Resume from checkpoint if specified
        checkpoint = None
        if args.resume_from_checkpoint:
            checkpoint = args.resume_from_checkpoint
            logger.info(f"Resuming training from checkpoint: {checkpoint}")

        # Static estimate of masked trainable parameters (LoRA down_proj rows)

        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Trainable parameters: {trainable_params:,}")

        if (
            applied_mask_mode in ("lesion", "preserve")
            and mask_union
            and ica_mask is not None
        ):
            masked_params_est = ica_mask.estimate_masked_params_for_lora_down_proj(
                model, mask_union, applied_mask_mode
            )
            effective_trainable = max(0, trainable_params - masked_params_est)
            logger.info(
                f"Effective trainable parameters (static est.): {effective_trainable:,} "
                f"[= {trainable_params:,} - masked {masked_params_est:,}]"
            )

        logger.info("Starting training...")
        trainer.train(resume_from_checkpoint=checkpoint)

        # Clean up forward hooks
        for h in mask_handles:
            h.remove()

        # Clean up any remaining forward hooks in param_handles (if any)
        # Note: param_handles may contain tuples from row parametrizations, not hooks
        for h in param_handles:
            if hasattr(h, "remove"):
                h.remove()

        # Clean up row parametrizations if they were applied
        if (
            has_row_parametrizations
            and ica_mask is not None
            and hasattr(ica_mask, "remove_row_parametrizations")
        ):
            ica_mask.remove_row_parametrizations(
                bake=args.anti_drift_unwrap_on_save, logger=logger
            )
        elif (
            args.anti_drift_unwrap_on_save
            and ica_mask is not None
            and hasattr(ica_mask, "remove_row_parametrizations")
        ):
            # Fallback for backward compatibility
            ica_mask.remove_row_parametrizations(bake=True, logger=logger)

        final_output_dir = os.path.join(args.output_dir, "final_model")
        save_model_and_tokenizer(model, tokenizer, final_output_dir, lora_args.use_peft)

        if args.merge_adapter_with_base and lora_args.use_peft:
            logger.info("Merging LoRA adapter with base model...")

            adapter_output_dir = os.path.join(args.output_dir, "adapter")
            merged_output_dir = os.path.join(args.output_dir, "merged_model")

            import shutil

            if os.path.exists(adapter_output_dir):
                shutil.rmtree(adapter_output_dir)
            shutil.copytree(final_output_dir, adapter_output_dir)
            logger.info(f"Adapter saved to: {adapter_output_dir}")

            from .utils.model_utils import merge_adapter_with_base_model

            try:
                merge_adapter_with_base_model(
                    adapter_path=final_output_dir,
                    output_path=merged_output_dir,
                    base_model_name=model_args.model_name_or_path,
                )
                logger.info(f"Merged model saved to: {merged_output_dir}")

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

        if args.push_to_hub:
            if not args.hub_repo_id:
                logger.error("--hub_repo_id is required when using --push_to_hub")
                raise ValueError("hub_repo_id must be specified for Hub upload")

            upload_model_path = final_output_dir
            upload_use_peft = lora_args.use_peft

            if (
                args.upload_merged_model
                and args.merge_adapter_with_base
                and lora_args.use_peft
            ):
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


from transformers.trainer_callback import TrainerCallback


class NoDriftCheckCallback(TrainerCallback):
    ##
    # A tiny callback that asserts the masked rows didn’t change on the first optimizer step.
    ##
    def __init__(self, model, ica_mask, sample_limit=1):
        self.model = model
        self.ica_mask = ica_mask
        self.snap = {}
        self.done = False
        self.sample_limit = sample_limit

    def on_step_begin(self, args, state, control, **kwargs):
        if self.done or not hasattr(self.ica_mask, "mask_dict_components"):
            return
        # Snapshot masked rows once
        count = 0
        if hasattr(self.ica_mask, "_row_parametrizations"):
            for mod, tname in self.ica_mask._row_parametrizations:
                if count >= self.sample_limit:
                    break
                with torch.no_grad():
                    self.snap[(id(mod), tname)] = getattr(mod, tname).detach().clone()
                count += 1

    def on_step_end(self, args, state, control, **kwargs):
        if self.done or not self.snap:
            return
        # Compare post-step
        for (mid, tname), prev in self.snap.items():
            mod = None
            # Find module by id (inefficient but tiny set)
            for m, tn in self.ica_mask._row_parametrizations:
                if id(m) == mid and tn == tname:
                    mod = m
                    break
            if mod is None:
                continue
            now = getattr(mod, tname).detach()
            # Because we parametrized all rows, checking exact equality is fine here
            if not torch.allclose(now, prev):
                print(
                    "[WARN] Anti-drift check detected a change in a parametrized tensor. "
                    "Ensure parametrization was applied to all intended rows."
                )
                break
        self.done = True


if __name__ == "__main__":
    main()
