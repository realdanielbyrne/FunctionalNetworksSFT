"""
Shared model utilities for quantization, LoRA setup, and model management.
"""

import json
import logging
import os
import platform
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers.utils.quantization_config import BitsAndBytesConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)

logger = logging.getLogger(__name__)


def get_optimal_device() -> Tuple[torch.device, str]:
    """
    Detect and return the optimal device for training/inference.

    Returns:
        Tuple of (device, device_name) where device_name is human-readable
    """
    if torch.cuda.is_available():
        device = torch.device("cuda:0")  # Specify index for consistency
        device_name = f"CUDA ({torch.cuda.get_device_name()})"

        # Log additional CUDA information
        cuda_version = torch.version.cuda
        device_count = torch.cuda.device_count()
        memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)

        logger.info(f"Using CUDA device: {device_name}")
        logger.info(f"CUDA version: {cuda_version}, Device count: {device_count}")
        logger.info(f"GPU memory: {memory_gb:.1f} GB")

        # Test basic CUDA operations to ensure compatibility
        try:
            test_tensor = torch.randn(10, 10, device=device)
            _ = torch.matmul(test_tensor, test_tensor)
            logger.info("CUDA operations verified successfully")
        except Exception as e:
            logger.warning(f"CUDA operation test failed: {e}")
            logger.warning("Falling back to CPU")
            device = torch.device("cpu")
            device_name = f"CPU ({platform.processor()})"

        return device, device_name
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        device_name = "Apple Silicon MPS"
        logger.info(f"Using MPS device: {device_name}")
        return device, device_name
    else:
        device = torch.device("cpu")
        device_name = f"CPU ({platform.processor()})"
        logger.info(f"Using CPU device: {device_name}")
        return device, device_name


def is_quantization_supported() -> bool:
    """
    Check if quantization (BitsAndBytes) is supported on current platform.

    Returns:
        True if quantization is supported, False otherwise
    """
    try:
        import bitsandbytes

        # BitsAndBytes requires CUDA
        if torch.cuda.is_available():
            logger.info("Quantization (BitsAndBytes) is supported")
            return True
        else:
            logger.warning(
                "Quantization (BitsAndBytes) requires CUDA - not available on this platform"
            )
            return False
    except ImportError:
        logger.warning("BitsAndBytes not installed - quantization not available")
        return False


def get_recommended_dtype() -> torch.dtype:
    """
    Get the recommended torch dtype based on the current device.

    Returns:
        Recommended torch dtype
    """
    device, _ = get_optimal_device()

    if device.type == "cuda":
        # CUDA supports bfloat16 on modern GPUs
        if torch.cuda.is_bf16_supported():
            logger.info("Using bfloat16 for CUDA")
            return torch.bfloat16
        else:
            logger.info("Using float16 for CUDA")
            return torch.float16
    elif device.type == "mps":
        # MPS works well with float16
        logger.info("Using float16 for MPS")
        return torch.float16
    else:
        # CPU typically uses float32
        logger.info("Using float32 for CPU")
        return torch.float32


def load_quantization_config(
    use_4bit: bool = True,
    use_8bit: bool = False,
    bnb_4bit_compute_dtype: str = "float16",
    bnb_4bit_quant_type: str = "nf4",
    bnb_4bit_use_double_quant: bool = True,
) -> Optional[BitsAndBytesConfig]:
    """
    Load quantization configuration with cross-platform support.

    Automatically disables quantization on platforms that don't support it (e.g., Apple Silicon).
    """
    if not (use_4bit or use_8bit):
        return None

    # Check if quantization is supported on current platform
    if not is_quantization_supported():
        logger.warning(
            "Quantization requested but not supported on this platform - disabling"
        )
        return None

    if use_8bit:
        logger.info("Using 8-bit quantization")
        return BitsAndBytesConfig(load_in_8bit=True)

    logger.info("Using 4-bit quantization")
    compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
    )


def setup_lora(
    model: Any,
    use_peft: bool = True,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.1,
    lora_target_modules: Optional[List[str]] = None,
    lora_bias: str = "none",
) -> Any:
    """Setup LoRA configuration for the model."""
    if not use_peft:
        logger.info("PEFT disabled - using full parameter fine-tuning")
        # For full fine-tuning, we still need to prepare quantized models if they are quantized
        is_quantized = (
            hasattr(model, "config")
            and hasattr(model.config, "quantization_config")
            and model.config.quantization_config is not None
        )

        if is_quantized:
            logger.info(
                "Model is quantized, preparing for k-bit training (full fine-tuning mode)"
            )
            model = prepare_model_for_kbit_training(model)
        else:
            logger.info("Model is not quantized, ready for full fine-tuning")

        # Enable gradient computation for all parameters
        for param in model.parameters():
            param.requires_grad = True

        # Log trainable parameters for full fine-tuning
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(
            f"Full fine-tuning mode: {trainable_params:,} trainable parameters out of {total_params:,} total parameters"
        )
        logger.info(
            f"Trainable parameters: {100 * trainable_params / total_params:.2f}%"
        )

        return model

    logger.info("Setting up LoRA configuration")

    # Only prepare model for k-bit training if it's actually quantized
    # Check if model has quantization config
    is_quantized = (
        hasattr(model, "config")
        and hasattr(model.config, "quantization_config")
        and model.config.quantization_config is not None
    )

    if is_quantized:
        logger.info("Model is quantized, preparing for k-bit training")
        model = prepare_model_for_kbit_training(model)
    else:
        logger.info("Model is not quantized, skipping k-bit training preparation")

    # Auto-detect target modules if not specified
    target_modules = lora_target_modules
    if target_modules is None:
        # Common target modules for different architectures
        if hasattr(model, "config") and hasattr(model.config, "model_type"):
            model_type = model.config.model_type.lower()
            if "llama" in model_type or "mistral" in model_type:
                target_modules = [
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ]
            elif "gpt" in model_type:
                target_modules = ["c_attn", "c_proj", "c_fc"]
            else:
                # Fallback: find all linear layers
                target_modules = []
                if hasattr(model, "named_modules"):
                    for name, module in model.named_modules():
                        if isinstance(module, nn.Linear):
                            target_modules.append(name.split(".")[-1])
                target_modules = list(set(target_modules))

        logger.info(f"Auto-detected target modules: {target_modules}")

    # Configure LoRA
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias=lora_bias,  # type: ignore[arg-type]
        task_type=TaskType.CAUSAL_LM,
    )

    # Apply LoRA to model
    model = get_peft_model(model, lora_config)
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()

    return model


def load_dataset_from_path(
    dataset_name_or_path: str,
    dataset_config_name: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Load dataset from local file or HuggingFace hub."""
    if os.path.isfile(dataset_name_or_path):
        logger.info(f"Loading dataset from local file: {dataset_name_or_path}")
        # Load from local file using HuggingFace datasets
        if dataset_name_or_path.endswith(".json"):
            dataset = load_dataset(
                "json", data_files=dataset_name_or_path, split="train"
            )
        elif dataset_name_or_path.endswith(".jsonl"):
            dataset = load_dataset(
                "json", data_files=dataset_name_or_path, split="train"
            )
        elif dataset_name_or_path.endswith(".csv"):
            dataset = load_dataset(
                "csv", data_files=dataset_name_or_path, split="train"
            )
        else:
            raise ValueError(f"Unsupported file format: {dataset_name_or_path}")
        data = [item for item in dataset]
    else:
        logger.info(f"Loading dataset from HuggingFace hub: {dataset_name_or_path}")
        dataset = load_dataset(dataset_name_or_path, dataset_config_name, split="train")
        data = [item for item in dataset]

    logger.info(f"Loaded {len(data)} examples")
    return data  # type: ignore


def split_dataset(
    data: List[Dict[str, Any]], validation_split: float = 0.1
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split dataset into train and validation sets."""
    if validation_split <= 0:
        return data, []

    split_idx = int(len(data) * (1 - validation_split))
    train_data = data[:split_idx]
    val_data = data[split_idx:]

    logger.info(f"Split dataset: {len(train_data)} train, {len(val_data)} validation")
    return train_data, val_data


def save_model_and_tokenizer(
    model: Any, tokenizer: Any, output_dir: str, use_peft: Optional[bool] = None
) -> None:
    """Save the fine-tuned model and tokenizer."""
    logger.info(f"Saving model to {output_dir}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Detect if this is a PEFT model if not explicitly specified
    if use_peft is None:
        use_peft = hasattr(model, "peft_config") and model.peft_config is not None

    if use_peft:
        logger.info("Saving PEFT model (adapters)")
        # For PEFT models, save the adapters
        if hasattr(model, "save_pretrained"):
            model.save_pretrained(output_dir)
        else:
            logger.warning("Model does not have save_pretrained method")
    else:
        logger.info("Saving full fine-tuned model")
        # For full fine-tuning, save the entire model
        if hasattr(model, "save_pretrained"):
            model.save_pretrained(output_dir)
        else:
            logger.warning("Model does not have save_pretrained method")

    # Save tokenizer (same for both modes)
    if hasattr(tokenizer, "save_pretrained"):
        tokenizer.save_pretrained(output_dir)
    else:
        logger.warning("Tokenizer does not have save_pretrained method")

    # Log what was saved
    saved_files = os.listdir(output_dir)
    logger.info(f"Saved files: {saved_files}")

    if use_peft:
        if "adapter_config.json" in saved_files:
            logger.info("✓ PEFT adapter configuration saved")
        if any(f.startswith("adapter_model") for f in saved_files):
            logger.info("✓ PEFT adapter weights saved")
    else:
        if "config.json" in saved_files:
            logger.info("✓ Model configuration saved")
        if any(
            f.startswith("pytorch_model") or f.endswith(".safetensors")
            for f in saved_files
        ):
            logger.info("✓ Full model weights saved")

    logger.info("Model and tokenizer saved successfully")
