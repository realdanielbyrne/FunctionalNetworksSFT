"""
Shared model utilities for quantization, LoRA setup, and model management.
"""

import json
import logging
import os
import platform
import subprocess
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers.utils.quantization_config import BitsAndBytesConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import AutoPeftModelForCausalLM
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


def preprocess_dataset_for_experiments(
    data: List[Dict[str, Any]],
    response_max_length: int = 4000,
    instruction_max_length: int = 2048,
) -> List[Dict[str, Any]]:
    """
    Preprocess dataset by filtering based on:
      1) Response length (response_max_length)
      2) Combined instruction length after format conversion (instruction_max_length)

    Args:
        data: List of dataset items
        response_max_length: Maximum allowed length for Response field (default: 4000)
        instruction_max_length: Maximum allowed length for combined instruction (default: 2048)

    Returns:
        Filtered dataset with items that meet the length criteria
    """
    from .dataset_utils import DatasetFormatter

    original_count = len(data)
    logger.info(f"Starting dataset preprocessing with {original_count} examples")

    # Detect dataset format for proper instruction length calculation
    detected_format = None
    if data:
        try:
            detected_format = DatasetFormatter.detect_format(data)
            logger.info(f"Detected format for preprocessing: {detected_format}")
        except Exception as e:
            logger.warning(f"Could not detect format: {e}. Using basic filtering.")

    # 1) Filter by Response length first
    response_filtered_data = []
    response_filtered_count = 0

    for item in data:
        response = item.get("Response", item.get("response", ""))
        if len(str(response)) <= response_max_length:
            response_filtered_data.append(item)
        else:
            response_filtered_count += 1

    logger.info(
        f"Filtered out {response_filtered_count} examples with Response length > {response_max_length} characters"
    )

    # Filter by combined instruction length (after format conversion)
    final_filtered_data = []
    instruction_filtered_count = 0

    for item in response_filtered_data:
        # Calculate the combined instruction length after format conversion
        combined_instruction_length = 0

        if detected_format:
            try:
                # Convert to standard format to get the actual combined instruction
                converted_item = DatasetFormatter.convert_to_standard_format(
                    item, detected_format
                )
                combined_instruction_length = len(
                    str(converted_item.get("instruction", ""))
                )
            except Exception as e:
                # Fallback: estimate combined length manually
                logger.debug(f"Format conversion failed for item, using fallback: {e}")
                instruction = str(item.get("instruction", item.get("Instruction", "")))
                context = str(item.get("context", item.get("Context", "")))

                if context.strip():
                    # Add the " Context for reference: " overhead (25 characters)
                    combined_instruction_length = len(instruction) + len(context) + 25
                else:
                    combined_instruction_length = len(instruction)
        else:
            # No format detected, use basic estimation
            instruction = str(item.get("instruction", item.get("Instruction", "")))
            context = str(item.get("context", item.get("Context", "")))

            if context.strip():
                combined_instruction_length = len(instruction) + len(context) + 25
            else:
                combined_instruction_length = len(instruction)

        if combined_instruction_length <= instruction_max_length:
            final_filtered_data.append(item)
        else:
            instruction_filtered_count += 1

    logger.info(
        f"Filtered out {instruction_filtered_count} examples with combined instruction length > {instruction_max_length} characters"
    )

    final_count = len(final_filtered_data)
    total_filtered = original_count - final_count

    logger.info("=" * 60)
    logger.info("DATASET PREPROCESSING SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Original dataset size: {original_count} examples")
    logger.info(
        f"Response length filter (>{response_max_length} chars): -{response_filtered_count} examples"
    )
    logger.info(
        f"Combined instruction length filter (>{instruction_max_length} chars): -{instruction_filtered_count} examples"
    )
    logger.info(f"Total filtered out: {total_filtered} examples")
    logger.info(f"Final dataset size: {final_count} examples")
    logger.info(f"Retention rate: {(final_count/original_count)*100:.1f}%")
    logger.info("=" * 60)

    return final_filtered_data


def load_dataset_from_path(
    dataset_name_or_path: str,
    dataset_config_name: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Load dataset from local file or HuggingFace hub (legacy function - loads train split only)."""
    splits_data = load_dataset_with_splits(dataset_name_or_path, dataset_config_name)
    return splits_data["train"]


def load_dataset_with_splits(
    dataset_name_or_path: str,
    dataset_config_name: Optional[str] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Load dataset from local file or HuggingFace hub with proper split handling.

    Returns:
        Dictionary with split names as keys and data as values.
        Always includes 'train' key, may include 'validation', 'test', etc.
    """
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
        logger.info(f"Loaded {len(data)} examples from local file")
        return {"train": data}

    else:
        logger.info(f"Loading dataset from HuggingFace hub: {dataset_name_or_path}")

        # Load the full dataset to check available splits
        try:
            full_dataset = load_dataset(dataset_name_or_path, dataset_config_name)
        except Exception as e:
            logger.error(f"Failed to load dataset {dataset_name_or_path}: {e}")
            raise

        available_splits = list(full_dataset.keys())
        logger.info(f"Available splits: {available_splits}")

        splits_data = {}

        # Load each available split
        for split_name in available_splits:
            split_data = [item for item in full_dataset[split_name]]
            splits_data[split_name] = split_data
            logger.info(f"Loaded {len(split_data)} examples from '{split_name}' split")

        # Ensure we always have a 'train' split
        if "train" not in splits_data:
            # If no train split, use the first available split as train
            first_split = available_splits[0]
            logger.warning(
                f"No 'train' split found. Using '{first_split}' split as training data."
            )
            splits_data["train"] = splits_data[first_split]

        return splits_data


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


def prepare_train_val_splits(
    splits_data: Dict[str, List[Dict[str, Any]]],
    validation_split: float = 0.1,
    prefer_existing_splits: bool = True,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Prepare training and validation data from dataset splits.

    Args:
        splits_data: Dictionary with split names as keys and data as values
        validation_split: Fraction for validation if creating custom split
        prefer_existing_splits: Whether to use existing validation split if available

    Returns:
        Tuple of (train_data, val_data)
    """
    train_data = splits_data["train"]
    val_data = []

    # Check if we should use existing validation split
    if prefer_existing_splits:
        # Look for validation split with common names
        validation_split_names = ["validation", "val", "dev", "valid"]

        for split_name in validation_split_names:
            if split_name in splits_data:
                val_data = splits_data[split_name]
                logger.info(
                    f"Using existing '{split_name}' split with {len(val_data)} examples for validation"
                )
                break

    # If no existing validation split found or not preferred, create custom split
    if not val_data and validation_split > 0:
        logger.info(
            f"No existing validation split found. Creating custom split with {validation_split:.1%} of training data"
        )
        train_data, val_data = split_dataset(train_data, validation_split)
    elif val_data:
        logger.info(
            f"Using {len(train_data)} training examples and {len(val_data)} validation examples from existing splits"
        )
    else:
        logger.info(f"Using {len(train_data)} training examples with no validation")

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
            logger.info("PEFT adapter configuration saved")
        if any(f.startswith("adapter_model") for f in saved_files):
            logger.info("PEFT adapter weights saved")
    else:
        if "config.json" in saved_files:
            logger.info("Model configuration saved")
        if any(
            f.startswith("pytorch_model") or f.endswith(".safetensors")
            for f in saved_files
        ):
            logger.info("Full model weights saved")

    logger.info("Model and tokenizer saved successfully")


def convert_to_gguf(
    model_path: str, output_path: str, quantization: str = "q4_0"
) -> None:
    """
    Convert model to GGUF format for Ollama compatibility.

    Args:
        model_path (str): Path to the model directory to convert
        output_path (str): Path where the GGUF file should be saved
        quantization (str): GGUF quantization type (e.g., "q4_0", "q8_0", "f16")

    Raises:
        FileNotFoundError: If the model path doesn't exist
        subprocess.CalledProcessError: If the conversion command fails
        Exception: For other conversion errors
    """
    try:
        logger.info(f"Converting model to GGUF format: {quantization}")

        # Validate inputs
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model path does not exist: {model_path}")

        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # Check if llama.cpp convert script exists
        convert_script = "convert-hf-to-gguf.py"

        cmd = [
            "python",
            convert_script,
            model_path,
            "--outfile",
            output_path,
            "--outtype",
            quantization,
        ]

        logger.info(f"Running GGUF conversion command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)

        if result.returncode == 0:
            logger.info(f"Successfully converted to GGUF: {output_path}")
        else:
            logger.error(f"GGUF conversion failed: {result.stderr}")
            raise subprocess.CalledProcessError(result.returncode, cmd, result.stderr)

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise
    except subprocess.CalledProcessError as e:
        logger.error(f"GGUF conversion command failed: {e}")
        raise
    except Exception as e:
        logger.error(f"Error during GGUF conversion: {e}")
        raise


def merge_adapter_with_base_model(
    adapter_path: str, output_path: str, base_model_name: Optional[str] = None
) -> None:
    """Merge LoRA adapter with base model to create unified model."""
    import os

    logger.info(f"Loading PEFT model from {adapter_path}")
    model = AutoPeftModelForCausalLM.from_pretrained(adapter_path)

    logger.info("Merging adapter with base model...")
    merged_model = model.merge_and_unload()

    logger.info(f"Saving merged model to {output_path}")
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    merged_model.save_pretrained(output_path)

    logger.info("Adapter merged successfully")
