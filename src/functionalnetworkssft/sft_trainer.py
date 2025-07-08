#!/usr/bin/env python3
"""
Supervised Fine-Tuning (SFT) Script for Quantized Language Models

This script provides a complete solution for fine-tuning quantized language models
using LoRA/QLoRA techniques with support for various model architectures and datasets.

Author: Daniel Byrne
License: MIT
"""

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

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import transformers
from transformers.utils.quantization_config import BitsAndBytesConfig
from transformers.training_args import TrainingArguments
from transformers.trainer import Trainer
from transformers.data.data_collator import DataCollatorForLanguageModeling
from transformers.trainer_callback import EarlyStoppingCallback
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.modeling_utils import PreTrainedModel
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from huggingface_hub import HfApi, login, whoami
from huggingface_hub.errors import RepositoryNotFoundError, HfHubHTTPError
import wandb
from tqdm.auto import tqdm

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("sft_training.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


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
    auto_detect_format: bool = field(
        default=True,
        metadata={"help": "Automatically detect and convert dataset format"},
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
    split_dataset,
    save_model_and_tokenizer,
)

# ===================== NEW: ICA helpers =====================
from sklearn.decomposition import FastICA
import numpy as np
import json
from collections import defaultdict
import itertools


def apply_ica_masks(
    model: PreTrainedModel, mask_dict: dict[str, list[int]], mask_mode: str = "key"
):
    """
    Inject forward pre-hooks that multiply the *input* of each MLP
    down-projection by a binary mask.  Implementation follows the
    reference design  [oai_citation:1‡ICA-Based Functional Network Masking for LLM Fine-Tuning.pdf](file-service://file-78U49V8bsfQqVCbLViHD19).
    """
    handles = []
    hidden_size = (
        getattr(model.config, "hidden_size", None)
        or getattr(model.config, "n_embd", None)
        or getattr(model.config, "d_model", None)
        or model.get_input_embeddings().embedding_dim
    )

    # Locate decoder blocks (works for GPT-like and Llama-like layouts)
    if hasattr(model, "transformer"):
        blocks = getattr(model.transformer, "h", None) or getattr(
            model.transformer, "blocks", None
        )
    elif hasattr(model, "model"):
        blocks = getattr(model.model, "layers", None) or getattr(
            model.model, "decoder", None
        )
    else:
        blocks = None
    if blocks is None:
        logger.warning("Could not find transformer blocks – no masking applied.")
        return handles

    for layer_idx, block in enumerate(blocks):
        for module in block.modules():
            # pick the *second* Linear in the MLP (in_features > hidden_size)
            if (
                isinstance(module, nn.Linear)
                or module.__class__.__name__ == "Linear8bitLt"
            ):
                if (
                    module.out_features == hidden_size
                    and module.in_features > hidden_size
                ):
                    neuron_ids = mask_dict.get(str(layer_idx), [])
                    if mask_mode == "key":  # zero the key neurons
                        mask = torch.ones(module.in_features)
                        mask[neuron_ids] = 0.0
                    else:  # zero everything *except* key neurons
                        mask = torch.zeros(module.in_features)
                        mask[neuron_ids] = 1.0
                    mask = mask.float()

                    def pre_hook(mod, inp, mask_tensor=mask):
                        x = inp[0]
                        return (x * mask_tensor.to(x.device, x.dtype),) + inp[1:]

                    handles.append(module.register_forward_pre_hook(pre_hook))
                    break  # stop after first matching linear in this block
    return handles


# ------------------------------------------------------------


def compute_ica_masks_for_model(
    model: PreTrainedModel,
    dataset: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    num_components: int = 20,
    percentile: float = 98.0,
    sample_batches: int = 100,
):
    """
    Lightweight, on-the-fly ICA (FastICA) over MLP activations.
    Adapted from the pseudocode in the paper  [oai_citation:2‡ICA-Based Functional Network Masking for LLM Fine-Tuning.pdf](file-service://file-78U49V8bsfQqVCbLViHD19).
    Only a *sample* of the dataset is streamed to limit RAM/CPU use.
    Returns a `mask_dict` ready for `apply_ica_masks`.
    """
    logger.info("Running ICA to discover functional networks – this can be slow…")
    model.eval()
    device = next(model.parameters()).device

    # 1. collect activations
    activations = defaultdict(list)
    hooks = []

    def capture(layer_idx):
        def _hook(_, __, out):
            # out: [B, T, d_int]  -> flatten B*T
            activations[layer_idx].append(out.detach().cpu().float())

        return _hook

    # attach capture hooks on the MLP *intermediate* output
    if hasattr(model, "transformer"):
        blocks = getattr(model.transformer, "h", None) or getattr(
            model.transformer, "blocks", None
        )
    else:
        blocks = getattr(model.model, "layers", None)

    for i, block in enumerate(blocks):
        # first linear (up-proj) output lies right after activation
        up_proj = next(
            m
            for n, m in block.named_modules()
            if isinstance(m, nn.Linear) and m.out_features > m.in_features
        )
        hooks.append(up_proj.register_forward_hook(capture(i)))

    # 2. feed a few mini-batches
    dl = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    with torch.no_grad():
        for idx, sample in enumerate(itertools.islice(dl, sample_batches)):
            model(
                input_ids=sample["input_ids"].to(device),
                attention_mask=sample["attention_mask"].to(device),
            )
            if idx and idx % 10 == 0:
                logger.info(f"  captured {idx}/{sample_batches} batches…")

    for h in hooks:
        h.remove()

    # 3. concatenate and run ICA
    layer_masks = {}
    for layer_idx, acts in activations.items():
        X = torch.cat(acts, dim=0).flatten(0, 1).numpy()  # [time, neurons]
        ica = FastICA(n_components=num_components, random_state=0)
        try:
            A = ica.fit_transform(
                X
            ).T  # components × time  (we only need mixing matrix)
            mixing = ica.mixing_  # [neurons, components]
        except ValueError:
            logger.warning(f"ICA failed on layer {layer_idx}, skipping.")
            continue
        # 4. pick top-|percentile| neurons across all components
        thr = np.percentile(np.abs(mixing), percentile)
        key_neurons = np.where(np.abs(mixing) >= thr)[0].tolist()
        if key_neurons:
            layer_masks[str(layer_idx)] = key_neurons

    logger.info(
        f"ICA complete – masking {sum(len(v) for v in layer_masks.values())} neurons."
    )
    return layer_masks


# ============================================================


class InstructionDataset(Dataset):
    """Enhanced dataset class for instruction-following data with automatic format detection."""

    def __init__(
        self,
        data: List[Dict[str, Any]],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 2048,
        instruction_template: str = "### Instruction:\n{instruction}\n\n### Response:\n{response}",
        auto_detect_format: bool = True,
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.instruction_template = instruction_template
        self.auto_detect_format = auto_detect_format

        # Set pad token if not exists
        if getattr(tokenizer, "pad_token", None) is None:
            eos_token = getattr(tokenizer, "eos_token", None)
            if eos_token is not None:
                tokenizer.pad_token = eos_token

        # Detect and log dataset format
        if self.auto_detect_format and data:
            self.detected_format = DatasetFormatter.detect_format(data)
            logger.info(f"Detected dataset format: {self.detected_format}")

            # Convert first sample to show the transformation
            if len(data) > 0:
                sample_converted = DatasetFormatter.convert_to_standard_format(
                    data[0], self.detected_format
                )
                logger.info(f"Sample conversion: {data[0]} -> {sample_converted}")
        else:
            self.detected_format = None

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]

        # Convert to standard format if auto-detection is enabled
        if self.auto_detect_format and self.detected_format:
            try:
                converted_item = DatasetFormatter.convert_to_standard_format(
                    item, self.detected_format
                )
            except Exception as e:
                logger.warning(
                    f"Failed to convert item {idx}: {e}. Using original format."
                )
                converted_item = item
        else:
            converted_item = item

        # Format the text
        if "instruction" in converted_item and "response" in converted_item:
            text = self.instruction_template.format(
                instruction=converted_item["instruction"],
                response=converted_item["response"],
            )
        elif "text" in converted_item:
            text = converted_item["text"]
        else:
            # Fallback for legacy behavior
            if "instruction" in item and "response" in item:
                text = self.instruction_template.format(
                    instruction=item["instruction"], response=item["response"]
                )
            elif "text" in item:
                text = item["text"]
            else:
                raise ValueError(
                    f"Dataset item {idx} must contain either 'instruction'+'response' or 'text' fields. "
                    f"Available keys: {list(item.keys())}"
                )

        # Tokenize
        encoding = self.tokenizer.__call__(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids.clone(),
        }


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
        torch_dtype = torch.float16
        if model_args.torch_dtype == "bfloat16":
            torch_dtype = torch.bfloat16
        elif model_args.torch_dtype == "float32":
            torch_dtype = torch.float32

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
        use_auth_token=model_args.use_auth_token,
    )
    # Set pad token if not exists
    if getattr(tokenizer, "pad_token", None) is None:
        eos_token = getattr(tokenizer, "eos_token", None)
        if eos_token is not None:
            tokenizer.pad_token = eos_token
            tokenizer.pad_token_id = getattr(tokenizer, "eos_token_id", None)

    # Load model with cross-platform device mapping
    device_map = "auto" if device.type in ["cuda", "mps"] else None

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        quantization_config=quant_config,
        torch_dtype=torch_dtype,
        trust_remote_code=model_args.trust_remote_code,
        use_auth_token=model_args.use_auth_token,
        device_map=device_map,
    )

    # Move to device if device_map wasn't used
    if device_map is None:
        model = model.to(device)

    return model, tokenizer


def setup_lora_from_args(
    model: PreTrainedModel, lora_args: LoRAArguments
) -> PreTrainedModel:
    """Setup LoRA configuration from arguments."""
    return setup_lora(
        model=model,
        lora_r=lora_args.lora_r,
        lora_alpha=lora_args.lora_alpha,
        lora_dropout=lora_args.lora_dropout,
        lora_target_modules=lora_args.lora_target_modules,
        lora_bias=lora_args.lora_bias,
    )


def load_dataset_from_args(data_args: DataArguments) -> List[Dict[str, Any]]:
    """Load dataset from arguments."""
    return load_dataset_from_path(
        dataset_name_or_path=data_args.dataset_name_or_path,
        dataset_config_name=data_args.dataset_config_name,
    )


def create_trainer(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    train_dataset: Dataset,
    eval_dataset: Optional[Dataset],
    training_args: TrainingArguments,
    data_collator: DataCollatorForLanguageModeling,
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


# save_model_and_tokenizer is now imported from utils.model_utils


def convert_to_gguf(
    model_path: str, output_path: str, quantization: str = "q4_0"
) -> None:
    """Convert model to GGUF format for Ollama compatibility."""
    try:
        import subprocess

        logger.info(f"Converting model to GGUF format: {quantization}")

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

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            logger.info(f"Successfully converted to GGUF: {output_path}")
        else:
            logger.error(f"GGUF conversion failed: {result.stderr}")

    except ImportError:
        logger.warning("llama.cpp not available for GGUF conversion")
    except Exception as e:
        logger.error(f"Error during GGUF conversion: {e}")


def load_config_from_yaml(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def upload_to_hub(
    model_path: str,
    tokenizer: PreTrainedTokenizerBase,
    repo_id: str,
    commit_message: Optional[str] = None,
    private: bool = False,
    token: Optional[str] = None,
    push_adapter_only: bool = False,
) -> None:
    try:
        logger.info(f"Starting upload to Hugging Face Hub: {repo_id}")

        # Validate inputs
        if not repo_id or "/" not in repo_id:
            raise ValueError(
                "repo_id must be in format 'username/repository-name' or 'organization/repository-name'"
            )

        if not os.path.exists(model_path):
            raise ValueError(f"Model path does not exist: {model_path}")

        # Set default commit message
        if commit_message is None:
            commit_message = "Upload fine-tuned model with LoRA adapters"

        # Handle authentication
        if token is None:
            token = os.getenv("HF_TOKEN")

        if token is None:
            logger.info(
                "No HF_TOKEN found in environment. Attempting to use cached credentials..."
            )
            try:
                # Check if user is already logged in
                user_info = whoami(token=token)
                logger.info(f"Using cached credentials for user: {user_info['name']}")
            except Exception:
                logger.info(
                    "No cached credentials found. Please log in to Hugging Face Hub..."
                )
                login()
        else:
            logger.info("Using provided authentication token")

        # Initialize HF API
        api = HfApi(token=token)

        # Check if repository exists, create if it doesn't
        try:
            api.repo_info(repo_id=repo_id, repo_type="model")
            logger.info(f"Repository {repo_id} exists")
        except RepositoryNotFoundError:
            logger.info(f"Creating new repository: {repo_id}")
            api.create_repo(
                repo_id=repo_id, repo_type="model", private=private, exist_ok=True
            )

        # Determine which files to upload
        files_to_upload = []

        if push_adapter_only:
            # Only upload LoRA adapter files
            adapter_files = [
                "adapter_config.json",
                "adapter_model.safetensors",
                "adapter_model.bin",  # fallback for older format
            ]

            for file_name in adapter_files:
                file_path = os.path.join(model_path, file_name)
                if os.path.exists(file_path):
                    files_to_upload.append(file_name)

            if not files_to_upload:
                raise ValueError(f"No LoRA adapter files found in {model_path}")

            logger.info(f"Uploading LoRA adapter files: {files_to_upload}")
        else:
            # Upload all model files
            logger.info("Uploading full model (base model + adapters)")

        # Upload tokenizer first
        logger.info("Uploading tokenizer...")
        if hasattr(tokenizer, "push_to_hub"):
            tokenizer.push_to_hub(
                repo_id=repo_id,
                commit_message=f"{commit_message} - tokenizer",
                token=token,
                private=private,
            )

        # Upload model files
        if push_adapter_only:
            # Upload individual adapter files
            for file_name in files_to_upload:
                file_path = os.path.join(model_path, file_name)
                logger.info(f"Uploading {file_name}...")
                api.upload_file(
                    path_or_fileobj=file_path,
                    path_in_repo=file_name,
                    repo_id=repo_id,
                    repo_type="model",
                    commit_message=f"{commit_message} - {file_name}",
                    token=token,
                )
        else:
            # Upload entire model directory
            logger.info("Uploading model files...")

            # Load and upload the model using transformers
            try:
                # Try to load as PEFT model first
                from peft import PeftModel, AutoPeftModelForCausalLM

                # Check if this is a PEFT model
                if os.path.exists(os.path.join(model_path, "adapter_config.json")):
                    logger.info("Detected PEFT model, uploading with PEFT support...")
                    model = AutoPeftModelForCausalLM.from_pretrained(model_path)
                    model.push_to_hub(
                        repo_id=repo_id,
                        commit_message=commit_message,
                        token=token,
                        private=private,
                    )
                else:
                    # Regular model upload
                    model = AutoModelForCausalLM.from_pretrained(model_path)
                    model.push_to_hub(
                        repo_id=repo_id,
                        commit_message=commit_message,
                        token=token,
                        private=private,
                    )
            except Exception as e:
                logger.warning(f"Failed to upload using transformers: {e}")
                logger.info("Falling back to file-by-file upload...")

                # Fallback: upload directory contents
                api.upload_folder(
                    folder_path=model_path,
                    repo_id=repo_id,
                    repo_type="model",
                    commit_message=commit_message,
                    token=token,
                )

        logger.info(
            f"✅ Successfully uploaded model to: https://huggingface.co/{repo_id}"
        )

    except RepositoryNotFoundError as e:
        logger.error(f"Repository not found and could not be created: {e}")
        raise
    except HfHubHTTPError as e:
        if "401" in str(e):
            logger.error(
                "Authentication failed. Please check your token or run 'huggingface-cli login'"
            )
        elif "403" in str(e):
            logger.error(
                "Permission denied. Check if you have write access to the repository"
            )
        elif "404" in str(e):
            logger.error(
                "Repository not found. Make sure the repository name is correct"
            )
        else:
            logger.error(f"HTTP error during upload: {e}")
        raise
    except ValueError as e:
        logger.error(f"Invalid input: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during upload: {e}")
        raise


def main():
    """Main training function."""
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
        required=True,
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
        required=True,
        help="Path to dataset file or HuggingFace dataset name",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="Configuration name for HuggingFace dataset",
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

    # Quantization arguments
    parser.add_argument(
        "--use_4bit", action="store_true", default=True, help="Use 4-bit quantization"
    )
    parser.add_argument(
        "--use_8bit", action="store_true", help="Use 8-bit quantization"
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

    # LoRA arguments
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
        required=True,
        help="Output directory for model checkpoints",
    )
    parser.add_argument(
        "--num_train_epochs", type=int, default=3, help="Number of training epochs"
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
        "--save_steps", type=int, default=500, help="Save checkpoint steps"
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
        "--push_adapter_only",
        action="store_true",
        help="Only upload LoRA adapter files to Hub (not the full model)",
    )

    # Masking and ICA arguments
    parser.add_argument(
        "--mask_mode",
        type=str,
        choices=["key", "complement"],
        default=None,
        help="When 'key' is selected, it should ablate (disable) the ICA-identified key neurons. When 'complement' is selected, it should keep only the key neurons active. When omitted, normal training should proceed without masking.",
    )
    parser.add_argument(
        "--ica_mask_path",
        type=str,
        default=None,
        help="A string argument defaulting to None that specifies the file path to a JSON file containing a dictionary mapping layer indices to lists of neuron indices (format: {layer-idx: [neuron-idx,…]}) as produced by an offline ICA analysis run.",
    )
    parser.add_argument(
        "--ica_components",
        type=int,
        default=20,
        help="An integer argument defaulting to 20 that specifies the number of independent components to extract when ICA needs to be performed on-the-fly during training.",
    )
    parser.add_argument(
        "--ica_percentile",
        type=float,
        default=98.0,
        help="A float argument defaulting to 98.0 that sets the percentile threshold (valid range 0-100) for selecting neurons within each component when ICA is executed on-the-fly.",
    )

    args = parser.parse_args()

    # Load configuration from YAML if provided
    if args.config:
        config = load_config_from_yaml(args.config)
        # Override command line args with config values
        for key, value in config.items():
            if not hasattr(args, key) or getattr(args, key) is None:
                setattr(args, key, value)

    # Create argument dataclasses
    model_args = ModelArguments(
        model_name_or_path=args.model_name_or_path,
        use_auth_token=args.use_auth_token,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=args.torch_dtype,
    )

    data_args = DataArguments(
        dataset_name_or_path=args.dataset_name_or_path,
        dataset_config_name=args.dataset_config_name,
        max_seq_length=args.max_seq_length,
        instruction_template=args.instruction_template,
        validation_split=args.validation_split,
        auto_detect_format=args.auto_detect_format,
    )

    quant_args = QuantizationArguments(
        use_4bit=args.use_4bit,
        use_8bit=args.use_8bit,
        bnb_4bit_compute_dtype=args.bnb_4bit_compute_dtype,
        bnb_4bit_quant_type=args.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=args.bnb_4bit_use_double_quant,
    )

    lora_args = LoRAArguments(
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=args.lora_target_modules,
        lora_bias=args.lora_bias,
    )

    # Initialize Weights & Biases if requested
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            config=vars(args),
            name=f"sft-{Path(args.model_name_or_path).name}-{Path(args.dataset_name_or_path).name}",
        )

    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
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
        save_strategy="steps",
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=args.load_best_model_at_end,
        metric_for_best_model=args.metric_for_best_model,
        greater_is_better=args.greater_is_better,
        report_to="wandb" if args.use_wandb else None,
        run_name=f"sft-{Path(args.model_name_or_path).name}",
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        gradient_checkpointing=True,
        fp16=args.torch_dtype == "float16",
        bf16=args.torch_dtype == "bfloat16",
    )

    logger.info("Starting supervised fine-tuning...")
    logger.info(f"Model: {model_args.model_name_or_path}")
    logger.info(f"Dataset: {data_args.dataset_name_or_path}")
    logger.info(f"Output directory: {args.output_dir}")

    try:
        # Load quantization config
        quant_config = load_quantization_config_from_args(quant_args)

        # Load model and tokenizer
        model, tokenizer = load_model_and_tokenizer(model_args, quant_config)

        # Setup LoRA
        model = setup_lora_from_args(model, lora_args)

        # Load and prepare dataset
        data = load_dataset_from_args(data_args)
        train_data, val_data = split_dataset(data, data_args.validation_split)

        # Create datasets
        train_dataset = InstructionDataset(
            train_data,
            tokenizer,
            data_args.max_seq_length,
            data_args.instruction_template,
            data_args.auto_detect_format,
        )

        eval_dataset = None
        if val_data:
            eval_dataset = InstructionDataset(
                val_data,
                tokenizer,
                data_args.max_seq_length,
                data_args.instruction_template,
                data_args.auto_detect_format,
            )

        # Create data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
        )

        # ---------- NEW: functional-network masking ----------
        mask_handles = []
        if args.mask_mode is not None:
            if args.ica_mask_path:
                with open(args.ica_mask_path) as f:
                    mask_dict = json.load(f)
                logger.info(f"Loaded pre-computed ICA mask from {args.ica_mask_path}")
            else:
                # use *training* split to estimate ICA masks quickly
                sample_for_ica = torch.utils.data.Subset(
                    train_dataset, range(min(1024, len(train_dataset)))
                )
                mask_dict = compute_ica_masks_for_model(
                    model,
                    sample_for_ica,
                    tokenizer,
                    num_components=args.ica_components,
                    percentile=args.ica_percentile,
                    sample_batches=50,
                )
            mask_handles = apply_ica_masks(model, mask_dict, mask_mode=args.mask_mode)
            logger.info(f"Applied functional-network masking: mode={args.mask_mode}")
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

        # Start training
        logger.info("Starting training...")
        trainer.train(resume_from_checkpoint=checkpoint)

        # Remove hooks to save an unmasked model
        for h in mask_handles:
            h.remove()

        # Save final model
        final_output_dir = os.path.join(args.output_dir, "final_model")
        save_model_and_tokenizer(model, tokenizer, final_output_dir)

        # Convert to GGUF if requested
        if args.convert_to_gguf:
            gguf_output_path = os.path.join(args.output_dir, "model.gguf")
            convert_to_gguf(final_output_dir, gguf_output_path, args.gguf_quantization)

        # Upload to Hugging Face Hub if requested
        if args.push_to_hub:
            if not args.hub_repo_id:
                logger.error("--hub_repo_id is required when using --push_to_hub")
                raise ValueError("hub_repo_id must be specified for Hub upload")

            try:
                upload_to_hub(
                    model_path=final_output_dir,
                    tokenizer=tokenizer,
                    repo_id=args.hub_repo_id,
                    commit_message=args.hub_commit_message,
                    private=args.hub_private,
                    token=args.hub_token,
                    push_adapter_only=args.push_adapter_only,
                )
            except Exception as e:
                logger.error(f"Failed to upload to Hub: {e}")
                # Don't raise here to allow training to complete successfully
                # even if upload fails

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
