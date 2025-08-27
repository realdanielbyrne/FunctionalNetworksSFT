#!/usr/bin/env python3
"""
Build ICA Templates from Datasets

This script builds ICA templates from multiple datasets without requiring model training.
The templates can be used later during training to apply pre-computed functional network masks.

Usage:
    # Using positional arguments (recommended):
    python build_ica_templates.py meta-llama/Llama-3.2-1B-Instruct tatsu-lab/alpaca

    # With multiple datasets:
    python build_ica_templates.py meta-llama/Llama-3.2-1B-Instruct databricks/databricks-dolly-15k tatsu-lab/alpaca

    # With optional parameters:
    python build_ica_templates.py meta-llama/Llama-3.2-1B-Instruct dataset1.json dataset2.jsonl \
        --ica_template_samples_per_ds 200 \
        --ica_template_output ./custom/output/ \
        --ica_components 15 \
        --ica_percentile 95.0

    # Using named arguments (also supported):
    python build_ica_templates.py \
        --ica_build_templates_from tatsu-lab/alpaca \
        --model_name_or_path meta-llama/Llama-3.2-1B-Instruct

    # Mixed usage (positional + named optional parameters):
    python build_ica_templates.py meta-llama/Llama-3.2-1B-Instruct tatsu-lab/alpaca \
        --ica_template_samples_per_ds 200

Supported Dataset Formats:
    - Local files: .json, .jsonl, .csv
    - Hugging Face Hub datasets: any dataset name (e.g., squad, alpaca)

Required Arguments (can be positional or named):
    model: Model to use for ICA computation (local path or HF model name) - FIRST positional argument
    datasets: One or more dataset paths (local .json/.jsonl/.csv files or HF dataset names) - remaining positional arguments

    Named argument equivalents:
    --model_name_or_path: Same as model (alternative syntax)
    --ica_build_templates_from: Same as datasets (alternative syntax)

Optional Arguments:
    --ica_template_samples_per_ds: Number of samples per dataset (default: 100)
    --ica_template_output: Output directory for templates (default: ./ica_templates/)
    --ica_components: Number of ICA components (default: 10)
    --ica_percentile: Percentile threshold (default: 98.0)
    --ica_dtype: Data type for ICA computation (default: auto)
    --max_seq_length: Maximum sequence length (default: 512)
    --template_format: Dataset format (default: auto)
"""

import argparse
import logging
import os
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import Subset

# Import from the same package
from .ica_mask import ICAMask
from .utils.model_utils import load_dataset_from_path
from .fnsft_trainer import InstructionDataset
from .utils.dataset_utils import DatasetFormatter
from .utils.config_defaults import ConfigDefaults

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Helpers for organized output structure
INVALID_FS_CHARS_PATTERN = r"[\\/:*?\"<>|]"


def sanitize_for_fs(name: str) -> str:
    """Sanitize a string for safe filesystem usage by replacing invalid characters.

    Replaces characters: / \\ : * ? " < > | with underscores, collapses repeats,
    and strips whitespace.
    """
    if not isinstance(name, str):
        name = str(name)
    # Replace invalid characters with underscore
    safe = re.sub(INVALID_FS_CHARS_PATTERN, "_", name)
    # Replace whitespace runs with single underscore
    safe = re.sub(r"\s+", "_", safe)
    # Collapse multiple underscores
    safe = re.sub(r"_+", "_", safe)
    # Trim leading/trailing underscores
    safe = safe.strip("._-")
    return safe or "unnamed"


def dataset_display_name(dataset_path: str) -> str:
    """Derive a human-friendly dataset name from a path or HF dataset ID."""
    try:
        if os.path.exists(dataset_path):
            return Path(dataset_path).stem
        # Not a local path; use as-is (e.g., tatsu-lab/alpaca)
        return dataset_path
    except Exception:
        return dataset_path


class DatasetLoader:
    """Handles loading and sampling from multiple datasets."""

    @staticmethod
    def load_and_sample_datasets(
        dataset_paths: List[str],
        samples_per_dataset: int,
    ) -> List[Dict[str, Any]]:
        """
        Load multiple datasets and sample specified number of examples from each.

        Args:
            dataset_paths: List of paths to dataset files
            samples_per_dataset: Number of samples to extract from each dataset
            tokenizer: Tokenizer for processing
            max_seq_length: Maximum sequence length
            template_format: Dataset format detection mode

        Returns:
            Combined list of sampled data from all datasets
        """
        combined_data = []

        for dataset_path in dataset_paths:
            logger.info(f"Loading dataset from: {dataset_path}")

            try:
                # Load dataset using existing utility
                data = load_dataset_from_path(dataset_path)
                logger.info(f"Loaded {len(data)} examples from {dataset_path}")

                # Sample specified number of examples
                if len(data) > samples_per_dataset:
                    sampled_indices = random.sample(
                        range(len(data)), samples_per_dataset
                    )
                    sampled_data = [data[i] for i in sampled_indices]
                    logger.info(
                        f"Sampled {len(sampled_data)} examples from {dataset_path}"
                    )
                else:
                    sampled_data = data
                    logger.info(
                        f"Using all {len(sampled_data)} examples from {dataset_path} (less than requested)"
                    )

                combined_data.extend(sampled_data)

            except Exception as e:
                logger.error(f"Failed to load dataset {dataset_path}: {e}")
                raise

        logger.info(f"Combined dataset contains {len(combined_data)} total examples")
        return combined_data


def create_instruction_dataset(
    data: List[Dict[str, Any]],
    tokenizer,
    max_seq_length: int,
    template_format: str = "auto",
) -> InstructionDataset:
    """Create an InstructionDataset from the combined data."""

    # Auto-detect format if needed
    detected_format = None
    if template_format == "auto" and data:
        detected_format = DatasetFormatter.detect_format(data)
        logger.info(f"Detected dataset format: {detected_format}")

    # Create instruction dataset
    dataset = InstructionDataset(
        data=data,
        tokenizer=tokenizer,
        max_length=max_seq_length,  # Fixed parameter name
        instruction_template="",  # Use default
        auto_detect_format=(template_format == "auto"),
        template_format=template_format,
        detected_format=detected_format,
    )

    return dataset


def build_ica_templates(
    dataset_paths: List[str],
    model_name_or_path: str,
    samples_per_dataset: int = 100,
    output_path: str = "./ica_templates/",
    ica_components: int = 10,
    ica_percentile: float = 98.0,
    ica_dtype: str = "auto",
    max_seq_length: int = 512,
    template_format: str = "auto",
) -> None:
    """
    Main function to build ICA templates from multiple datasets.

    Args:
        dataset_paths: List of paths to dataset files (required)
        model_name_or_path: Model to use for ICA computation (required)
        samples_per_dataset: Number of samples to extract from each dataset (default: 100)
        output_path: Output directory for saving templates (default: "./ica_templates/")
        ica_components: Number of ICA components to extract (default: 10)
        ica_percentile: Percentile threshold for component selection (default: 98.0)
        ica_dtype: Data type for ICA computation (default: "auto")
        max_seq_length: Maximum sequence length for tokenization (default: 512)
        template_format: Dataset format detection mode (default: "auto")
    """

    logger.info("Starting ICA template building process...")
    logger.info(f"Input datasets: {dataset_paths}")
    logger.info(f"Samples per dataset: {samples_per_dataset}")
    logger.info(f"Output path: {output_path}")
    logger.info(f"Model: {model_name_or_path}")
    logger.info(f"ICA components: {ica_components}")
    logger.info(f"ICA percentile: {ica_percentile}")

    # Load tokenizer and model
    logger.info("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.float32,  # Use float32 for stability
        device_map="auto",
    )
    model.eval()

    # Load and combine datasets
    logger.info("Loading and sampling datasets...")
    combined_data = DatasetLoader.load_and_sample_datasets(
        dataset_paths=dataset_paths, samples_per_dataset=samples_per_dataset
    )

    # Create instruction dataset
    logger.info("Creating instruction dataset...")
    dataset = create_instruction_dataset(
        data=combined_data,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        template_format=template_format,
    )

    # Initialize ICA mask handler
    logger.info("Initializing ICA mask handler...")
    ica_mask = ICAMask(
        num_components=ica_components,
        percentile=ica_percentile,
        sample_batches=100,
        ica_dtype=ica_dtype,
    )

    # Use subset for ICA computation (limit to reasonable size)
    max_samples_for_ica = min(1024, len(dataset))
    sample_for_ica = Subset(dataset, range(max_samples_for_ica))
    logger.info(f"Using {max_samples_for_ica} samples for ICA computation")

    # Compute global networks
    logger.info("Computing global ICA networks...")
    component_masks = ica_mask.compute_global_networks(
        model=model,
        dataset=sample_for_ica,
        tokenizer=tokenizer,
        target_layers=None,  # Use all layers
        n_components=ica_components,
        top_percentile_per_component=ica_percentile,
    )

    if not component_masks:
        logger.error("Global ICA produced no component masks!")
        return

    # Build templates
    logger.info("Building templates from computed components...")
    templates = ica_mask.build_templates_from_current_components(
        name="global_templates_v1"
    )

    # Save templates using organized directory structure: {base}/{model}/{dataset}/
    model_base = ConfigDefaults.extract_model_base_name(model_name_or_path)
    model_dir = sanitize_for_fs(model_base)

    # Derive dataset label (support multiple datasets)
    ds_names = [dataset_display_name(p) for p in dataset_paths]
    ds_label_raw = ds_names[0] if len(ds_names) == 1 else "__".join(ds_names)
    dataset_dir = sanitize_for_fs(ds_label_raw)

    template_dir = os.path.join(output_path, model_dir, dataset_dir)
    os.makedirs(template_dir, exist_ok=True)
    template_file_path = os.path.join(template_dir, "global_templates.json")

    logger.info(
        f"Saving templates to organized path: base='{output_path}', model='{model_dir}', dataset='{dataset_dir}'"
    )
    logger.info(f"Full template path: {template_file_path}")
    ica_mask.save_templates(template_file_path, templates)

    # Log summary
    logger.info("Template building completed successfully!")
    logger.info(f"Templates saved to: {template_file_path}")
    logger.info(f"Number of components: {len(templates['templates'])}")

    # Log component coverage summary
    logger.info("Component Coverage Summary:")
    for cid, comp in templates["templates"].items():
        layer_counts = {lid: len(chs) for lid, chs in comp.items()}
        total_channels = sum(layer_counts.values())
        logger.info(
            f"  • Component {cid}: {total_channels} channels across {len(layer_counts)} layers"
        )


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Build ICA templates from multiple datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Positional arguments (optional parameter names)
    parser.add_argument(
        "model",
        type=str,
        nargs="?",
        help="Model name or path to use for ICA computation (first positional argument)",
    )
    parser.add_argument(
        "datasets",
        type=str,
        nargs="*",
        help="One or more dataset paths to build templates from (remaining positional arguments)",
    )

    # Named arguments (alternative syntax)
    parser.add_argument(
        "--ica_build_templates_from",
        type=str,
        nargs="+",
        help="One or more dataset paths to build templates from (alternative to positional datasets)",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Model name or path to use for ICA computation (alternative to positional model)",
    )

    # Optional arguments with reasonable defaults
    parser.add_argument(
        "--ica_template_samples_per_ds",
        type=int,
        default=100,
        help="Number of samples to extract from each dataset (default: 100)",
    )
    parser.add_argument(
        "--ica_template_output",
        type=str,
        default="./ica_templates/",
        help="Output directory for saving the generated templates (default: ./ica_templates/)",
    )
    parser.add_argument(
        "--ica_components",
        type=int,
        default=10,
        help="Number of ICA components to extract (default: 10)",
    )
    parser.add_argument(
        "--ica_percentile",
        type=float,
        default=98.0,
        help="Percentile threshold for component selection (default: 98.0)",
    )
    parser.add_argument(
        "--ica_dtype",
        type=str,
        default="auto",
        choices=["auto", "float32", "float16", "bfloat16"],
        help="Data type for ICA computation (default: auto)",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=512,
        help="Maximum sequence length for tokenization (default: 512)",
    )
    parser.add_argument(
        "--template_format",
        type=str,
        default="auto",
        choices=["auto", "chat", "alpaca", "chatml", "basic"],
        help="Dataset format detection mode (default: auto)",
    )

    args = parser.parse_args()

    # Handle positional vs named arguments
    dataset_paths = None
    model_name_or_path = None

    # Check if positional arguments were provided
    if args.model and args.datasets:
        # Using positional arguments
        if len(args.datasets) == 0:
            parser.error("At least one dataset path is required")
        # First positional argument is the model, rest are datasets
        model_name_or_path = args.model
        dataset_paths = args.datasets
    elif args.ica_build_templates_from and args.model_name_or_path:
        # Using named arguments
        dataset_paths = args.ica_build_templates_from
        model_name_or_path = args.model_name_or_path
    elif args.model and args.ica_build_templates_from:
        # Mixed: positional model, named datasets
        model_name_or_path = args.model
        dataset_paths = args.ica_build_templates_from
    elif args.datasets and args.model_name_or_path:
        # Mixed: positional datasets, named model
        dataset_paths = args.datasets
        model_name_or_path = args.model_name_or_path
    else:
        # Neither complete set provided
        parser.error(
            "Required arguments missing. Provide either:\n"
            "  1. Positional: <model> <datasets...>\n"
            "  2. Named: --model_name_or_path <model> --ica_build_templates_from <datasets...>\n"
            "  3. Mixed: <model> --ica_build_templates_from <datasets...>\n"
            "  4. Mixed: --model_name_or_path <model> <datasets...>"
        )

    # Validate arguments
    if args.ica_template_samples_per_ds <= 0:
        parser.error("--ica_template_samples_per_ds must be positive")

    if args.ica_components <= 0:
        parser.error("--ica_components must be positive")

    if not (0 < args.ica_percentile <= 100):
        parser.error("--ica_percentile must be between 0 and 100")

    # Check dataset paths exist (only for local files, HF datasets will be validated during loading)
    for dataset_path in dataset_paths:
        # Only check existence for local file paths, not HuggingFace dataset names
        if (
            "/" not in dataset_path
            or os.path.isabs(dataset_path)
            or dataset_path.startswith("./")
            or dataset_path.startswith("../")
        ):
            # This looks like a local file path
            if not os.path.exists(dataset_path):
                parser.error(f"Local dataset file does not exist: {dataset_path}")
        # For HuggingFace dataset names (containing "/" but not local paths), skip validation here
        # They will be validated when load_dataset_from_path() is called

    # Ensure output directory is absolute path for clarity
    args.ica_template_output = os.path.abspath(args.ica_template_output)

    # Set random seed for reproducibility
    random.seed(42)
    torch.manual_seed(42)

    try:
        build_ica_templates(
            dataset_paths=dataset_paths,
            model_name_or_path=model_name_or_path,
            samples_per_dataset=args.ica_template_samples_per_ds,
            output_path=args.ica_template_output,
            ica_components=args.ica_components,
            ica_percentile=args.ica_percentile,
            ica_dtype=args.ica_dtype,
            max_seq_length=args.max_seq_length,
            template_format=args.template_format,
        )
    except Exception as e:
        logger.error(f"Template building failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
