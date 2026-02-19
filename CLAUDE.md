# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FunctionalNetworksSFT is a research framework for selective fine-tuning of Large Language Models using brain-inspired functional network masking. It implements ICA (Independent Component Analysis) to identify functional networks of neurons in LLMs and enables selective training/freezing of these networks during supervised fine-tuning to mitigate catastrophic forgetting.

**Key Concept**: Groups of neurons in LLMs form functional networks analogous to functional brain networks. By identifying these networks via ICA and selectively masking them during fine-tuning, we can preserve pre-trained knowledge while learning new tasks.

## Common Commands

### Development Setup

```bash
# Install dependencies (platform-specific)
# CUDA systems (NVIDIA GPUs)
poetry run python scripts/setup_cuda.py

# Apple Silicon (M1/M2/M3/M4)
poetry install --extras apple-silicon

# CPU-only systems
poetry install --extras cpu

# Verify installation
poetry run python -c "from functionalnetworkssft.utils.model_utils import get_optimal_device; print(get_optimal_device())"
```

### Running Tests

```bash
# Run all tests
poetry run pytest

# Run specific test file
poetry run pytest tests/test_ica_masking.py

# Run with verbose output
poetry run pytest -v

# Run installation verification
python tests/test_installation.py

# Run comprehensive ICA test suite
python tests/test_ica_suite.py
```

### Building ICA Templates

Templates must be built before using ICA masking during training:

```bash
# Build ICA templates from datasets (positional args - recommended)
poetry run python -m functionalnetworkssft.build_ica_templates \
    meta-llama/Llama-3.2-1B-Instruct \
    tatsu-lab/alpaca

# With multiple datasets and custom parameters
poetry run python -m functionalnetworkssft.build_ica_templates \
    meta-llama/Llama-3.2-1B-Instruct \
    dataset1.json dataset2.jsonl \
    --ica_template_samples_per_ds 200 \
    --ica_template_output ./ica_templates/ \
    --ica_components 15 \
    --ica_percentile 95.0

# Alternative: Using named arguments
poetry run buildtemplates \
    --model_name_or_path meta-llama/Llama-3.2-1B-Instruct \
    --ica_build_templates_from tatsu-lab/alpaca
```

### Training Models

```bash
# Basic fine-tuning (no ICA masking)
poetry run fnsft \
    --model_name_or_path meta-llama/Llama-3.2-1B-Instruct \
    --dataset_name_or_path your_dataset.json \
    --output_dir ./output \
    --num_train_epochs 2 \
    --per_device_train_batch_size 4 \
    --use_auth_token

# Training with ICA masking (lesion mode - zero out specific networks)
poetry run fnsft \
    --model_name_or_path meta-llama/Llama-3.2-1B-Instruct \
    --dataset_name_or_path your_dataset.json \
    --output_dir ./output \
    --mask_mode lesion \
    --ica_template_path ./ica_templates/global_templates.json \
    --ica_component_ids [0,1] \
    --ica_components 10 \
    --ica_percentile 98.0

# Training with ICA masking (preserve mode - train only specific networks)
poetry run fnsft \
    --model_name_or_path meta-llama/Llama-3.2-1B-Instruct \
    --dataset_name_or_path your_dataset.json \
    --output_dir ./output \
    --mask_mode preserve \
    --ica_template_path ./ica_templates/global_templates.json \
    --ica_component_ids [0,1]

# Using YAML config file
poetry run fnsft --config experiments/continual_learning/configs/llama_7b.yaml
```

### Running Continual Learning Experiments

```bash
# Run full experiment suite (all methods, all orders)
poetry run fnsft-cl-orchestrate \
    --config experiments/continual_learning/configs/llama_7b.yaml \
    --phase all

# Quick smoke test (2 tasks, 50 steps)
poetry run fnsft-cl-orchestrate \
    --model llama-3.2-1b \
    --methods lora ewc \
    --orders order_1 \
    --seeds 1 \
    --override_steps 50 \
    --skip_long_chains \
    --phase experiments

# Aggregate results into publication tables
poetry run fnsft-cl-aggregate \
    --results_csv ./experiments/continual_learning/results/experiments/llama-7b_results.csv \
    --model llama-7b
```

## High-Level Architecture

### Core Components

**1. ICA Masking System (`src/functionalnetworkssft/ica_mask.py`)**

- **ICAMask class**: Main interface for computing and applying functional network masks
- **Two analysis modes**:
  - Per-layer FFN-intermediate ICA: Selects key FFN neurons per layer
  - Global group-wise ICA: Runs ICA across all layers concatenated
- **Masking mechanism**: Uses PyTorch forward hooks on MLP down-projection layers to apply binary masks
- **RowMaskedDelta parametrization**: Enables training only unmasked neurons while keeping masked neurons frozen
- **Template system**: Supports pre-computed templates loaded from JSON files

**2. Training Pipeline (`src/functionalnetworkssft/fnsft_trainer.py`)**

- **Main entry point**: `fnsft` CLI command
- **Integrates with HuggingFace**: Uses Transformers Trainer, supports LoRA/QLoRA via PEFT
- **Two training modes**:
  - Standard PEFT fine-tuning (no masking)
  - ICA-masked fine-tuning (lesion or preserve mode)
- **InstructionDataset class**: Handles dataset loading and chat template formatting
- **Authentication handling**: Multi-method HuggingFace authentication (`.env`, environment variables, cached credentials)
- **Anti-drift compensation**: Optional mechanism to prevent optimizer drift for frozen parameters

**3. Template Building (`src/functionalnetworkssft/build_ica_templates.py`)**

- **Standalone tool**: Builds ICA templates without training
- **Supports multiple datasets**: Aggregates activations across datasets
- **Output format**: JSON files containing component masks per layer
- **Flexible input**: Local files (.json, .jsonl, .csv) or HuggingFace Hub datasets

**4. Dataset Utilities (`src/functionalnetworkssft/utils/dataset_utils.py`)**

- **DatasetFormatter class**: Auto-detects and converts various dataset formats
- **Format mappings**: Supports instruction-response, input-output, Alpaca, Dolly, and other common formats
- **Chat template handling**: Automatically applies model-specific chat templates

**5. Model Utilities (`src/functionalnetworkssft/utils/model_utils.py`)**

- **Device detection**: `get_optimal_device()` - Cross-platform CUDA/MPS/CPU detection
- **Quantization support**: BitsAndBytes 4-bit/8-bit loading (CUDA only)
- **LoRA configuration**: Automated PEFT setup for parameter-efficient training
- **Model loading**: Handles gated models, authentication, and dtype optimization

### Key Technical Details

**ICA Masking Implementation**:

- Masks are applied at MLP output (after activation, before residual connection)
- Binary masks (0 or 1) are broadcast across batch and sequence dimensions
- Forward hooks intercept down-projection input and apply masks element-wise
- Masked neurons receive zero gradient and don't update during training
- Two modes:
  - **Lesion**: Zero out selected components (ablate functional networks)
  - **Preserve**: Train only selected components (isolate functional networks)

**Hook Placement Strategy**:

- Targets MLP down-projection layer (e.g., `c_proj` in GPT-2, `down_proj` in LLaMA)
- Identifies correct layer by checking: `out_features == hidden_size` and `in_features > hidden_size`
- Works with quantized and LoRA-adapted models
- Skips attention projection layers (where `in_features == out_features`)

**Template Structure**:

```json
{
  "components": {
    "0": {
      "layer_0": [123, 456, 789],  // Channel indices for component 0, layer 0
      "layer_1": [234, 567, 890]
    },
    "1": { ... }
  },
  "metadata": {
    "num_components": 10,
    "percentile": 98.0,
    "model": "meta-llama/Llama-3.2-1B-Instruct"
  }
}
```

**Training Flow with ICA Masking**:

1. Load pre-trained model
2. Apply LoRA adapters (if using PEFT)
3. Load or compute ICA templates
4. Register forward hooks on MLP layers
5. Apply masks before each forward pass
6. Train (masked neurons receive no gradient)
7. Remove hooks and save model

### Project Structure

```
src/functionalnetworkssft/
├── fnsft_trainer.py          # Main training script (CLI: fnsft)
├── ica_mask.py                # ICA masking implementation
├── build_ica_templates.py     # Template building tool (CLI: buildtemplates)
├── cli_gguf.py                # GGUF conversion utility
└── utils/
    ├── config_defaults.py     # Default configuration values
    ├── dataset_utils.py       # Dataset format detection/conversion
    ├── hf_utilities.py        # HuggingFace authentication helpers
    ├── inference.py           # Model inference utilities
    ├── model_utils.py         # Model loading and device detection
    ├── platform_setup.py      # Cross-platform setup detection
    └── post_processing.py     # Post-training processing

experiments/continual_learning/  # Continual learning experiments
├── configs/                     # YAML experiment configurations
│   ├── llama_7b.yaml           # LLaMA-7B (paper)
│   ├── llama_13b.yaml          # LLaMA-13B (paper)
│   └── llama_3_2_1b.yaml      # LLaMA-3.2-1B (development)
├── scripts/                     # Helper scripts
└── results/                     # Output directory

tests/                         # Comprehensive test suite
├── test_ica_masking.py       # Core ICA functionality tests
├── test_ica_mask_application.py  # Mask application tests
├── test_ica_cli_integration.py   # CLI integration tests
├── test_installation.py      # Platform verification
└── test_ica_suite.py         # Comprehensive test suite
```

### Configuration System

**Configuration priority** (highest to lowest):

1. Command-line arguments
2. YAML config file (`--config path/to/config.yaml`)
3. Environment variables (e.g., `HF_TOKEN`)
4. Default values from `ConfigDefaults`

**Key configuration parameters**:

- **Model**: `model_name_or_path`, `torch_dtype`, `use_8bit`, `use_4bit`
- **Dataset**: `dataset_name_or_path`, `validation_split`, `max_seq_length`
- **PEFT**: `lora_r`, `lora_alpha`, `lora_dropout`, `lora_target_modules`
- **ICA**: `mask_mode`, `ica_template_path`, `ica_component_ids`, `ica_components`, `ica_percentile`
- **Training**: `num_train_epochs`, `learning_rate`, `per_device_train_batch_size`, `gradient_accumulation_steps`
- **Monitoring**: `use_wandb`, `wandb_project`, `push_to_hub`, `hub_repo_id`

### Authentication

HuggingFace authentication precedence:

1. `--hub_token` CLI parameter (for hub uploads)
2. `HF_TOKEN` environment variable (from `.env` or shell)
3. Cached credentials from `huggingface-cli login`
4. Interactive prompt

Verify authentication: `poetry run check-hf-token`

## Platform-Specific Notes

### CUDA (NVIDIA GPUs)

- Supports 4-bit/8-bit quantization via BitsAndBytes
- Flash Attention available when compatible
- Run `poetry run python scripts/setup_cuda.py` to install CUDA wheels
- Test: `poetry run python tests/test_cuda_configuration.py`

### Apple Silicon (MPS)

- MPS backend enabled automatically
- No quantization support (BitsAndBytes incompatible)
- Use `torch_dtype: float32` for compatibility
- Flash Attention not available

### CPU-Only

- Fallback for systems without GPU
- No quantization or acceleration
- Slower training, suitable for small models

## Development Workflow

### Adding New Features

1. **Modifying ICA masking**: Edit `ica_mask.py`
   - `ICAMask.compute_global_networks()`: ICA computation
   - `ICAMask.apply_component_masks()`: Hook registration
   - Test with `pytest tests/test_ica_masking.py`

2. **Adding dataset formats**: Edit `utils/dataset_utils.py`
   - Add mapping to `DatasetFormatter.FORMAT_MAPPINGS`
   - Update `detect_format()` if needed
   - Test with `pytest tests/test_dataset_formatter.py`

3. **Modifying training logic**: Edit `fnsft_trainer.py`
   - Training arguments in `SFTArguments` dataclass
   - Main training loop in `main()` function
   - Hook integration at model loading stage

### Debugging

- **Enable verbose logging**: Set `SFT_LOG_LEVEL=DEBUG` environment variable
- **Inspect ICA templates**: Templates are JSON files in `ica_templates/`
- **Check hook placement**: Set breakpoint in `ica_mask.py:apply_component_masks()`
- **Verify masking**: Check `test_ica_mask_application.py` for examples
- **Device issues**: Use `get_optimal_device()` to verify device detection

### Common Pitfalls

1. **Missing ICA template**: Must build templates before training with ICA masking
2. **Wrong target modules**: `lora_target_modules` must match architecture (use `["down_proj", "dense_4h_to_h"]` for most models)
3. **Quantization on MPS**: BitsAndBytes not supported on Apple Silicon - use `use_8bit: false` and `use_4bit: false`
4. **Chat template errors**: Use `template_format: auto` to auto-detect model's chat template
5. **Authentication failures**: Ensure `.env` file has `HF_TOKEN=hf_your_token_here` (no quotes)

## Research Context

Based on "Brain-Inspired Exploration of Functional Networks and Key Neurons in Large Language Models" (Liu et al., 2025). Key findings:

- Masking <2% of neurons (key networks) severely degrades performance
- Preserving ~10% of neurons (key networks) retains near-baseline capability
- Functional networks identified via ICA show recurring patterns across inputs

This codebase extends those findings to supervised fine-tuning, aiming to preserve pre-trained knowledge by selectively training functional networks rather than all parameters.
