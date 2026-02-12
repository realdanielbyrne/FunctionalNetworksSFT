# FunctionalNetworksSFT - AI Agent Instructions

## Project Overview

FunctionalNetworksSFT implements **brain-inspired selective fine-tuning** for LLMs using Independent Component Analysis (ICA) to identify and manipulate functional neuron networks. The system enables selective training/freezing of these networks to mitigate catastrophic forgetting during supervised fine-tuning.

**Core Innovation**: Instead of updating all weights during fine-tuning, ICA identifies coherent functional networks (groups of co-activating neurons), then applies binary masks to selectively lesion (ablate) or preserve (train only) these networks.

## Architecture Essentials

### The ICA Masking Pipeline

1. **Template Building** ([build_ica_templates.py](../src/functionalnetworkssft/build_ica_templates.py)): Pre-compute ICA components
   - Samples data, captures MLP activations across all layers
   - Runs PCA (dimensionality reduction) → FastICA → threshold selection
   - Outputs JSON templates: `{"components": {"0": {"layer_0": [channel_indices]}}, "metadata": {...}}`

2. **Hook-Based Masking** ([ica_mask.py](../src/functionalnetworkssft/ica_mask.py)): Apply masks during training
   - Registers PyTorch forward hooks on **MLP down-projection layers** (e.g., `down_proj`, `c_proj`)
   - Hook placement: Targets layers where `out_features == hidden_size` and `in_features > hidden_size`
   - Binary masks (0/1) zero out specific channels AFTER activation, BEFORE residual connection
   - Two modes: `lesion` (zero out networks) or `preserve` (train only selected networks)

3. **Training Integration** ([fnsft_trainer.py](../src/functionalnetworkssft/fnsft_trainer.py)): Combine with PEFT
   - Wraps HuggingFace Trainer with ICA masking support
   - Compatible with LoRA/QLoRA (masks and LoRA adapters coexist)
   - Masked neurons receive zero gradient → weights don't update

### Critical Implementation Detail

**Why down-projection layers?** Masks are applied to the OUTPUT of MLP activation (not the input). This targets the intermediate representation bottleneck where functional specialization emerges. Specifically:
```python
# Hook intercepts: intermediate_output = activation(up_proj(x))
# Before it reaches: output = down_proj(intermediate_output) + residual
mask = mask.view(1, 1, -1)  # Broadcast across (batch, seq_len, channels)
return input_tensor * mask  # Zero out masked channels
```

## Platform-Specific Workflows

### Setup Commands (CRITICAL - Always Use Poetry)

```bash
# CUDA (NVIDIA GPUs) - TWO-STEP PROCESS:
poetry install                              # Install base deps
poetry run python scripts/setup_cuda.py     # Install CUDA wheels + extras

# Apple Silicon (M1/M2/M3/M4):
poetry install --extras apple-silicon       # MPS-optimized PyTorch

# CPU-only:
poetry install --extras cpu
```

**Never** run `python scripts/setup_cuda.py` directly (installs globally). Always use `poetry run`.

### Platform Constraints

| Feature | CUDA | MPS (Apple Silicon) | CPU |
|---------|------|---------------------|-----|
| Quantization (4-bit/8-bit) | ✅ | ❌ | ❌ |
| Flash Attention | ✅ | ❌ | ❌ |
| Mixed Precision (fp16/bf16) | ✅ | ✅ (fp16 only) | ❌ |

**Apple Silicon workaround**: Use `torch_dtype: float32` or `float16`. Disable `use_4bit` and `use_8bit`.

## Development Workflows

### Building ICA Templates (Required Before Training)

```bash
# Positional syntax (recommended):
poetry run buildtemplates meta-llama/Llama-3.2-1B-Instruct tatsu-lab/alpaca

# Named arguments (alternative):
poetry run buildtemplates \
    --model_name_or_path meta-llama/Llama-3.2-1B-Instruct \
    --ica_build_templates_from tatsu-lab/alpaca camel-ai/physics \
    --ica_components 15 \
    --ica_percentile 95.0
```

**Output**: `ica_templates/global_templates.json` (or model/dataset-specific paths)

### Training with ICA Masking

```bash
# Lesion mode (zero out components 0,1 - ablate networks):
poetry run fnsft \
    --model_name_or_path meta-llama/Llama-3.2-1B-Instruct \
    --dataset_name_or_path your_dataset.json \
    --mask_mode lesion \
    --ica_template_path ./ica_templates/global_templates.json \
    --ica_component_ids [0,1] \
    --lora_target_modules ["down_proj"]  # REQUIRED for ICA compatibility

# Preserve mode (train ONLY components 0,1):
poetry run fnsft \
    --model_name_or_path meta-llama/Llama-3.2-1B-Instruct \
    --dataset_name_or_path your_dataset.json \
    --mask_mode preserve \
    --ica_component_ids [0,1]
```

**YAML config alternative**: `poetry run fnsft --config path/to/config.yaml`

### Running Experiments

Located in [experiments/peft_vs_peft-ica](../experiments/peft_vs_peft-ica):
- **Centralized config**: `common_config.yaml` contains shared parameters
- **Python overrides**: `run_experiments.py` programmatically defines experiment-specific settings
- **No duplicate configs**: Prevents drift across experiments

```bash
cd experiments/peft_vs_peft-ica
poetry run python run_experiments.py              # Run all (A, B, C)
poetry run python run_experiments.py --experiment b  # Run single experiment
poetry run python evaluate_models.py              # Compare results
```

## Project-Specific Conventions

### Configuration Hierarchy (Highest → Lowest Priority)

1. CLI arguments: `--learning_rate 1e-4`
2. YAML config: `config.yaml` loaded via `--config`
3. Environment variables: `HF_TOKEN`, `SFT_LOG_LEVEL`
4. Defaults: [ConfigDefaults](../src/functionalnetworkssft/utils/config_defaults.py)

### Authentication Flow

HuggingFace token resolution (checked in order):
1. `--hub_token` CLI parameter
2. `HF_TOKEN` environment variable (from `.env` or shell)
3. Cached `~/.huggingface/token` from `huggingface-cli login`
4. Interactive prompt

Verify: `poetry run check-hf-token`

### Dataset Format Auto-Detection

[DatasetFormatter](../src/functionalnetworkssft/utils/dataset_utils.py) handles diverse formats:
- Alpaca: `{instruction, input, output}`
- ShareGPT: `{conversations: [{from, value}]}`
- Dolly: `{instruction, context, response}`
- Simple: `{instruction, response}` or `{input, output}`

**Chat template priority**: `--template_format auto` → Uses model's built-in tokenizer template

### LoRA Target Module Selection

**Standard PEFT** (no ICA): Target all linear layers for max expressiveness
```python
lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"]
```

**ICA-compatible PEFT**: Target ONLY down-projection for hook compatibility
```python
lora_target_modules = ["down_proj"]  # Hooks are placed here
```

Mixing other modules with ICA is unsupported (hooks would conflict).

## Testing Strategy

### Test Organization

- **Unit tests**: [test_ica_masking.py](../tests/test_ica_masking.py), [test_ica_mask_application.py](../tests/test_ica_mask_application.py)
- **Integration tests**: [test_ica_cli_integration.py](../tests/test_ica_cli_integration.py)
- **System tests**: [test_installation.py](../tests/test_installation.py), [test_ica_suite.py](../tests/test_ica_suite.py)

```bash
poetry run pytest                           # All tests
poetry run pytest tests/test_ica_masking.py # Specific file
poetry run pytest -k "ica_mask"             # Pattern matching
poetry run pytest -v --cov=src/functionalnetworkssft  # With coverage
```

### Platform Verification

```bash
# Quick device check:
poetry run python -c "from functionalnetworkssft.utils.model_utils import get_optimal_device; print(get_optimal_device())"

# Comprehensive verification:
python tests/test_installation.py  # Note: Direct python (no poetry run)
```

## Common Pitfalls & Solutions

| Issue | Solution |
|-------|----------|
| **"No ICA template found"** | Run `buildtemplates` before training with ICA masking |
| **Quantization on MPS fails** | Set `use_4bit: false` and `use_8bit: false` in config |
| **Wrong target modules** | Use `lora_target_modules: ["down_proj"]` for ICA mode |
| **Chat template errors** | Use `template_format: auto` to auto-detect model template |
| **Authentication failures** | Create `.env` with `HF_TOKEN=hf_xxx` (no quotes) |
| **Hook placement errors** | Verify model architecture - some models use different layer names |

## Debugging Techniques

### Enable Verbose Logging
```bash
export SFT_LOG_LEVEL=DEBUG
poetry run fnsft ...
```

### Inspect ICA Templates
Templates are human-readable JSON:
```bash
cat ica_templates/global_templates.json | jq '.metadata'
```

### Verify Hook Placement
Set breakpoint at `ica_mask.py:apply_component_masks()` and inspect:
```python
# Check identified layers:
print(f"Applying hooks to: {[(n, m.__class__.__name__) for n, m in target_layers]}")
```

### Check Mask Application
Run [test_ica_mask_application.py](../tests/test_ica_mask_application.py) with modified component IDs to verify masking logic.

## Key Files for Modification

- **Add ICA features**: [ica_mask.py](../src/functionalnetworkssft/ica_mask.py) → Modify `compute_global_networks()` or hook logic
- **Add dataset formats**: [dataset_utils.py](../src/functionalnetworkssft/utils/dataset_utils.py) → Update `FORMAT_MAPPINGS`
- **Modify training logic**: [fnsft_trainer.py](../src/functionalnetworkssft/fnsft_trainer.py) → Edit `SFTArguments` or training loop
- **Add continual learning methods**: [continual_learning/methods/](../src/functionalnetworkssft/continual_learning/methods/) → Inherit from base class

## Research Context

Based on ["Brain-Inspired Exploration of Functional Networks and Key Neurons in Large Language Models"](https://arxiv.org/abs/2501.09803) (Liu et al., 2025):
- Masking <2% of neurons (key networks) → severe performance degradation
- Preserving ~10% of neurons (key networks) → near-baseline performance
- This codebase extends those findings to **training time** rather than just inference

Continual learning evaluation framework compares ICA masking against baselines (LoRA, EWC, LwF, O-LoRA, DOC) using Average Accuracy (AA), Backward Transfer (BWT), and Forward Transfer (FWT) metrics.
