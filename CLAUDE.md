# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FunctionalNetworksSFT is a research framework for brain-inspired selective fine-tuning of Large Language Models. It uses Independent Component Analysis (ICA) to identify functional neuron networks within transformer models, then selectively trains or freezes these networks to mitigate catastrophic forgetting while learning new tasks.

**Core Innovation**: Rather than updating all model weights during fine-tuning, this framework identifies coherent functional networks of neurons (analogous to brain networks) and applies binary masks to selectively train/freeze them during Supervised Fine-Tuning (SFT).

## Common Development Commands

### Installation and Setup

```bash
# CUDA-enabled systems (NVIDIA GPUs) - recommended approach
poetry install
poetry run python scripts/setup_cuda.py

# Apple Silicon (M1/M2/M3/M4)
poetry install --extras apple-silicon

# CPU-only systems
poetry install --extras cpu

# Verify installation
poetry run python tests/test_cuda_configuration.py
```

### Running Tests

```bash
# Run all tests
poetry run pytest

# Run specific test file
poetry run pytest tests/test_ica_masking.py

# Run ICA test suite
poetry run pytest tests/test_ica_suite.py

# Run with verbose output
poetry run pytest -v

# Run specific test function
poetry run pytest tests/test_ica_masking.py::test_component_mask_application
```

### Training Commands

```bash
# Basic training with PEFT (LoRA)
poetry run fnsft \
    --model_name_or_path meta-llama/Llama-3.2-1B-Instruct \
    --dataset_name_or_path tatsu-lab/alpaca \
    --output_dir ./output \
    --num_train_epochs 2 \
    --use_auth_token

# Training with YAML configuration
poetry run fnsft --config experiments/peft_vs_peft-ica/experiment_a_peft_only/config/experiment_a_config.yaml

# Training with ICA masking (lesion mode)
poetry run fnsft \
    --config experiment_config.yaml \
    --mask_mode lesion \
    --ica_component_ids [0,1,2]

# Training with ICA masking (preserve mode) and anti-drift
poetry run fnsft \
    --config experiment_config.yaml \
    --mask_mode preserve \
    --ica_component_ids [0] \
    --anti_drift_row_param true \
    --anti_drift_apply_to both
```

### ICA Template Building

```bash
# Build ICA templates (positional arguments - recommended)
poetry run buildtemplates \
    meta-llama/Llama-3.2-1B-Instruct \
    tatsu-lab/alpaca databricks/databricks-dolly-15k

# Build templates with custom parameters
poetry run buildtemplates \
    microsoft/DialoGPT-medium \
    dataset1.json dataset2.jsonl \
    --ica_template_samples_per_ds 500 \
    --ica_components 20 \
    --ica_percentile 95.0 \
    --ica_template_output ./custom_templates/

# Build templates using named arguments (alternative syntax)
poetry run buildtemplates \
    --model_name_or_path meta-llama/Llama-3.2-1B-Instruct \
    --ica_build_templates_from tatsu-lab/alpaca
```

### Running Experiments

```bash
# Run individual experiment
poetry run python experiments/peft_vs_peft-ica/experiment_a_peft_only/scripts/run_experiment_a.py

# Run all experiments sequentially
poetry run python experiments/peft_vs_peft-ica/run_experiments.py

# Evaluate trained models
poetry run python experiments/peft_vs_peft-ica/evaluate_models.py
```

### Utility Commands

```bash
# Check HuggingFace token configuration
poetry run check-hf-token

# Convert model to GGUF format
poetry run convert-gguf --model_path ./output/final_model --output_path ./output.gguf

# Check device configuration
poetry run python -c "from functionalnetworkssft.utils.model_utils import get_optimal_device; print(get_optimal_device())"
```

## Architecture Overview

### Core Components

**1. Training Orchestration (`fnsft_trainer.py`)**
- Main entry point for training via `fnsft` CLI command
- Handles YAML configuration loading and CLI argument parsing
- Orchestrates model loading, dataset preparation, PEFT setup, and ICA masking
- Integrates with HuggingFace Trainer for the training loop
- Supports model merging, GGUF conversion, and Hub uploads

**2. ICA Masking System (`ica_mask.py`)**
- `ICAMask` class: Computes ICA decomposition of MLP neuron activations
- Generates component-wise binary masks identifying functional networks
- Applies forward hooks to zero out masked neurons during forward pass
- Implements anti-drift row parametrization to prevent optimizer momentum drift on frozen parameters
- Supports two masking modes:
  - **lesion**: Zero out selected functional networks (ablate them)
  - **preserve**: Zero out everything except selected networks (isolate them)

**3. Template Builder (`build_ica_templates.py`)**
- Standalone tool for pre-computing ICA templates from datasets
- Aggregates samples across multiple datasets
- Saves reusable component masks as JSON templates
- Supports flexible argument parsing (positional or named)

**4. Dataset Utilities (`utils/dataset_utils.py`)**
- `DatasetFormatter`: Auto-detects and converts 17+ dataset formats
- `InstructionDataset`: Handles template-based text formatting with chat template support
- Supports standard formats: `(instruction, response)`, `(question, answer)`, `(messages,)`, etc.

**5. Model Utilities (`utils/model_utils.py`)**
- Device detection with caching: CUDA → MPS → CPU
- Platform-optimized dtype selection (bf16/fp16/fp32)
- Quantization config generation (4-bit/8-bit with BitsAndBytes)
- LoRA setup with architecture-specific target module auto-detection
- PEFT-aware model saving, merging, and GGUF conversion

### ICA Masking Integration Points

**Discovery Phase (Pre-training or Template Building):**
1. Hook MLP outputs across all transformer layers
2. Collect activations: `[batch, seq_len, hidden_size]` per layer
3. Concatenate across layers: `[time, num_layers * hidden_size]`
4. Apply PCA if matrix exceeds LAPACK limits (2^31-1 elements)
5. Run FastICA to extract independent components
6. Threshold component weights by percentile to identify key neurons
7. Map flat indices back to `(layer_idx, channel_idx)` pairs
8. Save as template: `{component_id: {layer: [channel_indices]}}`

**Application Phase (Training):**
1. Load pre-computed templates OR compute on-the-fly
2. Select components via `--ica_component_ids`
3. Generate binary masks per layer based on mode:
   - **lesion**: `mask[selected_channels] = 0`, rest = 1
   - **preserve**: `mask[selected_channels] = 1`, rest = 0
4. Register forward hooks on MLP modules: `output = output * mask`
5. Optional: Apply row parametrization for anti-drift:
   - Wrap trainable weights: `W_eff = frozen + row_mask * delta`
   - Frozen rows receive no gradient; trainable rows update normally

**Hook Placement:**
- Target: Second linear projection in each MLP (e.g., `down_proj`, `c_proj`)
- Condition: `out_features == hidden_size` and `in_features > hidden_size`
- Hook type: `forward_hook` to intercept and multiply output by mask

### PEFT Integration

**LoRA Target Detection:**
- Auto-detects modules by architecture: `q_proj`, `v_proj`, `k_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`
- Architecture-specific mappings for GPT-2, Llama, Mistral, Qwen, OPT, etc.
- Fallback: All `nn.Linear` layers if architecture unknown

**Anti-Drift Parametrization:**
- Applies to LoRA adapter weights (`lora_B.weight`) and/or base weights (`down_proj.weight`)
- Modes via `--anti_drift_apply_to`: `"lora"`, `"base"`, `"both"`, `"auto"`
- Prevents optimizer momentum from corrupting frozen parameters
- Cleanup: Unwrap parametrizations after training with optional baking

### Configuration System

**Priority Hierarchy (highest to lowest):**
1. CLI arguments (explicit command-line flags)
2. YAML config file (`--config experiment_config.yaml`)
3. Dataclass defaults

**Key Configuration Files:**
- `experiments/peft_vs_peft-ica/experiment_a_peft_only/config/experiment_a_config.yaml`: PEFT-only baseline
- `experiments/peft_vs_peft-ica/experiment_b_peft_ica/config/experiment_b_config.yaml`: PEFT + ICA lesion
- `experiments/peft_vs_peft-ica/experiment_c_peft_ica_preserve/config/experiment_c_config.yaml`: PEFT + ICA preserve

**Critical Parameters:**
- `mask_mode`: `null` (no masking), `"lesion"` (ablate networks), `"preserve"` (isolate networks)
- `ica_component_ids`: List of component indices to mask (e.g., `[0, 1, 2]`)
- `ica_template_path`: Path to pre-computed template JSON
- `ica_layers`: Layer specification (e.g., `"0-5"`, `"all"`, `[0, 2, 4]`)
- `anti_drift_row_param`: Enable row parametrization (default: `false`)
- `anti_drift_apply_to`: Where to apply parametrization (`"auto"`, `"lora"`, `"base"`, `"both"`)

### Dataset Format Auto-Detection

**Supported Formats (17+):**
- `(instruction, response)`, `(instruction, output)`
- `(instruction, context, response)`, `(instruction, input, output)`
- `(question, answer)`, `(question, response)`
- `(prompt, completion)`, `(prompt, response)`
- `(messages,)` - conversational format
- `(system, user, assistant)` - role-based
- `(message_1, message_2)` - custom formats (e.g., camel-ai/physics)

**Detection Strategy:**
1. Priority-ordered format matching by common keys
2. Per-item fallback for heterogeneous datasets
3. Auto-conversion to standard `{instruction, response}` or `{text}` format

**Template Formats:**
- `auto`: Uses tokenizer's chat template if available, else basic
- `chat`: Forces tokenizer's `apply_chat_template()`
- `alpaca`: `### Instruction:\n{instruction}\n\n### Response:\n{response}`
- `chatml`: `<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>`
- `basic`: Custom template string

### Platform-Specific Considerations

**CUDA (NVIDIA GPUs):**
- Use `poetry run python scripts/setup_cuda.py` for installation
- Enables: 4-bit/8-bit quantization, flash attention, bf16/fp16 mixed precision
- Recommended dtype: `"auto"` (selects bf16 if supported, else fp16)

**Apple Silicon (MPS):**
- Use `poetry install --extras apple-silicon`
- Force `torch_dtype: "float32"` in configs (MPS has limited fp16 support)
- Quantization NOT available (BitsAndBytes incompatible)

**CPU-Only:**
- Use `poetry install --extras cpu`
- Always uses fp32, no quantization, slower training

### Testing Strategy

**Test Categories:**
1. **ICA Functionality**: Core masking logic, hook behavior, layer parsing
2. **Dataset Handling**: Format detection, split management, preprocessing
3. **Template Management**: Save/load/match operations
4. **Training Integration**: PEFT vs full training, gradient flow validation
5. **Platform Support**: Device detection, CUDA configuration
6. **Configuration**: YAML loading, attribute access, default handling

**Running Specific Test Suites:**
```bash
# ICA tests
poetry run pytest tests/test_ica_*.py

# Dataset tests
poetry run pytest tests/test_dataset*.py

# Integration tests
poetry run pytest tests/test_*_integration.py
```

## Key Workflows

### Workflow 1: Standard PEFT Fine-Tuning (No ICA Masking)

1. Prepare YAML config with `mask_mode: null`
2. Run: `poetry run fnsft --config experiment_a_config.yaml`
3. Model trains all LoRA adapters normally
4. Optional: Merge adapter with base and upload to Hub

### Workflow 2: ICA-Masked Fine-Tuning (Lesion Mode)

1. Build ICA templates: `poetry run buildtemplates model_name dataset1 dataset2`
2. Prepare YAML config with:
   - `mask_mode: "lesion"`
   - `ica_template_path: "./ica_templates/global_templates.json"`
   - `ica_component_ids: [0, 1]`
3. Run: `poetry run fnsft --config experiment_b_config.yaml`
4. Selected functional networks are ablated (zeroed out) during training
5. Model learns with reduced capacity, testing network importance

### Workflow 3: ICA-Masked Fine-Tuning (Preserve Mode with Anti-Drift)

1. Build ICA templates (if not already done)
2. Prepare YAML config with:
   - `mask_mode: "preserve"`
   - `ica_component_ids: [0]`
   - `anti_drift_row_param: true`
   - `anti_drift_apply_to: "both"`
3. Run: `poetry run fnsft --config experiment_c_config.yaml`
4. Only selected networks are preserved (rest ablated)
5. Anti-drift prevents optimizer momentum corruption on frozen parameters
6. Model learns using isolated functional subnetwork

### Workflow 4: Multi-Experiment Comparison

1. Configure three experiments (A: PEFT-only, B: PEFT+ICA lesion, C: PEFT+ICA preserve)
2. Run: `poetry run python experiments/peft_vs_peft-ica/run_experiments.py`
3. Evaluate: `poetry run python experiments/peft_vs_peft-ica/evaluate_models.py`
4. Compare metrics: BLEU, ROUGE, perplexity, task-specific scores

## Important Implementation Details

### Forward Hook Mechanism

Hooks are registered on the **second linear projection** of each MLP:
```python
# Example for Llama: mlp.down_proj
# Example for GPT-2: mlp.c_proj
def fwd_hook(_mod, _inp, out, mask_tensor=mask):
    m = mask_tensor.to(device=out.device, dtype=out.dtype)
    return out * m  # Element-wise multiplication
```

**Why target this layer?**
- MLP structure: `hidden → intermediate (expand) → hidden (project)`
- Intermediate layer is where functional neurons exist (4x hidden size typically)
- Masking input to down projection = zeroing neuron outputs post-activation

### Anti-Drift Row Parametrization

**Problem:** Adam optimizer maintains momentum for all parameters, causing frozen rows to drift even with zero gradient.

**Solution:** Replace weight tensor with parametrization:
```python
W_effective = frozen_baseline + row_mask * trainable_delta
```

**Effect:**
- Frozen rows: `row_mask[i] = 0` → `W[i] = frozen[i]` (no gradient, no drift)
- Trainable rows: `row_mask[i] = 1` → `W[i] = frozen[i] + delta[i]` (gradient flows normally)

**Cleanup:**
- **Bake mode** (`bake=True`): Set weight to effective value, remove parametrization
- **Unbake mode** (`bake=False`): Restore original weight, discard delta

### Pre-Tokenization Caching

Dataset is pre-tokenized and cached to disk (default: `{output_dir}/tokenized_cache/`):
- Speeds up training startup on subsequent runs
- Cache invalidated if dataset/model/template changes
- Disable with: CLI does not expose flag, edit code if needed

### Authentication Precedence

**For Model Loading:**
1. `HF_TOKEN` environment variable (from `.env` file or shell)
2. Cached credentials from `huggingface-cli login`
3. No authentication (may fail for gated models)

**For Hub Uploads:**
1. `--hub_token` CLI parameter (highest priority)
2. `HF_TOKEN` environment variable
3. Cached credentials
4. Interactive login prompt

## Common Debugging Scenarios

### Issue: CUDA Out of Memory
- Reduce `per_device_train_batch_size` and/or `max_seq_length`
- Enable `gradient_checkpointing: true`
- Use quantization: `use_4bit: true` or `use_8bit: true`
- Reduce `lora_r` (e.g., from 16 to 8)

### Issue: Tests Failing with "LAPACK error"
- Matrix too large for LAPACK SVD solver in ICA
- Reduce `ica_components` or increase `ica_pca_preprocessing_components`
- Framework auto-applies PCA when matrix exceeds 2^31-1 elements

### Issue: MPS (Apple Silicon) Training Errors
- Ensure `torch_dtype: "float32"` in config (MPS has limited fp16 support)
- Disable quantization (not supported on MPS)
- Avoid flash attention (not available on MPS)

### Issue: Anti-Drift Not Working
- Verify `anti_drift_row_param: true` in config
- Check callback logs for drift detection
- Ensure correct `anti_drift_apply_to` setting matches your training mode
- If using quantized base model, set `anti_drift_apply_to: "lora"` (parametrization doesn't work on quantized weights)

### Issue: ICA Template Loading Fails
- Verify template path is correct and file exists
- Check template was built for same model architecture
- Ensure `ica_component_ids` indices exist in template (check component count in template file)

## File Structure Reference

```
src/functionalnetworkssft/
├── fnsft_trainer.py          # Main training CLI (1857 lines)
├── ica_mask.py               # ICA masking system (975 lines)
├── build_ica_templates.py    # Template builder CLI (558 lines)
├── cli_gguf.py               # GGUF conversion utility
└── utils/
    ├── model_utils.py        # Model/device/PEFT utilities (717 lines)
    ├── dataset_utils.py      # Dataset format detection (511 lines)
    ├── config_defaults.py    # Configuration defaults
    └── hf_utilities.py       # HuggingFace Hub utilities

experiments/peft_vs_peft-ica/
├── experiment_a_peft_only/   # Baseline PEFT experiment
├── experiment_b_peft_ica/    # PEFT + ICA lesion
├── experiment_c_peft_ica_preserve/  # PEFT + ICA preserve
├── run_experiments.py        # Sequential experiment runner
└── evaluate_models.py        # Model evaluation and comparison

tests/
├── test_ica_*.py            # ICA functionality tests
├── test_dataset*.py         # Dataset handling tests
├── test_*_integration.py    # Integration tests
└── test_cuda_configuration.py  # Platform setup verification
```

## Research Context

This framework operationalizes research from "Brain-Inspired Exploration of Functional Networks and Key Neurons in Large Language Models" (Liu et al., 2025). Key findings that informed this implementation:

1. **Functional Networks Exist**: LLMs contain recurring functional networks analogous to brain networks
2. **Key Networks are Sparse**: ~2% of neurons (key networks) can severely degrade performance if masked
3. **Preservation Efficiency**: ~10% of neurons (key networks) can retain near-baseline capability
4. **ICA Reveals Structure**: Independent Component Analysis on MLP activations reliably identifies these networks

This codebase enables empirical validation of whether selective fine-tuning of functional networks can mitigate catastrophic forgetting while maintaining learning capacity.
