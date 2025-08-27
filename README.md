# Functional Networks SFT

**FunctionalNetworksSFT** is a research-oriented framework for fine-tuning large language models using ICA-based functional network masking. This innovative approach leverages neuroscience insights about functional brain networks to selectively train or ablate specific neuron groups during supervised fine-tuning (SFT).

## Core Innovation

The framework implements **functional network masking** - a technique that applies binary masks to specific neurons during training based on Independent Component Analysis (ICA) of neuron activations. This allows researchers to:

- **Ablate key networks**: Mask important functional networks to study their role in model performance
- **Isolate networks**: Train only specific functional networks while masking all others
- **Targeted fine-tuning**: Update weights only in functionally relevant neurons, potentially reducing negative effects of full parameter fine-tuning

## Key Capabilities

- **ICA-Based Network Discovery**: Automatically identifies functional networks using FastICA analysis of MLP activations
- **Flexible Masking Modes**: Support for both ablation (`key`) and isolation (`complement`) masking strategies
- **Pre-computed or On-the-fly ICA**: Use existing ICA masks from JSON files or compute them dynamically during training
- **Hugging Face Integration**: Seamless compatibility with transformers, LoRA/QLoRA, and quantization techniques
- **Cross-Platform Optimization**: Native support for CUDA, Apple Silicon (MPS), and CPU-only environments

## Features

- 🚀 **Cross-Platform Support**: Works on CUDA-enabled systems and Apple Silicon Macs
- ⚡ **Hardware Acceleration**: Automatic detection and optimization for CUDA and MPS backends
- 🔧 **Quantization Support**: BitsAndBytes 4-bit/8-bit quantization on CUDA systems
- 🎯 **LoRA/QLoRA**: Parameter-efficient fine-tuning with automatic target module detection
- 📊 **Experiment Tracking**: Built-in Weights & Biases integration
- 🔄 **Flexible Data Formats**: Automatic detection and conversion of various dataset formats
- 🛡️ **Robust Error Handling**: Graceful fallbacks for platform-specific features

## Quick Start

### Installation

#### Prerequisites

- **Python 3.12+**
- **Poetry** for dependency management
- **Git** for cloning the repository

#### 1. Clone the Repository

```bash
git clone <repository-url>
cd FunctionalNetworksSFT
```

#### 2. Hardware-Specific Installation

Choose the installation method that matches your hardware configuration:

##### 🚀 CUDA-Enabled Systems (NVIDIA GPUs)

**Recommended for:** Windows/Linux systems with NVIDIA GPUs

```bash
# Step 1: Clone and navigate to the repository
git clone <repository-url>
cd FunctionalNetworksSFT

# Step 2: Install base dependencies (Poetry creates virtual environment automatically)
poetry install

# Step 3: Run automated CUDA setup (recommended)
poetry run python scripts/setup_cuda.py
```

**Alternative manual installation:**

```bash
# Steps 1-2 same as above, then:
# Install CUDA wheels explicitly (prevents CPU fallback)
poetry run pip uninstall -y torch torchvision torchaudio
poetry run pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
# Then install CUDA extras
poetry install --extras cuda
```

> **Important:** Always use `poetry run` or `poetry shell` to ensure commands run within Poetry's virtual environment. Never run `python scripts/setup_cuda.py` directly as this would install packages globally.

**Features enabled:**

- ✅ GPU acceleration with CUDA
- ✅ 4-bit/8-bit quantization with BitsAndBytes
- ✅ Flash Attention (when compatible)
- ✅ Automatic mixed precision training

**Verify installation:**

```bash
poetry run python tests/test_cuda_configuration.py
```

##### 🍎 Apple Silicon (M1/M2/M3/M4 Macs)

**Recommended for:** macOS systems with Apple Silicon processors

```bash
# Clone and navigate to the repository
git clone <repository-url>
cd FunctionalNetworksSFT

# Poetry automatically creates and manages the virtual environment
# Install with Apple Silicon optimizations
poetry install --extras apple-silicon
```

**Features enabled:**

- ✅ GPU acceleration with Metal Performance Shaders (MPS)
- ✅ Optimized for Apple Silicon architecture
- ✅ Native ARM64 performance
- ❌ Quantization not available (BitsAndBytes incompatible)

**Verify installation:**

```bash
python -c "import torch; print('MPS available:', torch.backends.mps.is_available())"
```

##### 💻 CPU-Only Systems

**Recommended for:** Any system without GPU acceleration or as a fallback

```bash
# Clone and navigate to the repository
git clone <repository-url>
cd FunctionalNetworksSFT

# Poetry automatically creates and manages the virtual environment
# Install CPU-only version
poetry install --extras cpu
```

**Features enabled:**

- ✅ CPU-based training (slower but universal)
- ✅ Full precision training (fp32)
- ✅ Compatible with any hardware
- ❌ No GPU acceleration
- ❌ No quantization support

#### 3. Verify Your Installation

After installation, verify everything is working correctly:

```bash
# Check platform detection and recommendations
poetry run python src/functionalnetworkssft/utils/platform_setup.py

# Run comprehensive tests (CUDA systems only)
poetry run python tests/test_cuda_configuration.py

# Quick verification
poetry run python -c "
import torch
from functionalnetworkssft.utils.model_utils import get_optimal_device
device, name = get_optimal_device()
print(f'Using device: {name}')
print(f'PyTorch version: {torch.__version__}')
"
```

#### 4. Troubleshooting Installation Issues

**CUDA Issues:**

```bash
# Check NVIDIA GPU detection
nvidia-smi

# Reinstall with CUDA support
poetry run python scripts/setup_cuda.py

# Manual PyTorch CUDA installation
poetry run pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**Apple Silicon Issues:**

```bash
# Verify MPS availability
python -c "import torch; print('MPS:', torch.backends.mps.is_available())"

# Reinstall if needed
poetry install --extras apple-silicon --force
```

**General Issues:**

```bash
# Clean installation
poetry env remove python
poetry install --extras <your-platform>

# Check Poetry environment
poetry env info
```

#### Platform Comparison

| Platform | Installation Command | GPU Acceleration | Quantization | Recommended Use |
|----------|---------------------|------------------|--------------|-----------------|
| **NVIDIA GPU** | `python scripts/setup_cuda.py` | ✅ CUDA | ✅ BitsAndBytes | Production training, large models |
| **Apple Silicon** | `poetry install --extras apple-silicon` | ✅ MPS | ❌ Not available | Development, medium models |
| **CPU Only** | `poetry install --extras cpu` | ❌ CPU only | ❌ Not available | Testing, small models |

For detailed platform-specific instructions and troubleshooting, see [CROSS_PLATFORM_SETUP.md](CROSS_PLATFORM_SETUP.md).

### Authentication Setup

FunctionalNetworksSFT supports multiple methods for HuggingFace authentication, with automatic fallback handling:

#### Method 1: Environment File (Recommended for Development)

Create a `.env` file in your project root:

```bash
# .env file
HF_TOKEN=hf_your_token_here
```

#### Method 2: Environment Variable

```bash
export HF_TOKEN=hf_your_token_here
```

#### Method 3: Interactive Login

```bash
huggingface-cli login
```

#### Token Precedence Order

The framework uses the following precedence for authentication:

**For Model Loading:**

1. `HF_TOKEN` environment variable (including from `.env` file)
2. Cached credentials from `huggingface-cli login`
3. No authentication (may fail for gated models)

**For Hub Uploads:**

1. `--hub_token` CLI parameter (highest priority)
2. `HF_TOKEN` environment variable (including from `.env` file)
3. Cached credentials from `huggingface-cli login`
4. Interactive login prompt

### Basic Usage

```bash
poetry run fnsft \
    --model_name_or_path microsoft/DialoGPT-medium \
    --dataset_name_or_path your_dataset.json \
    --output_dir ./output \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --torch_dtype auto \
    --use_auth_token  # Enable authentication for gated models
```

### Example: Fine-tuning Llama-3.2-1B on Sarcasm Dataset

This example demonstrates fine-tuning the Llama-3.2-1B model on a sarcasm detection dataset using intelligent chat template handling:

```bash
poetry run fnsft \
    --model_name_or_path meta-llama/Llama-3.2-1B-Instruct \
    --dataset_name_or_path sarcasm.csv \
    --output_dir ./fine-tuned-model \
    --template_format auto \
    --num_train_epochs 2 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --learning_rate 1e-5 \
    --torch_dtype float32 \
    --validation_split 0.05 \
    --max_seq_length 128 \
    --eval_steps 40 \
    --logging_steps 40 \
    --save_steps 160 \
    --max_grad_norm 2 \
    --use_auth_token  # Required for gated models
```

### Authentication Examples

#### Example 1: Using .env File (Recommended)

```bash
# Create .env file with your token
echo "HF_TOKEN=hf_your_token_here" > .env

# Run training - token automatically loaded
poetry run fnsft \
    --model_name_or_path meta-llama/Llama-3.2-1B-Instruct \
    --dataset_name_or_path sarcasm.csv \
    --output_dir ./fine-tuned-model \
    --use_auth_token
```

#### Example 2: Hub Upload with Different Token

```bash
# Use one token for model loading, different token for upload
poetry run fnsft \
    --model_name_or_path meta-llama/Llama-3.2-1B-Instruct \
    --dataset_name_or_path sarcasm.csv \
    --output_dir ./fine-tuned-model \
    --use_auth_token \
    --push_to_hub \
    --hub_repo_id username/my-fine-tuned-model \
    --hub_token hf_different_upload_token
```

#### Example 3: Using Cached Credentials

```bash
# Login once (credentials cached)
huggingface-cli login

# Run training without specifying tokens
poetry run fnsft \
    --model_name_or_path meta-llama/Llama-3.2-1B-Instruct \
    --dataset_name_or_path sarcasm.csv \
    --output_dir ./fine-tuned-model \
    --use_auth_token
```

**Key features demonstrated:**

- **Automatic Chat Template Detection**: `--template_format auto` automatically detects and uses Llama's chat template
- **CSV Dataset Support**: Direct loading of CSV files with `question,answer` columns
- **MacBook Compatibility**: `--torch_dtype float32` ensures compatibility with Apple Silicon
- **Intelligent Data Splitting**: `--validation_split 0.05` automatically creates train/validation splits
- **Optimized for Small Models**: Batch sizes and sequence length optimized for 1B parameter models
- **Seamless Authentication**: Automatic token loading from `.env` file or cached credentials

### Troubleshooting Authentication

#### Common Issues and Solutions

**Issue: "401 Client Error: Unauthorized"**

```bash
# Solution 1: Check if token is loaded
python -c "import os; print('HF_TOKEN found:', bool(os.getenv('HF_TOKEN')))"

# Solution 2: Verify .env file format (no quotes, no export)
cat .env  # Should show: HF_TOKEN=hf_your_token_here

# Solution 3: Use interactive login as fallback
huggingface-cli login
```

**Issue: "Access to model is restricted"**

- Ensure you've requested access to the gated model at the HuggingFace model page
- Wait for approval (can take time for popular models)
- Verify your token has the correct permissions

**Issue: Hub upload fails with different error**

```bash
# Use specific token for uploads
fnsft --push_to_hub --hub_token hf_write_token ...
```

#### Token Security Best Practices

- **Never commit `.env` files** to version control (already in `.gitignore`)
- **Use read-only tokens** for model loading when possible
- **Use write tokens** only for hub uploads with `--hub_token`
- **Rotate tokens regularly** and update `.env` file accordingly

### Token Verification Tool

The framework includes a built-in tool to verify your HuggingFace token configuration:

```bash
# Check token configuration and access to gated models
check-hf-token
```

This command will:

- ✅ Load and verify your `.env` file
- ✅ Test authentication with HuggingFace
- ✅ Verify access to gated models (like Llama)
- ✅ Provide troubleshooting guidance if issues are found

**Example output:**

```
🔍 Checking HuggingFace token configuration...

📁 Loading environment variables from .env
✅ Loaded HF_TOKEN from .env file: hf_NcgaE...VkLpCBgB

✅ HF_TOKEN found: hf_NcgaE...VkLpCBgB
✅ Authentication successful for user: realdanielbyrne

🔍 Testing access to meta-llama/Llama-3.2-1B-Instruct...
✅ Successfully accessed the gated model!
```

## ICA Template Building

FunctionalNetworksSFT includes a dedicated tool for building ICA templates from datasets without requiring model training. These templates can be used later during training to apply pre-computed functional network masks.

### Building ICA Templates

The `build_ica_templates.py` script supports both positional and named arguments for maximum flexibility:

#### Quick Start (Positional Arguments)

```bash
# Basic usage with positional arguments (recommended)
poetry run python -m functionalnetworkssft.build_ica_templates \
    meta-llama/Llama-3.2-1B-Instruct \
    tatsu-lab/alpaca

# Multiple datasets
poetry run python -m functionalnetworkssft.build_ica_templates \
    meta-llama/Llama-3.2-1B-Instruct \
    databricks/databricks-dolly-15k \
    tatsu-lab/alpaca

# With optional parameters
poetry run python -m functionalnetworkssft.build_ica_templates \
    meta-llama/Llama-3.2-1B-Instruct \
    dataset1.json dataset2.jsonl \
    --ica_template_samples_per_ds 200 \
    --ica_template_output ./custom/output/ \
    --ica_components 15 \
    --ica_percentile 95.0
```

#### Alternative Syntax (Named Arguments)

```bash
# Using named arguments (also supported)
poetry run python -m functionalnetworkssft.build_ica_templates \
    --model_name_or_path meta-llama/Llama-3.2-1B-Instruct \
    --ica_build_templates_from tatsu-lab/alpaca

# Mixed usage (positional model + named datasets)
poetry run python -m functionalnetworkssft.build_ica_templates \
    meta-llama/Llama-3.2-1B-Instruct \
    --ica_build_templates_from tatsu-lab/alpaca databricks/databricks-dolly-15k
```

### Supported Dataset Formats

The ICA template builder supports multiple dataset formats:

- **Local files**: `.json`, `.jsonl`, `.csv`
- **Hugging Face Hub datasets**: Any dataset name (e.g., `squad`, `alpaca`)

### Configuration Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `model` (positional) | Model name or path for ICA computation | Required |
| `datasets` (positional) | One or more dataset paths | Required |
| `--ica_template_samples_per_ds` | Number of samples per dataset | 100 |
| `--ica_template_output` | Output directory for templates | `./ica_templates/` |
| `--ica_components` | Number of ICA components | 10 |
| `--ica_percentile` | Percentile threshold | 98.0 |
| `--ica_dtype` | Data type for computation | `auto` |
| `--max_seq_length` | Maximum sequence length | 512 |
| `--template_format` | Dataset format detection | `auto` |

### Example: Building Templates for Code Tasks

```bash
# Build ICA templates for code-related fine-tuning
poetry run python -m functionalnetworkssft.build_ica_templates \
    microsoft/DialoGPT-medium \
    code_dataset.json logic_dataset.jsonl \
    --ica_template_samples_per_ds 500 \
    --ica_components 20 \
    --ica_percentile 95.0 \
    --ica_template_output ./code_ica_templates/
```

### Using Pre-computed Templates

Once built, ICA templates can be used during training:

```bash
poetry run fnsft \
    --model_name_or_path microsoft/DialoGPT-medium \
    --dataset_name_or_path your_dataset.json \
    --output_dir ./output \
    --ica_template_path ./ica_templates/global_templates.json \
    --mask_mode lesion \
    --ica_component_ids [0]
```

### Template Output

The script generates:

- **Template file**: `global_templates.json` containing component masks
- **Component coverage summary**: Detailed breakdown of channels per layer
- **Logging output**: Progress and configuration details

**Example output:**

```
✅ Template building completed successfully!
✅ Templates saved to: ./ica_templates/global_templates.json
✅ Number of components: 10

Component Coverage Summary:
  • Component 0: 656 channels across 12 layers
  • Component 1: 1024 channels across 8 layers
  • Component 2: 512 channels across 15 layers
  ...
```

## Platform Support

The package provides optimized installations for different hardware configurations:

### 🚀 NVIDIA GPU Systems (CUDA)

**Installation:** `python scripts/setup_cuda.py`

| Feature | Support | Notes |
|---------|---------|-------|
| GPU Acceleration | ✅ CUDA | Full NVIDIA GPU support |
| Mixed Precision | ✅ fp16/bf16 | Automatic precision selection |
| Quantization | ✅ 4-bit/8-bit | BitsAndBytes integration |
| Flash Attention | ✅ When compatible | Faster attention computation |
| Memory Optimization | ✅ Gradient checkpointing | Reduced memory usage |

**Recommended for:** Production training, large models (7B+ parameters)

### 🍎 Apple Silicon (MPS)

**Installation:** `poetry install --extras apple-silicon`

| Feature | Support | Notes |
|---------|---------|-------|
| GPU Acceleration | ✅ MPS | Metal Performance Shaders |
| Mixed Precision | ✅ fp16 | Optimized for Apple Silicon |
| Quantization | ❌ Not available | BitsAndBytes incompatible |
| Flash Attention | ❌ Not available | MPS limitations |
| Memory Optimization | ✅ Gradient checkpointing | Available |

**Recommended for:** Development, medium models (1B-7B parameters)

### 💻 CPU-Only Systems

**Installation:** `poetry install --extras cpu`

| Feature | Support | Notes |
|---------|---------|-------|
| GPU Acceleration | ❌ CPU only | No hardware acceleration |
| Mixed Precision | ❌ fp32 only | Full precision training |
| Quantization | ❌ Not available | CPU limitations |
| Flash Attention | ❌ Not available | CPU limitations |
| Memory Optimization | ✅ Gradient checkpointing | Available |

**Recommended for:** Testing, small models (<1B parameters), development without GPU

## Documentation

- **[Cross-Platform Setup Guide](CROSS_PLATFORM_SETUP.md)** - Detailed installation instructions for all platforms
- **[Training Guide](docs/training.md)** - Comprehensive training documentation (coming soon)
- **[API Reference](docs/api.md)** - API documentation (coming soon)

## Research

### ICA-Based Functional Network Masking for LLM Fine-Tuning

Large Language Models have been shown to contain functional networks of neurons that frequently co-activate and are crucial to model performance ￼. In this approach, we fine-tune a Hugging Face transformer model while masking out selected neurons based on an Independent Component Analysis (ICA) of neuron activations ￼. By toggling which functional networks (groups of neurons) are active during training, we can explore the model’s reliance on those networks. Below, we detail the implementation steps, CLI integration, use of precomputed ICA masks or on-the-fly computation, and how to integrate this extension with Hugging Face’s SFTTrainer (supporting LoRA and quantization). We also provide an example fine-tuning on a logic/code task to demonstrate usage.

#### Functional Network Masking in LLMs

Researchers have found that neurons in LLMs form functional networks analogous to functional brain networks ￼ ￼. These are sets of neurons that consistently co-activate under certain conditions ￼. Crucially, only a small fraction of neurons may constitute key networks essential for performance: masking these key networks (setting their outputs to zero) significantly degrades model performance, whereas retaining only these networks (masking all others) can still maintain much of the model’s functionality ￼ ￼. Prior work even showed that manipulating important neurons’ outputs via amplification or masking can steer model behavior ￼.

Our goal is to leverage these insights by introducing binary neuron masks during fine-tuning. This mask will zero-out either a chosen functional network (to ablate it) or all but that network (to isolate it). The masking is applied in the forward pass to the outputs of specific neurons, thereby affecting which neurons contribute to model computations and which gradients are updated. This allows us to fine-tune the model with or without certain functional subnetworks, potentially leading to insights on their role or even a lighter model focusing on key neurons ￼.

#### Implementing Binary Neuron Masks in the Forward Pass

To apply binary masks during the forward pass, we inject PyTorch forward hooks at the appropriate locations in the model’s architecture. In transformer decoders, each block contains an MLP (feed-forward network) that expands to a higher-dimensional “intermediate” layer (often 4× the hidden size) and then projects back to the hidden size ￼. We treat each dimension in this intermediate layer as a “neuron” and target them for masking (since these are the neurons identified by ICA analysis ￼).

**Where to mask**: We apply the mask right after the non-linear activation in the MLP, just before the second linear projection back to the hidden size. By zeroing out selected intermediate features at this point, we effectively remove those neurons’ contribution to the model’s output. This is equivalent to “masking their outputs,” as described in prior research ￼. It ensures masked neurons produce no activation and receive no gradient, while unmasked neurons function normally.

**How to mask via hooks**: We register a forward pre-hook on each MLP’s second linear layer (often named proj or down_proj), so that we can intercept its input. The hook multiplies the input tensor elementwise by a binary mask (1s for active neurons, 0s for masked neurons). Using a forward pre-hook means the mask is applied before the linear layer computes its output. This approach cleanly separates masking logic from the model code (no need to manually edit model source) and works with LoRA or quantized layers as long as we target the correct module.

**Verifying hook placement**: We identify the correct linear layer by checking for a linear whose out_features == hidden_size and in_features > hidden_size (since the intermediate layer is larger). This reliably catches the second projection of the MLP in architectures like GPT, LLaMA, OPT, BERT, etc., but skips attention output layers (which have in_features == out_features == hidden_size). For example, in GPT-2 each block’s MLP has c_fc (1024→4096) and c_proj (4096→1024); our code finds c_proj and masks its input of size 4096. In LLaMA-7B, each layer’s MLP uses two projections (gate_proj and up_proj both 4096→11008) and a down_proj (11008→4096); the hook targets down_proj’s input of size 11008. By masking that input, we zero out selected intermediate neurons after the SiLU activation and gating, which is effectively removing those functional units from the network’s computations.

**Trainable parameters**: With this forward masking in place, any neuron set to 0 will also receive zero gradient (since its output does not affect the loss). Thus, during fine-tuning those masked neurons’ weights won’t update (their gradient is essentially suppressed). The rest of the model trains normally. This fulfills the requirement that the mask affects which neurons are active and trainable: masked ones are inactive (and effectively not trained), while unmasked ones continue to learn.

## Requirements

- Python 3.12+
- Poetry for dependency management
- Platform-specific requirements (see setup guide)

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions are welcome! Please read our contributing guidelines before submitting pull requests.

## Support

For issues and questions:

1. Check the [Cross-Platform Setup Guide](CROSS_PLATFORM_SETUP.md)
2. Search existing GitHub issues
3. Create a new issue with platform details and error logs
