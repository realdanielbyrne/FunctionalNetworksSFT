# Functional Network-Based Selective Fine-Tuning of Large Language Models

A research framework for selective fine-tuning of LLMs using brain-inspired functional network masking. Uses ICA (Independent Component Analysis) to identify functional networks of neurons and enables selective training/freezing of these networks during supervised fine-tuning to mitigate catastrophic forgetting.

---

## Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd FunctionalNetworksSFT

# Install for your platform
poetry install --extras apple-silicon  # macOS with Apple Silicon
poetry install --extras cpu            # CPU-only systems
poetry run python scripts/setup_cuda.py  # NVIDIA GPU systems (CUDA)
```

For detailed platform-specific instructions, troubleshooting, and manual installation options, see the [Cross-Platform Setup Guide](docs/CROSS_PLATFORM_SETUP.md).

### Verify Installation

```bash
poetry run python -c "
from functionalnetworkssft.utils.model_utils import get_optimal_device
device, name = get_optimal_device()
print(f'Using device: {name}')
"
```

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

## ICA Template Building

FunctionalNetworksSFT includes a dedicated tool for building ICA templates from datasets without requiring model training. These templates can be used later during training to apply pre-computed functional network masks.

### Building ICA Templates

The `buildtemplates` CLI command supports both positional and named arguments for maximum flexibility:

#### Quick Start (Positional Arguments)

```bash
# Basic usage with positional arguments (recommended)
poetry run buildtemplates \
    meta-llama/Llama-3.2-1B-Instruct \
    tatsu-lab/alpaca

# Multiple datasets
poetry run buildtemplates \
    meta-llama/Llama-3.2-1B-Instruct \
    databricks/databricks-dolly-15k \
    tatsu-lab/alpaca

# With optional parameters
poetry run buildtemplates \
    meta-llama/Llama-3.2-1B-Instruct \
    dataset1.json dataset2.jsonl \
    --ica_template_samples_per_ds 200 \
    --ica_template_output ./custom/output/ \
    --ica_components 15 \
    --ica_percentile 95.0
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
Templates saved to: ./ica_templates/global_templates.json
Number of components: 10

Component Coverage Summary:
  Component 0: 656 channels across 12 layers
  Component 1: 1024 channels across 8 layers
  Component 2: 512 channels across 15 layers
  ...
```

## Documentation

- **[Cross-Platform Setup Guide](docs/CROSS_PLATFORM_SETUP.md)** - Detailed installation instructions, troubleshooting, and platform-specific notes
- **[Training Guide](docs/training.md)** - Comprehensive training documentation (coming soon)
- **[API Reference](docs/api.md)** - API documentation (coming soon)

---

## Research Background

### Abstract

Large Language Models (LLMs) have demonstrated remarkable capabilities, but fine-tuning them on new tasks can lead to catastrophic forgetting – the erosion of previously learned knowledge when adapting to new data ￼. Recent research in model interpretability suggests that groups of neurons in LLMs form functional networks analogous to functional brain networks in cognitive neuroscience ￼. In this work, we propose a brain-inspired selective fine-tuning methodology that targets these functional sub-networks during Supervised Fine-Tuning (SFT). By identifying independent components of neuron activations via Principal Component Analysis (PCA) and Independent Component Analysis (ICA), we isolate coherent functional networks within the model. We then selectively train (or conversely, freeze) these ICA-defined networks while keeping other parameters fixed. This approach is aimed at preserving the model’s pre-trained knowledge by minimizing disruptive weight updates, thereby potentially mitigating catastrophic forgetting while still integrating new information. We describe the theoretical motivation, methodology, and implementation of this framework. Results are pending as we continue to conduct experiments; however, this work lays the foundation for a novel fine-tuning paradigm that bridges insights from neuroscience and large-scale language model training.

### Introduction

The success of LLMs in NLP has led to widespread use of supervised fine-tuning (SFT) to adapt pre-trained models to specific tasks or instruction datasets. SFT involves updating model weights on new supervised data, which often improves performance on the fine-tuned task. However, a well-known challenge in this process is catastrophic forgetting, where the model’s performance on original or unrelated tasks deteriorates after fine-tuning ￼. Catastrophic forgetting occurs because adjusting all of a model’s parameters to fit new data can overwrite the knowledge gained during pre-training ￼ ￼. As model size increases, this forgetting effect may become even more severe ￼, undermining the model’s general-purpose capabilities.

Concurrently, understanding the internal mechanisms of LLMs has become an important research goal. Traditional interpretability studies often focus on identifying single “important” neurons or attention heads that correlate with specific behaviors or features. Yet, as Liu et al. (2025a) argue, such approaches neglect the fact that higher cognitive functions (in brains or in complex neural networks) arise from interactions among networks of neurons rather than single units ￼. In neuroscience, the concept of functional brain networks – distributed groups of neurons or regions with highly correlated activity – is well-established ￼. Brain-inspired analysis of LLMs suggests that analogous functional networks of artificial neurons may exist in these models ￼. Indeed, recent work identified recurring functional networks within GPT-style models using techniques akin to fMRI analysis ￼. These studies found that certain networks of neurons consistently co-activate and are crucial for the model’s performance: masking (lesioning) these key networks causes significant drops in performance, while preserving only these networks can still sustain much of the model’s functionality ￼. This insight implies that knowledge in LLMs might be compartmentalized into functional substructures.

Given this background, we hypothesize that selectively fine-tuning an LLM’s functional networks could improve training efficiency and reduce interference with existing skills. Rather than updating all weights indiscriminately, our approach restricts weight updates to a targeted subset of neurons – an ICA-derived functional network – or, alternatively, ablates certain networks during training (freezing them) to preserve their original function. By doing so, we aim to mitigate catastrophic forgetting: the untouched parts of the network maintain prior capabilities, while the model learns new information in a constrained subspace. This idea is a natural extension of the lesion/preservation experiments by Liu et al. (2025a) ￼, now applied during training. It also resonates with strategies in continual learning and parameter-efficient fine-tuning, where only a portion of the model’s weights are trained to prevent overwriting existing knowledge. For example, Low-Rank Adaptation (LoRA) trains only a small additional weight matrix and leaves the original weights mostly unchanged, thereby preserving the model’s pre-trained knowledge while learning new tasks ￼. Our method differs in that the subset of weights to update is determined by the model’s intrinsic functional organization (via ICA), rather than by architectural add-ons or generic criteria.

In summary, this work proposes a novel SFT algorithm that bridges research in functional networks with fine-tuning of LLMs. We present the motivation and methodology for identifying functional networks in transformer models and describe how selective training is performed. The ultimate goal is to enable LLMs to learn new tasks effectively with minimal loss of prior knowledge, improving continual learning for these models.

Key contributions of this project include: (1) introducing an ICA-based technique to decompose an LLM into functional sub-networks of neurons, (2) developing a selective fine-tuning approach that updates only chosen networks (or freezes them) during training to test their role in learning and forgetting, and (3) outlining an evaluation strategy to quantify new knowledge acquisition versus retention of original capabilities. An open-source implementation is provided to facilitate reproduction and further exploration of this approach.

### Prior Research

This project is inspired by the following research:

- Title: Brain-Inspired Exploration of Functional Networks and Key Neurons in Large Language Models
  Authors: Yiheng Liu, Xiaohui Gao, Haiyang Sun, Bao Ge, Tianming Liu, Junwei Han, Xintao Hu
  Link: <https://arxiv.org/html/2502.20408v1>

### Summary of the paper

The authors investigate whether large language models (LLMs) exhibit brain-like functional networks. Drawing on cognitive neuroscience, they apply Independent Component Analysis (ICA) to neuron activations (specifically MLP outputs) to decompose them into functional networks. They evaluate importance via two interventions: masking (lesion) and preservation. Key findings include that masking a small set of “key” networks (often comprising less than ~2% of neurons) can severely degrade performance, while preserving a compact subset of networks (on the order of ~10% of MLP neurons) can retain near-baseline capability. They also study group-wise ICA templates and similarity across inputs/models, supporting the view that LLMs contain recurring functional patterns.

### Concepts that informed this software

- Functional brain networks analogue: Treating LLM neuron activations like fMRI signals and recovering recurring functional networks via ICA.
- ICA over MLP activations: Running ICA on the final MLP outputs to obtain component-wise channel maps per layer.
- Group-wise templates and similarity: Building global (group) templates from multiple samples and relating per-sample networks via similarity metrics.
- Causal probes via interventions: Neuron/network lesion and preservation experiments to assess contribution to model behavior and efficiency.

### Relation to this codebase

- ICA-based discovery: This repo implements FastICA-based network extraction from MLP activations, including a global, group-wise mode. See `src/functionalnetworkssft/ica_mask.py` (`compute_global_networks`) and the CLI for template building in `src/functionalnetworkssft/build_ica_templates.py`.
- Network masking during SFT: We apply binary masks as forward hooks at MLP outputs to support both lesion and preserve modes. See `ICAMask.apply_component_masks` in `ica_mask.py` and the `--mask_mode` options in the training CLI.
- Template-driven workflows: Precomputed templates can be saved/loaded (`--ica_template_path`), selecting components (`--ica_component_ids`) and controlling sparsity via percentile thresholds (`--ica_percentile`).
- Experimental comparisons: The `experiments/peft_vs_peft-ica` directory includes runs contrasting standard PEFT with PEFT+ICA masking in lesion/preserve settings.

This repository is an independent, engineering-focused reimplementation that operationalizes the paper’s core ideas for supervised fine-tuning workflows (HF Transformers + LoRA/QLoRA), enabling reproducible ICA template building, component selection, and functional-network-aware training.

### Implementation Details

Large Language Models have been shown to contain functional networks of neurons that frequently co-activate and are crucial to model performance ￼. In this approach, we fine-tune a Hugging Face transformer model while masking out selected neurons based on an Independent Component Analysis (ICA) of neuron activations ￼. By toggling which functional networks (groups of neurons) are active during training, we can explore the model’s reliance on those networks. Below, we detail the implementation steps, CLI integration, use of precomputed ICA masks or on-the-fly computation, and how to integrate this extension with Hugging Face’s SFTTrainer (supporting LoRA and quantization). We also provide an example fine-tuning on a logic/code task to demonstrate usage.

Researchers have found that neurons in LLMs form functional networks analogous to functional brain networks ￼ ￼. These are sets of neurons that consistently co-activate under certain conditions ￼. Crucially, only a small fraction of neurons may constitute key networks essential for performance: masking these key networks (setting their outputs to zero) significantly degrades model performance, whereas retaining only these networks (masking all others) can still maintain much of the model’s functionality ￼ ￼. Prior work even showed that manipulating important neurons’ outputs via amplification or masking can steer model behavior ￼.

Our goal is to leverage these insights by introducing binary neuron masks during fine-tuning. This mask will zero-out either a chosen functional network (to ablate it) or all but that network (to isolate it). The masking is applied in the forward pass to the outputs of specific neurons, thereby affecting which neurons contribute to model computations and which gradients are updated. This allows us to fine-tune the model with or without certain functional subnetworks, potentially leading to insights on their role or even a lighter model focusing on key neurons ￼.

#### Implementing Binary Neuron Masks in the Forward Pass

To apply binary masks during the forward pass, we inject PyTorch forward hooks at the appropriate locations in the model’s architecture. In transformer decoders, each block contains an MLP (feed-forward network) that expands to a higher-dimensional “intermediate” layer (often 4× the hidden size) and then projects back to the hidden size ￼. We treat each dimension in this intermediate layer as a “neuron” and target them for masking (since these are the neurons identified by ICA analysis ￼).

**Where to mask**: We apply the mask right after the non-linear activation in the MLP, just before the second linear projection back to the hidden size. By zeroing out selected intermediate features at this point, we effectively remove those neurons’ contribution to the model’s output. This is equivalent to “masking their outputs,” as described in prior research ￼. It ensures masked neurons produce no activation and receive no gradient, while unmasked neurons function normally.

**How to mask via hooks**: We register a forward pre-hook on each MLP’s second linear layer (often named proj or down_proj), so that we can intercept its input. The hook multiplies the input tensor elementwise by a binary mask (1s for active neurons, 0s for masked neurons). Using a forward pre-hook means the mask is applied before the linear layer computes its output. This approach cleanly separates masking logic from the model code (no need to manually edit model source) and works with LoRA or quantized layers as long as we target the correct module.

**Verifying hook placement**: We identify the correct linear layer by checking for a linear whose out_features == hidden_size and in_features > hidden_size (since the intermediate layer is larger). This reliably catches the second projection of the MLP in architectures like GPT, LLaMA, OPT, BERT, etc., but skips attention output layers (which have in_features == out_features == hidden_size). For example, in GPT-2 each block’s MLP has c_fc (1024→4096) and c_proj (4096→1024); our code finds c_proj and masks its input of size 4096. In LLaMA-7B, each layer’s MLP uses two projections (gate_proj and up_proj both 4096→11008) and a down_proj (11008→4096); the hook targets down_proj’s input of size 11008. By masking that input, we zero out selected intermediate neurons after the SiLU activation and gating, which is effectively removing those functional units from the network’s computations.

**Trainable parameters**: With this forward masking in place, any neuron set to 0 will also receive zero gradient (since its output does not affect the loss). Thus, during fine-tuning those masked neurons’ weights won’t update (their gradient is essentially suppressed). The rest of the model trains normally. This fulfills the requirement that the mask affects which neurons are active and trainable: masked ones are inactive (and effectively not trained), while unmasked ones continue to learn.

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions are welcome! Please read our contributing guidelines before submitting pull requests.

## Support

For issues and questions:

1. Check the [Cross-Platform Setup Guide](docs/CROSS_PLATFORM_SETUP.md)
2. Search existing GitHub issues
3. Create a new issue with platform details and error logs
