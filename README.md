# Functional Networks SFT

<Description>

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

1. **Clone the repository:**

   ```bash
   git clone <repository-url>
   cd FunctionalNetworksSFT
   ```

2. **Install dependencies:**

   ```bash
   poetry install
   ```

3. **Platform-specific setup:**

   For detailed platform-specific installation instructions, see [CROSS_PLATFORM_SETUP.md](CROSS_PLATFORM_SETUP.md).

### Basic Usage

```bash
poetry run python -m functionalnetworkssft.fnsft_trainer \
    --model_name_or_path microsoft/DialoGPT-medium \
    --dataset_name_or_path your_dataset.json \
    --output_dir ./output \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --torch_dtype auto
```

## Platform Support

| Platform | GPU Acceleration | Quantization | Status |
|----------|------------------|--------------|--------|
| CUDA (NVIDIA) | ✅ CUDA | ✅ BitsAndBytes | Fully Supported |
| Apple Silicon | ✅ MPS | ❌ Not Available | Supported |
| CPU Only | ❌ CPU Only | ❌ Not Available | Basic Support |

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
