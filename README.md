# FunctionalNetworksSFT

A cross-platform supervised fine-tuning (SFT) framework for language models with support for both CUDA and Apple Silicon acceleration.

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
poetry run python -m functionalnetworkssft.sft_trainer \
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
