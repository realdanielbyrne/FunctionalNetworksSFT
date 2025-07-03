# Cross-Platform Setup Guide for FunctionalNetworksSFT

This guide provides installation instructions for running FunctionalNetworksSFT on different platforms with optimal hardware acceleration.

## Supported Platforms

- **CUDA-enabled systems** (NVIDIA GPUs on Windows/Linux)
- **Apple Silicon Macs** (M1/M2/M3/M4 with MPS acceleration)
- **CPU-only systems** (fallback for any platform)

## Prerequisites

- Python 3.12 or higher
- Poetry (for dependency management)
- Git

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd FunctionalNetworksSFT
```

### 2. Install Base Dependencies

```bash
poetry install
```

This installs the core dependencies that work across all platforms.

### 3. Platform-Specific Setup

#### For CUDA Systems (NVIDIA GPUs)

**Requirements:**
- NVIDIA GPU with CUDA Compute Capability 5.0 or higher
- CUDA 12.1 or compatible version
- Sufficient GPU memory (8GB+ recommended for training)

**Installation:**
```bash
# Install PyTorch with CUDA support
poetry run pip uninstall torch torchvision torchaudio -y
poetry run pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install CUDA-specific dependencies
poetry install --extras cuda
```

**Features Available:**
- ✅ GPU acceleration via CUDA
- ✅ BitsAndBytes quantization (4-bit/8-bit)
- ✅ Flash Attention (if compatible)
- ✅ Mixed precision training (fp16/bf16)

#### For Apple Silicon Macs

**Requirements:**
- Apple Silicon Mac (M1/M2/M3/M4)
- macOS 12.3 or later
- Sufficient unified memory (16GB+ recommended)

**Installation:**
```bash
# Base installation already includes MPS support
poetry install --extras apple-silicon
```

**Features Available:**
- ✅ GPU acceleration via Metal Performance Shaders (MPS)
- ✅ Mixed precision training (fp16)
- ❌ BitsAndBytes quantization (not supported)
- ❌ Flash Attention (not supported)

**Note:** Quantization is automatically disabled on Apple Silicon. The system will use full precision or fp16 instead.

#### For CPU-Only Systems

**Installation:**
```bash
# Base installation provides CPU support
poetry install
```

**Features Available:**
- ✅ CPU training (slower but functional)
- ✅ Full precision training (fp32)
- ❌ GPU acceleration
- ❌ Quantization
- ❌ Flash Attention

## Verification

After installation, verify your setup:

```bash
poetry run python utils/platform_setup.py
```

This will display:
- Platform information
- Available accelerators (CUDA/MPS/CPU)
- Library versions
- Quantization support status

## Usage

### Automatic Platform Detection

The training script automatically detects your platform and configures optimal settings:

```bash
poetry run python -m functionalnetworkssft.sft_trainer \
    --model_name_or_path microsoft/DialoGPT-medium \
    --dataset_name_or_path your_dataset.json \
    --output_dir ./output \
    --torch_dtype auto  # Automatically selects best dtype for your platform
```

### Manual Configuration

You can override automatic detection:

```bash
# Force specific dtype
--torch_dtype float16  # or bfloat16, float32

# Disable quantization (useful for debugging)
--use_4bit false --use_8bit false
```

## Platform-Specific Notes

### CUDA Systems

- **Memory Management**: Use `--gradient_checkpointing true` for large models
- **Batch Size**: Start with smaller batch sizes and increase based on GPU memory
- **Quantization**: 4-bit quantization can reduce memory usage by ~75%

### Apple Silicon

- **Memory Sharing**: Unified memory is shared between CPU and GPU
- **Batch Size**: Can often use larger batch sizes due to unified memory architecture
- **Performance**: MPS acceleration provides significant speedup over CPU

### CPU-Only

- **Performance**: Significantly slower than GPU acceleration
- **Memory**: May require smaller models or aggressive gradient checkpointing
- **Batch Size**: Use smaller batch sizes to fit in system RAM

## Troubleshooting

### CUDA Issues

**Problem**: CUDA out of memory
**Solution**: 
- Reduce batch size
- Enable gradient checkpointing
- Use 4-bit quantization
- Use smaller model

**Problem**: CUDA version mismatch
**Solution**: Install PyTorch version matching your CUDA installation

### Apple Silicon Issues

**Problem**: MPS not available
**Solution**: 
- Update to macOS 12.3+
- Ensure PyTorch 2.0+
- Check system compatibility

**Problem**: Memory issues
**Solution**:
- Close other applications
- Use smaller batch sizes
- Enable gradient checkpointing

### General Issues

**Problem**: Import errors
**Solution**: 
- Verify Poetry environment: `poetry env info`
- Reinstall dependencies: `poetry install --sync`
- Check Python version compatibility

## Performance Optimization

### CUDA Systems
- Use mixed precision training (`--fp16 true` or `--bf16 true`)
- Enable gradient checkpointing for large models
- Use 4-bit quantization for memory efficiency

### Apple Silicon
- Use fp16 precision (`--torch_dtype float16`)
- Leverage unified memory with larger batch sizes
- Monitor memory usage with Activity Monitor

### All Platforms
- Use gradient accumulation for effective larger batch sizes
- Enable early stopping to prevent overfitting
- Use learning rate scheduling

## Environment Variables

Set these for optimal performance:

```bash
# For CUDA systems
export CUDA_VISIBLE_DEVICES=0  # Use specific GPU
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# For Apple Silicon
export PYTORCH_ENABLE_MPS_FALLBACK=1

# For all platforms
export TOKENIZERS_PARALLELISM=false  # Avoid tokenizer warnings
```

## Next Steps

1. **Test Installation**: Run the verification script
2. **Prepare Data**: Format your training data according to the documentation
3. **Start Training**: Begin with a small model and dataset to verify setup
4. **Scale Up**: Gradually increase model size and dataset complexity

For more detailed training instructions, see the main README.md file.
