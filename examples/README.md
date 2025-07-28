# FunctionalNetworksSFT Examples

This directory contains example configurations and demonstration scripts for the FunctionalNetworksSFT framework.

## Files Overview

### Configuration Files
- **`peft_training_config.yaml`** - Complete PEFT (LoRA/QLoRA) training configuration
- **`full_finetuning_config.yaml`** - Complete full parameter fine-tuning configuration

### Demo Scripts
- **`training_mode_demo.py`** - Interactive demonstration of both training modes and configuration methods

## Quick Start

### 1. Using YAML Configuration Files

**PEFT Training:**
```bash
python -m functionalnetworkssft.fnsft_trainer --config examples/peft_training_config.yaml
```

**Full Fine-Tuning:**
```bash
python -m functionalnetworkssft.fnsft_trainer --config examples/full_finetuning_config.yaml
```

**Override specific parameters:**
```bash
# Use PEFT config but change output directory and epochs
python -m functionalnetworkssft.fnsft_trainer \
    --config examples/peft_training_config.yaml \
    --output_dir ./my_custom_model \
    --num_train_epochs 5

# Use full training config but switch to PEFT mode
python -m functionalnetworkssft.fnsft_trainer \
    --config examples/full_finetuning_config.yaml \
    --output_dir ./my_peft_model
    # Note: no --no_peft flag means PEFT is used (default)
```

### 2. Using CLI Parameters Only

**PEFT Training (Default):**
```bash
python -m functionalnetworkssft.fnsft_trainer \
    --model_name_or_path microsoft/DialoGPT-medium \
    --dataset_name_or_path tatsu-lab/alpaca \
    --output_dir ./models/peft-model \
    --num_train_epochs 3
```

**Full Fine-Tuning:**
```bash
python -m functionalnetworkssft.fnsft_trainer \
    --model_name_or_path microsoft/DialoGPT-medium \
    --dataset_name_or_path tatsu-lab/alpaca \
    --output_dir ./models/full-model \
    --no_peft \
    --learning_rate 5e-5 \
    --per_device_train_batch_size 2
```

## Interactive Demo

The `training_mode_demo.py` script provides an interactive way to explore both training modes and configuration methods:

### Run All Demonstrations
```bash
python examples/training_mode_demo.py
```
This runs all combinations: PEFT + Full training with both CLI and YAML configurations.

### Specific Demonstrations
```bash
# Only PEFT training examples
python examples/training_mode_demo.py --mode peft

# Only CLI parameter examples
python examples/training_mode_demo.py --method cli

# Only YAML configuration examples
python examples/training_mode_demo.py --method yaml

# Full training with YAML config only
python examples/training_mode_demo.py --mode full --method yaml

# Clean up outputs after completion
python examples/training_mode_demo.py --cleanup
```

### Demo Output Structure
```
demo_output/
├── peft_model_cli/          # PEFT training via CLI parameters
├── peft_model_yaml/         # PEFT training via YAML config
├── full_model_cli/          # Full training via CLI parameters
└── full_model_yaml/         # Full training via YAML config
```

## Configuration Priority

When using YAML configs with CLI overrides, the priority order is:
1. **CLI arguments** (highest priority)
2. **YAML configuration file**
3. **Default values** (lowest priority)

This allows you to:
- Use YAML for base configuration
- Override specific values via CLI
- Maintain reproducible configurations

## Key Differences

### PEFT vs Full Fine-Tuning
| Aspect | PEFT | Full Fine-Tuning |
|--------|------|------------------|
| **Memory** | Low (2-8GB) | High (16-64GB) |
| **Speed** | Fast | Slower |
| **Output Size** | Small (adapters) | Large (full model) |
| **CLI Flag** | Default (no flag) | `--no_peft` |

### CLI vs YAML Configuration
| Aspect | CLI Parameters | YAML Config |
|--------|----------------|-------------|
| **Best For** | Experimentation | Production |
| **Reproducibility** | Manual | Automatic |
| **Version Control** | Difficult | Easy |
| **Complex Configs** | Verbose | Clean |

## Next Steps

1. **Try the demo**: `python examples/training_mode_demo.py`
2. **Customize configs**: Edit the YAML files for your use case
3. **Read the guide**: See `../PEFT_VS_FULL_FINETUNING.md` for detailed information
4. **Run your training**: Use the configuration method that best fits your workflow

## Troubleshooting

- **Out of memory**: Use PEFT mode or reduce batch size
- **Config not found**: Ensure you're running from the repository root
- **Import errors**: Make sure the package is installed: `pip install -e .`
