# PEFT vs Full Parameter Fine-Tuning Guide

The FunctionalNetworksSFT framework now supports both Parameter-Efficient Fine-Tuning (PEFT) using LoRA/QLoRA and traditional full parameter fine-tuning. This guide helps you choose the right approach for your use case.

## Quick Start

### PEFT Training (Default)

```bash
python -m functionalnetworkssft.fnsft_trainer \
    --model_name_or_path microsoft/DialoGPT-medium \
    --dataset_name_or_path tatsu-lab/alpaca \
    --output_dir ./models/peft-model \
    --num_train_epochs 3
```

### Full Parameter Fine-Tuning

```bash
python -m functionalnetworkssft.fnsft_trainer \
    --model_name_or_path microsoft/DialoGPT-medium \
    --dataset_name_or_path tatsu-lab/alpaca \
    --output_dir ./models/full-model \
    --no_peft \
    --num_train_epochs 2 \
    --per_device_train_batch_size 2 \
    --learning_rate 5e-5
```

## Comparison Table

| Aspect | PEFT (LoRA/QLoRA) | Full Fine-Tuning |
|--------|-------------------|-------------------|
| **Memory Usage** | 🟢 Low (2-4GB) | 🔴 High (8-32GB+) |
| **Training Speed** | 🟢 Fast | 🟡 Slower |
| **Model Size** | 🟢 Small (adapters only) | 🔴 Large (full model) |
| **Performance** | 🟡 Good | 🟢 Potentially better |
| **Catastrophic Forgetting** | 🟢 Low risk | 🔴 Higher risk |
| **Setup Complexity** | 🟢 Simple | 🟡 More complex |
| **Resource Requirements** | 🟢 Consumer GPUs | 🔴 High-end GPUs |

## When to Use Each Approach

### Use PEFT When

- ✅ Limited GPU memory (< 16GB)
- ✅ Quick experimentation and iteration
- ✅ Multiple task-specific adaptations needed
- ✅ Preserving base model capabilities is important
- ✅ Storage and deployment efficiency matters
- ✅ Working with large models (7B+ parameters)

### Use Full Fine-Tuning When

- ✅ Maximum performance is critical
- ✅ Abundant computational resources available
- ✅ Significant domain shift from base model
- ✅ Complete model customization needed
- ✅ Working with smaller models (< 3B parameters)
- ✅ Research requiring full parameter control

## Configuration Examples

### PEFT Configuration

See `examples/peft_training_config.yaml` for a complete PEFT configuration with:

- Optimized hyperparameters for PEFT
- Memory-efficient settings
- Adapter-only Hub uploads

### Full Fine-Tuning Configuration  

See `examples/full_finetuning_config.yaml` for a complete full fine-tuning configuration with:

- Conservative hyperparameters for stability
- Memory optimization strategies
- Full model Hub uploads

## Memory Requirements

### PEFT (LoRA/QLoRA)

- **7B model**: 4-8GB GPU memory
- **13B model**: 8-12GB GPU memory  
- **30B model**: 16-24GB GPU memory

### Full Fine-Tuning

- **1B model**: 8-16GB GPU memory
- **3B model**: 16-32GB GPU memory
- **7B model**: 32-64GB GPU memory

## Best Practices

### For PEFT

1. Start with default LoRA parameters (r=16, alpha=32)
2. Use higher learning rates (1e-4 to 5e-4)
3. Enable quantization for memory efficiency
4. Upload only adapters to Hub for efficiency

### For Full Fine-Tuning

1. Use lower learning rates (1e-5 to 1e-4)
2. Enable gradient checkpointing
3. Use smaller batch sizes with gradient accumulation
4. Monitor for overfitting more carefully
5. Save checkpoints more frequently

## Troubleshooting

### Out of Memory Errors

**PEFT:**

- Enable 4-bit quantization: `--use_4bit`
- Reduce batch size: `--per_device_train_batch_size 1`
- Enable gradient checkpointing (enabled by default)

**Full Fine-Tuning:**

- Reduce batch size: `--per_device_train_batch_size 1`
- Increase gradient accumulation: `--gradient_accumulation_steps 4`
- Enable gradient checkpointing: `--gradient_checkpointing`
- Use mixed precision training (enabled by default)

### Poor Performance

**PEFT:**

- Increase LoRA rank: `--lora_r 32`
- Adjust learning rate: `--learning_rate 1e-4`
- Add more target modules: `--lora_target_modules q_proj k_proj v_proj o_proj`

**Full Fine-Tuning:**

- Lower learning rate: `--learning_rate 1e-5`
- Increase warmup: `--warmup_ratio 0.1`
- Add regularization: `--weight_decay 0.01`

## Advanced Features

Both training modes support all framework features:

- ICA-based functional network masking
- Multiple dataset formats and templates
- Weights & Biases integration
- Hugging Face Hub integration
- GGUF conversion for deployment

## Migration Guide

### From PEFT to Full Fine-Tuning

1. Add `--no_peft` flag
2. Reduce learning rate by 4-10x
3. Reduce batch size and increase gradient accumulation
4. Increase warmup ratio
5. Monitor memory usage carefully

### From Full Fine-Tuning to PEFT

1. Remove `--no_peft` flag (PEFT is default)
2. Increase learning rate by 4-10x
3. Can increase batch size
4. Reduce warmup ratio
5. Enable quantization for memory efficiency
