# Merge LoRA Adapter with Base Model

This document describes how to use the `--merge_adapter_with_base` feature to automatically merge a trained LoRA adapter with the base model at the conclusion of training.

## Overview

When training with LoRA/QLoRA (PEFT), you typically get adapter weights that need to be loaded on top of the base model. The merge feature allows you to create a standalone model that combines the base model with the trained adapter weights, eliminating the need to load adapters separately during inference.

## Usage

Add the `--merge_adapter_with_base` flag to your training command:

```bash
python -m functionalnetworkssft.fnsft_trainer \
    --model_name_or_path microsoft/DialoGPT-medium \
    --dataset_name_or_path your_dataset \
    --output_dir ./output \
    --use_peft \
    --merge_adapter_with_base \
    --num_train_epochs 3
```

## Uploading to Hugging Face Hub

When using both `--merge_adapter_with_base` and `--push_to_hub`, you can control which model gets uploaded:

### Upload Adapter (Default)

```bash
python -m functionalnetworkssft.fnsft_trainer \
    --model_name_or_path microsoft/DialoGPT-medium \
    --dataset_name_or_path your_dataset \
    --output_dir ./output \
    --use_peft \
    --merge_adapter_with_base \
    --push_to_hub \
    --hub_repo_id your-username/your-model \
    --num_train_epochs 3
```

### Upload Merged Model

```bash
python -m functionalnetworkssft.fnsft_trainer \
    --model_name_or_path microsoft/DialoGPT-medium \
    --dataset_name_or_path your_dataset \
    --output_dir ./output \
    --use_peft \
    --merge_adapter_with_base \
    --push_to_hub \
    --upload_merged_model \
    --hub_repo_id your-username/your-model \
    --num_train_epochs 3
```

## Output Directory Structure

When `--merge_adapter_with_base` is enabled, the following directory structure will be created in your output directory:

```
output/
├── final_model/          # Original adapter files (same as without merge flag)
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   └── tokenizer files...
├── adapter/              # Copy of adapter files for easy access
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   └── tokenizer files...
└── merged_model/         # Standalone merged model
    ├── config.json
    ├── pytorch_model.bin (or .safetensors)
    └── tokenizer files...
```

## Benefits

1. **Standalone Deployment**: The merged model can be used directly without needing to load adapters
2. **Simplified Inference**: No need to manage separate base model and adapter files
3. **Better Performance**: Eliminates the overhead of adapter loading during inference
4. **Easy Distribution**: Single model directory contains everything needed

## CLI Arguments

### `--merge_adapter_with_base`

- **Type**: Boolean flag
- **Purpose**: Merge trained LoRA adapter with base model after training completion
- **Requirements**: Must be used with PEFT training (`--use_peft` flag)

### `--upload_merged_model`

- **Type**: Boolean flag
- **Purpose**: Upload merged model instead of adapter when both `--merge_adapter_with_base` and `--push_to_hub` are enabled
- **Requirements**: Requires both `--merge_adapter_with_base` and `--push_to_hub` flags
- **Default**: If not specified, the adapter will be uploaded to Hub

## Requirements

- Must be used with PEFT training (`--use_peft` flag)
- Requires sufficient disk space for both adapter and merged model
- Base model must be compatible with the PEFT merge operation
- For Hub upload of merged model: requires `--upload_merged_model` flag

## Example Training Command

```bash
# Full example with common parameters
python -m functionalnetworkssft.fnsft_trainer \
    --model_name_or_path microsoft/DialoGPT-medium \
    --dataset_name_or_path ./data/my_conversations.json \
    --output_dir ./models/my_chatbot \
    --use_peft \
    --merge_adapter_with_base \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-4 \
    --warmup_ratio 0.03 \
    --logging_steps 10 \
    --save_steps 500 \
    --eval_steps 500
```

## Loading the Merged Model

After training, you can load the merged model directly:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the merged model (no adapter loading needed)
model = AutoModelForCausalLM.from_pretrained("./models/my_chatbot/merged_model")
tokenizer = AutoTokenizer.from_pretrained("./models/my_chatbot/merged_model")

# Use for inference
inputs = tokenizer("Hello, how are you?", return_tensors="pt")
outputs = model.generate(**inputs, max_length=50)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

## Notes

- The merge operation happens automatically after training completion
- If merge fails, training will still complete successfully and the adapter will be saved
- The original adapter files are preserved in both `final_model/` and `adapter/` directories
- Merging only works with LoRA adapters, not with full fine-tuning
