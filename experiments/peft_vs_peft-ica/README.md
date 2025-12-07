# PEFT vs PEFT+ICA Comparison Experiments

This directory contains a comprehensive experimental setup comparing three fine-tuning approaches using the centralized configuration architecture.

## Configuration Architecture

All experiments share a **single source of truth** for common parameters:

```text
experiments/peft_vs_peft-ica/
├── common_config.yaml          # Shared parameters (model, dataset, LoRA, training)
├── run_experiments.py          # Experiment runner with Python-defined overrides
├── evaluate_models.py          # Model evaluation and comparison
├── experiment_a_peft_only/     # Experiment A output directory
├── experiment_b_peft_ica/      # Experiment B output directory
└── experiment_c_peft_ica_preserve/  # Experiment C output directory
```

**Key Benefits:**

- No duplicate configuration files that could drift out of sync
- Guaranteed identical common parameters across all experiments
- Experiment-specific differences defined programmatically in Python
- Easy to modify shared settings in one place

## Experiment Overview

### Objective

Compare the effectiveness of Parameter Efficient Fine-Tuning (PEFT) with and without Independent Component Analysis (ICA) masking for domain-specific fine-tuning.

### Current Configuration

Settings are defined in `common_config.yaml`:

- **Model**: Configurable (default: `WeiboAI/VibeThinker-1.5B`)
- **Dataset**: Configurable (default: `camel-ai/physics`)
- **Training Epochs**: Configurable (default: 1)

## Experiments

### Experiment A: PEFT-Only

- **Method**: PEFT (LoRA) only
- **ICA Masking**: Disabled (`mask_mode: null`)
- **Output**: `experiment_a_peft_only/output/`

### Experiment B: PEFT + ICA Masking (Lesion)

- **Method**: PEFT (LoRA) + ICA masking
- **ICA Masking**: Enabled (`mask_mode: "lesion"`)
- **LoRA Target**: `down_proj` only (for ICA compatibility)
- **Output**: `experiment_b_peft_ica/output/`

### Experiment C: PEFT + ICA Masking (Preserve)

- **Method**: PEFT (LoRA) + ICA masking
- **ICA Masking**: Enabled (`mask_mode: "preserve"`)
- **LoRA Target**: `down_proj` only (for ICA compatibility)
- **Output**: `experiment_c_peft_ica_preserve/output/`

### Key Differences

The **only** parameters that differ between experiments are ICA-related settings:

| Parameter | Experiment A | Experiment B | Experiment C |
|-----------|--------------|--------------|--------------|
| `mask_mode` | `null` | `"lesion"` | `"preserve"` |
| `lora_target_modules` | (from common) | `["down_proj"]` | `["down_proj"]` |
| `ica_component_ids` | N/A | `[0, 1]` | `[0, 1]` |

All other parameters (model, dataset, LoRA rank/alpha, learning rate, etc.) are **guaranteed identical** via `common_config.yaml`.

## Running Experiments

### Prerequisites

1. Ensure Poetry virtual environment is activated: `poetry shell`
2. Configure `common_config.yaml` with desired model and dataset
3. Ensure you have HuggingFace authentication for gated models
4. Build ICA templates if using experiments B or C (see main project README)

### Option 1: Run All Experiments (Recommended)

```bash
# Run all three experiments sequentially
python experiments/peft_vs_peft-ica/run_experiments.py

# Run with verbose logging
python experiments/peft_vs_peft-ica/run_experiments.py --verbose
```

### Option 2: Run Individual Experiments

```bash
# Run only Experiment A (PEFT-only)
python experiments/peft_vs_peft-ica/run_experiments.py --experiment a

# Run only Experiment B (PEFT+ICA lesion)
python experiments/peft_vs_peft-ica/run_experiments.py --experiment b

# Run only Experiment C (PEFT+ICA preserve)
python experiments/peft_vs_peft-ica/run_experiments.py --experiment c
```

### Option 3: Run Evaluation Only

```bash
# Run evaluation on already trained models
python experiments/peft_vs_peft-ica/evaluate_models.py

# Run evaluation with custom test size
python experiments/peft_vs_peft-ica/evaluate_models.py --test-size 0.3

# Run evaluation with custom output directory
python experiments/peft_vs_peft-ica/evaluate_models.py --output-dir custom_results
```

## Expected Outputs

### Experiment A Output (`experiment_a_peft_only/output/`)

- `final_model/` - Final PEFT adapter files
- `merged_model/` - Merged base + adapter model (if merge enabled)
- `experiment_a.log` - Training logs
- `trainer_state.json` - Training state

### Experiment B Output (`experiment_b_peft_ica/output/`)

- `final_model/` - Final PEFT adapter files (with ICA masking applied)
- `merged_model/` - Merged base + adapter model (if merge enabled)
- `experiment_b.log` - Training logs (includes ICA computation details)
- `trainer_state.json` - Training state

### Experiment C Output (`experiment_c_peft_ica_preserve/output/`)

- `final_model/` - Final PEFT adapter files (with ICA preserve masking)
- `merged_model/` - Merged base + adapter model (if merge enabled)
- `experiment_c.log` - Training logs (includes ICA computation details)
- `trainer_state.json` - Training state

### Evaluation Results (`evaluation_results/`)

- `experiment_a_results.json` - Detailed metrics for PEFT-only model
- `experiment_b_results.json` - Detailed metrics for PEFT+ICA lesion model
- `experiment_c_results.json` - Detailed metrics for PEFT+ICA preserve model
- `model_comparison.json` - Side-by-side comparison data
- `evaluation_summary.md` - Human-readable summary report

### Master Run Logs (`logs/`)

- `experiment_run_YYYYMMDD_HHMMSS.log` - Timestamped run logs

## Configuration Details

All shared parameters are defined in `common_config.yaml`. To modify settings, edit this single file:

### Shared Parameters (from common_config.yaml)

- **LoRA rank**: 16
- **LoRA alpha**: 32
- **LoRA dropout**: 0.1
- **Learning rate**: 2e-4
- **Batch size**: 1 (train/eval)
- **Max sequence length**: 512
- **Validation split**: 10%
- **Gradient checkpointing**: Enabled

### Experiment-Specific Parameters (from run_experiments.py)

| Parameter | Experiment A | Experiment B | Experiment C |
|-----------|--------------|--------------|--------------|
| `mask_mode` | `null` | `"lesion"` | `"preserve"` |
| `lora_target_modules` | (from common) | `["down_proj"]` | `["down_proj"]` |
| `ica_components` | N/A | 5 | 5 |
| `ica_component_ids` | N/A | `[0, 1]` | `[0, 1]` |
| `ica_percentile` | N/A | 98.0 | 98.0 |

## Model Evaluation

### Automatic Evaluation

After running experiments, use the evaluation script to compare model performance:

```bash
python experiments/peft_vs_peft-ica/evaluate_models.py
```

### Evaluation Metrics

The evaluation uses HuggingFace's `evaluate` library to assess model performance:

#### Text Generation Quality

- **BLEU Score**: Measures n-gram overlap between generated and reference text
- **ROUGE-L**: Evaluates recall-oriented text similarity
- **Perplexity (PPL)**: Measures how well the model predicts the text
- **Negative Log-Likelihood (NLL)**: Per-example loss values

#### Response Quality

- **Average Response Length**: Word count statistics
- **Length Ratio**: Comparison of response lengths to reference answers
- **Sample Comparisons**: Side-by-side examples for qualitative analysis

### Interpreting Results

**Better Performance Indicators:**

- Higher BLEU, ROUGE scores (closer to reference text)
- Lower perplexity (more confident predictions)
- Length ratio closer to 1.0 (appropriate response length)

**The evaluation summary report (`evaluation_summary.md`) provides:**

- Performance comparison table with statistical significance
- Bootstrap 95% confidence intervals
- Pairwise p-values for NLL comparisons
- Visualization charts saved as PNG

## Monitoring & Logs

### Log Locations

- **Master logs**: `experiments/peft_vs_peft-ica/logs/experiment_run_*.log`
- **Experiment A logs**: `experiment_a_peft_only/output/experiment_a.log`
- **Experiment B logs**: `experiment_b_peft_ica/output/experiment_b.log`
- **Experiment C logs**: `experiment_c_peft_ica_preserve/output/experiment_c.log`
- **Evaluation logs**: Included in master logs when running full experiment suite

### What to Monitor

- Training loss progression
- Evaluation metrics during training
- ICA computation time (Experiments B and C)
- Memory usage
- Training duration differences
- Model evaluation metrics (BLEU, ROUGE, perplexity)

## Gitignore Coverage

All experiment outputs are automatically excluded from version control:

- Output directories (`*/output/`)
- Model checkpoints and artifacts
- Training logs
- Temporary config files (`temp_config_*.yaml`)

Only `common_config.yaml`, scripts, and this README are tracked in git.

## Troubleshooting

### Common Issues

1. **Missing common_config.yaml**: Ensure the file exists in `experiments/peft_vs_peft-ica/`
2. **Authentication**: Set up HuggingFace token for gated model access
3. **Memory issues**: Reduce batch size in `common_config.yaml`
4. **ICA computation slow**: This is expected for Experiments B and C
5. **Missing ICA templates**: Build templates first using `poetry run buildtemplates`

### Getting Help

- Check experiment logs for detailed error messages
- Verify Poetry virtual environment is activated: `poetry shell`
- Ensure all dependencies are installed: `poetry install`

## Next Steps

After running experiments:

1. **Review Evaluation Summary**: Check `evaluation_results/evaluation_summary.md` for comprehensive performance comparison
2. **Analyze Training Logs**: Compare training metrics and convergence patterns between experiments
3. **Examine Sample Outputs**: Review qualitative differences in generated responses
4. **Interpret Metrics**: Use the evaluation metrics to understand ICA masking impact:
   - Text quality (BLEU/ROUGE scores)
   - Perplexity and NLL values
   - Response appropriateness (length ratios)
5. **Consider Follow-up Experiments**: Based on results, consider:
   - Different ICA component counts or percentiles
   - Additional training epochs
   - Alternative masking strategies (lesion vs preserve)
   - Larger datasets or different domains

## Modifying Configuration

To change experiment settings, edit `common_config.yaml`:

```yaml
# Model and dataset (change these for different experiments)
model_name_or_path: "WeiboAI/VibeThinker-1.5B"
dataset_name_or_path: "camel-ai/physics"

# Training settings
num_train_epochs: 1
per_device_train_batch_size: 1
learning_rate: 2.0e-4

# LoRA settings
lora_r: 16
lora_alpha: 32
```

To modify ICA-specific settings (components, percentile, etc.), edit the experiment override dictionaries in `run_experiments.py`.
