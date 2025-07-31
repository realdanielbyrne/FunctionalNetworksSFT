# Fine-Tuning Experiments: PEFT vs PEFT+ICA

This directory contains a comprehensive experimental setup comparing two fine-tuning approaches for the `meta-llama/Llama-3.2-1B-Instruct` model using the sarcasm dataset.

## 🎯 Experiment Overview

### Objective
Compare the effectiveness of Parameter Efficient Fine-Tuning (PEFT) with and without Independent Component Analysis (ICA) masking for sarcasm detection fine-tuning.

### Model & Dataset
- **Model**: `meta-llama/Llama-3.2-1B-Instruct`
- **Dataset**: `datasets/sarcasm.csv` (200 question-answer pairs with sarcastic responses)
- **Training Epochs**: 2 (for both experiments)

## 🧪 Experiments

### Experiment A: PEFT-Only
- **Method**: PEFT (LoRA) only
- **ICA Masking**: Disabled
- **Location**: `experiment_a_peft_only/`

### Experiment B: PEFT + ICA Masking  
- **Method**: PEFT (LoRA) + ICA masking
- **ICA Masking**: Enabled (key mode)
- **ICA Components**: 20
- **ICA Percentile**: 98.0
- **Location**: `experiment_b_peft_ica/`

### Key Differences
The **only** parameter difference between experiments is the ICA masking setting:
- Experiment A: `mask_mode: null` (disabled)
- Experiment B: `mask_mode: "key"` (enabled)

All other parameters (LoRA settings, training hyperparameters, etc.) are identical.

## 📁 Directory Structure

```
experiments/
├── README.md                           # This documentation
├── run_experiments.py                  # Master experiment runner
├── logs/                              # Experiment run logs
├── experiment_a_peft_only/
│   ├── config/
│   │   └── experiment_a_config.yaml   # PEFT-only configuration
│   ├── scripts/
│   │   └── run_experiment_a.py        # Experiment A runner
│   └── output/                        # Experiment A results (gitignored)
└── experiment_b_peft_ica/
    ├── config/
    │   └── experiment_b_config.yaml   # PEFT+ICA configuration
    ├── scripts/
    │   └── run_experiment_b.py        # Experiment B runner
    └── output/                        # Experiment B results (gitignored)
```

## 🚀 Running Experiments

### Prerequisites
1. Ensure the virtual environment is activated
2. Verify the sarcasm dataset exists at `datasets/sarcasm.csv`
3. Ensure you have access to the Llama model (HuggingFace authentication)

### Option 1: Run All Experiments (Recommended)
```bash
# Run both experiments sequentially
python experiments/run_experiments.py

# Run with verbose logging
python experiments/run_experiments.py --verbose
```

### Option 2: Run Individual Experiments
```bash
# Run only Experiment A (PEFT-only)
python experiments/run_experiments.py --experiment a

# Run only Experiment B (PEFT+ICA)
python experiments/run_experiments.py --experiment b
```

### Option 3: Run Experiments Directly
```bash
# Run Experiment A directly
python experiments/experiment_a_peft_only/scripts/run_experiment_a.py

# Run Experiment B directly
python experiments/experiment_b_peft_ica/scripts/run_experiment_b.py
```

## 📊 Expected Outputs

### Experiment A Output (`experiment_a_peft_only/output/`)
- `final_model/` - Final PEFT adapter files
- `checkpoint-*/` - Training checkpoints
- `experiment_a.log` - Training logs
- `trainer_state.json` - Training state
- `training_args.bin` - Training arguments

### Experiment B Output (`experiment_b_peft_ica/output/`)
- `final_model/` - Final PEFT adapter files (with ICA masking applied)
- `checkpoint-*/` - Training checkpoints
- `experiment_b.log` - Training logs (includes ICA computation details)
- `trainer_state.json` - Training state
- `training_args.bin` - Training arguments

### Master Run Logs (`logs/`)
- `experiment_run_YYYYMMDD_HHMMSS.log` - Timestamped run logs

## ⚙️ Configuration Details

### Shared Parameters (Identical in Both Experiments)
- **LoRA rank**: 16
- **LoRA alpha**: 32  
- **LoRA dropout**: 0.1
- **Learning rate**: 2e-4
- **Batch size**: 4 (train/eval)
- **Max sequence length**: 1024
- **Validation split**: 10%
- **Gradient checkpointing**: Enabled

### Experiment-Specific Parameters
| Parameter | Experiment A | Experiment B |
|-----------|--------------|--------------|
| `mask_mode` | `null` | `"key"` |
| `ica_components` | 20 (unused) | 20 |
| `ica_percentile` | 98.0 (unused) | 98.0 |

## 🔍 Monitoring & Logs

### Log Locations
- **Master logs**: `experiments/logs/experiment_run_*.log`
- **Experiment A logs**: `experiment_a_peft_only/output/experiment_a.log`
- **Experiment B logs**: `experiment_b_peft_ica/output/experiment_b.log`

### What to Monitor
- Training loss progression
- Evaluation metrics
- ICA computation time (Experiment B only)
- Memory usage
- Training duration differences

## 🚫 Gitignore Coverage

All experiment outputs are automatically excluded from version control:
- Output directories (`*/output/`)
- Model checkpoints and artifacts
- Training logs
- Temporary files

Only configuration files and scripts are tracked in git.

## 🔧 Troubleshooting

### Common Issues
1. **Missing dataset**: Ensure `datasets/sarcasm.csv` exists
2. **Authentication**: Set up HuggingFace token for Llama model access
3. **Memory issues**: Reduce batch size in config files if needed
4. **ICA computation slow**: This is expected for Experiment B

### Getting Help
- Check experiment logs for detailed error messages
- Verify virtual environment is activated
- Ensure all dependencies are installed via Poetry

## 📈 Next Steps

After running experiments:
1. Compare training logs and metrics between experiments
2. Evaluate model performance on test data
3. Analyze the impact of ICA masking on convergence
4. Consider running additional epochs or hyperparameter variations
