# PEFT vs PEFT+ICA AB Comparison Test

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

## 🚀 Running Experiments

### Prerequisites

1. Ensure the virtual environment is activated
2. Verify the sarcasm dataset exists at `datasets/sarcasm.csv`
3. Ensure you have access to the Llama model (HuggingFace authentication)

### Option 1: Run All Experiments (Recommended)

```bash
# Run both experiments sequentially (includes automatic evaluation)
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

### Option 4: Run Evaluation Only

```bash
# Run evaluation on already trained models
python experiments/peft_vs_peft-ica/evaluate_models.py

# Run evaluation with custom test size
python experiments/peft_vs_peft-ica/evaluate_models.py --test-size 0.3

# Run evaluation with custom output directory
python experiments/peft_vs_peft-ica/evaluate_models.py --output-dir custom_results
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

### Evaluation Results (`evaluation_results/`)

- `experiment_a_results.json` - Detailed metrics for PEFT-only model
- `experiment_b_results.json` - Detailed metrics for PEFT+ICA model
- `model_comparison.json` - Side-by-side comparison data
- `evaluation_summary.md` - Human-readable summary report

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

## 📊 Model Evaluation

### Automatic Evaluation

When running both experiments together (`python experiments/run_experiments.py`), the system automatically evaluates both models using comprehensive metrics and generates comparison reports.

### Evaluation Metrics

The evaluation uses HuggingFace's `evaluate` library to assess model performance across multiple dimensions:

#### Text Generation Quality

- **BLEU Score**: Measures n-gram overlap between generated and reference text
- **ROUGE-1/2/L**: Evaluates recall-oriented text similarity
- **Perplexity**: Measures how well the model predicts the text

#### Sarcasm-Specific Metrics

- **Sarcasm Correlation**: Correlation between predicted and reference sarcasm intensity
- **Average Sarcasm Intensity**: Presence of sarcasm indicators in responses
- **Length Ratio**: Comparison of response lengths to reference answers

#### Response Quality

- **Average Response Length**: Word count statistics
- **Sample Comparisons**: Side-by-side examples for qualitative analysis

### Interpreting Results

**Better Performance Indicators:**

- Higher BLEU, ROUGE scores (closer to reference text)
- Lower perplexity (more confident predictions)
- Higher sarcasm correlation (better sarcasm detection)
- Length ratio closer to 1.0 (appropriate response length)

**The evaluation summary report (`evaluation_summary.md`) provides:**

- Performance comparison table with percentage improvements
- Sample outputs for qualitative assessment
- Clear indication of which model performs better for each metric

## 🔍 Monitoring & Logs

### Log Locations

- **Master logs**: `experiments/logs/experiment_run_*.log`
- **Experiment A logs**: `experiment_a_peft_only/output/experiment_a.log`
- **Experiment B logs**: `experiment_b_peft_ica/output/experiment_b.log`
- **Evaluation logs**: Included in master logs when running full experiment suite

### What to Monitor

- Training loss progression
- Evaluation metrics during training
- ICA computation time (Experiment B only)
- Memory usage
- Training duration differences
- Model evaluation metrics (BLEU, ROUGE, perplexity)
- Comparative performance analysis

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

1. **Review Evaluation Summary**: Check `evaluation_results/evaluation_summary.md` for comprehensive performance comparison
2. **Analyze Training Logs**: Compare training metrics and convergence patterns between experiments
3. **Examine Sample Outputs**: Review qualitative differences in generated sarcastic responses
4. **Interpret Metrics**: Use the evaluation metrics to understand ICA masking impact:
   - Text quality improvements (BLEU/ROUGE scores)
   - Sarcasm detection accuracy (correlation metrics)
   - Response appropriateness (length ratios)
5. **Consider Follow-up Experiments**: Based on results, consider:
   - Different ICA component counts or percentiles
   - Additional training epochs
   - Alternative masking strategies
   - Larger datasets or different domains

### Platform-Specific Installation

#### 🪟 Windows with CUDA Support

For optimal performance on Windows with NVIDIA GPUs:

```bash
git clone <repository-url>
cd FunctionalNetworksSFT

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate

# Install PyTorch with CUDA support first
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install LMPipeline with quantization support
pip install -e .[quantization]

# Verify CUDA installation
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

**CUDA Requirements:**

- NVIDIA GPU with CUDA Compute Capability 5.0+
- CUDA 11.8 or 12.1+ installed
- Latest NVIDIA drivers

**Note for RTX 40/50-series (RTX 4090, RTX 5090):** These GPUs work with current PyTorch but may show compatibility warnings:

```bash
# RTX 5090 users may see warnings about sm_120 compute capability
# The device is still detected and usable despite the warning
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**RTX 5090 Compatibility:** The RTX 5090 with sm_120 compute capability shows a compatibility warning with PyTorch 2.5.1, but the device is fully functional for training and inference.

### Alternative: Using Poetry

```bash
# If you prefer Poetry for dependency management
poetry install

# For CUDA support, install PyTorch separately first
poetry run pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Activate environment
poetry shell
```
