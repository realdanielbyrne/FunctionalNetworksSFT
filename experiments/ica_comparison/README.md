# ICA Masking Comparative Training Experiment

This experiment evaluates the effectiveness of ICA (Independent Component Analysis) masking in supervised fine-tuning by comparing it against a baseline approach without masking.

## Experiment Overview

### Objective
Compare the performance of ICA masking vs. standard SFT on financial instruction-following tasks.

### Configuration
- **Base Model:** `meta-llama/Llama-3.2-1B-Instruct`
- **Dataset:** `Josephgflowers/Finance-Instruct-500k` (500K financial instruction-following examples)
- **Training Method:** PEFT (LoRA) with identical hyperparameters
- **Training Duration:** 1 epoch each
- **Evaluation:** Financial instruction-following benchmarks

### Experimental Design
1. **Baseline:** Standard SFT without ICA masking
2. **Experimental:** SFT with ICA masking enabled (`mask_mode: "key"`)

Both experiments use identical:
- Random seeds (42)
- Hyperparameters
- Training conditions
- Evaluation metrics

## Dataset Format

The experiment handles the 3-column format with enhanced preprocessing:
- `system`: System prompt (uses default if empty)
- `user`: User instruction/question
- `assistant`: Expected response

Default system prompt: "You are a helpful assistant who thinks step by step when providing answers to user's questions."

## File Structure

```
experiments/ica_comparison/
├── README.md                           # This file
├── baseline_training_config.yaml       # Baseline training configuration
├── experimental_training_config.yaml   # ICA masking training configuration
├── financial_evaluation_config.yaml    # Evaluation configuration
├── run_comparative_experiment.py       # Main experiment runner
├── compare_results.py                  # Results analysis script
├── test_experiment_setup.py           # Validation script
├── baseline_model/                     # Baseline model output
├── experimental_model/                 # Experimental model output
└── comparison_reports/                 # Analysis results
    ├── comparison_table.csv
    ├── comparison_visualization.png
    └── detailed_comparison_report.md
```

## Quick Start

### 1. Validate Setup
```bash
cd experiments/ica_comparison
python test_experiment_setup.py
```

### 2. Run Full Experiment
```bash
python run_comparative_experiment.py
```

### 3. Generate Comparison Report
```bash
python compare_results.py
```

## Step-by-Step Usage

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended)
- Hugging Face account with access token
- Required packages (see validation script)

### Installation
```bash
pip install torch transformers datasets peft bitsandbytes accelerate
pip install pandas numpy matplotlib seaborn scikit-learn PyYAML
```

### Authentication
Set your Hugging Face token:
```bash
export HF_TOKEN="your_token_here"
# or
huggingface-cli login
```

### Running Individual Components

#### Baseline Training Only
```bash
python -m functionalnetworkssft.fnsft_trainer \
    --config experiments/ica_comparison/baseline_training_config.yaml
```

#### Experimental Training Only
```bash
python -m functionalnetworkssft.fnsft_trainer \
    --config experiments/ica_comparison/experimental_training_config.yaml
```

#### Evaluation Only
```bash
python -m functionalnetworkssft.evaluation.run_evaluation \
    --config experiments/ica_comparison/financial_evaluation_config.yaml \
    --model_name_or_path ./experiments/ica_comparison/baseline_model
```

## Configuration Details

### Training Configuration
Both configurations are identical except for ICA masking:

**Shared Settings:**
- Learning rate: 2e-4
- Batch size: 4 (with gradient accumulation: 4)
- LoRA rank: 16, alpha: 32
- 4-bit quantization enabled
- Cosine learning rate schedule

**Key Difference:**
- Baseline: `mask_mode: null`
- Experimental: `mask_mode: "key"`

### Evaluation Metrics
- **Language Understanding:** Perplexity, BLEU, ROUGE, BERTScore
- **MMLU:** Accuracy on financial/business subjects
- **Performance:** Inference speed, memory usage, model size
- **Financial Reasoning:** Custom financial domain evaluation
- **Instruction Following:** Adherence and response quality

## Expected Outputs

### Training Outputs
- `baseline_model/`: Baseline model checkpoints and adapters
- `experimental_model/`: ICA-masked model checkpoints and adapters
- `experiment.log`: Detailed training logs
- `experiment_results.json`: Complete experiment metadata

### Evaluation Outputs
- `comparison_table.csv`: Metric comparison table
- `comparison_visualization.png`: Visual comparison charts
- `detailed_comparison_report.md`: Comprehensive analysis report

### Key Metrics Compared
1. **Training Time:** Time to complete training
2. **Perplexity:** Language modeling performance
3. **BLEU/ROUGE:** Text generation quality
4. **MMLU Accuracy:** General knowledge retention
5. **Inference Speed:** Tokens per second
6. **Memory Usage:** Peak memory consumption
7. **Model Size:** Final model size

## Interpreting Results

### Success Indicators
- **Positive ICA Impact:** Improved evaluation metrics with minimal training overhead
- **Neutral Impact:** Similar performance with acceptable overhead
- **Negative Impact:** Degraded performance or excessive overhead

### Analysis Framework
The comparison script provides:
- Absolute metric differences
- Percentage improvements
- Statistical significance indicators
- Visual comparisons
- Detailed recommendations

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size in configs
   - Enable gradient checkpointing
   - Use smaller model or sequence length

2. **Dataset Access Errors**
   - Verify Hugging Face authentication
   - Check internet connection
   - Ensure dataset permissions

3. **Import Errors**
   - Run validation script to check dependencies
   - Install missing packages
   - Verify Python path

4. **Configuration Errors**
   - Validate YAML syntax
   - Check file paths
   - Ensure all required fields are present

### Debug Mode
Add `--log_level DEBUG` to training commands for verbose logging.

## Customization

### Modifying Experiment Parameters
Edit the YAML configuration files to adjust:
- Model size/type
- Dataset
- Training hyperparameters
- Evaluation benchmarks

### Adding Custom Metrics
Extend the evaluation configuration with additional benchmarks or modify the comparison script to include custom analysis.

### Scaling the Experiment
- Increase `num_train_epochs` for longer training
- Modify `max_samples` in evaluation configs for more comprehensive testing
- Add multiple random seeds for statistical robustness

## Results Interpretation Guide

### Training Performance
- Compare training times to assess ICA overhead
- Monitor convergence patterns in logs

### Evaluation Performance
- Focus on domain-relevant metrics (financial reasoning)
- Consider trade-offs between different metric types
- Evaluate statistical significance of differences

### Practical Implications
- Assess whether improvements justify computational overhead
- Consider deployment constraints (memory, speed)
- Evaluate generalization to other domains

## Citation

If you use this experimental framework, please cite:
```
FunctionalNetworksSFT ICA Masking Comparative Experiment
https://github.com/realdanielbyrne/FunctionalNetworksSFT
```

## Support

For issues or questions:
1. Check the troubleshooting section
2. Run the validation script
3. Review experiment logs
4. Open an issue on the project repository
