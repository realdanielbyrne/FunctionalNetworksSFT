# FunctionalNetworksSFT Evaluation Framework

A comprehensive evaluation framework for assessing fine-tuned language models using standard LLM benchmarks, performance metrics, and safety evaluations.

## Overview

The evaluation framework provides:

- **Core Language Understanding Metrics**: Perplexity, BLEU, ROUGE, BERTScore
- **Standard LLM Benchmarks**: MMLU, HellaSwag, ARC, GSM8K, HumanEval
- **Performance and Efficiency Metrics**: Inference speed, memory usage, model size, FLOPS
- **Safety and Bias Evaluation**: Toxicity detection, bias assessment, harmful content rates
- **Comprehensive Reporting**: Statistical analysis, visualizations, confidence intervals
- **CLI Integration**: Easy command-line usage and configuration

## Quick Start

### Installation

The evaluation framework is included with FunctionalNetworksSFT. Ensure you have the required dependencies:

```bash
pip install functionalnetworkssft[evaluation]
```

### Basic Usage

#### Command Line

```bash
# Quick evaluation with essential metrics
fnsft-eval --model_name_or_path your_model \
           --eval_benchmarks mmlu performance \
           --eval_max_samples 100

# Comprehensive evaluation
fnsft-eval --model_name_or_path your_model \
           --eval_config examples/evaluation_configs/comprehensive_evaluation.yaml
```

#### Python API

```python
from functionalnetworkssft.evaluation import ModelEvaluator, EvaluationConfig
from functionalnetworkssft.evaluation.core.config import get_default_evaluation_config

# Create evaluation configuration
eval_args = get_default_evaluation_config()
eval_args.model_name_or_path = "your_model"
eval_args.output_dir = "./evaluation_results"

# Run evaluation
config = EvaluationConfig(eval_args)
evaluator = ModelEvaluator(config)
report = evaluator.evaluate()

# Access results
for benchmark_name, benchmark in report.benchmarks.items():
    for metric_name, metric in benchmark.metrics.items():
        print(f"{benchmark_name}.{metric_name}: {metric.value:.4f}")
```

## Configuration

### YAML Configuration Files

Create evaluation configurations using YAML files:

```yaml
# evaluation_config.yaml
model_name_or_path: "your_model"
output_dir: "./evaluation_results"
batch_size: 8
max_length: 2048

benchmarks:
  - name: "mmlu"
    enabled: true
    dataset_name: "cais/mmlu"
    max_samples: 1000
    metrics:
      - name: "mmlu_accuracy"
        enabled: true
        
  - name: "performance"
    enabled: true
    max_samples: 50
    metrics:
      - name: "inference_speed"
        enabled: true
      - name: "memory_usage"
        enabled: true
```

### Programmatic Configuration

```python
from functionalnetworkssft.evaluation.core.config import (
    EvaluationArguments, 
    create_benchmark_config
)

eval_args = EvaluationArguments(
    model_name_or_path="your_model",
    output_dir="./results",
    benchmarks=[
        create_benchmark_config(
            name="mmlu",
            dataset_name="cais/mmlu",
            metrics=["mmlu_accuracy"],
            max_samples=500
        ),
        create_benchmark_config(
            name="performance",
            metrics=["inference_speed", "memory_usage"]
        )
    ]
)
```

## Available Benchmarks

### Language Understanding Metrics

- **Perplexity**: Language modeling quality
- **BLEU**: Text generation quality (n-gram overlap)
- **ROUGE**: Summarization quality (recall-oriented)
- **BERTScore**: Semantic similarity using BERT embeddings

### Standard LLM Benchmarks

- **MMLU**: Massive Multitask Language Understanding (57 subjects)
- **HellaSwag**: Commonsense reasoning completion
- **ARC**: AI2 Reasoning Challenge (science questions)
- **GSM8K**: Grade school math word problems
- **HumanEval**: Code generation evaluation

### Performance Metrics

- **Inference Speed**: Tokens per second generation rate
- **Memory Usage**: Peak memory consumption during inference
- **Model Size**: Parameter count and memory footprint
- **FLOPS**: Floating point operations per second

### Safety and Bias Evaluation

- **Toxicity**: Harmful language detection
- **Bias**: Demographic bias assessment
- **Harmful Content**: Violence, self-harm, illegal content detection

## Results and Reporting

### Evaluation Reports

The framework generates comprehensive reports including:

- **Metric Results**: Values, confidence intervals, sample sizes
- **Statistical Analysis**: Summary statistics, correlations
- **Benchmark Rankings**: Performance comparison across benchmarks
- **System Information**: Hardware, software environment details

### Result Storage

Results are saved in structured format:

```
evaluation_results/
├── reports/
│   └── evaluation_report_model_20240101_120000.json
├── raw_data/
│   └── predictions_model_20240101.pkl
└── visualizations/
    ├── benchmark_comparison.png
    └── performance_timeline.png
```

### Accessing Results

```python
from functionalnetworkssft.evaluation.core.results import ResultManager

# Load saved results
result_manager = ResultManager("./evaluation_results")
reports = result_manager.list_reports()
report = result_manager.load_report(reports[0])

# Get summary metrics
summary = report.get_summary_metrics()
print(f"Mean score: {summary['mean_score']:.4f}")

# Compare two reports
comparison = result_manager.compare_reports(report1, report2)
print(f"Mean improvement: {comparison['summary']['mean_difference']:.4f}")
```

## Advanced Usage

### Custom Metrics

Create custom evaluation metrics:

```python
from functionalnetworkssft.evaluation.core.metrics import BaseMetric, register_metric

@register_metric("custom_metric")
class CustomMetric(BaseMetric):
    def compute_score(self, predictions, references, **kwargs):
        # Your custom metric logic here
        return score

# Use in evaluation
eval_args.benchmarks[0].metrics.append(
    MetricConfig(name="custom_metric", enabled=True)
)
```

### Batch Evaluation

Evaluate multiple models:

```python
models = ["model1", "model2", "model3"]
reports = []

for model_path in models:
    eval_args.model_name_or_path = model_path
    config = EvaluationConfig(eval_args)
    evaluator = ModelEvaluator(config)
    report = evaluator.evaluate()
    reports.append(report)
    evaluator.cleanup()

# Compare all models
for i, report1 in enumerate(reports):
    for j, report2 in enumerate(reports[i+1:], i+1):
        comparison = result_manager.compare_reports(report1, report2)
        print(f"Model {i} vs Model {j}: {comparison['summary']['mean_difference']:.4f}")
```

### Integration with Training

Integrate evaluation into training pipeline:

```python
from functionalnetworkssft.fnsft_trainer import FNSFTTrainer
from functionalnetworkssft.evaluation import ModelEvaluator, EvaluationConfig

# After training
trainer = FNSFTTrainer(...)
trainer.train()

# Evaluate the trained model
eval_args = get_default_evaluation_config()
eval_args.model_name_or_path = trainer.args.output_dir
config = EvaluationConfig(eval_args)
evaluator = ModelEvaluator(config)
report = evaluator.evaluate()
```

## Configuration Reference

### Core Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_name_or_path` | str | Required | Model path or HuggingFace identifier |
| `output_dir` | str | `"./evaluation_results"` | Results output directory |
| `batch_size` | int | `8` | Evaluation batch size |
| `max_length` | int | `2048` | Maximum sequence length |
| `confidence_level` | float | `0.95` | Statistical confidence level |

### Benchmark Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | str | Required | Benchmark identifier |
| `enabled` | bool | `true` | Whether to run benchmark |
| `dataset_name` | str | Optional | HuggingFace dataset name |
| `max_samples` | int | Optional | Sample limit for faster testing |
| `metrics` | list | Required | List of metrics to compute |

### Metric Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | str | Required | Metric identifier |
| `enabled` | bool | `true` | Whether to compute metric |
| `parameters` | dict | `{}` | Metric-specific parameters |
| `weight` | float | `1.0` | Weight for aggregated scoring |

## Best Practices

### Performance Optimization

1. **Use appropriate batch sizes**: Larger batches for throughput, smaller for memory
2. **Limit samples for testing**: Use `max_samples` for quick validation
3. **Choose relevant benchmarks**: Don't run all benchmarks if not needed
4. **Monitor memory usage**: Use performance metrics to track resource consumption

### Statistical Rigor

1. **Use sufficient sample sizes**: Ensure adequate samples for reliable statistics
2. **Report confidence intervals**: Include uncertainty estimates
3. **Multiple runs**: Run evaluation multiple times for robust results
4. **Baseline comparison**: Compare against established baselines

### Safety Considerations

1. **Enable safety checks**: Always include safety and bias evaluation
2. **Review generated content**: Manually inspect samples for harmful content
3. **Set appropriate thresholds**: Adjust toxicity and bias detection thresholds
4. **Document limitations**: Report evaluation limitations and potential biases

## Troubleshooting

### Common Issues

1. **Out of memory errors**: Reduce batch size or max_length
2. **Slow evaluation**: Reduce max_samples or disable heavy benchmarks
3. **Missing dependencies**: Install evaluation extras with `pip install functionalnetworkssft[evaluation]`
4. **Authentication errors**: Set HF_TOKEN environment variable

### Performance Tips

1. Use GPU acceleration when available
2. Enable mixed precision with appropriate torch_dtype
3. Use device_map="auto" for multi-GPU setups
4. Cache datasets locally for repeated evaluations

## Examples

See the `examples/` directory for:

- `evaluation_configs/`: Sample configuration files
- `evaluation_demo.py`: Comprehensive demo script
- Integration examples with training pipeline

## Contributing

To add new benchmarks or metrics:

1. Implement the metric class inheriting from `BaseMetric`
2. Register the metric using `@register_metric` decorator
3. Add benchmark evaluator if needed
4. Update configuration schemas
5. Add tests and documentation
