"""
Configuration classes for the evaluation framework.

This module provides dataclasses and configuration management for evaluation settings,
following the same patterns as the existing training configuration system.
"""

import os
import yaml
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class MetricConfig:
    """Configuration for individual metrics."""

    name: str = field(metadata={"help": "Name of the metric"})
    enabled: bool = field(
        default=True, metadata={"help": "Whether to compute this metric"}
    )
    parameters: Dict[str, Any] = field(
        default_factory=dict, metadata={"help": "Metric-specific parameters"}
    )
    weight: float = field(
        default=1.0, metadata={"help": "Weight for aggregated scoring"}
    )


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark evaluations."""

    name: str = field(metadata={"help": "Name of the benchmark"})
    enabled: bool = field(
        default=True, metadata={"help": "Whether to run this benchmark"}
    )
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "HuggingFace dataset name or local path"}
    )
    dataset_config: Optional[str] = field(
        default=None, metadata={"help": "Dataset configuration name"}
    )
    subset: Optional[str] = field(
        default=None, metadata={"help": "Dataset subset to evaluate on"}
    )
    max_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Maximum number of samples to evaluate (for faster testing)"},
    )
    batch_size: int = field(default=8, metadata={"help": "Batch size for evaluation"})
    metrics: List[MetricConfig] = field(
        default_factory=list,
        metadata={"help": "List of metrics to compute for this benchmark"},
    )
    parameters: Dict[str, Any] = field(
        default_factory=dict, metadata={"help": "Benchmark-specific parameters"}
    )


@dataclass
class EvaluationArguments:
    """Arguments for evaluation configuration."""

    # Model and tokenizer
    model_name_or_path: str = field(
        metadata={"help": "Path to model or HuggingFace model identifier"}
    )
    tokenizer_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "Path to tokenizer (defaults to model path)"}
    )
    use_auth_token: bool = field(
        default=True, metadata={"help": "Use HuggingFace auth token"}
    )
    trust_remote_code: bool = field(
        default=True, metadata={"help": "Trust remote code when loading model"}
    )
    torch_dtype: str = field(
        default="auto", metadata={"help": "Torch dtype for model loading"}
    )
    device_map: str = field(
        default="auto", metadata={"help": "Device mapping strategy"}
    )

    # Evaluation settings
    output_dir: str = field(
        default="./evaluation_results",
        metadata={"help": "Directory to save evaluation results"},
    )
    run_name: Optional[str] = field(
        default=None, metadata={"help": "Name for this evaluation run"}
    )
    benchmarks: List[BenchmarkConfig] = field(
        default_factory=list, metadata={"help": "List of benchmarks to run"}
    )

    # Performance settings
    batch_size: int = field(
        default=8, metadata={"help": "Default batch size for evaluation"}
    )
    max_length: int = field(default=2048, metadata={"help": "Maximum sequence length"})
    num_workers: int = field(
        default=4, metadata={"help": "Number of data loading workers"}
    )

    # Statistical settings
    confidence_level: float = field(
        default=0.95, metadata={"help": "Confidence level for statistical tests"}
    )
    bootstrap_samples: int = field(
        default=1000,
        metadata={"help": "Number of bootstrap samples for confidence intervals"},
    )

    # Reporting settings
    generate_report: bool = field(
        default=True, metadata={"help": "Generate comprehensive evaluation report"}
    )
    include_visualizations: bool = field(
        default=True, metadata={"help": "Include charts and visualizations in report"}
    )
    save_predictions: bool = field(
        default=False, metadata={"help": "Save model predictions for analysis"}
    )

    # Comparison settings
    baseline_results: Optional[str] = field(
        default=None, metadata={"help": "Path to baseline results for comparison"}
    )
    compare_with_published: bool = field(
        default=True, metadata={"help": "Compare with published benchmark results"}
    )

    # Safety and efficiency
    enable_safety_checks: bool = field(
        default=True, metadata={"help": "Enable safety and bias evaluations"}
    )
    enable_efficiency_metrics: bool = field(
        default=True, metadata={"help": "Enable performance and efficiency metrics"}
    )

    # Logging and monitoring
    log_level: str = field(default="INFO", metadata={"help": "Logging level"})
    use_wandb: bool = field(
        default=False, metadata={"help": "Log results to Weights & Biases"}
    )
    wandb_project: str = field(
        default="llm-evaluation", metadata={"help": "Weights & Biases project name"}
    )


def load_evaluation_config(config_path: str) -> EvaluationArguments:
    """
    Load evaluation configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        EvaluationArguments instance with loaded configuration
    """
    logger.info(f"Loading evaluation configuration from {config_path}")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    # Convert benchmark configs
    benchmarks = []
    for benchmark_dict in config_dict.get("benchmarks", []):
        # Convert metric configs
        metrics = []
        for metric_dict in benchmark_dict.get("metrics", []):
            metrics.append(MetricConfig(**metric_dict))

        benchmark_dict["metrics"] = metrics
        benchmarks.append(BenchmarkConfig(**benchmark_dict))

    config_dict["benchmarks"] = benchmarks

    return EvaluationArguments(**config_dict)


def save_evaluation_config(config: EvaluationArguments, config_path: str) -> None:
    """
    Save evaluation configuration to YAML file.

    Args:
        config: EvaluationArguments instance to save
        config_path: Path where to save the configuration
    """
    logger.info(f"Saving evaluation configuration to {config_path}")

    # Convert to dictionary
    config_dict = {}
    for field_name, field_def in config.__dataclass_fields__.items():
        value = getattr(config, field_name)

        if field_name == "benchmarks":
            # Convert benchmark configs to dicts
            benchmarks = []
            for benchmark in value:
                benchmark_dict = {}
                for b_field_name, b_field_def in benchmark.__dataclass_fields__.items():
                    b_value = getattr(benchmark, b_field_name)

                    if b_field_name == "metrics":
                        # Convert metric configs to dicts
                        metrics = []
                        for metric in b_value:
                            metric_dict = {}
                            for (
                                m_field_name,
                                m_field_def,
                            ) in metric.__dataclass_fields__.items():
                                metric_dict[m_field_name] = getattr(
                                    metric, m_field_name
                                )
                            metrics.append(metric_dict)
                        benchmark_dict[b_field_name] = metrics
                    else:
                        benchmark_dict[b_field_name] = b_value
                benchmarks.append(benchmark_dict)
            config_dict[field_name] = benchmarks
        else:
            config_dict[field_name] = value

    # Ensure output directory exists
    os.makedirs(os.path.dirname(config_path), exist_ok=True)

    with open(config_path, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False, indent=2)


def get_default_evaluation_config() -> EvaluationArguments:
    """
    Get default evaluation configuration with common benchmarks enabled.

    Returns:
        EvaluationArguments with sensible defaults
    """
    # Define default benchmarks
    default_benchmarks = [
        BenchmarkConfig(
            name="language_understanding",
            enabled=True,
            metrics=[
                MetricConfig(name="perplexity", enabled=True),
                MetricConfig(name="bleu", enabled=True),
                MetricConfig(name="rouge", enabled=True),
                MetricConfig(name="bertscore", enabled=True),
            ],
        ),
        BenchmarkConfig(
            name="mmlu",
            enabled=True,
            dataset_name="cais/mmlu",
            max_samples=1000,
            metrics=[
                MetricConfig(name="accuracy", enabled=True),
                MetricConfig(name="category_accuracy", enabled=True),
            ],
        ),
        BenchmarkConfig(
            name="hellaswag",
            enabled=True,
            dataset_name="Rowan/hellaswag",
            max_samples=1000,
            metrics=[
                MetricConfig(name="accuracy", enabled=True),
            ],
        ),
        BenchmarkConfig(
            name="performance",
            enabled=True,
            metrics=[
                MetricConfig(name="inference_speed", enabled=True),
                MetricConfig(name="memory_usage", enabled=True),
                MetricConfig(name="model_size", enabled=True),
            ],
        ),
    ]

    return EvaluationArguments(
        model_name_or_path="",  # Must be provided by user
        benchmarks=default_benchmarks,
    )


def create_benchmark_config(
    name: str,
    dataset_name: Optional[str] = None,
    metrics: Optional[List[str]] = None,
    **kwargs,
) -> BenchmarkConfig:
    """
    Helper function to create benchmark configurations.

    Args:
        name: Benchmark name
        dataset_name: HuggingFace dataset name
        metrics: List of metric names to enable
        **kwargs: Additional benchmark parameters

    Returns:
        BenchmarkConfig instance
    """
    if metrics is None:
        metrics = ["accuracy"]

    metric_configs = [MetricConfig(name=metric, enabled=True) for metric in metrics]

    return BenchmarkConfig(
        name=name, dataset_name=dataset_name, metrics=metric_configs, **kwargs
    )
