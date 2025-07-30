"""
FunctionalNetworksSFT Evaluation Framework

A comprehensive evaluation framework for assessing fine-tuned language models
using standard LLM benchmarks, performance metrics, and safety evaluations.

This framework integrates seamlessly with the existing FunctionalNetworksSFT
training pipeline and supports:

- Core Language Understanding Metrics (Perplexity, BLEU, ROUGE, BERTScore)
- Standard LLM Benchmarks (MMLU, HellaSwag, ARC, GSM8K, HumanEval)
- Performance and Efficiency Metrics (Speed, Memory, FLOPS)
- Safety and Bias Evaluation (Toxicity, Bias Assessment)

Author: Daniel Byrne
License: MIT
"""

from .core.evaluator import (
    BaseEvaluator,
    ModelEvaluator,
    EvaluationConfig,
    EvaluationResult,
)
from .core.config import (
    EvaluationArguments,
    BenchmarkConfig,
    MetricConfig,
)
from .core.results import (
    ResultManager,
    EvaluationReport,
    MetricResult,
)

__version__ = "0.1.0"
__all__ = [
    "BaseEvaluator",
    "ModelEvaluator", 
    "EvaluationConfig",
    "EvaluationResult",
    "EvaluationArguments",
    "BenchmarkConfig",
    "MetricConfig",
    "ResultManager",
    "EvaluationReport",
    "MetricResult",
]
