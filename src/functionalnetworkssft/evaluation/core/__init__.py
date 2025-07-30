"""
Core evaluation infrastructure for FunctionalNetworksSFT.

This module provides the base classes and utilities for the evaluation framework.
"""

from .evaluator import BaseEvaluator, ModelEvaluator, EvaluationConfig, EvaluationResult
from .config import EvaluationArguments, BenchmarkConfig, MetricConfig
from .results import ResultManager, EvaluationReport, MetricResult
from .metrics import BaseMetric, MetricRegistry

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
    "BaseMetric",
    "MetricRegistry",
]
