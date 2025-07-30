"""
Base metric classes and metric registry for the evaluation framework.

This module provides the foundation for implementing various evaluation metrics
with a consistent interface and automatic registration system.
"""

import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Callable
import numpy as np
from scipy import stats

from .results import MetricResult

logger = logging.getLogger(__name__)


class BaseMetric(ABC):
    """Base class for all evaluation metrics."""
    
    def __init__(self, name: str, **kwargs):
        """
        Initialize metric.
        
        Args:
            name: Name of the metric
            **kwargs: Metric-specific parameters
        """
        self.name = name
        self.parameters = kwargs
        self._reset()
    
    def _reset(self) -> None:
        """Reset metric state for new evaluation."""
        self.scores = []
        self.metadata = {}
        self.start_time = None
        self.end_time = None
    
    @abstractmethod
    def compute_score(self, predictions: Any, references: Any, **kwargs) -> float:
        """
        Compute metric score for given predictions and references.
        
        Args:
            predictions: Model predictions
            references: Ground truth references
            **kwargs: Additional arguments
            
        Returns:
            Metric score
        """
        pass
    
    def add_batch(self, predictions: Any, references: Any, **kwargs) -> None:
        """
        Add a batch of predictions and references for evaluation.
        
        Args:
            predictions: Batch of model predictions
            references: Batch of ground truth references
            **kwargs: Additional arguments
        """
        if self.start_time is None:
            self.start_time = time.time()
        
        # Handle both single items and batches
        if not isinstance(predictions, (list, tuple)):
            predictions = [predictions]
        if not isinstance(references, (list, tuple)):
            references = [references]
        
        for pred, ref in zip(predictions, references):
            try:
                score = self.compute_score(pred, ref, **kwargs)
                self.scores.append(score)
            except Exception as e:
                logger.warning(f"Error computing {self.name} for batch item: {e}")
                # Optionally add NaN or skip
                self.scores.append(float('nan'))
    
    def compute_final_score(self) -> MetricResult:
        """
        Compute final metric result with statistics.
        
        Returns:
            MetricResult with final score and statistics
        """
        self.end_time = time.time()
        
        if not self.scores:
            logger.warning(f"No scores available for metric {self.name}")
            return MetricResult(
                name=self.name,
                value=0.0,
                metadata={'error': 'No scores available'}
            )
        
        # Filter out NaN values
        valid_scores = [s for s in self.scores if not np.isnan(s)]
        
        if not valid_scores:
            logger.warning(f"No valid scores for metric {self.name}")
            return MetricResult(
                name=self.name,
                value=0.0,
                metadata={'error': 'No valid scores'}
            )
        
        # Compute statistics
        mean_score = np.mean(valid_scores)
        std_score = np.std(valid_scores) if len(valid_scores) > 1 else 0.0
        
        # Compute confidence interval using bootstrap
        confidence_interval = self._compute_confidence_interval(valid_scores)
        
        # Prepare metadata
        metadata = {
            'total_samples': len(self.scores),
            'valid_samples': len(valid_scores),
            'invalid_samples': len(self.scores) - len(valid_scores),
            'min_score': np.min(valid_scores),
            'max_score': np.max(valid_scores),
            'median_score': np.median(valid_scores),
            'computation_time': self.end_time - self.start_time if self.start_time else 0.0,
            **self.metadata,
            **self.parameters
        }
        
        return MetricResult(
            name=self.name,
            value=mean_score,
            std=std_score,
            confidence_interval=confidence_interval,
            sample_size=len(valid_scores),
            metadata=metadata
        )
    
    def _compute_confidence_interval(
        self, 
        scores: List[float], 
        confidence_level: float = 0.95,
        n_bootstrap: int = 1000
    ) -> Optional[tuple]:
        """
        Compute confidence interval using bootstrap resampling.
        
        Args:
            scores: List of scores
            confidence_level: Confidence level (default 0.95)
            n_bootstrap: Number of bootstrap samples
            
        Returns:
            Tuple of (lower_bound, upper_bound) or None if computation fails
        """
        if len(scores) < 2:
            return None
        
        try:
            # Bootstrap resampling
            bootstrap_means = []
            for _ in range(n_bootstrap):
                bootstrap_sample = np.random.choice(scores, size=len(scores), replace=True)
                bootstrap_means.append(np.mean(bootstrap_sample))
            
            # Compute percentiles
            alpha = 1 - confidence_level
            lower_percentile = (alpha / 2) * 100
            upper_percentile = (1 - alpha / 2) * 100
            
            lower_bound = np.percentile(bootstrap_means, lower_percentile)
            upper_bound = np.percentile(bootstrap_means, upper_percentile)
            
            return (lower_bound, upper_bound)
        
        except Exception as e:
            logger.warning(f"Failed to compute confidence interval: {e}")
            return None
    
    def reset(self) -> None:
        """Reset metric for new evaluation."""
        self._reset()


class MetricRegistry:
    """Registry for managing available metrics."""
    
    _metrics: Dict[str, Callable[..., BaseMetric]] = {}
    
    @classmethod
    def register(cls, name: str, metric_class: Callable[..., BaseMetric]) -> None:
        """
        Register a metric class.
        
        Args:
            name: Name to register the metric under
            metric_class: Metric class or factory function
        """
        cls._metrics[name] = metric_class
        logger.debug(f"Registered metric: {name}")
    
    @classmethod
    def get_metric(cls, name: str, **kwargs) -> BaseMetric:
        """
        Get a metric instance by name.
        
        Args:
            name: Name of the metric
            **kwargs: Parameters to pass to metric constructor
            
        Returns:
            Metric instance
            
        Raises:
            ValueError: If metric is not registered
        """
        if name not in cls._metrics:
            available = list(cls._metrics.keys())
            raise ValueError(f"Metric '{name}' not found. Available metrics: {available}")
        
        return cls._metrics[name](name=name, **kwargs)
    
    @classmethod
    def list_metrics(cls) -> List[str]:
        """
        List all registered metrics.
        
        Returns:
            List of metric names
        """
        return list(cls._metrics.keys())
    
    @classmethod
    def is_registered(cls, name: str) -> bool:
        """
        Check if a metric is registered.
        
        Args:
            name: Metric name
            
        Returns:
            True if registered, False otherwise
        """
        return name in cls._metrics


def register_metric(name: str):
    """
    Decorator for registering metrics.
    
    Args:
        name: Name to register the metric under
    """
    def decorator(metric_class):
        MetricRegistry.register(name, metric_class)
        return metric_class
    return decorator


class CompositeMetric(BaseMetric):
    """Metric that combines multiple sub-metrics."""
    
    def __init__(self, name: str, metrics: List[BaseMetric], weights: Optional[List[float]] = None):
        """
        Initialize composite metric.
        
        Args:
            name: Name of the composite metric
            metrics: List of sub-metrics
            weights: Optional weights for combining metrics (defaults to equal weights)
        """
        super().__init__(name)
        self.metrics = metrics
        self.weights = weights or [1.0] * len(metrics)
        
        if len(self.weights) != len(self.metrics):
            raise ValueError("Number of weights must match number of metrics")
    
    def compute_score(self, predictions: Any, references: Any, **kwargs) -> float:
        """Compute composite score from sub-metrics."""
        scores = []
        for metric in self.metrics:
            try:
                score = metric.compute_score(predictions, references, **kwargs)
                scores.append(score)
            except Exception as e:
                logger.warning(f"Error computing {metric.name}: {e}")
                scores.append(0.0)
        
        # Weighted average
        weighted_sum = sum(score * weight for score, weight in zip(scores, self.weights))
        total_weight = sum(self.weights)
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    
    def add_batch(self, predictions: Any, references: Any, **kwargs) -> None:
        """Add batch to all sub-metrics."""
        super().add_batch(predictions, references, **kwargs)
        
        # Also add to sub-metrics for individual tracking
        for metric in self.metrics:
            metric.add_batch(predictions, references, **kwargs)
    
    def compute_final_score(self) -> MetricResult:
        """Compute final score with sub-metric details."""
        result = super().compute_final_score()
        
        # Add sub-metric results to metadata
        sub_results = {}
        for metric in self.metrics:
            sub_result = metric.compute_final_score()
            sub_results[metric.name] = sub_result.to_dict()
        
        result.metadata['sub_metrics'] = sub_results
        return result


class AccuracyMetric(BaseMetric):
    """Simple accuracy metric for classification tasks."""
    
    def compute_score(self, predictions: Any, references: Any, **kwargs) -> float:
        """Compute accuracy score."""
        if isinstance(predictions, str) and isinstance(references, str):
            return 1.0 if predictions.strip().lower() == references.strip().lower() else 0.0
        elif hasattr(predictions, '__iter__') and hasattr(references, '__iter__'):
            # For sequence comparison
            pred_list = list(predictions)
            ref_list = list(references)
            if len(pred_list) != len(ref_list):
                return 0.0
            return sum(p == r for p, r in zip(pred_list, ref_list)) / len(pred_list)
        else:
            return 1.0 if predictions == references else 0.0


# Register built-in metrics
MetricRegistry.register("accuracy", AccuracyMetric)
