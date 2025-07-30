"""
Statistical analysis utilities for evaluation results.

This module provides statistical analysis capabilities including confidence intervals,
significance testing, and summary statistics for evaluation metrics.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from scipy import stats
from scipy.stats import bootstrap

from ..core.results import EvaluationReport, BenchmarkResult, MetricResult

logger = logging.getLogger(__name__)


class StatisticalAnalyzer:
    """Provides statistical analysis for evaluation results."""

    def __init__(self, confidence_level: float = 0.95):
        """
        Initialize statistical analyzer.

        Args:
            confidence_level: Confidence level for intervals and tests
        """
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level

    def analyze_report(self, report: EvaluationReport) -> Dict[str, Any]:
        """
        Perform comprehensive statistical analysis of an evaluation report.

        Args:
            report: EvaluationReport to analyze

        Returns:
            Dictionary containing statistical analysis results
        """
        analysis = {
            "summary_statistics": self.compute_summary_statistics(report),
            "confidence_intervals": self.compute_all_confidence_intervals(report),
            "metric_correlations": self.compute_metric_correlations(report),
            "benchmark_rankings": self.rank_benchmarks(report),
        }

        return analysis

    def compute_summary_statistics(self, report: EvaluationReport) -> Dict[str, Any]:
        """Compute summary statistics across all metrics."""
        all_values = []
        metric_stats = {}

        for benchmark_name, benchmark in report.benchmarks.items():
            for metric_name, metric in benchmark.metrics.items():
                key = f"{benchmark_name}_{metric_name}"
                all_values.append(metric.value)

                metric_stats[key] = {
                    "value": metric.value,
                    "std": metric.std,
                    "sample_size": metric.sample_size,
                    "confidence_interval": metric.confidence_interval,
                }

        if all_values:
            summary = {
                "total_metrics": len(all_values),
                "mean_score": np.mean(all_values),
                "median_score": np.median(all_values),
                "std_score": np.std(all_values),
                "min_score": np.min(all_values),
                "max_score": np.max(all_values),
                "q25_score": np.percentile(all_values, 25),
                "q75_score": np.percentile(all_values, 75),
                "metric_details": metric_stats,
            }
        else:
            summary = {"total_metrics": 0, "metric_details": {}}

        return summary

    def compute_all_confidence_intervals(
        self, report: EvaluationReport
    ) -> Dict[str, Tuple[float, float]]:
        """Compute confidence intervals for all metrics."""
        intervals = {}

        for benchmark_name, benchmark in report.benchmarks.items():
            for metric_name, metric in benchmark.metrics.items():
                key = f"{benchmark_name}_{metric_name}"

                if metric.confidence_interval:
                    intervals[key] = metric.confidence_interval
                elif metric.std and metric.sample_size and metric.sample_size > 1:
                    # Compute using t-distribution
                    interval = self.compute_t_confidence_interval(
                        metric.value, metric.std, metric.sample_size
                    )
                    intervals[key] = interval

        return intervals

    def compute_t_confidence_interval(
        self, mean: float, std: float, n: int
    ) -> Tuple[float, float]:
        """
        Compute confidence interval using t-distribution.

        Args:
            mean: Sample mean
            std: Sample standard deviation
            n: Sample size

        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        if n <= 1:
            return (mean, mean)

        # Standard error
        se = std / np.sqrt(n)

        # t-critical value
        df = n - 1
        t_critical = stats.t.ppf(1 - self.alpha / 2, df)

        # Confidence interval
        margin_error = t_critical * se
        lower_bound = mean - margin_error
        upper_bound = mean + margin_error

        return (lower_bound, upper_bound)

    def compute_metric_correlations(self, report: EvaluationReport) -> Dict[str, float]:
        """Compute correlations between different metrics."""
        # Collect all metric values
        metric_data = {}

        for benchmark_name, benchmark in report.benchmarks.items():
            for metric_name, metric in benchmark.metrics.items():
                key = f"{benchmark_name}_{metric_name}"
                metric_data[key] = metric.value

        # Compute pairwise correlations
        correlations = {}
        metric_names = list(metric_data.keys())

        for i, metric1 in enumerate(metric_names):
            for j, metric2 in enumerate(metric_names[i + 1 :], i + 1):
                # For single values, correlation is not meaningful
                # This would be more useful with multiple evaluation runs
                correlations[f"{metric1}_vs_{metric2}"] = 0.0

        return correlations

    def rank_benchmarks(self, report: EvaluationReport) -> List[Dict[str, Any]]:
        """Rank benchmarks by average performance."""
        benchmark_scores = []

        for benchmark_name, benchmark in report.benchmarks.items():
            if benchmark.metrics:
                scores = [metric.value for metric in benchmark.metrics.values()]
                avg_score = np.mean(scores)
                std_score = np.std(scores) if len(scores) > 1 else 0.0

                benchmark_scores.append(
                    {
                        "benchmark": benchmark_name,
                        "average_score": avg_score,
                        "std_score": std_score,
                        "num_metrics": len(scores),
                        "duration": benchmark.duration,
                    }
                )

        # Sort by average score (descending)
        benchmark_scores.sort(key=lambda x: x["average_score"], reverse=True)

        return benchmark_scores

    def compare_reports(
        self, report1: EvaluationReport, report2: EvaluationReport
    ) -> Dict[str, Any]:
        """
        Compare two evaluation reports statistically.

        Args:
            report1: First evaluation report
            report2: Second evaluation report

        Returns:
            Statistical comparison results
        """
        comparison = {
            "model1": report1.model_name,
            "model2": report2.model_name,
            "metric_comparisons": {},
            "significance_tests": {},
            "effect_sizes": {},
            "summary": {},
        }

        # Find common metrics
        common_metrics = self._find_common_metrics(report1, report2)

        for metric_key in common_metrics:
            metric1 = self._get_metric_by_key(report1, metric_key)
            metric2 = self._get_metric_by_key(report2, metric_key)

            if metric1 and metric2:
                # Basic comparison
                comparison["metric_comparisons"][metric_key] = {
                    "value1": metric1.value,
                    "value2": metric2.value,
                    "difference": metric2.value - metric1.value,
                    "percent_change": (
                        ((metric2.value - metric1.value) / metric1.value * 100)
                        if metric1.value != 0
                        else float("inf")
                    ),
                    "std1": metric1.std,
                    "std2": metric2.std,
                }

                # Effect size (Cohen's d)
                if (
                    metric1.std
                    and metric2.std
                    and metric1.sample_size
                    and metric2.sample_size
                ):
                    effect_size = self._compute_cohens_d(
                        metric1.value,
                        metric1.std,
                        metric1.sample_size,
                        metric2.value,
                        metric2.std,
                        metric2.sample_size,
                    )
                    comparison["effect_sizes"][metric_key] = effect_size

        # Summary statistics
        if comparison["metric_comparisons"]:
            differences = [
                comp["difference"] for comp in comparison["metric_comparisons"].values()
            ]
            percent_changes = [
                comp["percent_change"]
                for comp in comparison["metric_comparisons"].values()
                if comp["percent_change"] != float("inf")
            ]

            comparison["summary"] = {
                "total_comparisons": len(differences),
                "mean_difference": np.mean(differences),
                "median_difference": np.median(differences),
                "std_difference": np.std(differences),
                "improvements": sum(1 for d in differences if d > 0),
                "degradations": sum(1 for d in differences if d < 0),
                "mean_percent_change": (
                    np.mean(percent_changes) if percent_changes else 0.0
                ),
            }

        return comparison

    def _find_common_metrics(
        self, report1: EvaluationReport, report2: EvaluationReport
    ) -> List[str]:
        """Find metrics that exist in both reports."""
        metrics1 = set()
        metrics2 = set()

        for benchmark_name, benchmark in report1.benchmarks.items():
            for metric_name in benchmark.metrics.keys():
                metrics1.add(f"{benchmark_name}_{metric_name}")

        for benchmark_name, benchmark in report2.benchmarks.items():
            for metric_name in benchmark.metrics.keys():
                metrics2.add(f"{benchmark_name}_{metric_name}")

        return list(metrics1 & metrics2)

    def _get_metric_by_key(
        self, report: EvaluationReport, key: str
    ) -> Optional[MetricResult]:
        """Get metric result by key (benchmark_metric format)."""
        parts = key.split("_", 1)
        if len(parts) != 2:
            return None

        benchmark_name, metric_name = parts
        benchmark = report.benchmarks.get(benchmark_name)
        if benchmark:
            return benchmark.metrics.get(metric_name)

        return None

    def _compute_cohens_d(
        self, mean1: float, std1: float, n1: int, mean2: float, std2: float, n2: int
    ) -> float:
        """Compute Cohen's d effect size."""
        # Pooled standard deviation
        pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))

        if pooled_std == 0:
            return 0.0

        # Cohen's d
        cohens_d = (mean2 - mean1) / pooled_std
        return cohens_d


def compute_confidence_intervals(
    values: List[float], confidence_level: float = 0.95, method: str = "bootstrap"
) -> Tuple[float, float]:
    """
    Compute confidence intervals for a list of values.

    Args:
        values: List of values
        confidence_level: Confidence level (default 0.95)
        method: Method to use ('bootstrap', 't_test')

    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    if len(values) < 2:
        mean_val = np.mean(values) if values else 0.0
        return (mean_val, mean_val)

    if method == "bootstrap":
        # Bootstrap confidence interval
        data = np.array(values)

        def statistic(x):
            return np.mean(x)

        try:
            res = bootstrap(
                (data,), statistic, n_resamples=1000, confidence_level=confidence_level
            )
            return (res.confidence_interval.low, res.confidence_interval.high)
        except Exception:
            # Fallback to t-test method if bootstrap fails
            return compute_confidence_intervals(
                values, confidence_level, method="t_test"
            )

    elif method == "t_test":
        # t-distribution confidence interval
        mean_val = np.mean(values)
        std_val = np.std(values, ddof=1)
        n = len(values)

        alpha = 1 - confidence_level
        t_critical = stats.t.ppf(1 - alpha / 2, n - 1)
        margin_error = t_critical * std_val / np.sqrt(n)

        return (mean_val - margin_error, mean_val + margin_error)

    else:
        raise ValueError(f"Unknown method: {method}")


def perform_significance_tests(
    values1: List[float], values2: List[float], test_type: str = "welch"
) -> Dict[str, float]:
    """
    Perform significance tests between two groups of values.

    Args:
        values1: First group of values
        values2: Second group of values
        test_type: Type of test ('welch', 'student', 'mann_whitney')

    Returns:
        Dictionary with test results
    """
    if len(values1) < 2 or len(values2) < 2:
        return {"p_value": 1.0, "statistic": 0.0, "test_type": test_type}

    if test_type == "welch":
        # Welch's t-test (unequal variances)
        statistic, p_value = stats.ttest_ind(values1, values2, equal_var=False)
    elif test_type == "student":
        # Student's t-test (equal variances)
        statistic, p_value = stats.ttest_ind(values1, values2, equal_var=True)
    elif test_type == "mann_whitney":
        # Mann-Whitney U test (non-parametric)
        statistic, p_value = stats.mannwhitneyu(
            values1, values2, alternative="two-sided"
        )
    else:
        raise ValueError(f"Unknown test type: {test_type}")

    return {
        "p_value": p_value,
        "statistic": statistic,
        "test_type": test_type,
        "significant": p_value < 0.05,
    }


def generate_summary_statistics(values: List[float]) -> Dict[str, float]:
    """
    Generate comprehensive summary statistics for a list of values.

    Args:
        values: List of values

    Returns:
        Dictionary with summary statistics
    """
    if not values:
        return {}

    values_array = np.array(values)

    return {
        "count": len(values),
        "mean": np.mean(values_array),
        "median": np.median(values_array),
        "std": np.std(values_array, ddof=1) if len(values) > 1 else 0.0,
        "var": np.var(values_array, ddof=1) if len(values) > 1 else 0.0,
        "min": np.min(values_array),
        "max": np.max(values_array),
        "q25": np.percentile(values_array, 25),
        "q75": np.percentile(values_array, 75),
        "iqr": np.percentile(values_array, 75) - np.percentile(values_array, 25),
        "skewness": stats.skew(values_array),
        "kurtosis": stats.kurtosis(values_array),
    }
