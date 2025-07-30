"""
Result management and storage for the evaluation framework.

This module provides classes for storing, managing, and analyzing evaluation results,
including statistical analysis and comparison capabilities.
"""

import json
import logging
import os
import pickle
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class MetricResult:
    """Result for a single metric evaluation."""
    
    name: str = field(metadata={"help": "Name of the metric"})
    value: float = field(metadata={"help": "Primary metric value"})
    std: Optional[float] = field(default=None, metadata={"help": "Standard deviation"})
    confidence_interval: Optional[tuple] = field(
        default=None, 
        metadata={"help": "Confidence interval (lower, upper)"}
    )
    sample_size: Optional[int] = field(
        default=None, 
        metadata={"help": "Number of samples used"}
    )
    metadata: Dict[str, Any] = field(
        default_factory=dict,
        metadata={"help": "Additional metric-specific information"}
    )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MetricResult':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class BenchmarkResult:
    """Result for a benchmark evaluation."""
    
    name: str = field(metadata={"help": "Name of the benchmark"})
    metrics: Dict[str, MetricResult] = field(
        default_factory=dict,
        metadata={"help": "Dictionary of metric results"}
    )
    duration: float = field(default=0.0, metadata={"help": "Evaluation duration in seconds"})
    samples_evaluated: int = field(default=0, metadata={"help": "Number of samples evaluated"})
    error_count: int = field(default=0, metadata={"help": "Number of errors encountered"})
    metadata: Dict[str, Any] = field(
        default_factory=dict,
        metadata={"help": "Additional benchmark-specific information"}
    )
    
    def add_metric(self, metric_result: MetricResult) -> None:
        """Add a metric result to this benchmark."""
        self.metrics[metric_result.name] = metric_result
    
    def get_metric(self, name: str) -> Optional[MetricResult]:
        """Get a specific metric result."""
        return self.metrics.get(name)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = asdict(self)
        # Convert metric results to dicts
        result['metrics'] = {name: metric.to_dict() for name, metric in self.metrics.items()}
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BenchmarkResult':
        """Create from dictionary."""
        metrics_data = data.pop('metrics', {})
        result = cls(**data)
        result.metrics = {
            name: MetricResult.from_dict(metric_data) 
            for name, metric_data in metrics_data.items()
        }
        return result


@dataclass
class EvaluationReport:
    """Complete evaluation report containing all results."""
    
    model_name: str = field(metadata={"help": "Name/path of the evaluated model"})
    timestamp: str = field(
        default_factory=lambda: datetime.now().isoformat(),
        metadata={"help": "Evaluation timestamp"}
    )
    benchmarks: Dict[str, BenchmarkResult] = field(
        default_factory=dict,
        metadata={"help": "Dictionary of benchmark results"}
    )
    total_duration: float = field(default=0.0, metadata={"help": "Total evaluation duration"})
    system_info: Dict[str, Any] = field(
        default_factory=dict,
        metadata={"help": "System and environment information"}
    )
    config: Dict[str, Any] = field(
        default_factory=dict,
        metadata={"help": "Evaluation configuration used"}
    )
    
    def add_benchmark(self, benchmark_result: BenchmarkResult) -> None:
        """Add a benchmark result to this report."""
        self.benchmarks[benchmark_result.name] = benchmark_result
    
    def get_benchmark(self, name: str) -> Optional[BenchmarkResult]:
        """Get a specific benchmark result."""
        return self.benchmarks.get(name)
    
    def get_summary_metrics(self) -> Dict[str, float]:
        """Get summary metrics across all benchmarks."""
        summary = {}
        
        # Collect all metrics
        all_metrics = {}
        for benchmark_name, benchmark in self.benchmarks.items():
            for metric_name, metric in benchmark.metrics.items():
                key = f"{benchmark_name}_{metric_name}"
                all_metrics[key] = metric.value
        
        if all_metrics:
            summary['mean_score'] = np.mean(list(all_metrics.values()))
            summary['median_score'] = np.median(list(all_metrics.values()))
            summary['std_score'] = np.std(list(all_metrics.values()))
            summary['total_benchmarks'] = len(self.benchmarks)
            summary['total_metrics'] = len(all_metrics)
        
        return summary
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = asdict(self)
        # Convert benchmark results to dicts
        result['benchmarks'] = {
            name: benchmark.to_dict() 
            for name, benchmark in self.benchmarks.items()
        }
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EvaluationReport':
        """Create from dictionary."""
        benchmarks_data = data.pop('benchmarks', {})
        report = cls(**data)
        report.benchmarks = {
            name: BenchmarkResult.from_dict(benchmark_data)
            for name, benchmark_data in benchmarks_data.items()
        }
        return report


class ResultManager:
    """Manages saving, loading, and analyzing evaluation results."""
    
    def __init__(self, output_dir: str):
        """
        Initialize result manager.
        
        Args:
            output_dir: Directory to save results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / "reports").mkdir(exist_ok=True)
        (self.output_dir / "raw_data").mkdir(exist_ok=True)
        (self.output_dir / "visualizations").mkdir(exist_ok=True)
    
    def save_report(self, report: EvaluationReport, filename: Optional[str] = None) -> str:
        """
        Save evaluation report to JSON file.
        
        Args:
            report: EvaluationReport to save
            filename: Optional filename (auto-generated if None)
            
        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = report.model_name.replace("/", "_").replace("\\", "_")
            filename = f"evaluation_report_{model_name}_{timestamp}.json"
        
        filepath = self.output_dir / "reports" / filename
        
        with open(filepath, 'w') as f:
            json.dump(report.to_dict(), f, indent=2, default=str)
        
        logger.info(f"Evaluation report saved to {filepath}")
        return str(filepath)
    
    def load_report(self, filepath: str) -> EvaluationReport:
        """
        Load evaluation report from JSON file.
        
        Args:
            filepath: Path to report file
            
        Returns:
            EvaluationReport instance
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        return EvaluationReport.from_dict(data)
    
    def save_raw_data(self, data: Any, filename: str) -> str:
        """
        Save raw evaluation data (predictions, etc.) using pickle.
        
        Args:
            data: Data to save
            filename: Filename for the data
            
        Returns:
            Path to saved file
        """
        filepath = self.output_dir / "raw_data" / filename
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        logger.info(f"Raw data saved to {filepath}")
        return str(filepath)
    
    def load_raw_data(self, filename: str) -> Any:
        """
        Load raw evaluation data from pickle file.
        
        Args:
            filename: Filename of the data
            
        Returns:
            Loaded data
        """
        filepath = self.output_dir / "raw_data" / filename
        
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    
    def compare_reports(
        self, 
        report1: EvaluationReport, 
        report2: EvaluationReport
    ) -> Dict[str, Any]:
        """
        Compare two evaluation reports.
        
        Args:
            report1: First report
            report2: Second report
            
        Returns:
            Comparison results
        """
        comparison = {
            'model1': report1.model_name,
            'model2': report2.model_name,
            'timestamp1': report1.timestamp,
            'timestamp2': report2.timestamp,
            'metric_comparisons': {},
            'summary': {}
        }
        
        # Compare common benchmarks
        common_benchmarks = set(report1.benchmarks.keys()) & set(report2.benchmarks.keys())
        
        for benchmark_name in common_benchmarks:
            bench1 = report1.benchmarks[benchmark_name]
            bench2 = report2.benchmarks[benchmark_name]
            
            # Compare common metrics
            common_metrics = set(bench1.metrics.keys()) & set(bench2.metrics.keys())
            
            for metric_name in common_metrics:
                metric1 = bench1.metrics[metric_name]
                metric2 = bench2.metrics[metric_name]
                
                key = f"{benchmark_name}_{metric_name}"
                comparison['metric_comparisons'][key] = {
                    'value1': metric1.value,
                    'value2': metric2.value,
                    'difference': metric2.value - metric1.value,
                    'percent_change': ((metric2.value - metric1.value) / metric1.value * 100) 
                                    if metric1.value != 0 else float('inf')
                }
        
        # Summary statistics
        if comparison['metric_comparisons']:
            differences = [comp['difference'] for comp in comparison['metric_comparisons'].values()]
            comparison['summary'] = {
                'mean_difference': np.mean(differences),
                'median_difference': np.median(differences),
                'std_difference': np.std(differences),
                'improvements': sum(1 for d in differences if d > 0),
                'degradations': sum(1 for d in differences if d < 0),
                'total_comparisons': len(differences)
            }
        
        return comparison
    
    def list_reports(self) -> List[str]:
        """
        List all available evaluation reports.
        
        Returns:
            List of report filenames
        """
        reports_dir = self.output_dir / "reports"
        return [f.name for f in reports_dir.glob("*.json")]
