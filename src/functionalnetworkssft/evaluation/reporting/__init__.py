"""
Reporting and visualization module for the evaluation framework.

This module provides comprehensive reporting capabilities including:
- HTML and PDF report generation
- Statistical analysis and confidence intervals
- Interactive visualizations and charts
- Comparison with baseline results
"""

from .generators import (
    ReportGenerator,
    HTMLReportGenerator,
    MarkdownReportGenerator,
)
from .visualizations import (
    VisualizationGenerator,
    create_metric_comparison_chart,
    create_benchmark_summary_chart,
    create_performance_timeline,
)
from .statistics import (
    StatisticalAnalyzer,
    compute_confidence_intervals,
    perform_significance_tests,
    generate_summary_statistics,
)

__all__ = [
    "ReportGenerator",
    "HTMLReportGenerator", 
    "MarkdownReportGenerator",
    "VisualizationGenerator",
    "create_metric_comparison_chart",
    "create_benchmark_summary_chart",
    "create_performance_timeline",
    "StatisticalAnalyzer",
    "compute_confidence_intervals",
    "perform_significance_tests",
    "generate_summary_statistics",
]
