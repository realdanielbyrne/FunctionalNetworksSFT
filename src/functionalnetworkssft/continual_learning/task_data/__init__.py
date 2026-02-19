"""
Dataset configurations, prompt templates, and loaders for CL evaluation.
"""

from .config import (
    ALL_DATASETS,
    TASK_ORDERS,
    DatasetConfig,
    get_dataset_config,
    get_task_order,
    is_standard_benchmark,
)
from .loaders import CLDatasetLoader
from .prompts import create_chat_format, format_example

__all__ = [
    "ALL_DATASETS",
    "CLDatasetLoader",
    "DatasetConfig",
    "TASK_ORDERS",
    "create_chat_format",
    "format_example",
    "get_dataset_config",
    "get_task_order",
    "is_standard_benchmark",
]

