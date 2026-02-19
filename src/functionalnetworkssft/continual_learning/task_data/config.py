"""
Dataset configurations and task orders for continual learning evaluation.

Task orders follow the DOC paper (Zhang et al., 2025) Table 7:
  - Orders 1-3: Standard CL Benchmark (5 tasks)
  - Orders 4-6: Long Chain (15 tasks from GLUE + SuperGLUE + IMDB + CL Benchmark)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class DatasetConfig:
    """Configuration for a single CL evaluation dataset."""

    name: str
    source: str  # HuggingFace dataset name
    subset: Optional[str] = None  # Dataset subset/config name
    num_classes: int = 2
    label_map: Dict[int, str] = field(default_factory=dict)
    text_fields: List[str] = field(default_factory=lambda: ["text"])
    task_type: str = "classification"  # classification, nli, etc.


# ---------------------------------------------------------------------------
# Dataset definitions (15 total)
# ---------------------------------------------------------------------------

_CL_BENCHMARK_DATASETS: Dict[str, DatasetConfig] = {
    "ag_news": DatasetConfig(
        name="ag_news",
        source="ag_news",
        num_classes=4,
        label_map={0: "World", 1: "Sports", 2: "Business", 3: "Technology"},
        text_fields=["text"],
        task_type="topic",
    ),
    "yelp": DatasetConfig(
        name="yelp",
        source="yelp_review_full",
        num_classes=5,
        label_map={0: "1 star", 1: "2 stars", 2: "3 stars", 3: "4 stars", 4: "5 stars"},
        text_fields=["text"],
        task_type="sentiment",
    ),
    "amazon": DatasetConfig(
        name="amazon",
        source="amazon_polarity",
        num_classes=2,
        label_map={0: "negative", 1: "positive"},
        text_fields=["content"],
        task_type="sentiment",
    ),
    "dbpedia": DatasetConfig(
        name="dbpedia",
        source="dbpedia_14",
        num_classes=14,
        label_map={
            0: "Company", 1: "EducationalInstitution", 2: "Artist",
            3: "Athlete", 4: "OfficeHolder", 5: "MeanOfTransportation",
            6: "Building", 7: "NaturalPlace", 8: "Village", 9: "Animal",
            10: "Plant", 11: "Album", 12: "Film", 13: "WrittenWork",
        },
        text_fields=["content"],
        task_type="topic",
    ),
    "yahoo": DatasetConfig(
        name="yahoo",
        source="yahoo_answers_topics",
        num_classes=10,
        label_map={
            0: "Society", 1: "Science", 2: "Health", 3: "Education",
            4: "Computers", 5: "Sports", 6: "Business", 7: "Entertainment",
            8: "Family", 9: "Politics",
        },
        text_fields=["question_title", "question_content"],
        task_type="topic",
    ),
}

_GLUE_DATASETS: Dict[str, DatasetConfig] = {
    "sst2": DatasetConfig(
        name="sst2",
        source="glue",
        subset="sst2",
        num_classes=2,
        label_map={0: "negative", 1: "positive"},
        text_fields=["sentence"],
        task_type="sentiment",
    ),
    "mnli": DatasetConfig(
        name="mnli",
        source="glue",
        subset="mnli",
        num_classes=3,
        label_map={0: "entailment", 1: "neutral", 2: "contradiction"},
        text_fields=["premise", "hypothesis"],
        task_type="nli",
    ),
    "qqp": DatasetConfig(
        name="qqp",
        source="glue",
        subset="qqp",
        num_classes=2,
        label_map={0: "not duplicate", 1: "duplicate"},
        text_fields=["question1", "question2"],
        task_type="paraphrase",
    ),
    "rte": DatasetConfig(
        name="rte",
        source="glue",
        subset="rte",
        num_classes=2,
        label_map={0: "entailment", 1: "not entailment"},
        text_fields=["sentence1", "sentence2"],
        task_type="nli",
    ),
}

_SUPERGLUE_DATASETS: Dict[str, DatasetConfig] = {
    "boolq": DatasetConfig(
        name="boolq",
        source="super_glue",
        subset="boolq",
        num_classes=2,
        label_map={0: "false", 1: "true"},
        text_fields=["question", "passage"],
        task_type="qa",
    ),
    "cb": DatasetConfig(
        name="cb",
        source="super_glue",
        subset="cb",
        num_classes=3,
        label_map={0: "entailment", 1: "contradiction", 2: "neutral"},
        text_fields=["premise", "hypothesis"],
        task_type="nli",
    ),
    "copa": DatasetConfig(
        name="copa",
        source="super_glue",
        subset="copa",
        num_classes=2,
        label_map={0: "choice1", 1: "choice2"},
        text_fields=["premise", "choice1", "choice2"],
        task_type="causal",
    ),
    "wic": DatasetConfig(
        name="wic",
        source="super_glue",
        subset="wic",
        num_classes=2,
        label_map={0: "false", 1: "true"},
        text_fields=["sentence1", "sentence2", "word"],
        task_type="wsd",
    ),
    "multirc": DatasetConfig(
        name="multirc",
        source="super_glue",
        subset="multirc",
        num_classes=2,
        label_map={0: "false", 1: "true"},
        text_fields=["paragraph", "question", "answer"],
        task_type="qa",
    ),
}

_OTHER_DATASETS: Dict[str, DatasetConfig] = {
    "imdb": DatasetConfig(
        name="imdb",
        source="imdb",
        num_classes=2,
        label_map={0: "negative", 1: "positive"},
        text_fields=["text"],
        task_type="sentiment",
    ),
}

# ---------------------------------------------------------------------------
# ALL_DATASETS: combined registry
# ---------------------------------------------------------------------------

ALL_DATASETS: Dict[str, DatasetConfig] = {
    **_CL_BENCHMARK_DATASETS,
    **_GLUE_DATASETS,
    **_SUPERGLUE_DATASETS,
    **_OTHER_DATASETS,
}

# ---------------------------------------------------------------------------
# Task Orders (DOC paper Table 7)
# ---------------------------------------------------------------------------

TASK_ORDERS: Dict[str, List[str]] = {
    # Standard CL Benchmark — 5 tasks each
    "order_1": ["ag_news", "yelp", "amazon", "dbpedia", "yahoo"],
    "order_2": ["dbpedia", "yahoo", "ag_news", "amazon", "yelp"],
    "order_3": ["yelp", "yahoo", "amazon", "dbpedia", "ag_news"],
    # Long Chain — 15 tasks each
    "order_4": [
        "sst2", "mnli", "qqp", "rte", "boolq",
        "wic", "cb", "copa", "multirc", "imdb",
        "ag_news", "yelp", "amazon", "dbpedia", "yahoo",
    ],
    "order_5": [
        "ag_news", "sst2", "imdb", "yelp", "amazon",
        "boolq", "rte", "cb", "copa", "wic",
        "multirc", "mnli", "qqp", "dbpedia", "yahoo",
    ],
    "order_6": [
        "dbpedia", "yahoo", "mnli", "qqp", "rte",
        "sst2", "boolq", "wic", "cb", "copa",
        "multirc", "imdb", "ag_news", "yelp", "amazon",
    ],
}

_STANDARD_ORDERS = {"order_1", "order_2", "order_3"}


# ---------------------------------------------------------------------------
# Public helper functions
# ---------------------------------------------------------------------------


def get_dataset_config(task_name: str) -> DatasetConfig:
    """Return the DatasetConfig for a given task name.

    Args:
        task_name: Dataset identifier (e.g. "ag_news", "sst2").

    Raises:
        ValueError: If the task name is not recognized.
    """
    if task_name not in ALL_DATASETS:
        raise ValueError(
            f"Unknown dataset: {task_name}. "
            f"Available: {sorted(ALL_DATASETS.keys())}"
        )
    return ALL_DATASETS[task_name]


def get_task_order(order_name: str) -> List[str]:
    """Return the ordered list of task names for a task order.

    Args:
        order_name: Order identifier (e.g. "order_1").

    Raises:
        ValueError: If the order name is not recognized.
    """
    if order_name not in TASK_ORDERS:
        raise ValueError(
            f"Unknown task order: {order_name}. "
            f"Available: {sorted(TASK_ORDERS.keys())}"
        )
    return list(TASK_ORDERS[order_name])  # return a copy


def is_standard_benchmark(order_name: str) -> bool:
    """Return True if the order is a standard CL benchmark (5 tasks)."""
    return order_name in _STANDARD_ORDERS
