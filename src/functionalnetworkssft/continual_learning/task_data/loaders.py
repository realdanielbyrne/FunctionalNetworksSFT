"""
Dataset loading and tokenization for continual learning evaluation.

The ``CLDatasetLoader`` class handles:
  - Downloading / caching HuggingFace datasets
  - Formatting examples with task-specific prompts
  - Tokenizing for causal-LM training
  - **Crucially**: including the answer in training data (``include_answer=True``)
    but **excluding** it from test data (``include_answer=False``) to prevent
    evaluation data leakage.
"""

import logging
from typing import Any, Dict, Optional

from datasets import Dataset, load_dataset

from .config import DatasetConfig, get_dataset_config
from .prompts import format_example

logger = logging.getLogger(__name__)

# Default limits — keep manageable for CL experiments
_DEFAULT_TRAIN_SAMPLES = 1000
_DEFAULT_TEST_SAMPLES = 500


class CLDatasetLoader:
    """Load, format, tokenize, and cache CL benchmark datasets.

    Args:
        tokenizer: HuggingFace tokenizer instance.
        max_seq_length: Maximum sequence length for tokenization.
        seed: Random seed for reproducible shuffling / sampling.
        train_samples: Number of training examples per task.
        test_samples: Number of test examples per task.
    """

    def __init__(
        self,
        tokenizer,
        max_seq_length: int = 512,
        seed: int = 42,
        train_samples: int = _DEFAULT_TRAIN_SAMPLES,
        test_samples: int = _DEFAULT_TEST_SAMPLES,
    ):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.seed = seed
        self.train_samples = train_samples
        self.test_samples = test_samples
        self._cache: Dict[str, Dict[str, Dataset]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_dataset(self, task_name: str) -> Dict[str, Dataset]:
        """Load and prepare a dataset for CL training and evaluation.

        Returns:
            ``{"train": Dataset, "test": Dataset}`` where
            *train* has columns ``[input_ids, attention_mask, labels]`` and
            *test*  has columns ``[input_ids, attention_mask, label_idx]``.
        """
        if task_name in self._cache:
            return self._cache[task_name]

        config = get_dataset_config(task_name)
        logger.info(f"Loading dataset: {task_name} (source={config.source})")

        raw = self._download_raw(config)
        train_split, test_split = self._get_splits(raw, config)

        train_ds = self._prepare_train(train_split, config)
        test_ds = self._prepare_test(test_split, config)

        result = {"train": train_ds, "test": test_ds}
        self._cache[task_name] = result
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _download_raw(self, config: DatasetConfig):
        """Download the raw HuggingFace dataset."""
        kwargs: Dict[str, Any] = {}
        if config.subset:
            kwargs["name"] = config.subset
        return load_dataset(config.source, **kwargs, trust_remote_code=True)

    def _get_splits(self, raw, config: DatasetConfig):
        """Extract and sample train/test splits."""
        # Determine split names (some datasets use "validation" not "test")
        if "test" in raw:
            test_key = "test"
        elif "validation" in raw:
            test_key = "validation"
        elif "validation_matched" in raw:  # MNLI
            test_key = "validation_matched"
        else:
            # Fallback: split train 90/10
            split = raw["train"].train_test_split(
                test_size=0.1, seed=self.seed
            )
            return (
                self._sample(split["train"], self.train_samples),
                self._sample(split["test"], self.test_samples),
            )

        train_split = self._sample(raw["train"], self.train_samples)
        test_split = self._sample(raw[test_key], self.test_samples)
        return train_split, test_split

    def _sample(self, dataset: Dataset, n: int) -> Dataset:
        """Shuffle and take up to *n* examples."""
        dataset = dataset.shuffle(seed=self.seed)
        if len(dataset) > n:
            dataset = dataset.select(range(n))
        return dataset

    def _prepare_train(self, split: Dataset, config: DatasetConfig) -> Dataset:
        """Format + tokenize for training (answer IS included)."""

        def _process(example):
            formatted = format_example(example, config, include_answer=True)
            enc = self.tokenizer(
                formatted["full_text"],
                truncation=True,
                max_length=self.max_seq_length,
                padding="max_length",
            )
            enc["labels"] = enc["input_ids"].copy()
            return enc

        return split.map(_process, remove_columns=split.column_names)

    def _prepare_test(self, split: Dataset, config: DatasetConfig) -> Dataset:
        """Format + tokenize for evaluation (answer NOT included)."""

        def _process(example):
            formatted = format_example(example, config, include_answer=False)
            enc = self.tokenizer(
                formatted["full_text"],  # prompt only — no answer
                truncation=True,
                max_length=self.max_seq_length,
            )
            enc["label_idx"] = example.get("label", 0)
            return enc

        return split.map(_process, remove_columns=split.column_names)

