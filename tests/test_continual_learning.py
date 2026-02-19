"""
Tests for the continual learning evaluation framework.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest


class TestContinualLearningMetrics:
    """Tests for ContinualLearningMetrics class."""

    def test_metrics_initialization(self):
        """Test metrics class initialization."""
        from functionalnetworkssft.continual_learning import ContinualLearningMetrics

        metrics = ContinualLearningMetrics(num_tasks=5)
        assert metrics.num_tasks == 5
        assert metrics.accuracy_matrix.shape == (5, 5)
        assert np.all(metrics.accuracy_matrix == 0)  # Initialized with zeros

    def test_record_accuracy(self):
        """Test recording accuracy values."""
        from functionalnetworkssft.continual_learning import ContinualLearningMetrics

        metrics = ContinualLearningMetrics(num_tasks=3)
        metrics.record_accuracy(task_idx=0, training_stage=0, accuracy=85.0)
        metrics.record_accuracy(task_idx=0, training_stage=1, accuracy=80.0)
        metrics.record_accuracy(task_idx=1, training_stage=1, accuracy=90.0)

        assert metrics.accuracy_matrix[0, 0] == 85.0
        assert metrics.accuracy_matrix[0, 1] == 80.0
        assert metrics.accuracy_matrix[1, 1] == 90.0

    def test_average_accuracy(self):
        """Test average accuracy computation."""
        from functionalnetworkssft.continual_learning import ContinualLearningMetrics

        metrics = ContinualLearningMetrics(num_tasks=3)
        metrics.record_accuracy(0, 0, 80.0)
        metrics.record_accuracy(0, 1, 75.0)
        metrics.record_accuracy(1, 1, 85.0)
        metrics.record_accuracy(0, 2, 70.0)
        metrics.record_accuracy(1, 2, 80.0)
        metrics.record_accuracy(2, 2, 90.0)

        aa = metrics.compute_average_accuracy(T=3)
        expected = (70.0 + 80.0 + 90.0) / 3
        assert abs(aa - expected) < 0.01

    def test_backward_transfer(self):
        """Test backward transfer computation."""
        from functionalnetworkssft.continual_learning import ContinualLearningMetrics

        metrics = ContinualLearningMetrics(num_tasks=3)
        metrics.record_accuracy(0, 0, 80.0)
        metrics.record_accuracy(0, 1, 75.0)
        metrics.record_accuracy(1, 1, 85.0)
        metrics.record_accuracy(0, 2, 70.0)
        metrics.record_accuracy(1, 2, 80.0)
        metrics.record_accuracy(2, 2, 90.0)

        bwt = metrics.compute_backward_transfer(T=3)
        # BWT = (1/(T-1)) * sum_{t=0}^{T-2} (a[t,T-1] - a[t,t])
        # = (1/2) * ((70-80) + (80-85)) = (1/2) * (-10 + -5) = -7.5
        expected = ((70.0 - 80.0) + (80.0 - 85.0)) / 2
        assert abs(bwt - expected) < 0.01

    def test_forward_transfer(self):
        """Test forward transfer computation."""
        from functionalnetworkssft.continual_learning import ContinualLearningMetrics

        metrics = ContinualLearningMetrics(num_tasks=3)
        # Set baseline accuracies using the correct API
        metrics.set_baseline_accuracy(0, 50.0)
        metrics.set_baseline_accuracy(1, 55.0)
        metrics.set_baseline_accuracy(2, 60.0)

        metrics.record_accuracy(0, 0, 80.0)
        metrics.record_accuracy(1, 1, 85.0)
        metrics.record_accuracy(2, 2, 90.0)

        fwt = metrics.compute_forward_transfer(T=3)
        # FWT = (1/(T-1)) * sum_{t=1}^{T-1} (a[t,t] - baseline[t])
        # = (1/2) * ((85-55) + (90-60)) = (1/2) * (30 + 30) = 30
        expected = ((85.0 - 55.0) + (90.0 - 60.0)) / 2
        assert abs(fwt - expected) < 0.01

    def test_save_and_load(self):
        """Test saving and loading metrics."""
        import tempfile
        from pathlib import Path

        from functionalnetworkssft.continual_learning import ContinualLearningMetrics

        metrics = ContinualLearningMetrics(
            num_tasks=3, task_names=["task_a", "task_b", "task_c"]
        )
        metrics.record_accuracy(0, 0, 80.0)
        metrics.record_accuracy(1, 1, 85.0)
        metrics.set_baseline_accuracy(0, 50.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "metrics.json"
            metrics.save(filepath)

            loaded = ContinualLearningMetrics.load(filepath)
            assert loaded.num_tasks == 3
            assert loaded.task_names == ["task_a", "task_b", "task_c"]
            assert loaded.accuracy_matrix[0, 0] == 80.0
            assert loaded.baseline_accuracies[0] == 50.0


class TestDatasetConfig:
    """Tests for dataset configuration."""

    def test_get_dataset_config(self):
        """Test getting dataset configuration."""
        from functionalnetworkssft.continual_learning.task_data.config import (
            get_dataset_config,
        )

        config = get_dataset_config("ag_news")
        assert config.name == "ag_news"
        assert config.num_classes == 4
        assert len(config.label_map) == 4

    def test_get_task_order(self):
        """Test getting task order."""
        from functionalnetworkssft.continual_learning.task_data.config import (
            get_task_order,
        )

        order = get_task_order("order_1")
        assert len(order) == 5  # order_1 has 5 tasks (CL Benchmark)
        assert "ag_news" in order
        assert "yelp" in order
        assert "dbpedia" in order

    def test_get_long_task_order(self):
        """Test getting long chain task order."""
        from functionalnetworkssft.continual_learning.task_data.config import (
            get_task_order,
        )

        order = get_task_order("order_4")
        assert len(order) == 15  # Long chain has 15 tasks

    def test_invalid_dataset(self):
        """Test error handling for invalid dataset."""
        from functionalnetworkssft.continual_learning.task_data.config import (
            get_dataset_config,
        )

        with pytest.raises(ValueError):
            get_dataset_config("nonexistent_dataset")

    def test_invalid_task_order(self):
        """Test error handling for invalid task order."""
        from functionalnetworkssft.continual_learning.task_data.config import (
            get_task_order,
        )

        with pytest.raises(ValueError):
            get_task_order("nonexistent_order")

    def test_is_standard_benchmark(self):
        """Test standard benchmark detection."""
        from functionalnetworkssft.continual_learning.task_data.config import (
            is_standard_benchmark,
        )

        assert is_standard_benchmark("order_1") is True
        assert is_standard_benchmark("order_2") is True
        assert is_standard_benchmark("order_3") is True
        assert is_standard_benchmark("order_4") is False
        assert is_standard_benchmark("order_5") is False


class TestPromptTemplates:
    """Tests for prompt templates."""

    def test_format_example_sentiment(self):
        """Test sentiment analysis prompt formatting."""
        from functionalnetworkssft.continual_learning.task_data.config import (
            get_dataset_config,
        )
        from functionalnetworkssft.continual_learning.task_data.prompts import (
            format_example,
        )

        config = get_dataset_config("sst2")
        example = {"sentence": "This movie is fantastic!", "label": 1}

        result = format_example(example, config, include_answer=True)

        assert "prompt" in result
        assert "answer" in result
        assert "full_text" in result
        assert "This movie is fantastic!" in result["prompt"]
        assert "positive" in result["answer"]

    def test_format_example_topic_classification(self):
        """Test topic classification prompt formatting."""
        from functionalnetworkssft.continual_learning.task_data.config import (
            get_dataset_config,
        )
        from functionalnetworkssft.continual_learning.task_data.prompts import (
            format_example,
        )

        config = get_dataset_config("ag_news")
        example = {"text": "The stock market rose sharply today.", "label": 2}

        result = format_example(example, config, include_answer=True)

        assert "prompt" in result
        assert "The stock market rose sharply today." in result["prompt"]
        assert "Business" in result["answer"]

    def test_format_example_nli(self):
        """Test NLI prompt formatting."""
        from functionalnetworkssft.continual_learning.task_data.config import (
            get_dataset_config,
        )
        from functionalnetworkssft.continual_learning.task_data.prompts import (
            format_example,
        )

        config = get_dataset_config("mnli")
        example = {
            "premise": "The cat sat on the mat.",
            "hypothesis": "An animal is on the mat.",
            "label": 0,
        }

        result = format_example(example, config, include_answer=True)

        assert "The cat sat on the mat." in result["prompt"]
        assert "An animal is on the mat." in result["prompt"]
        assert "entailment" in result["answer"]

    def test_create_chat_format(self):
        """Test chat format creation."""
        from functionalnetworkssft.continual_learning.task_data.prompts import (
            create_chat_format,
        )

        messages = create_chat_format(
            prompt="What is 2+2?", answer="4", system_message="You are helpful."
        )

        assert len(messages) == 3
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert messages[2]["role"] == "assistant"


class TestCLMethods:
    """Tests for continual learning methods."""

    def test_method_registry(self):
        """Test that all methods are registered."""
        from functionalnetworkssft.continual_learning.evaluation import METHODS

        assert "lora" in METHODS
        assert "ewc" in METHODS
        assert "lwf" in METHODS
        assert "o_lora" in METHODS
        assert "doc" in METHODS
        assert "ica_networks" in METHODS

    def test_model_configs(self):
        """Test model configurations."""
        from functionalnetworkssft.continual_learning.evaluation import MODEL_CONFIGS

        assert "llama-3.2-1b" in MODEL_CONFIGS
        assert "llama-7b" in MODEL_CONFIGS
        assert "llama-13b" in MODEL_CONFIGS
        assert "t5-large" in MODEL_CONFIGS
        config = MODEL_CONFIGS["llama-3.2-1b"]
        assert "model_name" in config
        assert "learning_rate" in config
        assert "lora_r" in config

    def test_lora_baseline_import(self):
        """Test LoRA baseline can be imported."""
        from functionalnetworkssft.continual_learning.methods import LoRABaseline

        assert LoRABaseline is not None

    def test_ewc_import(self):
        """Test EWC can be imported."""
        from functionalnetworkssft.continual_learning.methods import EWC

        assert EWC is not None

    def test_lwf_import(self):
        """Test LwF can be imported."""
        from functionalnetworkssft.continual_learning.methods import LwF

        assert LwF is not None

    def test_o_lora_import(self):
        """Test O-LoRA can be imported."""
        from functionalnetworkssft.continual_learning.methods import OLoRA

        assert OLoRA is not None

    def test_doc_import(self):
        """Test DOC can be imported."""
        from functionalnetworkssft.continual_learning.methods import DOC

        assert DOC is not None

    def test_ica_networks_import(self):
        """Test ICA Networks can be imported."""
        from functionalnetworkssft.continual_learning.methods import ICANetworksCL

        assert ICANetworksCL is not None


class TestTableGeneration:
    """Tests for table generation utilities."""

    def test_generate_accuracy_table(self):
        """Test accuracy table generation."""
        from functionalnetworkssft.continual_learning.utils import (
            generate_accuracy_table,
        )

        # Mock results
        results = {
            "llama-7b_lora_order_1": {"average_accuracy": 65.0},
            "llama-7b_ewc_order_1": {"average_accuracy": 70.0},
            "llama-7b_lwf_order_1": {"average_accuracy": 69.0},
            "llama-7b_o_lora_order_1": {"average_accuracy": 76.0},
            "llama-7b_doc_order_1": {"average_accuracy": 78.0},
            "llama-7b_ica_networks_order_1": {"average_accuracy": 75.0},
        }

        table = generate_accuracy_table(results, model="llama-7b")
        assert len(table) == 6  # 6 methods
        assert "Method" in table.columns

    def test_generate_bwt_fwt_table(self):
        """Test BWT/FWT table generation."""
        from functionalnetworkssft.continual_learning.utils import (
            generate_bwt_fwt_table,
        )

        results = {
            "llama-7b_lora_order_1": {
                "backward_transfer": -10.0,
                "forward_transfer": 1.0,
            },
            "llama-7b_doc_order_1": {
                "backward_transfer": -2.0,
                "forward_transfer": 1.5,
            },
        }

        table = generate_bwt_fwt_table(results, model="llama-7b")
        assert len(table) == 6  # 6 methods
        assert "Method" in table.columns


class TestCSVUtilities:
    """Tests for orchestrator CSV-based resumability functions."""

    def test_init_csv_creates_file(self):
        """Test CSV initialization creates file with header."""
        from functionalnetworkssft.continual_learning.orchestrator import (
            RESULTS_CSV_COLUMNS,
            init_csv,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "results.csv"
            init_csv(csv_path, RESULTS_CSV_COLUMNS)
            assert csv_path.exists()
            with open(csv_path) as f:
                header = f.readline().strip()
            assert "model" in header
            assert "method" in header
            assert "average_accuracy" in header

    def test_init_csv_idempotent(self):
        """Test that init_csv doesn't overwrite existing CSV."""
        from functionalnetworkssft.continual_learning.orchestrator import (
            RESULTS_CSV_COLUMNS,
            append_result,
            init_csv,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "results.csv"
            init_csv(csv_path, RESULTS_CSV_COLUMNS)
            # Append a row
            row = {col: "test" for col in RESULTS_CSV_COLUMNS}
            append_result(csv_path, row, RESULTS_CSV_COLUMNS)
            # Re-init should not overwrite
            init_csv(csv_path, RESULTS_CSV_COLUMNS)
            with open(csv_path) as f:
                lines = f.readlines()
            assert len(lines) == 2  # header + 1 row

    def test_result_exists(self):
        """Test result_exists check."""
        from functionalnetworkssft.continual_learning.orchestrator import (
            RESULTS_CSV_COLUMNS,
            append_result,
            init_csv,
            result_exists,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "results.csv"
            init_csv(csv_path, RESULTS_CSV_COLUMNS)

            assert not result_exists(csv_path, "llama", "lora", "order_1", 42)

            row = {col: "" for col in RESULTS_CSV_COLUMNS}
            row.update({
                "model": "llama",
                "method": "lora",
                "task_order": "order_1",
                "seed": "42",
            })
            append_result(csv_path, row, RESULTS_CSV_COLUMNS)

            assert result_exists(csv_path, "llama", "lora", "order_1", 42)
            assert not result_exists(csv_path, "llama", "lora", "order_2", 42)
            assert not result_exists(csv_path, "llama", "lora", "order_1", 43)

    def test_baseline_exists(self):
        """Test baseline_exists check."""
        from functionalnetworkssft.continual_learning.orchestrator import (
            BASELINE_CSV_COLUMNS,
            append_result,
            baseline_exists,
            init_csv,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "baselines.csv"
            init_csv(csv_path, BASELINE_CSV_COLUMNS)

            assert not baseline_exists(csv_path, "llama", "ag_news", 42)

            row = {
                "model": "llama",
                "task_name": "ag_news",
                "seed": "42",
                "accuracy": "85.0",
                "training_time_seconds": "10.0",
                "timestamp": "2025-01-01",
            }
            append_result(csv_path, row, BASELINE_CSV_COLUMNS)

            assert baseline_exists(csv_path, "llama", "ag_news", 42)
            assert not baseline_exists(csv_path, "llama", "yelp", 42)

    def test_load_baselines_from_csv(self):
        """Test loading baseline accuracies from CSV."""
        from functionalnetworkssft.continual_learning.orchestrator import (
            BASELINE_CSV_COLUMNS,
            append_result,
            init_csv,
            load_baselines_from_csv,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "baselines.csv"
            init_csv(csv_path, BASELINE_CSV_COLUMNS)

            task_order = ["ag_news", "yelp", "amazon"]
            for i, task in enumerate(task_order):
                row = {
                    "model": "llama",
                    "task_name": task,
                    "seed": "42",
                    "accuracy": str(80.0 + i * 5),
                    "training_time_seconds": "10.0",
                    "timestamp": "2025-01-01",
                }
                append_result(csv_path, row, BASELINE_CSV_COLUMNS)

            baselines = load_baselines_from_csv(
                csv_path, "llama", task_order, 42
            )
            assert baselines[0] == 80.0
            assert baselines[1] == 85.0
            assert baselines[2] == 90.0


class TestCheckpointing:
    """Tests for CLCheckpoint save/load cycle."""

    def test_empty_checkpoint(self):
        """Test behavior when no checkpoints exist."""
        from functionalnetworkssft.continual_learning.checkpointing import (
            CLCheckpoint,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = CLCheckpoint(Path(tmpdir) / "nonexistent")
            assert ckpt.get_last_completed_task() == -1
            assert not ckpt.is_run_complete()

    def test_save_and_load_checkpoint(self):
        """Test saving and loading a task checkpoint."""
        from unittest.mock import MagicMock

        from functionalnetworkssft.continual_learning.checkpointing import (
            CLCheckpoint,
        )
        from functionalnetworkssft.continual_learning.metrics import (
            ContinualLearningMetrics,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = CLCheckpoint(Path(tmpdir) / "test_ckpt")

            # Create mock model (not PeftModel)
            model = MagicMock()
            model.state_dict.return_value = {}

            # Create mock CL method with minimal state
            cl_method = MagicMock()
            cl_method.get_state_dict.return_value = {
                "task_history": [{"task_idx": 0, "task_name": "ag_news"}],
                "current_task_idx": 0,
            }

            metrics = ContinualLearningMetrics(num_tasks=5)
            metrics.record_accuracy(0, 0, 85.0)

            ckpt.save_task_checkpoint(0, model, cl_method, metrics)
            assert ckpt.get_last_completed_task() == 0

            loaded = ckpt.load_task_checkpoint(0)
            assert loaded["cl_method_state"]["current_task_idx"] == 0
            assert loaded["metrics"].accuracy_matrix[0, 0] == 85.0

    def test_run_complete_marker(self):
        """Test marking and checking run completion."""
        from functionalnetworkssft.continual_learning.checkpointing import (
            CLCheckpoint,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = CLCheckpoint(Path(tmpdir) / "test_ckpt")
            assert not ckpt.is_run_complete()
            ckpt.mark_run_complete()
            assert ckpt.is_run_complete()

    def test_cleanup(self):
        """Test checkpoint cleanup."""
        from functionalnetworkssft.continual_learning.checkpointing import (
            CLCheckpoint,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_dir = Path(tmpdir) / "test_ckpt"
            ckpt = CLCheckpoint(ckpt_dir)
            ckpt.mark_run_complete()
            assert ckpt_dir.exists()
            ckpt.cleanup()
            assert not ckpt_dir.exists()

    def test_multiple_task_checkpoints(self):
        """Test that get_last_completed_task returns highest completed."""
        from unittest.mock import MagicMock

        from functionalnetworkssft.continual_learning.checkpointing import (
            CLCheckpoint,
        )
        from functionalnetworkssft.continual_learning.metrics import (
            ContinualLearningMetrics,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = CLCheckpoint(Path(tmpdir) / "test_ckpt")
            model = MagicMock()
            model.state_dict.return_value = {}
            cl_method = MagicMock()
            cl_method.get_state_dict.return_value = {
                "task_history": [],
                "current_task_idx": 0,
            }
            metrics = ContinualLearningMetrics(num_tasks=5)

            ckpt.save_task_checkpoint(0, model, cl_method, metrics)
            ckpt.save_task_checkpoint(1, model, cl_method, metrics)
            ckpt.save_task_checkpoint(2, model, cl_method, metrics)

            assert ckpt.get_last_completed_task() == 2


class TestAggregation:
    """Tests for multi-seed result aggregation."""

    def test_aggregate_across_seeds(self):
        """Test aggregation computes correct mean/std."""
        import pandas as pd

        from functionalnetworkssft.continual_learning.aggregation import (
            aggregate_across_seeds,
        )

        data = {
            "model": ["llama"] * 3,
            "method": ["lora"] * 3,
            "task_order": ["order_1"] * 3,
            "seed": [42, 43, 44],
            "average_accuracy": [80.0, 82.0, 81.0],
            "backward_transfer": [-5.0, -4.0, -6.0],
            "forward_transfer": [1.0, 2.0, 1.5],
        }
        df = pd.DataFrame(data)

        agg = aggregate_across_seeds(df)
        assert len(agg) == 1
        assert abs(agg.iloc[0]["aa_mean"] - 81.0) < 0.01
        assert agg.iloc[0]["n_seeds"] == 3

    def test_generate_accuracy_table_with_stats(self):
        """Test accuracy table generation with stats."""
        import pandas as pd

        from functionalnetworkssft.continual_learning.aggregation import (
            aggregate_across_seeds,
            generate_accuracy_table_with_stats,
        )

        rows = []
        for method in ["lora", "ewc", "doc"]:
            for order in ["order_1", "order_2", "order_3"]:
                rows.append({
                    "model": "llama",
                    "method": method,
                    "task_order": order,
                    "seed": 42,
                    "average_accuracy": 70.0 + hash(method + order) % 20,
                    "backward_transfer": -5.0,
                    "forward_transfer": 1.0,
                })
        df = pd.DataFrame(rows)
        agg = aggregate_across_seeds(df)
        table = generate_accuracy_table_with_stats(agg, "llama", "standard")
        assert "Method" in table.columns
        assert "O1" in table.columns
        assert "Avg" in table.columns

    def test_load_experiment_csv(self):
        """Test loading experiment CSV."""
        from functionalnetworkssft.continual_learning.aggregation import (
            load_experiment_csv,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "results.csv"
            from functionalnetworkssft.continual_learning.orchestrator import (
                RESULTS_CSV_COLUMNS,
                append_result,
                init_csv,
            )

            init_csv(csv_path, RESULTS_CSV_COLUMNS)
            row = {col: "0" for col in RESULTS_CSV_COLUMNS}
            row.update({
                "model": "llama",
                "method": "lora",
                "task_order": "order_1",
                "seed": "42",
                "average_accuracy": "75.5",
                "backward_transfer": "-3.2",
                "forward_transfer": "1.1",
            })
            append_result(csv_path, row, RESULTS_CSV_COLUMNS)

            df = load_experiment_csv(csv_path)
            assert len(df) == 1
            assert df.iloc[0]["average_accuracy"] == 75.5


class TestMethodStateDicts:
    """Tests for method get_state_dict/load_state_dict round-trips."""

    def test_base_state_dict(self):
        """Test base class state dict."""
        from unittest.mock import MagicMock

        from functionalnetworkssft.continual_learning.methods.lora_baseline import (
            LoRABaseline,
        )

        model = MagicMock()
        model.parameters.return_value = iter([])
        config = {"learning_rate": 1e-4}

        method = LoRABaseline(model, config)
        method.current_task_idx = 2
        method.task_history = [
            {"task_idx": 0, "task_name": "a"},
            {"task_idx": 1, "task_name": "b"},
        ]

        state = method.get_state_dict()
        assert state["current_task_idx"] == 2
        assert len(state["task_history"]) == 2

        # Create fresh method and restore
        method2 = LoRABaseline(model, config)
        method2.load_state_dict(state)
        assert method2.current_task_idx == 2
        assert len(method2.task_history) == 2

    def test_ica_networks_state_dict(self):
        """Test ICA Networks state dict round-trip."""
        from unittest.mock import MagicMock

        from functionalnetworkssft.continual_learning.methods.ica_networks import (
            ICANetworksCL,
        )

        model = MagicMock()
        model.parameters.return_value = iter([])
        config = {"learning_rate": 1e-4}

        method = ICANetworksCL(model, config, mask_mode="lesion", ica_components=10)
        method.task_protected_components = {0: [0], 1: [0, 1]}
        method.current_task_idx = 2

        state = method.get_state_dict()
        assert state["mask_mode"] == "lesion"
        assert state["ica_components"] == 10

        method2 = ICANetworksCL(model, config)
        method2.load_state_dict(state)
        assert method2.mask_mode == "lesion"
        assert method2.task_protected_components[1] == [0, 1]
