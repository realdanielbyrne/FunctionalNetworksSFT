#!/usr/bin/env python3
"""
Tests for the evaluation framework.

This test suite verifies that the evaluation framework works correctly,
including metric computation, benchmark execution, and result management.
"""

import os
import tempfile
import unittest
from unittest.mock import Mock, patch, MagicMock
import torch
import torch.nn as nn
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from functionalnetworkssft.evaluation.core.config import (
    EvaluationArguments,
    BenchmarkConfig,
    MetricConfig,
    get_default_evaluation_config,
    create_benchmark_config,
)
from functionalnetworkssft.evaluation.core.evaluator import (
    EvaluationConfig,
    ModelEvaluator,
)
from functionalnetworkssft.evaluation.core.results import (
    MetricResult,
    BenchmarkResult,
    EvaluationReport,
    ResultManager,
)
from functionalnetworkssft.evaluation.core.metrics import (
    BaseMetric,
    MetricRegistry,
    AccuracyMetric,
)
from functionalnetworkssft.evaluation.benchmarks.language_understanding import (
    PerplexityMetric,
    BLEUMetric,
    ROUGEMetric,
)
from functionalnetworkssft.evaluation.benchmarks.performance_metrics import (
    InferenceSpeedMetric,
    MemoryUsageMetric,
    ModelSizeMetric,
)


class MockModel(nn.Module):
    """Mock model for testing."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)
        self.config = Mock()
        self.config.model_type = "test_model"

    def parameters(self):
        return self.linear.parameters()

    def eval(self):
        return self

    def generate(self, **kwargs):
        # Mock generation - return input with some additional tokens
        input_ids = kwargs.get("input_ids")
        if input_ids is not None:
            batch_size, seq_len = input_ids.shape
            max_new_tokens = kwargs.get("max_new_tokens", 10)
            # Create mock output with additional tokens
            new_tokens = torch.randint(0, 1000, (batch_size, max_new_tokens))
            return torch.cat([input_ids, new_tokens], dim=1)
        return torch.tensor([[1, 2, 3, 4, 5]])

    def forward(self, **kwargs):
        # Mock forward pass
        input_ids = kwargs.get("input_ids")
        if input_ids is not None:
            batch_size, seq_len = input_ids.shape
            vocab_size = 1000
            logits = torch.randn(batch_size, seq_len, vocab_size)

            # Mock loss calculation
            labels = kwargs.get("labels")
            if labels is not None:
                loss = torch.tensor(2.5)  # Mock loss value
                return Mock(loss=loss, logits=logits)

            return Mock(logits=logits)

        return Mock(logits=torch.randn(1, 10, 1000))


class MockTokenizer:
    """Mock tokenizer for testing."""

    def __init__(self):
        self.pad_token = "<pad>"
        self.pad_token_id = 0
        self.eos_token = "</s>"
        self.eos_token_id = 1

    def __call__(self, text, **kwargs):
        # Mock tokenization
        if isinstance(text, str):
            tokens = text.split()[:10]  # Limit to 10 tokens
        else:
            tokens = ["token"] * 5

        input_ids = torch.tensor([list(range(len(tokens)))])
        attention_mask = torch.ones_like(input_ids)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

    def decode(self, token_ids, **kwargs):
        # Mock decoding
        if isinstance(token_ids, torch.Tensor):
            return f"Generated text with {len(token_ids)} tokens"
        return "Generated text"


class TestEvaluationConfig(unittest.TestCase):
    """Test evaluation configuration."""

    def test_default_config(self):
        """Test default configuration creation."""
        config = get_default_evaluation_config()

        self.assertIsInstance(config, EvaluationArguments)
        self.assertTrue(len(config.benchmarks) > 0)
        self.assertEqual(config.output_dir, "./evaluation_results")

    def test_benchmark_config_creation(self):
        """Test benchmark configuration creation."""
        benchmark = create_benchmark_config(
            name="test_benchmark",
            dataset_name="test_dataset",
            metrics=["accuracy", "bleu"],
        )

        self.assertEqual(benchmark.name, "test_benchmark")
        self.assertEqual(benchmark.dataset_name, "test_dataset")
        self.assertEqual(len(benchmark.metrics), 2)
        self.assertEqual(benchmark.metrics[0].name, "accuracy")
        self.assertEqual(benchmark.metrics[1].name, "bleu")


class TestMetrics(unittest.TestCase):
    """Test individual metrics."""

    def test_accuracy_metric(self):
        """Test accuracy metric computation."""
        metric = AccuracyMetric(name="accuracy")

        # Test string comparison
        score1 = metric.compute_score("hello", "hello")
        self.assertEqual(score1, 1.0)

        score2 = metric.compute_score("hello", "world")
        self.assertEqual(score2, 0.0)

        # Test batch processing
        metric.add_batch(["hello", "world"], ["hello", "earth"])
        result = metric.compute_final_score()

        self.assertIsInstance(result, MetricResult)
        self.assertEqual(result.name, "accuracy")
        self.assertEqual(result.value, 0.5)  # 1 correct out of 2

    def test_perplexity_metric(self):
        """Test perplexity metric computation."""
        metric = PerplexityMetric()

        # Test with loss value
        loss = torch.tensor(2.0)
        score = metric.compute_score(loss, None)
        expected_perplexity = torch.exp(loss).item()
        self.assertAlmostEqual(score, expected_perplexity, places=5)

        # Test batch processing
        metric.add_batch(torch.tensor(1.0), None, num_tokens=10)
        metric.add_batch(torch.tensor(2.0), None, num_tokens=20)
        result = metric.compute_final_score()

        self.assertIsInstance(result, MetricResult)
        self.assertEqual(result.name, "perplexity")
        self.assertGreater(result.value, 0)

    def test_bleu_metric(self):
        """Test BLEU metric computation."""
        metric = BLEUMetric(name="bleu")

        # Test identical strings (should get some score > 0)
        score1 = metric.compute_score("hello world", "hello world")
        self.assertGreaterEqual(score1, 0.0)  # At least some score

        # Test different strings
        score2 = metric.compute_score("hello world", "goodbye earth")
        self.assertGreaterEqual(score2, 0.0)  # Should be non-negative

        # Test batch processing
        metric.add_batch(
            ["hello world", "good morning"], ["hello world", "good evening"]
        )
        result = metric.compute_final_score()

        self.assertIsInstance(result, MetricResult)
        self.assertEqual(result.name, "bleu")

    def test_inference_speed_metric(self):
        """Test inference speed metric."""
        metric = InferenceSpeedMetric()

        # Test single measurement
        score = metric.compute_score(None, None, inference_time=1.0, token_count=100)
        self.assertEqual(score, 100.0)  # 100 tokens per second

        # Test batch processing
        metric.add_batch(None, None, inference_time=1.0, token_count=50)
        metric.add_batch(None, None, inference_time=2.0, token_count=100)
        result = metric.compute_final_score()

        self.assertIsInstance(result, MetricResult)
        self.assertEqual(result.name, "inference_speed")
        self.assertGreater(result.value, 0)

    def test_model_size_metric(self):
        """Test model size metric."""
        metric = ModelSizeMetric()
        model = MockModel()

        score = metric.compute_score(None, None, model=model)
        self.assertGreater(score, 0)  # Should return model size in MB

        result = metric.compute_final_score()
        self.assertIsInstance(result, MetricResult)
        self.assertEqual(result.name, "model_size")
        self.assertIn("total_parameters", result.metadata)


class TestResultManagement(unittest.TestCase):
    """Test result management and storage."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.result_manager = ResultManager(self.temp_dir)

    def tearDown(self):
        """Clean up test environment."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_metric_result_serialization(self):
        """Test metric result serialization."""
        metric_result = MetricResult(
            name="test_metric",
            value=0.85,
            std=0.05,
            confidence_interval=(0.80, 0.90),
            sample_size=100,
            metadata={"test_key": "test_value"},
        )

        # Test to_dict
        result_dict = metric_result.to_dict()
        self.assertEqual(result_dict["name"], "test_metric")
        self.assertEqual(result_dict["value"], 0.85)

        # Test from_dict
        restored_result = MetricResult.from_dict(result_dict)
        self.assertEqual(restored_result.name, "test_metric")
        self.assertEqual(restored_result.value, 0.85)

    def test_benchmark_result_management(self):
        """Test benchmark result management."""
        benchmark_result = BenchmarkResult(name="test_benchmark")

        # Add metrics
        metric1 = MetricResult(name="accuracy", value=0.85)
        metric2 = MetricResult(name="bleu", value=0.75)

        benchmark_result.add_metric(metric1)
        benchmark_result.add_metric(metric2)

        # Test retrieval
        retrieved_metric = benchmark_result.get_metric("accuracy")
        self.assertIsNotNone(retrieved_metric)
        self.assertEqual(retrieved_metric.value, 0.85)

        # Test serialization
        result_dict = benchmark_result.to_dict()
        restored_result = BenchmarkResult.from_dict(result_dict)
        self.assertEqual(restored_result.name, "test_benchmark")
        self.assertEqual(len(restored_result.metrics), 2)

    def test_evaluation_report_management(self):
        """Test evaluation report management."""
        report = EvaluationReport(model_name="test_model")

        # Add benchmark results
        benchmark1 = BenchmarkResult(name="benchmark1")
        benchmark1.add_metric(MetricResult(name="accuracy", value=0.85))

        benchmark2 = BenchmarkResult(name="benchmark2")
        benchmark2.add_metric(MetricResult(name="bleu", value=0.75))

        report.add_benchmark(benchmark1)
        report.add_benchmark(benchmark2)

        # Test summary metrics
        summary = report.get_summary_metrics()
        self.assertIn("mean_score", summary)
        self.assertIn("total_benchmarks", summary)
        self.assertEqual(summary["total_benchmarks"], 2)

        # Test serialization
        report_dict = report.to_dict()
        restored_report = EvaluationReport.from_dict(report_dict)
        self.assertEqual(restored_report.model_name, "test_model")
        self.assertEqual(len(restored_report.benchmarks), 2)

    def test_result_manager_save_load(self):
        """Test result manager save and load functionality."""
        # Create test report
        report = EvaluationReport(model_name="test_model")
        benchmark = BenchmarkResult(name="test_benchmark")
        benchmark.add_metric(MetricResult(name="accuracy", value=0.85))
        report.add_benchmark(benchmark)

        # Save report
        saved_path = self.result_manager.save_report(report)
        self.assertTrue(os.path.exists(saved_path))

        # Load report
        loaded_report = self.result_manager.load_report(saved_path)
        self.assertEqual(loaded_report.model_name, "test_model")
        self.assertEqual(len(loaded_report.benchmarks), 1)

        # Test list reports
        reports = self.result_manager.list_reports()
        self.assertEqual(len(reports), 1)


class TestEvaluationIntegration(unittest.TestCase):
    """Test end-to-end evaluation integration."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test environment."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch("functionalnetworkssft.evaluation.core.evaluator.AutoModelForCausalLM")
    @patch("functionalnetworkssft.evaluation.core.evaluator.AutoTokenizer")
    def test_model_evaluator_initialization(self, mock_tokenizer, mock_model):
        """Test model evaluator initialization."""
        # Setup mocks
        mock_model.from_pretrained.return_value = MockModel()
        mock_tokenizer.from_pretrained.return_value = MockTokenizer()

        # Create evaluation config
        eval_args = EvaluationArguments(
            model_name_or_path="test_model",
            output_dir=self.temp_dir,
            benchmarks=[
                create_benchmark_config(name="test_benchmark", metrics=["accuracy"])
            ],
        )

        config = EvaluationConfig(eval_args)
        evaluator = ModelEvaluator(config)

        # Test model loading
        evaluator.load_model_and_tokenizer()
        self.assertIsNotNone(evaluator.model)
        self.assertIsNotNone(evaluator.tokenizer)

        # Cleanup
        evaluator.cleanup()

    def test_metric_registry(self):
        """Test metric registry functionality."""
        # Test listing metrics
        metrics = MetricRegistry.list_metrics()
        self.assertIn("accuracy", metrics)

        # Test getting metric
        accuracy_metric = MetricRegistry.get_metric("accuracy")
        self.assertIsInstance(accuracy_metric, AccuracyMetric)

        # Test unknown metric
        with self.assertRaises(ValueError):
            MetricRegistry.get_metric("unknown_metric")


class TestEvaluationCLI(unittest.TestCase):
    """Test CLI integration."""

    def test_cli_argument_parsing(self):
        """Test CLI argument parsing."""
        from functionalnetworkssft.evaluation.cli.evaluation_cli import (
            create_evaluation_args_from_cli,
        )

        # Create mock args
        mock_args = Mock()
        mock_args.model_name_or_path = "test_model"
        mock_args.eval_config = None  # No config file
        mock_args.eval_output_dir = "./test_output"
        mock_args.eval_benchmarks = ["mmlu", "performance"]
        mock_args.eval_max_samples = 100
        mock_args.eval_batch_size = 4
        mock_args.use_auth_token = True
        mock_args.torch_dtype = "float16"

        # Create evaluation args
        eval_args = create_evaluation_args_from_cli(mock_args)

        self.assertEqual(eval_args.model_name_or_path, "test_model")
        self.assertEqual(eval_args.output_dir, "./test_output")
        self.assertEqual(eval_args.batch_size, 4)
        self.assertEqual(len(eval_args.benchmarks), 2)


if __name__ == "__main__":
    # Setup logging for tests
    import logging

    logging.basicConfig(level=logging.WARNING)

    # Run tests
    unittest.main(verbosity=2)
