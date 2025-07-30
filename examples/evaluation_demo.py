#!/usr/bin/env python3
"""
Evaluation Framework Demo Script

This script demonstrates how to use the FunctionalNetworksSFT evaluation framework
to evaluate fine-tuned language models using various benchmarks and metrics.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from functionalnetworkssft.evaluation.core.config import (
    EvaluationArguments,
    BenchmarkConfig,
    MetricConfig,
    get_default_evaluation_config,
    create_benchmark_config,
    load_evaluation_config,
)
from functionalnetworkssft.evaluation.core.evaluator import (
    EvaluationConfig,
    ModelEvaluator,
)
from functionalnetworkssft.evaluation.cli.evaluation_cli import (
    create_evaluation_args_from_cli,
    run_evaluation_from_args,
)

logger = logging.getLogger(__name__)


def demo_basic_evaluation():
    """Demonstrate basic evaluation setup and execution."""
    print("=" * 60)
    print("BASIC EVALUATION DEMO")
    print("=" * 60)
    
    # Create a simple evaluation configuration
    eval_args = EvaluationArguments(
        model_name_or_path="microsoft/DialoGPT-small",  # Small model for demo
        output_dir="./demo_evaluation_results",
        run_name="basic_demo",
        batch_size=4,
        max_length=256,
        benchmarks=[
            create_benchmark_config(
                name="language_understanding",
                metrics=["perplexity", "bleu"],
                max_samples=10  # Very small for demo
            ),
            create_benchmark_config(
                name="performance",
                metrics=["inference_speed", "memory_usage", "model_size"],
                max_samples=5
            )
        ]
    )
    
    print(f"Model: {eval_args.model_name_or_path}")
    print(f"Output directory: {eval_args.output_dir}")
    print(f"Benchmarks: {[b.name for b in eval_args.benchmarks]}")
    
    # Initialize and run evaluation
    try:
        config = EvaluationConfig(eval_args)
        evaluator = ModelEvaluator(config)
        
        print("\nRunning evaluation...")
        report = evaluator.evaluate()
        
        print(f"\nEvaluation completed in {report.total_duration:.2f} seconds")
        print(f"Benchmarks completed: {len(report.benchmarks)}")
        
        # Print results summary
        for benchmark_name, benchmark in report.benchmarks.items():
            print(f"\n{benchmark_name.upper()} Results:")
            for metric_name, metric in benchmark.metrics.items():
                print(f"  {metric_name}: {metric.value:.4f}")
                if metric.confidence_interval:
                    ci_low, ci_high = metric.confidence_interval
                    print(f"    95% CI: [{ci_low:.4f}, {ci_high:.4f}]")
        
        evaluator.cleanup()
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        return False
    
    return True


def demo_config_file_evaluation():
    """Demonstrate evaluation using configuration files."""
    print("\n" + "=" * 60)
    print("CONFIG FILE EVALUATION DEMO")
    print("=" * 60)
    
    # Use the quick evaluation config
    config_path = project_root / "examples" / "evaluation_configs" / "quick_evaluation.yaml"
    
    if not config_path.exists():
        print(f"Config file not found: {config_path}")
        return False
    
    try:
        # Load configuration
        eval_args = load_evaluation_config(str(config_path))
        
        # Override model for demo (use smaller model)
        eval_args.model_name_or_path = "microsoft/DialoGPT-small"
        eval_args.output_dir = "./demo_config_results"
        
        # Further reduce samples for demo
        for benchmark in eval_args.benchmarks:
            benchmark.max_samples = min(benchmark.max_samples or 10, 10)
        
        print(f"Loaded config from: {config_path}")
        print(f"Model: {eval_args.model_name_or_path}")
        print(f"Benchmarks: {[b.name for b in eval_args.benchmarks]}")
        
        # Run evaluation
        config = EvaluationConfig(eval_args)
        evaluator = ModelEvaluator(config)
        
        print("\nRunning evaluation from config...")
        report = evaluator.evaluate()
        
        print(f"\nEvaluation completed in {report.total_duration:.2f} seconds")
        
        # Print summary
        summary = report.get_summary_metrics()
        if summary:
            print("\nSummary Metrics:")
            for metric_name, value in summary.items():
                print(f"  {metric_name}: {value:.4f}")
        
        evaluator.cleanup()
        
    except Exception as e:
        print(f"Error during config evaluation: {e}")
        return False
    
    return True


def demo_cli_integration():
    """Demonstrate CLI integration."""
    print("\n" + "=" * 60)
    print("CLI INTEGRATION DEMO")
    print("=" * 60)
    
    # Simulate CLI arguments
    class MockArgs:
        def __init__(self):
            self.model_name_or_path = "microsoft/DialoGPT-small"
            self.eval_output_dir = "./demo_cli_results"
            self.eval_run_name = "cli_demo"
            self.eval_benchmarks = ["performance"]
            self.eval_max_samples = 5
            self.eval_batch_size = 2
            self.eval_log_level = "INFO"
            self.use_auth_token = False
            self.trust_remote_code = True
            self.torch_dtype = "auto"
    
    mock_args = MockArgs()
    
    try:
        # Create evaluation args from CLI
        eval_args = create_evaluation_args_from_cli(mock_args)
        
        print(f"CLI Model: {eval_args.model_name_or_path}")
        print(f"CLI Output: {eval_args.output_dir}")
        print(f"CLI Benchmarks: {[b.name for b in eval_args.benchmarks]}")
        
        # This would normally call run_evaluation_from_args(mock_args)
        # but we'll do a simplified version for demo
        print("\nCLI evaluation would run here...")
        print("(Skipped for demo to avoid long execution)")
        
    except Exception as e:
        print(f"Error in CLI demo: {e}")
        return False
    
    return True


def demo_custom_metrics():
    """Demonstrate creating custom metrics."""
    print("\n" + "=" * 60)
    print("CUSTOM METRICS DEMO")
    print("=" * 60)
    
    from functionalnetworkssft.evaluation.core.metrics import BaseMetric, MetricRegistry
    
    # Define a custom metric
    class CustomMetric(BaseMetric):
        """Example custom metric that counts words."""
        
        def compute_score(self, predictions, references, **kwargs):
            if isinstance(predictions, str):
                return len(predictions.split())
            return 0.0
    
    # Register the custom metric
    MetricRegistry.register("word_count", CustomMetric)
    
    print("Registered custom metric: word_count")
    
    # Test the custom metric
    try:
        metric = MetricRegistry.get_metric("word_count")
        
        # Test with sample data
        metric.add_batch("Hello world", "Reference text")
        metric.add_batch("This is a longer sentence", "Another reference")
        
        result = metric.compute_final_score()
        print(f"Custom metric result: {result.value:.2f}")
        print(f"Sample size: {result.sample_size}")
        
    except Exception as e:
        print(f"Error in custom metrics demo: {e}")
        return False
    
    return True


def main():
    """Main demo function."""
    parser = argparse.ArgumentParser(description="Evaluation Framework Demo")
    parser.add_argument(
        "--demo",
        choices=["basic", "config", "cli", "custom", "all"],
        default="all",
        help="Which demo to run"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    
    print("FunctionalNetworksSFT Evaluation Framework Demo")
    print("=" * 60)
    
    demos = {
        "basic": demo_basic_evaluation,
        "config": demo_config_file_evaluation,
        "cli": demo_cli_integration,
        "custom": demo_custom_metrics,
    }
    
    success_count = 0
    total_count = 0
    
    if args.demo == "all":
        demo_list = list(demos.keys())
    else:
        demo_list = [args.demo]
    
    for demo_name in demo_list:
        total_count += 1
        try:
            if demos[demo_name]():
                success_count += 1
                print(f"\n✅ {demo_name.upper()} demo completed successfully")
            else:
                print(f"\n❌ {demo_name.upper()} demo failed")
        except Exception as e:
            print(f"\n❌ {demo_name.upper()} demo crashed: {e}")
    
    print("\n" + "=" * 60)
    print(f"DEMO SUMMARY: {success_count}/{total_count} demos successful")
    print("=" * 60)
    
    if success_count == total_count:
        print("🎉 All demos completed successfully!")
        print("\nNext steps:")
        print("1. Try running: fnsft-eval --model_name_or_path your_model --eval_benchmarks mmlu performance")
        print("2. Create your own evaluation config based on the examples")
        print("3. Integrate evaluation into your training pipeline")
    else:
        print("⚠️  Some demos failed. Check the error messages above.")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
