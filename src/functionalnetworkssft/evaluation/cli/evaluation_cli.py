"""
Command-line interface for the evaluation framework.

This module provides CLI integration for running model evaluations,
following the same patterns as the existing training CLI.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from functionalnetworkssft.evaluation.core.config import (
    EvaluationArguments,
    BenchmarkConfig,
    MetricConfig,
    load_evaluation_config,
    save_evaluation_config,
    get_default_evaluation_config,
    create_benchmark_config,
)
from functionalnetworkssft.evaluation.core.evaluator import (
    EvaluationConfig,
    ModelEvaluator,
)

logger = logging.getLogger(__name__)


def add_evaluation_arguments(parser: argparse.ArgumentParser) -> None:
    """
    Add evaluation-specific arguments to an existing argument parser.
    
    Args:
        parser: ArgumentParser to add evaluation arguments to
    """
    eval_group = parser.add_argument_group('Evaluation Options')
    
    # Configuration
    eval_group.add_argument(
        '--eval_config',
        type=str,
        help='Path to evaluation configuration YAML file'
    )
    eval_group.add_argument(
        '--eval_output_dir',
        type=str,
        default='./evaluation_results',
        help='Directory to save evaluation results'
    )
    eval_group.add_argument(
        '--eval_run_name',
        type=str,
        help='Name for this evaluation run'
    )
    
    # Benchmarks
    eval_group.add_argument(
        '--eval_benchmarks',
        type=str,
        nargs='+',
        choices=[
            'language_understanding', 'perplexity', 'mmlu', 'hellaswag', 
            'arc', 'gsm8k', 'humaneval', 'performance', 'safety', 'bias'
        ],
        help='Benchmarks to run'
    )
    eval_group.add_argument(
        '--eval_max_samples',
        type=int,
        help='Maximum number of samples per benchmark (for faster testing)'
    )
    
    # Performance settings
    eval_group.add_argument(
        '--eval_batch_size',
        type=int,
        default=8,
        help='Batch size for evaluation'
    )
    eval_group.add_argument(
        '--eval_max_length',
        type=int,
        default=2048,
        help='Maximum sequence length for evaluation'
    )
    
    # Reporting
    eval_group.add_argument(
        '--eval_generate_report',
        action='store_true',
        default=True,
        help='Generate comprehensive evaluation report'
    )
    eval_group.add_argument(
        '--eval_include_visualizations',
        action='store_true',
        default=True,
        help='Include charts and visualizations in report'
    )
    eval_group.add_argument(
        '--eval_save_predictions',
        action='store_true',
        help='Save model predictions for analysis'
    )
    
    # Comparison
    eval_group.add_argument(
        '--eval_baseline_results',
        type=str,
        help='Path to baseline results for comparison'
    )
    eval_group.add_argument(
        '--eval_compare_with_published',
        action='store_true',
        default=True,
        help='Compare with published benchmark results'
    )
    
    # Safety and efficiency
    eval_group.add_argument(
        '--eval_enable_safety_checks',
        action='store_true',
        default=True,
        help='Enable safety and bias evaluations'
    )
    eval_group.add_argument(
        '--eval_enable_efficiency_metrics',
        action='store_true',
        default=True,
        help='Enable performance and efficiency metrics'
    )
    
    # Logging
    eval_group.add_argument(
        '--eval_log_level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level for evaluation'
    )
    eval_group.add_argument(
        '--eval_use_wandb',
        action='store_true',
        help='Log evaluation results to Weights & Biases'
    )
    eval_group.add_argument(
        '--eval_wandb_project',
        type=str,
        default='llm-evaluation',
        help='Weights & Biases project name for evaluation'
    )


def create_evaluation_args_from_cli(args: argparse.Namespace) -> EvaluationArguments:
    """
    Create EvaluationArguments from CLI arguments.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        EvaluationArguments instance
    """
    # Load from config file if provided
    if hasattr(args, 'eval_config') and args.eval_config:
        eval_args = load_evaluation_config(args.eval_config)
    else:
        eval_args = get_default_evaluation_config()
    
    # Override with CLI arguments
    if hasattr(args, 'model_name_or_path') and args.model_name_or_path:
        eval_args.model_name_or_path = args.model_name_or_path
    
    if hasattr(args, 'eval_output_dir') and args.eval_output_dir:
        eval_args.output_dir = args.eval_output_dir
    
    if hasattr(args, 'eval_run_name') and args.eval_run_name:
        eval_args.run_name = args.eval_run_name
    
    if hasattr(args, 'eval_batch_size') and args.eval_batch_size:
        eval_args.batch_size = args.eval_batch_size
    
    if hasattr(args, 'eval_max_length') and args.eval_max_length:
        eval_args.max_length = args.eval_max_length
    
    if hasattr(args, 'eval_log_level') and args.eval_log_level:
        eval_args.log_level = args.eval_log_level
    
    if hasattr(args, 'eval_use_wandb') and args.eval_wandb:
        eval_args.use_wandb = args.eval_use_wandb
    
    if hasattr(args, 'eval_wandb_project') and args.eval_wandb_project:
        eval_args.wandb_project = args.eval_wandb_project
    
    # Handle benchmark selection
    if hasattr(args, 'eval_benchmarks') and args.eval_benchmarks:
        # Create benchmark configs for selected benchmarks
        benchmarks = []
        for benchmark_name in args.eval_benchmarks:
            if benchmark_name == 'language_understanding':
                benchmark = create_benchmark_config(
                    name='language_understanding',
                    metrics=['perplexity', 'bleu', 'rouge', 'bertscore']
                )
            elif benchmark_name == 'mmlu':
                benchmark = create_benchmark_config(
                    name='mmlu',
                    dataset_name='cais/mmlu',
                    metrics=['mmlu_accuracy']
                )
            elif benchmark_name == 'hellaswag':
                benchmark = create_benchmark_config(
                    name='hellaswag',
                    dataset_name='Rowan/hellaswag',
                    metrics=['hellaswag_accuracy']
                )
            elif benchmark_name == 'arc':
                benchmark = create_benchmark_config(
                    name='arc',
                    dataset_name='ai2_arc',
                    metrics=['arc_accuracy']
                )
            elif benchmark_name == 'gsm8k':
                benchmark = create_benchmark_config(
                    name='gsm8k',
                    dataset_name='gsm8k',
                    metrics=['gsm8k_accuracy']
                )
            elif benchmark_name == 'humaneval':
                benchmark = create_benchmark_config(
                    name='humaneval',
                    dataset_name='openai_humaneval',
                    metrics=['humaneval_pass_at_k']
                )
            elif benchmark_name == 'performance':
                benchmark = create_benchmark_config(
                    name='performance',
                    metrics=['inference_speed', 'memory_usage', 'model_size', 'flops']
                )
            elif benchmark_name in ['safety', 'bias']:
                benchmark = create_benchmark_config(
                    name=benchmark_name,
                    metrics=['toxicity', 'bias', 'harmful_content']
                )
            else:
                logger.warning(f"Unknown benchmark: {benchmark_name}")
                continue
            
            # Set max samples if specified
            if hasattr(args, 'eval_max_samples') and args.eval_max_samples:
                benchmark.max_samples = args.eval_max_samples
            
            benchmarks.append(benchmark)
        
        eval_args.benchmarks = benchmarks
    
    # Copy other relevant arguments from training args
    if hasattr(args, 'use_auth_token'):
        eval_args.use_auth_token = args.use_auth_token
    
    if hasattr(args, 'trust_remote_code'):
        eval_args.trust_remote_code = args.trust_remote_code
    
    if hasattr(args, 'torch_dtype'):
        eval_args.torch_dtype = args.torch_dtype
    
    return eval_args


def run_evaluation_from_args(args: argparse.Namespace) -> None:
    """
    Run evaluation from command line arguments.
    
    Args:
        args: Parsed command line arguments
    """
    # Create evaluation arguments
    eval_args = create_evaluation_args_from_cli(args)
    
    # Validate model path
    if not eval_args.model_name_or_path:
        raise ValueError("Model path must be specified for evaluation")
    
    # Setup logging
    log_level = getattr(logging, eval_args.log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(eval_args.output_dir, "evaluation.log")),
            logging.StreamHandler(sys.stdout),
        ],
    )
    
    logger.info("Starting model evaluation")
    logger.info(f"Model: {eval_args.model_name_or_path}")
    logger.info(f"Output directory: {eval_args.output_dir}")
    logger.info(f"Benchmarks: {[b.name for b in eval_args.benchmarks]}")
    
    # Initialize evaluator
    config = EvaluationConfig(eval_args)
    evaluator = ModelEvaluator(config)
    
    try:
        # Run evaluation
        report = evaluator.evaluate()
        
        # Log summary
        logger.info("Evaluation completed successfully")
        logger.info(f"Total duration: {report.total_duration:.2f} seconds")
        logger.info(f"Benchmarks completed: {len(report.benchmarks)}")
        
        # Print summary metrics
        summary = report.get_summary_metrics()
        if summary:
            logger.info("Summary metrics:")
            for metric_name, value in summary.items():
                logger.info(f"  {metric_name}: {value:.4f}")
        
        # Log results location
        logger.info(f"Results saved to: {eval_args.output_dir}")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise
    finally:
        # Cleanup
        evaluator.cleanup()


def main():
    """Main entry point for evaluation CLI."""
    parser = argparse.ArgumentParser(
        description="Evaluate fine-tuned language models using comprehensive benchmarks"
    )
    
    # Model arguments
    parser.add_argument(
        '--model_name_or_path',
        type=str,
        required=True,
        help='Path to model or HuggingFace model identifier'
    )
    parser.add_argument(
        '--use_auth_token',
        action='store_true',
        help='Use HuggingFace auth token'
    )
    parser.add_argument(
        '--trust_remote_code',
        action='store_true',
        help='Trust remote code when loading model'
    )
    parser.add_argument(
        '--torch_dtype',
        type=str,
        default='auto',
        choices=['auto', 'float16', 'bfloat16', 'float32'],
        help='Torch dtype for model loading'
    )
    
    # Add evaluation arguments
    add_evaluation_arguments(parser)
    
    # Parse arguments
    args = parser.parse_args()
    
    # Run evaluation
    try:
        run_evaluation_from_args(args)
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
