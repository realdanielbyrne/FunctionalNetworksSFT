"""
Core evaluator classes for the evaluation framework.

This module provides the main evaluation orchestration classes that coordinate
model loading, benchmark execution, and result collection.
"""

import logging
import os
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from .config import EvaluationArguments, BenchmarkConfig
from .results import EvaluationReport, BenchmarkResult, ResultManager
from .metrics import MetricRegistry, BaseMetric
from ...utils.model_utils import get_optimal_device, get_recommended_dtype

logger = logging.getLogger(__name__)


class EvaluationConfig:
    """Configuration container for evaluation runs."""
    
    def __init__(self, args: EvaluationArguments):
        """
        Initialize evaluation configuration.
        
        Args:
            args: EvaluationArguments instance
        """
        self.args = args
        self.device, self.device_name = get_optimal_device()
        self.torch_dtype = self._get_torch_dtype()
        
        logger.info(f"Evaluation device: {self.device_name}")
        logger.info(f"Evaluation dtype: {self.torch_dtype}")
    
    def _get_torch_dtype(self) -> torch.dtype:
        """Get appropriate torch dtype for evaluation."""
        if self.args.torch_dtype == "auto":
            return get_recommended_dtype()
        elif self.args.torch_dtype == "float16":
            return torch.float16
        elif self.args.torch_dtype == "bfloat16":
            return torch.bfloat16
        elif self.args.torch_dtype == "float32":
            return torch.float32
        else:
            logger.warning(f"Unknown dtype {self.args.torch_dtype}, using auto")
            return get_recommended_dtype()


class EvaluationResult:
    """Container for evaluation results."""
    
    def __init__(self):
        """Initialize empty evaluation result."""
        self.benchmarks: Dict[str, BenchmarkResult] = {}
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.errors: List[str] = []
    
    def add_benchmark_result(self, result: BenchmarkResult) -> None:
        """Add a benchmark result."""
        self.benchmarks[result.name] = result
    
    def get_total_duration(self) -> float:
        """Get total evaluation duration."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return 0.0
    
    def to_report(self, model_name: str, config: Dict[str, Any]) -> EvaluationReport:
        """Convert to EvaluationReport."""
        return EvaluationReport(
            model_name=model_name,
            benchmarks=self.benchmarks,
            total_duration=self.get_total_duration(),
            config=config
        )


class BaseEvaluator(ABC):
    """Base class for all evaluators."""
    
    def __init__(self, config: EvaluationConfig):
        """
        Initialize evaluator.
        
        Args:
            config: EvaluationConfig instance
        """
        self.config = config
        self.result_manager = ResultManager(config.args.output_dir)
    
    @abstractmethod
    def evaluate(self) -> EvaluationReport:
        """
        Run evaluation and return results.
        
        Returns:
            EvaluationReport with results
        """
        pass
    
    def _setup_logging(self) -> None:
        """Setup logging for evaluation."""
        log_level = getattr(logging, self.config.args.log_level.upper(), logging.INFO)
        logging.getLogger().setLevel(log_level)
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information for the report."""
        import platform
        import psutil
        
        info = {
            'platform': platform.platform(),
            'python_version': platform.python_version(),
            'cpu_count': psutil.cpu_count(),
            'memory_total': psutil.virtual_memory().total,
            'device': self.config.device_name,
            'torch_version': torch.__version__,
        }
        
        if torch.cuda.is_available():
            info['cuda_version'] = torch.version.cuda
            info['gpu_count'] = torch.cuda.device_count()
            if torch.cuda.device_count() > 0:
                info['gpu_name'] = torch.cuda.get_device_name(0)
                info['gpu_memory'] = torch.cuda.get_device_properties(0).total_memory
        
        return info


class ModelEvaluator(BaseEvaluator):
    """Main evaluator for language models."""
    
    def __init__(self, config: EvaluationConfig):
        """
        Initialize model evaluator.
        
        Args:
            config: EvaluationConfig instance
        """
        super().__init__(config)
        self.model: Optional[PreTrainedModel] = None
        self.tokenizer: Optional[PreTrainedTokenizerBase] = None
        self._setup_logging()
    
    def load_model_and_tokenizer(self) -> None:
        """Load model and tokenizer for evaluation."""
        logger.info(f"Loading model: {self.config.args.model_name_or_path}")
        
        # Get authentication token
        auth_token = None
        if self.config.args.use_auth_token:
            auth_token = os.getenv("HF_TOKEN")
            if auth_token:
                logger.info("Using HF_TOKEN from environment")
            else:
                try:
                    from huggingface_hub import whoami
                    user_info = whoami()
                    logger.info(f"Using cached HuggingFace credentials for user: {user_info['name']}")
                    auth_token = True
                except Exception as e:
                    logger.warning(f"No authentication available: {e}")
                    auth_token = None
        
        # Load tokenizer
        tokenizer_path = (
            self.config.args.tokenizer_name_or_path 
            or self.config.args.model_name_or_path
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=self.config.args.trust_remote_code,
            token=auth_token,
        )
        
        # Set pad token if not exists
        if getattr(self.tokenizer, "pad_token", None) is None:
            eos_token = getattr(self.tokenizer, "eos_token", None)
            if eos_token is not None:
                self.tokenizer.pad_token = eos_token
                self.tokenizer.pad_token_id = getattr(self.tokenizer, "eos_token_id", None)
        
        # Load model
        device_map = self.config.args.device_map
        if device_map == "auto" and self.config.device.type not in ["cuda", "mps"]:
            device_map = None
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.args.model_name_or_path,
            torch_dtype=self.config.torch_dtype,
            trust_remote_code=self.config.args.trust_remote_code,
            token=auth_token,
            device_map=device_map,
        )
        
        # Move to device if device_map wasn't used
        if device_map is None:
            self.model = self.model.to(self.config.device)
        
        # Set to evaluation mode
        self.model.eval()
        
        logger.info("Model and tokenizer loaded successfully")
    
    def evaluate(self) -> EvaluationReport:
        """
        Run complete evaluation pipeline.
        
        Returns:
            EvaluationReport with all results
        """
        logger.info("Starting model evaluation")
        
        # Load model if not already loaded
        if self.model is None or self.tokenizer is None:
            self.load_model_and_tokenizer()
        
        # Initialize result container
        result = EvaluationResult()
        result.start_time = time.time()
        
        # Run each enabled benchmark
        for benchmark_config in self.config.args.benchmarks:
            if not benchmark_config.enabled:
                logger.info(f"Skipping disabled benchmark: {benchmark_config.name}")
                continue
            
            logger.info(f"Running benchmark: {benchmark_config.name}")
            
            try:
                benchmark_result = self._run_benchmark(benchmark_config)
                result.add_benchmark_result(benchmark_result)
                logger.info(f"Completed benchmark: {benchmark_config.name}")
            except Exception as e:
                logger.error(f"Error running benchmark {benchmark_config.name}: {e}")
                result.errors.append(f"Benchmark {benchmark_config.name}: {str(e)}")
        
        result.end_time = time.time()
        
        # Create final report
        report = result.to_report(
            model_name=self.config.args.model_name_or_path,
            config=self.config.args.__dict__
        )
        
        # Add system information
        report.system_info = self._get_system_info()
        
        # Save report
        if self.config.args.generate_report:
            self.result_manager.save_report(report)
        
        logger.info(f"Evaluation completed in {report.total_duration:.2f} seconds")
        return report
    
    def _run_benchmark(self, benchmark_config: BenchmarkConfig) -> BenchmarkResult:
        """
        Run a single benchmark.
        
        Args:
            benchmark_config: Configuration for the benchmark
            
        Returns:
            BenchmarkResult with metrics
        """
        start_time = time.time()
        
        # Create benchmark result container
        benchmark_result = BenchmarkResult(name=benchmark_config.name)
        
        # Import and get the appropriate benchmark evaluator
        benchmark_evaluator = self._get_benchmark_evaluator(benchmark_config)
        
        # Run the benchmark
        metrics_results = benchmark_evaluator.evaluate(
            model=self.model,
            tokenizer=self.tokenizer,
            config=benchmark_config,
            device=self.config.device
        )
        
        # Add metric results
        for metric_result in metrics_results:
            benchmark_result.add_metric(metric_result)
        
        benchmark_result.duration = time.time() - start_time
        
        return benchmark_result
    
    def _get_benchmark_evaluator(self, benchmark_config: BenchmarkConfig):
        """
        Get the appropriate benchmark evaluator for the given configuration.
        
        Args:
            benchmark_config: Benchmark configuration
            
        Returns:
            Benchmark evaluator instance
        """
        # Import benchmark evaluators
        from ..benchmarks.language_understanding import LanguageUnderstandingEvaluator
        from ..benchmarks.standard_benchmarks import StandardBenchmarkEvaluator
        from ..benchmarks.performance_metrics import PerformanceEvaluator
        from ..benchmarks.safety_bias import SafetyBiasEvaluator
        
        # Map benchmark names to evaluators
        evaluator_map = {
            'language_understanding': LanguageUnderstandingEvaluator,
            'perplexity': LanguageUnderstandingEvaluator,
            'mmlu': StandardBenchmarkEvaluator,
            'hellaswag': StandardBenchmarkEvaluator,
            'arc': StandardBenchmarkEvaluator,
            'gsm8k': StandardBenchmarkEvaluator,
            'humaneval': StandardBenchmarkEvaluator,
            'performance': PerformanceEvaluator,
            'safety': SafetyBiasEvaluator,
            'bias': SafetyBiasEvaluator,
        }
        
        evaluator_class = evaluator_map.get(benchmark_config.name)
        if evaluator_class is None:
            raise ValueError(f"Unknown benchmark: {benchmark_config.name}")
        
        return evaluator_class()
    
    def cleanup(self) -> None:
        """Clean up resources."""
        if self.model is not None:
            del self.model
            self.model = None
        
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("Evaluation resources cleaned up")
