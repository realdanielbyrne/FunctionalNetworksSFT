"""
Performance and efficiency metrics for the evaluation framework.

This module implements metrics for evaluating model performance including:
- Inference speed (tokens per second)
- Memory usage during inference
- Model size and parameter count
- FLOPS (floating point operations per second)
"""

import gc
import logging
import os
import time
from typing import Any, Dict, List, Optional, Union
import torch
import psutil
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from ..core.metrics import BaseMetric, register_metric
from ..core.results import MetricResult
from ..core.config import BenchmarkConfig

logger = logging.getLogger(__name__)


@register_metric("inference_speed")
class InferenceSpeedMetric(BaseMetric):
    """Metric for measuring inference speed in tokens per second."""
    
    def __init__(self, name: str = "inference_speed", **kwargs):
        super().__init__(name, **kwargs)
        self.total_tokens = 0
        self.total_time = 0.0
        self.batch_times = []
        self.batch_token_counts = []
    
    def compute_score(self, predictions: Any, references: Any, **kwargs) -> float:
        """Compute tokens per second for a single inference."""
        inference_time = kwargs.get('inference_time', 0.0)
        token_count = kwargs.get('token_count', 0)
        
        if inference_time > 0 and token_count > 0:
            return token_count / inference_time
        return 0.0
    
    def add_batch(self, predictions: Any, references: Any, **kwargs) -> None:
        """Add batch timing information."""
        inference_time = kwargs.get('inference_time', 0.0)
        token_count = kwargs.get('token_count', 0)
        
        if inference_time > 0 and token_count > 0:
            self.total_tokens += token_count
            self.total_time += inference_time
            self.batch_times.append(inference_time)
            self.batch_token_counts.append(token_count)
            
            # Also add individual score
            score = self.compute_score(predictions, references, **kwargs)
            self.scores.append(score)
    
    def compute_final_score(self) -> MetricResult:
        """Compute final inference speed metrics."""
        if self.total_time > 0 and self.total_tokens > 0:
            overall_speed = self.total_tokens / self.total_time
            
            # Compute per-batch speeds
            batch_speeds = [
                tokens / time_taken for tokens, time_taken 
                in zip(self.batch_token_counts, self.batch_times)
                if time_taken > 0
            ]
            
            import numpy as np
            
            metadata = {
                'total_tokens': self.total_tokens,
                'total_time': self.total_time,
                'num_batches': len(self.batch_times),
                'mean_batch_speed': np.mean(batch_speeds) if batch_speeds else 0.0,
                'std_batch_speed': np.std(batch_speeds) if batch_speeds else 0.0,
                'min_batch_speed': np.min(batch_speeds) if batch_speeds else 0.0,
                'max_batch_speed': np.max(batch_speeds) if batch_speeds else 0.0,
                **self.metadata,
                **self.parameters
            }
            
            return MetricResult(
                name=self.name,
                value=overall_speed,
                std=np.std(batch_speeds) if batch_speeds else 0.0,
                sample_size=len(batch_speeds),
                metadata=metadata
            )
        else:
            return super().compute_final_score()


@register_metric("memory_usage")
class MemoryUsageMetric(BaseMetric):
    """Metric for measuring memory usage during inference."""
    
    def __init__(self, name: str = "memory_usage", **kwargs):
        super().__init__(name, **kwargs)
        self.peak_memory = 0
        self.memory_measurements = []
        self.baseline_memory = self._get_current_memory()
    
    def _get_current_memory(self) -> Dict[str, float]:
        """Get current memory usage."""
        memory_info = {
            'cpu_memory_mb': psutil.virtual_memory().used / 1024 / 1024,
            'cpu_memory_percent': psutil.virtual_memory().percent,
        }
        
        if torch.cuda.is_available():
            memory_info.update({
                'gpu_memory_allocated_mb': torch.cuda.memory_allocated() / 1024 / 1024,
                'gpu_memory_reserved_mb': torch.cuda.memory_reserved() / 1024 / 1024,
                'gpu_memory_percent': (torch.cuda.memory_allocated() / 
                                     torch.cuda.get_device_properties(0).total_memory * 100),
            })
        
        return memory_info
    
    def compute_score(self, predictions: Any, references: Any, **kwargs) -> float:
        """Compute memory usage for a single inference."""
        current_memory = self._get_current_memory()
        self.memory_measurements.append(current_memory)
        
        # Return GPU memory if available, otherwise CPU memory
        if torch.cuda.is_available():
            memory_used = current_memory['gpu_memory_allocated_mb']
            self.peak_memory = max(self.peak_memory, memory_used)
            return memory_used
        else:
            memory_used = current_memory['cpu_memory_mb']
            self.peak_memory = max(self.peak_memory, memory_used)
            return memory_used
    
    def compute_final_score(self) -> MetricResult:
        """Compute final memory usage metrics."""
        if not self.memory_measurements:
            return super().compute_final_score()
        
        import numpy as np
        
        # Compute statistics
        if torch.cuda.is_available():
            gpu_allocated = [m['gpu_memory_allocated_mb'] for m in self.memory_measurements]
            gpu_reserved = [m['gpu_memory_reserved_mb'] for m in self.memory_measurements]
            
            avg_allocated = np.mean(gpu_allocated)
            peak_allocated = np.max(gpu_allocated)
            avg_reserved = np.mean(gpu_reserved)
            peak_reserved = np.max(gpu_reserved)
            
            metadata = {
                'peak_gpu_allocated_mb': peak_allocated,
                'avg_gpu_allocated_mb': avg_allocated,
                'peak_gpu_reserved_mb': peak_reserved,
                'avg_gpu_reserved_mb': avg_reserved,
                'baseline_gpu_allocated_mb': self.baseline_memory.get('gpu_memory_allocated_mb', 0),
                'memory_increase_mb': peak_allocated - self.baseline_memory.get('gpu_memory_allocated_mb', 0),
                **self.metadata,
                **self.parameters
            }
            
            return MetricResult(
                name=self.name,
                value=peak_allocated,
                std=np.std(gpu_allocated),
                sample_size=len(gpu_allocated),
                metadata=metadata
            )
        else:
            cpu_memory = [m['cpu_memory_mb'] for m in self.memory_measurements]
            
            avg_memory = np.mean(cpu_memory)
            peak_memory = np.max(cpu_memory)
            
            metadata = {
                'peak_cpu_memory_mb': peak_memory,
                'avg_cpu_memory_mb': avg_memory,
                'baseline_cpu_memory_mb': self.baseline_memory.get('cpu_memory_mb', 0),
                'memory_increase_mb': peak_memory - self.baseline_memory.get('cpu_memory_mb', 0),
                **self.metadata,
                **self.parameters
            }
            
            return MetricResult(
                name=self.name,
                value=peak_memory,
                std=np.std(cpu_memory),
                sample_size=len(cpu_memory),
                metadata=metadata
            )


@register_metric("model_size")
class ModelSizeMetric(BaseMetric):
    """Metric for measuring model size and parameter count."""
    
    def __init__(self, name: str = "model_size", **kwargs):
        super().__init__(name, **kwargs)
        self.model_info = {}
    
    def compute_score(self, predictions: Any, references: Any, **kwargs) -> float:
        """Compute model size metrics."""
        model = kwargs.get('model')
        if model is None:
            return 0.0
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Estimate model size in MB
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
        model_size_mb = (param_size + buffer_size) / 1024 / 1024
        
        self.model_info = {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'non_trainable_parameters': total_params - trainable_params,
            'model_size_mb': model_size_mb,
            'parameter_size_mb': param_size / 1024 / 1024,
            'buffer_size_mb': buffer_size / 1024 / 1024,
        }
        
        return model_size_mb
    
    def compute_final_score(self) -> MetricResult:
        """Compute final model size metrics."""
        if not self.model_info:
            return super().compute_final_score()
        
        return MetricResult(
            name=self.name,
            value=self.model_info['model_size_mb'],
            sample_size=1,
            metadata={
                **self.model_info,
                **self.metadata,
                **self.parameters
            }
        )


@register_metric("flops")
class FLOPSMetric(BaseMetric):
    """Metric for measuring floating point operations per second."""
    
    def __init__(self, name: str = "flops", **kwargs):
        super().__init__(name, **kwargs)
        self.flop_measurements = []
    
    def compute_score(self, predictions: Any, references: Any, **kwargs) -> float:
        """Compute FLOPS for a single inference."""
        # This is a simplified FLOPS estimation
        # In practice, you would use profiling tools like torch.profiler
        
        model = kwargs.get('model')
        input_length = kwargs.get('input_length', 0)
        inference_time = kwargs.get('inference_time', 0.0)
        
        if model is None or input_length == 0 or inference_time == 0:
            return 0.0
        
        # Rough estimation based on model parameters and sequence length
        total_params = sum(p.numel() for p in model.parameters())
        estimated_flops = total_params * input_length * 2  # Forward pass approximation
        flops_per_second = estimated_flops / inference_time
        
        self.flop_measurements.append({
            'flops': estimated_flops,
            'time': inference_time,
            'flops_per_second': flops_per_second
        })
        
        return flops_per_second
    
    def compute_final_score(self) -> MetricResult:
        """Compute final FLOPS metrics."""
        if not self.flop_measurements:
            return super().compute_final_score()
        
        import numpy as np
        
        flops_per_second = [m['flops_per_second'] for m in self.flop_measurements]
        total_flops = sum(m['flops'] for m in self.flop_measurements)
        total_time = sum(m['time'] for m in self.flop_measurements)
        
        avg_flops_per_second = np.mean(flops_per_second)
        overall_flops_per_second = total_flops / total_time if total_time > 0 else 0.0
        
        metadata = {
            'total_flops': total_flops,
            'total_time': total_time,
            'overall_flops_per_second': overall_flops_per_second,
            'num_measurements': len(flops_per_second),
            'min_flops_per_second': np.min(flops_per_second),
            'max_flops_per_second': np.max(flops_per_second),
            **self.metadata,
            **self.parameters
        }
        
        return MetricResult(
            name=self.name,
            value=avg_flops_per_second,
            std=np.std(flops_per_second),
            sample_size=len(flops_per_second),
            metadata=metadata
        )


class PerformanceEvaluator:
    """Evaluator for performance and efficiency metrics."""
    
    def __init__(self):
        self.supported_metrics = {
            'inference_speed': InferenceSpeedMetric,
            'memory_usage': MemoryUsageMetric,
            'model_size': ModelSizeMetric,
            'flops': FLOPSMetric,
        }
    
    def evaluate(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        config: BenchmarkConfig,
        device: torch.device,
        **kwargs
    ) -> List[MetricResult]:
        """Evaluate performance metrics."""
        logger.info("Running performance evaluation")
        
        results = []
        
        # Initialize metrics
        metrics = {}
        for metric_config in config.metrics:
            if not metric_config.enabled:
                continue
            
            if metric_config.name in self.supported_metrics:
                metric_class = self.supported_metrics[metric_config.name]
                metrics[metric_config.name] = metric_class(
                    name=metric_config.name,
                    **metric_config.parameters
                )
            else:
                logger.warning(f"Unsupported performance metric: {metric_config.name}")
        
        if not metrics:
            logger.warning("No supported performance metrics enabled")
            return results
        
        # Model size metric (computed once)
        if 'model_size' in metrics:
            metrics['model_size'].compute_score(None, None, model=model)
            results.append(metrics['model_size'].compute_final_score())
        
        # Performance benchmarking
        test_inputs = self._generate_test_inputs(tokenizer, config)
        
        model.eval()
        with torch.no_grad():
            for test_input in test_inputs:
                # Prepare input
                inputs = tokenizer(
                    test_input,
                    return_tensors='pt',
                    truncation=True,
                    max_length=config.parameters.get('max_length', 512)
                ).to(device)
                
                input_length = inputs['input_ids'].shape[1]
                
                # Clear cache before measurement
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                
                # Measure inference
                start_time = time.time()
                
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=config.parameters.get('max_new_tokens', 50),
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )
                
                end_time = time.time()
                inference_time = end_time - start_time
                
                # Count generated tokens
                generated_tokens = outputs.shape[1] - inputs['input_ids'].shape[1]
                total_tokens = outputs.shape[1]
                
                # Update metrics
                metric_kwargs = {
                    'model': model,
                    'inference_time': inference_time,
                    'token_count': generated_tokens,
                    'input_length': input_length,
                    'total_tokens': total_tokens,
                }
                
                for metric_name, metric in metrics.items():
                    if metric_name != 'model_size':  # Already computed
                        metric.add_batch(None, None, **metric_kwargs)
        
        # Compute final results for remaining metrics
        for metric_name, metric in metrics.items():
            if metric_name != 'model_size':  # Already added
                result = metric.compute_final_score()
                results.append(result)
        
        return results
    
    def _generate_test_inputs(
        self, 
        tokenizer: PreTrainedTokenizerBase, 
        config: BenchmarkConfig
    ) -> List[str]:
        """Generate test inputs for performance benchmarking."""
        # Generate inputs of varying lengths for comprehensive testing
        test_inputs = [
            "Hello, how are you today?",  # Short
            "The quick brown fox jumps over the lazy dog. " * 5,  # Medium
            "This is a longer text that will be used to test the performance of the model. " * 10,  # Long
        ]
        
        # Limit number of test inputs based on config
        max_samples = config.max_samples or len(test_inputs)
        return test_inputs[:max_samples]
