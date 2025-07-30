"""
Language understanding metrics for the evaluation framework.

This module implements core language understanding metrics including:
- Perplexity for language modeling evaluation
- BLEU scores for text generation quality
- ROUGE scores for summarization tasks
- BERTScore for semantic similarity
"""

import logging
import math
import time
from typing import Any, Dict, List, Optional, Union
import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from ..core.metrics import BaseMetric, register_metric
from ..core.results import MetricResult
from ..core.config import BenchmarkConfig

logger = logging.getLogger(__name__)


@register_metric("perplexity")
class PerplexityMetric(BaseMetric):
    """Perplexity metric for language modeling evaluation."""
    
    def __init__(self, name: str = "perplexity", **kwargs):
        """
        Initialize perplexity metric.
        
        Args:
            name: Metric name
            **kwargs: Additional parameters
        """
        super().__init__(name, **kwargs)
        self.total_loss = 0.0
        self.total_tokens = 0
    
    def compute_score(self, predictions: Any, references: Any, **kwargs) -> float:
        """
        Compute perplexity for a single example.
        
        Args:
            predictions: Model logits or loss
            references: Target token IDs
            **kwargs: Additional arguments
            
        Returns:
            Perplexity score
        """
        if isinstance(predictions, torch.Tensor) and predictions.dim() == 0:
            # Single loss value
            loss = predictions.item()
            return math.exp(loss)
        elif isinstance(predictions, (int, float)):
            # Numeric loss
            return math.exp(predictions)
        else:
            # Compute loss from logits and targets
            if not isinstance(predictions, torch.Tensor):
                predictions = torch.tensor(predictions)
            if not isinstance(references, torch.Tensor):
                references = torch.tensor(references)
            
            # Reshape for cross entropy
            if predictions.dim() > 2:
                predictions = predictions.view(-1, predictions.size(-1))
            if references.dim() > 1:
                references = references.view(-1)
            
            # Compute cross entropy loss
            loss = F.cross_entropy(predictions, references, ignore_index=-100, reduction='mean')
            return math.exp(loss.item())
    
    def add_batch(self, predictions: Any, references: Any, **kwargs) -> None:
        """Add batch for perplexity computation."""
        if isinstance(predictions, torch.Tensor) and predictions.dim() == 0:
            # Single loss value
            loss = predictions.item()
            num_tokens = kwargs.get('num_tokens', 1)
            self.total_loss += loss * num_tokens
            self.total_tokens += num_tokens
        else:
            super().add_batch(predictions, references, **kwargs)
    
    def compute_final_score(self) -> MetricResult:
        """Compute final perplexity score."""
        if self.total_tokens > 0:
            # Use accumulated loss and token counts
            avg_loss = self.total_loss / self.total_tokens
            perplexity = math.exp(avg_loss)
            
            return MetricResult(
                name=self.name,
                value=perplexity,
                sample_size=self.total_tokens,
                metadata={
                    'average_loss': avg_loss,
                    'total_tokens': self.total_tokens,
                    **self.metadata,
                    **self.parameters
                }
            )
        else:
            return super().compute_final_score()


@register_metric("bleu")
class BLEUMetric(BaseMetric):
    """BLEU score metric for text generation evaluation."""
    
    def __init__(self, name: str = "bleu", **kwargs):
        """
        Initialize BLEU metric.
        
        Args:
            name: Metric name
            **kwargs: Additional parameters (max_order, smooth, etc.)
        """
        super().__init__(name, **kwargs)
        self.max_order = kwargs.get('max_order', 4)
        self.smooth = kwargs.get('smooth', False)
    
    def compute_score(self, predictions: Any, references: Any, **kwargs) -> float:
        """
        Compute BLEU score for a single example.
        
        Args:
            predictions: Generated text
            references: Reference text(s)
            **kwargs: Additional arguments
            
        Returns:
            BLEU score
        """
        try:
            import sacrebleu
            
            # Ensure inputs are strings
            if not isinstance(predictions, str):
                predictions = str(predictions)
            
            if isinstance(references, str):
                references = [references]
            elif not isinstance(references, list):
                references = [str(references)]
            
            # Compute BLEU score
            bleu = sacrebleu.sentence_bleu(predictions, references)
            return bleu.score / 100.0  # Convert to 0-1 range
            
        except ImportError:
            logger.warning("sacrebleu not available, using simple BLEU approximation")
            return self._simple_bleu(predictions, references)
        except Exception as e:
            logger.warning(f"Error computing BLEU: {e}")
            return 0.0
    
    def _simple_bleu(self, prediction: str, references: List[str]) -> float:
        """Simple BLEU approximation when sacrebleu is not available."""
        if not prediction or not references:
            return 0.0
        
        pred_tokens = prediction.lower().split()
        ref_tokens = references[0].lower().split() if references else []
        
        if not pred_tokens or not ref_tokens:
            return 0.0
        
        # Simple unigram precision
        matches = sum(1 for token in pred_tokens if token in ref_tokens)
        precision = matches / len(pred_tokens) if pred_tokens else 0.0
        
        # Simple brevity penalty
        bp = min(1.0, len(pred_tokens) / len(ref_tokens)) if ref_tokens else 0.0
        
        return precision * bp


@register_metric("rouge")
class ROUGEMetric(BaseMetric):
    """ROUGE score metric for summarization evaluation."""
    
    def __init__(self, name: str = "rouge", **kwargs):
        """
        Initialize ROUGE metric.
        
        Args:
            name: Metric name
            **kwargs: Additional parameters (rouge_types, use_stemmer, etc.)
        """
        super().__init__(name, **kwargs)
        self.rouge_types = kwargs.get('rouge_types', ['rouge1', 'rouge2', 'rougeL'])
        self.use_stemmer = kwargs.get('use_stemmer', True)
    
    def compute_score(self, predictions: Any, references: Any, **kwargs) -> float:
        """
        Compute ROUGE score for a single example.
        
        Args:
            predictions: Generated text
            references: Reference text(s)
            **kwargs: Additional arguments
            
        Returns:
            Average ROUGE score
        """
        try:
            from rouge_score import rouge_scorer
            
            # Ensure inputs are strings
            if not isinstance(predictions, str):
                predictions = str(predictions)
            
            if isinstance(references, str):
                references = [references]
            elif not isinstance(references, list):
                references = [str(references)]
            
            # Initialize scorer
            scorer = rouge_scorer.RougeScorer(
                self.rouge_types, 
                use_stemmer=self.use_stemmer
            )
            
            # Compute ROUGE scores for each reference
            scores = []
            for reference in references:
                rouge_scores = scorer.score(reference, predictions)
                # Average F1 scores across ROUGE types
                f1_scores = [score.fmeasure for score in rouge_scores.values()]
                scores.append(sum(f1_scores) / len(f1_scores))
            
            # Return maximum score across references
            return max(scores) if scores else 0.0
            
        except ImportError:
            logger.warning("rouge-score not available, using simple overlap metric")
            return self._simple_rouge(predictions, references)
        except Exception as e:
            logger.warning(f"Error computing ROUGE: {e}")
            return 0.0
    
    def _simple_rouge(self, prediction: str, references: List[str]) -> float:
        """Simple ROUGE approximation when rouge-score is not available."""
        if not prediction or not references:
            return 0.0
        
        pred_tokens = set(prediction.lower().split())
        ref_tokens = set(references[0].lower().split()) if references else set()
        
        if not pred_tokens or not ref_tokens:
            return 0.0
        
        # Simple overlap-based score
        overlap = len(pred_tokens & ref_tokens)
        precision = overlap / len(pred_tokens) if pred_tokens else 0.0
        recall = overlap / len(ref_tokens) if ref_tokens else 0.0
        
        if precision + recall == 0:
            return 0.0
        
        f1 = 2 * precision * recall / (precision + recall)
        return f1


@register_metric("bertscore")
class BERTScoreMetric(BaseMetric):
    """BERTScore metric for semantic similarity evaluation."""
    
    def __init__(self, name: str = "bertscore", **kwargs):
        """
        Initialize BERTScore metric.
        
        Args:
            name: Metric name
            **kwargs: Additional parameters (model_type, num_layers, etc.)
        """
        super().__init__(name, **kwargs)
        self.model_type = kwargs.get('model_type', 'microsoft/deberta-xlarge-mnli')
        self.num_layers = kwargs.get('num_layers', None)
        self.verbose = kwargs.get('verbose', False)
    
    def compute_score(self, predictions: Any, references: Any, **kwargs) -> float:
        """
        Compute BERTScore for a single example.
        
        Args:
            predictions: Generated text
            references: Reference text(s)
            **kwargs: Additional arguments
            
        Returns:
            BERTScore F1
        """
        try:
            from bert_score import score
            
            # Ensure inputs are strings
            if not isinstance(predictions, str):
                predictions = str(predictions)
            
            if isinstance(references, str):
                references = [references]
            elif not isinstance(references, list):
                references = [str(references)]
            
            # Compute BERTScore
            P, R, F1 = score(
                [predictions] * len(references),
                references,
                model_type=self.model_type,
                num_layers=self.num_layers,
                verbose=self.verbose
            )
            
            # Return maximum F1 score across references
            return F1.max().item()
            
        except ImportError:
            logger.warning("bert-score not available, using simple similarity metric")
            return self._simple_similarity(predictions, references)
        except Exception as e:
            logger.warning(f"Error computing BERTScore: {e}")
            return 0.0
    
    def _simple_similarity(self, prediction: str, references: List[str]) -> float:
        """Simple similarity approximation when bert-score is not available."""
        if not prediction or not references:
            return 0.0
        
        pred_tokens = set(prediction.lower().split())
        ref_tokens = set(references[0].lower().split()) if references else set()
        
        if not pred_tokens or not ref_tokens:
            return 0.0
        
        # Jaccard similarity
        intersection = len(pred_tokens & ref_tokens)
        union = len(pred_tokens | ref_tokens)
        
        return intersection / union if union > 0 else 0.0


class LanguageUnderstandingEvaluator:
    """Evaluator for language understanding metrics."""
    
    def __init__(self):
        """Initialize language understanding evaluator."""
        self.supported_metrics = {
            'perplexity': PerplexityMetric,
            'bleu': BLEUMetric,
            'rouge': ROUGEMetric,
            'bertscore': BERTScoreMetric,
        }
    
    def evaluate(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        config: BenchmarkConfig,
        device: torch.device,
        **kwargs
    ) -> List[MetricResult]:
        """
        Evaluate language understanding metrics.
        
        Args:
            model: Model to evaluate
            tokenizer: Tokenizer for the model
            config: Benchmark configuration
            device: Device to run evaluation on
            **kwargs: Additional arguments
            
        Returns:
            List of metric results
        """
        logger.info("Running language understanding evaluation")
        
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
                logger.warning(f"Unsupported metric: {metric_config.name}")
        
        if not metrics:
            logger.warning("No supported metrics enabled")
            return results
        
        # Load evaluation dataset
        eval_data = self._load_evaluation_data(config)
        
        # Run evaluation
        model.eval()
        with torch.no_grad():
            for batch in eval_data:
                # Process batch based on metric requirements
                for metric_name, metric in metrics.items():
                    if metric_name == 'perplexity':
                        self._evaluate_perplexity_batch(
                            metric, model, tokenizer, batch, device
                        )
                    else:
                        self._evaluate_generation_batch(
                            metric, model, tokenizer, batch, device
                        )
        
        # Compute final results
        for metric in metrics.values():
            result = metric.compute_final_score()
            results.append(result)
        
        return results
    
    def _load_evaluation_data(self, config: BenchmarkConfig) -> List[Dict[str, Any]]:
        """Load evaluation dataset."""
        # For now, return a simple test dataset
        # In a full implementation, this would load from HuggingFace datasets
        # or other data sources based on config.dataset_name
        
        test_data = [
            {
                'input': 'The quick brown fox jumps over the lazy dog.',
                'target': 'A fast brown fox leaps over a sleepy dog.',
            },
            {
                'input': 'Machine learning is a subset of artificial intelligence.',
                'target': 'ML is part of AI technology.',
            },
            # Add more test examples as needed
        ]
        
        max_samples = config.max_samples
        if max_samples and max_samples < len(test_data):
            test_data = test_data[:max_samples]
        
        return test_data
    
    def _evaluate_perplexity_batch(
        self,
        metric: PerplexityMetric,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        batch: Dict[str, Any],
        device: torch.device
    ) -> None:
        """Evaluate perplexity for a batch."""
        text = batch['input']
        
        # Tokenize
        inputs = tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=512
        ).to(device)
        
        # Forward pass
        outputs = model(**inputs, labels=inputs['input_ids'])
        loss = outputs.loss
        
        # Add to metric
        num_tokens = inputs['input_ids'].numel()
        metric.add_batch(loss, None, num_tokens=num_tokens)
    
    def _evaluate_generation_batch(
        self,
        metric: BaseMetric,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        batch: Dict[str, Any],
        device: torch.device
    ) -> None:
        """Evaluate generation metrics for a batch."""
        input_text = batch['input']
        target_text = batch['target']
        
        # Generate text
        inputs = tokenizer(
            input_text,
            return_tensors='pt',
            truncation=True,
            max_length=256
        ).to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode generated text
        generated_text = tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        # Add to metric
        metric.add_batch(generated_text, target_text)
