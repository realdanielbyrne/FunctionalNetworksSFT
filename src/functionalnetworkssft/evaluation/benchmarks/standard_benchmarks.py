"""
Standard LLM benchmark evaluations.

This module implements evaluations for standard language model benchmarks including:
- MMLU (Massive Multitask Language Understanding)
- HellaSwag (Commonsense reasoning)
- ARC (AI2 Reasoning Challenge)
- GSM8K (Mathematical reasoning)
- HumanEval (Code generation)
"""

import logging
import re
import time
from typing import Any, Dict, List, Optional, Union
import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from ..core.metrics import BaseMetric, register_metric
from ..core.results import MetricResult
from ..core.config import BenchmarkConfig

logger = logging.getLogger(__name__)


@register_metric("mmlu_accuracy")
class MMLUAccuracyMetric(BaseMetric):
    """Accuracy metric for MMLU evaluation."""
    
    def compute_score(self, predictions: Any, references: Any, **kwargs) -> float:
        """Compute accuracy for MMLU multiple choice."""
        if isinstance(predictions, str) and isinstance(references, str):
            # Extract choice letter (A, B, C, D)
            pred_choice = self._extract_choice(predictions)
            ref_choice = self._extract_choice(references)
            return 1.0 if pred_choice == ref_choice else 0.0
        return 0.0
    
    def _extract_choice(self, text: str) -> str:
        """Extract choice letter from text."""
        # Look for patterns like "A)", "(A)", "A.", "A:"
        match = re.search(r'\b([ABCD])\b', text.upper())
        return match.group(1) if match else ""


@register_metric("hellaswag_accuracy")
class HellaSwagAccuracyMetric(BaseMetric):
    """Accuracy metric for HellaSwag evaluation."""
    
    def compute_score(self, predictions: Any, references: Any, **kwargs) -> float:
        """Compute accuracy for HellaSwag multiple choice."""
        if isinstance(predictions, (int, str)) and isinstance(references, (int, str)):
            try:
                pred_idx = int(predictions) if isinstance(predictions, str) else predictions
                ref_idx = int(references) if isinstance(references, str) else references
                return 1.0 if pred_idx == ref_idx else 0.0
            except ValueError:
                return 0.0
        return 0.0


@register_metric("arc_accuracy")
class ARCAccuracyMetric(BaseMetric):
    """Accuracy metric for ARC evaluation."""
    
    def compute_score(self, predictions: Any, references: Any, **kwargs) -> float:
        """Compute accuracy for ARC multiple choice."""
        if isinstance(predictions, str) and isinstance(references, str):
            # Normalize and compare
            pred_clean = predictions.strip().lower()
            ref_clean = references.strip().lower()
            return 1.0 if pred_clean == ref_clean else 0.0
        return 0.0


@register_metric("gsm8k_accuracy")
class GSM8KAccuracyMetric(BaseMetric):
    """Accuracy metric for GSM8K mathematical reasoning."""
    
    def compute_score(self, predictions: Any, references: Any, **kwargs) -> float:
        """Compute accuracy for GSM8K numerical answers."""
        pred_num = self._extract_number(str(predictions))
        ref_num = self._extract_number(str(references))
        
        if pred_num is not None and ref_num is not None:
            # Allow small numerical tolerance
            return 1.0 if abs(pred_num - ref_num) < 1e-6 else 0.0
        return 0.0
    
    def _extract_number(self, text: str) -> Optional[float]:
        """Extract numerical answer from text."""
        # Look for numbers in the text
        numbers = re.findall(r'-?\d+\.?\d*', text)
        if numbers:
            try:
                return float(numbers[-1])  # Take the last number found
            except ValueError:
                pass
        return None


@register_metric("humaneval_pass_at_k")
class HumanEvalPassAtKMetric(BaseMetric):
    """Pass@k metric for HumanEval code generation."""
    
    def __init__(self, name: str = "humaneval_pass_at_k", **kwargs):
        super().__init__(name, **kwargs)
        self.k = kwargs.get('k', 1)
        self.test_results = []
    
    def compute_score(self, predictions: Any, references: Any, **kwargs) -> float:
        """Compute pass@k for code generation."""
        # This would require code execution which is complex
        # For now, return a simple syntax check
        if isinstance(predictions, str):
            return 1.0 if self._is_valid_python(predictions) else 0.0
        return 0.0
    
    def _is_valid_python(self, code: str) -> bool:
        """Simple Python syntax validation."""
        try:
            compile(code, '<string>', 'exec')
            return True
        except SyntaxError:
            return False


class MMLUEvaluator:
    """Evaluator for MMLU benchmark."""
    
    def __init__(self):
        self.subjects = [
            'abstract_algebra', 'anatomy', 'astronomy', 'business_ethics',
            'clinical_knowledge', 'college_biology', 'college_chemistry',
            'college_computer_science', 'college_mathematics', 'college_medicine',
            'college_physics', 'computer_security', 'conceptual_physics',
            'econometrics', 'electrical_engineering', 'elementary_mathematics',
            'formal_logic', 'global_facts', 'high_school_biology',
            'high_school_chemistry', 'high_school_computer_science',
            'high_school_european_history', 'high_school_geography',
            'high_school_government_and_politics', 'high_school_macroeconomics',
            'high_school_mathematics', 'high_school_microeconomics',
            'high_school_physics', 'high_school_psychology',
            'high_school_statistics', 'high_school_us_history',
            'high_school_world_history', 'human_aging', 'human_sexuality',
            'international_law', 'jurisprudence', 'logical_fallacies',
            'machine_learning', 'management', 'marketing', 'medical_genetics',
            'miscellaneous', 'moral_disputes', 'moral_scenarios', 'nutrition',
            'philosophy', 'prehistory', 'professional_accounting',
            'professional_law', 'professional_medicine', 'professional_psychology',
            'public_relations', 'security_studies', 'sociology', 'us_foreign_policy',
            'virology', 'world_religions'
        ]
    
    def evaluate(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        config: BenchmarkConfig,
        device: torch.device,
        **kwargs
    ) -> List[MetricResult]:
        """Evaluate MMLU benchmark."""
        logger.info("Running MMLU evaluation")
        
        # Initialize metrics
        accuracy_metric = MMLUAccuracyMetric()
        
        # Load MMLU data (simplified for demo)
        eval_data = self._load_mmlu_data(config)
        
        # Evaluate
        model.eval()
        with torch.no_grad():
            for item in eval_data:
                prediction = self._generate_mmlu_answer(
                    model, tokenizer, item, device
                )
                accuracy_metric.add_batch(prediction, item['answer'])
        
        return [accuracy_metric.compute_final_score()]
    
    def _load_mmlu_data(self, config: BenchmarkConfig) -> List[Dict[str, Any]]:
        """Load MMLU evaluation data."""
        # Simplified test data for demonstration
        test_data = [
            {
                'question': 'What is the capital of France?',
                'choices': ['A) London', 'B) Berlin', 'C) Paris', 'D) Madrid'],
                'answer': 'C',
                'subject': 'geography'
            },
            {
                'question': 'What is 2 + 2?',
                'choices': ['A) 3', 'B) 4', 'C) 5', 'D) 6'],
                'answer': 'B',
                'subject': 'mathematics'
            },
        ]
        
        max_samples = config.max_samples
        if max_samples and max_samples < len(test_data):
            test_data = test_data[:max_samples]
        
        return test_data
    
    def _generate_mmlu_answer(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        item: Dict[str, Any],
        device: torch.device
    ) -> str:
        """Generate answer for MMLU question."""
        # Format prompt
        prompt = f"Question: {item['question']}\n"
        for choice in item['choices']:
            prompt += f"{choice}\n"
        prompt += "Answer:"
        
        # Tokenize and generate
        inputs = tokenizer(prompt, return_tensors='pt', truncation=True).to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=5,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode answer
        answer = tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        ).strip()
        
        return answer


class HellaSwagEvaluator:
    """Evaluator for HellaSwag benchmark."""
    
    def evaluate(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        config: BenchmarkConfig,
        device: torch.device,
        **kwargs
    ) -> List[MetricResult]:
        """Evaluate HellaSwag benchmark."""
        logger.info("Running HellaSwag evaluation")
        
        accuracy_metric = HellaSwagAccuracyMetric()
        eval_data = self._load_hellaswag_data(config)
        
        model.eval()
        with torch.no_grad():
            for item in eval_data:
                prediction = self._evaluate_hellaswag_item(
                    model, tokenizer, item, device
                )
                accuracy_metric.add_batch(prediction, item['label'])
        
        return [accuracy_metric.compute_final_score()]
    
    def _load_hellaswag_data(self, config: BenchmarkConfig) -> List[Dict[str, Any]]:
        """Load HellaSwag evaluation data."""
        # Simplified test data
        test_data = [
            {
                'ctx': 'A person is cooking in the kitchen.',
                'endings': [
                    'They turn on the stove.',
                    'They go to sleep.',
                    'They start dancing.',
                    'They call their friend.'
                ],
                'label': 0
            },
        ]
        
        max_samples = config.max_samples
        if max_samples and max_samples < len(test_data):
            test_data = test_data[:max_samples]
        
        return test_data
    
    def _evaluate_hellaswag_item(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        item: Dict[str, Any],
        device: torch.device
    ) -> int:
        """Evaluate single HellaSwag item using likelihood scoring."""
        ctx = item['ctx']
        endings = item['endings']
        
        scores = []
        for ending in endings:
            full_text = ctx + ' ' + ending
            
            # Tokenize
            inputs = tokenizer(full_text, return_tensors='pt').to(device)
            
            # Get likelihood
            with torch.no_grad():
                outputs = model(**inputs, labels=inputs['input_ids'])
                loss = outputs.loss.item()
                scores.append(-loss)  # Higher is better
        
        # Return index of highest scoring ending
        return scores.index(max(scores))


class ARCEvaluator:
    """Evaluator for ARC benchmark."""
    
    def evaluate(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        config: BenchmarkConfig,
        device: torch.device,
        **kwargs
    ) -> List[MetricResult]:
        """Evaluate ARC benchmark."""
        logger.info("Running ARC evaluation")
        
        accuracy_metric = ARCAccuracyMetric()
        eval_data = self._load_arc_data(config)
        
        model.eval()
        with torch.no_grad():
            for item in eval_data:
                prediction = self._generate_arc_answer(
                    model, tokenizer, item, device
                )
                accuracy_metric.add_batch(prediction, item['answerKey'])
        
        return [accuracy_metric.compute_final_score()]
    
    def _load_arc_data(self, config: BenchmarkConfig) -> List[Dict[str, Any]]:
        """Load ARC evaluation data."""
        # Simplified test data
        test_data = [
            {
                'question': 'What do plants need to grow?',
                'choices': {
                    'text': ['Water', 'Darkness', 'Cold', 'Noise'],
                    'label': ['A', 'B', 'C', 'D']
                },
                'answerKey': 'A'
            },
        ]
        
        max_samples = config.max_samples
        if max_samples and max_samples < len(test_data):
            test_data = test_data[:max_samples]
        
        return test_data
    
    def _generate_arc_answer(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        item: Dict[str, Any],
        device: torch.device
    ) -> str:
        """Generate answer for ARC question."""
        question = item['question']
        choices = item['choices']
        
        # Format prompt
        prompt = f"Question: {question}\n"
        for label, text in zip(choices['label'], choices['text']):
            prompt += f"{label}) {text}\n"
        prompt += "Answer:"
        
        # Generate answer
        inputs = tokenizer(prompt, return_tensors='pt', truncation=True).to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=5,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        
        answer = tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        ).strip()
        
        return answer


class GSM8KEvaluator:
    """Evaluator for GSM8K mathematical reasoning benchmark."""
    
    def evaluate(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        config: BenchmarkConfig,
        device: torch.device,
        **kwargs
    ) -> List[MetricResult]:
        """Evaluate GSM8K benchmark."""
        logger.info("Running GSM8K evaluation")
        
        accuracy_metric = GSM8KAccuracyMetric()
        eval_data = self._load_gsm8k_data(config)
        
        model.eval()
        with torch.no_grad():
            for item in eval_data:
                prediction = self._generate_gsm8k_answer(
                    model, tokenizer, item, device
                )
                accuracy_metric.add_batch(prediction, item['answer'])
        
        return [accuracy_metric.compute_final_score()]
    
    def _load_gsm8k_data(self, config: BenchmarkConfig) -> List[Dict[str, Any]]:
        """Load GSM8K evaluation data."""
        # Simplified test data
        test_data = [
            {
                'question': 'If John has 5 apples and gives away 2, how many does he have left?',
                'answer': '3'
            },
            {
                'question': 'A store sells 15 items per hour. How many items do they sell in 4 hours?',
                'answer': '60'
            },
        ]
        
        max_samples = config.max_samples
        if max_samples and max_samples < len(test_data):
            test_data = test_data[:max_samples]
        
        return test_data
    
    def _generate_gsm8k_answer(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        item: Dict[str, Any],
        device: torch.device
    ) -> str:
        """Generate answer for GSM8K question."""
        prompt = f"Question: {item['question']}\nAnswer:"
        
        inputs = tokenizer(prompt, return_tensors='pt', truncation=True).to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        
        answer = tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        ).strip()
        
        return answer


class HumanEvalEvaluator:
    """Evaluator for HumanEval code generation benchmark."""
    
    def evaluate(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        config: BenchmarkConfig,
        device: torch.device,
        **kwargs
    ) -> List[MetricResult]:
        """Evaluate HumanEval benchmark."""
        logger.info("Running HumanEval evaluation")
        
        pass_at_k_metric = HumanEvalPassAtKMetric()
        eval_data = self._load_humaneval_data(config)
        
        model.eval()
        with torch.no_grad():
            for item in eval_data:
                prediction = self._generate_code(
                    model, tokenizer, item, device
                )
                pass_at_k_metric.add_batch(prediction, item['canonical_solution'])
        
        return [pass_at_k_metric.compute_final_score()]
    
    def _load_humaneval_data(self, config: BenchmarkConfig) -> List[Dict[str, Any]]:
        """Load HumanEval evaluation data."""
        # Simplified test data
        test_data = [
            {
                'prompt': 'def add_two_numbers(a, b):\n    """Add two numbers and return the result."""\n    ',
                'canonical_solution': 'return a + b',
                'test': 'assert add_two_numbers(2, 3) == 5'
            },
        ]
        
        max_samples = config.max_samples
        if max_samples and max_samples < len(test_data):
            test_data = test_data[:max_samples]
        
        return test_data
    
    def _generate_code(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        item: Dict[str, Any],
        device: torch.device
    ) -> str:
        """Generate code completion."""
        prompt = item['prompt']
        
        inputs = tokenizer(prompt, return_tensors='pt', truncation=True).to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                stop_strings=['\n\n', 'def ', 'class ']
            )
        
        code = tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        ).strip()
        
        return code


class StandardBenchmarkEvaluator:
    """Main evaluator for standard LLM benchmarks."""
    
    def __init__(self):
        self.evaluators = {
            'mmlu': MMLUEvaluator(),
            'hellaswag': HellaSwagEvaluator(),
            'arc': ARCEvaluator(),
            'gsm8k': GSM8KEvaluator(),
            'humaneval': HumanEvalEvaluator(),
        }
    
    def evaluate(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        config: BenchmarkConfig,
        device: torch.device,
        **kwargs
    ) -> List[MetricResult]:
        """Evaluate standard benchmarks."""
        benchmark_name = config.name.lower()
        
        if benchmark_name in self.evaluators:
            evaluator = self.evaluators[benchmark_name]
            return evaluator.evaluate(model, tokenizer, config, device, **kwargs)
        else:
            logger.warning(f"Unknown benchmark: {benchmark_name}")
            return []
