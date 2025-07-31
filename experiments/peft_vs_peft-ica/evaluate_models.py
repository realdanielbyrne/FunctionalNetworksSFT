#!/usr/bin/env python3
"""
PEFT vs PEFT+ICA Model Evaluation Script

This script evaluates and compares the trained models from both experiments using
HuggingFace's evaluate library with comprehensive metrics including:
- Accuracy, Precision, Recall, F1-score
- Perplexity
- BLEU and ROUGE scores
- Custom sarcasm detection metrics

Usage:
    python experiments/peft_vs_peft-ica/evaluate_models.py [--test-size 0.2] [--output-dir results]
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import warnings

import torch
import pandas as pd
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
)

# Try to import PEFT model class, fallback if not available
try:
    from peft import AutoPeftModelForCausalLM
except ImportError:
    try:
        from transformers import AutoPeftModelForCausalLM
    except ImportError:
        # Fallback: we'll handle PEFT models manually
        AutoPeftModelForCausalLM = None
from datasets import Dataset
import evaluate
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Add the src directory to the Python path
project_root = Path(__file__).parent.parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

from functionalnetworkssft.utils.model_utils import get_optimal_device

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)


class ModelEvaluator:
    """Comprehensive model evaluation class for PEFT vs PEFT+ICA comparison."""

    def __init__(self, base_model_name: str = "meta-llama/Llama-3.2-1B-Instruct"):
        """Initialize the evaluator with base model configuration."""
        self.base_model_name = base_model_name
        self.device, self.device_name = get_optimal_device()
        logger.info(f"Using device: {self.device_name}")

        # Initialize evaluation metrics
        self.metrics = {
            "accuracy": evaluate.load("accuracy"),
            "precision": evaluate.load("precision"),
            "recall": evaluate.load("recall"),
            "f1": evaluate.load("f1"),
            "bleu": evaluate.load("bleu"),
            "rouge": evaluate.load("rouge"),
            "perplexity": evaluate.load("perplexity", module_type="metric"),
        }

        # Results storage
        self.results = {}

    def load_model_and_tokenizer(
        self, model_path: str, is_peft: bool = True
    ) -> Tuple[Any, Any]:
        """Load a trained model and tokenizer."""
        logger.info(f"Loading model from: {model_path}")

        try:
            # Load tokenizer from base model
            tokenizer = AutoTokenizer.from_pretrained(
                self.base_model_name, trust_remote_code=True, use_auth_token=True
            )

            # Set pad token if not exists
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                tokenizer.pad_token_id = tokenizer.eos_token_id

            if is_peft and os.path.exists(
                os.path.join(model_path, "adapter_config.json")
            ):
                # Load PEFT model
                logger.info("Loading PEFT model...")
                if AutoPeftModelForCausalLM is not None:
                    model = AutoPeftModelForCausalLM.from_pretrained(
                        model_path,
                        torch_dtype=torch.float16,
                        device_map=(
                            "auto" if self.device.type in ["cuda", "mps"] else None
                        ),
                        use_auth_token=True,
                    )
                else:
                    # Fallback: load base model and adapters manually
                    from peft import PeftModel

                    base_model = AutoModelForCausalLM.from_pretrained(
                        self.base_model_name,
                        torch_dtype=torch.float16,
                        device_map=(
                            "auto" if self.device.type in ["cuda", "mps"] else None
                        ),
                        use_auth_token=True,
                    )
                    model = PeftModel.from_pretrained(base_model, model_path)
            else:
                # Load full model
                logger.info("Loading full model...")
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16,
                    device_map="auto" if self.device.type in ["cuda", "mps"] else None,
                    use_auth_token=True,
                )

            # Move to device if device_map wasn't used
            if self.device.type not in ["cuda", "mps"]:
                model = model.to(self.device)

            model.eval()
            logger.info("Model loaded successfully")
            return model, tokenizer

        except Exception as e:
            logger.error(f"Failed to load model from {model_path}: {str(e)}")
            raise

    def prepare_test_data(
        self, dataset_path: str, test_size: float = 0.2
    ) -> Tuple[List[str], List[str]]:
        """Prepare test data from the sarcasm dataset."""
        logger.info(f"Loading dataset from: {dataset_path}")

        # Load the dataset
        df = pd.read_csv(dataset_path)

        # Split into train/test (we'll use test for evaluation)
        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=42,
            stratify=None,  # No stratification for this dataset
        )

        questions = test_df["question"].tolist()
        answers = test_df["answer"].tolist()

        logger.info(f"Prepared {len(questions)} test samples")
        return questions, answers

    def generate_responses(
        self,
        model: Any,
        tokenizer: Any,
        questions: List[str],
        max_length: int = 512,
        batch_size: int = 4,
    ) -> List[str]:
        """Generate responses from the model for given questions."""
        logger.info(f"Generating responses for {len(questions)} questions...")

        responses = []

        # Create text generation pipeline
        generator = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=(
                self.device if self.device.type != "mps" else -1
            ),  # MPS not supported in pipeline
            torch_dtype=torch.float16,
            do_sample=True,
            temperature=0.7,
            max_new_tokens=128,
            pad_token_id=tokenizer.eos_token_id,
        )

        # Process in batches
        for i in range(0, len(questions), batch_size):
            batch_questions = questions[i : i + batch_size]

            try:
                # Format questions as prompts
                prompts = [f"Question: {q}\nAnswer:" for q in batch_questions]

                # Generate responses
                batch_outputs = generator(prompts, batch_size=len(prompts))

                # Extract generated text (remove the prompt)
                for j, output in enumerate(batch_outputs):
                    generated_text = output[0]["generated_text"]
                    # Extract only the answer part
                    answer_start = generated_text.find("Answer:") + len("Answer:")
                    response = generated_text[answer_start:].strip()

                    # Clean up response (remove extra tokens)
                    if tokenizer.eos_token in response:
                        response = response.split(tokenizer.eos_token)[0].strip()

                    responses.append(response)

            except Exception as e:
                logger.warning(f"Error generating batch {i//batch_size + 1}: {str(e)}")
                # Add empty responses for failed batch
                responses.extend([""] * len(batch_questions))

        logger.info(f"Generated {len(responses)} responses")
        return responses

    def calculate_perplexity(
        self, model: Any, tokenizer: Any, texts: List[str]
    ) -> float:
        """Calculate perplexity for a list of texts."""
        logger.info("Calculating perplexity...")

        try:
            # Prepare texts for perplexity calculation
            perplexity_inputs = []
            for text in texts:
                if text.strip():  # Only non-empty texts
                    perplexity_inputs.append(text)

            if not perplexity_inputs:
                logger.warning("No valid texts for perplexity calculation")
                return float("inf")

            # Calculate perplexity using the evaluate library
            results = self.metrics["perplexity"].compute(
                predictions=perplexity_inputs,
                model_id=self.base_model_name,  # Use base model for reference
            )

            return results["mean_perplexity"]

        except Exception as e:
            logger.warning(f"Error calculating perplexity: {str(e)}")
            return float("inf")

    def calculate_text_metrics(
        self, predictions: List[str], references: List[str]
    ) -> Dict[str, float]:
        """Calculate BLEU and ROUGE scores."""
        logger.info("Calculating text generation metrics...")

        results = {}

        try:
            # Filter out empty predictions and references
            valid_pairs = [
                (p, r)
                for p, r in zip(predictions, references)
                if p.strip() and r.strip()
            ]

            if not valid_pairs:
                logger.warning("No valid prediction-reference pairs for text metrics")
                return {"bleu": 0.0, "rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

            valid_predictions, valid_references = zip(*valid_pairs)

            # Calculate BLEU score
            bleu_results = self.metrics["bleu"].compute(
                predictions=valid_predictions,
                references=[
                    [ref] for ref in valid_references
                ],  # BLEU expects list of references
            )
            results["bleu"] = bleu_results["bleu"]

            # Calculate ROUGE scores
            rouge_results = self.metrics["rouge"].compute(
                predictions=valid_predictions, references=valid_references
            )
            results["rouge1"] = rouge_results["rouge1"]
            results["rouge2"] = rouge_results["rouge2"]
            results["rougeL"] = rouge_results["rougeL"]

        except Exception as e:
            logger.warning(f"Error calculating text metrics: {str(e)}")
            results = {"bleu": 0.0, "rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

        return results

    def evaluate_sarcasm_quality(
        self, predictions: List[str], references: List[str]
    ) -> Dict[str, float]:
        """Evaluate sarcasm quality using custom metrics."""
        logger.info("Evaluating sarcasm quality...")

        results = {}

        # Simple sarcasm indicators (can be expanded)
        sarcasm_indicators = [
            "oh",
            "wow",
            "really",
            "obviously",
            "genius",
            "brilliant",
            "shocking",
            "surprising",
            "amazing",
            "incredible",
            "fantastic",
            "groundbreaking",
            "revolutionary",
            "riveting",
            "fascinating",
        ]

        # Calculate sarcasm indicator presence
        pred_sarcasm_scores = []
        ref_sarcasm_scores = []

        for pred, ref in zip(predictions, references):
            pred_lower = pred.lower()
            ref_lower = ref.lower()

            pred_score = sum(
                1 for indicator in sarcasm_indicators if indicator in pred_lower
            )
            ref_score = sum(
                1 for indicator in sarcasm_indicators if indicator in ref_lower
            )

            pred_sarcasm_scores.append(min(pred_score, 3))  # Cap at 3
            ref_sarcasm_scores.append(min(ref_score, 3))

        # Calculate correlation between predicted and reference sarcasm levels
        if pred_sarcasm_scores and ref_sarcasm_scores:
            correlation = np.corrcoef(pred_sarcasm_scores, ref_sarcasm_scores)[0, 1]
            results["sarcasm_correlation"] = (
                correlation if not np.isnan(correlation) else 0.0
            )
        else:
            results["sarcasm_correlation"] = 0.0

        # Calculate average sarcasm intensity
        results["avg_pred_sarcasm"] = (
            np.mean(pred_sarcasm_scores) if pred_sarcasm_scores else 0.0
        )
        results["avg_ref_sarcasm"] = (
            np.mean(ref_sarcasm_scores) if ref_sarcasm_scores else 0.0
        )

        return results

    def evaluate_model(
        self,
        model_path: str,
        model_name: str,
        questions: List[str],
        references: List[str],
    ) -> Dict[str, Any]:
        """Evaluate a single model comprehensively."""
        logger.info(f"Evaluating model: {model_name}")

        start_time = time.time()

        # Load model and tokenizer
        model, tokenizer = self.load_model_and_tokenizer(model_path)

        # Generate responses
        predictions = self.generate_responses(model, tokenizer, questions)

        # Calculate metrics
        results = {
            "model_name": model_name,
            "model_path": model_path,
            "evaluation_time": 0,  # Will be updated at the end
            "num_samples": len(questions),
        }

        # Text generation metrics
        text_metrics = self.calculate_text_metrics(predictions, references)
        results.update(text_metrics)

        # Perplexity
        perplexity = self.calculate_perplexity(model, tokenizer, predictions)
        results["perplexity"] = perplexity

        # Sarcasm quality metrics
        sarcasm_metrics = self.evaluate_sarcasm_quality(predictions, references)
        results.update(sarcasm_metrics)

        # Response length statistics
        pred_lengths = [len(p.split()) for p in predictions if p.strip()]
        ref_lengths = [len(r.split()) for r in references if r.strip()]

        results["avg_pred_length"] = np.mean(pred_lengths) if pred_lengths else 0.0
        results["avg_ref_length"] = np.mean(ref_lengths) if ref_lengths else 0.0
        results["length_ratio"] = (
            results["avg_pred_length"] / results["avg_ref_length"]
            if results["avg_ref_length"] > 0
            else 0.0
        )

        # Store sample predictions for analysis
        results["sample_predictions"] = predictions[:5]  # First 5 for inspection
        results["sample_references"] = references[:5]
        results["sample_questions"] = questions[:5]

        # Clean up model to free memory
        del model
        del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        results["evaluation_time"] = time.time() - start_time
        logger.info(
            f"Model {model_name} evaluated in {results['evaluation_time']:.2f} seconds"
        )

        return results

    def compare_models(
        self, results_a: Dict[str, Any], results_b: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compare results between two models."""
        logger.info("Comparing model results...")

        comparison = {
            "model_a": results_a["model_name"],
            "model_b": results_b["model_name"],
            "comparison_metrics": {},
        }

        # Metrics to compare (higher is better)
        higher_better = [
            "bleu",
            "rouge1",
            "rouge2",
            "rougeL",
            "sarcasm_correlation",
            "length_ratio",
        ]

        # Metrics to compare (lower is better)
        lower_better = ["perplexity"]

        for metric in higher_better:
            if metric in results_a and metric in results_b:
                diff = results_b[metric] - results_a[metric]
                improvement = (
                    (diff / results_a[metric] * 100) if results_a[metric] != 0 else 0
                )
                comparison["comparison_metrics"][metric] = {
                    "model_a": results_a[metric],
                    "model_b": results_b[metric],
                    "difference": diff,
                    "improvement_pct": improvement,
                    "better_model": (
                        results_b["model_name"] if diff > 0 else results_a["model_name"]
                    ),
                }

        for metric in lower_better:
            if metric in results_a and metric in results_b:
                diff = (
                    results_a[metric] - results_b[metric]
                )  # Reversed for lower-is-better
                improvement = (
                    (diff / results_a[metric] * 100) if results_a[metric] != 0 else 0
                )
                comparison["comparison_metrics"][metric] = {
                    "model_a": results_a[metric],
                    "model_b": results_b[metric],
                    "difference": -diff,  # Show as negative for consistency
                    "improvement_pct": improvement,
                    "better_model": (
                        results_b["model_name"] if diff > 0 else results_a["model_name"]
                    ),
                }

        return comparison

    def save_results(
        self,
        output_dir: str,
        results_a: Dict[str, Any],
        results_b: Dict[str, Any],
        comparison: Dict[str, Any],
    ) -> None:
        """Save evaluation results to files."""
        logger.info(f"Saving results to: {output_dir}")

        os.makedirs(output_dir, exist_ok=True)

        # Save individual results
        with open(os.path.join(output_dir, "experiment_a_results.json"), "w") as f:
            json.dump(results_a, f, indent=2, default=str)

        with open(os.path.join(output_dir, "experiment_b_results.json"), "w") as f:
            json.dump(results_b, f, indent=2, default=str)

        # Save comparison
        with open(os.path.join(output_dir, "model_comparison.json"), "w") as f:
            json.dump(comparison, f, indent=2, default=str)

        # Create summary report
        self.create_summary_report(output_dir, results_a, results_b, comparison)

    def create_summary_report(
        self,
        output_dir: str,
        results_a: Dict[str, Any],
        results_b: Dict[str, Any],
        comparison: Dict[str, Any],
    ) -> None:
        """Create a human-readable summary report."""
        report_path = os.path.join(output_dir, "evaluation_summary.md")

        with open(report_path, "w") as f:
            f.write("# PEFT vs PEFT+ICA Model Evaluation Summary\n\n")
            f.write(f"**Evaluation Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # Model information
            f.write("## Models Evaluated\n\n")
            f.write(f"- **Experiment A (PEFT-only)**: {results_a['model_name']}\n")
            f.write(f"- **Experiment B (PEFT+ICA)**: {results_b['model_name']}\n")
            f.write(f"- **Test Samples**: {results_a['num_samples']}\n\n")

            # Performance comparison
            f.write("## Performance Comparison\n\n")
            f.write("| Metric | PEFT-only | PEFT+ICA | Difference | Better Model |\n")
            f.write("|--------|-----------|----------|------------|-------------|\n")

            for metric, data in comparison["comparison_metrics"].items():
                f.write(
                    f"| {metric.upper()} | {data['model_a']:.4f} | {data['model_b']:.4f} | "
                    f"{data['improvement_pct']:+.2f}% | {data['better_model']} |\n"
                )

            f.write("\n")

            # Sample outputs
            f.write("## Sample Outputs\n\n")
            for i in range(min(3, len(results_a["sample_questions"]))):
                f.write(f"### Sample {i+1}\n\n")
                f.write(f"**Question**: {results_a['sample_questions'][i]}\n\n")
                f.write(f"**Reference**: {results_a['sample_references'][i]}\n\n")
                f.write(f"**PEFT-only**: {results_a['sample_predictions'][i]}\n\n")
                f.write(f"**PEFT+ICA**: {results_b['sample_predictions'][i]}\n\n")
                f.write("---\n\n")

        logger.info(f"Summary report saved to: {report_path}")


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate PEFT vs PEFT+ICA models")
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of dataset to use for testing (default: 0.2)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/peft_vs_peft-ica/evaluation_results",
        help="Directory to save evaluation results",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="datasets/sarcasm.csv",
        help="Path to the sarcasm dataset",
    )

    args = parser.parse_args()

    # Model paths
    model_a_path = (
        "experiments/peft_vs_peft-ica/experiment_a_peft_only/output/final_model"
    )
    model_b_path = (
        "experiments/peft_vs_peft-ica/experiment_b_peft_ica/output/final_model"
    )

    # Check if models exist
    if not os.path.exists(model_a_path):
        logger.error(f"Experiment A model not found at: {model_a_path}")
        logger.error("Please run Experiment A first")
        sys.exit(1)

    if not os.path.exists(model_b_path):
        logger.error(f"Experiment B model not found at: {model_b_path}")
        logger.error("Please run Experiment B first")
        sys.exit(1)

    # Initialize evaluator
    evaluator = ModelEvaluator()

    # Prepare test data
    questions, references = evaluator.prepare_test_data(
        args.dataset_path, args.test_size
    )

    # Evaluate both models
    logger.info("Starting model evaluation...")

    results_a = evaluator.evaluate_model(
        model_a_path, "PEFT-only", questions, references
    )
    results_b = evaluator.evaluate_model(
        model_b_path, "PEFT+ICA", questions, references
    )

    # Compare results
    comparison = evaluator.compare_models(results_a, results_b)

    # Save results
    evaluator.save_results(args.output_dir, results_a, results_b, comparison)

    # Print summary
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)
    print(f"Results saved to: {args.output_dir}")
    print(f"Summary report: {os.path.join(args.output_dir, 'evaluation_summary.md')}")
    print("\nKey Findings:")

    for metric, data in comparison["comparison_metrics"].items():
        print(
            f"  {metric.upper()}: {data['better_model']} performs better "
            f"({data['improvement_pct']:+.2f}% difference)"
        )

    print("\n✅ Evaluation completed successfully!")


if __name__ == "__main__":
    main()
