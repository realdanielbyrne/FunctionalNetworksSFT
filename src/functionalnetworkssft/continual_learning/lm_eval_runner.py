"""
lm-eval harness integration for general capability benchmarks.

Runs MMLU (and optionally other benchmarks) on base and fine-tuned models
to measure whether continual learning preserves general capabilities.
Corresponds to DOC paper Table 9.
"""

import csv
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

LM_EVAL_CSV_COLUMNS = [
    "model",
    "method",
    "task_order",
    "seed",
    "benchmark",
    "metric",
    "value",
    "timestamp",
]


def _init_csv(path: Path, columns: List[str]) -> None:
    """Create CSV with header if it doesn't exist."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()


def _result_exists(
    path: Path, model: str, method: str, task_order: str, seed: int, benchmark: str
) -> bool:
    """Check if an lm-eval result row already exists."""
    if not path.exists():
        return False
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if (
                row["model"] == model
                and row["method"] == method
                and row["task_order"] == task_order
                and row["seed"] == str(seed)
                and row["benchmark"] == benchmark
            ):
                return True
    return False


def run_lm_eval_benchmarks(
    model: torch.nn.Module,
    tokenizer: Any,
    benchmarks: List[str] = None,
    batch_size: int = 8,
    device: str = "cuda",
) -> Dict[str, Dict[str, float]]:
    """Run lm-eval benchmarks on a model.

    Args:
        model: Model to evaluate.
        tokenizer: Tokenizer for the model.
        benchmarks: List of benchmark names (default: ["mmlu"]).
        batch_size: Batch size for evaluation.
        device: Device for evaluation.

    Returns:
        Dict mapping benchmark name -> metric name -> score.
    """
    if benchmarks is None:
        benchmarks = ["mmlu"]

    try:
        from lm_eval import evaluator
        from lm_eval.models.huggingface import HFLM
    except ImportError:
        logger.error(
            "lm_eval not installed. Install with: pip install lm-eval"
        )
        return {}

    logger.info(f"Running lm-eval benchmarks: {benchmarks}")

    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size)
    results = evaluator.simple_evaluate(
        model=lm,
        tasks=benchmarks,
        batch_size=batch_size,
    )

    parsed = {}
    for task_name, task_results in results.get("results", {}).items():
        parsed[task_name] = {}
        for metric_key, metric_val in task_results.items():
            if isinstance(metric_val, (int, float)):
                parsed[task_name][metric_key] = float(metric_val)

    return parsed


def run_lm_eval_on_base_model(
    model_key: str,
    model_name: str,
    benchmarks: List[str],
    csv_path: Path,
    device: str = "cuda",
) -> None:
    """Run lm-eval on the base (pre-CL) model.

    Provides the baseline for DOC Table 9 comparison.

    Args:
        model_key: Model identifier (e.g. "llama-7b").
        model_name: HuggingFace model name.
        benchmarks: Benchmarks to run.
        csv_path: Path to output CSV.
        device: Device to use.
    """
    _init_csv(csv_path, LM_EVAL_CSV_COLUMNS)

    # Check if base model results already exist
    all_exist = all(
        _result_exists(csv_path, model_key, "base", "none", 0, b)
        for b in benchmarks
    )
    if all_exist:
        logger.info(f"[SKIP] Base model lm-eval already computed for {model_key}")
        return

    logger.info(f"Running lm-eval on base model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, token=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map=device,
        token=True,
    )

    results = run_lm_eval_benchmarks(model, tokenizer, benchmarks, device=device)

    for benchmark, metrics in results.items():
        for metric_name, value in metrics.items():
            if _result_exists(csv_path, model_key, "base", "none", 0, benchmark):
                continue
            row = {
                "model": model_key,
                "method": "base",
                "task_order": "none",
                "seed": 0,
                "benchmark": benchmark,
                "metric": metric_name,
                "value": f"{value:.4f}",
                "timestamp": datetime.now().isoformat(),
            }
            with open(csv_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=LM_EVAL_CSV_COLUMNS)
                writer.writerow(row)

    logger.info(f"Base model lm-eval results saved to {csv_path}")

    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_lm_eval_on_completed_experiments(
    model_key: str,
    results_csv: Path,
    lm_eval_csv: Path,
    final_models_dir: Path,
    benchmarks: List[str],
    device: str = "cuda",
) -> None:
    """Run lm-eval on all completed CL experiments.

    Loads final models saved after CL training and evaluates them.

    Args:
        model_key: Model identifier.
        results_csv: Path to the experiments results CSV.
        lm_eval_csv: Path to the lm-eval output CSV.
        final_models_dir: Directory containing saved final models.
        benchmarks: Benchmarks to run.
        device: Device to use.
    """
    _init_csv(lm_eval_csv, LM_EVAL_CSV_COLUMNS)

    if not results_csv.exists():
        logger.warning(f"No experiment results found: {results_csv}")
        return

    import pandas as pd

    df = pd.read_csv(results_csv)
    model_df = df[df["model"] == model_key]

    for _, row in model_df.iterrows():
        method = row["method"]
        task_order = row["task_order"]
        seed = int(row["seed"])

        # Check if already evaluated
        all_exist = all(
            _result_exists(lm_eval_csv, model_key, method, task_order, seed, b)
            for b in benchmarks
        )
        if all_exist:
            logger.info(
                f"[SKIP] lm-eval already computed: {method}/{task_order}/seed{seed}"
            )
            continue

        # Find the saved final model
        model_dir = (
            final_models_dir
            / f"{model_key}_{method}_{task_order}_seed{seed}"
        )
        if not model_dir.exists():
            logger.warning(
                f"Final model not found: {model_dir}. "
                f"Skipping lm-eval for {method}/{task_order}/seed{seed}."
            )
            continue

        logger.info(
            f"Running lm-eval: {method}/{task_order}/seed{seed}"
        )

        try:
            tokenizer = AutoTokenizer.from_pretrained(model_dir, token=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            model = AutoModelForCausalLM.from_pretrained(
                model_dir,
                torch_dtype=torch.float16
                if torch.cuda.is_available()
                else torch.float32,
                device_map=device,
                token=True,
            )

            results = run_lm_eval_benchmarks(
                model, tokenizer, benchmarks, device=device
            )

            for benchmark, metrics in results.items():
                for metric_name, value in metrics.items():
                    csv_row = {
                        "model": model_key,
                        "method": method,
                        "task_order": task_order,
                        "seed": seed,
                        "benchmark": benchmark,
                        "metric": metric_name,
                        "value": f"{value:.4f}",
                        "timestamp": datetime.now().isoformat(),
                    }
                    with open(lm_eval_csv, "a", newline="") as f:
                        writer = csv.DictWriter(
                            f, fieldnames=LM_EVAL_CSV_COLUMNS
                        )
                        writer.writerow(csv_row)

            logger.info(
                f"  lm-eval done: {method}/{task_order}/seed{seed}"
            )

        except Exception as e:
            logger.error(
                f"  lm-eval failed: {method}/{task_order}/seed{seed}: {e}",
                exc_info=True,
            )
        finally:
            try:
                del model, tokenizer
            except NameError:
                pass
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
