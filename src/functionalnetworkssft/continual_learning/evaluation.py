"""
Main evaluation loop for continual learning experiments.
"""

import argparse
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch.optim import AdamW
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from .task_data.config import TASK_ORDERS, get_dataset_config, get_task_order
from .task_data.loaders import CLDatasetLoader
from .methods.base import ContinualLearningMethod
from .methods.doc import DOC
from .methods.ewc import EWC
from .methods.ica_networks import ICANetworksCL
from .methods.lora_baseline import LoRABaseline
from .methods.lwf import LwF
from .methods.o_lora import OLoRA
from .metrics import ContinualLearningMetrics

logger = logging.getLogger(__name__)

METHODS = {
    "lora": LoRABaseline,
    "ewc": EWC,
    "lwf": LwF,
    "o_lora": OLoRA,
    "doc": DOC,
    "ica_networks": ICANetworksCL,
}

MODEL_CONFIGS = {
    "llama-7b": {
        "model_name": "meta-llama/Llama-2-7b-hf",
        "learning_rate": 1e-4,
        "batch_size": 8,
        "num_steps_per_task": 1000,
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "lora_target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
    },
    "llama-13b": {
        "model_name": "meta-llama/Llama-2-13b-hf",
        "learning_rate": 1e-4,
        "batch_size": 8,
        "num_steps_per_task": 1000,
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "lora_target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
    },
    "llama-3.2-1b": {
        "model_name": "meta-llama/Llama-3.2-1B-Instruct",
        "learning_rate": 1e-4,
        "batch_size": 8,
        "num_steps_per_task": 1000,
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "lora_target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
    },
    "t5-large": {
        "model_name": "google/t5-large",
        "learning_rate": 1e-3,
        "batch_size": 64,
        "num_steps_per_task": 1000,
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "lora_target_modules": ["q", "v"],
    },
}


def _is_seq2seq_model(model_key: str) -> bool:
    """Check if the model key refers to a seq2seq model (e.g., T5)."""
    return "t5" in model_key.lower()


def load_model_and_tokenizer(
    model_key: str, device: str = "auto", use_auth_token: bool = True
):
    """Load model and tokenizer with LoRA configuration."""
    config = MODEL_CONFIGS[model_key]
    is_seq2seq = _is_seq2seq_model(model_key)
    logger.info(f"Loading model: {config['model_name']} (seq2seq={is_seq2seq})")

    tokenizer = AutoTokenizer.from_pretrained(
        config["model_name"], token=use_auth_token
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_cls = AutoModelForSeq2SeqLM if is_seq2seq else AutoModelForCausalLM
    model = model_cls.from_pretrained(
        config["model_name"],
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map=device,
        token=use_auth_token,
    )

    task_type = TaskType.SEQ_2_SEQ_LM if is_seq2seq else TaskType.CAUSAL_LM
    lora_config = LoraConfig(
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        target_modules=config["lora_target_modules"],
        task_type=task_type,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer, config


def _match_prediction(
    predicted_text: str, ground_truth: str, valid_answers: List[str]
) -> bool:
    """Robust answer matching: exact > normalized prefix > best-matching answer.

    Args:
        predicted_text: Raw generated text (already stripped).
        ground_truth: Expected answer label (lowercase).
        valid_answers: All valid answer labels for the task (lowercase).

    Returns:
        True if the prediction matches the ground truth.
    """
    pred = re.sub(r"[^\w\s]", "", predicted_text.strip()).split()
    if not pred:
        return False
    pred_word = pred[0].lower()
    gt = ground_truth.lower()

    # 1. Exact match
    if pred_word == gt:
        return True

    # 2. Ground truth starts with prediction or vice versa (min 3 chars)
    if len(pred_word) >= 3 and (gt.startswith(pred_word) or pred_word.startswith(gt)):
        return True

    # 3. Check if pred matches any valid answer uniquely
    matches = [
        a for a in valid_answers
        if a.startswith(pred_word) or pred_word.startswith(a)
    ]
    if len(matches) == 1 and matches[0] == gt:
        return True

    return False


def evaluate_task(
    model: torch.nn.Module,
    tokenizer,
    task_name: str,
    test_data,
    device: str = "cuda",
) -> float:
    """Evaluate model accuracy on a task using generation.

    Test data contains prompt-only input_ids (no answer) and label_idx
    for ground truth comparison.
    """
    model.eval()
    correct = 0
    total = 0

    config = get_dataset_config(task_name)
    valid_answers = [v.lower() for v in config.label_map.values()]

    with torch.no_grad():
        for example in tqdm(test_data, desc=f"Eval {task_name}", leave=False):
            input_ids = torch.tensor([example["input_ids"]]).to(device)
            attention_mask = torch.tensor([example["attention_mask"]]).to(device)

            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

            # Extract only the newly generated tokens
            new_tokens = outputs[0][input_ids.shape[1]:]
            generated = tokenizer.decode(new_tokens, skip_special_tokens=True)

            # Ground truth from stored label index
            label_idx = example["label_idx"]
            ground_truth = config.label_map[label_idx].lower()

            if _match_prediction(generated, ground_truth, valid_answers):
                correct += 1
            total += 1

    accuracy = (correct / total) * 100 if total > 0 else 0.0

    # Sanity checks
    num_classes = config.num_classes
    chance_level = 100.0 / num_classes
    if accuracy < chance_level * 0.5:
        logger.warning(
            f"Evaluation sanity check: {task_name} accuracy {accuracy:.1f}% is below "
            f"half of chance level ({chance_level * 0.5:.1f}%). "
            f"This may indicate a prompt/generation mismatch."
        )
    if accuracy > 99.5:
        logger.warning(
            f"Evaluation sanity check: {task_name} accuracy {accuracy:.1f}% is "
            f"suspiciously high (>99.5%). Check for possible data leakage."
        )

    return accuracy


def train_on_task(
    cl_method: ContinualLearningMethod,
    train_data,
    config: Dict,
    device: str = "cuda",
    optimizer_state: Optional[Dict] = None,
    scheduler_state: Optional[Dict] = None,
    start_step: int = 0,
) -> Tuple[Dict, Dict]:
    """Train model on a single task using the CL method.

    Args:
        cl_method: The continual learning method instance.
        train_data: Training dataset.
        config: Model/training configuration dict.
        device: Device for tensors.
        optimizer_state: Optional optimizer state dict to resume from.
        scheduler_state: Optional scheduler state dict to resume from.
        start_step: Step to resume training from.

    Returns:
        Tuple of (optimizer_state_dict, scheduler_state_dict) for checkpointing.
    """
    model = cl_method.model
    model.train()

    optimizer = AdamW(cl_method.get_trainable_parameters(), lr=config["learning_rate"])
    num_steps = config["num_steps_per_task"]
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_steps // 10, num_training_steps=num_steps
    )

    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
    if scheduler_state is not None:
        scheduler.load_state_dict(scheduler_state)

    step = start_step
    pbar = tqdm(total=num_steps, initial=start_step, desc="Training")

    while step < num_steps:
        train_data = train_data.shuffle()
        for i in range(0, len(train_data), config["batch_size"]):
            if step >= num_steps:
                break

            batch_data = train_data[i : i + config["batch_size"]]
            batch = {
                "input_ids": torch.tensor(batch_data["input_ids"]).to(device),
                "attention_mask": torch.tensor(batch_data["attention_mask"]).to(device),
                "labels": torch.tensor(batch_data["labels"]).to(device),
            }

            loss = cl_method.compute_loss(batch, cl_method.current_task_idx)

            optimizer.zero_grad()
            loss.backward()

            # Apply gradient projection for DOC method
            if hasattr(cl_method, 'apply_gradient_projection'):
                cl_method.apply_gradient_projection()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            step += 1
            pbar.update(1)
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    pbar.close()
    return optimizer.state_dict(), scheduler.state_dict()


def run_single_task_cycle(
    T: int,
    task_order: List[str],
    cl_method: ContinualLearningMethod,
    dataset_loader: CLDatasetLoader,
    tokenizer,
    config: Dict,
    metrics: ContinualLearningMetrics,
    device: str = "cuda",
    optimizer_state: Optional[Dict] = None,
    scheduler_state: Optional[Dict] = None,
) -> Tuple[Dict, Dict]:
    """Train on task T and evaluate all tasks 0..T.

    This is the core task cycle extracted for reuse by the orchestrator.
    It handles before_task, training, after_task, and evaluation.

    Args:
        T: Index of the current task in the task order (0-indexed).
        task_order: List of task names in sequence.
        cl_method: The continual learning method instance.
        dataset_loader: Dataset loader with caching.
        tokenizer: Model tokenizer for evaluation.
        config: Model/training configuration dict.
        metrics: ContinualLearningMetrics to record accuracy into.
        device: Device for tensors.
        optimizer_state: Optional optimizer state to resume from.
        scheduler_state: Optional scheduler state to resume from.

    Returns:
        Tuple of (optimizer_state_dict, scheduler_state_dict).
    """
    num_tasks = len(task_order)
    task_name = task_order[T]
    logger.info(f"\nTask {T + 1}/{num_tasks}: {task_name}")

    task_data = dataset_loader.load_dataset(task_name)
    cl_method.before_task(T, task_name, task_data)
    opt_state, sched_state = train_on_task(
        cl_method, task_data["train"], config, device,
        optimizer_state=optimizer_state,
        scheduler_state=scheduler_state,
    )
    cl_method.after_task(T, task_name, task_data)

    eval_model = cl_method.get_model_for_inference()
    logger.info(f"Evaluating on tasks 1-{T + 1}...")

    for t in range(T + 1):
        eval_task = task_order[t]
        eval_data = dataset_loader.load_dataset(eval_task)
        accuracy = evaluate_task(
            eval_model, tokenizer, eval_task, eval_data["test"], device
        )
        metrics.record_accuracy(t, T, accuracy)
        logger.info(f"  Task {t + 1} ({eval_task}): {accuracy:.2f}%")

    logger.info(f"Current AA: {metrics.compute_average_accuracy(T + 1):.2f}%")
    if T > 0:
        logger.info(f"Current BWT: {metrics.compute_backward_transfer(T + 1):.2f}")

    return opt_state, sched_state


def compute_baseline_accuracies(
    model_key: str,
    task_order: List[str],
    dataset_loader: CLDatasetLoader,
    tokenizer,
    seed: int = 42,
    device: str = "cuda",
) -> Dict[int, float]:
    """
    Compute single-task baseline accuracies for FWT computation.

    For each task, trains a fresh LoRA model on that task alone and
    records the accuracy. These serve as the 'standard fine-tuning'
    baselines in the FWT formula.

    Returns:
        Dictionary mapping task index to baseline accuracy.
    """
    baselines = {}

    for t, task_name in enumerate(task_order):
        logger.info(f"Computing baseline for task {t + 1}/{len(task_order)}: {task_name}")
        torch.manual_seed(seed)

        model, tok, config = load_model_and_tokenizer(model_key, device)
        baseline_method = LoRABaseline(model, config)

        task_data = dataset_loader.load_dataset(task_name)
        baseline_method.before_task(0, task_name, task_data)
        train_on_task(baseline_method, task_data["train"], config, device)

        accuracy = evaluate_task(model, tok, task_name, task_data["test"], device)
        baselines[t] = accuracy
        logger.info(f"  Baseline accuracy for {task_name}: {accuracy:.2f}%")

        # Free memory
        del model, tok, baseline_method
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return baselines


def run_continual_learning_evaluation(
    model_key: str,
    method_name: str,
    task_order_name: str,
    output_dir: Path,
    seed: int = 42,
    compute_baselines: bool = False,
    method_kwargs: Optional[Dict] = None,
    device: str = "cuda",
) -> Dict:
    """
    Run complete continual learning evaluation.

    Args:
        model_key: Model identifier (llama-7b, llama-13b, etc.)
        method_name: CL method name
        task_order_name: Task order identifier
        output_dir: Output directory for results
        seed: Random seed
        compute_baselines: Whether to compute FWT baselines
        method_kwargs: Additional kwargs for the CL method
        device: Device to use

    Returns:
        Dictionary containing all metrics and results
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    task_order = get_task_order(task_order_name)
    num_tasks = len(task_order)

    logger.info(f"Running {method_name} on {task_order_name}")
    logger.info(f"Tasks: {task_order}")

    model, tokenizer, config = load_model_and_tokenizer(model_key, device)
    dataset_loader = CLDatasetLoader(tokenizer, max_seq_length=512, seed=seed)
    metrics = ContinualLearningMetrics(num_tasks=num_tasks, task_names=task_order)

    method_kwargs = method_kwargs or {}
    if method_name not in METHODS:
        raise ValueError(
            f"Unknown method: {method_name}. Available: {list(METHODS.keys())}"
        )

    cl_method = METHODS[method_name](model, config, **method_kwargs)

    # Compute FWT baselines if requested
    if compute_baselines:
        logger.info("Computing single-task baseline accuracies for FWT...")
        baselines = compute_baseline_accuracies(
            model_key, task_order, dataset_loader, tokenizer, seed, device
        )
        for t, acc in baselines.items():
            metrics.set_baseline_accuracy(t, acc)
        logger.info("Baseline computation complete.")

    opt_state, sched_state = None, None
    for T in range(num_tasks):
        opt_state, sched_state = run_single_task_cycle(
            T, task_order, cl_method, dataset_loader,
            tokenizer, config, metrics, device,
            optimizer_state=opt_state,
            scheduler_state=sched_state,
        )

    results = metrics.get_full_report()
    results.update(
        {
            "model": model_key,
            "method": method_name,
            "task_order": task_order_name,
            "config": config,
            "seed": seed,
            "timestamp": datetime.now().isoformat(),
        }
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = (
        output_dir / f"{model_key}_{method_name}_{task_order_name}_{seed}.json"
    )
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nResults saved to {output_file}")
    logger.info(f"Final AA: {results['average_accuracy']:.2f}%")
    logger.info(f"Final BWT: {results['backward_transfer']:.2f}")
    if compute_baselines:
        logger.info(f"Final FWT: {results['forward_transfer']:.2f}")

    return results


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Continual Learning Evaluation Framework"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="llama-3.2-1b",
        choices=list(MODEL_CONFIGS.keys()),
        help="Model to evaluate",
    )
    parser.add_argument(
        "--method",
        type=str,
        required=True,
        choices=list(METHODS.keys()),
        help="Continual learning method",
    )
    parser.add_argument(
        "--task_order",
        type=str,
        default="order_1",
        choices=list(TASK_ORDERS.keys()),
        help="Task order to use",
    )
    parser.add_argument(
        "--output_dir", type=str, default="./cl_results", help="Output directory"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--no_baselines",
        action="store_true",
        help="Skip computing baselines (faster but no FWT)",
    )
    parser.add_argument("--device", type=str, default="auto", help="Device to use")

    # ICA-specific arguments
    parser.add_argument(
        "--ica_template_path", type=str, default=None, help="Path to ICA templates"
    )
    parser.add_argument(
        "--ica_mask_mode",
        type=str,
        default="lesion",
        choices=["lesion", "preserve"],
        help="ICA masking mode",
    )
    parser.add_argument(
        "--ica_components", type=int, default=10, help="Number of ICA components"
    )

    # EWC-specific arguments
    parser.add_argument(
        "--ewc_lambda", type=float, default=0.4, help="EWC regularization weight"
    )

    # LwF-specific arguments
    parser.add_argument(
        "--lwf_alpha", type=float, default=1.0, help="LwF distillation weight"
    )
    parser.add_argument(
        "--lwf_temperature", type=float, default=2.0, help="LwF temperature"
    )

    # O-LoRA-specific arguments
    parser.add_argument(
        "--olora_lambda", type=float, default=0.1, help="O-LoRA orthogonality weight"
    )

    # DOC-specific arguments
    parser.add_argument(
        "--doc_lambda", type=float, default=0.5, help="DOC regularization weight"
    )
    parser.add_argument(
        "--doc_subspace_fraction",
        type=float,
        default=0.1,
        help="Fraction of directions to preserve per task",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    method_kwargs = {}
    if args.method == "ica_networks":
        method_kwargs = {
            "ica_template_path": args.ica_template_path,
            "mask_mode": args.ica_mask_mode,
            "ica_components": args.ica_components,
        }
    elif args.method == "ewc":
        method_kwargs = {
            "ewc_lambda": args.ewc_lambda,
        }
    elif args.method == "lwf":
        method_kwargs = {
            "lwf_alpha": args.lwf_alpha,
            "temperature": args.lwf_temperature,
        }
    elif args.method == "o_lora":
        method_kwargs = {
            "ortho_lambda": args.olora_lambda,
        }
    elif args.method == "doc":
        method_kwargs = {
            "doc_lambda": args.doc_lambda,
            "subspace_fraction": args.doc_subspace_fraction,
        }

    results = run_continual_learning_evaluation(
        model_key=args.model,
        method_name=args.method,
        task_order_name=args.task_order,
        output_dir=Path(args.output_dir),
        seed=args.seed,
        compute_baselines=not args.no_baselines,
        method_kwargs=method_kwargs,
        device=args.device,
    )

    return results


if __name__ == "__main__":
    main()
