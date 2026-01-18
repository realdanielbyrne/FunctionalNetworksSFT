"""
Main evaluation loop for continual learning experiments.
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch.optim import AdamW
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from .datasets.config import TASK_ORDERS, get_dataset_config, get_task_order
from .datasets.loaders import CLDatasetLoader
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
        "num_steps_per_task": 500,
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


def load_model_and_tokenizer(
    model_key: str, device: str = "auto", use_auth_token: bool = True
):
    """Load model and tokenizer with LoRA configuration."""
    config = MODEL_CONFIGS[model_key]
    logger.info(f"Loading model: {config['model_name']}")

    tokenizer = AutoTokenizer.from_pretrained(
        config["model_name"], token=use_auth_token
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config["model_name"],
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map=device,
        token=use_auth_token,
    )

    lora_config = LoraConfig(
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        target_modules=config["lora_target_modules"],
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer, config


def evaluate_task(
    model: torch.nn.Module,
    tokenizer,
    task_name: str,
    test_data,
    device: str = "cuda",
) -> float:
    """Evaluate model accuracy on a task using generation."""
    model.eval()
    correct = 0
    total = 0

    config = get_dataset_config(task_name)
    valid_answers = list(config.label_map.values())

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

            generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
            predicted = generated.split()[-1].strip().lower()

            labels = example["labels"]
            ground_truth = None
            for ans in valid_answers:
                if ans.lower() in tokenizer.decode(labels).lower():
                    ground_truth = ans.lower()
                    break

            if ground_truth and predicted.startswith(ground_truth[:3]):
                correct += 1
            total += 1

    accuracy = (correct / total) * 100 if total > 0 else 0.0
    return accuracy


def train_on_task(
    cl_method: ContinualLearningMethod,
    train_data,
    config: Dict,
    device: str = "cuda",
) -> None:
    """Train model on a single task using the CL method."""
    model = cl_method.model
    model.train()

    optimizer = AdamW(cl_method.get_trainable_parameters(), lr=config["learning_rate"])
    num_steps = config["num_steps_per_task"]
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_steps // 10, num_training_steps=num_steps
    )

    step = 0
    pbar = tqdm(total=num_steps, desc="Training")

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
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            step += 1
            pbar.update(1)
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    pbar.close()


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

    for T, task_name in enumerate(task_order):
        logger.info(f"\nTask {T + 1}/{num_tasks}: {task_name}")

        task_data = dataset_loader.load_dataset(task_name)
        cl_method.before_task(T, task_name, task_data)
        train_on_task(cl_method, task_data["train"], config, device)
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
