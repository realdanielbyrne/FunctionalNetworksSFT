"""
Experiment orchestrator for continual learning evaluation.

Provides autonomous, resumable execution of the full experimental suite:
  Phase 0: ICA template building (skip if exists)
  Phase 1: FWT baseline computation (skip completed rows)
  Phase 2: CL experiments - method x order x seed (skip completed, checkpoint per task)
  Phase 3: lm-eval benchmarks (general capability evaluation)
  Phase 4: Result aggregation and publication tables

Inspired by the N-BEATS-Lightning experiment runner pattern:
CSV-as-source-of-truth resumability with immediate row-by-row appending.
"""

import argparse
import csv
import gc
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from .checkpointing import CLCheckpoint
from .task_data.config import (
    ALL_DATASETS,
    TASK_ORDERS,
    get_dataset_config,
    get_task_order,
)
from .task_data.loaders import CLDatasetLoader
from .evaluation import (
    METHODS,
    MODEL_CONFIGS,
    evaluate_task,
    load_model_and_tokenizer,
    run_single_task_cycle,
    train_on_task,
)
from .methods.lora_baseline import LoRABaseline
from .metrics import ContinualLearningMetrics

logger = logging.getLogger(__name__)

BASE_SEED = 42

# ---------------------------------------------------------------------------
# CSV Schema
# ---------------------------------------------------------------------------

RESULTS_CSV_COLUMNS = [
    "model",
    "method",
    "task_order",
    "seed",
    "run_idx",
    "num_tasks",
    "average_accuracy",
    "backward_transfer",
    "forward_transfer",
    "per_task_accuracies",
    "accuracy_matrix",
    "training_time_seconds",
    "method_params",
    "git_hash",
    "timestamp",
]

BASELINE_CSV_COLUMNS = [
    "model",
    "task_name",
    "seed",
    "accuracy",
    "training_time_seconds",
    "timestamp",
]

# ---------------------------------------------------------------------------
# CSV Utilities (N-BEATS pattern)
# ---------------------------------------------------------------------------


def init_csv(path: Path, columns: List[str]) -> None:
    """Create CSV with header if it doesn't exist."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()


def append_result(path: Path, row_dict: Dict[str, Any], columns: List[str]) -> None:
    """Append a single result row to CSV."""
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writerow(row_dict)


def result_exists(
    path: Path, model: str, method: str, task_order: str, seed: int
) -> bool:
    """Check if a result row already exists in the CSV."""
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
            ):
                return True
    return False


def baseline_exists(path: Path, model: str, task_name: str, seed: int) -> bool:
    """Check if a FWT baseline row already exists in the CSV."""
    if not path.exists():
        return False
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if (
                row["model"] == model
                and row["task_name"] == task_name
                and row["seed"] == str(seed)
            ):
                return True
    return False


def load_baselines_from_csv(
    path: Path, model: str, task_order: List[str], seed: int
) -> Dict[int, float]:
    """Load FWT baseline accuracies from CSV for a specific model/seed."""
    baselines = {}
    if not path.exists():
        return baselines
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["model"] == model and row["seed"] == str(seed):
                task_name = row["task_name"]
                if task_name in task_order:
                    idx = task_order.index(task_name)
                    baselines[idx] = float(row["accuracy"])
    return baselines


# ---------------------------------------------------------------------------
# Device Detection
# ---------------------------------------------------------------------------


def _get_device(device_arg: str) -> str:
    """Resolve device string."""
    if device_arg != "auto":
        return device_arg
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _cleanup_memory() -> None:
    """Free GPU/CPU memory between experiments."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()


# ---------------------------------------------------------------------------
# Method kwargs builder
# ---------------------------------------------------------------------------

DEFAULT_METHOD_KWARGS: Dict[str, Dict[str, Any]] = {
    "lora": {},
    "ewc": {"ewc_lambda": 0.4},
    "lwf": {"lwf_alpha": 1.0, "temperature": 2.0},
    "o_lora": {"ortho_lambda": 0.1},
    "doc": {"doc_lambda": 0.5, "subspace_fraction": 0.1},
    "ica_networks": {"mask_mode": "lesion", "ica_components": 10},
}

# ICA variation defaults
ICA_VARIATION_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "ica_lesion": {"mask_mode": "lesion", "anti_drift": False},
    "ica_preserve": {"mask_mode": "preserve", "anti_drift": False},
    "ica_lesion_antidrift": {"mask_mode": "lesion", "anti_drift": True},
    "ica_preserve_antidrift": {"mask_mode": "preserve", "anti_drift": True},
}


def _build_method_kwargs(
    method_name: str,
    template_path: Optional[Path] = None,
    ica_components: int = 10,
    ica_percentile: float = 98.0,
    ica_mask_mode: str = "lesion",
    method_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build method-specific kwargs, injecting ICA template path."""
    # Check if it's an ICA variation
    if method_name in ICA_VARIATION_DEFAULTS:
        kwargs = dict(ICA_VARIATION_DEFAULTS[method_name])
        kwargs["ica_template_path"] = str(template_path) if template_path else None
        kwargs["ica_components"] = ica_components
        kwargs["ica_percentile"] = ica_percentile
        if method_overrides:
            kwargs.update(method_overrides)
        return kwargs

    kwargs = dict(DEFAULT_METHOD_KWARGS.get(method_name, {}))
    if method_name == "ica_networks":
        kwargs["ica_template_path"] = str(template_path) if template_path else None
        kwargs["ica_components"] = ica_components
        kwargs["ica_percentile"] = ica_percentile
        kwargs["mask_mode"] = ica_mask_mode

    # Apply any per-method overrides from YAML config
    if method_overrides:
        kwargs.update(method_overrides)

    return kwargs


def _resolve_method_class(method_name: str):
    """Resolve a method name to its class, handling ICA variations."""
    if method_name in ICA_VARIATION_DEFAULTS:
        return METHODS["ica_networks"]
    return METHODS.get(method_name)


# ---------------------------------------------------------------------------
# ICA Template Management
# ---------------------------------------------------------------------------


def _resolve_ica_template_path(
    model_key: str,
    output_dir: Path,
    task_orders: List[str],
    ica_components: int = 10,
    ica_percentile: float = 98.0,
    ica_template_datasets: Optional[List[str]] = None,
) -> Optional[Path]:
    """Resolve ICA template path, building if needed.

    Checks standard location for existing templates, triggers building
    if missing, and fails fast if building fails.

    Returns:
        Path to template file, or None if unavailable.
    """
    model_name = MODEL_CONFIGS[model_key]["model_name"]
    sanitized = model_name.replace("/", "_").replace(".", "_")
    template_dir = output_dir / "ica_templates" / sanitized
    template_file = template_dir / "global_templates.json"

    if template_file.exists():
        # Validate template
        try:
            with open(template_file) as f:
                data = json.load(f)
            meta = data.get("metadata", {})
            if meta.get("num_components", 0) < 1:
                logger.warning(f"Template has 0 components: {template_file}")
                return None
            logger.info(f"Using existing ICA templates: {template_file}")
            return template_file
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Invalid template file {template_file}: {e}")
            return None

    # Try building
    logger.info(f"ICA templates not found, attempting to build for {model_key}...")
    return build_ica_templates_if_needed(
        model_key, output_dir, task_orders,
        ica_components, ica_percentile, ica_template_datasets,
    )


def build_ica_templates_if_needed(
    model_key: str,
    output_dir: Path,
    task_orders: List[str],
    ica_components: int = 10,
    ica_percentile: float = 98.0,
    ica_template_datasets: Optional[List[str]] = None,
) -> Optional[Path]:
    """Build ICA templates if they don't already exist.

    Returns:
        Path to template file, or None if building failed.
    """
    model_name = MODEL_CONFIGS[model_key]["model_name"]
    sanitized = model_name.replace("/", "_").replace(".", "_")
    template_dir = output_dir / "ica_templates" / sanitized
    template_file = template_dir / "global_templates.json"

    if template_file.exists():
        logger.info(f"[SKIP] ICA templates already exist: {template_file}")
        return template_file

    logger.info(f"Building ICA templates for {model_name}...")

    # Collect unique dataset sources
    if ica_template_datasets:
        dataset_sources = ica_template_datasets
    else:
        all_tasks = set()
        for order_name in task_orders:
            all_tasks.update(get_task_order(order_name))
        dataset_sources = []
        seen = set()
        for task_name in sorted(all_tasks):
            cfg = get_dataset_config(task_name)
            source = cfg.source
            if cfg.subset:
                source = f"{source}/{cfg.subset}"
            if source not in seen:
                dataset_sources.append(source)
                seen.add(source)

    logger.info(f"Using datasets for templates: {dataset_sources}")

    try:
        from ..build_ica_templates import build_ica_templates

        build_ica_templates(
            dataset_paths=dataset_sources,
            model_name_or_path=model_name,
            samples_per_dataset=100,
            output_path=str(template_dir),
            ica_components=ica_components,
            ica_percentile=ica_percentile,
        )

        if template_file.exists():
            logger.info(f"ICA templates saved to {template_file}")
            return template_file

        # Template builder may use different filename convention
        json_files = list(template_dir.glob("*.json"))
        if json_files:
            logger.info(f"ICA templates saved to {json_files[0]}")
            return json_files[0]

        logger.error("Template building completed but no output file found")
        return None

    except Exception as e:
        logger.error(f"ICA template building failed: {e}", exc_info=True)
        return None
    finally:
        _cleanup_memory()


# ---------------------------------------------------------------------------
# Phase 1: FWT Baselines
# ---------------------------------------------------------------------------


def compute_fwt_baselines(
    model_key: str,
    task_orders: List[str],
    seeds: List[int],
    csv_path: Path,
    device: str = "cuda",
) -> None:
    """Compute single-task baseline accuracies for FWT, with CSV-based skip."""
    init_csv(csv_path, BASELINE_CSV_COLUMNS)

    # Collect unique tasks across all orders
    all_tasks = set()
    for order_name in task_orders:
        all_tasks.update(get_task_order(order_name))

    total = len(all_tasks) * len(seeds)
    done = 0

    for task_name in sorted(all_tasks):
        for seed in seeds:
            done += 1
            if baseline_exists(csv_path, model_key, task_name, seed):
                logger.info(
                    f"  [SKIP] Baseline {task_name}/seed{seed} — already exists "
                    f"({done}/{total})"
                )
                continue

            logger.info(
                f"  Computing baseline {task_name}/seed{seed} ({done}/{total})..."
            )
            torch.manual_seed(seed)
            np.random.seed(seed)
            start_time = time.time()

            try:
                model, tokenizer, config = load_model_and_tokenizer(
                    model_key, device
                )
                dataset_loader = CLDatasetLoader(
                    tokenizer, max_seq_length=512, seed=seed
                )
                baseline_method = LoRABaseline(model, config)

                task_data = dataset_loader.load_dataset(task_name)
                baseline_method.before_task(0, task_name, task_data)
                train_on_task(baseline_method, task_data["train"], config, device)

                accuracy = evaluate_task(
                    model, tokenizer, task_name, task_data["test"], device
                )
                elapsed = time.time() - start_time

                row = {
                    "model": model_key,
                    "task_name": task_name,
                    "seed": seed,
                    "accuracy": f"{accuracy:.4f}",
                    "training_time_seconds": f"{elapsed:.1f}",
                    "timestamp": datetime.now().isoformat(),
                }
                append_result(csv_path, row, BASELINE_CSV_COLUMNS)
                logger.info(
                    f"    {task_name}: {accuracy:.2f}% ({elapsed:.0f}s)"
                )

            except Exception as e:
                logger.error(
                    f"    Baseline {task_name}/seed{seed} failed: {e}",
                    exc_info=True,
                )
            finally:
                # Cleanup
                try:
                    del model, tokenizer, dataset_loader, baseline_method
                except NameError:
                    pass
                _cleanup_memory()


# ---------------------------------------------------------------------------
# Phase 2: CL Experiments
# ---------------------------------------------------------------------------


def run_single_experiment(
    model_key: str,
    method_name: str,
    task_order_name: str,
    seed: int,
    run_idx: int,
    results_csv: Path,
    checkpoint_base: Path,
    baseline_csv: Path,
    template_path: Optional[Path],
    device: str,
    method_kwargs: Dict[str, Any],
    git_hash: str = "unknown",
    save_final_model: bool = False,
    final_models_dir: Optional[Path] = None,
) -> Optional[Dict]:
    """Run one CL experiment with per-task checkpointing and CSV skip logic.

    Returns result dict or None if skipped/failed.
    """
    # CSV-level skip
    if result_exists(results_csv, model_key, method_name, task_order_name, seed):
        logger.info(
            f"  [SKIP] {method_name}/{task_order_name}/seed{seed} — already in CSV"
        )
        return None

    task_order = get_task_order(task_order_name)
    num_tasks = len(task_order)

    ckpt_dir = (
        checkpoint_base / f"{model_key}_{method_name}_{task_order_name}_seed{seed}"
    )
    checkpoint = CLCheckpoint(ckpt_dir)

    if checkpoint.is_run_complete():
        logger.info(
            f"  [SKIP] {method_name}/{task_order_name}/seed{seed} — checkpoint COMPLETE"
        )
        return None

    torch.manual_seed(seed)
    np.random.seed(seed)
    start_time = time.time()

    # Resolve the method class (handles ICA variations)
    method_cls = _resolve_method_class(method_name)
    if method_cls is None:
        logger.error(f"Unknown method: {method_name}")
        return None

    try:
        model, tokenizer, config = load_model_and_tokenizer(model_key, device)
        dataset_loader = CLDatasetLoader(tokenizer, max_seq_length=512, seed=seed)
        cl_method = method_cls(model, config, **method_kwargs)
        metrics = ContinualLearningMetrics(num_tasks=num_tasks, task_names=task_order)

        # Load FWT baselines into metrics
        baselines = load_baselines_from_csv(
            baseline_csv, model_key, task_order, seed
        )
        for t, acc in baselines.items():
            metrics.set_baseline_accuracy(t, acc)

        # Check for partial checkpoint
        last_task = checkpoint.get_last_completed_task()
        start_task = 0
        opt_state, sched_state = None, None

        if last_task >= 0:
            logger.info(
                f"  Resuming from task {last_task + 1}/{num_tasks} "
                f"(checkpoint found)"
            )
            ckpt_data = checkpoint.load_task_checkpoint(last_task)

            # Restore metrics
            metrics = ckpt_data["metrics"]
            # Re-load baselines into restored metrics
            for t, acc in baselines.items():
                metrics.set_baseline_accuracy(t, acc)

            # Restore model adapter weights
            if ckpt_data["adapter_path"]:
                from peft import PeftModel

                model = model.base_model.model  # unwrap PEFT
                model = PeftModel.from_pretrained(
                    model, str(ckpt_data["adapter_path"])
                )
            elif ckpt_data["model_state_path"]:
                state = torch.load(
                    ckpt_data["model_state_path"],
                    map_location=device,
                    weights_only=True,
                )
                model.load_state_dict(state, strict=False)

            # Restore CL method state
            cl_method = method_cls(model, config, **method_kwargs)
            cl_method.load_state_dict(ckpt_data["cl_method_state"])

            # Restore optimizer/scheduler state
            opt_state = ckpt_data.get("optimizer_state")
            sched_state = ckpt_data.get("scheduler_state")

            start_task = last_task + 1

        # Run remaining tasks
        for T in range(start_task, num_tasks):
            logger.info(
                f"  [{method_name}/{task_order_name}/seed{seed}] "
                f"Task {T + 1}/{num_tasks}: {task_order[T]}"
            )
            opt_state, sched_state = run_single_task_cycle(
                T, task_order, cl_method, dataset_loader,
                tokenizer, config, metrics, device,
                optimizer_state=opt_state,
                scheduler_state=sched_state,
            )
            checkpoint.save_task_checkpoint(
                T, model, cl_method, metrics,
                optimizer_state=opt_state,
                scheduler_state=sched_state,
            )

        elapsed = time.time() - start_time

        # Save final model for lm-eval if requested
        if save_final_model and final_models_dir is not None:
            final_dir = (
                final_models_dir
                / f"{model_key}_{method_name}_{task_order_name}_seed{seed}"
            )
            final_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(final_dir)
            tokenizer.save_pretrained(final_dir)
            logger.info(f"  Final model saved to {final_dir}")

        # Build result row
        report = metrics.get_full_report()
        row = {
            "model": model_key,
            "method": method_name,
            "task_order": task_order_name,
            "seed": seed,
            "run_idx": run_idx,
            "num_tasks": num_tasks,
            "average_accuracy": f"{report['average_accuracy']:.4f}",
            "backward_transfer": f"{report['backward_transfer']:.4f}",
            "forward_transfer": f"{report['forward_transfer']:.4f}",
            "per_task_accuracies": json.dumps(report["per_task_final_accuracy"]),
            "accuracy_matrix": json.dumps(report["accuracy_matrix"]),
            "training_time_seconds": f"{elapsed:.1f}",
            "method_params": json.dumps(method_kwargs),
            "git_hash": git_hash,
            "timestamp": datetime.now().isoformat(),
        }
        append_result(results_csv, row, RESULTS_CSV_COLUMNS)
        checkpoint.mark_run_complete()

        # Save full JSON report
        individual_dir = results_csv.parent / "individual_runs"
        individual_dir.mkdir(parents=True, exist_ok=True)
        report.update(
            {
                "model": model_key,
                "method": method_name,
                "task_order": task_order_name,
                "config": config,
                "seed": seed,
                "git_hash": git_hash,
                "timestamp": datetime.now().isoformat(),
            }
        )
        json_file = (
            individual_dir
            / f"{model_key}_{method_name}_{task_order_name}_seed{seed}.json"
        )
        with open(json_file, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(
            f"  DONE {method_name}/{task_order_name}/seed{seed}: "
            f"AA={report['average_accuracy']:.1f}% "
            f"BWT={report['backward_transfer']:.2f} "
            f"({elapsed:.0f}s)"
        )
        return report

    except torch.cuda.OutOfMemoryError:
        logger.error(
            f"  OOM: {method_name}/{task_order_name}/seed{seed}",
            exc_info=True,
        )
        return None
    except Exception as e:
        logger.error(
            f"  FAILED: {method_name}/{task_order_name}/seed{seed}: {e}",
            exc_info=True,
        )
        return None
    finally:
        try:
            del model, tokenizer, cl_method, dataset_loader
        except NameError:
            pass
        _cleanup_memory()


def run_experiment_suite(
    model_key: str,
    methods: List[str],
    orders: List[str],
    num_seeds: int,
    output_dir: Path,
    device: str,
    template_path: Optional[Path],
    baseline_csv: Path,
    ica_components: int = 10,
    ica_percentile: float = 98.0,
    method_overrides: Optional[Dict[str, Dict[str, Any]]] = None,
    dry_run: bool = False,
    git_hash: str = "unknown",
    save_final_models: bool = False,
) -> None:
    """Run the full experiment suite: method x order x seed."""
    results_csv = output_dir / "experiments" / f"{model_key}_results.csv"
    checkpoint_base = output_dir / "checkpoints"
    final_models_dir = output_dir / "final_models" if save_final_models else None
    init_csv(results_csv, RESULTS_CSV_COLUMNS)

    seeds = [BASE_SEED + i for i in range(num_seeds)]
    total = len(methods) * len(orders) * len(seeds)

    logger.info(f"\n{'='*60}")
    logger.info(f"Experiment Suite: {total} runs")
    logger.info(f"  Model:   {model_key}")
    logger.info(f"  Methods: {methods}")
    logger.info(f"  Orders:  {orders}")
    logger.info(f"  Seeds:   {seeds}")
    logger.info(f"  Output:  {results_csv}")
    logger.info(f"{'='*60}\n")

    if dry_run:
        for method in methods:
            for order in orders:
                for seed in seeds:
                    exists = result_exists(
                        results_csv, model_key, method, order, seed
                    )
                    status = "[SKIP]" if exists else "[RUN]"
                    print(f"  {status} {method}/{order}/seed{seed}")
        return

    method_overrides = method_overrides or {}

    run_count = 0
    for method in methods:
        for order in orders:
            for i, seed in enumerate(seeds):
                run_count += 1
                logger.info(
                    f"\n--- Run {run_count}/{total}: "
                    f"{method}/{order}/seed{seed} ---"
                )

                method_kwargs = _build_method_kwargs(
                    method, template_path, ica_components,
                    ica_percentile,
                    method_overrides=method_overrides.get(method),
                )
                run_single_experiment(
                    model_key=model_key,
                    method_name=method,
                    task_order_name=order,
                    seed=seed,
                    run_idx=i,
                    results_csv=results_csv,
                    checkpoint_base=checkpoint_base,
                    baseline_csv=baseline_csv,
                    template_path=template_path,
                    device=device,
                    method_kwargs=method_kwargs,
                    git_hash=git_hash,
                    save_final_model=save_final_models,
                    final_models_dir=final_models_dir,
                )

    logger.info(f"\nAll {total} runs processed. Results: {results_csv}")


# ---------------------------------------------------------------------------
# Main Orchestrator
# ---------------------------------------------------------------------------


def main():
    """CLI entry point for the experiment orchestrator."""
    parser = argparse.ArgumentParser(
        description="Continual Learning Experiment Orchestrator"
    )

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        choices=list(MODEL_CONFIGS.keys()),
        help="Model to evaluate (overrides YAML config)",
    )
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=None,
        help="CL methods to run (default: from config or all)",
    )
    parser.add_argument(
        "--orders",
        type=str,
        nargs="+",
        default=None,
        choices=list(TASK_ORDERS.keys()),
        help="Task orders (default: from config or all 6)",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=None,
        help="Number of seeds (default: from config or 1)",
    )
    parser.add_argument(
        "--phase",
        type=str,
        default="all",
        choices=["all", "templates", "baselines", "experiments", "lm_eval", "aggregate"],
        help="Run specific phase only",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Base output directory (overrides YAML config)",
    )
    parser.add_argument(
        "--device", type=str, default="auto", help="Device (auto/cuda/mps/cpu)"
    )
    parser.add_argument(
        "--skip_long_chains",
        action="store_true",
        help="Skip 15-task long-chain orders (order_4-6)",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print what would be run without executing",
    )

    # ICA-specific
    parser.add_argument(
        "--ica_components", type=int, default=10, help="ICA components"
    )
    parser.add_argument(
        "--ica_percentile", type=float, default=98.0, help="ICA percentile"
    )
    parser.add_argument(
        "--ica_template_datasets",
        type=str,
        nargs="+",
        default=None,
        help="Override datasets for ICA template building",
    )

    # Override training steps (for smoke testing)
    parser.add_argument(
        "--override_steps",
        type=int,
        default=None,
        help="Override num_steps_per_task (for testing)",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # --- Load YAML config if provided ---
    cl_config = None
    method_overrides: Dict[str, Dict[str, Any]] = {}
    save_final_models = False
    lm_eval_benchmarks = ["mmlu"]

    if args.config:
        from .config import (
            CLExperimentConfig,
            config_to_model_config,
            expand_ica_variations,
            load_cl_config,
        )

        cl_config = load_cl_config(args.config)

        # Override MODEL_CONFIGS with YAML values
        MODEL_CONFIGS[cl_config.model_key] = config_to_model_config(cl_config)
        logger.info(f"Updated MODEL_CONFIGS[{cl_config.model_key}] from YAML")

        # Store per-method overrides
        method_overrides = dict(cl_config.methods)
        save_final_models = cl_config.save_models
        lm_eval_benchmarks = cl_config.lm_eval_benchmarks

    # --- Resolve arguments (CLI > YAML > defaults) ---
    model_key = args.model
    if model_key is None:
        model_key = cl_config.model_key if cl_config else "llama-3.2-1b"

    # Ensure model_key is in MODEL_CONFIGS
    if model_key not in MODEL_CONFIGS:
        parser.error(f"Unknown model: {model_key}. Available: {list(MODEL_CONFIGS.keys())}")

    num_seeds = args.seeds
    if num_seeds is None:
        num_seeds = cl_config.num_seeds if cl_config else 1

    output_dir_str = args.output_dir
    if output_dir_str is None:
        output_dir_str = cl_config.results_dir if cl_config else "./experiments/continual_learning/results"
    output_dir = Path(output_dir_str)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = _get_device(args.device)

    # Apply step override for smoke testing
    if args.override_steps is not None:
        for key in MODEL_CONFIGS:
            MODEL_CONFIGS[key]["num_steps_per_task"] = args.override_steps
        logger.info(f"Overriding num_steps_per_task to {args.override_steps}")

    # Build methods list
    methods = args.methods
    if methods is None:
        if cl_config:
            methods = list(cl_config.methods.keys())
            # Expand ICA variations into the methods list
            if cl_config.ica_variations:
                from .config import expand_ica_variations

                ica_vars = expand_ica_variations(cl_config)
                for var_name, var_kwargs in ica_vars:
                    methods.append(var_name)
                    method_overrides[var_name] = var_kwargs
        else:
            methods = list(METHODS.keys())

    # Build orders list
    orders = args.orders
    if orders is None:
        if cl_config:
            orders = cl_config.standard_orders + cl_config.long_chain_orders
        else:
            orders = list(TASK_ORDERS.keys())

    if args.skip_long_chains:
        orders = [o for o in orders if o in ("order_1", "order_2", "order_3")]

    # --- Reproducibility: capture environment ---
    git_hash = "unknown"
    try:
        from .reproducibility import dump_full_config, save_environment

        git_hash = save_environment(output_dir)
        if cl_config:
            dump_full_config(cl_config, output_dir / "config_resolved.json")
    except Exception as e:
        logger.warning(f"Could not capture environment info: {e}")

    logger.info(f"Device: {device}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Model: {model_key}")
    logger.info(f"Methods: {methods}")
    logger.info(f"Orders: {orders}")
    logger.info(f"Seeds: {num_seeds}")
    logger.info(f"Phase: {args.phase}")
    logger.info(f"Git hash: {git_hash}")

    # --- Determine which methods need ICA templates ---
    ica_methods = [m for m in methods if m.startswith("ica_") or m == "ica_networks"]

    # --- Phase 0: ICA Templates ---
    template_path = None
    if args.phase in ("all", "templates"):
        if ica_methods:
            logger.info("\n=== Phase 0: ICA Template Building ===")
            template_path = _resolve_ica_template_path(
                model_key=model_key,
                output_dir=output_dir,
                task_orders=orders,
                ica_components=args.ica_components,
                ica_percentile=args.ica_percentile,
                ica_template_datasets=args.ica_template_datasets,
            )
            if template_path is None and ica_methods:
                raise RuntimeError(
                    f"ICA templates required but could not be built for {model_key}. "
                    f"ICA methods ({ica_methods}) cannot run without templates."
                )
        else:
            logger.info("Skipping ICA templates (no ICA methods in list)")

    if args.phase == "templates":
        return

    # --- Phase 1: FWT Baselines ---
    baseline_csv = output_dir / "baselines" / f"{model_key}_baselines.csv"
    if args.phase in ("all", "baselines"):
        logger.info("\n=== Phase 1: FWT Baselines ===")
        seeds = [BASE_SEED + i for i in range(num_seeds)]
        compute_fwt_baselines(
            model_key=model_key,
            task_orders=orders,
            seeds=seeds,
            csv_path=baseline_csv,
            device=device,
        )

    if args.phase == "baselines":
        return

    # --- Phase 2: CL Experiments ---
    if args.phase in ("all", "experiments"):
        logger.info("\n=== Phase 2: CL Experiments ===")

        # If template wasn't built in this run, try to find existing one
        if template_path is None and ica_methods:
            model_name = MODEL_CONFIGS[model_key]["model_name"]
            sanitized = model_name.replace("/", "_").replace(".", "_")
            candidate = (
                output_dir / "ica_templates" / sanitized / "global_templates.json"
            )
            if candidate.exists():
                template_path = candidate
                logger.info(f"Found existing ICA templates: {template_path}")
            else:
                logger.error(
                    f"ICA templates not found for {model_key}. "
                    f"ICA methods will be skipped."
                )
                methods = [m for m in methods if m not in ica_methods]

        run_experiment_suite(
            model_key=model_key,
            methods=methods,
            orders=orders,
            num_seeds=num_seeds,
            output_dir=output_dir,
            device=device,
            template_path=template_path,
            baseline_csv=baseline_csv,
            ica_components=args.ica_components,
            ica_percentile=args.ica_percentile,
            method_overrides=method_overrides,
            dry_run=args.dry_run,
            git_hash=git_hash,
            save_final_models=save_final_models,
        )

    if args.phase == "experiments":
        return

    # --- Phase 3: lm-eval Benchmarks ---
    if args.phase in ("all", "lm_eval"):
        logger.info("\n=== Phase 3: lm-eval Benchmarks ===")
        try:
            from .lm_eval_runner import (
                run_lm_eval_on_base_model,
                run_lm_eval_on_completed_experiments,
            )

            lm_eval_csv = output_dir / "lm_eval" / f"{model_key}_lm_eval.csv"
            results_csv = output_dir / "experiments" / f"{model_key}_results.csv"
            final_models = output_dir / "final_models"

            # Base model evaluation
            model_name = MODEL_CONFIGS[model_key]["model_name"]
            run_lm_eval_on_base_model(
                model_key, model_name, lm_eval_benchmarks, lm_eval_csv, device
            )

            # Post-CL evaluations
            if save_final_models and final_models.exists():
                run_lm_eval_on_completed_experiments(
                    model_key, results_csv, lm_eval_csv,
                    final_models, lm_eval_benchmarks, device,
                )
            else:
                logger.info(
                    "Skipping post-CL lm-eval (save_models not enabled or no final models)"
                )

        except ImportError as e:
            logger.warning(f"lm-eval not available: {e}")
        except Exception as e:
            logger.error(f"lm-eval phase failed: {e}", exc_info=True)

    if args.phase == "lm_eval":
        return

    # --- Phase 4: Aggregation ---
    if args.phase in ("all", "aggregate"):
        logger.info("\n=== Phase 4: Aggregation & Tables ===")
        try:
            from .aggregation import generate_all_tables

            results_csv = output_dir / "experiments" / f"{model_key}_results.csv"
            lm_eval_csv = output_dir / "lm_eval" / f"{model_key}_lm_eval.csv"
            tables_dir = output_dir / "tables"
            generate_all_tables(
                results_csv=results_csv,
                baseline_csv=baseline_csv,
                model=model_key,
                output_dir=tables_dir,
                lm_eval_csv=lm_eval_csv if lm_eval_csv.exists() else None,
            )
        except ImportError:
            logger.warning(
                "Aggregation module not available. "
                "Run fnsft-cl-aggregate separately."
            )
        except Exception as e:
            logger.error(f"Aggregation failed: {e}", exc_info=True)

    logger.info("\n=== Orchestrator Complete ===")


if __name__ == "__main__":
    main()
