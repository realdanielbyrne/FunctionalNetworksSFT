"""
Per-task checkpoint management for continual learning experiment runs.

Enables resuming CL sequences from the last completed task rather than
restarting from scratch. Checkpoints include model adapter weights,
CL method state, accuracy metrics, and optimizer state.
"""

import json
import logging
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from peft import PeftModel

from .methods.base import ContinualLearningMethod
from .metrics import ContinualLearningMetrics

logger = logging.getLogger(__name__)


class CLCheckpoint:
    """Manages per-task checkpoints within a CL experiment run."""

    COMPLETE_MARKER = "COMPLETE"

    def __init__(self, checkpoint_dir: Path):
        self.checkpoint_dir = Path(checkpoint_dir)

    def _task_dir(self, task_idx: int) -> Path:
        return self.checkpoint_dir / f"task_{task_idx}"

    def save_task_checkpoint(
        self,
        task_idx: int,
        model: torch.nn.Module,
        cl_method: ContinualLearningMethod,
        metrics: ContinualLearningMetrics,
        optimizer_state: Optional[Dict] = None,
        scheduler_state: Optional[Dict] = None,
    ) -> None:
        """Save checkpoint after completing a task.

        Args:
            task_idx: Index of the task just completed.
            model: The model (with LoRA adapters).
            cl_method: The CL method instance with accumulated state.
            metrics: Accuracy matrix recorded so far.
            optimizer_state: Optimizer state dict (recommended for resume).
            scheduler_state: Scheduler state dict (recommended for resume).
        """
        task_dir = self._task_dir(task_idx)
        task_dir.mkdir(parents=True, exist_ok=True)

        # Save LoRA adapter weights
        if isinstance(model, PeftModel):
            model.save_pretrained(task_dir / "adapter")
        else:
            torch.save(
                {k: v.cpu() for k, v in model.state_dict().items() if v.requires_grad},
                task_dir / "model_trainable.pt",
            )

        # Save CL method state
        cl_state = cl_method.get_state_dict()
        # Separate tensor state from JSON-serializable state
        tensor_state = {}
        json_state = {}
        for k, v in cl_state.items():
            if isinstance(v, torch.Tensor):
                tensor_state[k] = v.cpu()
            elif isinstance(v, dict) and any(
                isinstance(vv, torch.Tensor) for vv in v.values()
            ):
                tensor_state[k] = {
                    kk: vv.cpu() if isinstance(vv, torch.Tensor) else vv
                    for kk, vv in v.items()
                }
            else:
                json_state[k] = v

        with open(task_dir / "cl_method_state.json", "w") as f:
            json.dump(json_state, f, indent=2)

        if tensor_state:
            torch.save(tensor_state, task_dir / "cl_method_tensors.pt")

        # Save metrics
        metrics.save(task_dir / "metrics.json")

        # Save optimizer and scheduler state
        if optimizer_state is not None:
            torch.save(optimizer_state, task_dir / "optimizer.pt")
        if scheduler_state is not None:
            torch.save(scheduler_state, task_dir / "scheduler.pt")

        if optimizer_state is None:
            logger.warning(
                f"Checkpoint for task {task_idx} saved WITHOUT optimizer state. "
                "Resume will start with a fresh optimizer."
            )

        # Write completion marker for this task
        (task_dir / "DONE").touch()

        logger.info(f"Checkpoint saved for task {task_idx} at {task_dir}")

    def get_last_completed_task(self) -> int:
        """Return index of last completed task, or -1 if none."""
        if not self.checkpoint_dir.exists():
            return -1

        last = -1
        for task_dir in sorted(self.checkpoint_dir.glob("task_*")):
            if (task_dir / "DONE").exists():
                try:
                    idx = int(task_dir.name.split("_")[1])
                    last = max(last, idx)
                except (ValueError, IndexError):
                    continue
        return last

    def load_task_checkpoint(
        self, task_idx: int
    ) -> Dict[str, Any]:
        """Load checkpoint from after task_idx was completed.

        Returns:
            Dictionary with keys:
            - 'adapter_path': Path to saved adapter weights (or None)
            - 'model_state_path': Path to trainable weights (or None)
            - 'cl_method_state': Dict of CL method state
            - 'metrics': ContinualLearningMetrics instance
            - 'optimizer_state': Optimizer state dict (or None)
            - 'scheduler_state': Scheduler state dict (or None)
        """
        task_dir = self._task_dir(task_idx)
        if not (task_dir / "DONE").exists():
            raise FileNotFoundError(
                f"No completed checkpoint at task {task_idx}"
            )

        result: Dict[str, Any] = {}

        # Load adapter/model path
        adapter_path = task_dir / "adapter"
        if adapter_path.exists():
            result["adapter_path"] = adapter_path
            result["model_state_path"] = None
        else:
            result["adapter_path"] = None
            model_path = task_dir / "model_trainable.pt"
            result["model_state_path"] = model_path if model_path.exists() else None

        # Load CL method state
        cl_state = {}
        json_path = task_dir / "cl_method_state.json"
        if json_path.exists():
            with open(json_path) as f:
                cl_state = json.load(f)

        tensor_path = task_dir / "cl_method_tensors.pt"
        if tensor_path.exists():
            tensor_state = torch.load(tensor_path, map_location="cpu", weights_only=False)
            cl_state.update(tensor_state)

        result["cl_method_state"] = cl_state

        # Load metrics
        metrics_path = task_dir / "metrics.json"
        result["metrics"] = ContinualLearningMetrics.load(metrics_path)

        # Load optimizer state
        opt_path = task_dir / "optimizer.pt"
        result["optimizer_state"] = (
            torch.load(opt_path, map_location="cpu", weights_only=False)
            if opt_path.exists()
            else None
        )

        # Load scheduler state
        sched_path = task_dir / "scheduler.pt"
        result["scheduler_state"] = (
            torch.load(sched_path, map_location="cpu", weights_only=False)
            if sched_path.exists()
            else None
        )

        logger.info(f"Loaded checkpoint from task {task_idx}")
        return result

    def mark_run_complete(self) -> None:
        """Mark the full run as complete (all tasks done)."""
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        (self.checkpoint_dir / self.COMPLETE_MARKER).touch()

    def is_run_complete(self) -> bool:
        """Check if the run was already fully completed."""
        return (self.checkpoint_dir / self.COMPLETE_MARKER).exists()

    def cleanup(self) -> None:
        """Remove checkpoint files after successful CSV append."""
        if self.checkpoint_dir.exists():
            shutil.rmtree(self.checkpoint_dir)
            logger.info(f"Cleaned up checkpoints at {self.checkpoint_dir}")
