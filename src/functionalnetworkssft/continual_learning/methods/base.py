"""
Base class for continual learning methods.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict

import torch
from torch import nn
from transformers import PreTrainedModel


class ContinualLearningMethod(ABC):
    """
    Abstract base class for continual learning methods.

    All methods should implement:
    - before_task(): Called before training on a new task
    - compute_loss(): Compute the training loss (may include regularization)
    - after_task(): Called after training on a task
    """

    def __init__(self, model: PreTrainedModel, config: Dict[str, Any]):
        """
        Initialize the CL method.

        Args:
            model: The base model (may have LoRA adapters)
            config: Training configuration
        """
        self.model = model
        self.config = config
        self.current_task_idx = 0
        self.task_history = []

    def before_task(
        self, task_idx: int, task_name: str, task_data: Any
    ) -> None:
        """
        Called before training on a new task.

        Override to perform task-specific setup like:
        - Initializing task-specific parameters
        - Recording initial model state
        - Computing importance weights

        Args:
            task_idx: Index of the current task (0-indexed)
            task_name: Name of the current task
            task_data: Training data for the task
        """
        self.current_task_idx = task_idx

    @abstractmethod
    def compute_loss(
        self, batch: Dict[str, torch.Tensor], task_idx: int
    ) -> torch.Tensor:
        """
        Compute the training loss for a batch.

        Override to add method-specific regularization terms.

        Args:
            batch: Batch of training data
            task_idx: Index of the current task

        Returns:
            Total loss (task loss + any regularization)
        """
        pass

    def after_task(
        self, task_idx: int, task_name: str, task_data: Any
    ) -> None:
        """
        Called after training on a task.

        Override to:
        - Store important directions/weights
        - Update regularization parameters
        - Save task-specific state

        Args:
            task_idx: Index of the completed task
            task_name: Name of the completed task
            task_data: Training data for the task
        """
        self.task_history.append({"task_idx": task_idx, "task_name": task_name})

    def get_model_for_inference(self) -> nn.Module:
        """Return the model ready for inference."""
        return self.model

    def get_trainable_parameters(self) -> list:
        """Get list of trainable parameters."""
        return [p for p in self.model.parameters() if p.requires_grad]

    def get_state_dict(self) -> Dict[str, Any]:
        """Return serializable state for checkpointing.

        Subclasses should override to include method-specific state
        (Fisher matrices, subspace bases, etc.) and call super().
        """
        return {
            "task_history": self.task_history,
            "current_task_idx": self.current_task_idx,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore state from checkpoint.

        Subclasses should override to restore method-specific state
        and call super().
        """
        self.task_history = state.get("task_history", [])
        self.current_task_idx = state.get("current_task_idx", 0)

    def save_state(self, path: str) -> None:
        """Save method-specific state."""
        pass

    def load_state(self, path: str) -> None:
        """Load method-specific state."""
        pass

