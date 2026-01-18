"""
Learning without Forgetting (LwF) implementation.
Li & Hoiem, 2017 - "Learning without Forgetting"

LwF uses knowledge distillation to preserve knowledge from previous tasks.
The old model's outputs on new task data serve as soft targets.
"""

import logging
from copy import deepcopy
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn

from .base import ContinualLearningMethod

logger = logging.getLogger(__name__)


class LwF(ContinualLearningMethod):
    """
    Learning without Forgetting for continual learning.

    Uses knowledge distillation from the previous task's model to prevent
    forgetting. The loss combines:
    - Standard cross-entropy on current task
    - KL divergence between current and previous model outputs (distillation)

    Loss = L_task + alpha * L_distill

    where L_distill = KL(softmax(z_old/T) || softmax(z_new/T))
    and T is the temperature for softening distributions.
    """

    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        lwf_alpha: float = 1.0,
        temperature: float = 2.0,
    ):
        """
        Initialize LwF method.

        Args:
            model: The base model
            config: Training configuration
            lwf_alpha: Weight for distillation loss
            temperature: Temperature for softening probability distributions
        """
        super().__init__(model, config)
        self.lwf_alpha = lwf_alpha
        self.temperature = temperature
        self.old_model: Optional[nn.Module] = None

    def before_task(
        self, task_idx: int, task_name: str, task_data: Any
    ) -> None:
        """Store a copy of the model before training on new task."""
        super().before_task(task_idx, task_name, task_data)

        if task_idx > 0 and self.old_model is None:
            # This shouldn't happen if after_task was called properly
            logger.warning("Old model not found, creating copy now")
            self._store_old_model()

    def compute_loss(
        self, batch: Dict[str, torch.Tensor], task_idx: int
    ) -> torch.Tensor:
        """
        Compute task loss plus knowledge distillation loss.

        For the first task, only task loss is used.
        For subsequent tasks, distillation loss is added.
        """
        # Current model forward pass
        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        task_loss = outputs.loss
        current_logits = outputs.logits

        # Add distillation loss for tasks after the first
        if task_idx > 0 and self.old_model is not None:
            distill_loss = self._compute_distillation_loss(
                batch, current_logits
            )
            total_loss = task_loss + self.lwf_alpha * distill_loss
        else:
            total_loss = task_loss

        return total_loss

    def _compute_distillation_loss(
        self,
        batch: Dict[str, torch.Tensor],
        current_logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute knowledge distillation loss.

        Uses KL divergence between soft targets from old model and
        current model's predictions.
        """
        # Get old model's predictions (no gradient needed)
        with torch.no_grad():
            old_outputs = self.old_model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
            old_logits = old_outputs.logits

        # Compute soft targets with temperature scaling
        # KL(P_old || P_new) where P = softmax(logits/T)
        old_probs = F.softmax(old_logits / self.temperature, dim=-1)
        new_log_probs = F.log_softmax(current_logits / self.temperature, dim=-1)

        # KL divergence loss (reduction='batchmean' for proper scaling)
        # Multiply by T^2 as per Hinton et al. (2015) to match gradient magnitudes
        distill_loss = F.kl_div(
            new_log_probs,
            old_probs,
            reduction="batchmean",
        ) * (self.temperature ** 2)

        return distill_loss

    def _store_old_model(self) -> None:
        """Store a frozen copy of the current model."""
        logger.info("Storing old model for knowledge distillation")
        self.old_model = deepcopy(self.model)
        self.old_model.eval()
        for param in self.old_model.parameters():
            param.requires_grad = False

    def after_task(
        self, task_idx: int, task_name: str, task_data: Any
    ) -> None:
        """Store model copy after task completion for next task's distillation."""
        super().after_task(task_idx, task_name, task_data)

        # Store current model as old model for next task
        self._store_old_model()
        logger.info(f"Stored model after task {task_idx} for LwF distillation")

    def get_model_for_inference(self) -> nn.Module:
        """Return the current model for inference."""
        return self.model
