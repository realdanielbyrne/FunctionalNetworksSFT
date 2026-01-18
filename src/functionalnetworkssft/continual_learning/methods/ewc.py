"""
Elastic Weight Consolidation (EWC) implementation.
Kirkpatrick et al., 2017
"""

import logging
from typing import Any, Dict

import torch
from torch import nn

from .base import ContinualLearningMethod

logger = logging.getLogger(__name__)


class EWC(ContinualLearningMethod):
    """
    Elastic Weight Consolidation for continual learning.

    Adds a regularization term based on Fisher information matrix
    to prevent important weights from changing too much.

    Loss = L_task + lambda * sum_i F_i * (theta_i - theta*_i)^2

    where F_i is the Fisher information for parameter i,
    theta*_i is the optimal parameter after previous task.
    """

    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        ewc_lambda: float = 0.4,
        fisher_sample_size: int = 200,
    ):
        super().__init__(model, config)
        self.ewc_lambda = ewc_lambda
        self.fisher_sample_size = fisher_sample_size
        self.fisher_matrices: Dict[int, Dict[str, torch.Tensor]] = {}
        self.optimal_params: Dict[int, Dict[str, torch.Tensor]] = {}

    def compute_loss(
        self, batch: Dict[str, torch.Tensor], task_idx: int
    ) -> torch.Tensor:
        """Compute task loss plus EWC regularization."""
        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        task_loss = outputs.loss

        ewc_loss = torch.tensor(0.0, device=task_loss.device)

        for prev_task_idx in range(task_idx):
            if prev_task_idx not in self.fisher_matrices:
                continue

            fisher = self.fisher_matrices[prev_task_idx]
            optimal = self.optimal_params[prev_task_idx]

            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue
                if name not in fisher:
                    continue
                penalty = fisher[name] * (param - optimal[name]).pow(2)
                ewc_loss = ewc_loss + penalty.sum()

        total_loss = task_loss + self.ewc_lambda * ewc_loss
        return total_loss

    def after_task(
        self, task_idx: int, task_name: str, task_data: Any
    ) -> None:
        """Compute and store Fisher information after task completion."""
        super().after_task(task_idx, task_name, task_data)

        logger.info(f"Computing Fisher information for task {task_idx}")

        self.optimal_params[task_idx] = {
            name: param.clone().detach()
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }

        fisher = self._compute_fisher(task_data)
        self.fisher_matrices[task_idx] = fisher

    def _compute_fisher(self, task_data: Any) -> Dict[str, torch.Tensor]:
        """
        Compute diagonal Fisher information matrix.

        Fisher = E[grad log p(y|x,theta) * grad log p(y|x,theta)^T]
        """
        fisher = {
            name: torch.zeros_like(param)
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }

        self.model.eval()
        device = next(self.model.parameters()).device

        samples = task_data["train"].shuffle().select(
            range(min(self.fisher_sample_size, len(task_data["train"])))
        )

        for example in samples:
            self.model.zero_grad()

            inputs = {
                k: torch.tensor([v]).to(device)
                for k, v in example.items()
                if k in ["input_ids", "attention_mask", "labels"]
            }

            outputs = self.model(**inputs)
            loss = outputs.loss
            loss.backward()

            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher[name] += param.grad.pow(2)

        for name in fisher:
            fisher[name] /= len(samples)

        self.model.train()
        return fisher

