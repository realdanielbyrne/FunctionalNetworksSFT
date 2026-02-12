"""
Orthogonal LoRA (O-LoRA) implementation for continual learning.
Wang et al., 2023 - "Orthogonal Subspace Learning for Language Model Continual Learning"

O-LoRA constrains new task learning to be orthogonal to previous tasks'
subspaces to minimize interference and forgetting.
"""

import logging
from typing import Any, Dict, List, Optional

import torch
from torch import nn

from .base import ContinualLearningMethod

logger = logging.getLogger(__name__)


class OLoRA(ContinualLearningMethod):
    """
    Orthogonal LoRA for continual learning.

    Maintains orthogonal subspaces for different tasks by:
    1. Storing the LoRA directions used for previous tasks
    2. Projecting gradients to be orthogonal to previous directions
    3. Adding regularization to enforce orthogonality

    Loss = L_task + lambda * L_ortho

    where L_ortho penalizes overlap with previous task subspaces.
    """

    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        ortho_lambda: float = 0.1,
        subspace_dim: Optional[int] = None,
    ):
        """
        Initialize O-LoRA method.

        Args:
            model: The base model with LoRA adapters
            config: Training configuration
            ortho_lambda: Weight for orthogonality regularization loss
            subspace_dim: Dimension of subspace to preserve per task (default: LoRA rank)
        """
        super().__init__(model, config)
        self.ortho_lambda = ortho_lambda
        self.subspace_dim = subspace_dim or config.get("lora_r", 16)

        # Store previous task subspaces (as basis vectors)
        # Key: parameter name, Value: list of basis vectors (one per previous task)
        self.task_subspaces: Dict[str, List[torch.Tensor]] = {}

        # Store LoRA parameter names for easy access
        self.lora_param_names: List[str] = []
        self._identify_lora_params()

    def _identify_lora_params(self) -> None:
        """Identify LoRA A and B matrix parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and ("lora_A" in name or "lora_B" in name):
                self.lora_param_names.append(name)
                self.task_subspaces[name] = []

        logger.info(f"Identified {len(self.lora_param_names)} LoRA parameters")

    def _compute_subspace_basis(self, param_name: str) -> torch.Tensor:
        """
        Compute the principal subspace basis for a LoRA parameter.

        Uses SVD to find the principal directions of the current parameter.
        """
        # Get the parameter
        param = dict(self.model.named_parameters())[param_name]

        # Flatten to 2D if needed
        if param.dim() > 2:
            param_flat = param.view(param.size(0), -1)
        else:
            param_flat = param

        # Compute SVD to get principal directions
        try:
            U, S, Vh = torch.linalg.svd(param_flat.float(), full_matrices=False)
            # Take top-k directions based on subspace_dim
            k = min(self.subspace_dim, U.size(1))
            basis = U[:, :k].detach()
        except Exception as e:
            logger.warning(f"SVD failed for {param_name}: {e}")
            basis = param_flat[:, : self.subspace_dim].detach()

        return basis

    def compute_loss(
        self, batch: Dict[str, torch.Tensor], task_idx: int
    ) -> torch.Tensor:
        """Compute task loss plus orthogonality regularization."""
        # Standard task loss
        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        task_loss = outputs.loss

        # Add orthogonality loss for tasks after the first
        if task_idx > 0 and self.task_subspaces:
            ortho_loss = self._compute_orthogonality_loss()
            total_loss = task_loss + self.ortho_lambda * ortho_loss
        else:
            total_loss = task_loss

        return total_loss

    def _compute_orthogonality_loss(self) -> torch.Tensor:
        """
        Compute orthogonality regularization loss.

        Penalizes current parameters for having components in the
        subspaces of previous tasks.
        """
        device = next(self.model.parameters()).device
        ortho_loss = torch.tensor(0.0, device=device)
        num_terms = 0

        for name, param in self.model.named_parameters():
            if name not in self.task_subspaces:
                continue
            if not self.task_subspaces[name]:
                continue

            # Flatten parameter
            if param.dim() > 2:
                param_flat = param.view(param.size(0), -1)
            else:
                param_flat = param

            # Compute overlap with each previous task's subspace
            for prev_basis in self.task_subspaces[name]:
                prev_basis = prev_basis.to(device)

                # Ensure dimensions match
                if prev_basis.size(0) != param_flat.size(0):
                    continue

                # Project current param onto previous subspace
                # projection = basis @ basis.T @ param
                projection = prev_basis @ (prev_basis.T @ param_flat)

                # Penalize the magnitude of the projection
                ortho_loss = ortho_loss + torch.norm(projection) ** 2
                num_terms += 1

        if num_terms > 0:
            ortho_loss = ortho_loss / num_terms

        return ortho_loss

    def after_task(
        self, task_idx: int, task_name: str, task_data: Any
    ) -> None:
        """Store the subspace used for this task."""
        super().after_task(task_idx, task_name, task_data)

        logger.info(f"Extracting subspace for task {task_idx}")

        # Extract and store subspace basis for each LoRA parameter
        for name in self.lora_param_names:
            if name in dict(self.model.named_parameters()):
                basis = self._compute_subspace_basis(name)
                self.task_subspaces[name].append(basis.cpu())

        logger.info(
            f"Stored subspaces for {len(self.lora_param_names)} parameters"
        )

    def get_model_for_inference(self) -> nn.Module:
        """Return the model for inference."""
        return self.model

    def get_state_dict(self) -> Dict[str, Any]:
        """Return O-LoRA state: subspace bases per parameter per task."""
        state = super().get_state_dict()
        state["ortho_lambda"] = self.ortho_lambda
        state["subspace_dim"] = self.subspace_dim
        state["lora_param_names"] = self.lora_param_names
        state["task_subspaces"] = {
            name: [b.cpu() for b in bases]
            for name, bases in self.task_subspaces.items()
        }
        return state

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore O-LoRA state from checkpoint."""
        super().load_state_dict(state)
        self.ortho_lambda = state.get("ortho_lambda", self.ortho_lambda)
        self.subspace_dim = state.get("subspace_dim", self.subspace_dim)
        if "task_subspaces" in state:
            self.task_subspaces = state["task_subspaces"]

    def save_state(self, path: str) -> None:
        """Save O-LoRA state including subspaces."""
        import json
        from pathlib import Path

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save subspace info (just metadata, not the actual tensors for now)
        state = {
            "ortho_lambda": self.ortho_lambda,
            "subspace_dim": self.subspace_dim,
            "num_tasks": len(self.task_history),
            "lora_param_names": self.lora_param_names,
        }

        with open(path / "o_lora_state.json", "w") as f:
            json.dump(state, f, indent=2)

        # Save subspace tensors
        torch.save(self.task_subspaces, path / "task_subspaces.pt")
