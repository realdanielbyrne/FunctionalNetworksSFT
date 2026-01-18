"""
Dynamic Orthogonal Continual Fine-Tuning (DOC) implementation.
Zhang et al., 2025 - "DOC: Towards Dynamic Orthogonal Continual Fine-Tuning"

DOC tracks functional directions important for previous tasks and projects
gradients to be orthogonal to these directions, minimizing forgetting while
enabling positive transfer through shared non-critical directions.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn

from .base import ContinualLearningMethod

logger = logging.getLogger(__name__)


class DOC(ContinualLearningMethod):
    """
    Dynamic Orthogonal Continual Fine-Tuning.

    DOC improves upon O-LoRA by:
    1. Tracking gradient directions rather than parameter values
    2. Using dynamic updates to the orthogonal projection matrix
    3. Allowing beneficial transfer through non-critical shared directions

    The method maintains a projection matrix P that projects gradients
    to be orthogonal to important directions from previous tasks:

    g' = (I - P @ P.T) @ g

    where P contains the important directions accumulated over tasks.
    """

    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        doc_lambda: float = 0.5,
        gradient_accumulation_steps: int = 100,
        subspace_fraction: float = 0.1,
        use_gradient_projection: bool = True,
    ):
        """
        Initialize DOC method.

        Args:
            model: The base model with LoRA adapters
            config: Training configuration
            doc_lambda: Weight for orthogonality regularization
            gradient_accumulation_steps: Steps to accumulate gradients for direction estimation
            subspace_fraction: Fraction of directions to preserve per task
            use_gradient_projection: Whether to project gradients (True) or use regularization (False)
        """
        super().__init__(model, config)
        self.doc_lambda = doc_lambda
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.subspace_fraction = subspace_fraction
        self.use_gradient_projection = use_gradient_projection

        # Storage for important directions per parameter
        # Key: param name, Value: orthonormal basis matrix [param_dim, num_directions]
        self.important_directions: Dict[str, torch.Tensor] = {}

        # Gradient accumulator for estimating important directions
        self.gradient_accumulator: Dict[str, torch.Tensor] = {}
        self.gradient_count = 0

        # Track which parameters to apply DOC to
        self.target_param_names: List[str] = []
        self._identify_target_params()

    def _identify_target_params(self) -> None:
        """Identify parameters to apply DOC to (LoRA parameters)."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # Focus on LoRA parameters
                if "lora_A" in name or "lora_B" in name:
                    self.target_param_names.append(name)
                    self.gradient_accumulator[name] = None

        logger.info(f"DOC targeting {len(self.target_param_names)} parameters")

    def _flatten_param(self, param: torch.Tensor) -> torch.Tensor:
        """Flatten parameter to 1D."""
        return param.view(-1)

    def _accumulate_gradients(self) -> None:
        """Accumulate gradients for direction estimation."""
        for name, param in self.model.named_parameters():
            if name not in self.target_param_names:
                continue
            if param.grad is None:
                continue

            grad_flat = self._flatten_param(param.grad.detach())

            if self.gradient_accumulator[name] is None:
                # Initialize accumulator as list of gradients
                self.gradient_accumulator[name] = [grad_flat.cpu()]
            else:
                self.gradient_accumulator[name].append(grad_flat.cpu())

        self.gradient_count += 1

    def _compute_important_directions(self) -> None:
        """
        Compute important directions from accumulated gradients using SVD.

        The gradient covariance matrix G @ G.T reveals the principal directions
        of parameter updates during training. We preserve the top-k directions.
        """
        logger.info("Computing important directions from gradients")

        for name in self.target_param_names:
            if self.gradient_accumulator[name] is None:
                continue

            # Stack gradients: [num_samples, param_dim]
            grads = torch.stack(self.gradient_accumulator[name], dim=0)

            # Compute SVD of gradient matrix
            # U contains left singular vectors (directions in parameter space)
            try:
                U, S, Vh = torch.linalg.svd(grads.T.float(), full_matrices=False)

                # Determine number of directions to keep
                total_dims = U.size(1)
                num_keep = max(1, int(total_dims * self.subspace_fraction))

                # Take top-k directions (highest singular values)
                new_directions = U[:, :num_keep].detach()

                # Merge with existing directions if any
                if name in self.important_directions:
                    existing = self.important_directions[name]
                    combined = torch.cat([existing, new_directions], dim=1)

                    # Re-orthogonalize using SVD
                    U_combined, _, _ = torch.linalg.svd(
                        combined.float(), full_matrices=False
                    )

                    # Keep reasonable number of directions
                    max_directions = min(
                        U_combined.size(1),
                        int(U_combined.size(0) * self.subspace_fraction * 2),
                    )
                    self.important_directions[name] = U_combined[:, :max_directions]
                else:
                    self.important_directions[name] = new_directions

                logger.debug(
                    f"{name}: kept {self.important_directions[name].size(1)} directions"
                )

            except Exception as e:
                logger.warning(f"SVD failed for {name}: {e}")

        # Clear accumulator
        for name in self.target_param_names:
            self.gradient_accumulator[name] = None
        self.gradient_count = 0

    def _project_gradient(
        self, grad: torch.Tensor, name: str
    ) -> torch.Tensor:
        """
        Project gradient to be orthogonal to important directions.

        g' = g - P @ P.T @ g

        where P contains the important directions as columns.
        """
        if name not in self.important_directions:
            return grad

        P = self.important_directions[name].to(grad.device)
        grad_flat = self._flatten_param(grad)

        # Project out components along important directions
        # projection = P @ (P.T @ grad_flat)
        coefficients = P.T @ grad_flat  # [num_directions]
        projection = P @ coefficients  # [param_dim]

        # Orthogonal component
        grad_orthogonal = grad_flat - projection

        return grad_orthogonal.view_as(grad)

    def before_task(
        self, task_idx: int, task_name: str, task_data: Any
    ) -> None:
        """Setup for new task."""
        super().before_task(task_idx, task_name, task_data)

        # Clear gradient accumulator
        for name in self.target_param_names:
            self.gradient_accumulator[name] = None
        self.gradient_count = 0

        if task_idx > 0:
            logger.info(
                f"Task {task_idx}: Using {len(self.important_directions)} "
                f"direction sets from previous tasks"
            )

    def compute_loss(
        self, batch: Dict[str, torch.Tensor], task_idx: int
    ) -> torch.Tensor:
        """
        Compute loss with gradient projection or regularization.

        For gradient projection mode, we modify gradients after backward.
        For regularization mode, we add a penalty term.
        """
        # Standard forward pass
        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        task_loss = outputs.loss

        # Add regularization loss if not using gradient projection
        if not self.use_gradient_projection and task_idx > 0:
            reg_loss = self._compute_regularization_loss()
            total_loss = task_loss + self.doc_lambda * reg_loss
        else:
            total_loss = task_loss

        return total_loss

    def _compute_regularization_loss(self) -> torch.Tensor:
        """
        Compute regularization loss penalizing overlap with important directions.
        """
        device = next(self.model.parameters()).device
        reg_loss = torch.tensor(0.0, device=device)
        num_terms = 0

        for name, param in self.model.named_parameters():
            if name not in self.important_directions:
                continue

            P = self.important_directions[name].to(device)
            param_flat = self._flatten_param(param)

            # Compute projection magnitude
            coefficients = P.T @ param_flat
            reg_loss = reg_loss + torch.sum(coefficients ** 2)
            num_terms += 1

        if num_terms > 0:
            reg_loss = reg_loss / num_terms

        return reg_loss

    def apply_gradient_projection(self) -> None:
        """
        Apply gradient projection after backward pass.

        Call this after loss.backward() but before optimizer.step().
        """
        if not self.use_gradient_projection:
            return

        # Accumulate gradients for direction estimation
        self._accumulate_gradients()

        # Project gradients if we have important directions
        if self.current_task_idx > 0:
            for name, param in self.model.named_parameters():
                if name not in self.important_directions:
                    continue
                if param.grad is None:
                    continue

                param.grad.data = self._project_gradient(param.grad, name)

    def after_task(
        self, task_idx: int, task_name: str, task_data: Any
    ) -> None:
        """Compute and store important directions after task completion."""
        super().after_task(task_idx, task_name, task_data)

        # Compute important directions from accumulated gradients
        self._compute_important_directions()

        logger.info(
            f"Task {task_idx} complete. "
            f"Tracking {sum(d.size(1) for d in self.important_directions.values())} "
            f"total directions across {len(self.important_directions)} parameters"
        )

    def get_model_for_inference(self) -> nn.Module:
        """Return the model for inference."""
        return self.model

    def save_state(self, path: str) -> None:
        """Save DOC state including important directions."""
        import json
        from pathlib import Path

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        state = {
            "doc_lambda": self.doc_lambda,
            "subspace_fraction": self.subspace_fraction,
            "use_gradient_projection": self.use_gradient_projection,
            "num_tasks": len(self.task_history),
            "target_param_names": self.target_param_names,
        }

        with open(path / "doc_state.json", "w") as f:
            json.dump(state, f, indent=2)

        # Save important directions
        torch.save(self.important_directions, path / "important_directions.pt")

    def load_state(self, path: str) -> None:
        """Load DOC state."""
        import json
        from pathlib import Path

        path = Path(path)

        with open(path / "doc_state.json", "r") as f:
            state = json.load(f)

        self.doc_lambda = state["doc_lambda"]
        self.subspace_fraction = state["subspace_fraction"]
        self.use_gradient_projection = state["use_gradient_projection"]

        # Load important directions
        directions_path = path / "important_directions.pt"
        if directions_path.exists():
            self.important_directions = torch.load(directions_path)
