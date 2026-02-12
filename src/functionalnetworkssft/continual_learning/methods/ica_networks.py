"""
ICA-based Functional Network method for continual learning.
Integrates with FunctionalNetworksSFT's ICA masking capabilities.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch import nn

from .base import ContinualLearningMethod
from ...ica_mask import ICAMask

logger = logging.getLogger(__name__)


class ICANetworksCL(ContinualLearningMethod):
    """
    ICA-based Functional Network method for continual learning.

    This method leverages the functional network decomposition from
    FunctionalNetworksSFT to selectively train or protect network
    components during continual learning.

    Modes:
    - 'lesion': Mask (zero out) specific ICA components during training
               to protect them from being modified
    - 'preserve': Only allow training through specific ICA components,
                 protecting all other neurons
    """

    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        ica_template_path: Optional[str] = None,
        mask_mode: str = "lesion",
        ica_components: int = 10,
        ica_percentile: float = 98.0,
        component_selection_strategy: str = "cumulative",
        protected_component_ids: Optional[List[int]] = None,
    ):
        """
        Initialize ICA Networks CL method.

        Args:
            model: The base model
            config: Training configuration
            ica_template_path: Path to precomputed ICA templates
            mask_mode: 'lesion' to mask components, 'preserve' to keep only components
            ica_components: Number of ICA components to extract
            ica_percentile: Percentile for thresholding component masks
            component_selection_strategy: How to select components to protect
            protected_component_ids: Specific component IDs to protect (optional)
        """
        super().__init__(model, config)

        self.ica_template_path = ica_template_path
        self.mask_mode = mask_mode
        self.ica_components = ica_components
        self.ica_percentile = ica_percentile
        self.component_selection_strategy = component_selection_strategy
        self.protected_component_ids = protected_component_ids or []

        self.ica_mask: Optional[ICAMask] = None
        self.task_protected_components: Dict[int, List[int]] = {}

        if ica_template_path:
            self._load_ica_templates(ica_template_path)
        else:
            logger.warning(
                "ICA Networks method initialized without template path. "
                "No masking will be applied. Provide --ica_template_path to enable ICA masking."
            )

    def _load_ica_templates(self, template_path: str) -> None:
        """Load precomputed ICA templates."""
        logger.info(f"Loading ICA templates from {template_path}")
        with open(template_path, "r") as f:
            data = json.load(f)

        self.ica_mask = ICAMask(
            num_components=data.get("metadata", {}).get(
                "num_components", self.ica_components
            ),
            percentile=data.get("metadata", {}).get(
                "percentile", self.ica_percentile
            ),
        )

        components = data.get("components", {})
        self.ica_mask.mask_dict_components = {
            int(k): v for k, v in components.items()
        }

        if "layout" in data:
            self.ica_mask.global_feature_layout = data["layout"]

    def _select_components_to_protect(
        self, task_idx: int, task_data: Any
    ) -> List[int]:
        """Select which ICA components to protect for this task."""
        if self.protected_component_ids:
            return self.protected_component_ids

        if self.component_selection_strategy == "fixed":
            return list(range(min(task_idx + 1, self.ica_components)))
        elif self.component_selection_strategy == "cumulative":
            return list(range(min(task_idx + 1, self.ica_components)))
        else:
            return list(range(min(task_idx + 1, self.ica_components)))

    def before_task(
        self, task_idx: int, task_name: str, task_data: Any
    ) -> None:
        """Setup ICA masking before training on a task."""
        super().before_task(task_idx, task_name, task_data)

        if task_idx > 0 and self.ica_mask is not None:
            protected = self._select_components_to_protect(task_idx, task_data)
            self.task_protected_components[task_idx] = protected
            logger.info(f"Task {task_idx}: Protecting components {protected}")

            self.ica_mask.apply_component_masks(
                self.model, component_ids=protected, mode=self.mask_mode
            )

    def compute_loss(
        self, batch: Dict[str, torch.Tensor], task_idx: int
    ) -> torch.Tensor:
        """Compute loss with ICA masking applied via forward hooks."""
        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        return outputs.loss

    def after_task(
        self, task_idx: int, task_name: str, task_data: Any
    ) -> None:
        """Update ICA components after task completion."""
        super().after_task(task_idx, task_name, task_data)

        if self.ica_mask and self.ica_mask.mask_handles:
            for handle in self.ica_mask.mask_handles:
                handle.remove()
            self.ica_mask.mask_handles = []

    def get_model_for_inference(self) -> nn.Module:
        """Return model without masks for inference."""
        if self.ica_mask and self.ica_mask.mask_handles:
            for handle in self.ica_mask.mask_handles:
                handle.remove()
            self.ica_mask.mask_handles = []
        return self.model

    def save_state(self, path: str) -> None:
        """Save ICA state and protected components."""
        state = {
            "task_protected_components": self.task_protected_components,
            "ica_components": self.ica_components,
            "ica_percentile": self.ica_percentile,
            "mask_mode": self.mask_mode,
        }
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "ica_cl_state.json", "w") as f:
            json.dump(state, f, indent=2)

