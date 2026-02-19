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
from torch.nn.utils import parametrize

from .base import ContinualLearningMethod
from ...ica_mask import ICAMask, RowMaskedDelta, _build_row_mask, _union_selected_rows

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

    Anti-drift:
        When enabled, applies RowMaskedDelta parametrization to prevent
        optimizer state drift on frozen (masked) neurons. Without this,
        Adam momentum can accumulate on frozen rows and cause a sudden
        jump when the mask changes.
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
        anti_drift: bool = False,
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
            anti_drift: If True, apply RowMaskedDelta parametrization to
                prevent optimizer drift on frozen neurons.
        """
        super().__init__(model, config)

        self.ica_template_path = ica_template_path
        self.mask_mode = mask_mode
        self.ica_components = ica_components
        self.ica_percentile = ica_percentile
        self.component_selection_strategy = component_selection_strategy
        self.protected_component_ids = protected_component_ids or []
        self.anti_drift = anti_drift

        self.ica_mask: Optional[ICAMask] = None
        self.task_protected_components: Dict[int, List[int]] = {}
        self._parametrized_modules: List[str] = []

        if ica_template_path:
            self._load_ica_templates(ica_template_path)
        else:
            logger.error(
                "ICA Networks method initialized WITHOUT a template path. "
                "No ICA masking will be applied — the method degrades to "
                "vanilla LoRA and CL benchmark results will be INVALID. "
                "Provide ica_template_path (or --ica_template_path on the CLI) "
                "to enable ICA masking."
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

    def _apply_anti_drift(self, component_ids: List[int]) -> None:
        """Apply RowMaskedDelta parametrization to freeze masked rows.

        This prevents optimizer momentum from accumulating on frozen
        neurons, which would cause discontinuous jumps when masks change.
        """
        if self.ica_mask is None:
            return

        # Gather rows that should be trainable from the ICA mask
        components = self.ica_mask.mask_dict_components

        # Find MLP down-projection layers and apply parametrization
        for name, module in self.model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            # Identify down-projection layers (out_features == hidden_size, in > out)
            if module.out_features >= module.in_features:
                continue

            # Extract layer index from module name
            layer_idx = None
            for part in name.split("."):
                try:
                    layer_idx = int(part)
                except ValueError:
                    continue

            if layer_idx is None:
                continue

            layer_key = f"layer_{layer_idx}"

            # Collect trainable rows from selected components
            train_rows = []
            for comp_id in component_ids:
                comp_data = components.get(comp_id, {})
                rows = comp_data.get(layer_key, [])
                train_rows.extend(rows)
            train_rows = sorted(set(train_rows))

            if not train_rows:
                continue

            # Invert for lesion mode (protect = freeze selected, train = rest)
            if self.mask_mode == "lesion":
                all_rows = set(range(module.in_features))
                train_rows = sorted(all_rows - set(train_rows))

            row_mask = _build_row_mask(
                module.in_features,
                train_rows,
                module.weight.device,
                module.weight.dtype,
            )

            # Transpose mask for weight parametrization (weight is [out, in])
            # RowMaskedDelta expects [out_features, 1]
            # Here we need to mask input features (intermediate dim)
            # Parametrize the weight
            try:
                parametrize.register_parametrization(
                    module, "weight",
                    RowMaskedDelta(
                        row_mask.T,  # [1, in_features] for broadcast over rows
                        module.weight.data,
                    ),
                )
                self._parametrized_modules.append(name)
                logger.debug(f"Anti-drift parametrization on {name}")
            except Exception as e:
                logger.warning(f"Could not apply anti-drift to {name}: {e}")

    def _remove_anti_drift(self) -> None:
        """Remove RowMaskedDelta parametrizations."""
        for name in self._parametrized_modules:
            parts = name.split(".")
            module = self.model
            for part in parts:
                module = getattr(module, part)
            try:
                parametrize.remove_parametrizations(module, "weight")
            except Exception:
                pass
        self._parametrized_modules = []

    def before_task(
        self, task_idx: int, task_name: str, task_data: Any
    ) -> None:
        """Setup ICA masking before training on a task."""
        super().before_task(task_idx, task_name, task_data)

        if self.ica_mask is None:
            if task_idx > 0:
                logger.warning(
                    f"Task {task_idx} ({task_name}): ICA mask is None — "
                    "skipping masking. Results will not reflect ICA method."
                )
            return

        if task_idx > 0:
            protected = self._select_components_to_protect(task_idx, task_data)
            self.task_protected_components[task_idx] = protected
            logger.info(f"Task {task_idx}: Protecting components {protected}")

            self.ica_mask.apply_component_masks(
                self.model, component_ids=protected, mode=self.mask_mode
            )

            if self.anti_drift:
                logger.info(f"Task {task_idx}: Applying anti-drift parametrization")
                self._apply_anti_drift(protected)

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

        if self.anti_drift:
            self._remove_anti_drift()

    def get_model_for_inference(self) -> nn.Module:
        """Return model without masks for inference."""
        if self.ica_mask and self.ica_mask.mask_handles:
            for handle in self.ica_mask.mask_handles:
                handle.remove()
            self.ica_mask.mask_handles = []

        if self.anti_drift:
            self._remove_anti_drift()

        return self.model

    def get_state_dict(self) -> Dict[str, Any]:
        """Return ICA Networks state for checkpointing."""
        state = super().get_state_dict()
        state["task_protected_components"] = {
            str(k): v for k, v in self.task_protected_components.items()
        }
        state["ica_components"] = self.ica_components
        state["ica_percentile"] = self.ica_percentile
        state["mask_mode"] = self.mask_mode
        state["ica_template_path"] = self.ica_template_path
        state["anti_drift"] = self.anti_drift
        return state

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore ICA Networks state from checkpoint."""
        super().load_state_dict(state)
        self.task_protected_components = {
            int(k): v
            for k, v in state.get("task_protected_components", {}).items()
        }
        self.ica_components = state.get("ica_components", self.ica_components)
        self.ica_percentile = state.get("ica_percentile", self.ica_percentile)
        self.mask_mode = state.get("mask_mode", self.mask_mode)
        self.anti_drift = state.get("anti_drift", self.anti_drift)

    def save_state(self, path: str) -> None:
        """Save ICA state and protected components."""
        state = {
            "task_protected_components": self.task_protected_components,
            "ica_components": self.ica_components,
            "ica_percentile": self.ica_percentile,
            "mask_mode": self.mask_mode,
            "anti_drift": self.anti_drift,
        }
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "ica_cl_state.json", "w") as f:
            json.dump(state, f, indent=2)
