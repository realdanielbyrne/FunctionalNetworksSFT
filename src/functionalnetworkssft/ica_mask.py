#!/usr/bin/env python3
"""
ICA-based functional network masking utilities for FunctionalNetworksSFT.

Refactored to add a global (group-wise) ICA over concatenated final MLP outputs
and component-wise masking at the MLP output, while preserving the original
per-layer ICA over FFN intermediate activations.

Author: Daniel Byrne
License: MIT
"""

import json
import logging
import os
from collections import defaultdict
from typing import Dict, List, Optional, Literal, Any, Tuple
import itertools

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from sklearn.decomposition import FastICA
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


class ICAMask:
    """
    Compute and apply ICA-based masks to transformer models.

    Two analysis modes:
      1) Per-layer FFN-intermediate ICA (original): selects key FFN neurons (d_ff)
         and masks via pre-hook on the down-projection input.
      2) Global group-wise ICA (new): concatenates final MLP outputs (hidden_size) across
         all layers, runs one ICA, and defines one mask per ICA component (functional network).
         Masks are applied at the MLP output (post-activation, post down-projection).
    """

    def __init__(
        self,
        num_components: int = 10,
        percentile: float = 98.0,
        sample_batches: int = 100,
        ica_dtype: Optional[str] = None,
    ):
        """
        Args:
            num_components: Number of ICA components to extract (default for both modes)
            percentile: Percentile threshold for neuron/component selection

            sample_batches: Number of batches to sample for ICA

            ica_dtype: 'float32'|'float16'|'bfloat16'|'auto'|None – dtype for ICA math
        """
        self.num_components = num_components
        self.percentile = percentile
        self.sample_batches = sample_batches

        self.ica_dtype = ica_dtype

        # Outputs
        self.mask_dict: Optional[Dict[str, List[int]]] = None  # per-layer FFN masks
        self.mask_handles: List[Any] = []

        # NEW: global component-wise outputs
        self.mask_dict_components: Optional[Dict[int, Dict[str, List[int]]]] = None
        self.global_feature_layout: Optional[Dict[str, Any]] = (
            None  # layers order, hidden_size
        )

    # ------------------------ Shared utilities ------------------------

    def _get_ica_dtype(self, model_dtype: torch.dtype) -> torch.dtype:
        """
        Determine the optimal dtype for ICA computation based on configuration and model dtype.

        Args:
            model_dtype: The model's current dtype

        Returns:
            torch.dtype to use for ICA computation
        """
        if self.ica_dtype is None or self.ica_dtype == "float32":
            return torch.float32
        elif self.ica_dtype == "auto":
            if model_dtype in [torch.float16, torch.bfloat16]:
                return torch.float32
            else:
                return model_dtype
        elif self.ica_dtype == "float16":
            return torch.float16
        elif self.ica_dtype == "bfloat16":
            return torch.bfloat16
        else:
            logger.warning(f"Unknown ica_dtype '{self.ica_dtype}', using float32")
            return torch.float32

    def parse_layer_specification(
        self, layer_spec: str, total_layers: int
    ) -> List[int]:
        if not layer_spec or not layer_spec.strip():
            raise ValueError("Layer specification cannot be empty")

        layer_indices = set()
        parts = [part.strip() for part in layer_spec.split(",")]
        for part in parts:
            if not part:
                continue
            if ":" in part:
                start_str, end_str = part.split(":", 1)
                start = 0 if start_str == "" else int(start_str)
                end = total_layers if end_str == "" else int(end_str)
                if (
                    start < 0
                    or end < 0
                    or start >= total_layers
                    or end > total_layers
                    or start >= end
                ):
                    raise ValueError(
                        f"Invalid range: '{part}' for total_layers={total_layers}"
                    )
                layer_indices.update(range(start, end))
            else:
                idx = int(part)
                if idx < 0 or idx >= total_layers:
                    raise ValueError(f"Layer index out of range: {idx}")
                layer_indices.add(idx)
        if not layer_indices:
            raise ValueError("No valid layer indices found in specification")
        return sorted(list(layer_indices))

    def filter_mask_by_layers(
        self, mask_dict: Dict[str, List[int]], target_layers: List[int]
    ) -> Dict[str, List[int]]:
        """
        Filter mask dictionary to only include specified target layers.

        Args:
            mask_dict: Original mask dictionary
            target_layers: List of layer indices to keep

        Returns:
            Filtered mask dictionary
        """
        return {
            str(layer): mask_dict.get(str(layer), [])
            for layer in target_layers
            if str(layer) in mask_dict
        }

    def _find_decoder_blocks_and_mlps(
        self, actual_model: Any
    ) -> Tuple[Optional[List[Any]], Optional[List[Optional[nn.Module]]]]:
        """Return (blocks, mlp_modules) if found, else (None, None)."""
        if hasattr(actual_model, "transformer"):
            blocks = getattr(actual_model.transformer, "h", None) or getattr(
                actual_model.transformer, "blocks", None
            )
        elif hasattr(actual_model, "model"):
            blocks = getattr(actual_model.model, "layers", None) or getattr(
                actual_model.model, "decoder", None
            )
        elif hasattr(actual_model, "layers"):
            blocks = actual_model.layers
        else:
            blocks = None
        if blocks is None:
            return None, None

        mlps: List[Optional[nn.Module]] = []
        for b in blocks:
            if hasattr(b, "mlp") and isinstance(b.mlp, nn.Module):
                mlps.append(b.mlp)
            else:
                mlps.append(None)  # not found (model-specific case)
        return blocks, mlps

    def compute_global_networks(
        self,
        model: PreTrainedModel,
        dataset: Dataset,
        tokenizer: PreTrainedTokenizerBase,  # noqa: ARG002
        target_layers: Optional[List[int]] = None,
        n_components: Optional[int] = None,
        top_percentile_per_component: Optional[float] = None,
    ) -> Dict[int, Dict[str, List[int]]]:
        """
        NEW MODE:
        Run ONE ICA over a global feature space formed by concatenating ALL layers' final MLP outputs.
        Returns: component_masks[comp_id][layer_str] = [hidden_channel indices]
        """
        logger.info(
            "Global ICA: capturing final MLP outputs from all layers (post-activation, post down-proj)."
        )
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        model.eval()
        device = next(model.parameters()).device

        # dtype for ICA math
        model_dtype = next(model.parameters()).dtype
        ica_dtype = self._get_ica_dtype(model_dtype)

        # unwrap PEFT to get base model
        actual_model: Any = model
        if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
            actual_model = model.base_model.model
        elif hasattr(model, "base_model"):
            actual_model = model.base_model

        blocks, mlps = self._find_decoder_blocks_and_mlps(actual_model)
        if blocks is None or mlps is None:
            logger.error("Could not find transformer blocks/MLPs.")
            return {}

        num_layers_total = len(blocks)
        if target_layers is None:
            target_layers = list(range(num_layers_total))
        else:
            target_layers = [i for i in target_layers if 0 <= i < num_layers_total]

        # per-step storage of each layer's MLP output
        per_step: Dict[int, Dict[int, torch.Tensor]] = defaultdict(dict)
        last_mask = {"tensor": None}  # attention_mask for pad filtering
        cur_step = {"id": -1}

        def capture(layer_idx: int):
            def _hook(_, __, out):
                # out: [B, T, hidden_size] final MLP output
                y = out.detach().cpu().to(ica_dtype)
                if last_mask["tensor"] is not None:
                    m = last_mask["tensor"].reshape(-1) > 0  # [B*T]
                    y = y.reshape(-1, y.shape[-1])[m]
                else:
                    y = y.reshape(-1, y.shape[-1])
                per_step[cur_step["id"]][layer_idx] = y  # [time_kept, hidden_size]

            return _hook

        # attach hooks on each target layer's MLP output
        handles = []
        for i, mlp in enumerate(mlps):
            if i not in target_layers or mlp is None:
                continue
            base_mlp = getattr(mlp, "base_layer", mlp)
            handles.append(base_mlp.register_forward_hook(capture(i)))

        # run dataloader to collect signals
        dl = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
        with torch.no_grad():
            for idx, sample in enumerate(itertools.islice(dl, self.sample_batches)):
                cur_step["id"] = idx
                last_mask["tensor"] = sample["attention_mask"].to(device).detach().cpu()
                _ = model(
                    input_ids=sample["input_ids"].to(device),
                    attention_mask=sample["attention_mask"].to(device),
                )
                if idx and idx % 10 == 0:
                    logger.info(f"  captured {idx}/{self.sample_batches} batches…")

        for h in handles:
            h.remove()
        logger.info("Removed MLP output capture hooks")

        # Identify which layers we actually captured (intersection over steps)
        captured_layers_sorted = (
            sorted(set().union(*[d.keys() for d in per_step.values()]))
            if per_step
            else []
        )
        if not captured_layers_sorted:
            logger.warning("No MLP outputs were captured.")
            return {}

        # infer hidden_size
        some_step = next(iter(per_step.values()))
        some_layer = next(iter(some_step.values()))
        hidden_size = some_layer.shape[-1]

        # build the global design matrix X: [total_time, num_layers * hidden_size]
        rows = []
        for sid in sorted(per_step.keys()):
            layer_to_Y = per_step[sid]
            # ensure all requested layers are present for this step; skip otherwise
            if not all(i in layer_to_Y for i in captured_layers_sorted):
                continue
            Ys = [
                layer_to_Y[i] for i in captured_layers_sorted
            ]  # each [time_kept, hidden_size]
            tmin = min(y.shape[0] for y in Ys)
            if tmin == 0:
                continue
            Ys = [y[:tmin] for y in Ys]
            step_mat = torch.cat(Ys, dim=1)  # [tmin, L*hidden_size]
            rows.append(step_mat)

        if not rows:
            logger.warning("No usable steps after alignment.")
            return {}

        X = torch.cat(rows, dim=0).numpy()  # [time, L*hidden_size]

        # standardize across time
        X_mean = np.mean(X, axis=0)
        X_std = np.maximum(np.std(X, axis=0), 1e-8)
        Xz = (X - X_mean) / X_std

        k = n_components if n_components is not None else self.num_components
        logger.info(
            f"Global ICA: fitting FastICA with n_components={k} on shape {Xz.shape}"
        )
        ica = FastICA(n_components=k, random_state=0, max_iter=200, tol=1e-3)
        _ = ica.fit_transform(Xz)  # time signals (unused downstream)
        A = ica.mixing_  # [L*hidden_size, k] spatial maps

        if A is None or np.any(~np.isfinite(A)):
            logger.warning("FastICA mixing matrix invalid or None.")
            return {}

        # per-component masks by thresholding |A[:, c]|
        p = (
            top_percentile_per_component
            if top_percentile_per_component is not None
            else self.percentile
        )
        component_masks: Dict[int, Dict[str, List[int]]] = {}
        for c in range(A.shape[1]):
            w = np.abs(A[:, c])
            thr = np.percentile(w, p)
            idxs = np.flatnonzero(w >= thr)
            comp_mask: Dict[str, List[int]] = defaultdict(list)
            for j in idxs:
                layer_local = j // hidden_size
                ch = j % hidden_size
                layer_idx = captured_layers_sorted[layer_local]  # remap local→original
                comp_mask[str(layer_idx)].append(int(ch))
            component_masks[c] = {k: sorted(v) for k, v in comp_mask.items()}

        # stash for later application
        self.mask_dict_components = component_masks
        self.global_feature_layout = {
            "captured_layers_sorted": captured_layers_sorted,
            "hidden_size": hidden_size,
        }

        total_on = sum(
            sum(len(v) for v in m.values()) for m in component_masks.values()
        )
        logger.info(
            f"Global ICA complete – built {len(component_masks)} component masks with {total_on} total channel selections."
        )
        return component_masks

    def apply_component_masks(
        self,
        model: PreTrainedModel,
        component_ids: List[int],
        mode: Literal["lesion", "preserve"] = "lesion",
    ) -> List[Any]:
        """
        NEW MODE APPLY:
        Multiply MLP outputs by a per-channel mask. 'lesion' zeros selected channels;
        'preserve' zeros everything else. Operates at the MLP output (post-activation).
        """
        if not self.mask_dict_components:
            raise ValueError(
                "No component masks available. Run compute_global_networks first."
            )

        # union selected components into layer→channels
        union_sets: Dict[str, set] = defaultdict(set)
        for cid in component_ids:
            comp = self.mask_dict_components.get(cid, {})
            for layer, chans in comp.items():
                for ch in chans:
                    union_sets[layer].add(ch)
        union = {k: sorted(v) for k, v in union_sets.items()}

        # unwrap PEFT
        actual_model: Any = model
        if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
            actual_model = model.base_model.model
        elif hasattr(model, "base_model"):
            actual_model = model.base_model

        blocks, mlps = self._find_decoder_blocks_and_mlps(actual_model)
        if blocks is None or mlps is None:
            logger.warning(
                "Could not find transformer blocks/MLPs – no masking applied."
            )
            return []

        hidden_size = (
            getattr(model.config, "hidden_size", None)
            or getattr(model.config, "n_embd", None)
            or getattr(model.config, "d_model", None)
        )
        if hidden_size is None:
            embedding_dim = getattr(model.get_input_embeddings(), "embedding_dim", None)
            if embedding_dim is not None:
                hidden_size = int(embedding_dim)
            else:
                raise ValueError(
                    "Could not determine hidden_size from model config or embeddings"
                )
        else:
            hidden_size = int(hidden_size)

        handles: List[Any] = []
        for i, mlp in enumerate(mlps):
            if mlp is None:
                continue
            chans = union.get(str(i), [])
            chans = [ch for ch in chans if 0 <= ch < hidden_size]
            if not chans:
                continue

            with torch.no_grad():
                if mode == "lesion":
                    mask = torch.ones((hidden_size,), dtype=torch.float32)
                    mask[chans] = 0.0
                else:  # preserve
                    mask = torch.zeros((hidden_size,), dtype=torch.float32)
                    mask[chans] = 1.0
                mask.requires_grad_(False)

            def fwd_hook(_mod, _inp, out, mask_tensor=mask):
                m = mask_tensor.to(device=out.device, dtype=out.dtype)  # [hidden_size]
                return out * m  # out: [B, T, hidden_size]

            base_mlp = getattr(mlp, "base_layer", mlp)
            handles.append(base_mlp.register_forward_hook(fwd_hook))

        self.mask_handles = handles
        return handles

    def build_templates_from_current_components(
        self, name: str = "default"
    ) -> Dict[str, Any]:
        """
        Freeze current global component masks as 'templates'.
        Returns a dict with layout metadata and per-component supports.
        """
        if not getattr(self, "mask_dict_components", None) or not getattr(
            self, "global_feature_layout", None
        ):
            raise ValueError(
                "No global components/layout available. Run compute_global_networks() first."
            )

        # Type assertions after None check
        assert self.mask_dict_components is not None
        assert self.global_feature_layout is not None

        templates = {
            "name": name,
            "layout": {
                "captured_layers_sorted": list(
                    self.global_feature_layout["captured_layers_sorted"]
                ),
                "hidden_size": int(self.global_feature_layout["hidden_size"]),
            },
            # copy to avoid accidental mutation
            "templates": {
                int(cid): {str(ly): list(map(int, chs)) for ly, chs in comp.items()}
                for cid, comp in self.mask_dict_components.items()
            },
        }
        # keep in-memory reference for convenience
        self.component_templates = templates
        return templates

    def save_templates(
        self, path: str, templates: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Save template dictionary (as produced by build_templates_from_current_components) to JSON.
        """
        if templates is None:
            templates = getattr(self, "component_templates", None)
        if templates is None:
            raise ValueError(
                "No templates to save. Call build_templates_from_current_components() first."
            )
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(templates, f, indent=2)

    def load_templates(self, path: str) -> Dict[str, Any]:
        """
        Load previously saved templates JSON and keep them on self.component_templates.
        """
        with open(path, "r") as f:
            templates = json.load(f)
        # normalize keys
        templates["templates"] = {
            int(k): {str(ly): list(map(int, v2)) for ly, v2 in v.items()}
            for k, v in templates["templates"].items()
        }
        self.component_templates = templates
        return templates

    @staticmethod
    def _flatten_support(
        comp_map: Dict[str, List[int]],
        captured_layers_sorted: List[int],
        hidden_size: int,
    ) -> np.ndarray:
        """
        Convert a per-layer channel index map into a flat boolean vector of length L*hidden_size.
        """
        layer_to_local = {int(ly): i for i, ly in enumerate(captured_layers_sorted)}
        flat = np.zeros(len(captured_layers_sorted) * hidden_size, dtype=bool)
        for ly_str, chans in comp_map.items():
            ly = int(ly_str)
            if ly not in layer_to_local:
                continue
            offset = layer_to_local[ly] * hidden_size
            for ch in chans:
                idx = offset + int(ch)
                if 0 <= idx < flat.size:
                    flat[idx] = True
        return flat

    @staticmethod
    def _iou_bool(a: np.ndarray, b: np.ndarray) -> float:
        """
        IoU between two boolean support vectors.
        """
        inter = np.logical_and(a, b).sum()
        union = np.logical_or(a, b).sum()
        return float(inter) / float(union) if union > 0 else 0.0

    def match_components_to_templates(
        self,
        candidate_components: Optional[Dict[int, Dict[str, List[int]]]] = None,
        templates: Optional[Dict[str, Any]] = None,
        iou_threshold: float = 0.2,
        top_k: int = 1,
    ) -> Dict[int, List[Tuple[int, float]]]:
        """
        For each candidate component, compute IoU with each template.
        Return mapping: cand_id -> [(template_id, iou), ...] filtered by iou_threshold, sorted desc, up to top_k.

        Notes:
        - Set top_k=None to return all matches above threshold.
        - This is many-to-one by default (multiple candidates can match the same template).
        """
        # defaults
        if candidate_components is None:
            if not getattr(self, "mask_dict_components", None):
                raise ValueError(
                    "No candidate components. Provide candidate_components or run compute_global_networks()."
                )
            candidate_components = self.mask_dict_components

        # Type assertion after None check
        assert candidate_components is not None

        if templates is None:
            templates = getattr(self, "component_templates", None)
        if templates is None:
            # fall back to using current components as 'templates' (useful for A/B matches)
            if not getattr(self, "mask_dict_components", None) or not getattr(
                self, "global_feature_layout", None
            ):
                raise ValueError(
                    "No templates and no current components; cannot match."
                )
            templates = self.build_templates_from_current_components(name="auto")

        # unpack layout
        cap_layers = list(templates["layout"]["captured_layers_sorted"])
        hidden_size = int(templates["layout"]["hidden_size"])

        # pre-flatten templates
        flat_templates = {
            int(tid): self._flatten_support(tcomp, cap_layers, hidden_size)
            for tid, tcomp in templates["templates"].items()
        }

        # match
        results: Dict[int, List[Tuple[int, float]]] = {}
        for cid, cmap in candidate_components.items():
            cand_flat = self._flatten_support(cmap, cap_layers, hidden_size)
            pairs = []
            for tid, tflat in flat_templates.items():
                iou = self._iou_bool(cand_flat, tflat)
                if iou >= iou_threshold:
                    pairs.append((tid, float(iou)))
            pairs.sort(key=lambda x: x[1], reverse=True)
            if top_k is not None:
                pairs = pairs[:top_k]
            results[int(cid)] = pairs
        return results

    def count_matches_by_template(
        self,
        match_result: Dict[int, List[Tuple[int, float]]],
    ) -> Dict[int, int]:
        """
        Reduce match_result to counts per template id (uses only the top match per candidate if available).
        """
        counts: Dict[int, int] = {}
        for _cid, lst in match_result.items():
            if not lst:
                continue
            top_tid = int(lst[0][0])
            counts[top_tid] = counts.get(top_tid, 0) + 1
        return counts
