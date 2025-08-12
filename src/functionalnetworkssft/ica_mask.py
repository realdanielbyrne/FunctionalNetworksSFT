#!/usr/bin/env python3
"""
ICA-based functional network masking utilities for FunctionalNetworksSFT.

This module contains the ICAMask class and related utilities for computing
and applying ICA-based masks to transformer models for functional network
analysis and selective fine-tuning.

Author: Daniel Byrne
License: MIT
"""

import json
import logging
import os
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Literal, Any, Tuple
import itertools

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from sklearn.decomposition import FastICA
from joblib import Parallel, delayed
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


class ICAMask:
    """
    A class for computing and applying ICA-based functional network masks to transformer models.

    This class encapsulates all functionality related to ICA mask computation, application,
    saving, and loading. It provides a clean interface for functional network masking
    operations in transformer fine-tuning.
    """

    def __init__(
        self,
        num_components: int = 20,
        percentile: float = 98.0,
        selection_mode: Literal["max_abs", "l2", "topk"] = "max_abs",
        sample_batches: int = 100,
        clip_activations: bool = False,
        n_jobs: int = -1,
        backend: str = "threading",
        ica_dtype: Optional[str] = None,
    ):
        """
        Initialize the ICAMask instance.

        Args:
            num_components: Number of ICA components to extract
            percentile: Percentile threshold for neuron selection
            selection_mode: Selection mode ("max_abs", "l2", "topk")
            sample_batches: Number of batches to sample for ICA
            clip_activations: Whether to clip extreme activation values
            n_jobs: Number of parallel jobs for ICA computation (-1 uses all cores)
            backend: Joblib backend for parallelization ('threading', 'loky', 'multiprocessing')
            ica_dtype: Data type for ICA computation ('float32', 'float16', 'auto', or None for float32)
        """
        self.num_components = num_components
        self.percentile = percentile
        self.selection_mode = selection_mode
        self.sample_batches = sample_batches
        self.clip_activations = clip_activations
        self.n_jobs = n_jobs
        self.backend = backend
        self.ica_dtype = ica_dtype
        self.mask_dict: Optional[Dict[str, List[int]]] = None
        self.mask_handles: List[Any] = []

    def _get_ica_dtype(self, model_dtype: torch.dtype) -> torch.dtype:
        """
        Determine the optimal dtype for ICA computation based on configuration and model dtype.

        Args:
            model_dtype: The model's current dtype

        Returns:
            torch.dtype to use for ICA computation
        """
        if self.ica_dtype is None or self.ica_dtype == "float32":
            # Default: use float32 for maximum numerical stability
            return torch.float32
        elif self.ica_dtype == "auto":
            # Auto: match model dtype but ensure minimum precision
            if model_dtype in [torch.float16, torch.bfloat16]:
                # For half precision models, use float32 for ICA stability
                # unless explicitly requested otherwise
                return torch.float32
            else:
                return model_dtype
        elif self.ica_dtype == "float16":
            return torch.float16
        elif self.ica_dtype == "bfloat16":
            return torch.bfloat16
        else:
            # Fallback to float32 for unknown values
            logger.warning(f"Unknown ica_dtype '{self.ica_dtype}', using float32")
            return torch.float32

    def parse_layer_specification(
        self, layer_spec: str, total_layers: int
    ) -> List[int]:
        """
        Parse layer specification string into a list of layer indices.

        Args:
            layer_spec: String specifying layers (e.g., "0", "0,3,7", "0:4,5:6,9:", "0,2:5,8")
            total_layers: Total number of layers in the model

        Returns:
            List of layer indices that should receive ICA masking

        Raises:
            ValueError: If the specification format is invalid
        """
        if not layer_spec or not layer_spec.strip():
            raise ValueError("Layer specification cannot be empty")

        layer_indices = set()

        # Split by comma to handle multiple specifications
        parts = [part.strip() for part in layer_spec.split(",")]

        for part in parts:
            if not part:
                continue

            if ":" in part:
                # Handle range specification (e.g., "0:4", ":3", "5:", "2:8")
                try:
                    start_str, end_str = part.split(":", 1)

                    # Parse start index
                    if start_str == "":
                        start = 0
                    else:
                        start = int(start_str)
                        if start < 0:
                            raise ValueError(f"Start index cannot be negative: {start}")

                    # Parse end index
                    if end_str == "":
                        end = total_layers
                    else:
                        end = int(end_str)
                        if end < 0:
                            raise ValueError(f"End index cannot be negative: {end}")

                    # Validate range
                    if start >= total_layers:
                        raise ValueError(
                            f"Start index {start} exceeds total layers {total_layers}"
                        )
                    if end > total_layers:
                        raise ValueError(
                            f"End index {end} exceeds total layers {total_layers}"
                        )
                    if start >= end:
                        raise ValueError(f"Invalid range: start {start} >= end {end}")

                    # Add range to set
                    layer_indices.update(range(start, end))

                except ValueError as e:
                    if "invalid literal for int()" in str(e):
                        raise ValueError(
                            f"Invalid range format: '{part}'. Expected format like '0:4', ':3', '5:', etc."
                        )
                    else:
                        raise
            else:
                # Handle single layer specification
                try:
                    layer_idx = int(part)
                    if layer_idx < 0:
                        raise ValueError(f"Layer index cannot be negative: {layer_idx}")
                    if layer_idx >= total_layers:
                        raise ValueError(
                            f"Layer index {layer_idx} exceeds total layers {total_layers}"
                        )
                    layer_indices.add(layer_idx)
                except ValueError as e:
                    if "invalid literal for int()" in str(e):
                        raise ValueError(
                            f"Invalid layer index: '{part}'. Expected integer."
                        )
                    else:
                        raise

        if not layer_indices:
            raise ValueError("No valid layer indices found in specification")

        return sorted(list(layer_indices))

    def save_mask(self, mask_dict: Dict[str, List[int]], file_path: str) -> None:
        """
        Save ICA mask dictionary to a JSON file.

        Args:
            mask_dict: Dictionary mapping layer indices to lists of neuron indices
            file_path: Path where to save the mask file
        """
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as f:
            json.dump(mask_dict, f, indent=2)
        logger.info(f"Saved ICA mask to {file_path}")

    def load_mask(self, file_path: str) -> Dict[str, List[int]]:
        """
        Load ICA mask dictionary from a JSON file.

        Args:
            file_path: Path to the mask file

        Returns:
            Dictionary mapping layer indices to lists of neuron indices
        """
        with open(file_path, "r") as f:
            mask_dict = json.load(f)
        logger.info(f"Loaded ICA mask from {file_path}")
        self.mask_dict = mask_dict
        return mask_dict

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
        filtered_mask_dict = {
            str(layer): mask_dict[str(layer)]
            for layer in target_layers
            if str(layer) in mask_dict
        }
        logger.info(f"Filtered mask to {len(filtered_mask_dict)} layers")
        return filtered_mask_dict

    def _process_layer_ica(
        self,
        layer_idx: int,
        acts: List[torch.Tensor],
    ) -> Optional[Tuple[int, List[int]]]:
        """
        Process ICA computation for a single layer.

        Args:
            layer_idx: Layer index
            acts: List of activation tensors for this layer

        Returns:
            Tuple of (layer_idx, key_neurons) if successful, None if failed
        """
        try:
            logger.info(
                f"Layer {layer_idx}: Processing {len(acts)} activation batches..."
            )
            X = torch.cat(acts, dim=0).flatten(0, 1).numpy()  # [time, neurons]
            logger.info(f"Layer {layer_idx}: Concatenated data shape: {X.shape}")

            # Check if we have enough data for ICA
            if X.shape[0] < self.num_components or X.shape[1] < self.num_components:
                logger.warning(
                    f"Insufficient data for ICA on layer {layer_idx} (shape: {X.shape}), skipping."
                )
                return None

            # More robust numerical checks and preprocessing
            logger.debug(
                f"Layer {layer_idx}: Raw data shape: {X.shape}, dtype: {X.dtype}"
            )
            logger.debug(
                f"Layer {layer_idx}: Raw data range: [{np.min(X):.6f}, {np.max(X):.6f}]"
            )
            logger.debug(
                f"Layer {layer_idx}: NaN count: {np.sum(np.isnan(X))}, Inf count: {np.sum(np.isinf(X))}"
            )

            # Check for numerical issues after cleaning
            if np.any(np.isnan(X)) or np.any(np.isinf(X)):
                logger.warning(
                    f"NaN or Inf values still present in layer {layer_idx} after cleaning, skipping ICA."
                )
                return None

            # Optional outlier removal and clipping (disabled by default)
            if self.clip_activations:
                # More aggressive outlier removal - remove extreme values
                X_percentiles = np.percentile(X, [1, 99], axis=0)
                X_clipped = np.clip(X, X_percentiles[0], X_percentiles[1])
                logger.debug(
                    f"Layer {layer_idx}: Applied outlier clipping to 1st-99th percentiles"
                )
            else:
                X_clipped = X

            # Standard standardization (zero-mean, unit-variance)
            X_mean = np.mean(X_clipped, axis=0)
            X_std_dev = np.std(X_clipped, axis=0)
            # Avoid division by very small numbers
            X_std_dev = np.maximum(X_std_dev, 1e-8)  # More conservative threshold
            X_std = (X_clipped - X_mean) / X_std_dev

            # Optional final clipping for numerical stability (disabled by default)
            if self.clip_activations:
                X_std = np.clip(X_std, -5.0, 5.0)
                logger.debug(
                    f"Layer {layer_idx}: Applied final clipping to [-5.0, 5.0]"
                )

            # Final check after all preprocessing
            if np.any(np.isnan(X_std)) or np.any(np.isinf(X_std)):
                logger.warning(
                    f"NaN or Inf values in preprocessed data for layer {layer_idx}, skipping ICA."
                )
                return None

            logger.debug(
                f"Layer {layer_idx}: Preprocessed data range: [{np.min(X_std):.6f}, {np.max(X_std):.6f}]"
            )

            # Use a more conservative number of components if needed
            effective_components = min(
                self.num_components, X.shape[0] // 2, X.shape[1] // 2
            )
            logger.info(
                f"Layer {layer_idx}: Using {effective_components} ICA components"
            )
            if effective_components < 2:
                logger.warning(
                    f"Too few effective components ({effective_components}) for layer {layer_idx}, skipping."
                )
                return None

            logger.info(f"Layer {layer_idx}: Starting FastICA computation...")
            ica = FastICA(
                n_components=effective_components,
                random_state=0,
                max_iter=100,
                tol=1e-3,
            )

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                logger.info(f"Layer {layer_idx}: Running ICA fit_transform...")
                A = ica.fit_transform(X_std).T  # components × time
                mixing = ica.mixing_  # [neurons, components]
                logger.info(f"Layer {layer_idx}: ICA completed successfully")

            # Check if ICA converged properly
            if np.any(np.isnan(mixing)) or np.any(np.isinf(mixing)):
                logger.warning(
                    f"ICA produced NaN/Inf values for layer {layer_idx}, skipping."
                )
                return None

            # Score neurons and select based on selection_mode
            if self.selection_mode == "max_abs":
                scores = np.max(np.abs(mixing), axis=1)
                thr = np.percentile(scores, self.percentile)
                key_neurons = np.flatnonzero(scores >= thr).tolist()
            elif self.selection_mode == "l2":
                scores = np.linalg.norm(mixing, ord=2, axis=1)
                thr = np.percentile(scores, self.percentile)
                key_neurons = np.flatnonzero(scores >= thr).tolist()
            elif self.selection_mode == "topk":
                scores = np.max(np.abs(mixing), axis=1)
                k = max(
                    1, int(round((100.0 - self.percentile) / 100.0 * scores.shape[0]))
                )
                key_neurons = np.argsort(scores)[-k:].tolist()
            else:
                raise ValueError(f"Unknown ICA selection_mode: {self.selection_mode}")

            if key_neurons:
                # Deduplicate while preserving order
                seen = set()
                key_neurons = [n for n in key_neurons if not (n in seen or seen.add(n))]
                logger.debug(
                    f"Layer {layer_idx}: selection_mode={self.selection_mode}, selected {len(key_neurons)} neurons"
                )
                return (layer_idx, key_neurons)
            else:
                return None

        except (ValueError, RuntimeWarning, np.linalg.LinAlgError) as e:
            logger.warning(f"ICA failed on layer {layer_idx}: {str(e)}, skipping.")
            return None
        except Exception as e:
            logger.warning(
                f"Failed to compute percentile threshold for layer {layer_idx}: {str(e)}, skipping."
            )
            return None

    def compute_masks_for_model(
        self,
        model: PreTrainedModel,
        dataset: Dataset,
        tokenizer: PreTrainedTokenizerBase,
        target_layers: Optional[List[int]] = None,
    ) -> Dict[str, List[int]]:
        """
        Compute ICA masks for a model using the provided dataset.

        Args:
            model: The transformer model to analyze
            dataset: Dataset to sample activations from
            tokenizer: Tokenizer for the model
            target_layers: List of layer indices to process. If None, process all layers.

        Returns:
            Dictionary mapping layer indices to lists of neuron indices
        """
        logger.info("Running ICA to discover functional networks – this can be slow…")
        if target_layers is not None:
            logger.info(f"ICA will be applied only to layers: {target_layers}")
            logger.info(
                f"num_components: {self.num_components}, percentile: {self.percentile}, selection_mode: {self.selection_mode}"
            )
        else:
            logger.info("ICA will be applied to all layers (default behavior)")

        # Set environment variable to avoid tokenizer parallelism conflicts
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        model.eval()
        device = next(model.parameters()).device

        # Determine optimal dtype for ICA computation
        model_dtype = next(model.parameters()).dtype
        ica_dtype = self._get_ica_dtype(model_dtype)
        logger.info(
            f"Using dtype {ica_dtype} for ICA computation (model dtype: {model_dtype})"
        )

        # 1. collect activations
        activations = defaultdict(list)
        hooks = []

        def capture(layer_idx):
            def _hook(_, __, out):
                # out: [B, T, d_int]  -> flatten B*T
                # Convert to optimal dtype for ICA computation
                out_converted = out.detach().cpu().to(ica_dtype)

                # Debug: Check what we're actually capturing
                if len(activations[layer_idx]) == 0:  # First capture for this layer
                    logger.debug(
                        f"Layer {layer_idx}: First capture - shape: {out_converted.shape}, "
                        f"dtype: {out_converted.dtype}, "
                        f"range: [{torch.min(out_converted):.6f}, {torch.max(out_converted):.6f}], "
                        f"mean: {torch.mean(out_converted):.6f}, std: {torch.std(out_converted):.6f}"
                    )

                activations[layer_idx].append(out_converted)

            return _hook

        # attach capture hooks on the MLP *intermediate* output
        # Use the same logic as apply_masks for consistency

        # Handle PEFT models - get the actual base model for ICA computation
        # IMPORTANT: We use the base model because LoRA adapters are blank at training start
        actual_model: Any = model
        if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
            # PEFT model: model.base_model.model is the actual transformer
            actual_model = model.base_model.model
        elif hasattr(model, "base_model"):
            # Some PEFT configurations: model.base_model is the actual transformer
            actual_model = model.base_model

        # Now find transformer blocks in the actual model
        if hasattr(actual_model, "transformer"):
            blocks = getattr(actual_model.transformer, "h", None) or getattr(
                actual_model.transformer, "blocks", None
            )
        elif hasattr(actual_model, "model"):
            blocks = getattr(actual_model.model, "layers", None) or getattr(
                actual_model.model, "decoder", None
            )
        elif hasattr(actual_model, "layers"):
            # Direct access to layers (some model architectures)
            blocks = actual_model.layers
        else:
            blocks = None

        if blocks is None:
            logger.error(
                "Could not find transformer blocks in model for ICA computation"
            )
            return {}

        hooks_attached = 0
        for i, block in enumerate(blocks):
            # Skip layers not in target_layers if filtering is enabled
            if target_layers is not None and i not in target_layers:
                logger.debug(f"Block {i}: Skipping (not in target layers)")
                continue

            # Hook into the base model layers, not LoRA adapters
            up_proj = None
            logger.debug(f"Block {i}: Searching for base model linear layers...")

            # For GPT-style models, look for the MLP layers
            if hasattr(block, "mlp"):
                mlp = block.mlp
                # Look for the first linear layer in MLP (usually c_fc for GPT models)
                if hasattr(mlp, "c_fc"):
                    # Get the base layer, not the LoRA wrapper
                    base_layer = mlp.c_fc
                    if hasattr(base_layer, "base_layer"):
                        # This is a LoRA-wrapped layer, get the original
                        up_proj = base_layer.base_layer
                        logger.debug(f"  Found base layer: c_fc (base_layer)")
                    else:
                        # This is the original layer
                        up_proj = base_layer
                        logger.debug(f"  Found original layer: c_fc")
                elif hasattr(mlp, "up_proj"):
                    # Llama-style models
                    base_layer = mlp.up_proj
                    if hasattr(base_layer, "base_layer"):
                        up_proj = base_layer.base_layer
                        logger.debug(f"  Found base layer: up_proj (base_layer)")
                    else:
                        up_proj = base_layer
                        logger.debug(f"  Found original layer: up_proj")

            # Fallback: search for any suitable linear layer in the base model
            if up_proj is None:
                for n, m in block.named_modules():
                    # Skip LoRA adapter layers
                    if "lora" in n.lower() or "adapter" in n.lower():
                        continue

                    # Check for Linear layers
                    if isinstance(m, nn.Linear) and m.out_features > m.in_features:
                        up_proj = m
                        logger.debug(f"  Found fallback layer '{n}'")
                        break
                    # Check for Conv1D layers (used in GPT models)
                    elif hasattr(m, "weight") and len(m.weight.shape) == 2:
                        # Conv1D layer - weight shape is [out_features, in_features]
                        in_feat, out_feat = m.weight.shape[1], m.weight.shape[0]
                        if out_feat > in_feat:
                            up_proj = m
                            logger.debug(f"  Found fallback Conv1D layer '{n}'")
                            break

            if up_proj is not None:
                hooks.append(up_proj.register_forward_hook(capture(i)))
                hooks_attached += 1
                logger.debug(f"Attached hook to block {i} base layer")
            else:
                logger.warning(
                    f"No suitable base model layer found in block {i}, skipping ICA hook"
                )

        logger.debug(f"Total hooks attached: {hooks_attached}")

        # 2. feed a few mini-batches
        dl = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
        with torch.no_grad():
            for idx, sample in enumerate(itertools.islice(dl, self.sample_batches)):
                model(
                    input_ids=sample["input_ids"].to(device),
                    attention_mask=sample["attention_mask"].to(device),
                )
                if idx and idx % 10 == 0:
                    logger.info(f"  captured {idx}/{self.sample_batches} batches…")

        for h in hooks:
            h.remove()
        logger.info("Removed activation capture hooks")

        # 3. Run ICA in parallel across layers
        logger.info(f"Processing activations for {len(activations)} layers...")
        if self.n_jobs == 1:
            logger.info("Running ICA sequentially (n_jobs=1)")
            # Sequential processing for debugging or when parallelization is not desired
            results = []
            for layer_idx, acts in activations.items():
                result = self._process_layer_ica(layer_idx, acts)
                if result is not None:
                    results.append(result)
        else:
            logger.info(f"Running ICA in parallel with n_jobs={self.n_jobs}")
            try:
                # Try parallel processing with specified backend
                logger.info(f"Using joblib backend: {self.backend}")
                results = Parallel(n_jobs=self.n_jobs, backend=self.backend, verbose=1)(
                    delayed(self._process_layer_ica)(layer_idx, acts)
                    for layer_idx, acts in activations.items()
                )
                # Filter out None results
                results = [r for r in results if r is not None]
            except Exception as e:
                logger.warning(
                    f"Parallel processing with backend '{self.backend}' failed: {e}"
                )
                if self.backend != "threading":
                    logger.info("Trying fallback to threading backend...")
                    try:
                        results = Parallel(
                            n_jobs=self.n_jobs, backend="threading", verbose=1
                        )(
                            delayed(self._process_layer_ica)(layer_idx, acts)
                            for layer_idx, acts in activations.items()
                        )
                        results = [r for r in results if r is not None]
                    except Exception as e2:
                        logger.warning(f"Threading backend also failed: {e2}")
                        logger.info("Falling back to sequential processing...")
                        # Fallback to sequential processing
                        results = []
                        for layer_idx, acts in activations.items():
                            result = self._process_layer_ica(layer_idx, acts)
                            if result is not None:
                                results.append(result)
                else:
                    logger.info("Falling back to sequential processing...")
                    # Fallback to sequential processing
                    results = []
                    for layer_idx, acts in activations.items():
                        result = self._process_layer_ica(layer_idx, acts)
                        if result is not None:
                            results.append(result)

        # Convert results to layer_masks dictionary
        layer_masks = {}
        for layer_idx, key_neurons in results:
            layer_masks[str(layer_idx)] = key_neurons

        total_masked_neurons = sum(len(v) for v in layer_masks.values())
        if total_masked_neurons == 0:
            logger.warning(
                "ICA failed to identify any neurons for masking. Training will proceed without masking."
            )
        else:
            logger.info(
                f"ICA complete – masking {total_masked_neurons} neurons across {len(layer_masks)} layers."
            )

        # Clean up memory after ICA computation
        logger.info("Cleaning up ICA computation memory...")
        del activations
        # Also clean up any other large variables
        import gc

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("Cleared CUDA cache")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
            logger.info("Cleared MPS cache")

        self.mask_dict = layer_masks
        return layer_masks

    def apply_masks(
        self,
        model: PreTrainedModel,
        mask_dict: Optional[Dict[str, List[int]]] = None,
        mask_mode: str = "key",
    ) -> List[Any]:
        """
        Apply ICA masks to a model by injecting forward pre-hooks.

        Args:
            model: The transformer model to apply masks to
            mask_dict: Dictionary mapping layer indices to neuron indices. If None, uses self.mask_dict
            mask_mode: Masking mode - "key" to mask key neurons, "complement" to mask all but key neurons

        Returns:
            List of hook handles that can be used to remove the masks later
        """
        if mask_dict is None:
            if self.mask_dict is None:
                raise ValueError(
                    "No mask dictionary provided and none computed. Call compute_masks_for_model first."
                )
            mask_dict = self.mask_dict

        handles = []
        hidden_size = (
            getattr(model.config, "hidden_size", None)
            or getattr(model.config, "n_embd", None)
            or getattr(model.config, "d_model", None)
            or model.get_input_embeddings().embedding_dim
        )

        # Locate decoder blocks (works for GPT-like and Llama-like layouts)
        # Handle PEFT models by accessing the base model
        actual_model: Any = model
        if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
            # PEFT model: access the underlying model
            actual_model = model.base_model.model
        elif hasattr(model, "base_model"):
            # PEFT model with different structure
            actual_model = model.base_model

        if hasattr(actual_model, "transformer"):
            blocks = getattr(actual_model.transformer, "h", None) or getattr(
                actual_model.transformer, "blocks", None
            )
        elif hasattr(actual_model, "model"):
            blocks = getattr(actual_model.model, "layers", None) or getattr(
                actual_model.model, "decoder", None
            )
        else:
            blocks = None
        if blocks is None:
            logger.warning("Could not find transformer blocks – no masking applied.")
            return handles

        for layer_idx, block in enumerate(blocks):
            for name, module in block.named_modules():
                # pick the *second* Linear in the MLP (in_features > hidden_size)
                # For PEFT models, we need to apply hooks to the base layer, not the PEFT wrapper
                target_module = module

                # Identify linear or equivalent modules by type and shape
                if isinstance(module, torch.nn.Linear):
                    in_features, out_features = module.in_features, module.out_features
                elif (
                    module.__class__.__name__ == "Linear8bitLt"
                ):  # bitsandbytes 8-bit quantized linear
                    in_features, out_features = module.in_features, module.out_features
                elif module.__class__.__name__ == "Linear4bit":  # bitsandbytes 4-bit
                    in_features, out_features = module.in_features, module.out_features
                elif hasattr(module, "base_layer"):
                    # PEFT wrapped layer (e.g., LoRA) - use the base layer for hooks
                    base_layer = module.base_layer
                    target_module = (
                        base_layer  # Apply hook to base layer, not PEFT wrapper
                    )
                    if hasattr(base_layer, "in_features"):
                        # Standard Linear layer
                        in_features, out_features = (
                            base_layer.in_features,
                            base_layer.out_features,
                        )
                    elif hasattr(base_layer, "nf") and hasattr(base_layer, "nx"):
                        # Conv1D layer (used in GPT models) - nf is out_features, nx is in_features
                        in_features, out_features = base_layer.nx, base_layer.nf
                    else:
                        continue
                elif hasattr(module, "nf") and hasattr(module, "nx"):
                    # Direct Conv1D layer
                    in_features, out_features = module.nx, module.nf
                else:
                    continue
                if out_features == hidden_size and in_features > out_features:
                    neuron_ids = mask_dict.get(str(layer_idx), [])

                    # Create mask completely outside gradient computation
                    with torch.no_grad():
                        if mask_mode == "key":  # zero the key neurons
                            mask = torch.ones(in_features, dtype=torch.float32)
                            mask[neuron_ids] = 0.0
                        else:  # zero everything *except* key neurons
                            mask = torch.zeros(in_features, dtype=torch.float32)
                            mask[neuron_ids] = 1.0
                        # Ensure mask requires no gradients
                        mask.requires_grad_(False)

                    def pre_hook(mod, inp, mask_tensor=mask):
                        # Handle both tuple and single tensor inputs
                        if isinstance(inp, tuple):
                            x = inp[0]
                            # Create mask on correct device and ensure it doesn't require gradients
                            mask_device = mask_tensor.to(device=x.device, dtype=x.dtype)
                            mask_device.requires_grad_(False)
                            # Apply mask by element-wise multiplication, preserving gradient flow for unmasked elements
                            masked_x = x * mask_device
                            return (masked_x,) + inp[1:]
                        else:
                            # Single tensor input
                            mask_device = mask_tensor.to(
                                device=inp.device, dtype=inp.dtype
                            )
                            mask_device.requires_grad_(False)
                            return inp * mask_device

                    # Apply hook to the target module (base layer for PEFT, original module otherwise)
                    handles.append(target_module.register_forward_pre_hook(pre_hook))
                    break  # stop after first matching linear in this block

        self.mask_handles = handles
        return handles

    def remove_masks(self) -> None:
        """Remove all applied masks by removing the forward hooks."""
        for handle in self.mask_handles:
            handle.remove()
        self.mask_handles = []
        logger.info("Removed all ICA mask hooks")
