"""
Configuration defaults utility for functionalnetworkssft.

This module provides intelligent default configuration values to improve user experience
and reduce required configuration. It handles:
- Storage path defaults (checkpoints, model outputs)
- Model naming defaults based on transformations
- Environment variable configuration
- Directory creation with error handling
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
import re

logger = logging.getLogger(__name__)


class ConfigDefaults:
    """Handles intelligent default configuration values for LMPipeline."""

    # Default directory structure
    DEFAULT_MODELS_DIR = "./models"
    DEFAULT_CHECKPOINTS_DIR = "./models/checkpoints"
    DEFAULT_OUTPUT_DIR = "./models/output"

    # Environment variable names for configuration
    ENV_MODELS_DIR = "LMPIPELINE_MODELS_DIR"
    ENV_CHECKPOINTS_DIR = "LMPIPELINE_CHECKPOINTS_DIR"
    ENV_OUTPUT_DIR = "LMPIPELINE_OUTPUT_DIR"

    # Model naming suffixes
    TRANSFORMATION_SUFFIXES = {
        "quantized": "-quantized",
        "gguf": "-gguf",
        "finetuned": "-finetuned",
        "fp16": "-fp16",
        "float16": "-fp16",  # Map float16 to fp16
        "fp32": "-fp32",
        "float32": "-fp32",  # Map float32 to fp32
        "bf16": "-bf16",
        "bfloat16": "-bf16",  # Map bfloat16 to bf16
        "int8": "-int8",
        "int4": "-int4",
        "4bit": "-4bit",
        "8bit": "-8bit",
    }

    @classmethod
    def get_default_checkpoints_dir(cls) -> str:
        """Get the default checkpoints directory."""
        return os.environ.get(cls.ENV_CHECKPOINTS_DIR, cls.DEFAULT_CHECKPOINTS_DIR)

    @classmethod
    def get_default_output_dir(cls) -> str:
        """Get the default output directory."""
        return os.environ.get(cls.ENV_OUTPUT_DIR, cls.DEFAULT_OUTPUT_DIR)

    @classmethod
    def get_default_models_dir(cls) -> str:
        """Get the default models directory."""
        return os.environ.get(cls.ENV_MODELS_DIR, cls.DEFAULT_MODELS_DIR)

    @classmethod
    def ensure_directory_exists(cls, directory: str) -> bool:
        """
        Ensure a directory exists, creating it if necessary.

        Args:
            directory: Path to the directory

        Returns:
            True if directory exists or was created successfully, False otherwise
        """
        try:
            directory_path = Path(directory)

            # Check if directory already exists
            if directory_path.exists():
                if directory_path.is_dir():
                    logger.debug(f"Directory already exists: {directory}")
                    return True
                else:
                    logger.error(f"Path exists but is not a directory: {directory}")
                    return False

            # Try to create the directory
            directory_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Directory created: {directory}")
            return True

        except PermissionError:
            logger.error(f"Permission denied creating directory: {directory}")
            logger.error(
                "Please check directory permissions or run with appropriate privileges"
            )
            return False
        except OSError as e:
            if e.errno == 28:  # No space left on device
                logger.error(
                    f"No space left on device when creating directory: {directory}"
                )
                logger.error("Please free up disk space and try again")
            elif e.errno == 30:  # Read-only file system
                logger.error(
                    f"Cannot create directory on read-only file system: {directory}"
                )
                logger.error("Please choose a writable location")
            else:
                logger.error(f"Failed to create directory {directory}: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error creating directory {directory}: {e}")
            return False

    @classmethod
    def extract_model_base_name(cls, model_name_or_path: str) -> str:
        """
        Extract a clean base name from a model name or path.

        Args:
            model_name_or_path: Original model name or path

        Returns:
            Clean base name suitable for use in generated names
        """
        # Handle HuggingFace model names (e.g., "microsoft/DialoGPT-medium")
        if "/" in model_name_or_path:
            base_name = model_name_or_path.split("/")[-1]
        else:
            # Handle local paths
            base_name = Path(model_name_or_path).name

        # Clean up the name - remove common suffixes and normalize
        base_name = base_name.lower()

        # Remove common model suffixes that might be redundant
        suffixes_to_remove = ["-hf", "-chat", "-instruct", "-base"]
        for suffix in suffixes_to_remove:
            if base_name.endswith(suffix):
                base_name = base_name[: -len(suffix)]
                break

        return base_name

    @classmethod
    def generate_model_name(
        cls,
        base_model_name: str,
        stages: List[str],
        quantization_config: Optional[Dict[str, Any]] = None,
        torch_dtype: Optional[str] = None,
        convert_to_gguf: bool = False,
        gguf_quantization: Optional[str] = None,
    ) -> str:
        """
        Generate an intelligent model name based on transformations.

        Args:
            base_model_name: Original model name
            stages: List of pipeline stages applied
            quantization_config: Quantization configuration if any
            torch_dtype: Torch dtype used
            convert_to_gguf: Whether GGUF conversion is enabled
            gguf_quantization: GGUF quantization type

        Returns:
            Generated model name with appropriate suffixes
        """
        base_name = cls.extract_model_base_name(base_model_name)
        suffixes = []

        # Add stage-based suffixes
        if stages:
            # Check for fine-tuning stages
            finetuning_stages = {"sft", "dpo", "rlaif", "rl", "cot_distillation"}
            if any(stage in finetuning_stages for stage in stages):
                suffixes.append(cls.TRANSFORMATION_SUFFIXES["finetuned"])

        # Add quantization suffix
        if quantization_config:
            if quantization_config.get("use_4bit"):
                suffixes.append(cls.TRANSFORMATION_SUFFIXES["4bit"])
            elif quantization_config.get("use_8bit"):
                suffixes.append(cls.TRANSFORMATION_SUFFIXES["8bit"])
            else:
                suffixes.append(cls.TRANSFORMATION_SUFFIXES["quantized"])

        # Add precision suffix
        if torch_dtype and torch_dtype != "auto":
            dtype_suffix = cls.TRANSFORMATION_SUFFIXES.get(torch_dtype)
            if dtype_suffix:
                suffixes.append(dtype_suffix)

        # Add GGUF suffix with quantization info
        if convert_to_gguf:
            if gguf_quantization and gguf_quantization != "f16":
                # Add specific GGUF quantization info for non-default quantizations
                gguf_suffix = f"-gguf-{gguf_quantization}"
                suffixes.append(gguf_suffix)
            else:
                suffixes.append(cls.TRANSFORMATION_SUFFIXES["gguf"])

        # Combine base name with suffixes
        final_name = base_name + "".join(suffixes)

        logger.info(f"Generated model name: {final_name} (from {base_model_name})")
        return final_name

    @classmethod
    def apply_storage_defaults(cls, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply storage path defaults to configuration.

        Args:
            config_dict: Configuration dictionary

        Returns:
            Updated configuration with storage defaults applied
        """
        updated_config = config_dict.copy()

        # Apply default output directory if not specified
        if not updated_config.get("output_dir"):
            # Build a model-specific output directory under the default root
            default_output_root = cls.get_default_output_dir()
            base_model = updated_config.get("model_name_or_path", "model")
            model_dir_name = cls.extract_model_base_name(base_model)
            model_output_dir = os.path.join(default_output_root, model_dir_name)

            updated_config["output_dir"] = model_output_dir
            logger.info(
                f"Applied default output directory (model-specific): {model_output_dir}"
            )

            # Ensure the directory exists
            if not cls.ensure_directory_exists(model_output_dir):
                logger.warning(
                    f"Could not create default output directory: {model_output_dir}"
                )

        # Apply checkpoint directory defaults to stage configs
        updated_config = cls._apply_checkpoint_defaults(updated_config)

        return updated_config

    @classmethod
    def _apply_checkpoint_defaults(cls, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply checkpoint directory defaults to stage configurations.

        Args:
            config_dict: Configuration dictionary

        Returns:
            Updated configuration with checkpoint defaults applied
        """
        updated_config = config_dict.copy()
        stage_configs = updated_config.get("stage_configs", {})

        default_checkpoints_dir = cls.get_default_checkpoints_dir()

        # Ensure checkpoints directory exists
        if not cls.ensure_directory_exists(default_checkpoints_dir):
            logger.warning(
                f"Could not create default checkpoints directory: {default_checkpoints_dir}"
            )
            return updated_config

        # Apply checkpoint defaults to each stage that supports checkpointing
        for stage_name, stage_config in stage_configs.items():
            if isinstance(stage_config, dict):
                # Add default checkpoint directory if not specified
                if "output_dir" not in stage_config:
                    # Use stage-specific subdirectory under checkpoints
                    stage_checkpoint_dir = os.path.join(
                        default_checkpoints_dir, stage_name
                    )
                    stage_config["checkpoint_dir"] = stage_checkpoint_dir
                    logger.info(
                        f"Applied default checkpoint directory for {stage_name}: {stage_checkpoint_dir}"
                    )

                    # Ensure stage checkpoint directory exists
                    cls.ensure_directory_exists(stage_checkpoint_dir)

                # Set default checkpoint saving behavior if not specified
                if "save_steps" not in stage_config:
                    stage_config["save_steps"] = 500  # Save checkpoint every 500 steps
                    logger.info(f"Applied default save_steps for {stage_name}: 500")

                if "save_total_limit" not in stage_config:
                    stage_config["save_total_limit"] = (
                        3  # Keep only 3 most recent checkpoints
                    )
                    logger.info(f"Applied default save_total_limit for {stage_name}: 3")

        updated_config["stage_configs"] = stage_configs
        return updated_config

    @classmethod
    def apply_model_naming_defaults(cls, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply model naming defaults to configuration.

        Args:
            config_dict: Configuration dictionary

        Returns:
            Updated configuration with model naming defaults applied
        """
        updated_config = config_dict.copy()

        # Generate default GGUF output path if GGUF conversion is enabled but no path specified
        if updated_config.get("convert_to_gguf") and not updated_config.get(
            "gguf_output_path"
        ):
            base_model_name = updated_config.get("model_name_or_path", "model")
            stages = updated_config.get("stages", [])

            # Extract quantization config from stage configs
            quantization_config = cls._extract_quantization_config(updated_config)

            generated_name = cls.generate_model_name(
                base_model_name=base_model_name,
                stages=stages,
                quantization_config=quantization_config,
                torch_dtype=updated_config.get("torch_dtype"),
                convert_to_gguf=True,
                gguf_quantization=updated_config.get("gguf_quantization"),
            )

            output_dir = updated_config.get("output_dir", cls.get_default_output_dir())
            gguf_path = os.path.join(output_dir, f"{generated_name}.gguf")
            updated_config["gguf_output_path"] = gguf_path
            logger.info(f"Applied default GGUF output path: {gguf_path}")

        return updated_config

    @classmethod
    def _extract_quantization_config(
        cls, config_dict: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Extract quantization configuration from stage configs.

        Args:
            config_dict: Configuration dictionary

        Returns:
            Quantization configuration if found, None otherwise
        """
        stage_configs = config_dict.get("stage_configs", {})

        # Look for quantization config in any stage (typically SFT)
        for _, stage_config in stage_configs.items():
            if isinstance(stage_config, dict):
                if stage_config.get("use_4bit") or stage_config.get("use_8bit"):
                    return {
                        "use_4bit": stage_config.get("use_4bit", False),
                        "use_8bit": stage_config.get("use_8bit", False),
                    }

        return None

    @classmethod
    def apply_all_defaults(cls, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply all default configurations with fallback handling.

        Args:
            config_dict: Configuration dictionary

        Returns:
            Updated configuration with all defaults applied
        """
        logger.info("Applying intelligent configuration defaults...")

        # Apply defaults in order with fallback handling
        updated_config = cls.apply_storage_defaults_with_fallback(config_dict)
        updated_config = cls.apply_model_naming_defaults(updated_config)

        logger.info("Configuration defaults applied successfully")
        return updated_config

    @classmethod
    def generate_final_model_name(cls, config_dict: Dict[str, Any]) -> str:
        """
        Generate a final model name for the completed pipeline.

        Args:
            config_dict: Configuration dictionary

        Returns:
            Generated final model name
        """
        base_model_name = config_dict.get("model_name_or_path", "model")
        stages = config_dict.get("stages", [])
        quantization_config = cls._extract_quantization_config(config_dict)

        return cls.generate_model_name(
            base_model_name=base_model_name,
            stages=stages,
            quantization_config=quantization_config,
            torch_dtype=config_dict.get("torch_dtype"),
            convert_to_gguf=config_dict.get("convert_to_gguf", False),
            gguf_quantization=config_dict.get("gguf_quantization"),
        )

    @classmethod
    def get_default_final_model_path(cls, config_dict: Dict[str, Any]) -> str:
        """
        Get the default path for the final model.

        Args:
            config_dict: Configuration dictionary

        Returns:
            Default final model path
        """
        output_dir = config_dict.get("output_dir", cls.get_default_output_dir())
        model_name = cls.generate_final_model_name(config_dict)
        return os.path.join(output_dir, model_name)

    @classmethod
    def validate_and_create_directories(cls, config_dict: Dict[str, Any]) -> List[str]:
        """
        Validate and create all necessary directories for the pipeline.

        Args:
            config_dict: Configuration dictionary

        Returns:
            List of directories that could not be created (empty if all successful)
        """
        failed_dirs = []

        # Main output directory
        output_dir = config_dict.get("output_dir")
        if output_dir and not cls.ensure_directory_exists(output_dir):
            failed_dirs.append(output_dir)

        # Checkpoints directory
        checkpoints_dir = cls.get_default_checkpoints_dir()
        if not cls.ensure_directory_exists(checkpoints_dir):
            failed_dirs.append(checkpoints_dir)

        # Stage-specific directories
        stage_configs = config_dict.get("stage_configs", {})
        for _, stage_config in stage_configs.items():
            if isinstance(stage_config, dict):
                checkpoint_dir = stage_config.get("checkpoint_dir")
                if checkpoint_dir and not cls.ensure_directory_exists(checkpoint_dir):
                    failed_dirs.append(checkpoint_dir)

        return failed_dirs

    @classmethod
    def get_fallback_directory(cls, preferred_dir: str) -> str:
        """
        Get a fallback directory when the preferred directory cannot be created.

        Args:
            preferred_dir: The preferred directory that failed to create

        Returns:
            A fallback directory path
        """
        # Try current working directory first
        fallback_options = [
            "./outputs",  # Simple outputs directory in current location
            ".",  # Current directory as last resort
        ]

        for fallback in fallback_options:
            if cls.ensure_directory_exists(fallback):
                logger.warning(
                    f"Using fallback directory {fallback} instead of {preferred_dir}"
                )
                return fallback

        # If all else fails, use current directory
        logger.error(
            f"All fallback options failed, using current directory instead of {preferred_dir}"
        )
        return "."

    @classmethod
    def apply_storage_defaults_with_fallback(
        cls, config_dict: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Apply storage defaults with fallback handling for directory creation failures.

        Args:
            config_dict: Configuration dictionary

        Returns:
            Updated configuration with storage defaults and fallbacks applied
        """
        updated_config = config_dict.copy()

        # Apply default output directory if not specified
        if not updated_config.get("output_dir"):
            # Build a model-specific output directory under the default root
            default_output_root = cls.get_default_output_dir()
            base_model = updated_config.get("model_name_or_path", "model")
            model_dir_name = cls.extract_model_base_name(base_model)
            model_output_dir = os.path.join(default_output_root, model_dir_name)

            # Try to create the model-specific directory
            if cls.ensure_directory_exists(model_output_dir):
                updated_config["output_dir"] = model_output_dir
                logger.info(
                    f"Applied default output directory (model-specific): {model_output_dir}"
                )
            else:
                # Use fallback directory
                fallback_dir = cls.get_fallback_directory(model_output_dir)
                updated_config["output_dir"] = fallback_dir
                logger.warning(f"Using fallback output directory: {fallback_dir}")

        # Apply checkpoint directory defaults with fallback
        updated_config = cls._apply_checkpoint_defaults_with_fallback(updated_config)

        return updated_config

    @classmethod
    def _apply_checkpoint_defaults_with_fallback(
        cls, config_dict: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Apply checkpoint directory defaults with fallback handling.

        Args:
            config_dict: Configuration dictionary

        Returns:
            Updated configuration with checkpoint defaults and fallbacks applied
        """
        updated_config = config_dict.copy()
        stage_configs = updated_config.get("stage_configs", {})

        default_checkpoints_dir = cls.get_default_checkpoints_dir()

        # Try to create checkpoints directory, use fallback if it fails
        if not cls.ensure_directory_exists(default_checkpoints_dir):
            logger.warning(
                f"Could not create default checkpoints directory: {default_checkpoints_dir}"
            )
            default_checkpoints_dir = cls.get_fallback_directory(
                default_checkpoints_dir
            )

        # Apply checkpoint defaults to each stage that supports checkpointing
        for stage_name, stage_config in stage_configs.items():
            if isinstance(stage_config, dict):
                # Add default checkpoint directory if not specified
                if "output_dir" not in stage_config:
                    # Use stage-specific subdirectory under checkpoints
                    stage_checkpoint_dir = os.path.join(
                        default_checkpoints_dir, stage_name
                    )

                    # Try to create stage checkpoint directory
                    if cls.ensure_directory_exists(stage_checkpoint_dir):
                        stage_config["checkpoint_dir"] = stage_checkpoint_dir
                        logger.info(
                            f"Applied default checkpoint directory for {stage_name}: {stage_checkpoint_dir}"
                        )
                    else:
                        # Use fallback for stage checkpoint directory
                        fallback_stage_dir = os.path.join(
                            default_checkpoints_dir, f"stage_{stage_name}"
                        )
                        if cls.ensure_directory_exists(fallback_stage_dir):
                            stage_config["checkpoint_dir"] = fallback_stage_dir
                            logger.warning(
                                f"Using fallback checkpoint directory for {stage_name}: {fallback_stage_dir}"
                            )
                        else:
                            # Last resort: use the main checkpoints directory
                            stage_config["checkpoint_dir"] = default_checkpoints_dir
                            logger.warning(
                                f"Using main checkpoints directory for {stage_name}: {default_checkpoints_dir}"
                            )

                # Set default checkpoint saving behavior if not specified
                if "save_steps" not in stage_config:
                    stage_config["save_steps"] = 500
                    logger.info(f"Applied default save_steps for {stage_name}: 500")

                if "save_total_limit" not in stage_config:
                    stage_config["save_total_limit"] = 3
                    logger.info(f"Applied default save_total_limit for {stage_name}: 3")

        updated_config["stage_configs"] = stage_configs
        return updated_config
