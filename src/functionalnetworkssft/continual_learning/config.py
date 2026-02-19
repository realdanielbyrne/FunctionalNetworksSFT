"""
YAML-driven configuration for continual learning experiments.

Loads experiment configs from YAML files and bridges them to the
flat MODEL_CONFIGS / method kwargs used by the evaluation pipeline.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

logger = logging.getLogger(__name__)


@dataclass
class CLExperimentConfig:
    """Parsed experiment configuration from a YAML file."""

    # Model
    model_key: str = "llama-7b"
    model_name: str = "meta-llama/Llama-2-7b-hf"
    torch_dtype: str = "float16"
    device_map: str = "auto"

    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"]
    )

    # Training
    learning_rate: float = 1e-4
    batch_size: int = 8
    num_steps_per_task: int = 1000
    gradient_clip: float = 1.0
    warmup_ratio: float = 0.1

    # Evaluation
    max_seq_length: int = 512
    num_seeds: int = 3

    # Task orders
    standard_orders: List[str] = field(
        default_factory=lambda: ["order_1", "order_2", "order_3"]
    )
    long_chain_orders: List[str] = field(
        default_factory=lambda: ["order_4", "order_5", "order_6"]
    )

    # Methods
    methods: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # ICA variations
    ica_variations: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # ICA global settings
    ica_components: int = 10
    ica_percentile: float = 98.0
    ica_template_path: Optional[str] = None

    # lm-eval
    lm_eval_benchmarks: List[str] = field(default_factory=lambda: ["mmlu"])

    # Output
    results_dir: str = "./experiments/continual_learning/results"
    save_models: bool = False


def load_cl_config(yaml_path: str) -> CLExperimentConfig:
    """Load and validate a CL experiment configuration from YAML.

    Args:
        yaml_path: Path to the YAML configuration file.

    Returns:
        Parsed CLExperimentConfig.

    Raises:
        FileNotFoundError: If the YAML file does not exist.
        ValueError: If required fields are missing or invalid.
    """
    path = Path(yaml_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {yaml_path}")

    with open(path) as f:
        raw = yaml.safe_load(f)

    config = CLExperimentConfig()

    # Model section
    model = raw.get("model", {})
    config.model_key = model.get("name", config.model_key)
    config.model_name = model.get("model_name", config.model_name)
    config.torch_dtype = model.get("torch_dtype", config.torch_dtype)
    config.device_map = model.get("device_map", config.device_map)

    # LoRA section
    lora = raw.get("lora", {})
    config.lora_r = lora.get("r", config.lora_r)
    config.lora_alpha = lora.get("alpha", config.lora_alpha)
    config.lora_dropout = lora.get("dropout", config.lora_dropout)
    config.lora_target_modules = lora.get(
        "target_modules", config.lora_target_modules
    )

    # Training section
    training = raw.get("training", {})
    config.learning_rate = training.get("learning_rate", config.learning_rate)
    config.batch_size = training.get("batch_size", config.batch_size)
    config.num_steps_per_task = training.get(
        "num_steps_per_task", config.num_steps_per_task
    )
    config.gradient_clip = training.get("gradient_clip", config.gradient_clip)
    config.warmup_ratio = training.get("warmup_ratio", config.warmup_ratio)

    # Evaluation section
    evaluation = raw.get("evaluation", {})
    config.max_seq_length = evaluation.get("max_seq_length", config.max_seq_length)
    config.num_seeds = evaluation.get("num_seeds", config.num_seeds)
    config.lm_eval_benchmarks = evaluation.get(
        "lm_eval_benchmarks", config.lm_eval_benchmarks
    )

    # Task orders
    task_orders = raw.get("task_orders", {})
    if isinstance(task_orders, dict):
        config.standard_orders = task_orders.get(
            "standard", config.standard_orders
        )
        config.long_chain_orders = task_orders.get(
            "long_chain", config.long_chain_orders
        )
    elif isinstance(task_orders, list):
        # Flat list of order names
        config.standard_orders = [o for o in task_orders if o in ("order_1", "order_2", "order_3")]
        config.long_chain_orders = [o for o in task_orders if o in ("order_4", "order_5", "order_6")]

    # Methods section -- can be a list of names or a dict with kwargs
    methods_raw = raw.get("methods", {})
    if isinstance(methods_raw, list):
        # List of method names (backward compat): treat all as defaults
        config.methods = {m: {} for m in methods_raw}
    elif isinstance(methods_raw, dict):
        config.methods = {
            k: (v if isinstance(v, dict) else {})
            for k, v in methods_raw.items()
        }

    # ICA variations
    config.ica_variations = raw.get("ica_variations", {})

    # ICA global settings
    ica = raw.get("ica", {})
    config.ica_components = ica.get("components", config.ica_components)
    config.ica_percentile = ica.get("percentile", config.ica_percentile)
    config.ica_template_path = ica.get("template_path", config.ica_template_path)

    # Output
    output = raw.get("output", {})
    config.results_dir = output.get("results_dir", config.results_dir)
    config.save_models = output.get("save_models", config.save_models)

    logger.info(f"Loaded CL config from {yaml_path}: model={config.model_key}")
    return config


def expand_ica_variations(
    config: CLExperimentConfig,
) -> List[Tuple[str, Dict[str, Any]]]:
    """Expand ICA variations into (method_key, kwargs) pairs.

    Each ICA variation becomes a distinct method in the experiment matrix.

    Args:
        config: Parsed experiment configuration.

    Returns:
        List of (method_key, method_kwargs) tuples.
    """
    variations = []
    for var_name, var_params in config.ica_variations.items():
        kwargs = {
            "mask_mode": var_params.get("mask_mode", "lesion"),
            "anti_drift": var_params.get("anti_drift", False),
            "ica_components": config.ica_components,
            "ica_percentile": config.ica_percentile,
        }
        if config.ica_template_path:
            kwargs["ica_template_path"] = config.ica_template_path
        variations.append((var_name, kwargs))
    return variations


def config_to_model_config(config: CLExperimentConfig) -> Dict[str, Any]:
    """Convert CLExperimentConfig to the flat MODEL_CONFIGS dict format.

    Returns:
        Dict suitable for use as a MODEL_CONFIGS entry.
    """
    return {
        "model_name": config.model_name,
        "learning_rate": config.learning_rate,
        "batch_size": config.batch_size,
        "num_steps_per_task": config.num_steps_per_task,
        "lora_r": config.lora_r,
        "lora_alpha": config.lora_alpha,
        "lora_dropout": config.lora_dropout,
        "lora_target_modules": config.lora_target_modules,
    }
