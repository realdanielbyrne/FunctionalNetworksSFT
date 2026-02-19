"""
Reproducibility infrastructure for continual learning experiments.

Captures environment information (git hash, package versions, hardware)
and dumps resolved configurations for experiment auditing.
"""

import json
import logging
import platform
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def capture_environment_info() -> Dict[str, Any]:
    """Capture full environment information for reproducibility.

    Returns:
        Dict with git hash, Python/package versions, hardware info.
    """
    info: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python_version": platform.python_version(),
        },
    }

    # Git hash
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        info["git_hash"] = result.stdout.strip() if result.returncode == 0 else "unknown"

        result = subprocess.run(
            ["git", "diff", "--stat"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        info["git_dirty"] = bool(result.stdout.strip()) if result.returncode == 0 else None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        info["git_hash"] = "unknown"
        info["git_dirty"] = None

    # Package versions
    packages = {}
    for pkg_name in ["torch", "transformers", "peft", "lm_eval", "datasets", "numpy", "scipy"]:
        try:
            mod = __import__(pkg_name)
            packages[pkg_name] = getattr(mod, "__version__", "unknown")
        except ImportError:
            packages[pkg_name] = "not installed"
    info["packages"] = packages

    # CUDA info
    try:
        import torch

        info["cuda"] = {
            "available": torch.cuda.is_available(),
            "version": torch.version.cuda if torch.cuda.is_available() else None,
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        }
        if torch.cuda.is_available():
            devices = []
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                devices.append({
                    "name": props.name,
                    "total_memory_gb": round(props.total_memory / 1e9, 1),
                })
            info["cuda"]["devices"] = devices
    except ImportError:
        info["cuda"] = {"available": False}

    return info


def get_git_hash() -> str:
    """Get the current git commit hash.

    Returns:
        Short git hash string, or "unknown" if not in a git repo.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return "unknown"


def dump_full_config(config: Any, output_path: Path) -> None:
    """Dump the fully resolved experiment configuration as JSON.

    Args:
        config: Configuration object (dataclass or dict).
        output_path: Path to write the JSON file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if hasattr(config, "__dataclass_fields__"):
        import dataclasses
        config_dict = dataclasses.asdict(config)
    elif isinstance(config, dict):
        config_dict = config
    else:
        config_dict = {"config": str(config)}

    config_dict["_dumped_at"] = datetime.now().isoformat()

    with open(output_path, "w") as f:
        json.dump(config_dict, f, indent=2, default=str)

    logger.info(f"Config dumped to {output_path}")


def save_environment(output_dir: Path) -> str:
    """Save environment info to output directory.

    Args:
        output_dir: Directory to save environment.json.

    Returns:
        The git hash string.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    env_info = capture_environment_info()
    env_path = output_dir / "environment.json"
    with open(env_path, "w") as f:
        json.dump(env_info, f, indent=2)

    logger.info(f"Environment info saved to {env_path}")
    return env_info.get("git_hash", "unknown")
