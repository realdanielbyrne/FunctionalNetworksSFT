#!/usr/bin/env python3
"""
Experiment A: PEFT-Only Fine-Tuning Script

This script runs Experiment A using the centralized configuration architecture.
It uses common_config.yaml for shared parameters and applies experiment-specific
overrides for PEFT-only training (no ICA masking).

Usage:
    python experiment_a_peft_only/scripts/run_experiment_a.py

    Or via the main runner:
    python experiments/peft_vs_peft-ica/run_experiments.py --experiment a
"""

import os
import sys
import logging
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from run_experiments import run_experiment_a as _run_experiment_a

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def run_experiment_a():
    """
    Run Experiment A: PEFT-only fine-tuning using centralized configuration.
    """
    # Ensure we're in project root
    project_root = Path(__file__).parent.parent.parent.parent.parent
    original_cwd = os.getcwd()
    os.chdir(project_root)

    try:
        return _run_experiment_a()
    finally:
        os.chdir(original_cwd)


if __name__ == "__main__":
    success = run_experiment_a()
    if success:
        print("\nExperiment A completed successfully!")
    else:
        print("\n!!! Experiment A failed!")
        sys.exit(1)
