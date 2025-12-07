#!/usr/bin/env python3
"""
Experiment B: PEFT + ICA Masking Fine-Tuning Script (Lesion Mode)

This script runs Experiment B using the centralized configuration architecture.
It uses common_config.yaml for shared parameters and applies experiment-specific
overrides for PEFT + ICA masking in lesion mode.

Usage:
    python experiment_b_peft_ica/scripts/run_experiment_b.py

    Or via the main runner:
    python experiments/peft_vs_peft-ica/run_experiments.py --experiment b
"""

import os
import sys
import logging
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from run_experiments import run_experiment_b as _run_experiment_b

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def run_experiment_b():
    """
    Run Experiment B: PEFT + ICA masking (lesion mode) using centralized configuration.
    """
    # Ensure we're in project root
    project_root = Path(__file__).parent.parent.parent.parent.parent
    original_cwd = os.getcwd()
    os.chdir(project_root)

    try:
        return _run_experiment_b()
    finally:
        os.chdir(original_cwd)


if __name__ == "__main__":
    success = run_experiment_b()
    if success:
        print("\nExperiment B (Lesion Mode) completed successfully!")
    else:
        print("\n!!! Experiment B failed!")
        sys.exit(1)
