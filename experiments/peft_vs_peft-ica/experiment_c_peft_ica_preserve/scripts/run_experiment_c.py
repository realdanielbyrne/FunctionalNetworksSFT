#!/usr/bin/env python3
"""
Experiment C: PEFT + ICA Masking Fine-Tuning Script (Preserve Mode)

This script runs Experiment C using the centralized configuration architecture.
It uses common_config.yaml for shared parameters and applies experiment-specific
overrides for PEFT + ICA masking in preserve mode.

Usage:
    python experiment_c_peft_ica_preserve/scripts/run_experiment_c.py

    Or via the main runner:
    python experiments/peft_vs_peft-ica/run_experiments.py --experiment c
"""

import os
import sys
import logging
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from run_experiments import run_experiment_c as _run_experiment_c

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def run_experiment_c_preserve():
    """
    Run Experiment C: PEFT + ICA masking (preserve mode) using centralized configuration.
    """
    # Ensure we're in project root
    project_root = Path(__file__).parent.parent.parent.parent.parent
    original_cwd = os.getcwd()
    os.chdir(project_root)

    try:
        return _run_experiment_c()
    finally:
        os.chdir(original_cwd)


if __name__ == "__main__":
    success = run_experiment_c_preserve()
    if success:
        print("\nExperiment C (Preserve Mode) completed successfully!")
    else:
        print("\n!!! Experiment C failed!")
        sys.exit(1)
