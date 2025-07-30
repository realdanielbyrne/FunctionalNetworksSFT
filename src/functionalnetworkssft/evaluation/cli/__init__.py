"""
CLI integration for the evaluation framework.

This module provides command-line interface integration for running evaluations
and integrates with the existing FunctionalNetworksSFT CLI system.
"""

from .evaluation_cli import (
    add_evaluation_arguments,
    run_evaluation_from_args,
    main as evaluation_main,
)

__all__ = [
    "add_evaluation_arguments",
    "run_evaluation_from_args", 
    "evaluation_main",
]
