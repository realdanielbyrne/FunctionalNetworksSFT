"""
Continual Learning Evaluation Framework for FunctionalNetworksSFT.

This module implements the evaluation methodology from the DOC paper
(Zhang et al., 2025) to compare ICA-based functional network masking
against baseline continual learning methods.

Main components:
- metrics: AA, BWT, FWT metric calculations
- datasets: Dataset configurations, prompts, and loaders
- methods: CL method implementations (LoRA, EWC, LwF, O-LoRA, DOC, ICA Networks)
- evaluation: Main evaluation loop and CLI
- orchestrator: Automated experiment suite with CSV-based resumability
- checkpointing: Per-task checkpoint management for CL sequences
- aggregation: Multi-seed result aggregation and publication tables
"""

from .metrics import ContinualLearningMetrics
from .evaluation import run_continual_learning_evaluation, run_single_task_cycle, main

__all__ = [
    "ContinualLearningMetrics",
    "run_continual_learning_evaluation",
    "run_single_task_cycle",
    "main",
]

