"""
Continual Learning Metrics Implementation.
Based on DOC paper (Zhang et al., 2025) methodology.

Metrics:
- Average Accuracy (AA): Mean accuracy across all tasks after training
- Backward Transfer (BWT): Measures forgetting (negative = forgetting occurred)
- Forward Transfer (FWT): Measures transfer to new tasks vs baseline
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


@dataclass
class ContinualLearningMetrics:
    """
    Maintains an accuracy matrix A where A[t][T] = accuracy on task t
    after training on task T.

    Paper notation (1-indexed) vs code (0-indexed):
    - Paper: a_{t,T} = accuracy on task t after training through task T
    - Code: accuracy_matrix[t][T] where t, T are 0-indexed

    Attributes:
        num_tasks: Total number of tasks in the sequence
        task_names: List of task names for reporting
        accuracy_matrix: 2D array storing accuracies [task_evaluated][stage_trained]
        baseline_accuracies: Accuracies from standard fine-tuning (for FWT)
    """

    num_tasks: int
    task_names: List[str] = field(default_factory=list)
    accuracy_matrix: np.ndarray = field(init=False)
    baseline_accuracies: np.ndarray = field(init=False)

    def __post_init__(self):
        self.accuracy_matrix = np.zeros((self.num_tasks, self.num_tasks))
        self.baseline_accuracies = np.zeros(self.num_tasks)
        if not self.task_names:
            self.task_names = [f"task_{i}" for i in range(self.num_tasks)]

    def record_accuracy(
        self, task_idx: int, training_stage: int, accuracy: float
    ) -> None:
        """
        Record accuracy on task `task_idx` after training on task `training_stage`.

        Args:
            task_idx: Index of the task being evaluated (0-indexed)
            training_stage: Index of the task just trained on (0-indexed)
            accuracy: Test accuracy (0-100 scale)
        """
        if not (0 <= task_idx < self.num_tasks):
            raise ValueError(f"task_idx {task_idx} out of range [0, {self.num_tasks})")
        if not (0 <= training_stage < self.num_tasks):
            raise ValueError(
                f"training_stage {training_stage} out of range [0, {self.num_tasks})"
            )
        self.accuracy_matrix[task_idx, training_stage] = accuracy

    def set_baseline_accuracy(self, task_idx: int, accuracy: float) -> None:
        """Set the baseline accuracy for a task (standard fine-tuning without CL)."""
        self.baseline_accuracies[task_idx] = accuracy

    def compute_average_accuracy(self, T: Optional[int] = None) -> float:
        """
        Compute Average Accuracy (AA) after training on T tasks.

        Formula: AA(T) = (1/T) * sum_{t=1}^{T} a_{t,T}

        Args:
            T: Number of tasks trained (1-indexed in paper). If None, uses all tasks.

        Returns:
            Average accuracy across all tasks seen so far (0-100 scale).
        """
        if T is None:
            T = self.num_tasks
        accuracies = self.accuracy_matrix[:T, T - 1]
        return float(np.mean(accuracies))

    def compute_backward_transfer(self, T: Optional[int] = None) -> float:
        """
        Compute Backward Transfer (BWT) after training on T tasks.

        Formula: BWT(T) = (1/(T-1)) * sum_{t=1}^{T-1} (a_{t,T} - a_{t,t})

        Negative BWT indicates forgetting.

        Args:
            T: Number of tasks trained. If None, uses all tasks.

        Returns:
            Backward transfer score.
        """
        if T is None:
            T = self.num_tasks
        if T <= 1:
            return 0.0

        bwt_sum = 0.0
        for t in range(T - 1):
            final_acc = self.accuracy_matrix[t, T - 1]
            initial_acc = self.accuracy_matrix[t, t]
            bwt_sum += final_acc - initial_acc
        return float(bwt_sum / (T - 1))

    def compute_forward_transfer(self, T: Optional[int] = None) -> float:
        """
        Compute Forward Transfer (FWT) after training on T tasks.

        Formula: FWT(T) = (1/(T-1)) * sum_{t=2}^{T} (a_{t,t} - baseline_t)

        Args:
            T: Number of tasks trained. If None, uses all tasks.

        Returns:
            Forward transfer score.
        """
        if T is None:
            T = self.num_tasks
        if T <= 1:
            return 0.0

        fwt_sum = 0.0
        for t in range(1, T):
            cl_acc = self.accuracy_matrix[t, t]
            baseline_acc = self.baseline_accuracies[t]
            fwt_sum += cl_acc - baseline_acc
        return float(fwt_sum / (T - 1))

    def get_per_task_final_accuracy(self) -> Dict[str, float]:
        """Get final accuracy for each task after all training."""
        return {
            name: float(self.accuracy_matrix[i, -1])
            for i, name in enumerate(self.task_names)
        }

    def get_full_report(self) -> Dict:
        """Generate complete evaluation report."""
        T = self.num_tasks
        return {
            "num_tasks": T,
            "task_names": self.task_names,
            "average_accuracy": self.compute_average_accuracy(T),
            "backward_transfer": self.compute_backward_transfer(T),
            "forward_transfer": self.compute_forward_transfer(T),
            "accuracy_matrix": self.accuracy_matrix.tolist(),
            "baseline_accuracies": self.baseline_accuracies.tolist(),
            "per_task_final_accuracy": self.get_per_task_final_accuracy(),
        }

    def save(self, filepath: Path) -> None:
        """Save metrics to JSON file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(self.get_full_report(), f, indent=2)

    @classmethod
    def load(cls, filepath: Path) -> "ContinualLearningMetrics":
        """Load metrics from JSON file."""
        with open(filepath, "r") as f:
            data = json.load(f)

        metrics = cls(num_tasks=data["num_tasks"], task_names=data["task_names"])
        metrics.accuracy_matrix = np.array(data["accuracy_matrix"])
        metrics.baseline_accuracies = np.array(data["baseline_accuracies"])
        return metrics

    def __str__(self) -> str:
        return (
            f"ContinualLearningMetrics(\n"
            f"  AA={self.compute_average_accuracy():.2f}%,\n"
            f"  BWT={self.compute_backward_transfer():.2f},\n"
            f"  FWT={self.compute_forward_transfer():.2f}\n"
            f")"
        )
