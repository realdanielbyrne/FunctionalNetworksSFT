"""
Vanilla LoRA sequential fine-tuning baseline.
No continual learning mechanism - just trains sequentially.
"""

from typing import Any, Dict

import torch

from .base import ContinualLearningMethod


class LoRABaseline(ContinualLearningMethod):
    """
    Vanilla LoRA sequential fine-tuning.

    This is the baseline that demonstrates catastrophic forgetting.
    No regularization or protection of previous task knowledge.
    """

    def compute_loss(
        self, batch: Dict[str, torch.Tensor], task_idx: int
    ) -> torch.Tensor:
        """Standard cross-entropy loss, no regularization."""
        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        return outputs.loss

