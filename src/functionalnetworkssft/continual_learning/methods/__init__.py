"""
Continual learning method implementations.

Available methods:
- LoRABaseline: Vanilla LoRA sequential fine-tuning (baseline showing forgetting)
- EWC: Elastic Weight Consolidation (Kirkpatrick et al., 2017)
- LwF: Learning without Forgetting (Li & Hoiem, 2017)
- OLoRA: Orthogonal LoRA (Wang et al., 2023)
- DOC: Dynamic Orthogonal Continual Fine-Tuning (Zhang et al., 2025)
- ICANetworksCL: ICA-based Functional Network method (this project)
"""

from .base import ContinualLearningMethod
from .doc import DOC
from .ewc import EWC
from .ica_networks import ICANetworksCL
from .lora_baseline import LoRABaseline
from .lwf import LwF
from .o_lora import OLoRA

__all__ = [
    "ContinualLearningMethod",
    "LoRABaseline",
    "EWC",
    "LwF",
    "OLoRA",
    "DOC",
    "ICANetworksCL",
]
