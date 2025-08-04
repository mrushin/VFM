"""VFM - Vision Foundation Model for Multi-Domain Object Detection."""

__version__ = "0.1.0"
__author__ = "Michael Rushin"
__email__ = "michael.r.rushin@gmail.com"

from .models import UnifiedVisionFM
from .data import SyntheticMultiDomainDataset
from .training import VFMTrainer
from .evaluation import VFMEvaluator

__all__ = [
    "UnifiedVisionFM",
    "SyntheticMultiDomainDataset",
    "VFMTrainer", 
    "VFMEvaluator",
]
