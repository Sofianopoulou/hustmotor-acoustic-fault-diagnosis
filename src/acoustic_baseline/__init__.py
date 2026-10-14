"""Acoustic-only HUSTmotor domain-generalization baseline."""

from .model import AcousticCNN1D
from .training import set_seed, paper_lr, train_model
from .evaluation import evaluate

__all__ = ["AcousticCNN1D", "set_seed", "paper_lr", "train_model", "evaluate"]
