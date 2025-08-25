"""
RelativisticAdam: A Physics-Inspired Optimizer for Gradient Explosion Prevention

This package provides PyTorch optimizers that implement relativistic gradient clipping
mechanisms, inspired by the theory of special relativity.
"""

from .optimizer import RelativisticAdam, RelativisticAdamW

__version__ = "1.0.0"
__author__ = "Souradeep Nanda"
__email__ = ""

__all__ = ["RelativisticAdam", "RelativisticAdamW"]
