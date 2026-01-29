"""
Deep Kernel GP - Deep Kernel Learning for Gaussian Process Regression
"""

from .models import DeepKernelGP, ImageFeatureExtractor
from .training import train_dkgp, fit_dkgp
from .prediction import predict
from .acquisition import (
    expected_improvement,
    upper_confidence_bound,
    probability_of_improvement,
    thompson_sampling,
    expected_improvement_with_constraints,
)
from . import utils

__version__ = "0.1.0"

__all__ = [
    "DeepKernelGP",
    "ImageFeatureExtractor",
    "train_dkgp",
    "fit_dkgp",
    "predict",
    "expected_improvement",
    "upper_confidence_bound",
    "probability_of_improvement",
    "thompson_sampling",
    "expected_improvement_with_constraints",
    "utils",
]
