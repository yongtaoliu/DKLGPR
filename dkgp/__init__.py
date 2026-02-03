"""
Deep Kernel GP - Deep Kernel Learning for Gaussian Process Regression and Classification
"""

# Feature extractors
from .models import (
    FCFeatureExtractor,
    FCBNFeatureExtractor,
    ResNetFeatureExtractor,
    AttentionFeatureExtractor,
    WideDeepFeatureExtractor,
    get_feature_extractor,
    ImageFeatureExtractor,  # Backward compatibility
)

# Regression
from .gpr import (
    DeepKernelGP,
    ConfidenceWeightedMLL,
    train_dkgp,
    fit_dkgp,
)

# Classification
from .gpc import (
    DeepKernelGPClassifier,
    BinaryGPClassificationModel,
    MultiClassGPClassificationModel,
    ConfidenceWeightedELBO,
    train_dkgp_classifier,
    fit_dkgp_classifier,
    predict_classifier,
)

# Prediction
from .prediction import predict

# Acquisition functions
from .acquisition import (
    expected_improvement,
    upper_confidence_bound,
    probability_of_improvement,
    thompson_sampling,
    expected_improvement_with_constraints,
)

# Submodules - for convenience imports
from . import gpr as dkgpr  # Allow: from dkgp import dkgpr
from . import gpc as dkgpc  # Allow: from dkgp import dkgpc
from . import acquisition
from . import utils
from . import models
from . import prediction

__version__ = "0.2.0"

__all__ = [
    # Feature extractors
    "FCFeatureExtractor",
    "FCBNFeatureExtractor",
    "ResNetFeatureExtractor",
    "AttentionFeatureExtractor",
    "WideDeepFeatureExtractor",
    "get_feature_extractor",
    "ImageFeatureExtractor",
    # Regression
    "DeepKernelGP",
    "ConfidenceWeightedMLL",
    "train_dkgp",
    "fit_dkgp",
    # Classification
    "DeepKernelGPClassifier",
    "BinaryGPClassificationModel",
    "MultiClassGPClassificationModel",
    "ConfidenceWeightedELBO",
    "train_dkgp_classifier",
    "fit_dkgp_classifier",
    "predict_classifier",
    # Prediction
    "predict",
    # Acquisition
    "expected_improvement",
    "upper_confidence_bound",
    "probability_of_improvement",
    "thompson_sampling",
    "expected_improvement_with_constraints",
    # Submodules
    "dkgpr",
    "dkgpc",
    "acquisition",
    "utils",
    "models",
    "prediction",
]
