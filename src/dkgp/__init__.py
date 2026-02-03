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

# Utilities
from . import utils

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
    "DKGPR",
    "ConfidenceWeightedMLL",
    "train_dkgpr",
    "fit_dkgpr",
    # Classification
    "DKGPC",
    "BinaryGPClassificationModel",
    "MultiClassGPClassificationModel",
    "ConfidenceWeightedELBO",
    "train_dkgpc",
    "fit_dkgpc",
    "predict_dkgpc",
    # Prediction
    "predict",
    # Acquisition
    "expected_improvement",
    "upper_confidence_bound",
    "probability_of_improvement",
    "thompson_sampling",
    "expected_improvement_with_constraints",
    # Utils
    "utils",
]
