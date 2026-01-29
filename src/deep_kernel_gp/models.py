"""
Core model classes for Deep Kernel GP.
"""
import torch
import torch.nn as nn
from botorch.models import SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood


class ImageFeatureExtractor(nn.Module):
    """
    Neural network feature extractor for high-dimensional inputs.
    
    Parameters
    ----------
    input_dim : int
        Dimensionality of input data
    feature_dim : int
        Dimensionality of learned feature space
    hidden_dims : list of int
        Hidden layer dimensions
    dropout : float
        Dropout rate
    """
    
    def __init__(self, input_dim, feature_dim=16, hidden_dims=None, dropout=0.2):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, feature_dim))
        self.network = nn.Sequential(*layers)
        
        self.input_dim = input_dim
        self.feature_dim = feature_dim

    def forward(self, x):
        """
        Forward pass through feature extractor.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, input_dim)
            
        Returns
        -------
        torch.Tensor
            Features of shape (batch_size, feature_dim)
        """
        return self.network(x)


class ConfidenceWeightedMLL(nn.Module):
    """
    Marginal log likelihood with confidence weighting for regression.
    
    This allows different data points to have different importance weights,
    useful when some observations are more reliable than others.
    
    Parameters
    ----------
    likelihood : gpytorch.likelihoods.Likelihood
        GP likelihood
    model : gpytorch.models.GP
        GP model
    confidence_weights : torch.Tensor
        Confidence weights for each data point, shape (n,)
    """
    
    def __init__(self, likelihood, model, confidence_weights):
        super().__init__()
        self.likelihood = likelihood
        self.model = model
        
        if confidence_weights.dtype != torch.float64:
            confidence_weights = confidence_weights.double()
        self.confidence_weights = confidence_weights
        
        # Normalize weights to maintain scale
        if confidence_weights.sum() > 0:
            self.normalized_weights = (
                confidence_weights / confidence_weights.sum() * len(confidence_weights)
            )
        else:
            self.normalized_weights = confidence_weights
    
    def forward(self, output, target):
        """
        Compute weighted marginal log likelihood.
        
        Parameters
        ----------
        output : gpytorch.distributions.MultivariateNormal
            GP posterior
        target : torch.Tensor
            Target values
            
        Returns
        -------
        torch.Tensor
            Weighted log likelihood
        """
        mean = output.mean
        variance = output.variance
        
        # Compute residuals
        residuals = target - mean
        
        # Compute log probabilities
        log_probs = -0.5 * (
            torch.log(2 * torch.pi * variance) + 
            (residuals ** 2) / variance
        )
        
        # Weight by confidence
        weighted_log_probs = self.normalized_weights * log_probs
        
        return weighted_log_probs.sum()


class DeepKernelGP(nn.Module):
    """
    Deep Kernel Learning for Gaussian Process Regression.
    
    Combines a neural network feature extractor with a Gaussian Process
    to handle high-dimensional inputs while maintaining GP benefits.
    
    Parameters
    ----------
    datapoints : torch.Tensor
        Training inputs, shape (n, input_dim)
    targets : torch.Tensor
        Training targets, shape (n,) or (n, 1)
    input_dim : int
        Dimensionality of input data
    feature_dim : int
        Dimensionality of learned feature space
    hidden_dims : list of int, optional
        Hidden layer dimensions for feature extractor
    confidence_weights : torch.Tensor, optional
        Confidence weights for each data point
    noise_constraint : gpytorch.constraints.Constraint, optional
        Constraint on observation noise
    dropout : float
        Dropout rate for feature extractor
        
    Attributes
    ----------
    feature_extractor : ImageFeatureExtractor
        Neural network for dimensionality reduction
    gp_model : SingleTaskGP
        Gaussian Process model in feature space
    """
    
    def __init__(
        self,
        datapoints,
        targets,
        input_dim,
        feature_dim=16,
        hidden_dims=None,
        confidence_weights=None,
        noise_constraint=None,
        dropout=0.2
    ):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]

        # Initialize feature extractor
        self.feature_extractor = ImageFeatureExtractor(
            input_dim=input_dim,
            feature_dim=feature_dim,
            hidden_dims=hidden_dims,
            dropout=dropout
        )

        self.feature_extractor = self.feature_extractor.to(
            device=datapoints.device,
            dtype=datapoints.dtype
        )

        # Extract initial features
        with torch.no_grad():
            train_features = self.feature_extractor(datapoints)

        # Set up GP kernel
        covar_module = ScaleKernel(RBFKernel(ard_num_dims=feature_dim))
        
        # Initialize likelihood
        likelihood = GaussianLikelihood()
        if noise_constraint is not None:
            likelihood.noise_covar.register_constraint("raw_noise", noise_constraint)

        # Initialize GP model
        self.gp_model = SingleTaskGP(
            train_X=train_features,
            train_Y=targets.unsqueeze(-1) if targets.ndim == 1 else targets,
            covar_module=covar_module,
            likelihood=likelihood,
            input_transform=Normalize(d=feature_dim),
            outcome_transform=Standardize(m=1)
        )

        self.train_datapoints = datapoints
        self.train_targets = targets
        self.feature_dim = feature_dim
        self.input_dim = input_dim

        # Store confidence weights
        if confidence_weights is not None:
            if confidence_weights.dtype != torch.float64:
                confidence_weights = confidence_weights.double()
            self.confidence_weights = confidence_weights.to(datapoints.device)
        else:
            self.confidence_weights = torch.ones(
                len(datapoints),
                dtype=torch.float64, 
                device=datapoints.device
            )
            
        self.register_buffer('_confidence_weights', self.confidence_weights)

    def forward(self, x):
        """
        Forward pass through the model.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor, shape (batch_size, input_dim)
            
        Returns
        -------
        gpytorch.distributions.MultivariateNormal
            GP posterior distribution
        """
        features = self.feature_extractor(x)
        return self.gp_model(features)

    def update_gp_data(self):
        """
        Update GP training data with current features from feature extractor.
        
        This should be called during training to update the GP with the
        latest learned features.
        """
        features = self.feature_extractor(self.train_datapoints)
        targets = (
            self.train_targets.unsqueeze(-1) 
            if self.train_targets.ndim == 1 
            else self.train_targets
        )
        self.gp_model.set_train_data(features, targets, strict=False)
