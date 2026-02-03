"""
Gaussian Process Regression with Deep Kernel Learning.
"""
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from botorch.models import SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood

from .models import get_feature_extractor


# ============================================================================
# Confidence-Weighted MLL for Regression
# ============================================================================

class ConfidenceWeightedMLL(nn.Module):
    """
    Marginal log likelihood with confidence weighting for regression.
    
    FIXED: Ensures scalar output for backpropagation.
    
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
            Weighted log likelihood (SCALAR)
        """
        mean = output.mean
        variance = output.variance
        
        # FIX: Ensure everything is 1D to avoid shape issues
        if target.dim() > 1:
            target = target.squeeze()
        if mean.dim() > 1:
            mean = mean.squeeze()
        if variance.dim() > 1:
            variance = variance.squeeze()
        
        # Compute residuals
        residuals = target - mean
        
        # Compute log probabilities
        log_probs = -0.5 * (
            torch.log(2 * torch.pi * variance) + 
            (residuals ** 2) / variance
        )
        
        # Weight by confidence
        weighted_log_probs = self.normalized_weights * log_probs
        
        # FIX: CRITICAL - Always return scalar by summing
        return weighted_log_probs.sum()


# ============================================================================
# Deep Kernel GP Regression Model
# ============================================================================

class DKGPR(nn.Module):
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
    extractor_type : str
        Type of feature extractor ('fc', 'fcbn', 'resnet', 'attention', 'wide_deep', 'custom')
    extractor_kwargs : dict, optional
        Additional arguments for feature extractor
    confidence_weights : torch.Tensor, optional
        Confidence weights for each data point
    noise_constraint : gpytorch.constraints.Constraint, optional
        Constraint on observation noise
    dropout : float
        Dropout rate for feature extractor
        
    Attributes
    ----------
    feature_extractor : nn.Module
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
        extractor_type='fcbn',
        extractor_kwargs=None,
        confidence_weights=None,
        noise_constraint=None,
        dropout=0.2
    ):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]
        
        if extractor_kwargs is None:
            extractor_kwargs = {}

        # Create feature extractor
        self.feature_extractor = get_feature_extractor(
            extractor_type=extractor_type,
            input_dim=input_dim,
            feature_dim=feature_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
            **extractor_kwargs
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

        # Handle target shapes
        if targets.ndim == 1:
            targets_for_gp = targets.unsqueeze(-1)
        elif targets.ndim == 2 and targets.shape[-1] == 1:
            targets_for_gp = targets
        else:
            targets_for_gp = targets.squeeze().unsqueeze(-1)

        # Initialize GP model
        self.gp_model = SingleTaskGP(
            train_X=train_features,
            train_Y=targets_for_gp,
            covar_module=covar_module,
            likelihood=likelihood,
            input_transform=Normalize(d=feature_dim),
            outcome_transform=Standardize(m=1)
        )

        self.train_datapoints = datapoints
        self.train_targets = targets.squeeze() if targets.ndim > 1 else targets
        self.feature_dim = feature_dim
        self.input_dim = input_dim
        self.extractor_type = extractor_type

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
        """Update GP training data with current features."""
        features = self.feature_extractor(self.train_datapoints)
        targets = self.train_targets
        if targets.ndim > 1:
            targets = targets.squeeze()
        targets = targets.unsqueeze(-1)
        self.gp_model.set_train_data(features, targets, strict=False)


# ============================================================================
# Training Functions
# ============================================================================

def train_dkgpr(
    datapoints,
    targets,
    input_dim,
    feature_dim=16,
    hidden_dims=None,
    extractor_type='fcbn',
    extractor_kwargs=None,
    confidence_weights=None,
    use_custom_mll=None,
    num_epochs=1000,
    lr_features=1e-4,
    lr_gp=1e-2,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    verbose=True,
    patience=None,
    min_delta=1e-4
):
    """
    Train Deep Kernel GP for regression.
    
    Parameters
    ----------
    datapoints : np.ndarray or torch.Tensor
        Training datapoints, shape (n, input_dim)
    targets : np.ndarray or torch.Tensor
        Target values, shape (n,) or (n, 1)
    input_dim : int
        Input dimensionality
    feature_dim : int
        Learned feature dimensionality
    hidden_dims : list of int, optional
        Hidden layer dimensions. Default: [256, 128, 64]
    extractor_type : str
        Feature extractor type ('fc', 'fcbn', 'resnet', 'attention', 'wide_deep', 'custom')
    extractor_kwargs : dict, optional
        Additional arguments for feature extractor
    confidence_weights : np.ndarray or torch.Tensor, optional
        Confidence weights for data points, shape (n,)
    use_custom_mll : bool, optional
        If True, use ConfidenceWeightedMLL
        If False, use standard ExactMarginalLogLikelihood
        If None (default), auto-select based on confidence_weights
    num_epochs : int
        Number of training epochs
    lr_features : float
        Learning rate for feature extractor
    lr_gp : float
        Learning rate for GP parameters
    device : str
        Device to use ('cuda' or 'cpu')
    verbose : bool
        Print training progress
    patience : int, optional
        Early stopping patience (epochs without improvement)
    min_delta : float
        Minimum improvement to reset patience counter
    
    Returns
    -------
    model : DeepKernelGP
        Trained model
    losses : list
        Training losses per epoch
    """
    if hidden_dims is None:
        hidden_dims = [256, 128, 64]
    
    if extractor_kwargs is None:
        extractor_kwargs = {}
    
    # Convert to tensors
    if not isinstance(datapoints, torch.Tensor):
        datapoints = torch.from_numpy(datapoints).double()
    else:
        datapoints = datapoints.double()

    if not isinstance(targets, torch.Tensor):
        targets = torch.from_numpy(targets).double()
    else:
        targets = targets.double()

    # Handle confidence weights
    if confidence_weights is not None:
        if not isinstance(confidence_weights, torch.Tensor):
            confidence_weights = torch.from_numpy(confidence_weights).double()  
        else:
            confidence_weights = confidence_weights.double()  
        confidence_weights = confidence_weights.to(device)
    else:
        confidence_weights = torch.ones(
            len(datapoints), 
            dtype=torch.float64,  
            device=device
        )

    datapoints = datapoints.to(device)
    targets = targets.to(device)

    # MLL selection logic
    if use_custom_mll is None:
        has_varying_confidence = not torch.allclose(
            confidence_weights, 
            torch.ones_like(confidence_weights)
        )
        use_custom_mll = has_varying_confidence
        
        if verbose:
            if has_varying_confidence:
                print("\nAuto-selected: ConfidenceWeightedMLL")
            else:
                print("\nAuto-selected: ExactMarginalLogLikelihood")
    else:
        if verbose:
            if use_custom_mll:
                print("\nUser-selected: ConfidenceWeightedMLL")
            else:
                print("\nUser-selected: ExactMarginalLogLikelihood")

    # Create model
    model = DKGPR(
        datapoints=datapoints,
        targets=targets,
        input_dim=input_dim,
        feature_dim=feature_dim,
        hidden_dims=hidden_dims,
        extractor_type=extractor_type,
        extractor_kwargs=extractor_kwargs,
        confidence_weights=confidence_weights
    ).to(device)

    # Optimizer
    optimizer = torch.optim.Adam([
        {'params': model.feature_extractor.parameters(), 'lr': lr_features},
        {'params': model.gp_model.parameters(), 'lr': lr_gp}
    ])

    # Select MLL
    if use_custom_mll:
        mll = ConfidenceWeightedMLL(
            model.gp_model.likelihood,
            model.gp_model,
            confidence_weights
        )
        mll_name = "ConfidenceWeightedMLL"
    else:
        mll = ExactMarginalLogLikelihood(
            model.gp_model.likelihood,
            model.gp_model
        )
        mll_name = "ExactMarginalLogLikelihood"

    model.train()
    losses = []
    best_loss = float('inf')
    patience_counter = 0

    if verbose:
        print(f"\nTraining Deep Kernel GP")
        print("=" * 60)
        print(f"  Device: {device}")
        print(f"  Extractor type: {extractor_type}")
        print(f"  Input dim: {input_dim} → Feature dim: {feature_dim}")
        print(f"  Data points: {len(datapoints)}")
        print(f"  Hidden layers: {hidden_dims}")
        print(f"  MLL: {mll_name}")
        if patience:
            print(f"  Early stopping: patience={patience}, min_delta={min_delta}")
        print("=" * 60)

    # Training loop
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        model.update_gp_data()
        output = model.gp_model(*model.gp_model.train_inputs)
        
        loss = -mll(output, model.gp_model.train_targets)
        
        # Ensure scalar
        if isinstance(loss, torch.Tensor):
            if loss.dim() > 0 or loss.numel() != 1:
                loss = loss.sum()
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        current_loss = loss.item()
        losses.append(current_loss)

        # Early stopping
        if patience is not None:
            if current_loss < best_loss - min_delta:
                best_loss = current_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                if verbose:
                    print(f"\nEarly stopping at epoch {epoch+1}")
                break

        if verbose and (epoch + 1) % 100 == 0:
            print(f"  Epoch {epoch+1:4d}/{num_epochs}, Loss: {current_loss:.4f}")

    model.eval()
    
    if verbose:
        print("=" * 60)
        print(f"Training complete! Final loss: {losses[-1]:.4f}")
        print("=" * 60)
    
    return model, losses


def fit_dkgpr(
    X_train, 
    y_train, 
    confidence_weights=None, 
    use_custom_mll=None, 
    feature_dim=16, 
    hidden_dims=None,
    extractor_type='fcbn',
    extractor_kwargs=None,
    num_epochs=2000, 
    lr_features=1e-4,
    lr_gp=1e-2,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    verbose=True,
    plot_loss=True,
    patience=None,
    min_delta=1e-4
):
    """
    Fit Deep Kernel GP regression model.
    
    This is a high-level interface that handles training and returns all
    necessary objects for prediction and further analysis.

    Parameters
    ----------
    X_train : np.ndarray or torch.Tensor
        High-dimensional features (N, D)
    y_train : np.ndarray or torch.Tensor
        Target values (N,) or (N, 1)
    confidence_weights : np.ndarray or torch.Tensor, optional
        Confidence weights for each data point, shape (N,)
    use_custom_mll : bool, optional
        Explicitly choose MLL type (None = auto-select)
    feature_dim : int
        Dimensionality of learned feature space
    hidden_dims : list of int, optional
        Hidden layer dimensions. Default: [256, 128, 64]
    extractor_type : str
        Feature extractor type: 'fc', 'fcbn', 'resnet', 'attention', 'wide_deep', 'custom'
    extractor_kwargs : dict, optional
        Additional arguments for feature extractor
    num_epochs : int
        Number of training epochs
    lr_features : float
        Learning rate for feature extractor
    lr_gp : float
        Learning rate for GP parameters
    device : str
        Device to use
    verbose : bool
        Print training information
    plot_loss : bool
        Plot training loss curve
    patience : int, optional
        Early stopping patience
    min_delta : float
        Minimum improvement for early stopping

    Returns
    -------
    mll : MarginalLogLikelihood
        Marginal log likelihood object
    gp_model : SingleTaskGP
        The GP model (operates in feature space)
    dkl_model : DeepKernelGP
        Complete deep kernel model with feature extractor
    losses : list
        Training losses
        
    Examples
    --------
    >>> # Default extractor (FC + BatchNorm)
    >>> mll, gp, dkl, losses = fit_dkgp(X, y, feature_dim=16)
    
    >>> # Simple FC extractor
    >>> mll, gp, dkl, losses = fit_dkgp(X, y, extractor_type='fc')
    
    >>> # ResNet extractor
    >>> mll, gp, dkl, losses = fit_dkgp(X, y, extractor_type='resnet',
    ...                                 extractor_kwargs={'hidden_dim': 256, 'num_blocks': 3})
    
    >>> # Custom extractor
    >>> my_net = nn.Sequential(nn.Linear(100, 64), nn.ReLU(), nn.Linear(64, 16))
    >>> mll, gp, dkl, losses = fit_dkgp(X, y, extractor_type='custom',
    ...                                 extractor_kwargs={'custom_extractor': my_net})
    """
    if hidden_dims is None:
        hidden_dims = [256, 128, 64]
    
    if extractor_kwargs is None:
        extractor_kwargs = {}
        
    if verbose:
        print("=" * 60)
        print("Training Deep Kernel GP Regression Model")
        print(f"Feature Extractor: {extractor_type}")
        print("=" * 60)

    input_dim = X_train.shape[-1]

    dkl_model, losses = train_dkgpr(
        datapoints=X_train,
        targets=y_train,
        input_dim=input_dim,
        feature_dim=feature_dim,
        hidden_dims=hidden_dims,
        extractor_type=extractor_type,
        extractor_kwargs=extractor_kwargs,
        confidence_weights=confidence_weights,
        use_custom_mll=use_custom_mll,
        num_epochs=num_epochs,
        lr_features=lr_features,
        lr_gp=lr_gp,
        device=device,
        verbose=verbose,
        patience=patience,
        min_delta=min_delta
    )

    gp_model = dkl_model.gp_model
    
    if confidence_weights is not None or use_custom_mll:
        conf_weights = dkl_model.confidence_weights
        mll = ConfidenceWeightedMLL(
            gp_model.likelihood,
            gp_model,
            conf_weights
        )
    else:
        mll = ExactMarginalLogLikelihood(
            gp_model.likelihood,
            gp_model
        )
        
    if plot_loss and verbose:
        plt.figure(figsize=(10, 5))
        plt.plot(losses, linewidth=2, color='#2E86AB')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Negative Log Likelihood', fontsize=12)
        plt.title(f'Training Loss ({extractor_type} extractor)', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.tight_layout()
        plt.show()

    return mll, gp_model, dkl_model, losses
