"""
Gaussian Process Classification with Deep Kernel Learning.
"""
import torch
import torch.nn as nn
import numpy as np
import gpytorch
import matplotlib.pyplot as plt
from gpytorch.likelihoods import BernoulliLikelihood, SoftmaxLikelihood
from gpytorch.mlls import VariationalELBO
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy

from .models import get_feature_extractor


# ============================================================================
# Confidence-Weighted ELBO for Classification
# ============================================================================

class ConfidenceWeightedELBO(nn.Module):
    """
    Variational ELBO with confidence weighting for classification.
    
    Parameters
    ----------
    likelihood : gpytorch.likelihoods.Likelihood
        GP likelihood (BernoulliLikelihood or SoftmaxLikelihood)
    model : ApproximateGP
        GP model
    num_data : int
        Total number of data points
    confidence_weights : torch.Tensor
        Confidence weights for each data point, shape (n,)
    beta : float
        Scaling factor for KL divergence (default: 1.0)
    """
    
    def __init__(self, likelihood, model, num_data, confidence_weights, beta=1.0):
        super().__init__()
        self.likelihood = likelihood
        self.model = model
        self.num_data = num_data
        self.beta = beta
        
        if confidence_weights.dtype != torch.float32:
            confidence_weights = confidence_weights.float()
        self.confidence_weights = confidence_weights
        
        # Normalize weights to maintain scale
        if confidence_weights.sum() > 0:
            self.normalized_weights = (
                confidence_weights / confidence_weights.sum() * len(confidence_weights)
            )
        else:
            self.normalized_weights = confidence_weights
    
    def forward(self, variational_dist_f, target, **kwargs):
        """
        Compute weighted ELBO.
        
        Parameters
        ----------
        variational_dist_f : gpytorch.distributions.MultivariateNormal
            Variational distribution over function values
        target : torch.Tensor
            Target labels
            
        Returns
        -------
        torch.Tensor
            Weighted ELBO (scalar)
        """
        # Expected log likelihood
        log_likelihood = self.likelihood.expected_log_prob(target, variational_dist_f).sum(-1)
        
        # Weight by confidence
        weighted_log_likelihood = (self.normalized_weights * log_likelihood).sum()
        
        # KL divergence
        kl_divergence = self.model.variational_strategy.kl_divergence().sum()
        
        # ELBO = E[log p(y|f)] - β * KL[q(f)||p(f)]
        # Scale by num_data for mini-batch training
        elbo = weighted_log_likelihood - self.beta * kl_divergence / self.num_data
        
        return elbo


# ============================================================================
# GP Classification Models
# ============================================================================

class BinaryGPClassificationModel(ApproximateGP):
    """
    Variational GP for binary classification.
    
    Parameters
    ----------
    train_x : torch.Tensor
        Training features in feature space
    train_y : torch.Tensor
        Training labels (0 or 1)
    feature_dim : int
        Feature dimensionality
    num_inducing : int
        Number of inducing points
    """
    
    def __init__(self, train_x, train_y, feature_dim, num_inducing=100):
        # Inducing points
        inducing_points = train_x[:num_inducing].clone()
        
        # Variational distribution
        variational_distribution = CholeskyVariationalDistribution(
            num_inducing_points=inducing_points.size(0)
        )
        
        # Variational strategy
        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True
        )
        
        super().__init__(variational_strategy)
        
        # Mean and covariance
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = ScaleKernel(RBFKernel(ard_num_dims=feature_dim))
        
        # Likelihood
        self.likelihood = BernoulliLikelihood()
        
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class MultiClassGPClassificationModel(ApproximateGP):
    """
    Variational GP for multi-class classification.
    
    Parameters
    ----------
    train_x : torch.Tensor
        Training features in feature space
    train_y : torch.Tensor
        Training labels
    feature_dim : int
        Feature dimensionality
    num_classes : int
        Number of classes
    num_inducing : int
        Number of inducing points
    """
    
    def __init__(self, train_x, train_y, feature_dim, num_classes, num_inducing=100):
        self.num_classes = num_classes
        
        # Inducing points
        inducing_points = train_x[:num_inducing].clone()
        
        # Variational distribution
        variational_distribution = CholeskyVariationalDistribution(
            num_inducing_points=inducing_points.size(0),
            batch_shape=torch.Size([num_classes])
        )
        
        # Variational strategy
        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True
        )
        
        super().__init__(variational_strategy)
        
        # Mean and covariance (independent for each class)
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([num_classes]))
        self.covar_module = ScaleKernel(
            RBFKernel(ard_num_dims=feature_dim, batch_shape=torch.Size([num_classes])),
            batch_shape=torch.Size([num_classes])
        )
        
        # Likelihood
        self.likelihood = SoftmaxLikelihood(num_classes=num_classes, mixing_weights=False)
        
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


# ============================================================================
# Deep Kernel GP Classifier
# ============================================================================

class DKGPC(nn.Module):
    """
    Deep Kernel Learning for GP Classification.
    
    Supports both binary and multi-class classification.
    
    Parameters
    ----------
    datapoints : torch.Tensor
        Training inputs, shape (n, input_dim)
    targets : torch.Tensor
        Training labels, shape (n,) with values in {0, 1, ..., num_classes-1}
    input_dim : int
        Dimensionality of input data
    num_classes : int
        Number of classes (2 for binary)
    feature_dim : int
        Dimensionality of learned feature space
    hidden_dims : list of int, optional
        Hidden layer dimensions for feature extractor
    extractor_type : str
        Type of feature extractor
    extractor_kwargs : dict, optional
        Additional arguments for feature extractor
    num_inducing : int
        Number of inducing points for variational GP
    dropout : float
        Dropout rate for feature extractor
    """
    
    def __init__(
        self,
        datapoints,
        targets,
        input_dim,
        num_classes=2,
        feature_dim=16,
        hidden_dims=None,
        extractor_type='fcbn',
        extractor_kwargs=None,
        num_inducing=100,
        dropout=0.2
    ):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]
        
        if extractor_kwargs is None:
            extractor_kwargs = {}
        
        self.num_classes = num_classes
        self.input_dim = input_dim
        self.feature_dim = feature_dim
        self.extractor_type = extractor_type
        
        # Feature extractor
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
            dtype=torch.float32
        )
        
        # Extract initial features
        with torch.no_grad():
            train_features = self.feature_extractor(datapoints)
        
        # Initialize GP model based on number of classes
        if num_classes == 2:
            # Binary classification
            self.gp_model = BinaryGPClassificationModel(
                train_features,
                targets.long(),
                feature_dim,
                num_inducing
            )
        else:
            # Multi-class classification
            self.gp_model = MultiClassGPClassificationModel(
                train_features,
                targets.long(),
                feature_dim,
                num_classes,
                num_inducing
            )
        
        self.train_datapoints = datapoints
        self.train_targets = targets
        
        # Store confidence weights (for classification, default to ones)
        self.confidence_weights = torch.ones(
            len(datapoints),
            dtype=torch.float32,
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
        output
            GP output (distribution over classes)
        """
        features = self.feature_extractor(x)
        return self.gp_model(features)
    
    def predict_proba(self, x):
        """
        Predict class probabilities.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor, shape (batch_size, input_dim)
            
        Returns
        -------
        probabilities : torch.Tensor
            Class probabilities, shape (batch_size, num_classes)
        """
        self.eval()
        with torch.no_grad():
            output = self(x)
            
            if self.num_classes == 2:
                # Binary classification
                prob_1 = torch.sigmoid(output.mean)
                prob_0 = 1 - prob_1
                probs = torch.stack([prob_0, prob_1], dim=-1)
            else:
                # Multi-class classification
                probs = torch.softmax(output.mean, dim=0).T
            
        return probs
    
    def predict(self, x):
        """
        Predict class labels.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor
            
        Returns
        -------
        predictions : torch.Tensor
            Predicted class labels
        """
        probs = self.predict_proba(x)
        # For multi-class, argmax along class dimension (dim=0 after transpose)
        if probs.dim() > 1 and probs.shape[0] != x.shape[0]:
            return torch.argmax(probs, dim=0)
        return torch.argmax(probs, dim=-1)
    
    def update_gp_data(self):
        """Update GP training data with current features."""
        features = self.feature_extractor(self.train_datapoints)
        self.gp_model.set_train_data(features, self.train_targets, strict=False)


# ============================================================================
# Training Functions
# ============================================================================

def train_dkgpc(
    datapoints,
    targets,
    input_dim,
    num_classes=2,
    feature_dim=16,
    hidden_dims=None,
    extractor_type='fcbn',
    extractor_kwargs=None,
    confidence_weights=None,
    use_confidence_weighted=None,
    num_inducing=100,
    num_epochs=1000,
    lr_features=1e-4,
    lr_gp=1e-2,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    verbose=True,
    patience=None,
    min_delta=1e-4
):
    """
    Train Deep Kernel GP Classifier.
    
    Parameters
    ----------
    datapoints : np.ndarray or torch.Tensor
        Training inputs, shape (n, input_dim)
    targets : np.ndarray or torch.Tensor
        Training labels, shape (n,) with integer class labels
    input_dim : int
        Input dimensionality
    num_classes : int
        Number of classes
    feature_dim : int
        Learned feature dimensionality
    hidden_dims : list of int, optional
        Hidden layer dimensions
    extractor_type : str
        Feature extractor type ('fc', 'fcbn', 'resnet', 'attention', 'wide_deep', 'custom')
    extractor_kwargs : dict, optional
        Additional arguments for feature extractor
    confidence_weights : np.ndarray or torch.Tensor, optional
        Confidence weights for data points, shape (n,)
    use_confidence_weighted : bool, optional
        If True, use ConfidenceWeightedELBO
        If False, use standard VariationalELBO
        If None (default), auto-select based on confidence_weights
    num_inducing : int
        Number of inducing points
    num_epochs : int
        Training epochs
    lr_features : float
        Learning rate for feature extractor
    lr_gp : float
        Learning rate for GP
    device : str
        Device to use
    verbose : bool
        Print progress
    patience : int, optional
        Early stopping patience
    min_delta : float
        Minimum improvement for early stopping
    
    Returns
    -------
    model : DeepKernelGPClassifier
        Trained model
    losses : list
        Training losses
    """
    if hidden_dims is None:
        hidden_dims = [256, 128, 64]
    
    if extractor_kwargs is None:
        extractor_kwargs = {}
    
    # Convert to tensors
    if not isinstance(datapoints, torch.Tensor):
        datapoints = torch.from_numpy(datapoints).float()
    else:
        datapoints = datapoints.float()
    
    if not isinstance(targets, torch.Tensor):
        targets = torch.from_numpy(targets).long()
    else:
        targets = targets.long()
    
    # Handle confidence weights
    if confidence_weights is not None:
        if not isinstance(confidence_weights, torch.Tensor):
            confidence_weights = torch.from_numpy(confidence_weights).float()
        else:
            confidence_weights = confidence_weights.float()
        confidence_weights = confidence_weights.to(device)
        
        if verbose:
            print(f"Confidence weights statistics:")
            print(f"  Min: {confidence_weights.min():.3f}")
            print(f"  Max: {confidence_weights.max():.3f}")
            print(f"  Mean: {confidence_weights.mean():.3f}")
            print(f"  Std: {confidence_weights.std():.3f}")
    else:
        confidence_weights = torch.ones(
            len(datapoints),
            dtype=torch.float32,
            device=device
        )
    
    datapoints = datapoints.to(device)
    targets = targets.to(device)
    
    # ELBO selection logic
    if use_confidence_weighted is None:
        has_varying_confidence = not torch.allclose(
            confidence_weights,
            torch.ones_like(confidence_weights)
        )
        use_confidence_weighted = has_varying_confidence
        
        if verbose:
            if has_varying_confidence:
                print("\nAuto-selected: ConfidenceWeightedELBO")
            else:
                print("\nAuto-selected: VariationalELBO")
    else:
        if verbose:
            if use_confidence_weighted:
                print("\nUser-selected: ConfidenceWeightedELBO")
            else:
                print("\nUser-selected: VariationalELBO")
    
    # Create model
    model = DeepKernelGPClassifier(
        datapoints=datapoints,
        targets=targets,
        input_dim=input_dim,
        num_classes=num_classes,
        feature_dim=feature_dim,
        hidden_dims=hidden_dims,
        extractor_type=extractor_type,
        extractor_kwargs=extractor_kwargs,
        num_inducing=num_inducing
    ).to(device)
    
    # Store confidence weights in model
    model.confidence_weights = confidence_weights
    model.register_buffer('_confidence_weights', confidence_weights)
    
    # Optimizer
    optimizer = torch.optim.Adam([
        {'params': model.feature_extractor.parameters(), 'lr': lr_features},
        {'params': model.gp_model.parameters(), 'lr': lr_gp}
    ])
    
    # Select ELBO
    if use_confidence_weighted:
        mll = ConfidenceWeightedELBO(
            model.gp_model.likelihood,
            model.gp_model,
            num_data=len(targets),
            confidence_weights=confidence_weights
        )
        mll_name = "ConfidenceWeightedELBO"
    else:
        mll = VariationalELBO(
            model.gp_model.likelihood,
            model.gp_model,
            num_data=len(targets)
        )
        mll_name = "VariationalELBO"
    
    model.train()
    losses = []
    
    # Early stopping
    best_loss = float('inf')
    patience_counter = 0
    
    if verbose:
        print(f"\nTraining Deep Kernel GP Classifier")
        print("=" * 60)
        print(f"  Device: {device}")
        print(f"  Extractor type: {extractor_type}")
        print(f"  Input dim: {input_dim} → Feature dim: {feature_dim}")
        print(f"  Classes: {num_classes}")
        print(f"  Samples: {len(datapoints)}")
        print(f"  Inducing points: {num_inducing}")
        print(f"  ELBO: {mll_name}")
        if patience:
            print(f"  Early stopping: patience={patience}")
        print("=" * 60)
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        # Forward pass
        output = model(datapoints)
        
        # Compute loss
        loss = -mll(output, targets)

        # Ensure scalar
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


def fit_dkgpc(
    X_train,
    y_train,
    num_classes=None,
    feature_dim=16,
    hidden_dims=None,
    extractor_type='fcbn',
    extractor_kwargs=None,
    confidence_weights=None,
    use_confidence_weighted=None,
    num_inducing=100,
    num_epochs=1000,
    lr_features=1e-4,
    lr_gp=1e-2,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    verbose=True,
    plot_loss=True,
    patience=None
):
    """
    Fit Deep Kernel GP Classifier (high-level interface).
    
    Parameters
    ----------
    X_train : np.ndarray or torch.Tensor
        Training features, shape (N, D)
    y_train : np.ndarray or torch.Tensor
        Training labels, shape (N,)
    num_classes : int, optional
        Number of classes (auto-detected if None)
    feature_dim : int
        Feature space dimension
    hidden_dims : list of int, optional
        Hidden layer dimensions
    extractor_type : str
        Feature extractor type: 'fc', 'fcbn', 'resnet', 'attention', 'wide_deep', 'custom'
    extractor_kwargs : dict, optional
        Additional arguments for feature extractor
    confidence_weights : np.ndarray or torch.Tensor, optional
        Confidence weights for data points, shape (N,)
    use_confidence_weighted : bool, optional
        Explicitly choose ELBO type (None = auto-select)
    num_inducing : int
        Number of inducing points
    num_epochs : int
        Training epochs
    lr_features : float
        Learning rate for features
    lr_gp : float
        Learning rate for GP
    device : str
        Device to use
    verbose : bool
        Print progress
    plot_loss : bool
        Plot training loss
    patience : int, optional
        Early stopping patience
    
    Returns
    -------
    model : DeepKernelGPClassifier
        Trained model
    losses : list
        Training losses
        
    Examples
    --------
    >>> # Default extractor (FC + BatchNorm)
    >>> model, losses = fit_dkgp_classifier(X, y, num_classes=4)
    
    >>> # With confidence weights
    >>> weights = np.array([1.0, 0.8, 1.0, 0.5, ...])  # Lower weight for noisy samples
    >>> model, losses = fit_dkgp_classifier(X, y, confidence_weights=weights)
    
    >>> # ResNet extractor with confidence
    >>> model, losses = fit_dkgp_classifier(X, y, extractor_type='resnet',
    ...                                     confidence_weights=weights,
    ...                                     extractor_kwargs={'hidden_dim': 256, 'num_blocks': 3})
    """
    if hidden_dims is None:
        hidden_dims = [256, 128, 64]
    
    if extractor_kwargs is None:
        extractor_kwargs = {}
    
    # Auto-detect number of classes
    if num_classes is None:
        if isinstance(y_train, torch.Tensor):
            num_classes = len(torch.unique(y_train))
        else:
            num_classes = len(np.unique(y_train))
    
    if verbose:
        print("=" * 60)
        print("Training Deep Kernel GP Classifier")
        print(f"Feature Extractor: {extractor_type}")
        print("=" * 60)
    
    input_dim = X_train.shape[-1]
    
    model, losses = train_dkgpc(
        datapoints=X_train,
        targets=y_train,
        input_dim=input_dim,
        num_classes=num_classes,
        feature_dim=feature_dim,
        hidden_dims=hidden_dims,
        extractor_type=extractor_type,
        extractor_kwargs=extractor_kwargs,
        confidence_weights=confidence_weights,
        use_confidence_weighted=use_confidence_weighted,
        num_inducing=num_inducing,
        num_epochs=num_epochs,
        lr_features=lr_features,
        lr_gp=lr_gp,
        device=device,
        verbose=verbose,
        patience=patience
    )
    
    if plot_loss and verbose:
        plt.figure(figsize=(10, 5))
        plt.plot(losses, linewidth=2, color='#2E86AB')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Negative ELBO', fontsize=12)
        plt.title(f'Training Loss ({extractor_type} extractor)', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.tight_layout()
        plt.show()
    
    return model, losses


# ============================================================================
# Prediction Functions
# ============================================================================

def predict_dkgpc(
    model,
    test_data,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    return_proba=False,
    batch_size=None
):
    """
    Predict with GP classifier.
    
    Parameters
    ----------
    model : DeepKernelGPClassifier
        Trained model
    test_data : np.ndarray or torch.Tensor
        Test features
    device : str
        Device to use
    return_proba : bool
        If True, return probabilities instead of labels
    batch_size : int, optional
        Batch size for prediction
    
    Returns
    -------
    predictions : np.ndarray
        Predicted labels or probabilities
    """
    if not isinstance(test_data, torch.Tensor):
        test_data = torch.from_numpy(test_data).float()
    else:
        test_data = test_data.float()
    
    test_data = test_data.to(device)
    model.eval()
    
    if batch_size is None:
        # Process all at once
        with torch.no_grad():
            if return_proba:
                predictions = model.predict_proba(test_data).cpu().numpy()
            else:
                predictions = model.predict(test_data).cpu().numpy()
    else:
        # Process in batches
        n_test = len(test_data)
        all_preds = []
        
        with torch.no_grad():
            for i in range(0, n_test, batch_size):
                batch = test_data[i:i+batch_size]
                if return_proba:
                    preds = model.predict_proba(batch).cpu().numpy()
                else:
                    preds = model.predict(batch).cpu().numpy()
                all_preds.append(preds)
        
        predictions = np.concatenate(all_preds, axis=0)
    
    return predictions
