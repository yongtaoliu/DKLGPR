"""
Pairwise Gaussian Process for Deep Kernel Learning.

Pairwise GP learns from preference data (A > B) rather than absolute values.
Supports ties, confidence weighting, and multiple feature extractors.
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from gpytorch.kernels import RBFKernel, ScaleKernel
from botorch.models.pairwise_gp import PairwiseGP
from botorch.models.transforms import Normalize
from botorch.models.pairwise_gp import PairwiseLaplaceMarginalLogLikelihood
from botorch.acquisition.preference import AnalyticExpectedUtilityOfBestOption
import random

from .models import get_feature_extractor


# ============================================================================
# Confidence-Weighted MLL Classes (from your original code)
# ============================================================================

class ConfidenceWeightedMLL(nn.Module):
    """
    Marginal log likelihood with confidence weighting.
    """
    
    def __init__(self, likelihood, model, confidence_weights):
        super().__init__()
        self.likelihood = likelihood
        self.model = model
        
        if confidence_weights.dtype != torch.float64:
            confidence_weights = confidence_weights.double()
        self.confidence_weights = confidence_weights
        
        # Normalize weights
        if confidence_weights.sum() > 0:
            self.normalized_weights = confidence_weights / confidence_weights.sum() * len(confidence_weights)
        else:
            self.normalized_weights = confidence_weights
    
    def forward(self, output, target):
        """
        Compute weighted marginal log likelihood.
        """
        mean = output.mean
        comparisons = target
        
        total_weighted_ll = torch.tensor(0.0, dtype=torch.float64, device=mean.device)
        
        for i in range(len(comparisons)):
            winner_idx = comparisons[i, 0].long()
            loser_idx = comparisons[i, 1].long()
            
            # Utility difference
            mean_diff = mean[winner_idx] - mean[loser_idx]
            
            # Variance of difference
            var_winner = output.variance[winner_idx]
            var_loser = output.variance[loser_idx]
            var_diff = var_winner + var_loser
            
            # Log probability
            std_diff = torch.sqrt(var_diff + 1e-6)
            z_score = mean_diff / std_diff
            
            # Normal CDF
            normal_cdf = 0.5 * (1 + torch.erf(z_score / torch.sqrt(torch.tensor(2.0, dtype=torch.float64))))
            log_prob = torch.log(normal_cdf + 1e-8)
            
            # Weight by confidence (all in float64)
            confidence = self.normalized_weights[i]
            weighted_log_prob = confidence * log_prob
            
            total_weighted_ll = total_weighted_ll + weighted_log_prob
        
        return total_weighted_ll


class ConfidenceWeightedMLLWithTies(nn.Module):
    """
    Marginal log likelihood with confidence weighting and tie support.
    """
    
    def __init__(self, likelihood, model, confidence_weights, tolerance=0.1):
        super().__init__()
        self.likelihood = likelihood
        self.model = model
        self.tolerance = tolerance
        
        if confidence_weights.dtype != torch.float64:
            confidence_weights = confidence_weights.double()
        self.confidence_weights = confidence_weights
        
        # Normalize weights
        if confidence_weights.sum() > 0:
            self.normalized_weights = confidence_weights / confidence_weights.sum() * len(confidence_weights)
        else:
            self.normalized_weights = confidence_weights
    
    def forward(self, output, comparisons):
        """
        Compute weighted marginal log likelihood with tie support.
        
        Parameters
        ----------
        output : gpytorch posterior
            GP posterior
        comparisons : torch.Tensor
            Comparisons with types, shape (n_comparisons, 3)
            [:, 0] = first point index
            [:, 1] = second point index
            [:, 2] = type (0=first>second, 1=second>first, 2=equal)
        """
        mean = output.mean
        variance = output.variance
        
        total_weighted_ll = torch.tensor(0.0, dtype=torch.float64, device=mean.device)
        
        for i in range(len(comparisons)):
            idx_a = comparisons[i, 0].long()
            idx_b = comparisons[i, 1].long()
            comp_type = comparisons[i, 2].long()
            
            # Utility difference
            mean_diff = mean[idx_a] - mean[idx_b]
            
            # Variance of difference
            var_diff = variance[idx_a] + variance[idx_b]
            std_diff = torch.sqrt(var_diff + 1e-6)
            
            if comp_type == 0:  # A > B
                # P(A > B) = Φ((mean_diff - tolerance) / std_diff)
                z_score = (mean_diff - self.tolerance) / std_diff
                normal_cdf = 0.5 * (1 + torch.erf(z_score / torch.sqrt(torch.tensor(2.0, dtype=torch.float64))))
                log_prob = torch.log(normal_cdf + 1e-8)
                
            elif comp_type == 1:  # B > A
                # P(B > A) = Φ((-mean_diff - tolerance) / std_diff)
                z_score = (-mean_diff - self.tolerance) / std_diff
                normal_cdf = 0.5 * (1 + torch.erf(z_score / torch.sqrt(torch.tensor(2.0, dtype=torch.float64))))
                log_prob = torch.log(normal_cdf + 1e-8)
                
            elif comp_type == 2:  # A ≈ B (equal/tie)
                # P(|diff| < tolerance) = Φ((tolerance - |mean_diff|) / std_diff) - Φ((-tolerance - |mean_diff|) / std_diff)
                abs_diff = torch.abs(mean_diff)
                z_upper = (self.tolerance - abs_diff) / std_diff
                z_lower = (-self.tolerance - abs_diff) / std_diff
                
                prob_upper = 0.5 * (1 + torch.erf(z_upper / torch.sqrt(torch.tensor(2.0, dtype=torch.float64))))
                prob_lower = 0.5 * (1 + torch.erf(z_lower / torch.sqrt(torch.tensor(2.0, dtype=torch.float64))))
                
                log_prob = torch.log(prob_upper - prob_lower + 1e-8)
            
            else:
                raise ValueError(f"Unknown comparison type: {comp_type}. Expected 0, 1, or 2.")
            
            # Weight by confidence
            confidence = self.normalized_weights[i]
            weighted_log_prob = confidence * log_prob
            total_weighted_ll = total_weighted_ll + weighted_log_prob
        
        return total_weighted_ll


# ============================================================================
# Deep Kernel Pairwise GP with Flexible Feature Extractors
# ============================================================================

class DeepKernelPairwiseGP(nn.Module):
    """
    PairwiseGP with deep kernel learning and flexible feature extractors.
    
    Supports:
    - Multiple feature extractor types (FC, FCBN, ResNet, Attention, Wide&Deep)
    - Confidence weighting
    - Tie/equal comparisons
    - Custom extractors
    
    Parameters
    ----------
    datapoints : torch.Tensor
        Training inputs, shape (n, input_dim)
    comparisons : torch.Tensor
        Pairwise comparisons (see allow_ties parameter)
    input_dim : int
        Dimensionality of input data
    feature_dim : int
        Dimensionality of learned feature space
    hidden_dims : list of int, optional
        Hidden layer dimensions for feature extractor
    extractor_type : str
        Type of feature extractor: 'fc', 'fcbn', 'resnet', 'attention', 'wide_deep', 'custom'
    extractor_kwargs : dict, optional
        Additional arguments for feature extractor
    confidence_weights : torch.Tensor, optional
        Confidence weights for comparisons
    allow_ties : bool
        If True, comparisons should have shape (n, 3) with type column
    jitter : float
        Jitter for numerical stability
    """
    
    def __init__(
        self,
        datapoints,
        comparisons,
        input_dim,
        feature_dim=16,
        hidden_dims=None,
        extractor_type='fc',
        extractor_kwargs=None,
        confidence_weights=None,
        allow_ties=False,
        jitter=1e-4
    ):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]

        if extractor_kwargs is None:
            extractor_kwargs = {}

        # extractor_type=None → standard GP: identity extractor, GP lives in input space
        if extractor_type is None:
            feature_dim = input_dim

        self.input_dim = input_dim
        self.feature_dim = feature_dim
        self.extractor_type = extractor_type
        self.allow_ties = allow_ties

        # Create feature extractor using factory function
        self.feature_extractor = get_feature_extractor(
            extractor_type=extractor_type,
            input_dim=input_dim,
            feature_dim=feature_dim,
            hidden_dims=hidden_dims,
            **extractor_kwargs
        )

        self.feature_extractor = self.feature_extractor.to(
            device=datapoints.device,
            dtype=datapoints.dtype
        )

        # Extract features
        with torch.no_grad():
            train_features = self.feature_extractor(datapoints)

        covar_module = ScaleKernel(RBFKernel(ard_num_dims=feature_dim))

        # ===== EXTRACT COMPARISONS FOR PAIRWISE GP =====
    
        if comparisons.shape[1] == 3:
            # We have comparison types (3 columns)
            # Extract only strict preferences (type 0 or 1)
            strict_mask = comparisons[:, 2] != 2  # Not ties
            strict_comparisons = comparisons[strict_mask, :2].clone()
            
            # Convert type 1 (second>first) to type 0 (first>second) by swapping
            type_1_mask = comparisons[strict_mask, 2] == 1
            strict_comparisons[type_1_mask] = strict_comparisons[type_1_mask].flip(1)
            
            # Store full comparisons for later use
            self.full_comparisons = comparisons
            self.has_ties = (comparisons[:, 2] == 2).any().item()
        else:
            # Standard 2-column format
            strict_comparisons = comparisons
            self.full_comparisons = comparisons
            self.has_ties = False

        # Initialize PairwiseGP with only strict preferences
        self.gp_model = PairwiseGP(
            datapoints=train_features,
            comparisons=strict_comparisons,  # Only (n, 2) format
            covar_module=covar_module,
            input_transform=Normalize(d=feature_dim),
            jitter=jitter
        )

        self.train_datapoints = datapoints

        # Store confidence weights (for full comparisons)
        if confidence_weights is not None:
            if confidence_weights.dtype != torch.float64:
                confidence_weights = confidence_weights.double()
            self.confidence_weights = confidence_weights.to(datapoints.device)
        else:
            self.confidence_weights = torch.ones(
                len(comparisons),  # Length of FULL comparisons
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
        posterior
            GP posterior distribution
        """
        features = self.feature_extractor(x)
        return self.gp_model.posterior(features)

    def update_gp_data(self):
        """Update GP training data with current features."""
        features = self.feature_extractor(self.train_datapoints)
        self.gp_model.set_train_data(features, strict=False)
    
    def predict_utilities(self, x):
        """
        Predict latent utility values.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor
            
        Returns
        -------
        mean : torch.Tensor
            Predicted utility means
        variance : torch.Tensor
            Predicted variances
        """
        self.eval()
        with torch.no_grad():
            posterior = self(x)
            return posterior.mean, posterior.variance
    
    def predict_preferences(self, x1, x2):
        """
        Predict preference: P(x1 > x2)
        
        Parameters
        ----------
        x1 : torch.Tensor
            First set of inputs
        x2 : torch.Tensor
            Second set of inputs
            
        Returns
        -------
        torch.Tensor
            Probability that x1 is preferred over x2
        """
        self.eval()
        with torch.no_grad():
            mean1, var1 = self.predict_utilities(x1)
            mean2, var2 = self.predict_utilities(x2)
            
            # Compute probability via probit function
            # P(f1 > f2) = Φ((f1 - f2) / √(var1 + var2))
            diff = mean1 - mean2
            var_diff = var1 + var2
            std_diff = torch.sqrt(var_diff + 1e-6)
            
            z = diff / std_diff
            prob = 0.5 * (1 + torch.erf(z / np.sqrt(2.0)))
            
        return prob


# ============================================================================
# Training Functions
# ============================================================================

def train_dkgppw(
    datapoints,
    comparisons,
    input_dim,
    feature_dim=16,
    hidden_dims=None,
    extractor_type='fc',
    extractor_kwargs=None,
    confidence_weights=None,
    use_custom_mll=None,
    allow_ties=False,
    tolerance=0.1,
    num_epochs=1000,
    lr_features=1e-4,
    lr_gp=1e-2,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    verbose=True,
    patience=None,
    min_delta=1e-4
):
    """
    Train Deep Kernel PairwiseGP with flexible MLL selection and tie support.
    
    Parameters
    ----------
    datapoints : np.ndarray or torch.Tensor
        Training datapoints, shape (n, input_dim)
    comparisons : np.ndarray or torch.Tensor
        Pairwise comparisons:
        - If allow_ties=False: shape (m, 2), each row is [winner_idx, loser_idx]
        - If allow_ties=True: shape (m, 3), each row is [idx1, idx2, type]
          where type: 0=idx1>idx2, 1=idx2>idx1, 2=equal
    input_dim : int
        Input dimensionality
    feature_dim : int
        Learned feature dimensionality
    hidden_dims : list of int, optional
        Hidden layer dimensions
    extractor_type : str
        Feature extractor type: 'fc', 'fcbn', 'resnet', 'attention', 'wide_deep', 'custom'
    extractor_kwargs : dict, optional
        Additional arguments for feature extractor
    confidence_weights : np.ndarray or torch.Tensor, optional
        Confidence weights for comparisons, shape (m,)
    use_custom_mll : bool, optional
        If True, use ConfidenceWeightedMLL/ConfidenceWeightedMLLWithTies
        If False, use standard PairwiseLaplaceMarginalLogLikelihood
        If None (default), auto-select based on confidence_weights
    allow_ties : bool
        If True, expect comparisons to have 3 columns with type information
        If False, expect standard 2-column format
    tolerance : float
        Tolerance for equal comparisons (only used if allow_ties=True)
    num_epochs : int
        Number of training epochs
    lr_features : float
        Learning rate for feature extractor
    lr_gp : float
        Learning rate for GP parameters
    device : str
        Device to use
    verbose : bool
        Print training progress
    patience : int, optional
        Early stopping patience
    min_delta : float
        Minimum improvement for early stopping
    
    Returns
    -------
    model : DeepKernelPairwiseGP
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
        datapoints = torch.from_numpy(datapoints).double()
    else:
        datapoints = datapoints.double()

    if not isinstance(comparisons, torch.Tensor):
        comparisons = torch.from_numpy(comparisons).long()
    else:
        comparisons = comparisons.long()

    # Handle comparison format
    if allow_ties:
        if comparisons.shape[1] == 2:
            # Add type column (all type 0)
            types = torch.zeros(len(comparisons), 1, dtype=torch.long, device=comparisons.device)
            comparisons = torch.cat([comparisons, types], dim=1)
            if verbose:
                print("Warning: allow_ties=True but comparisons have 2 columns. Adding type column (all strict).")
        
        # Count comparison types
        n_strict = ((comparisons[:, 2] == 0) | (comparisons[:, 2] == 1)).sum().item()
        n_ties = (comparisons[:, 2] == 2).sum().item()
    else:
        if comparisons.shape[1] == 3:
            if verbose:
                print("Warning: comparisons have 3 columns but allow_ties=False. Using only first 2 columns.")
            comparisons = comparisons[:, :2]
        n_strict = len(comparisons)
        n_ties = 0

    # Confidence weights
    if confidence_weights is not None:
        if not isinstance(confidence_weights, torch.Tensor):
            confidence_weights = torch.from_numpy(confidence_weights).double()  
        else:
            confidence_weights = confidence_weights.double()  
        confidence_weights = confidence_weights.to(device)
        
        if verbose:
            print(f"Confidence weights:")
            print(f"  min: {confidence_weights.min():.3f}, "
                  f"max: {confidence_weights.max():.3f}, "
                  f"mean: {confidence_weights.mean():.3f}")
    else:
        confidence_weights = torch.ones(
            len(comparisons), 
            dtype=torch.float64,  
            device=device
        )

    datapoints = datapoints.to(device)
    comparisons = comparisons.to(device)

    # ===== MLL SELECTION LOGIC =====
    if use_custom_mll is None:
        has_varying_confidence = not torch.allclose(
            confidence_weights, 
            torch.ones_like(confidence_weights)
        )
        has_ties = allow_ties and n_ties > 0
        
        use_custom_mll = has_varying_confidence or has_ties
        
        if verbose:
            if has_ties:
                print("\nAuto-selected: ConfidenceWeightedMLLWithTies (tie support)")
            elif has_varying_confidence:
                print("\nAuto-selected: ConfidenceWeightedMLL (varying confidence)")
            else:
                print("\nAuto-selected: Standard PairwiseLaplaceMarginalLogLikelihood")
    else:
        if verbose:
            if use_custom_mll:
                mll_name = "ConfidenceWeightedMLLWithTies" if (allow_ties and n_ties > 0) else "ConfidenceWeightedMLL"
                print(f"\nUser-selected: {mll_name}")
            else:
                print("\nUser-selected: Standard PairwiseLaplaceMarginalLogLikelihood")

    # ===== CREATE MODEL =====
    model = DeepKernelPairwiseGP(
        datapoints=datapoints,
        comparisons=comparisons,  # Can be (n, 2) or (n, 3)
        input_dim=input_dim,
        feature_dim=feature_dim,
        hidden_dims=hidden_dims,
        extractor_type=extractor_type,
        extractor_kwargs=extractor_kwargs,
        confidence_weights=confidence_weights,
        allow_ties=allow_ties
    ).to(device)

    # Optimizer — skip feature extractor group when it has no parameters (extractor_type=None)
    has_extractor_params = len(list(model.feature_extractor.parameters())) > 0
    param_groups = []
    if has_extractor_params:
        param_groups.append({'params': model.feature_extractor.parameters(), 'lr': lr_features})
    param_groups.append({'params': model.gp_model.parameters(), 'lr': lr_gp})
    optimizer = torch.optim.Adam(param_groups)

    # ===== SELECT MLL =====
    if use_custom_mll:
        if allow_ties and n_ties > 0:
            # Use tie-aware MLL with FULL comparisons
            mll = ConfidenceWeightedMLLWithTies(
                model.gp_model.likelihood,
                model.gp_model,
                confidence_weights,
                tolerance=tolerance
            )
            mll_name = "ConfidenceWeightedMLLWithTies"
            train_with_full_comparisons = True
        else:
            # Standard confidence-weighted MLL
            # Only use weights for strict comparisons
            if allow_ties and comparisons.shape[1] == 3:
                strict_mask = comparisons[:, 2] != 2
                conf_weights_strict = confidence_weights[strict_mask]
            else:
                conf_weights_strict = confidence_weights
                
            mll = ConfidenceWeightedMLL(
                model.gp_model.likelihood,
                model.gp_model,
                conf_weights_strict
            )
            mll_name = "ConfidenceWeightedMLL"
            train_with_full_comparisons = False
    else:
        # Standard BoTorch MLL
        mll = PairwiseLaplaceMarginalLogLikelihood(
            model.gp_model.likelihood,
            model.gp_model
        )
        mll_name = "PairwiseLaplaceMarginalLogLikelihood"
        train_with_full_comparisons = False

    model.train()
    losses = []
    best_loss = float('inf')
    patience_counter = 0

    if verbose:
        if extractor_type is None:
            print(f"\nTraining Standard PairwiseGP (no deep kernel)")
        else:
            print(f"\nTraining Deep Kernel PairwiseGP")
        print("="*60)
        print(f"  Device: {device}")
        print(f"  Extractor type: {extractor_type if extractor_type is not None else 'None (standard GP)'}")
        print(f"  Input dim: {input_dim} → Feature dim: {model.feature_dim}")
        print(f"  Total comparisons: {len(comparisons)}")
        if allow_ties:
            print(f"    Strict: {n_strict}, Ties: {n_ties}")
        print(f"  MLL: {mll_name}")
        if patience:
            print(f"  Early stopping: patience={patience}")
        print("="*60)

    # ===== TRAINING LOOP =====
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        model.update_gp_data()
        output = model.gp_model(*model.gp_model.train_inputs)
        
        # Compute loss
        if train_with_full_comparisons:
            # Pass full comparisons (including ties) to tie-aware MLL
            loss = -mll(output, comparisons)
        else:
            # Pass only strict comparisons to standard MLL
            loss = -mll(output, model.gp_model.train_targets)
        
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
        print("="*60)
        print(f"Training complete! Final loss: {losses[-1]:.4f}")
        print("="*60)
    
    return model, losses


def fit_dkgppw(
    X_train,
    train_comp,
    confidence_weights=None,
    use_custom_mll=None,
    allow_ties=False,
    tolerance=0.1,
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
    patience=None
):
    """
    Fit Deep Kernel PairwiseGP model (high-level interface).

    Parameters
    ----------
    X_train : np.ndarray or torch.Tensor
        High-dimensional features (N, D)
    train_comp : np.ndarray or torch.Tensor
        Pairwise comparisons:
        - If allow_ties=False: shape (M, 2), [winner_idx, loser_idx]
        - If allow_ties=True: shape (M, 3), [idx1, idx2, type]
          where type: 0=idx1>idx2, 1=idx2>idx1, 2=equal
    confidence_weights : np.ndarray or torch.Tensor, optional
        Confidence weights for each comparison, shape (M,)
    use_custom_mll : bool, optional
        Explicitly choose MLL type (None = auto-select)
    allow_ties : bool
        If True, support tie/equal comparisons
    tolerance : float
        Tolerance for considering utilities equal (only if allow_ties=True)
    feature_dim : int
        Dimensionality of learned feature space
    hidden_dims : list of int, optional
        Hidden layer dimensions
    extractor_type : str
        Feature extractor type: 'fc', 'fcbn', 'resnet', 'attention', 'wide_deep'
    extractor_kwargs : dict, optional
        Additional arguments for feature extractor
    num_epochs : int
        Number of training epochs
    lr_features : float
        Learning rate for feature extractor
    lr_gp : float
        Learning rate for GP
    device : str
        Device to use
    verbose : bool
        Print training information
    plot_loss : bool
        Plot training loss
    patience : int, optional
        Early stopping patience

    Returns
    -------
    mll : MarginalLogLikelihood
        Marginal log likelihood object
    pref_model : PairwiseGP
        The GP model (operates in feature space)
    dkl_model : DeepKernelPairwiseGP
        Complete deep kernel model with feature extractor
        
    Examples
    --------
    >>> # Standard comparisons
    >>> model, losses = fit_dkpg(X, comparisons, extractor_type='fcbn')
    
    >>> # With ties and attention extractor
    >>> model, losses = fit_dkpg(
    ...     X, comparisons,
    ...     allow_ties=True,
    ...     extractor_type='attention',
    ...     extractor_kwargs={'hidden_dim': 128, 'num_heads': 4}
    ... )
    """
    if hidden_dims is None:
        hidden_dims = [256, 128, 64]
    
    if extractor_kwargs is None:
        extractor_kwargs = {}
    
    if verbose:
        print("="*60)
        if extractor_type is None:
            print("Training Standard PairwiseGP (no deep kernel)")
        else:
            print("Training Deep Kernel PairwiseGP Model")
            print(f"Feature Extractor: {extractor_type}")
        print("="*60)

    input_dim = X_train.shape[-1]

    dkl_model, losses = train_dkgppw(
        datapoints=X_train,
        comparisons=train_comp,
        input_dim=input_dim,
        feature_dim=feature_dim,
        hidden_dims=hidden_dims,
        extractor_type=extractor_type,
        extractor_kwargs=extractor_kwargs,
        confidence_weights=confidence_weights,
        use_custom_mll=use_custom_mll,
        allow_ties=allow_ties,
        tolerance=tolerance,
        num_epochs=num_epochs,
        lr_features=lr_features,
        lr_gp=lr_gp,
        device=device,
        verbose=verbose,
        patience=patience
    )

    pref_model = dkl_model.gp_model
    
    # Create MLL for reference (matches what was used in training)
    if allow_ties:
        if not isinstance(train_comp, torch.Tensor):
            train_comp_tensor = torch.from_numpy(train_comp).long()
        else:
            train_comp_tensor = train_comp
        
        has_ties = train_comp_tensor.shape[1] == 3 and (train_comp_tensor[:, 2] == 2).any()
        
        if has_ties:
            conf_weights = dkl_model.confidence_weights
            mll = ConfidenceWeightedMLLWithTies(
                pref_model.likelihood,
                pref_model,
                conf_weights,
                tolerance=tolerance
            )
        elif confidence_weights is not None or use_custom_mll:
            conf_weights = dkl_model.confidence_weights
            mll = ConfidenceWeightedMLL(
                pref_model.likelihood,
                pref_model,
                conf_weights
            )
        else:
            mll = PairwiseLaplaceMarginalLogLikelihood(
                pref_model.likelihood,
                pref_model
            )
    elif confidence_weights is not None or use_custom_mll:
        conf_weights = dkl_model.confidence_weights
        mll = ConfidenceWeightedMLL(
            pref_model.likelihood,
            pref_model,
            conf_weights
        )
    else:
        mll = PairwiseLaplaceMarginalLogLikelihood(
            pref_model.likelihood,
            pref_model
        )
        
    if plot_loss and verbose:
        plt.figure(figsize=(10, 5))
        plt.plot(losses, linewidth=2, color='#2E86AB')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Negative MLL', fontsize=12)
        plt.title(f'Training Loss ({extractor_type if extractor_type is not None else "standard GP"})', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.tight_layout()
        plt.show()

    return mll, pref_model, dkl_model


def predict_utility(
    model,
    test_data,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    return_std=False
):
    """
    Predict utility for test data.
    
    Parameters
    ----------
    model : DeepKernelPairwiseGP
        Trained model
    test_data : np.ndarray or torch.Tensor
        Test features, shape (n_test, input_dim)
    device : str
        Device to use
    return_std : bool
        If True, return std instead of variance
    
    Returns
    -------
    mean : np.ndarray
        Predicted utility means, shape (n_test,)
    uncertainty : np.ndarray
        Predicted variance (or std if return_std=True), shape (n_test,)
    """
    if not isinstance(test_data, torch.Tensor):
        test_data = torch.from_numpy(test_data).double()
    else:
        test_data = test_data.double()

    test_data = test_data.to(device)
    model.eval()

    with torch.no_grad():
        mean, variance = model.predict_utilities(test_data)
        mean = mean.cpu().numpy()
        variance = variance.cpu().numpy()

    if return_std:
        return mean, np.sqrt(variance)
    return mean, variance
# ============================================================================
# Acquisition Functions
# ============================================================================
def dkgppw_ucb(dkl_model, X_pool, previous_comparisons=None, 
                            top_k=100, beta=2.0, strategy='max_ucb'):
    """
    Get next comparison pair using UCB (Upper Confidence Bound) acquisition.
    
    UCB balances exploitation (high predicted utility) and exploration (high uncertainty)
    by computing: UCB(x) = mean(x) + beta * std(x)
    
    Parameters
    ----------
    dkl_model : DeepKernelPairwiseGP
        Deep kernel model with feature extractor
    X_pool : torch.Tensor or np.ndarray
        Candidate pool of shape (n_candidates, input_dim)
    previous_comparisons : set of tuples, optional
        Set of (idx1, idx2) tuples representing already-asked comparisons
    top_k : int, optional (default=100)
        Number of candidates to consider (for efficiency)
    beta : float, optional (default=2.0)
        Exploration parameter. Higher beta = more exploration
        - beta=0: Pure exploitation (highest predicted utility)
        - beta→∞: Pure exploration (highest uncertainty)
        - beta=2: Balanced (recommended)
    strategy : str, optional (default='max_ucb')
        Strategy for selecting pairs:
        - 'max_ucb': Compare top UCB vs second top UCB
        - 'max_vs_uncertain': Compare top UCB vs most uncertain
        - 'top_ucb_diverse': Compare top UCB with diverse high-UCB points
    
    Returns
    -------
    best_pair : tuple of (int, int)
        Indices of the two candidates to compare next
    """
    if not isinstance(X_pool, torch.Tensor):
        X_pool = torch.tensor(X_pool, dtype=torch.float64, 
                             device=next(dkl_model.parameters()).device)
    
    n = len(X_pool)
    
    print(f"\nUCB Acquisition (beta={beta}, strategy='{strategy}')")
    
    # Get predictions for all candidates
    with torch.no_grad():
        X_features = dkl_model.feature_extractor(X_pool)
        posterior = dkl_model.gp_model.posterior(X_features)
        means = posterior.mean.squeeze(-1)
        stds = torch.sqrt(posterior.variance.squeeze(-1))
    
    # Compute UCB scores
    ucb_scores = means + beta * stds
    
    # Statistics
    print(f"  Mean utility: {means.mean().item():.4f} ± {means.std().item():.4f}")
    print(f"  Mean uncertainty: {stds.mean().item():.4f} ± {stds.std().item():.4f}")
    print(f"  UCB range: [{ucb_scores.min().item():.4f}, {ucb_scores.max().item():.4f}]")
    
    # Pre-filter to top-k by UCB for efficiency
    top_ucb_idx = torch.argsort(ucb_scores, descending=True)[:min(top_k, n)]
    
    selected_points = None
    best_value = float('-inf')
    evaluated_pairs = 0
    skipped_duplicates = 0
    skipped_previous = 0
    
    if strategy == 'max_ucb':
        # Strategy 1: Compare top UCB vs second top UCB
        # This focuses on finding the best among top candidates
        for i_idx, i in enumerate(top_ucb_idx):
            for j in top_ucb_idx[i_idx + 1:]:
                i_val, j_val = i.item(), j.item()
                
                # Skip identical points
                if torch.allclose(X_pool[i_val], X_pool[j_val], atol=1e-6):
                    skipped_duplicates += 1
                    continue
                
                # Skip previously compared pairs
                if previous_comparisons is not None:
                    if (i_val, j_val) in previous_comparisons or (j_val, i_val) in previous_comparisons:
                        skipped_previous += 1
                        continue
                
                # Score this pair: prefer pairs with high combined UCB
                pair_score = ucb_scores[i_val] + ucb_scores[j_val]
                
                evaluated_pairs += 1
                
                if pair_score > best_value:
                    best_value = pair_score
                    selected_points = (i_val, j_val)
    
    elif strategy == 'max_vs_uncertain':
        # Strategy 2: Compare highest UCB vs most uncertain
        # This validates the top candidate against uncertain ones
        top_idx = top_ucb_idx[0].item()
        uncertain_idx = torch.argsort(stds, descending=True)[:top_k]
        
        for j in uncertain_idx:
            j_val = j.item()
            
            if j_val == top_idx:
                continue
            
            # Skip identical points
            if torch.allclose(X_pool[top_idx], X_pool[j_val], atol=1e-6):
                skipped_duplicates += 1
                continue
            
            # Skip previously compared pairs
            if previous_comparisons is not None:
                if (top_idx, j_val) in previous_comparisons or (j_val, top_idx) in previous_comparisons:
                    skipped_previous += 1
                    continue
            
            # Score: prefer high UCB (exploitation) + high uncertainty (exploration)
            pair_score = ucb_scores[top_idx] + stds[j_val]
            
            evaluated_pairs += 1
            
            if pair_score > best_value:
                best_value = pair_score
                selected_points = (top_idx, j_val)
    
    elif strategy == 'top_ucb_diverse':
        # Strategy 3: Compare top UCB with diverse high-UCB points
        # This explores among promising candidates
        top_idx = top_ucb_idx[0].item()
        
        for j in top_ucb_idx[1:]:
            j_val = j.item()
            
            # Skip identical points
            if torch.allclose(X_pool[top_idx], X_pool[j_val], atol=1e-6):
                skipped_duplicates += 1
                continue
            
            # Skip previously compared pairs
            if previous_comparisons is not None:
                if (top_idx, j_val) in previous_comparisons or (j_val, top_idx) in previous_comparisons:
                    skipped_previous += 1
                    continue
            
            # Score based on UCB difference (want meaningful comparisons)
            ucb_diff = abs(ucb_scores[top_idx] - ucb_scores[j_val])
            # Small difference = informative comparison
            pair_score = 1.0 / (ucb_diff + 0.1)  # Prefer close UCB values
            
            evaluated_pairs += 1
            
            if pair_score > best_value:
                best_value = pair_score
                selected_points = (top_idx, j_val)

    print(f"  Evaluated {evaluated_pairs} pairs")
    print(f"  Skipped {skipped_duplicates} duplicate points")
    print(f"  Skipped {skipped_previous} previously compared pairs")
    
    # Validation
    if selected_points is None:
        raise RuntimeError(
            f"Could not find valid comparison pair!\n"
            f"Evaluated: {evaluated_pairs}, "
            f"Skipped duplicates: {skipped_duplicates}, "
            f"Skipped previous: {skipped_previous}"
        )
    
    if selected_points[0] == selected_points[1]:
        print (f"Same index selected twice! best_pair={selected_points}")
        return [selected_points[0]]
    
    if torch.allclose(X_pool[selected_points[0]], X_pool[selected_points[1]], atol=1e-6):
        print (f"Identical points selected! Indices: {selected_points}")
        return [selected_points[0]]
    
    # Detailed diagnostics
    i, j = selected_points
    print(f"Selected pair: ({i}, {j})")
    
    return selected_points

def dkgppw_eubo (dkl_model, X_pool, previous_comparisons=None, top_k=100):
    """
    Get next point(s) using BoTorch Expected Utility of Best Option acquisition with DKL.

    Parameters
    ----------
    dkl_model : DeepKernelPairwiseGP
        Deep kernel model with feature extractor
    X_pool : torch.Tensor or np.ndarray
        Candidate pool of shape (n_candidates, input_dim)
    previous_comparisons : set of tuples, optional
        Set of (idx1, idx2) tuples representing already-asked comparisons
    top_k : int, optional (default=100)
        Number of most uncertain candidates to consider

    Returns
    -------
    best_pair : tuple of (int, int)
        Indices of the two candidates to compare next
    """
    if not isinstance(X_pool, torch.Tensor):
        X_pool = torch.tensor(X_pool, dtype=torch.float64,  # Use double for consistency
                             device=next(dkl_model.parameters()).device)

    n = len(X_pool)

    # Pre-filter by uncertainty - work in FEATURE space
    with torch.no_grad():
        # Extract features for all candidates
        X_features = dkl_model.feature_extractor(X_pool)  # [n, feature_dim]

        # Get uncertainties in feature space
        posterior = dkl_model.gp_model.posterior(X_features)
        uncertainties = torch.sqrt(posterior.variance.squeeze(-1))

    top_uncertain_idx = torch.argsort(uncertainties, descending=True)[:min(top_k, n)]

    # Create acquisition function on the GP model (which operates in feature space)
    acq = AnalyticExpectedUtilityOfBestOption(pref_model=dkl_model.gp_model)

    best_value = float('-inf')
    selected_points = None
    evaluated_pairs = 0
    skipped_duplicates = 0
    skipped_previous = 0

    for i_idx, i in enumerate(top_uncertain_idx):
        for j in top_uncertain_idx[i_idx + 1:]:
            i_val, j_val = i.item(), j.item()

            # Skip identical points in ORIGINAL space
            if torch.allclose(X_pool[i_val], X_pool[j_val], atol=1e-6):
                skipped_duplicates += 1
                continue

            # Skip previously compared pairs
            if previous_comparisons is not None:
                if (i_val, j_val) in previous_comparisons or (j_val, i_val) in previous_comparisons:
                    skipped_previous += 1
                    continue

            # Extract features for this pair
            with torch.no_grad():
                features_i = X_features[i_val]  # Already computed above
                features_j = X_features[j_val]

            # Stack features for acquisition evaluation [1, 2, feature_dim]
            comparison_pair = torch.stack([features_i, features_j]).unsqueeze(0)

            with torch.no_grad():
                acq_value = acq(comparison_pair).item()

            evaluated_pairs += 1

            if acq_value > best_value:
                best_value = acq_value
                selected_points = (i_val, j_val)

    print(f"Evaluated {evaluated_pairs} pairs")
    print(f"Skipped {skipped_duplicates} duplicate points")
    print(f"Skipped {skipped_previous} previously compared pairs")

    # Validation
    if selected_points is None:
        raise RuntimeError(
            f"Could not find valid comparison pair!\n"
            f"Evaluated: {evaluated_pairs}, "
            f"Skipped duplicates: {skipped_duplicates}, "
            f"Skipped previous: {skipped_previous}"
        )

    if selected_points[0] == selected_points[1]:
        print(f"Same index selected twice: {selected_points}")
        return [selected_points[0]]

    if torch.allclose(X_pool[selected_points[0]], X_pool[selected_points[1]], atol=1e-6):
        print(f"Identical points selected: {selected_points}")
        return [selected_points[0]]

    print(f"Selected: indices {selected_points}")

    return selected_points

def get_user_preference(train_idx1, train_idx2, pair_num=1, total_pairs=1, 
                       confidence_factors=[0.2, 0.75, 1.0], allow_ties=True):
    """
    Get user preference between two options with optional confidence weighting.

    Parameters
    ----------
    train_idx1 : int
        First training index to compare
    train_idx2 : int
        Second training index to compare
    pair_num : int, optional (default=1)
        Current pair number (for display)
    total_pairs : int, optional (default=1)
        Total number of pairs to compare (for display)
    confidence_factors : list of float or None, optional (default=[0.5, 0.75, 1.0])
        Confidence weights for 3 confidence levels:
        - confidence_factors[0]: Slightly prefer (low confidence)
        - confidence_factors[1]: Moderately prefer (medium confidence)
        - confidence_factors[2]: Strongly prefer (high confidence)
        If None, skip confidence collection and always return 1.0
    allow_ties : bool, optional (default=False)
        If True, allow user to say options are equal/tie
        If False, force strict preference
    
    Returns
    -------
    idx1 : int
        First training index
    idx2 : int
        Second training index
    comp_type : int
        Comparison type:
        - 0: idx1 > idx2 (first is better)
        - 1: idx2 > idx1 (second is better)
        - 2: idx1 ≈ idx2 (equal/tie) [only if allow_ties=True]
    confidence : float
        Confidence weight in range [0, 1]
    """
    # Get preference
    while True:
        if allow_ties:
            prompt = (
                f"Pair {pair_num}/{total_pairs} → Which is better?\n"
                f"  0 = First option is better\n"
                f"  1 = Second option is better\n"
                f"  2 = They are tie\n"
                f"Enter 0, 1, or 2: "
            )
        else:
            prompt = f"Pair {pair_num}/{total_pairs} → Which is better? Enter 0 or 1:\n"

        choice = input(prompt).strip()

        if choice == "0":
            comp_type = 0  # train_idx1 > train_idx2
            break
        elif choice == "1":
            comp_type = 1  # train_idx2 > train_idx1
            break
        elif choice == "2" and allow_ties:
            comp_type = 2  # train_idx1 ≈ train_idx2
            break
        else:
            if allow_ties:
                print("Invalid input. Please enter 0, 1, or 2.")
            else:
                print("Invalid input. Please enter 0 or 1.")
    
    # Get confidence if requested
    if confidence_factors is not None:
        while True:
            conf_input = input(
                "How confident are you?\n"
                "  1 = Slightly prefer\n"
                "  2 = Moderately prefer\n"
                "  3 = Strongly prefer\n"
                "Enter 1, 2, or 3: "
            ).strip()
            
            if conf_input == "1":
                confidence = confidence_factors[0]
                break
            elif conf_input == "2":
                confidence = confidence_factors[1]
                break
            elif conf_input == "3":
                confidence = confidence_factors[2]
                break
            else:
                print("Invalid input. Please enter 1, 2, or 3.")
        
    else:
        confidence = 1.0  # Default: fully confident

        # Print summary
    if comp_type == 0:
        print(f"Recorded: {train_idx1} > {train_idx2} (confidence={confidence:.2f})")
    elif comp_type == 1:
        print(f"Recorded: {train_idx2} > {train_idx1} (confidence={confidence:.2f})")
    elif comp_type == 2:
        print(f"Recorded: {train_idx1} ≈ {train_idx2} (equal, confidence={confidence:.2f})")

    return train_idx1, train_idx2, comp_type, confidence

def sample_comparison_pairs(train_indices, n_pairs_per_point=1, best_train_idx=None, seed=None):
    """
    Create comparison pairs for newly added points.

    Strategy:
    - If 2 new points added: compare them with each other + random old points
    - If 1 new point added with known best: compare new vs best + random old points

    Parameters
    ----------
    train_indices : np.ndarray
        All training indices (into full pool)
    n_pairs_per_point : int
        Number of random comparisons per new point
    best_train_idx : int, optional
        Training index of current best point (if known)
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    pairs_log : list of tuples
        Pairs of training indices to compare
    """
    if seed is not None:
        random.seed(seed)

    n_train = len(train_indices)
    last_idx = n_train - 1  # Newest point
    old_indices = list(range(n_train - 2))  # All points except last 2

    pairs_log = []

    if best_train_idx is None:
        # Two new points added: compare them against each other
        second_last_idx = n_train - 2
        pairs_log.append((second_last_idx, last_idx))
        new_points = [second_last_idx, last_idx]
    else:
        # One new point added: compare against known best
        pairs_log.append((best_train_idx, last_idx))
        new_points = [last_idx]

    # Compare each new point against n_pairs random old points
    for new_idx in new_points:
        selected_old = random.sample(old_indices, min(n_pairs_per_point, len(old_indices)))
        for old_idx in selected_old:
            pairs_log.append((old_idx, new_idx))

    return pairs_log

# ============================================================================
# Utility Functions
# ============================================================================
def plot_option(img_full, coords, spectra, v_step, pool_idx, train_idx=None):
    """
    Plot a single option for user comparison.

    Parameters
    ----------
    img_full : np.ndarray
        Full image
    coords : np.ndarray
        All coordinates
    spectra : np.ndarray
        All spectra
    v_step : np.ndarray
        Voltage steps
    pool_idx : tuple
        Indexes in the full pool
    train_idx : int, optional
        Index in training set (if applicable)
    option_label : str
        Label for the option
    """
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3), dpi=80)

    # Image with location markers for comparision
    ax1.imshow(img_full, origin='lower')
    ax1.scatter(coords[pool_idx[0], 1], coords[pool_idx[0], 0], marker='x', s=50, c='r')
    ax1.scatter(coords[pool_idx[1], 1], coords[pool_idx[1], 0], marker='x', s=50, c='k')
    ax1.axis("off")

    # Spectrum
    ax2.plot(v_step, spectra[coords[pool_idx[0], 0], coords[pool_idx[0], 1]], c = 'r', label = f"Opt 0; idx: pool-{pool_idx[0]}, train-{train_idx[0]}")
    ax2.plot(v_step, spectra[coords[pool_idx[1], 0], coords[pool_idx[1], 1]], c = 'k', label = f"Opt 1; idx: pool-{pool_idx[1]}, train-{train_idx[1]}")
    ax2.legend(loc = 1)
    plt.show()
    plt.pause(0.1)
    plt.close()

def plot_predictions(coords, y, coord_train, mean, var, step, total_steps):
    """
    Visualize ground truth, predicted mean, and predicted variance.

    Parameters
    ----------
    coords : np.ndarray
        All candidate coordinates, shape (n_candidates, 2)
    y : np.ndarray
        Ground truth values for all candidates
    coord_train : np.ndarray
        Training set coordinates
    mean : np.ndarray
        Predicted utility means
    var : np.ndarray
        Predicted utility variances
    step : int
        Current exploration step (1-indexed)
    total_steps : int
        Total number of exploration steps
    """
    _, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4), dpi=100)

    # Ground truth
    a = ax1.scatter(coords[:, 1], coords[:, 0], c=y, cmap="viridis")
    plt.colorbar(a, ax=ax1)
    ax1.scatter(coord_train[:, 1], coord_train[:, 0], s=50, c='red', marker='x')
    ax1.set_title(f'Ground Truth (Step {step}/{total_steps})')
    ax1.set_aspect('equal')

    # Predicted utility mean
    b = ax2.scatter(coords[:, 1], coords[:, 0], c=mean, cmap="viridis")
    plt.colorbar(b, ax=ax2)
    ax2.scatter(coord_train[:, 1], coord_train[:, 0], s=50, c='red', marker='x')
    ax2.set_title(f'Utility Mean (Step {step}/{total_steps})')
    ax2.set_aspect('equal')

    # Predicted variance
    c = ax3.scatter(coords[:, 1], coords[:, 0], c=var, cmap="viridis")
    plt.colorbar(c, ax=ax3)
    ax3.scatter(coord_train[:, 1], coord_train[:, 0], s=50, c='red', marker='x')
    ax3.set_title(f'Predicted Variance (Step {step}/{total_steps})')
    ax3.set_aspect('equal')

    plt.tight_layout()
    plt.show()
    plt.pause(0.1)
    plt.close()

def acquire_preference(img_full, train_indices, comparison_pairs, coords, spectra, v_step, 
                       y_groundtruth=None, mode='human', confidence_factors=[0.5, 0.75, 1.0],
                       allow_ties=True, tie_threshold=0.1):
    """
    Display all comparison pairs and acquire user preferences.

    Parameters
    ----------
    img_full : np.ndarray
        Full image
    train_indices : np.ndarray
        Training indices (into full pool)
    comparison_pairs : list of tuples
        Pairs of training indices to compare
    coords : np.ndarray
        All coordinates
    spectra : np.ndarray
        All targets
    v_step : np.ndarray
        Voltage steps
    y_groundtruth: np.ndarray
        Ground Truth for simulating experiment
    mode: str
        Human or simulated mode
    confidence_factors: list
        confidence factors.
    allow_ties : bool, optional (default=False)
        If True, allow tie/equal comparisons
    tie_threshold : float, optional (default=0.1)
        Threshold for simulation (only if allow_ties=True)

    Returns
    -------
    new_comparisons : torch.Tensor
        Comparisons in format [winner_idx, loser_idx], shape (n_pairs, 2)
    """
    new_comparisons = []
    confidence_weights = []

    for pair_idx, (train_idx1, train_idx2) in enumerate(comparison_pairs):
        pool_idx1 = train_indices[train_idx1]
        pool_idx2 = train_indices[train_idx2]

        # Plot both options
        print(f"Comparison {pair_idx + 1}/{len(comparison_pairs)}")

        plot_option(img_full, coords, spectra, v_step,
                   pool_idx=[pool_idx1,pool_idx2], train_idx=[train_idx1,train_idx2])

        # Get preference
        if mode == 'simulated':
            if y_groundtruth is None:
                raise ValueError("Ground truth 'y' required for simulated mode")
            
            idx1, idx2, comp_type, confidence = get_simulated_preference(
                train_idx1, train_idx2, train_indices, y_groundtruth,
                tie_threshold=tie_threshold,
                confidence_factors=confidence_factors,
                allow_ties=allow_ties
            )
        
        elif mode == 'human':
            idx1, idx2, comp_type, confidence = get_user_preference(
                train_idx1, train_idx2,
                pair_num=pair_idx + 1,
                total_pairs=len(comparison_pairs),
                confidence_factors=confidence_factors,
                allow_ties=allow_ties
            )
        
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'human' or 'simulated'.")
        
        new_comparisons.append([idx1, idx2, comp_type])
        confidence_weights.append(confidence)
        print(f"Recorded: train_idx {idx1} > train_idx {idx2}")

    return torch.tensor(new_comparisons, dtype=torch.long), torch.tensor(confidence_weights, dtype=torch.float64)

def get_simulated_preference(train_idx1, train_idx2, train_indices, y_groundtruth, tie_threshold=0.1, 
                            confidence_factors=[0.5, 0.75, 1.0], allow_ties=True):
    """
    Simulate user preference using ground truth.

    Parameters
    ----------
    train_idx1, train_idx2 : int
        Training indices to compare
    train_indices : np.ndarray
        Mapping from training indices to pool indices
    y_groundtruth : np.ndarray
        Ground truth utilities
    tie_threshold : float, optional (default=0.1)
        If |y1 - y2| < tie_threshold, consider it a tie
        Only used if allow_ties=True
    confidence_factors : list of float
        Confidence levels for weak/medium/strong preferences
    allow_ties : bool, optional (default=False)
        If True, can return tie comparisons

    Returns
    -------
    idx1, idx2 : int
        Training indices
    comp_type : int
        0, 1, or 2
    confidence : float
        Confidence weight
    
    """
    pool_idx1 = train_indices[train_idx1]
    pool_idx2 = train_indices[train_idx2]
    
    y1 = y_groundtruth[pool_idx1]
    y2 = y_groundtruth[pool_idx2]
    print (f"y1: {y1}")
    print (f"y2: {y2}")
    utility_diff = abs(y1 - y2)

    # Determine comparison type based on ground truth
    if y1<0.2 and y2<0.2:
        true_confidence = confidence_factors[0]
        if allow_ties and utility_diff < tie_threshold:
            # They're close enough to be equal
            true_comp_type = 2
        elif y1 > y2:
            true_comp_type = 0
        else:
            true_comp_type = 1
    else:
        true_confidence = confidence_factors[2]
        if allow_ties and utility_diff < tie_threshold:
            # They're close enough to be equal
            true_comp_type = 2
        elif y1 > y2:
            true_comp_type = 0
        else:
            true_comp_type = 1
    
    if true_comp_type == 0:
        print(f"{train_idx1} > {train_idx2} (confidence={true_confidence:.2f})")
    elif true_comp_type == 1:
        print(f"{train_idx2} > {train_idx1} (confidence={true_confidence:.2f})")
    elif true_comp_type == 2:
        print(f"{train_idx1} ≈ {train_idx2} (equal, confidence={true_confidence:.2f})")
    
    return train_idx1, train_idx2, true_comp_type, true_confidence