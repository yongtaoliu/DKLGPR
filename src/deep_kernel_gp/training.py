"""
Training utilities for Deep Kernel GP models.
"""
import torch
import matplotlib.pyplot as plt
from gpytorch.mlls import ExactMarginalLogLikelihood
from .models import DeepKernelGP, ConfidenceWeightedMLL


def train_dkgp(
    datapoints,
    targets,
    input_dim,
    feature_dim=16,
    hidden_dims=None,
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
        
        if verbose:
            print(f"Confidence weights statistics:")
            print(f"  Min: {confidence_weights.min():.3f}")
            print(f"  Max: {confidence_weights.max():.3f}")
            print(f"  Mean: {confidence_weights.mean():.3f}")
            print(f"  Std: {confidence_weights.std():.3f}")
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
                print("\nAuto-selected: ConfidenceWeightedMLL (varying confidence)")
            else:
                print("\nAuto-selected: ExactMarginalLogLikelihood")
    else:
        if verbose:
            if use_custom_mll:
                print("\nUser-selected: ConfidenceWeightedMLL")
            else:
                print("\nUser-selected: ExactMarginalLogLikelihood")

    # Create model
    model = DeepKernelGP(
        datapoints=datapoints,
        targets=targets,
        input_dim=input_dim,
        feature_dim=feature_dim,
        hidden_dims=hidden_dims,
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
    
    # Early stopping
    best_loss = float('inf')
    patience_counter = 0

    if verbose:
        print(f"\nTraining Deep Kernel GP")
        print("=" * 60)
        print(f"  Device: {device}")
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
        
        # Compute loss
        loss = -mll(output, model.gp_model.train_targets)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        current_loss = loss.item()
        losses.append(current_loss)

        # Early stopping check
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


def fit_dkgp(
    X_train, 
    y_train, 
    confidence_weights=None, 
    use_custom_mll=None, 
    feature_dim=16, 
    hidden_dims=None,
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
    Fit Deep Kernel GP regression model with optional confidence weighting.
    
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
    """
    if hidden_dims is None:
        hidden_dims = [256, 128, 64]
        
    if verbose:
        print("=" * 60)
        print("Training Deep Kernel GP Regression Model")
        print("=" * 60)

    input_dim = X_train.shape[-1]

    dkl_model, losses = train_dkgp(
        datapoints=X_train,
        targets=y_train,
        input_dim=input_dim,
        feature_dim=feature_dim,
        hidden_dims=hidden_dims,
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
    
    # Create MLL for reference
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
        plt.title('Training Loss', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.tight_layout()
        plt.show()

    return mll, gp_model, dkl_model, losses
