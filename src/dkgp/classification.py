"""
Deep Kernel GP Classification for binary and multi-class problems.
"""
import torch
import torch.nn as nn
import numpy as np
from botorch.models import SingleTaskGP
from botorch.models.transforms.input import Normalize
from gpytorch.likelihoods import BernoulliLikelihood, SoftmaxLikelihood
from gpytorch.mlls import VariationalELBO
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
from .models import ImageFeatureExtractor


class DeepKernelGPClassifier(nn.Module):
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
        num_inducing=100,
        dropout=0.2
    ):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]
        
        self.num_classes = num_classes
        self.input_dim = input_dim
        self.feature_dim = feature_dim
        
        # Feature extractor
        self.feature_extractor = ImageFeatureExtractor(
            input_dim=input_dim,
            feature_dim=feature_dim,
            hidden_dims=hidden_dims,
            dropout=dropout
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
                probs = torch.softmax(output.mean, dim=-1)
            
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
        return torch.argmax(probs, dim=-1)
    
    def update_gp_data(self):
        """Update GP training data with current features."""
        features = self.feature_extractor(self.train_datapoints)
        self.gp_model.set_train_data(features, self.train_targets, strict=False)


class BinaryGPClassificationModel(ApproximateGP):
    """
    Variational GP for binary classification.
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


def train_dkgp_classifier(
    datapoints,
    targets,
    input_dim,
    num_classes=2,
    feature_dim=16,
    hidden_dims=None,
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
    
    # Convert to tensors
    if not isinstance(datapoints, torch.Tensor):
        datapoints = torch.from_numpy(datapoints).double()
    else:
        datapoints = datapoints.double()
    
    if not isinstance(targets, torch.Tensor):
        targets = torch.from_numpy(targets).long()
    else:
        targets = targets.long()
    
    datapoints = datapoints.to(device)
    targets = targets.to(device)
    
    # Create model
    model = DeepKernelGPClassifier(
        datapoints=datapoints,
        targets=targets,
        input_dim=input_dim,
        num_classes=num_classes,
        feature_dim=feature_dim,
        hidden_dims=hidden_dims,
        num_inducing=num_inducing
    ).to(device)
    
    # Optimizer
    optimizer = torch.optim.Adam([
        {'params': model.feature_extractor.parameters(), 'lr': lr_features},
        {'params': model.gp_model.parameters(), 'lr': lr_gp}
    ])
    
    # Loss function (Variational ELBO)
    mll = VariationalELBO(model.gp_model.likelihood, model.gp_model, num_data=len(targets))
    
    model.train()
    losses = []
    
    # Early stopping
    best_loss = float('inf')
    patience_counter = 0
    
    if verbose:
        print(f"\nTraining Deep Kernel GP Classifier")
        print("=" * 60)
        print(f"  Device: {device}")
        print(f"  Input dim: {input_dim} → Feature dim: {feature_dim}")
        print(f"  Classes: {num_classes}")
        print(f"  Samples: {len(datapoints)}")
        print(f"  Inducing points: {num_inducing}")
        if patience:
            print(f"  Early stopping: patience={patience}")
        print("=" * 60)
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        # Forward pass
        output = model(datapoints)
        
        # Compute loss
        loss = -mll(output, targets)
        
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


def fit_dkgp_classifier(
    X_train,
    y_train,
    num_classes=None,
    feature_dim=16,
    hidden_dims=None,
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
    """
    if hidden_dims is None:
        hidden_dims = [256, 128, 64]
    
    # Auto-detect number of classes
    if num_classes is None:
        if isinstance(y_train, torch.Tensor):
            num_classes = len(torch.unique(y_train))
        else:
            num_classes = len(np.unique(y_train))
    
    if verbose:
        print("=" * 60)
        print("Training Deep Kernel GP Classifier")
        print("=" * 60)
    
    input_dim = X_train.shape[-1]
    
    model, losses = train_dkgp_classifier(
        datapoints=X_train,
        targets=y_train,
        input_dim=input_dim,
        num_classes=num_classes,
        feature_dim=feature_dim,
        hidden_dims=hidden_dims,
        num_inducing=num_inducing,
        num_epochs=num_epochs,
        lr_features=lr_features,
        lr_gp=lr_gp,
        device=device,
        verbose=verbose,
        patience=patience
    )
    
    if plot_loss and verbose:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 5))
        plt.plot(losses, linewidth=2, color='#2E86AB')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Negative ELBO', fontsize=12)
        plt.title('Training Loss', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.tight_layout()
        plt.show()
    
    return model, losses


def predict_classifier(
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
        test_data = torch.from_numpy(test_data).double()
    else:
        test_data = test_data.double()
    
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


# Import gpytorch at module level
import gpytorch
