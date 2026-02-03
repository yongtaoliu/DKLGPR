"""
Utility functions for Deep Kernel GP.
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import norm


def standardize_data(X, y, X_mean=None, X_std=None, y_mean=None, y_std=None):
    """
    Standardize training data (zero mean, unit variance).
    
    Parameters
    ----------
    X : np.ndarray
        Input features
    y : np.ndarray
        Target values
    X_mean, X_std, y_mean, y_std : np.ndarray, optional
        Pre-computed statistics for test data
    
    Returns
    -------
    X_scaled : np.ndarray
        Standardized inputs
    y_scaled : np.ndarray
        Standardized targets
    stats : dict
        Dictionary with mean and std for inverse transform
    """
    if X_mean is None:
        X_mean = X.mean(axis=0)
        X_std = X.std(axis=0) + 1e-8
    
    if y_mean is None:
        y_mean = y.mean()
        y_std = y.std() + 1e-8
    
    X_scaled = (X - X_mean) / X_std
    y_scaled = (y - y_mean) / y_std
    
    stats = {
        'X_mean': X_mean,
        'X_std': X_std,
        'y_mean': y_mean,
        'y_std': y_std
    }
    
    return X_scaled, y_scaled, stats


def inverse_standardize(y_scaled, y_mean, y_std):
    """
    Inverse standardization transform.
    
    Parameters
    ----------
    y_scaled : np.ndarray
        Standardized values
    y_mean : float
        Original mean
    y_std : float
        Original std
    
    Returns
    -------
    y : np.ndarray
        Original scale values
    """
    return y_scaled * y_std + y_mean


def compute_calibration(y_true, mean, std, n_bins=10):
    """
    Compute calibration statistics for uncertainty estimates.
    
    Parameters
    ----------
    y_true : np.ndarray
        True values
    mean : np.ndarray
        Predicted means
    std : np.ndarray
        Predicted standard deviations
    n_bins : int
        Number of bins for calibration curve
    
    Returns
    -------
    calibration_dict : dict
        Dictionary with calibration statistics
    """
    # Compute standardized residuals
    z_scores = (y_true - mean) / (std + 1e-8)
    
    # Expected vs observed coverage
    confidence_levels = np.linspace(0.1, 0.9, n_bins)
    observed_coverage = []
    
    for conf in confidence_levels:
        # Number of std devs for this confidence level
        z = norm.ppf((1 + conf) / 2)
        
        # Count points within interval
        in_interval = np.abs(z_scores) <= z
        observed_coverage.append(in_interval.mean())
    
    # Mean absolute calibration error
    mace = np.mean(np.abs(confidence_levels - observed_coverage))
    
    return {
        'confidence_levels': confidence_levels,
        'observed_coverage': np.array(observed_coverage),
        'mace': mace,
        'z_scores': z_scores
    }


def plot_calibration(calibration_dict, save_path=None):
    """
    Plot calibration curve.
    
    Parameters
    ----------
    calibration_dict : dict
        Output from compute_calibration
    save_path : str, optional
        Path to save figure
    """
    conf = calibration_dict['confidence_levels']
    obs = calibration_dict['observed_coverage']
    mace = calibration_dict['mace']
    
    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    plt.plot(conf, obs, 'o-', linewidth=2, markersize=8, label='Model')
    plt.xlabel('Expected Coverage', fontsize=12)
    plt.ylabel('Observed Coverage', fontsize=12)
    plt.title(f'Calibration Curve (MACE = {mace:.3f})', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_predictions(
    y_true, 
    y_pred, 
    uncertainty=None,
    title='Predictions vs True Values',
    save_path=None
):
    """
    Plot predictions against true values.
    
    Parameters
    ----------
    y_true : np.ndarray
        True values
    y_pred : np.ndarray
        Predicted values
    uncertainty : np.ndarray, optional
        Uncertainty estimates (std or variance)
    title : str
        Plot title
    save_path : str, optional
        Path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Scatter plot
    axes[0].scatter(y_true, y_pred, alpha=0.6, s=50, edgecolors='k', linewidth=0.5)
    
    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect')
    
    # Compute R²
    r2 = 1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - y_true.mean())**2)
    
    axes[0].set_xlabel('True Values', fontsize=12)
    axes[0].set_ylabel('Predicted Values', fontsize=12)
    axes[0].set_title(f'{title}\n$R^2$ = {r2:.3f}', fontsize=13, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Residual plot
    residuals = y_true - y_pred
    axes[1].scatter(y_pred, residuals, alpha=0.6, s=50, edgecolors='k', linewidth=0.5)
    axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
    
    if uncertainty is not None:
        # Add uncertainty bands
        axes[1].fill_between(
            np.sort(y_pred),
            -2*np.sort(uncertainty),
            2*np.sort(uncertainty),
            alpha=0.2,
            label='±2σ'
        )
        axes[1].legend()
    
    axes[1].set_xlabel('Predicted Values', fontsize=12)
    axes[1].set_ylabel('Residuals', fontsize=12)
    axes[1].set_title('Residual Plot', fontsize=13, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def split_train_test(X, y, test_size=0.2, random_state=None):
    """
    Split data into train and test sets.
    
    Parameters
    ----------
    X : np.ndarray
        Features
    y : np.ndarray
        Targets
    test_size : float
        Fraction of data for testing
    random_state : int, optional
        Random seed
    
    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    n = len(X)
    indices = np.random.permutation(n)
    n_test = int(n * test_size)
    
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]
    
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def save_model(model, filepath):
    """
    Save trained model to disk.
    
    Parameters
    ----------
    model : DeepKernelGP
        Trained model
    filepath : str
        Path to save model
    """
    torch.save({
        'feature_extractor_state': model.feature_extractor.state_dict(),
        'gp_model_state': model.gp_model.state_dict(),
        'input_dim': model.input_dim,
        'feature_dim': model.feature_dim,
        'train_datapoints': model.train_datapoints,
        'train_targets': model.train_targets,
    }, filepath)


def load_model(filepath, device='cpu'):
    """
    Load trained model from disk.
    
    Parameters
    ----------
    filepath : str
        Path to saved model
    device : str
        Device to load model to
    
    Returns
    -------
    model : DeepKernelGP
        Loaded model
    """
    from .models import DeepKernelGP
    
    checkpoint = torch.load(filepath, map_location=device)
    
    model = DeepKernelGP(
        datapoints=checkpoint['train_datapoints'],
        targets=checkpoint['train_targets'],
        input_dim=checkpoint['input_dim'],
        feature_dim=checkpoint['feature_dim']
    )
    
    model.feature_extractor.load_state_dict(checkpoint['feature_extractor_state'])
    model.gp_model.load_state_dict(checkpoint['gp_model_state'])
    model.to(device)
    model.eval()
    
    return model


def compute_metrics(y_true, y_pred, y_std=None):
    """
    Compute various regression metrics.
    
    Parameters
    ----------
    y_true : np.ndarray
        True values
    y_pred : np.ndarray
        Predicted values
    y_std : np.ndarray, optional
        Predicted standard deviations
    
    Returns
    -------
    metrics : dict
        Dictionary of metrics
    """
    mse = np.mean((y_true - y_pred)**2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))
    r2 = 1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - y_true.mean())**2)
    
    metrics = {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }
    
    if y_std is not None:
        # Negative log-likelihood
        nll = 0.5 * np.mean(
            np.log(2 * np.pi * y_std**2) + ((y_true - y_pred)**2) / (y_std**2)
        )
        metrics['nll'] = nll
        
        # Mean standardized log loss
        msll = np.mean(
            0.5 * np.log(2 * np.pi * y_std**2) + 
            0.5 * ((y_true - y_pred) / y_std)**2
        )
        metrics['msll'] = msll
    
    return metrics


def print_metrics(metrics, title="Model Performance"):
    """
    Pretty print metrics.
    
    Parameters
    ----------
    metrics : dict
        Dictionary of metrics
    title : str
        Title to print
    """
    print("\n" + "="*50)
    print(f"{title:^50}")
    print("="*50)
    
    for key, value in metrics.items():
        print(f"  {key.upper():10s}: {value:>12.6f}")
    
    print("="*50 + "\n")
