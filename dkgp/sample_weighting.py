"""
Sample-level attention weighting for robust training.
"""
import torch
import torch.nn as nn
import numpy as np


class SampleWeightModule(nn.Module):
    """
    Learnable per-sample weights for robust training.
    
    Each training sample gets a learnable weight that is optimized
    during training. Noisy/outlier samples automatically get lower weights.
    
    Parameters
    ----------
    n_samples : int
        Number of training samples
    
    Examples
    --------
    >>> weight_module = SampleWeightModule(n_samples=100)
    >>> weights = weight_module.get_weights()  # (100,) in [0, 1]
    >>> noisy_samples = np.where(weights < 0.5)[0]
    """
    
    def __init__(self, n_samples):
        super().__init__()
        
        self.n_samples = n_samples
        
        # Learnable log-weights (use log for numerical stability)
        # Initialize to 0 → sigmoid(0) = 0.5 → normalized = 1.0
        self.log_weights = nn.Parameter(
            torch.zeros(n_samples, dtype=torch.float64)
        )
    
    def get_weights(self, normalize=True):
        """
        Get sample weights in [0, 1].
        
        Parameters
        ----------
        normalize : bool
            If True, normalize so mean = 1.0
            
        Returns
        -------
        weights : torch.Tensor
            Sample weights, shape (n_samples,)
        """
        # Sigmoid to map to [0, 1]
        weights = torch.sigmoid(self.log_weights)
        
        if normalize:
            # Normalize so average weight = 1.0
            # This keeps the loss scale similar to unweighted case
            weights = weights / weights.mean()
        
        return weights
    
    def forward(self, sample_indices=None):
        """
        Get weights for specific samples.
        
        Parameters
        ----------
        sample_indices : torch.Tensor or None
            Indices of samples to get weights for.
            If None, return all weights.
            
        Returns
        -------
        weights : torch.Tensor
            Weights for specified samples
        """
        weights = self.get_weights()
        
        if sample_indices is not None:
            return weights[sample_indices]
        return weights


def apply_sample_weighting(loss_fn, sample_weight_module, predictions, targets):
    """
    Apply sample weighting to loss function.
    
    Parameters
    ----------
    loss_fn : callable
        Base loss function (e.g., MSE)
    sample_weight_module : SampleWeightModule
        Module containing learnable sample weights
    predictions : torch.Tensor
        Model predictions
    targets : torch.Tensor
        Target values
        
    Returns
    -------
    weighted_loss : torch.Tensor
        Loss weighted by sample importance
    """
    # Get sample weights
    weights = sample_weight_module.get_weights()
    
    # Compute base loss per sample
    sample_losses = (predictions - targets) ** 2
    
    # Weight each sample's loss
    weighted_losses = weights * sample_losses
    
    # Return mean
    return weighted_losses.mean()


def analyze_sample_weights(sample_weights, y_train, predictions=None, 
                          threshold=0.5, verbose=True):
    """
    Analyze learned sample weights to identify noisy samples.
    
    Parameters
    ----------
    sample_weights : np.ndarray
        Learned sample weights, shape (n_samples,)
    y_train : np.ndarray
        Training targets
    predictions : np.ndarray, optional
        Model predictions on training data
    threshold : float
        Weight threshold below which samples are considered noisy
    verbose : bool
        Print analysis
        
    Returns
    -------
    analysis : dict
        Dictionary with analysis results:
        - 'noisy_indices': Indices of detected noisy samples
        - 'clean_indices': Indices of clean samples
        - 'weight_stats': Statistics about weights
        - 'noisy_stats': Statistics about noisy samples
    """
    # Find noisy samples
    noisy_indices = np.where(sample_weights < threshold)[0]
    clean_indices = np.where(sample_weights >= threshold)[0]
    
    # Statistics
    weight_stats = {
        'min': sample_weights.min(),
        'max': sample_weights.max(),
        'mean': sample_weights.mean(),
        'std': sample_weights.std(),
        'median': np.median(sample_weights)
    }
    
    noisy_stats = {
        'count': len(noisy_indices),
        'percentage': len(noisy_indices) / len(sample_weights) * 100,
        'avg_weight': sample_weights[noisy_indices].mean() if len(noisy_indices) > 0 else 0,
        'avg_target': y_train[noisy_indices].mean() if len(noisy_indices) > 0 else 0
    }
    
    if predictions is not None:
        errors = np.abs(predictions - y_train)
        noisy_stats['avg_error'] = errors[noisy_indices].mean() if len(noisy_indices) > 0 else 0
        noisy_stats['clean_avg_error'] = errors[clean_indices].mean() if len(clean_indices) > 0 else 0
    
    if verbose:
        print("="*70)
        print("Sample Weight Analysis")
        print("="*70)
        
        print(f"\nWeight Statistics:")
        print(f"  Min:    {weight_stats['min']:.4f}")
        print(f"  Max:    {weight_stats['max']:.4f}")
        print(f"  Mean:   {weight_stats['mean']:.4f}")
        print(f"  Median: {weight_stats['median']:.4f}")
        print(f"  Std:    {weight_stats['std']:.4f}")
        
        print(f"\nNoisy Sample Detection (threshold < {threshold}):")
        print(f"  Detected: {noisy_stats['count']} / {len(sample_weights)} "
              f"({noisy_stats['percentage']:.1f}%)")
        
        if noisy_stats['count'] > 0:
            print(f"  Avg weight (noisy): {noisy_stats['avg_weight']:.4f}")
            print(f"  Avg target (noisy): {noisy_stats['avg_target']:.4f}")
            
            if predictions is not None:
                print(f"  Avg error (noisy):  {noisy_stats['avg_error']:.4f}")
                print(f"  Avg error (clean):  {noisy_stats['clean_avg_error']:.4f}")
                
                if noisy_stats['avg_error'] > noisy_stats['clean_avg_error']:
                    print(f"  ✓ Noisy samples have higher error (as expected)")
                else:
                    print(f"  ⚠ Warning: Noisy samples don't have higher error")
        
        print("="*70)
    
    return {
        'noisy_indices': noisy_indices,
        'clean_indices': clean_indices,
        'weight_stats': weight_stats,
        'noisy_stats': noisy_stats
    }
