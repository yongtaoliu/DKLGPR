"""
Utility functions for Deep Kernel GP.
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import norm

### Data Preprocessing Functions ###
def get_grid_coords (img, step=1):
    """
    Generate coordinate grid for a single 2D image.
    
    Args:
        img: 2D numpy array
        step: distance between grid points
    
    Returns:
        N x 2 array of (y, x) coordinates
    """
    h, w = img.shape[:2]
    coords = []
    for i in range(0, h, step):
        for j in range(0, w, step):
            coords.append([i, j])
    return np.array(coords)

def get_subimages(img, coordinates, window_size):
    """
    Extract subimages centered at given coordinates.
    
    Args:
        img: 2D or 3D numpy array (h, w) or (h, w, c)
        coordinates: N x 2 array of (y, x) coordinates
        window_size: size of square window to extract
    
    Returns:
        subimages: (N, window_size, window_size, channels) array
        valid_coords: coordinates where extraction succeeded
        valid_indices: indices of valid extractions
    """
    if img.ndim == 2:
        img = img[..., None]
    
    h, w, c = img.shape
    half_w = window_size // 2
    
    subimages = []
    valid_coords = []
    valid_indices = []
    
    for idx, (y, x) in enumerate(coordinates):
        # Check boundaries
        if (y - half_w >= 0 and y + half_w <= h and
            x - half_w >= 0 and x + half_w <= w):
            
            patch = img[y - half_w:y + half_w,
                       x - half_w:x + half_w, :]
            
            if patch.shape[0] == window_size and patch.shape[1] == window_size:
                subimages.append(patch)
                valid_coords.append([y, x])
                valid_indices.append(idx)
    
    return (np.array(subimages), 
            np.array(valid_coords), 
            np.array(valid_indices))

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

### Attention Analysis Functions ###
def get_attention_scores(model, X):
    """
    Extract attention scores from a model with attention extractor.
    
    Parameters
    ----------
    model : DeepKernelPairwiseGP, DeepKernelGP, or DeepKernelGPClassifier
        Trained model with attention extractor
    X : np.ndarray or torch.Tensor
        Input data, shape (n_samples, input_dim)
        
    Returns
    -------
    attention_scores : np.ndarray
        Attention scores, shape (n_samples, num_heads, num_heads)
        
    Examples
    --------
    >>> # After training with attention extractor
    >>> attention = get_attention_scores(dkl_model, X_test)
    >>> print(attention.shape)  # (n_test, 4, 4) for 4 heads
    """
    if not isinstance(X, torch.Tensor):
        X = torch.from_numpy(X).double()
    
    device = next(model.parameters()).device
    X = X.to(device)
    
    # Check if model has feature extractor
    if not hasattr(model, 'feature_extractor'):
        raise ValueError("Model does not have feature_extractor attribute")
    
    extractor = model.feature_extractor
    
    # Check if extractor is attention-based
    if not hasattr(extractor, 'get_attention_maps'):
        raise ValueError(
            f"Feature extractor type '{type(extractor).__name__}' does not support attention extraction. "
            "Use extractor_type='attention' when training the model."
        )
    
    # Extract attention
    attention = extractor.get_attention_maps(X)
    
    return attention.cpu().numpy()

def get_attention_for_sample(model, x, average_heads=False):
    """
    Get attention scores for a single sample.
    
    Parameters
    ----------
    model : trained model
        Model with attention extractor
    x : np.ndarray or torch.Tensor
        Single sample, shape (input_dim,) or (1, input_dim)
    average_heads : bool
        If True, average across attention heads
        
    Returns
    -------
    attention : np.ndarray
        Attention scores
        - If average_heads=False: shape (num_heads, num_heads)
        - If average_heads=True: shape (num_heads,)
        
    Examples
    --------
    >>> # Get attention for one spectrum
    >>> x = X_spectra[0]
    >>> attention = get_attention_for_sample(model, x)
    >>> print(attention.shape)  # (4, 4) for 4 attention heads
    
    >>> # Average across heads
    >>> avg_attention = get_attention_for_sample(model, x, average_heads=True)
    >>> print(avg_attention.shape)  # (4,)
    """
    if not isinstance(x, torch.Tensor):
        x = torch.from_numpy(x).double()
    
    if x.ndim == 1:
        x = x.unsqueeze(0)
    
    scores = get_attention_scores(model, x)
    scores = scores[0]  # Get first (only) sample
    
    if average_heads:
        # Average across heads dimension
        scores = scores.mean(axis=0)
    
    return scores

def analyze_attention_locality(attention_scores):
    """
    Analyze how local vs global the attention patterns are.
    
    Parameters
    ----------
    attention_scores : np.ndarray
        Attention scores, shape (num_heads, num_heads) or (num_heads,)
        
    Returns
    -------
    locality_scores : dict
        Dictionary with locality analysis
        
    Examples
    --------
    >>> attention = get_attention_for_sample(model, x)
    >>> locality = analyze_attention_locality(attention)
    >>> print(locality['interpretation'])
    """
    if attention_scores.ndim == 1:
        # Single head or averaged
        attention_scores = attention_scores[np.newaxis, :]
    
    num_heads = attention_scores.shape[0]
    results = {
        'num_heads': num_heads,
        'per_head': []
    }
    
    for h in range(num_heads):
        attn = attention_scores[h]
        
        # Entropy (higher = more dispersed attention)
        entropy = -np.sum(attn * np.log(attn + 1e-10))
        
        # Maximum attention (lower = more distributed)
        max_attn = attn.max()
        
        # Concentration ratio (top-3 / total)
        top_3 = np.partition(attn, -3)[-3:].sum()
        concentration = top_3
        
        # Interpretation
        if concentration > 0.8:
            interpretation = "Very focused (attends to few positions)"
        elif concentration > 0.5:
            interpretation = "Moderately focused"
        else:
            interpretation = "Distributed (attends globally)"
        
        results['per_head'].append({
            'head': h,
            'entropy': float(entropy),
            'max_attention': float(max_attn),
            'top3_concentration': float(concentration),
            'interpretation': interpretation
        })
    
    # Overall interpretation
    avg_concentration = np.mean([h['top3_concentration'] for h in results['per_head']])
    if avg_concentration > 0.8:
        results['overall'] = "Model uses very focused attention"
    elif avg_concentration > 0.5:
        results['overall'] = "Model uses balanced attention"
    else:
        results['overall'] = "Model uses distributed attention"
    
    return results


def summarize_attention(model, X, sample_idx=None):
    """
    Print a summary of attention patterns.
    
    Parameters
    ----------
    model : trained model
        Model with attention extractor
    X : np.ndarray
        Input data
    sample_idx : int, optional
        Specific sample to analyze. If None, analyzes first sample.
        
    Examples
    --------
    >>> summarize_attention(model, X_spectra)
    >>> summarize_attention(model, X_spectra, sample_idx=5)
    """
    if sample_idx is None:
        sample_idx = 0
    
    x = X[sample_idx:sample_idx+1]
    attention = get_attention_for_sample(model, x)
    
    print("="*60)
    print(f"Attention Analysis - Sample {sample_idx}")
    print("="*60)
    print(f"\nAttention shape: {attention.shape}")
    print(f"Number of heads: {attention.shape[0]}")
    
    # Analyze each head
    locality = analyze_attention_locality(attention)
    
    print(f"\nPer-Head Analysis:")
    print("-"*60)
    for head_info in locality['per_head']:
        h = head_info['head']
        print(f"\nHead {h}:")
        print(f"  Entropy: {head_info['entropy']:.4f}")
        print(f"  Max attention: {head_info['max_attention']:.4f}")
        print(f"  Top-3 concentration: {head_info['top3_concentration']:.4f}")
        print(f"  → {head_info['interpretation']}")
        
        # Show top attended positions
        head_attn = attention[h]
        top_3_idx = np.argsort(head_attn)[-3:][::-1]
        print(f"  Top-3 positions: {top_3_idx.tolist()}")
        print(f"  Weights: {head_attn[top_3_idx].tolist()}")
    
    print(f"\n{'='*60}")
    print(f"Overall: {locality['overall']}")
    print(f"{'='*60}")

### Save and Load Model Functions ###
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