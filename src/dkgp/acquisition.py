"""
Acquisition functions for Bayesian optimization with DKGP.
"""
import torch
import numpy as np
from scipy.stats import norm


def expected_improvement(
    model,
    candidates,
    best_f,
    xi=0.01,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    maximize=True
):
    """
    Expected Improvement acquisition function.
    
    EI measures the expected improvement over the current best observation.
    
    Parameters
    ----------
    model : DeepKernelGP
        Trained model
    candidates : np.ndarray or torch.Tensor
        Candidate points, shape (n_candidates, input_dim)
    best_f : float
        Best observed function value
    xi : float
        Exploration-exploitation trade-off parameter
    device : str
        Device to use
    maximize : bool
        If True, maximize the function (default)
        If False, minimize the function
    
    Returns
    -------
    ei_values : np.ndarray
        Expected improvement values for each candidate
    """
    if not isinstance(candidates, torch.Tensor):
        candidates = torch.from_numpy(candidates).double()
    else:
        candidates = candidates.double()
    
    candidates = candidates.to(device)
    model.eval()
    
    with torch.no_grad():
        posterior = model(candidates)
        mean = posterior.mean.cpu().numpy().squeeze()
        std = posterior.variance.cpu().numpy().squeeze() ** 0.5
    
    # Flip sign for minimization
    if not maximize:
        mean = -mean
        best_f = -best_f
    
    # Compute EI
    improvement = mean - best_f - xi
    Z = improvement / (std + 1e-9)
    
    ei = improvement * norm.cdf(Z) + std * norm.pdf(Z)
    ei[std == 0.0] = 0.0
    
    return ei


def upper_confidence_bound(
    model,
    candidates,
    beta=2.0,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    maximize=True
):
    """
    Upper Confidence Bound (UCB) acquisition function.
    
    UCB balances exploitation (high mean) with exploration (high uncertainty).
    
    Parameters
    ----------
    model : DeepKernelGP
        Trained model
    candidates : np.ndarray or torch.Tensor
        Candidate points, shape (n_candidates, input_dim)
    beta : float
        Exploration parameter (higher values favor exploration)
    device : str
        Device to use
    maximize : bool
        If True, maximize (UCB). If False, minimize (LCB)
    
    Returns
    -------
    ucb_values : np.ndarray
        UCB values for each candidate
    """
    if not isinstance(candidates, torch.Tensor):
        candidates = torch.from_numpy(candidates).double()
    else:
        candidates = candidates.double()
    
    candidates = candidates.to(device)
    model.eval()
    
    with torch.no_grad():
        posterior = model(candidates)
        mean = posterior.mean.cpu().numpy().squeeze()
        std = posterior.variance.cpu().numpy().squeeze() ** 0.5
    
    if maximize:
        ucb = mean + beta * std
    else:
        ucb = mean - beta * std
    
    return ucb


def probability_of_improvement(
    model,
    candidates,
    best_f,
    xi=0.01,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    maximize=True
):
    """
    Probability of Improvement acquisition function.
    
    PI measures the probability that a candidate will improve over the best.
    
    Parameters
    ----------
    model : DeepKernelGP
        Trained model
    candidates : np.ndarray or torch.Tensor
        Candidate points, shape (n_candidates, input_dim)
    best_f : float
        Best observed function value
    xi : float
        Exploration-exploitation trade-off parameter
    device : str
        Device to use
    maximize : bool
        If True, maximize the function
        If False, minimize the function
    
    Returns
    -------
    pi_values : np.ndarray
        Probability of improvement for each candidate
    """
    if not isinstance(candidates, torch.Tensor):
        candidates = torch.from_numpy(candidates).double()
    else:
        candidates = candidates.double()
    
    candidates = candidates.to(device)
    model.eval()
    
    with torch.no_grad():
        posterior = model(candidates)
        mean = posterior.mean.cpu().numpy().squeeze()
        std = posterior.variance.cpu().numpy().squeeze() ** 0.5
    
    # Flip sign for minimization
    if not maximize:
        mean = -mean
        best_f = -best_f
    
    # Compute PI
    improvement = mean - best_f - xi
    Z = improvement / (std + 1e-9)
    
    pi = norm.cdf(Z)
    pi[std == 0.0] = 0.0
    
    return pi