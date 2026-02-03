"""
Prediction utilities for Deep Kernel GP models.
"""
import torch
import numpy as np

def predict(
    model, 
    test_data,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    return_std=False,
    batch_size=None
):
    """
    Predict outputs for test data.
    
    Parameters
    ----------
    model : DeepKernelGP
        Trained model
    test_data : np.ndarray or torch.Tensor
        Test features, shape (n_test, input_dim)
    device : str
        Device to use ('cuda' or 'cpu')
    return_std : bool
        If True, return standard deviation instead of variance
    batch_size : int, optional
        Process test data in batches to save memory
    
    Returns
    -------
    mean : np.ndarray
        Predicted means, shape (n_test,)
    uncertainty : np.ndarray
        Predicted variance (or std if return_std=True), shape (n_test,)
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
            posterior = model(test_data)
            mean = posterior.mean.cpu().numpy().squeeze()
            variance = posterior.variance.cpu().numpy().squeeze()
    else:
        # Process in batches
        n_test = len(test_data)
        means = []
        variances = []
        
        with torch.no_grad():
            for i in range(0, n_test, batch_size):
                batch = test_data[i:i+batch_size]
                posterior = model(batch)
                means.append(posterior.mean.cpu().numpy())
                variances.append(posterior.variance.cpu().numpy())
        
        mean = np.concatenate(means, axis=0).squeeze()
        variance = np.concatenate(variances, axis=0).squeeze()

    if return_std:
        return mean, np.sqrt(variance)
    return mean, variance


def predict_with_gradients(
    model,
    test_data,
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    Predict with gradient computation enabled.
    
    Useful for acquisition function optimization where we need gradients
    with respect to the input.
    
    Parameters
    ----------
    model : DeepKernelGP
        Trained model
    test_data : torch.Tensor
        Test features, shape (n_test, input_dim)
        Must be a torch.Tensor with requires_grad=True for gradient computation
    device : str
        Device to use
    
    Returns
    -------
    posterior : gpytorch.distributions.MultivariateNormal
        GP posterior with gradient tracking
    """
    if not isinstance(test_data, torch.Tensor):
        raise TypeError("test_data must be a torch.Tensor for gradient computation")
    
    test_data = test_data.to(device)
    model.eval()
    
    posterior = model(test_data)
    return posterior


def predict_quantiles(
    model,
    test_data,
    quantiles=[0.025, 0.25, 0.5, 0.75, 0.975],
    device='cuda' if torch.cuda.is_available() else 'cpu',
    n_samples=1000
):
    """
    Predict quantiles of the posterior distribution.
    
    Parameters
    ----------
    model : DeepKernelGP
        Trained model
    test_data : np.ndarray or torch.Tensor
        Test features, shape (n_test, input_dim)
    quantiles : list of float
        Quantiles to compute (between 0 and 1)
    device : str
        Device to use
    n_samples : int
        Number of samples for quantile estimation
    
    Returns
    -------
    quantile_dict : dict
        Dictionary mapping quantile values to arrays of predictions
    """
    if not isinstance(test_data, torch.Tensor):
        test_data = torch.from_numpy(test_data).double()
    else:
        test_data = test_data.double()

    test_data = test_data.to(device)
    model.eval()
    
    with torch.no_grad():
        posterior = model(test_data)
        samples = posterior.sample(torch.Size([n_samples]))
        samples_np = samples.cpu().numpy().squeeze()
        
    quantile_dict = {}
    for q in quantiles:
        quantile_dict[q] = np.quantile(samples_np, q, axis=0)
    
    return quantile_dict


def predict_with_epistemic_aleatoric(
    model,
    test_data,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    n_samples=100
):
    """
    Separate epistemic (model) and aleatoric (noise) uncertainty.
    
    Parameters
    ----------
    model : DeepKernelGP
        Trained model
    test_data : np.ndarray or torch.Tensor
        Test features, shape (n_test, input_dim)
    device : str
        Device to use
    n_samples : int
        Number of samples for epistemic uncertainty estimation
    
    Returns
    -------
    mean : np.ndarray
        Predicted means
    epistemic : np.ndarray
        Epistemic (model) uncertainty
    aleatoric : np.ndarray
        Aleatoric (observation noise) uncertainty
    total : np.ndarray
        Total uncertainty (epistemic + aleatoric)
    """
    if not isinstance(test_data, torch.Tensor):
        test_data = torch.from_numpy(test_data).double()
    else:
        test_data = test_data.double()

    test_data = test_data.to(device)
    model.eval()
    
    with torch.no_grad():
        posterior = model(test_data)
        mean = posterior.mean.cpu().numpy().squeeze()
        
        # Total variance includes observation noise
        total_var = posterior.variance.cpu().numpy().squeeze()
        
        # Aleatoric uncertainty from likelihood noise
        noise_var = model.gp_model.likelihood.noise.item()
        aleatoric = np.full_like(total_var, noise_var)
        
        # Epistemic uncertainty (model uncertainty)
        epistemic = total_var - aleatoric
        
    return mean, epistemic, aleatoric, total_var


def batch_predict(
    model,
    test_data,
    batch_size=100,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    return_std=False,
    show_progress=False
):
    """
    Batch prediction for large test sets to manage memory.
    
    Parameters
    ----------
    model : DeepKernelGP
        Trained model
    test_data : np.ndarray or torch.Tensor
        Test features, shape (n_test, input_dim)
    batch_size : int
        Number of test points per batch
    device : str
        Device to use
    return_std : bool
        Return standard deviation instead of variance
    show_progress : bool
        Show progress bar (requires tqdm)
    
    Returns
    -------
    mean : np.ndarray
        Predicted means
    uncertainty : np.ndarray
        Predicted variance or standard deviation
    """
    if not isinstance(test_data, torch.Tensor):
        test_data = torch.from_numpy(test_data).double()
    else:
        test_data = test_data.double()

    n_test = len(test_data)
    model.eval()
    
    means = []
    variances = []
    
    iterator = range(0, n_test, batch_size)
    if show_progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(iterator, desc="Predicting")
        except ImportError:
            print("Warning: tqdm not installed, progress bar disabled")
    
    with torch.no_grad():
        for i in iterator:
            batch = test_data[i:i+batch_size].to(device)
            posterior = model(batch)
            means.append(posterior.mean.cpu().numpy())
            variances.append(posterior.variance.cpu().numpy())
    
    mean = np.concatenate(means, axis=0).squeeze()
    variance = np.concatenate(variances, axis=0).squeeze()
    
    if return_std:
        return mean, np.sqrt(variance)
    return mean, variance
