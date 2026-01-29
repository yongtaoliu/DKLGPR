# Deep Kernel GP - Minimal Version

Deep Kernel Learning for Gaussian Process Regression - Essential Components Only

## Installation

```bash
pip install -e .
```

## Quick Start

```python
import numpy as np
from deep_kernel_gp import fit_dkgp, predict
from deep_kernel_gp.acquisition import expected_improvement

# Generate data
X_train = np.random.randn(100, 50)
y_train = np.random.randn(100)

# Train model
mll, gp_model, dkl_model, losses = fit_dkgp(
    X_train, 
    y_train,
    feature_dim=16,
    num_epochs=1000
)

# Predict
X_test = np.random.randn(20, 50)
mean, std = predict(dkl_model, X_test, return_std=True)

# Bayesian Optimization
best_f = y_train.max()
ei = expected_improvement(dkl_model, X_test, best_f)
```

## What's Included

### Source Code (src/deep_kernel_gp/)
- models.py - Core model classes
- training.py - Training functions
- prediction.py - Prediction utilities
- acquisition.py - Acquisition functions
- utils.py - Helper functions
- __init__.py - Package initialization

### Root Files
- pyproject.toml - Package configuration
- setup.py - Setup script
- requirements.txt - Dependencies
- README.md - This file
- LICENSE - MIT License

## License

MIT License
