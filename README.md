# DKGP


```
dkgp/
├── models.py         # All feature extractors
├── gpr.py           # Gaussian Process Regression
├── gpc.py           # Gaussian Process Classification
├── gppw.py          # Gaussian Process Pairwise
├── utils.py         # Utility functions
└── __init__.py      # Package exports
```

## Feature Extractors

### Available Extractors

1. **`FCFeatureExtractor`** - Simple fully-connected
   - Fast, lightweight
   - Good for prototyping
   - No regularization

2. **`FCBNFeatureExtractor`** - FC + BatchNorm + Dropout (Default)
   - Recommended for general use
   - Prevents overfitting
   - Most robust

3. **`ResNetFeatureExtractor`** - ResNet with skip connections
   - Better gradient flow
   - Good for deeper networks
   - Handles vanishing gradients

4. **`AttentionFeatureExtractor`** - Self-attention based
   - Learns feature interactions
   - Good for relational data
   - More parameters

### Factory Function

Use `get_feature_extractor()` to create extractors:

```python
from deep_kernel_gp import get_feature_extractor

# Simple FC
extractor = get_feature_extractor('fc', input_dim=100, feature_dim=16)

# FC + BatchNorm 
extractor = get_feature_extractor('fcbn', input_dim=100, feature_dim=16,
                                  hidden_dims=[512, 256, 128], dropout=0.3)

# ResNet
extractor = get_feature_extractor('resnet', input_dim=100, feature_dim=16,
                                  hidden_dim=256, num_blocks=3)

# Attention
extractor = get_feature_extractor('attention', input_dim=100, feature_dim=16,
                                  hidden_dim=128, num_heads=4)

# Custom
import torch.nn as nn
my_net = nn.Sequential(nn.Linear(100, 64), nn.ReLU(), nn.Linear(64, 16))
extractor = get_feature_extractor('custom', custom_extractor=my_net)
```

## Usage

### Regression

```python
from deep_kernel_gp import fit_dkgp, predict

# Default extractor (FC + BatchNorm)
mll, gp, dkl, losses = fit_dkgp(X_train, y_train, feature_dim=16)

# Predict
mean, std = predict(dkl, X_test, return_std=True)
```

### Classification

```python
from dkgp import fit_dkgp_classifier, predict_classifier

# Default extractor (FC + BatchNorm)
model, losses = fit_dkgp_classifier(X_train, y_train, num_classes=4)

# Predict
y_pred = predict_classifier(model, X_test)
y_proba = predict_classifier(model, X_test, return_proba=True)
```

## Comparison of Extractors

**`fc`:**
- Prototyping quickly
- Small datasets (<100 samples)
- Simple patterns

**`fcbn`:**
- General purpose modeling
- Medium datasets (100-10,000 samples)
- Want robustness

**`resnet`:**
- Need deeper networks
- Experiencing vanishing gradients
- Want stable training

**`attention`:**
- Feature interactions matter
- Have relational data
- Computational cost is okay

## Complete Example

```python
import numpy as np
from dkgp import fit_dkgp, predict, get_feature_extractor

# Generate data
np.random.seed(42)
X_train = np.random.randn(200, 50)
y_train = np.sum(X_train[:, :5], axis=1) + 0.1 * np.random.randn(200)
X_test = np.random.randn(50, 50)
y_test = np.sum(X_test[:, :5], axis=1) + 0.1 * np.random.randn(50)

# Fit
mll, gp, dkl, losses = fit_dkgp(
    X_train, y_train,
    feature_dim=16,
    extractor_type='resnet',
    extractor_kwargs={'hidden_dim': 128, 'num_blocks': 2},
    num_epochs=1000,
    lr_features=1e-4,
    lr_gp=1e-2
)

# Predict
mean, std = predict(dkl, X_test, return_std=True)

```

## API Reference

### Regression (`gpr.py`)

- `DeepKernelGP` - Main regression model class
- `ConfidenceWeightedMLL` - Weighted loss for heteroscedastic data
- `train_dkgp()` - Low-level training function
- `fit_dkgp()` - High-level training interface

### Classification (`gpc.py`)

- `DeepKernelGPClassifier` - Main classification model class
- `BinaryGPClassificationModel` - Binary GP model
- `MultiClassGPClassificationModel` - Multi-class GP model
- `train_dkgp_classifier()` - Low-level training function
- `fit_dkgp_classifier()` - High-level training interface
- `predict_classifier()` - Prediction function

### Feature Extractors (`models.py`)

- `FCFeatureExtractor` - Simple FC
- `FCBNFeatureExtractor` - FC + BatchNorm + Dropout (default)
- `ResNetFeatureExtractor` - ResNet style
- `AttentionFeatureExtractor` - Self-attention
- `get_feature_extractor()` - Factory function


### Custom Hybrid Extractor

```python
import torch.nn as nn
from deep_kernel_gp import get_feature_extractor, fit_dkgp

class HybridExtractor(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super().__init__()
        # Combine ResNet and Attention
        self.resnet = get_feature_extractor('resnet', input_dim, feature_dim//2)
        self.attention = get_feature_extractor('attention', input_dim, feature_dim//2)
        self.input_dim = input_dim
        self.feature_dim = feature_dim
    
    def forward(self, x):
        res_features = self.resnet(x)
        attn_features = self.attention(x)
        return torch.cat([res_features, attn_features], dim=-1)

# Use it
hybrid = HybridExtractor(input_dim=100, feature_dim=16)
mll, gp, dkl, losses = fit_dkgp(
    X, y,
    extractor_type='custom',
    extractor_kwargs={'custom_extractor': hybrid}
)
```

## Version History

- **v0.2.0** - Restructured with multiple feature extractors
- **v0.1.0** - Initial release

## License

MIT
