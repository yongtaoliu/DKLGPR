# DKGP


```
dkgp/
├── models.py         # All feature extractors
├── gpr.py           # Gaussian Process Regression
├── gpc.py           # Gaussian Process Classification
├── prediction.py    # Prediction utilities
├── acquisition.py   # Acquisition functions for BO
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

5. **`WideDeepFeatureExtractor`** - Wide & Deep architecture
   - Combines linear + nonlinear paths
   - Good for mixed feature types
   - Flexible

### Factory Function

Use `get_feature_extractor()` to create extractors:

```python
from deep_kernel_gp import get_feature_extractor

# Simple FC
extractor = get_feature_extractor('fc', input_dim=100, feature_dim=16)

# FC + BatchNorm (recommended)
extractor = get_feature_extractor('fcbn', input_dim=100, feature_dim=16,
                                  hidden_dims=[512, 256, 128], dropout=0.3)

# ResNet
extractor = get_feature_extractor('resnet', input_dim=100, feature_dim=16,
                                  hidden_dim=256, num_blocks=3)

# Attention
extractor = get_feature_extractor('attention', input_dim=100, feature_dim=16,
                                  hidden_dim=128, num_heads=4)

# Wide & Deep
extractor = get_feature_extractor('wide_deep', input_dim=100, feature_dim=16,
                                  deep_dims=[256, 128])

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

# Simple FC extractor
mll, gp, dkl, losses = fit_dkgp(X_train, y_train, extractor_type='fc')

# ResNet extractor
mll, gp, dkl, losses = fit_dkgp(
    X_train, y_train,
    extractor_type='resnet',
    extractor_kwargs={'hidden_dim': 256, 'num_blocks': 3}
)

# Custom extractor
import torch.nn as nn
my_net = nn.Sequential(nn.Linear(100, 64), nn.ReLU(), nn.Linear(64, 16))
mll, gp, dkl, losses = fit_dkgp(
    X_train, y_train,
    extractor_type='custom',
    extractor_kwargs={'custom_extractor': my_net}
)

# Predict
mean, std = predict(dkl, X_test, return_std=True)
```

### Classification

```python
from dkgp import fit_dkgp_classifier, predict_classifier

# Default extractor (FC + BatchNorm)
model, losses = fit_dkgp_classifier(X_train, y_train, num_classes=4)

# Simple FC extractor
model, losses = fit_dkgp_classifier(
    X_train, y_train,
    num_classes=4,
    extractor_type='fc'
)

# ResNet extractor
model, losses = fit_dkgp_classifier(
    X_train, y_train,
    num_classes=4,
    extractor_type='resnet',
    extractor_kwargs={'hidden_dim': 256, 'num_blocks': 3}
)

# Attention extractor
model, losses = fit_dkgp_classifier(
    X_train, y_train,
    num_classes=4,
    extractor_type='attention',
    extractor_kwargs={'hidden_dim': 128, 'num_heads': 4}
)

# Predict
y_pred = predict_classifier(model, X_test)
y_proba = predict_classifier(model, X_test, return_proba=True)
```

## Comparison of Extractors

### Performance Characteristics

| Extractor | Speed | Parameters | Regularization | Best For |
|-----------|-------|------------|----------------|----------|
| `fc` | ⚡⚡⚡ | Low | None | Prototyping, small data |
| `fcbn` | ⚡⚡ | Medium | BatchNorm + Dropout | General use (recommended) |
| `resnet` | ⚡ | Medium-High | BatchNorm + Skip | Deep networks, gradients |
| `attention` | 🐌 | High | LayerNorm | Feature interactions |
| `wide_deep` | ⚡⚡ | Medium | BatchNorm + Dropout | Mixed features |

### When to Use Which

**Use `fc` when:**
- Prototyping quickly
- Small datasets (<100 samples)
- Simple patterns

**Use `fcbn` (default) when:**
- General purpose modeling
- Medium datasets (100-10,000 samples)
- Want robustness

**Use `resnet` when:**
- Need deeper networks
- Experiencing vanishing gradients
- Want stable training

**Use `attention` when:**
- Feature interactions matter
- Have relational data
- Computational cost is okay

**Use `wide_deep` when:**
- Mixed feature types (categorical + continuous)
- Need both memorization and generalization
- Medium-large datasets

## Complete Example

```python
import numpy as np
from deep_kernel_gp import fit_dkgp, predict, get_feature_extractor

# Generate data
np.random.seed(42)
X_train = np.random.randn(200, 50)
y_train = np.sum(X_train[:, :5], axis=1) + 0.1 * np.random.randn(200)
X_test = np.random.randn(50, 50)
y_test = np.sum(X_test[:, :5], axis=1) + 0.1 * np.random.randn(50)

# Method 1: Use high-level interface
mll, gp, dkl, losses = fit_dkgp(
    X_train, y_train,
    feature_dim=16,
    extractor_type='resnet',
    extractor_kwargs={'hidden_dim': 128, 'num_blocks': 2},
    num_epochs=1000,
    lr_features=1e-4,
    lr_gp=1e-2
)

# Method 2: Use factory function
extractor = get_feature_extractor(
    'attention',
    input_dim=50,
    feature_dim=16,
    hidden_dim=128,
    num_heads=4
)

mll, gp, dkl, losses = fit_dkgp(
    X_train, y_train,
    extractor_type='custom',
    extractor_kwargs={'custom_extractor': extractor}
)

# Predict
mean, std = predict(dkl, X_test, return_std=True)

# Evaluate
from deep_kernel_gp.utils import compute_metrics
metrics = compute_metrics(y_test, mean, std)
print(metrics)
```

## Backward Compatibility

The old `ImageFeatureExtractor` still works:

```python
from deep_kernel_gp import ImageFeatureExtractor

# Old way (still works)
extractor = ImageFeatureExtractor(input_dim=100, feature_dim=16)

# Equivalent to
extractor = FCBNFeatureExtractor(input_dim=100, feature_dim=16)
```

## Migration Guide

### From Old Structure

**Old:**
```python
from deep_kernel_gp import fit_dkgp
mll, gp, dkl, losses = fit_dkgp(X, y)
```

**New (same, but more options):**
```python
from deep_kernel_gp import fit_dkgp

# Default (same as before)
mll, gp, dkl, losses = fit_dkgp(X, y)

# Or choose extractor
mll, gp, dkl, losses = fit_dkgp(X, y, extractor_type='resnet')
```

### Custom Extractors

**Old:**
```python
# Had to modify source code
```

**New:**
```python
import torch.nn as nn
my_net = nn.Sequential(...)
mll, gp, dkl, losses = fit_dkgp(
    X, y,
    extractor_type='custom',
    extractor_kwargs={'custom_extractor': my_net}
)
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
- `WideDeepFeatureExtractor` - Wide & Deep
- `get_feature_extractor()` - Factory function

## Advanced Usage

### Compare Multiple Extractors

```python
from deep_kernel_gp import fit_dkgp, predict
from deep_kernel_gp.utils import compute_metrics

extractors = ['fc', 'fcbn', 'resnet', 'attention']
results = {}

for ext_type in extractors:
    print(f"\nTesting {ext_type}...")
    
    mll, gp, dkl, losses = fit_dkgp(
        X_train, y_train,
        extractor_type=ext_type,
        num_epochs=500,
        verbose=False
    )
    
    mean, std = predict(dkl, X_test, return_std=True)
    metrics = compute_metrics(y_test, mean, std)
    results[ext_type] = metrics
    
    print(f"  R²: {metrics['r2']:.4f}, RMSE: {metrics['rmse']:.4f}")

# Best extractor
best = max(results.items(), key=lambda x: x[1]['r2'])
print(f"\nBest: {best[0]} (R² = {best[1]['r2']:.4f})")
```

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
