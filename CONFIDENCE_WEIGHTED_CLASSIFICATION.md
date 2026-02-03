# Confidence-Weighted Classification Examples

## Overview

Confidence-weighted classification allows you to assign different importance to training samples. This is useful when:

- Some samples are **noisier** than others
- Some samples are **more reliable** (e.g., expert-labeled vs crowd-sourced)
- Dealing with **imbalanced data** (assign higher weights to minority class)
- **Active learning** (higher weights for informative samples)
- **Temporal data** (higher weights for recent samples)

## Basic Usage

### Example 1: Noisy Labels

```python
import numpy as np
from deep_kernel_gp import fit_dkgp_classifier, predict_classifier

# Generate data
np.random.seed(42)
X_train = np.random.randn(200, 10)
y_train = (X_train[:, 0] + X_train[:, 1] > 0).astype(int)

# Add label noise to some samples
noisy_indices = np.random.choice(200, 40, replace=False)
y_train[noisy_indices] = 1 - y_train[noisy_indices]  # Flip labels

# Assign lower confidence to noisy samples
confidence_weights = np.ones(200)
confidence_weights[noisy_indices] = 0.3  # Lower weight for noisy samples

# Train with confidence weights
model, losses = fit_dkgp_classifier(
    X_train, 
    y_train,
    confidence_weights=confidence_weights,  # Pass weights
    num_classes=2,
    num_epochs=500,
    verbose=True
)

# Compare with standard training (no weights)
model_standard, _ = fit_dkgp_classifier(
    X_train, y_train,
    num_classes=2,
    num_epochs=500,
    verbose=False
)

# Test
X_test = np.random.randn(50, 10)
y_test = (X_test[:, 0] + X_test[:, 1] > 0).astype(int)

acc_weighted = (predict_classifier(model, X_test) == y_test).mean()
acc_standard = (predict_classifier(model_standard, X_test) == y_test).mean()

print(f"Accuracy with confidence weighting: {acc_weighted:.2%}")
print(f"Accuracy without confidence weighting: {acc_standard:.2%}")
```

### Example 2: Imbalanced Data

```python
# Generate imbalanced data
X_class0 = np.random.randn(180, 10)  # Majority: 90%
X_class1 = np.random.randn(20, 10)   # Minority: 10%

X_train = np.vstack([X_class0, X_class1])
y_train = np.array([0]*180 + [1]*20)

# Assign higher weights to minority class
confidence_weights = np.ones(200)
confidence_weights[y_train == 1] = 9.0  # Balance: 180/20 = 9

# Train
model, losses = fit_dkgp_classifier(
    X_train, y_train,
    confidence_weights=confidence_weights,
    num_classes=2
)
```

### Example 3: Multi-Class with Varying Confidence

```python
# Generate multi-class data
X_train = np.random.randn(300, 20)
y_train = np.random.randint(0, 4, 300)  # 4 classes

# Assign confidence based on some criterion
# E.g., confidence from labeler, prediction certainty, etc.
confidence_weights = np.random.uniform(0.5, 1.0, 300)

# Train multi-class classifier with weights
model, losses = fit_dkgp_classifier(
    X_train, y_train,
    num_classes=4,
    confidence_weights=confidence_weights,
    feature_dim=16,
    num_epochs=800
)

# Predict
X_test = np.random.randn(50, 20)
y_pred = predict_classifier(model, X_test)
y_proba = predict_classifier(model, X_test, return_proba=True)
```

## Advanced Usage

### Example 4: Active Learning with Confidence

```python
from deep_kernel_gp import fit_dkgp_classifier, predict_classifier
import numpy as np

# Initial labeled pool
X_labeled = np.random.randn(50, 15)
y_labeled = np.random.randint(0, 3, 50)
confidence_labeled = np.ones(50)

# Unlabeled pool
X_unlabeled = np.random.randn(200, 15)

# Active learning loop
for iteration in range(10):
    print(f"\n=== Iteration {iteration+1} ===")
    
    # Train model
    model, _ = fit_dkgp_classifier(
        X_labeled, y_labeled,
        num_classes=3,
        confidence_weights=confidence_labeled,
        verbose=False
    )
    
    # Query most uncertain samples
    probs = predict_classifier(model, X_unlabeled, return_proba=True)
    uncertainty = 1 - probs.max(axis=1)
    
    # Select top 5 most uncertain
    query_indices = np.argsort(uncertainty)[-5:]
    X_query = X_unlabeled[query_indices]
    
    # Simulate oracle labeling (in practice, ask human)
    y_query = np.random.randint(0, 3, 5)
    
    # Assign high confidence to actively queried samples
    confidence_query = np.ones(5) * 1.5  # Higher weight
    
    # Add to labeled pool
    X_labeled = np.vstack([X_labeled, X_query])
    y_labeled = np.append(y_labeled, y_query)
    confidence_labeled = np.append(confidence_labeled, confidence_query)
    
    # Remove from unlabeled pool
    X_unlabeled = np.delete(X_unlabeled, query_indices, axis=0)
    
    print(f"Labeled pool size: {len(X_labeled)}")
    print(f"Unlabeled pool size: {len(X_unlabeled)}")
```

### Example 5: Temporal Weighting (Recent Samples More Important)

```python
# Time series classification
n_samples = 300
X_train = np.random.randn(n_samples, 10)
y_train = np.random.randint(0, 2, n_samples)

# Assign exponentially decaying weights (recent samples weighted more)
time_indices = np.arange(n_samples)
decay_rate = 0.01
confidence_weights = np.exp(decay_rate * time_indices)
confidence_weights = confidence_weights / confidence_weights.max()  # Normalize

print(f"First sample weight: {confidence_weights[0]:.3f}")
print(f"Last sample weight: {confidence_weights[-1]:.3f}")

# Train
model, losses = fit_dkgp_classifier(
    X_train, y_train,
    confidence_weights=confidence_weights,
    num_classes=2
)
```

### Example 6: Combining with Different Feature Extractors

```python
# Use confidence weighting with ResNet extractor
model, losses = fit_dkgp_classifier(
    X_train, y_train,
    num_classes=4,
    confidence_weights=confidence_weights,
    extractor_type='resnet',
    extractor_kwargs={'hidden_dim': 256, 'num_blocks': 3},
    num_epochs=1000
)

# Use confidence weighting with Attention extractor
model, losses = fit_dkgp_classifier(
    X_train, y_train,
    num_classes=4,
    confidence_weights=confidence_weights,
    extractor_type='attention',
    extractor_kwargs={'hidden_dim': 128, 'num_heads': 4},
    num_epochs=1000
)
```

### Example 7: Manual Control of ELBO Type

```python
# Force use of ConfidenceWeightedELBO even with uniform weights
model, losses = fit_dkgp_classifier(
    X_train, y_train,
    num_classes=2,
    confidence_weights=np.ones(len(X_train)),
    use_confidence_weighted=True,  # Force confidence-weighted ELBO
    verbose=True
)

# Force standard ELBO even with non-uniform weights
model, losses = fit_dkgp_classifier(
    X_train, y_train,
    num_classes=2,
    confidence_weights=varying_weights,
    use_confidence_weighted=False,  # Force standard ELBO
    verbose=True
)
```

## How Confidence Weights Work

### Mathematical Formulation

**Standard ELBO:**
```
ELBO = Σᵢ log p(yᵢ|fᵢ) - KL[q(f)||p(f)]
```

**Confidence-Weighted ELBO:**
```
ELBO = Σᵢ wᵢ · log p(yᵢ|fᵢ) - KL[q(f)||p(f)]
```

Where:
- `wᵢ` = confidence weight for sample i (normalized)
- Higher weight → sample contributes more to loss
- Lower weight → sample contributes less (e.g., noisy labels)

### Weight Guidelines

| Weight Value | Interpretation | Use Case |
|--------------|----------------|----------|
| `0.0` | Ignore sample | Known bad labels |
| `0.1 - 0.5` | Low confidence | Noisy/uncertain labels |
| `1.0` | Normal confidence | Standard samples |
| `1.5 - 3.0` | High confidence | Expert labels, clean data |
| `5.0+` | Very high confidence | Critical samples |

### Auto-Selection Logic

The library automatically selects the appropriate ELBO:

```python
# If all weights are equal → VariationalELBO
confidence_weights = np.ones(100)
fit_dkgp_classifier(X, y, confidence_weights=weights)
# → Uses VariationalELBO

# If weights vary → ConfidenceWeightedELBO
confidence_weights = np.array([1.0, 0.5, 1.0, 0.3, ...])
fit_dkgp_classifier(X, y, confidence_weights=weights)
# → Uses ConfidenceWeightedELBO
```

## Best Practices

### 1. Normalize Weights
```python
# Good: Weights around 1.0
weights = confidence_scores / confidence_scores.mean()

# Bad: Extreme weights
weights = [0.01, 100.0, 0.001, ...]  # Avoid!
```

### 2. Don't Over-Weight
```python
# Good: Reasonable range
weights = np.clip(raw_weights, 0.1, 3.0)

# Bad: Too extreme
weights[minority_class] = 1000.0  # Too high!
```

### 3. Monitor Training
```python
model, losses = fit_dkgp_classifier(
    X, y,
    confidence_weights=weights,
    verbose=True  # Watch for issues
)

# Check if loss behaves normally
import matplotlib.pyplot as plt
plt.plot(losses)
plt.title('Training Loss')
plt.show()
```

### 4. Validate on Clean Data
```python
# Always test on clean, balanced test set
# Don't apply confidence weights to test data!

y_pred = predict_classifier(model, X_test_clean)
accuracy = (y_pred == y_test_clean).mean()
```

## Comparison: Weighted vs Unweighted

```python
import numpy as np
from deep_kernel_gp import fit_dkgp_classifier, predict_classifier
from sklearn.metrics import accuracy_score, classification_report

# Setup
X_train = np.random.randn(200, 10)
y_train = np.random.randint(0, 2, 200)

# Add noise
noisy_idx = np.random.choice(200, 50, replace=False)
y_train[noisy_idx] = 1 - y_train[noisy_idx]

# Weights
weights = np.ones(200)
weights[noisy_idx] = 0.2

# Train both
model_weighted, _ = fit_dkgp_classifier(
    X_train, y_train,
    confidence_weights=weights,
    verbose=False
)

model_standard, _ = fit_dkgp_classifier(
    X_train, y_train,
    verbose=False
)

# Test
X_test = np.random.randn(100, 10)
y_test = np.random.randint(0, 2, 100)

y_pred_weighted = predict_classifier(model_weighted, X_test)
y_pred_standard = predict_classifier(model_standard, X_test)

print("Weighted Model:")
print(classification_report(y_test, y_pred_weighted))
print("\nStandard Model:")
print(classification_report(y_test, y_pred_standard))
```

## Summary

**Use confidence weights when:**
- Dealing with noisy labels
- Handling imbalanced data
- Incorporating label reliability
- Temporal dynamics matter
- Active learning scenarios

**Don't use confidence weights when:**
- All samples equally reliable
- Clean, balanced dataset
- Unsure about weight values
- Small dataset (<50 samples)

🎯 **Key takeaway:** Confidence weighting helps the model focus on reliable samples while still learning from uncertain ones!
