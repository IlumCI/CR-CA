# Monitoring & Drift Detection

Monitoring systems and drift detection for policy engines.

## Overview

Monitoring systems track policy performance and detect when system behavior deviates from expected patterns.

## Drift Detection

Drift detection identifies when system behavior deviates from expected patterns.

### Mathematical Foundation

Drift is detected when:

$$D(\mathbf{x}_t, \mathbf{x}_{expected}) > \theta$$

Where:
- $D$: Distance metric
- $\mathbf{x}_t$: Current state
- $\mathbf{x}_{expected}$: Expected state
- $\theta$: Drift threshold

### Distance Metrics

**Euclidean Distance**:
$$D_{Euclidean} = \|\mathbf{x}_t - \mathbf{x}_{expected}\|_2$$

**Mahalanobis Distance**:
$$D_{Mahalanobis} = \sqrt{(\mathbf{x}_t - \mathbf{x}_{expected})' \Sigma^{-1} (\mathbf{x}_t - \mathbf{x}_{expected})}$$

### Usage

```python
from templates.drift_detection import DriftDetector

detector = DriftDetector(threshold=0.1)
drift_detected = detector.check(current_state, expected_state)
```

## Next Steps

- [Overview](overview.md) - Return to overview
