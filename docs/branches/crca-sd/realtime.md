# Real-Time Control

Real-time control system for CRCA-SD.

## Overview

Real-time control enables live operation with data acquisition, state estimation, and automated policy execution.

## Mathematical Foundation

Real-time control uses state estimation:

$$\hat{\mathbf{x}}_t = E[\mathbf{x}_t | \mathbf{z}_{0:t}]$$

Where $\mathbf{z}_{0:t}$ are observations.

## Usage

```python
from branches.crca_sd.crca_sd_realtime import RealTimeController

controller = RealTimeController()
controller.start()
```

## Next Steps

- [TUI](tui.md) - Terminal user interface
