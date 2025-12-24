# CRCA-SD TUI (Terminal User Interface)

## Overview

The CRCA-SD TUI provides a rich, interactive terminal interface for monitoring and controlling CRCA-SD systems in real-time.

## Features

- **Live Updating Dashboard**: Real-time updates every second
- **State Visualization**: Current economic state with color-coded metrics
- **Vision Progress Tracking**: Progress bars and metrics towards goals
- **Policy Display**: Current policy with budget allocation visualization
- **System Status**: Health, executions, pending approvals
- **Alert Monitoring**: Real-time alerts with severity indicators
- **Execution History**: Recent policy executions

## Usage

### Basic Usage

```python
from crca_sd.crca_sd_tui import CRCA_SD_TUI
from crca_sd.crca_sd_core import StateVector, ControlVector

# Create TUI
tui = CRCA_SD_TUI(title="My Control System")

# Update state
tui.update_state(
    current_state=state_vector,
    vision_progress={"overall": 0.45, "unemployment": 0.60, "gdp": 0.50},
    system_status={"is_running": True, "n_executions": 5},
    policy=current_policy
)

# Run with live updates
def update_callback(tui_instance):
    # Update TUI state from your controller
    tui_instance.update_state(...)

tui.run_live(update_callback, refresh_rate=1.0)
```

### With Pridnestrovia Example

The Pridnestrovia real-time controller automatically uses TUI if `rich` is installed:

```bash
python3 examples/pridnestrovia_realtime.py
```

If `rich` is not available, it falls back to text dashboard.

## Layout

The TUI is organized into panels:

```
┌─────────────────────────────────────────────────────────┐
│ Header: Title, Uptime, Last Update                      │
├──────────────────┬───────────────────────────────────────┤
│                  │                                      │
│  Current State   │  System Status                      │
│  (Key Metrics)   │  (Health, Executions)                │
├──────────────────┼───────────────────────────────────────┤
│  Vision Progress │  Alerts                               │
│  (Progress Bars) │  (Recent Alerts)                      │
├──────────────────┼───────────────────────────────────────┤
│  Current Policy  │  Execution History                    │
│  (Budget Shares) │  (Recent Executions)                  │
├──────────────────┴───────────────────────────────────────┤
│ Footer: Controls (Q=Quit, R=Refresh, etc.)               │
└─────────────────────────────────────────────────────────┘
```

## Color Coding

- **Green**: Good/Healthy/Normal
- **Yellow**: Warning/Degraded
- **Red**: Critical/Unhealthy/Alert
- **Cyan**: Information/Data
- **Magenta**: Policy/Controls

## Requirements

- `rich>=13.0` (already in requirements.txt)

## Installation

```bash
pip install rich
```

## API

### CRCA_SD_TUI

Main TUI class.

**Methods:**
- `update_state(...)`: Update TUI state
- `render()`: Render the layout
- `run_live(update_callback, refresh_rate)`: Run with live updates

### TUIState

Data class for TUI state.

**Fields:**
- `current_state`: StateVector
- `vision_progress`: Dict[str, float]
- `vision_target`: StateVector
- `system_status`: Dict[str, Any]
- `execution_history`: List[Dict[str, Any]]
- `pending_approvals`: List[Dict[str, Any]]
- `alerts`: List[Dict[str, Any]]
- `policy`: ControlVector

## Example

See `examples/pridnestrovia_realtime.py` for a complete example.

