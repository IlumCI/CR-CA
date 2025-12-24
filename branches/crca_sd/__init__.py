"""
CRCA-SD: CRCA for Socioeconomic Dynamics & Logistics

A constrained, stochastic, multi-objective model-predictive control system
for socioeconomic dynamics and logistics optimization.

Integrates CRCA (Causal Reasoning and Counterfactual Analysis) for:
- Causal scenario generation (not just random)
- Causal policy validation
- Understanding why policies work
"""

from typing import Optional, Any

__version__ = "0.1.0"

# Optional CRCA integration
try:
    # Import from parent directory's __init__.py
    import sys
    import os
    import importlib.util
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    root_init_path = os.path.join(parent_dir, "__init__.py")
    if os.path.exists(root_init_path):
        spec = importlib.util.spec_from_file_location("root_init", root_init_path)
        root_init = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(root_init)  # type: ignore
        _CRCAAgent = root_init.load_crca_agent()
        CRCA_AVAILABLE = True
    else:
        _CRCAAgent = None
        CRCA_AVAILABLE = False
except Exception:
    _CRCAAgent = None
    CRCA_AVAILABLE = False


def get_crca_agent(**kwargs) -> Optional[Any]:
    """
    Get CRCAAgent instance for causal reasoning (if available).
    
    Args:
        **kwargs: Arguments to pass to CRCAAgent constructor
        
    Returns:
        Optional[CRCAAgent]: CRCAAgent instance or None if not available
    """
    if not CRCA_AVAILABLE or _CRCAAgent is None:
        return None
    
    try:
        return _CRCAAgent(**kwargs)
    except Exception:
        return None
__all__ = [
    "StateVector",
    "ControlVector",
    "DynamicsModel",
    "ConstraintChecker",
    "ForwardSimulator",
    "ObjectiveVector",
    "CVaRComputer",
    "MPCSolver",
    "ScenarioGenerator",
    "StabilityEnforcer",
    "ParetoExtractor",
    "StateEstimator",
    "BoardMember",
    "Board",
    "Arbitration",
    "GovernanceSystem",
    "LogisticsNetwork",
    "Visualization",
    "Config",
    "RiskAssessment",
    "MetricsCollector",
    # Real-time components
    "DataAcquisition",
    "DataPipeline",
    "RealTimeStateEstimator",
    "RealTimeMonitor",
    "PolicyExecutor",
    "SafetyInterlocks",
    "ControlInterface",
    "AlertingSystem",
    "ComplianceSystem",
    "AccountabilitySystem",
    "RollbackSystem",
    "ModelAdaptation",
    "PerformanceFeedback",
    "FaultTolerance",
    # TUI
    "CRCA_SD_TUI",
    "TUIState",
    # Utils
    "get_formatter",
]

# Lazy imports to avoid circular dependencies
def _lazy_import(module_name: str, class_name: str):
    """Lazy import helper."""
    import importlib
    module = importlib.import_module(f"crca_sd.{module_name}")
    return getattr(module, class_name)

# Core components
def get_StateVector():
    """Get StateVector class."""
    from crca_sd.crca_sd_core import StateVector
    return StateVector

def get_ControlVector():
    """Get ControlVector class."""
    from crca_sd.crca_sd_core import ControlVector
    return ControlVector

def get_DynamicsModel():
    """Get DynamicsModel class."""
    from crca_sd.crca_sd_core import DynamicsModel
    return DynamicsModel

def get_ConstraintChecker():
    """Get ConstraintChecker class."""
    from crca_sd.crca_sd_core import ConstraintChecker
    return ConstraintChecker

def get_ForwardSimulator():
    """Get ForwardSimulator class."""
    from crca_sd.crca_sd_core import ForwardSimulator
    return ForwardSimulator


def get_formatter(md: bool = True):
    """
    Get Formatter instance from utils.
    
    Args:
        md: Enable markdown output rendering (default: True)
        
    Returns:
        Formatter instance or None if not available
    """
    try:
        from utils.formatter import Formatter
        return Formatter(md=md)
    except ImportError:
        return None

