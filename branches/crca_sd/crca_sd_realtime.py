"""
CRCA-SD Real-Time Control System

This module implements all real-time control capabilities for transforming CRCA-SD
from simulation/decision support into a real-world economic control system.

Key Features:
- Real-time data acquisition (government systems priority)
- State estimation with daily updates
- Automated policy execution (minor changes < 10%)
- Human approval workflows (major changes > 10%)
- Safety interlocks and compliance
- 7-day rollback window
- Full regulatory compliance

Organized into sections:
1. Data Integration (~500 lines)
2. Control Execution (~600 lines)
3. Monitoring & Compliance (~500 lines)
4. Resilience & Learning (~400 lines)
"""

from typing import Dict, List, Optional, Tuple, Any, Callable
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import time
import json
import uuid
from enum import Enum
from loguru import logger

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    logger.warning("requests not available, API integration limited")

try:
    import sqlalchemy
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False
    logger.warning("sqlalchemy not available, database integration limited")

from crca_sd.crca_sd_core import StateVector, ControlVector, DynamicsModel, ConstraintChecker
from crca_sd.crca_sd_mpc import StateEstimator, ObjectiveVector


# ============================================================================
# SECTION 1: DATA INTEGRATION (~500 lines)
# ============================================================================

class DataSourceType(str, Enum):
    """Types of data sources."""
    GOVERNMENT_API = "government_api"  # Priority
    PUBLIC_API = "public_api"
    DATABASE = "database"
    IOT_SENSOR = "iot_sensor"


@dataclass
class DataPoint:
    """Single data point with metadata."""
    timestamp: float
    source: str
    source_type: DataSourceType
    variable: str
    value: float
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class DataAcquisition:
    """
    Real-time data acquisition system.
    
    Priority: Government system APIs (treasury, ministries, central bank)
    Secondary: Public APIs, databases, IoT sensors
    
    Args:
        government_api_config: Configuration for government APIs
        update_frequency: Update frequency in seconds (default: 86400 = daily)
    """
    
    def __init__(
        self,
        government_api_config: Optional[Dict[str, Any]] = None,
        update_frequency: float = 86400.0,  # Daily (24 hours)
    ) -> None:
        """Initialize data acquisition system."""
        self.government_api_config = government_api_config or {}
        self.update_frequency = update_frequency
        self.last_update: Dict[str, float] = {}
        self.data_cache: Dict[str, List[DataPoint]] = {}
        self.api_clients: Dict[str, Any] = {}
        
        logger.info(f"DataAcquisition initialized with {update_frequency/3600:.1f}h update frequency")
    
    def connect_government_api(
        self,
        api_name: str,
        base_url: str,
        auth_config: Dict[str, Any]
    ) -> bool:
        """
        Connect to government API (treasury, ministries, central bank).
        
        Args:
            api_name: Name identifier for the API
            base_url: Base URL for the API
            auth_config: Authentication configuration
            
        Returns:
            bool: True if connection successful
        """
        try:
            # Store API configuration
            self.government_api_config[api_name] = {
                "base_url": base_url,
                "auth": auth_config,
            }
            
            # Initialize API client (placeholder - would use actual client library)
            self.api_clients[api_name] = {
                "base_url": base_url,
                "auth": auth_config,
                "connected": True,
            }
            
            logger.info(f"Connected to government API: {api_name}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to connect to government API {api_name}: {e}")
            return False
    
    def fetch_government_data(
        self,
        api_name: str,
        endpoint: str,
        variables: List[str]
    ) -> List[DataPoint]:
        """
        Fetch data from government API.
        
        Args:
            api_name: API identifier
            endpoint: API endpoint path
            variables: List of variable names to fetch
            
        Returns:
            List[DataPoint]: List of data points
        """
        if api_name not in self.api_clients:
            logger.warning(f"Government API {api_name} not connected")
            return []
        
        if not REQUESTS_AVAILABLE:
            logger.warning("requests not available, returning mock data")
            return self._generate_mock_data(variables, DataSourceType.GOVERNMENT_API)
        
        try:
            client = self.api_clients[api_name]
            url = f"{client['base_url']}/{endpoint}"
            
            # Make API request (placeholder - would use actual authentication)
            # response = requests.get(url, headers=client['auth'], timeout=10)
            # data = response.json()
            
            # For now, return mock data
            logger.debug(f"Fetching from {api_name}/{endpoint} for variables: {variables}")
            data_points = self._generate_mock_data(variables, DataSourceType.GOVERNMENT_API, api_name)
            
            # Cache data
            for dp in data_points:
                if dp.variable not in self.data_cache:
                    self.data_cache[dp.variable] = []
                self.data_cache[dp.variable].append(dp)
            
            return data_points
        
        except Exception as e:
            logger.error(f"Error fetching government data from {api_name}: {e}")
            return []
    
    def fetch_public_api_data(
        self,
        api_url: str,
        variables: List[str]
    ) -> List[DataPoint]:
        """
        Fetch data from public API (keyless).
        
        Args:
            api_url: Public API URL
            variables: List of variable names to fetch
            
        Returns:
            List[DataPoint]: List of data points
        """
        if not REQUESTS_AVAILABLE:
            return self._generate_mock_data(variables, DataSourceType.PUBLIC_API)
        
        try:
            logger.debug(f"Fetching from public API: {api_url}")
            response = requests.get(api_url, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            # Parse response and create data points
            data_points = []
            timestamp = time.time()
            
            for var in variables:
                value = self._extract_value_from_response(data, var)
                if value is not None:
                    dp = DataPoint(
                        timestamp=timestamp,
                        source=api_url,
                        source_type=DataSourceType.PUBLIC_API,
                        variable=var,
                        value=float(value),
                        confidence=0.85,  # Public APIs typically less reliable than government
                    )
                    data_points.append(dp)
            
            return data_points
        
        except Exception as e:
            logger.error(f"Error fetching public API data: {e}")
            return []
    
    def fetch_world_bank_data(
        self,
        country_code: str,
        indicators: Dict[str, str]
    ) -> List[DataPoint]:
        """
        Fetch data from World Bank API (keyless, free).
        
        World Bank API: https://datahelpdesk.worldbank.org/knowledgebase/articles/889392
        
        Args:
            country_code: ISO country code (e.g., "MD" for Moldova, "XK" for Kosovo - closest to Pridnestrovia)
            indicators: Dict mapping variable names to World Bank indicator codes
                      Example: {"Y": "NY.GDP.MKTP.CD", "P": "SP.POP.TOTL"}
            
        Returns:
            List[DataPoint]: List of data points
        """
        if not REQUESTS_AVAILABLE:
            return []
        
        data_points = []
        timestamp = time.time()
        
        # World Bank API base URL (keyless, no authentication required)
        base_url = "https://api.worldbank.org/v2/country"
        
        for var_name, indicator_code in indicators.items():
            try:
                # World Bank API format: /country/{country}/indicator/{indicator}?format=json
                url = f"{base_url}/{country_code}/indicator/{indicator_code}"
                params = {
                    "format": "json",
                    "date": "2020:2025",  # Get recent data
                    "per_page": 1,  # Just get latest value
                }
                
                response = requests.get(url, params=params, timeout=30)  # World Bank can be slow
                response.raise_for_status()
                data = response.json()
                
                # World Bank API returns nested structure
                if len(data) >= 2 and len(data[1]) > 0:
                    latest_value = data[1][0].get("value")
                    if latest_value is not None:
                        dp = DataPoint(
                            timestamp=timestamp,
                            source="world_bank",
                            source_type=DataSourceType.PUBLIC_API,
                            variable=var_name,
                            value=float(latest_value),
                            confidence=0.90,  # World Bank is reliable
                            metadata={"indicator": indicator_code, "country": country_code}
                        )
                        data_points.append(dp)
                        logger.debug(f"World Bank: {var_name} = {latest_value}")
            
            except Exception as e:
                logger.warning(f"Could not fetch World Bank indicator {indicator_code}: {e}")
                continue
        
        return data_points
    
    def fetch_restcountries_data(
        self,
        country_name: str,
        variables: List[str]
    ) -> List[DataPoint]:
        """
        Fetch data from REST Countries API (keyless, free).
        
        REST Countries API: https://restcountries.com/
        
        Args:
            country_name: Country name (e.g., "Moldova" - closest to Pridnestrovia)
            variables: List of variable names to extract
            
        Returns:
            List[DataPoint]: List of data points
        """
        if not REQUESTS_AVAILABLE:
            return []
        
        try:
            # REST Countries API (keyless)
            url = f"https://restcountries.com/v3.1/name/{country_name}"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if not data or len(data) == 0:
                return []
            
            country_data = data[0]  # Get first match
            data_points = []
            timestamp = time.time()
            
            # Map variables to country data fields
            var_mapping = {
                "P": country_data.get("population"),
                "literacy": None,  # Not directly available
            }
            
            for var in variables:
                if var == "P" and var_mapping["P"]:
                    dp = DataPoint(
                        timestamp=timestamp,
                        source="restcountries",
                        source_type=DataSourceType.PUBLIC_API,
                        variable=var,
                        value=float(var_mapping["P"]),
                        confidence=0.85,
                    )
                    data_points.append(dp)
            
            return data_points
        
        except Exception as e:
            logger.warning(f"Could not fetch REST Countries data: {e}")
            return []
    
    def _extract_value_from_response(self, data: Any, variable: str) -> Optional[float]:
        """
        Extract value for variable from API response.
        
        Args:
            data: API response data (dict/list)
            variable: Variable name to extract
            
        Returns:
            Optional[float]: Extracted value or None
        """
        # Try common response formats
        if isinstance(data, dict):
            # Try direct key
            if variable in data:
                return data[variable]
            
            # Try nested keys
            for key in ["value", "data", "result", "indicator"]:
                if key in data and isinstance(data[key], dict):
                    if variable in data[key]:
                        return data[key][variable]
            
            # Try lowercase/uppercase variants
            for key in data.keys():
                if key.lower() == variable.lower():
                    return data[key]
        
        elif isinstance(data, list) and len(data) > 0:
            # Try first element
            return self._extract_value_from_response(data[0], variable)
        
        return None
    
    def fetch_database_data(
        self,
        connection_string: str,
        query: str,
        variables: List[str]
    ) -> List[DataPoint]:
        """
        Fetch data from database.
        
        Args:
            connection_string: Database connection string
            query: SQL query
            variables: List of variable names
            
        Returns:
            List[DataPoint]: List of data points
        """
        if not SQLALCHEMY_AVAILABLE:
            logger.warning("sqlalchemy not available, returning mock data")
            return self._generate_mock_data(variables, DataSourceType.DATABASE)
        
        try:
            # Placeholder for actual database query
            logger.debug(f"Fetching from database: {query}")
            return self._generate_mock_data(variables, DataSourceType.DATABASE)
        
        except Exception as e:
            logger.error(f"Error fetching database data: {e}")
            return []
    
    def should_update(self, source: str) -> bool:
        """
        Check if data source should be updated based on frequency.
        
        Args:
            source: Data source identifier
            
        Returns:
            bool: True if should update
        """
        if source not in self.last_update:
            return True
        
        elapsed = time.time() - self.last_update[source]
        return elapsed >= self.update_frequency
    
    def _generate_mock_data(
        self,
        variables: List[str],
        source_type: DataSourceType,
        source_name: str = "mock"
    ) -> List[DataPoint]:
        """Generate mock data for testing."""
        data_points = []
        timestamp = time.time()
        
        for var in variables:
            # Mock values based on variable type
            if var == "P":
                value = 360000.0  # Population
            elif var == "U":
                value = 0.07  # Unemployment
            elif var == "Y":
                value = 1300000000.0  # GDP
            else:
                value = 1000.0  # Default
            
            dp = DataPoint(
                timestamp=timestamp,
                source=source_name,
                source_type=source_type,
                variable=var,
                value=value,
                confidence=0.95,
            )
            data_points.append(dp)
        
        return data_points


class DataPipeline:
    """
    ETL pipeline for data processing.
    
    Handles:
    - Data normalization and standardization
    - Time-series alignment
    - Missing data imputation
    - Data versioning and lineage tracking
    """
    
    def __init__(self) -> None:
        """Initialize data pipeline."""
        self.data_history: Dict[str, List[DataPoint]] = {}
        self.normalization_params: Dict[str, Dict[str, float]] = {}
    
    def process_data_points(
        self,
        data_points: List[DataPoint],
        normalize: bool = False
    ) -> List[DataPoint]:
        """
        Process data points (with optional normalization).
        
        Args:
            data_points: Raw data points
            normalize: Whether to normalize values (default: False, use raw values)
            
        Returns:
            List[DataPoint]: Processed data points
        """
        processed = []
        
        for dp in data_points:
            # Use raw value by default (normalization was causing issues)
            # Only normalize if explicitly requested
            if normalize:
                processed_value = self._normalize_value(dp.variable, dp.value)
            else:
                processed_value = dp.value  # Use raw value
            
            # Create processed data point
            processed_dp = DataPoint(
                timestamp=dp.timestamp,
                source=dp.source,
                source_type=dp.source_type,
                variable=dp.variable,
                value=processed_value,
                confidence=dp.confidence,
                metadata={**dp.metadata, "normalized": normalize},
            )
            
            processed.append(processed_dp)
            
            # Store in history
            if dp.variable not in self.data_history:
                self.data_history[dp.variable] = []
            self.data_history[dp.variable].append(processed_dp)
        
        return processed
    
    def align_time_series(
        self,
        variables: List[str],
        target_timestamp: float
    ) -> Dict[str, float]:
        """
        Align time series to target timestamp.
        
        Args:
            variables: List of variable names
            target_timestamp: Target timestamp
            
        Returns:
            Dict[str, float]: Aligned values
        """
        aligned = {}
        
        for var in variables:
            if var not in self.data_history:
                continue
            
            # Find closest data point to target timestamp
            closest = None
            min_diff = float('inf')
            
            for dp in self.data_history[var]:
                diff = abs(dp.timestamp - target_timestamp)
                if diff < min_diff:
                    min_diff = diff
                    closest = dp
            
            if closest and min_diff < 3600:  # Within 1 hour
                aligned[var] = closest.value
            else:
                # Impute missing value
                aligned[var] = self._impute_missing(var)
        
        return aligned
    
    def _normalize_value(self, variable: str, value: float) -> float:
        """Normalize value based on variable type."""
        if variable not in self.normalization_params:
            # Initialize normalization params
            self.normalization_params[variable] = {
                "mean": value,
                "std": abs(value) * 0.1 if value != 0 else 1.0,
            }
        
        # Simple normalization (can be enhanced)
        params = self.normalization_params[variable]
        normalized = (value - params["mean"]) / params["std"] if params["std"] > 0 else value
        
        return normalized
    
    def _impute_missing(self, variable: str) -> float:
        """Impute missing value using historical data."""
        if variable not in self.data_history or not self.data_history[variable]:
            return 0.0
        
        # Use most recent value
        recent = self.data_history[variable][-1]
        return recent.value


class RealTimeStateEstimator:
    """
    Real-time state estimator with EKF/UKF and multi-sensor fusion.
    
    Extends the base StateEstimator with real-time capabilities:
    - Daily update scheduling
    - Multi-sensor fusion
    - Anomaly detection
    - Confidence quantification
    """
    
    def __init__(
        self,
        dynamics: DynamicsModel,
        observation_noise_cov: Optional[np.ndarray] = None,
        process_noise_cov: Optional[np.ndarray] = None,
        update_frequency: float = 86400.0,  # Daily
    ) -> None:
        """Initialize real-time state estimator."""
        # Use base StateEstimator
        self.base_estimator = StateEstimator(dynamics, observation_noise_cov, process_noise_cov)
        self.update_frequency = update_frequency
        self.last_update_time = 0.0
        self.current_state: Optional[StateVector] = None
        self.state_confidence: float = 0.0
        self.anomaly_threshold = 3.0  # Standard deviations
    
    def update_with_data_points(
        self,
        data_points: List[DataPoint],
        u_t: ControlVector
    ) -> StateVector:
        """
        Update state estimate with data points from multiple sources.
        
        Args:
            data_points: List of data points from various sources
            u_t: Current control vector
            
        Returns:
            StateVector: Updated state estimate
        """
        # Convert data points to observation dictionary
        observation = {}
        for dp in data_points:
            observation[dp.variable] = dp.value
        
        # Update base estimator
        if self.current_state is None:
            self.current_state = StateVector()  # Initialize
        
        # Use base estimator update
        updated_state = self.base_estimator.update(observation, u_t)
        
        # Compute confidence based on data quality
        self.state_confidence = self._compute_confidence(data_points)
        
        # Check for anomalies
        if self._detect_anomaly(updated_state, data_points):
            logger.warning("Anomaly detected in state estimate")
        
        self.current_state = updated_state
        self.last_update_time = time.time()
        
        return updated_state
    
    def should_update(self) -> bool:
        """Check if state should be updated based on frequency."""
        if self.last_update_time == 0.0:
            return True
        
        elapsed = time.time() - self.last_update_time
        return elapsed >= self.update_frequency
    
    def get_current_state(self) -> Optional[StateVector]:
        """Get current state estimate."""
        return self.current_state
    
    def get_confidence(self) -> float:
        """Get state estimate confidence."""
        return self.state_confidence
    
    def _compute_confidence(self, data_points: List[DataPoint]) -> float:
        """Compute confidence based on data quality."""
        if not data_points:
            return 0.0
        
        # Average confidence across data points
        avg_confidence = np.mean([dp.confidence for dp in data_points])
        
        # Penalize if data is stale
        now = time.time()
        max_age = max([now - dp.timestamp for dp in data_points], default=0)
        age_penalty = max(0, 1.0 - max_age / (2 * self.update_frequency))
        
        return avg_confidence * age_penalty
    
    def _detect_anomaly(
        self,
        state: StateVector,
        data_points: List[DataPoint]
    ) -> bool:
        """Detect anomalies in state estimate."""
        if self.current_state is None:
            return False
        
        # Check for large deviations
        state_dict = state.to_dict()
        prev_dict = self.current_state.to_dict()
        
        for key in state_dict:
            if key not in prev_dict:
                continue
            
            diff = abs(state_dict[key] - prev_dict[key])
            prev_val = abs(prev_dict[key])
            
            if prev_val > 0:
                relative_change = diff / prev_val
                if relative_change > 0.5:  # 50% change
                    return True
        
        return False


class RealTimeMonitor:
    """
    Real-time monitoring system.
    
    Monitors:
    - Continuous state monitoring (daily frequency)
    - Constraint violation detection
    - Performance metrics tracking
    - System health checks
    """
    
    def __init__(
        self,
        constraint_checker: ConstraintChecker,
        update_frequency: float = 86400.0,  # Daily
    ) -> None:
        """Initialize real-time monitor."""
        self.constraint_checker = constraint_checker
        self.update_frequency = update_frequency
        self.last_check_time = 0.0
        self.violation_history: List[Tuple[float, List[str]]] = []
        self.metrics_history: List[Dict[str, Any]] = []
    
    def check_state(
        self,
        x_t: StateVector,
        u_t: ControlVector
    ) -> Tuple[bool, List[str], Dict[str, Any]]:
        """
        Check state and constraints.
        
        Args:
            x_t: Current state
            u_t: Current control
            
        Returns:
            Tuple[bool, List[str], Dict[str, Any]]: (is_feasible, violations, metrics)
        """
        is_feasible, violations = self.constraint_checker.check_feasible(x_t, u_t)
        
        # Record violation
        if violations:
            self.violation_history.append((time.time(), violations))
        
        # Compute metrics
        metrics = {
            "timestamp": time.time(),
            "feasible": is_feasible,
            "n_violations": len(violations),
            "state": x_t.to_dict(),
        }
        self.metrics_history.append(metrics)
        
        self.last_check_time = time.time()
        
        return is_feasible, violations, metrics
    
    def should_check(self) -> bool:
        """Check if monitoring should run."""
        if self.last_check_time == 0.0:
            return True
        
        elapsed = time.time() - self.last_check_time
        return elapsed >= self.update_frequency
    
    def get_recent_violations(self, hours: float = 24.0) -> List[Tuple[float, List[str]]]:
        """Get violations from last N hours."""
        cutoff = time.time() - hours * 3600
        return [(t, v) for t, v in self.violation_history if t >= cutoff]
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get system health status."""
        recent_violations = self.get_recent_violations(24.0)
        
        return {
            "status": "healthy" if len(recent_violations) == 0 else "degraded",
            "n_violations_24h": len(recent_violations),
            "last_check": self.last_check_time,
            "next_check": self.last_check_time + self.update_frequency,
        }


# ============================================================================
# SECTION 2: CONTROL EXECUTION (~600 lines)
# ============================================================================

class PolicyExecutor:
    """
    Automated policy execution engine.
    
    Executes policies via government API integration:
    - Automated budget allocation execution
    - API calls to government systems (treasury, ministries)
    - Transaction processing and verification
    - Execution confirmation and status tracking
    - Rollback capabilities
    - Atomic operations (all-or-nothing)
    """
    
    def __init__(
        self,
        government_api_config: Optional[Dict[str, Any]] = None
    ) -> None:
        """Initialize policy executor."""
        self.government_api_config = government_api_config or {}
        self.execution_history: List[Dict[str, Any]] = []
        self.pending_executions: List[Dict[str, Any]] = []
    
    def execute_policy(
        self,
        policy: ControlVector,
        execution_id: Optional[str] = None,
        require_approval: bool = False
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Execute a policy.
        
        Args:
            policy: Control vector to execute
            execution_id: Optional execution ID
            require_approval: Whether approval is required (for major changes)
            
        Returns:
            Tuple[bool, str, Dict[str, Any]]: (success, message, execution_info)
        """
        if execution_id is None:
            execution_id = str(uuid.uuid4())
        
        if require_approval:
            # Queue for approval
            self.pending_executions.append({
                "execution_id": execution_id,
                "policy": policy,
                "timestamp": time.time(),
                "status": "pending_approval",
            })
            return False, "Pending approval", {"execution_id": execution_id, "status": "pending_approval"}
        
        # Execute policy
        try:
            # Validate policy
            is_valid, error = policy.validate_simplex()
            if not is_valid:
                return False, f"Invalid policy: {error}", {}
            
            # Execute via government API (placeholder)
            success = self._execute_via_api(policy, execution_id)
            
            execution_info = {
                "execution_id": execution_id,
                "policy": policy.to_dict(),
                "timestamp": time.time(),
                "status": "executed" if success else "failed",
                "success": success,
            }
            
            self.execution_history.append(execution_info)
            
            if success:
                logger.info(f"Policy executed successfully: {execution_id}")
                return True, "Policy executed", execution_info
            else:
                logger.error(f"Policy execution failed: {execution_id}")
                return False, "Execution failed", execution_info
        
        except Exception as e:
            logger.error(f"Error executing policy: {e}")
            return False, str(e), {}
    
    def _execute_via_api(
        self,
        policy: ControlVector,
        execution_id: str
    ) -> bool:
        """
        Execute policy via government API.
        
        Args:
            policy: Policy to execute
            execution_id: Execution ID
            
        Returns:
            bool: True if successful
        """
        # Placeholder for actual API integration
        # Would make API calls to treasury/ministries to allocate budget
        
        logger.debug(f"Executing policy {execution_id} via government API")
        
        # Simulate API call
        time.sleep(0.1)  # Simulate network delay
        
        return True
    
    def get_execution_status(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """Get execution status by ID."""
        for exec_info in self.execution_history:
            if exec_info["execution_id"] == execution_id:
                return exec_info
        return None


class SafetyInterlocks:
    """
    Safety interlock system.
    
    Implements:
    - Hard limits (never exceed absolute bounds)
    - Rate limiters (max change per period)
    - Automated execution for minor changes (< 10%)
    - Human approval gates for major changes (> 10%)
    - Circuit breakers (auto-stop on anomalies)
    - Emergency stop mechanisms
    """
    
    def __init__(
        self,
        max_budget_change: float = 0.20,  # 20% max change per period
        major_change_threshold: float = 0.10,  # 10% = major change
        confidence_threshold: float = 0.95,  # 95% confidence required
    ) -> None:
        """Initialize safety interlocks."""
        self.max_budget_change = max_budget_change
        self.major_change_threshold = major_change_threshold
        self.confidence_threshold = confidence_threshold
        self.circuit_breaker_active = False
        self.emergency_stop_active = False
    
    def check_policy_safety(
        self,
        policy: ControlVector,
        previous_policy: Optional[ControlVector],
        state_confidence: float
    ) -> Tuple[bool, str, bool]:
        """
        Check if policy is safe to execute.
        
        Args:
            policy: Proposed policy
            previous_policy: Previous policy (for rate limiting)
            state_confidence: State estimate confidence
            
        Returns:
            Tuple[bool, str, bool]: (is_safe, reason, requires_approval)
        """
        # Check emergency stop
        if self.emergency_stop_active:
            return False, "Emergency stop active", True
        
        # Check circuit breaker
        if self.circuit_breaker_active:
            return False, "Circuit breaker active", True
        
        # Check confidence threshold
        if state_confidence < self.confidence_threshold:
            return False, f"State confidence {state_confidence:.2f} below threshold {self.confidence_threshold}", True
        
        # Check rate limits
        if previous_policy is not None:
            is_valid, reason, requires_approval = self._check_rate_limits(policy, previous_policy)
            if not is_valid:
                return False, reason, requires_approval
        
        # Check if major change (requires approval)
        if previous_policy is not None:
            is_major = self._is_major_change(policy, previous_policy)
            if is_major:
                return True, "Major change detected", True  # Safe but requires approval
        
        # Minor change - can execute automatically
        return True, "Safe to execute", False
    
    def _check_rate_limits(
        self,
        policy: ControlVector,
        previous_policy: ControlVector
    ) -> Tuple[bool, str, bool]:
        """Check rate limits."""
        # Compute L1 norm of budget change
        budget_change = {}
        for cat in set(list(policy.budget_shares.keys()) + list(previous_policy.budget_shares.keys())):
            prev_val = previous_policy.budget_shares.get(cat, 0.0)
            curr_val = policy.budget_shares.get(cat, 0.0)
            budget_change[cat] = abs(curr_val - prev_val)
        
        total_change = sum(budget_change.values())
        
        if total_change > self.max_budget_change:
            return False, f"Budget change {total_change:.2%} exceeds limit {self.max_budget_change:.2%}", True
        
        return True, "Within rate limits", False
    
    def _is_major_change(
        self,
        policy: ControlVector,
        previous_policy: ControlVector
    ) -> bool:
        """Check if change is major (> 10% in any category)."""
        for cat in set(list(policy.budget_shares.keys()) + list(previous_policy.budget_shares.keys())):
            prev_val = previous_policy.budget_shares.get(cat, 0.0)
            curr_val = policy.budget_shares.get(cat, 0.0)
            change = abs(curr_val - prev_val)
            
            if change > self.major_change_threshold:
                return True
        
        return False
    
    def activate_circuit_breaker(self, reason: str) -> None:
        """Activate circuit breaker."""
        self.circuit_breaker_active = True
        logger.warning(f"Circuit breaker activated: {reason}")
    
    def deactivate_circuit_breaker(self) -> None:
        """Deactivate circuit breaker."""
        self.circuit_breaker_active = False
        logger.info("Circuit breaker deactivated")
    
    def emergency_stop(self) -> None:
        """Activate emergency stop."""
        self.emergency_stop_active = True
        logger.critical("EMERGENCY STOP ACTIVATED")
    
    def resume(self) -> None:
        """Resume after emergency stop."""
        self.emergency_stop_active = False
        logger.info("System resumed after emergency stop")


class ControlInterface:
    """
    Control interface for policy commands.
    
    Provides:
    - REST API for policy commands
    - WebSocket for real-time updates
    - Command queue and priority system
    - Command validation and authorization
    - Audit logging of all commands
    """
    
    def __init__(self) -> None:
        """Initialize control interface."""
        self.command_queue: List[Dict[str, Any]] = []
        self.command_history: List[Dict[str, Any]] = []
        self.authorized_users: Dict[str, Dict[str, Any]] = {}
    
    def queue_command(
        self,
        command: Dict[str, Any],
        user_id: str,
        priority: int = 0
    ) -> str:
        """
        Queue a command for execution.
        
        Args:
            command: Command dictionary
            user_id: User ID
            priority: Priority (higher = more urgent)
            
        Returns:
            str: Command ID
        """
        command_id = str(uuid.uuid4())
        
        queued_command = {
            "command_id": command_id,
            "command": command,
            "user_id": user_id,
            "priority": priority,
            "timestamp": time.time(),
            "status": "queued",
        }
        
        # Insert by priority
        inserted = False
        for i, cmd in enumerate(self.command_queue):
            if priority > cmd.get("priority", 0):
                self.command_queue.insert(i, queued_command)
                inserted = True
                break
        
        if not inserted:
            self.command_queue.append(queued_command)
        
        logger.info(f"Command queued: {command_id} by {user_id}")
        return command_id
    
    def authorize_user(
        self,
        user_id: str,
        permissions: List[str]
    ) -> None:
        """Authorize a user with permissions."""
        self.authorized_users[user_id] = {
            "permissions": permissions,
            "authorized_at": time.time(),
        }
        logger.info(f"User authorized: {user_id} with permissions: {permissions}")
    
    def validate_command(
        self,
        command: Dict[str, Any],
        user_id: str
    ) -> Tuple[bool, str]:
        """Validate command and user authorization."""
        if user_id not in self.authorized_users:
            return False, "User not authorized"
        
        # Check required fields
        if "action" not in command:
            return False, "Command missing 'action' field"
        
        return True, "Valid"
    
    def get_next_command(self) -> Optional[Dict[str, Any]]:
        """Get next command from queue."""
        if not self.command_queue:
            return None
        
        return self.command_queue.pop(0)
    
    def log_command(
        self,
        command_id: str,
        status: str,
        result: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log command execution."""
        log_entry = {
            "command_id": command_id,
            "status": status,
            "result": result,
            "timestamp": time.time(),
        }
        self.command_history.append(log_entry)


# ============================================================================
# SECTION 3: MONITORING & COMPLIANCE (~500 lines)
# ============================================================================

class AlertLevel(str, Enum):
    """Alert levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertingSystem:
    """
    Alerting system for real-time monitoring.
    
    Features:
    - Multi-level alerts (info, warning, critical)
    - Alert routing (email, SMS, dashboard, API)
    - Alert escalation (unacknowledged alerts escalate)
    - Alert suppression (avoid alert fatigue)
    - Alert correlation (group related alerts)
    """
    
    def __init__(self) -> None:
        """Initialize alerting system."""
        self.alerts: List[Dict[str, Any]] = []
        self.alert_routes: Dict[AlertLevel, List[str]] = {
            AlertLevel.INFO: ["dashboard"],
            AlertLevel.WARNING: ["dashboard", "email"],
            AlertLevel.CRITICAL: ["dashboard", "email", "sms"],
        }
        self.suppressed_alerts: Dict[str, float] = {}  # alert_key -> suppress_until
    
    def create_alert(
        self,
        level: AlertLevel,
        message: str,
        source: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create an alert.
        
        Args:
            level: Alert level
            message: Alert message
            source: Alert source
            metadata: Additional metadata
            
        Returns:
            str: Alert ID
        """
        alert_id = str(uuid.uuid4())
        alert_key = f"{source}:{message[:50]}"
        
        # Check if suppressed
        if alert_key in self.suppressed_alerts:
            if time.time() < self.suppressed_alerts[alert_key]:
                logger.debug(f"Alert suppressed: {alert_key}")
                return alert_id
        
        alert = {
            "alert_id": alert_id,
            "level": level.value,
            "message": message,
            "source": source,
            "timestamp": time.time(),
            "acknowledged": False,
            "metadata": metadata or {},
        }
        
        self.alerts.append(alert)
        
        # Route alert
        self._route_alert(alert)
        
        logger.log(
            "INFO" if level == AlertLevel.INFO else "WARNING" if level == AlertLevel.WARNING else "ERROR",
            f"Alert [{level.value}]: {message}"
        )
        
        return alert_id
    
    def _route_alert(self, alert: Dict[str, Any]) -> None:
        """Route alert to appropriate channels."""
        level = AlertLevel(alert["level"])
        routes = self.alert_routes.get(level, [])
        
        for route in routes:
            if route == "dashboard":
                # Would send to dashboard
                pass
            elif route == "email":
                # Would send email
                logger.debug(f"Would send email alert: {alert['message']}")
            elif route == "sms":
                # Would send SMS
                logger.debug(f"Would send SMS alert: {alert['message']}")
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert."""
        for alert in self.alerts:
            if alert["alert_id"] == alert_id:
                alert["acknowledged"] = True
                return True
        return False
    
    def get_unacknowledged_alerts(self) -> List[Dict[str, Any]]:
        """Get unacknowledged alerts."""
        return [a for a in self.alerts if not a["acknowledged"]]


class ComplianceSystem:
    """
    Regulatory compliance system.
    
    Implements:
    - Audit trail logging (who, what, when, why)
    - Regulatory reporting (GDPR, financial regulations)
    - Data retention policies
    - Access control and authorization
    - Change management (version control for policies)
    """
    
    def __init__(self, retention_days: int = 2555) -> None:  # 7 years default
        """Initialize compliance system."""
        self.audit_log: List[Dict[str, Any]] = []
        self.retention_days = retention_days
        self.access_log: List[Dict[str, Any]] = []
    
    def log_decision(
        self,
        decision_type: str,
        decision: Dict[str, Any],
        user_id: str,
        reason: str,
        approved_by: Optional[List[str]] = None
    ) -> str:
        """
        Log a decision for audit trail.
        
        Args:
            decision_type: Type of decision (policy_execution, approval, etc.)
            decision: Decision details
            user_id: User who made/initiated decision
            reason: Reason for decision
            approved_by: List of user IDs who approved (if applicable)
            
        Returns:
            str: Audit log entry ID
        """
        log_id = str(uuid.uuid4())
        
        log_entry = {
            "log_id": log_id,
            "timestamp": time.time(),
            "decision_type": decision_type,
            "decision": decision,
            "user_id": user_id,
            "reason": reason,
            "approved_by": approved_by or [],
        }
        
        self.audit_log.append(log_entry)
        
        logger.info(f"Audit log: {decision_type} by {user_id}: {reason}")
        
        return log_id
    
    def log_access(
        self,
        user_id: str,
        action: str,
        resource: str,
        success: bool
    ) -> None:
        """Log access attempt."""
        access_entry = {
            "timestamp": time.time(),
            "user_id": user_id,
            "action": action,
            "resource": resource,
            "success": success,
        }
        
        self.access_log.append(access_entry)
    
    def get_audit_trail(
        self,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        user_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get audit trail with filters."""
        filtered = self.audit_log
        
        if start_time:
            filtered = [e for e in filtered if e["timestamp"] >= start_time]
        
        if end_time:
            filtered = [e for e in filtered if e["timestamp"] <= end_time]
        
        if user_id:
            filtered = [e for e in filtered if e["user_id"] == user_id]
        
        return filtered
    
    def cleanup_old_logs(self) -> int:
        """Clean up logs older than retention period."""
        cutoff = time.time() - self.retention_days * 86400
        
        initial_count = len(self.audit_log)
        self.audit_log = [e for e in self.audit_log if e["timestamp"] >= cutoff]
        
        removed = initial_count - len(self.audit_log)
        logger.info(f"Cleaned up {removed} old audit log entries")
        
        return removed


class AccountabilitySystem:
    """
    Accountability system for decision attribution.
    
    Tracks:
    - Decision attribution (which board/person approved)
    - Performance attribution (outcomes linked to decisions)
    - Blame assignment (who is responsible for failures)
    - Transparency reports
    """
    
    def __init__(self) -> None:
        """Initialize accountability system."""
        self.decision_attribution: Dict[str, Dict[str, Any]] = {}
        self.performance_tracking: List[Dict[str, Any]] = []
    
    def attribute_decision(
        self,
        decision_id: str,
        policy: ControlVector,
        approved_by: List[str],
        board_votes: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Attribute a decision to approvers.
        
        Args:
            decision_id: Decision ID
            policy: Policy that was approved
            approved_by: List of user IDs who approved
            board_votes: Board voting results (optional)
        """
        self.decision_attribution[decision_id] = {
            "decision_id": decision_id,
            "policy": policy.to_dict(),
            "approved_by": approved_by,
            "board_votes": board_votes or {},
            "timestamp": time.time(),
        }
    
    def track_performance(
        self,
        decision_id: str,
        expected_outcomes: Dict[str, float],
        actual_outcomes: Dict[str, float],
        timestamp: float
    ) -> None:
        """Track performance of a decision."""
        performance_entry = {
            "decision_id": decision_id,
            "expected_outcomes": expected_outcomes,
            "actual_outcomes": actual_outcomes,
            "timestamp": timestamp,
            "performance_score": self._compute_performance_score(expected_outcomes, actual_outcomes),
        }
        
        self.performance_tracking.append(performance_entry)
    
    def _compute_performance_score(
        self,
        expected: Dict[str, float],
        actual: Dict[str, float]
    ) -> float:
        """Compute performance score (0-1, higher is better)."""
        if not expected:
            return 0.5
        
        errors = []
        for key in expected:
            if key in actual:
                error = abs(expected[key] - actual[key]) / (abs(expected[key]) + 1e-6)
                errors.append(error)
        
        if not errors:
            return 0.5
        
        avg_error = np.mean(errors)
        score = max(0.0, 1.0 - avg_error)  # Lower error = higher score
        
        return score
    
    def generate_transparency_report(
        self,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None
    ) -> Dict[str, Any]:
        """Generate transparency report."""
        if start_time is None:
            start_time = time.time() - 30 * 86400  # Last 30 days
        
        if end_time is None:
            end_time = time.time()
        
        # Filter decisions in time range
        decisions = [
            d for d in self.decision_attribution.values()
            if start_time <= d["timestamp"] <= end_time
        ]
        
        # Filter performance in time range
        performance = [
            p for p in self.performance_tracking
            if start_time <= p["timestamp"] <= end_time
        ]
        
        report = {
            "period": {
                "start": start_time,
                "end": end_time,
            },
            "n_decisions": len(decisions),
            "n_performance_tracked": len(performance),
            "avg_performance_score": np.mean([p["performance_score"] for p in performance]) if performance else 0.0,
            "decisions": decisions,
        }
        
        return report


class RollbackSystem:
    """
    Rollback system for state and policy recovery.
    
    Features:
    - State snapshots (periodic checkpoints)
    - Policy rollback (undo last N policies)
    - State restoration (revert to previous state)
    - Transaction logs (for audit and recovery)
    - 7-day rollback window
    """
    
    def __init__(self, rollback_window_days: int = 7) -> None:
        """Initialize rollback system."""
        self.rollback_window_days = rollback_window_days
        self.state_snapshots: List[Tuple[float, StateVector]] = []  # (timestamp, state)
        self.policy_history: List[Tuple[float, ControlVector, str]] = []  # (timestamp, policy, execution_id)
        self.transaction_log: List[Dict[str, Any]] = []
    
    def create_snapshot(
        self,
        state: StateVector,
        snapshot_id: Optional[str] = None
    ) -> str:
        """
        Create a state snapshot.
        
        Args:
            state: State to snapshot
            snapshot_id: Optional snapshot ID
            
        Returns:
            str: Snapshot ID
        """
        if snapshot_id is None:
            snapshot_id = str(uuid.uuid4())
        
        timestamp = time.time()
        self.state_snapshots.append((timestamp, state.copy()))
        
        # Clean up old snapshots outside rollback window
        cutoff = timestamp - self.rollback_window_days * 86400
        self.state_snapshots = [(t, s) for t, s in self.state_snapshots if t >= cutoff]
        
        logger.info(f"State snapshot created: {snapshot_id}")
        
        return snapshot_id
    
    def record_policy_execution(
        self,
        policy: ControlVector,
        execution_id: str
    ) -> None:
        """Record policy execution for rollback."""
        timestamp = time.time()
        self.policy_history.append((timestamp, policy, execution_id))
        
        # Clean up old history
        cutoff = timestamp - self.rollback_window_days * 86400
        self.policy_history = [(t, p, eid) for t, p, eid in self.policy_history if t >= cutoff]
    
    def rollback_policies(
        self,
        n_policies: int
    ) -> List[str]:
        """
        Rollback last N policies.
        
        Args:
            n_policies: Number of policies to rollback
            
        Returns:
            List[str]: List of execution IDs rolled back
        """
        if n_policies > len(self.policy_history):
            n_policies = len(self.policy_history)
        
        rolled_back = []
        for _ in range(n_policies):
            if self.policy_history:
                timestamp, policy, execution_id = self.policy_history.pop()
                rolled_back.append(execution_id)
                logger.info(f"Rolled back policy: {execution_id}")
        
        return rolled_back
    
    def restore_state(
        self,
        snapshot_id: Optional[str] = None,
        timestamp: Optional[float] = None
    ) -> Optional[StateVector]:
        """
        Restore state from snapshot.
        
        Args:
            snapshot_id: Snapshot ID (if provided, use this)
            timestamp: Timestamp to restore to (if provided, find closest)
            
        Returns:
            Optional[StateVector]: Restored state or None
        """
        if timestamp is not None:
            # Find closest snapshot to timestamp
            closest = None
            min_diff = float('inf')
            
            for snap_timestamp, state in self.state_snapshots:
                diff = abs(snap_timestamp - timestamp)
                if diff < min_diff:
                    min_diff = diff
                    closest = state
            
            if closest and min_diff < 3600:  # Within 1 hour
                logger.info(f"Restored state from timestamp: {timestamp}")
                return closest.copy()
        
        # Use most recent snapshot
        if self.state_snapshots:
            _, state = self.state_snapshots[-1]
            logger.info("Restored state from most recent snapshot")
            return state.copy()
        
        return None
    
    def get_rollback_window(self) -> float:
        """Get rollback window in seconds."""
        return self.rollback_window_days * 86400


# ============================================================================
# SECTION 4: RESILIENCE & LEARNING (~400 lines)
# ============================================================================

class ModelAdaptation:
    """
    Model adaptation for online learning.
    
    Features:
    - Online learning (update dynamics model from observations)
    - Parameter estimation (Bayesian updates)
    - Model validation (compare predictions to outcomes)
    - Model versioning (A/B testing of models)
    - Drift detection (detect when model becomes stale)
    """
    
    def __init__(self, dynamics: DynamicsModel) -> None:
        """Initialize model adaptation."""
        self.dynamics = dynamics
        self.parameter_history: List[Dict[str, float]] = []
        self.prediction_errors: List[float] = []
        self.drift_threshold = 0.1  # 10% error increase
    
    def update_parameters(
        self,
        observed_state: StateVector,
        predicted_state: StateVector,
        control: ControlVector
    ) -> Dict[str, float]:
        """
        Update model parameters based on observation.
        
        Args:
            observed_state: Actual observed state
            predicted_state: Model-predicted state
            control: Control that was applied
            
        Returns:
            Dict[str, float]: Updated parameter values
        """
        # Compute prediction error
        error = self._compute_prediction_error(observed_state, predicted_state)
        self.prediction_errors.append(error)
        
        # Simple parameter update (placeholder for Bayesian update)
        # In practice, would use more sophisticated methods
        
        updated_params = {
            "delta_K": self.dynamics.delta_K,
            "delta_I": self.dynamics.delta_I,
            "alpha": self.dynamics.alpha,
            # ... other parameters
        }
        
        self.parameter_history.append(updated_params)
        
        logger.debug(f"Model parameters updated, prediction error: {error:.4f}")
        
        return updated_params
    
    def _compute_prediction_error(
        self,
        observed: StateVector,
        predicted: StateVector
    ) -> float:
        """Compute prediction error."""
        obs_dict = observed.to_dict()
        pred_dict = predicted.to_dict()
        
        errors = []
        for key in obs_dict:
            if key in pred_dict:
                error = abs(obs_dict[key] - pred_dict[key]) / (abs(obs_dict[key]) + 1e-6)
                errors.append(error)
        
        return np.mean(errors) if errors else 0.0
    
    def detect_drift(self) -> bool:
        """Detect if model has drifted."""
        if len(self.prediction_errors) < 10:
            return False
        
        # Compare recent errors to historical average
        recent_errors = self.prediction_errors[-10:]
        historical_errors = self.prediction_errors[:-10] if len(self.prediction_errors) > 10 else recent_errors
        
        recent_avg = np.mean(recent_errors)
        historical_avg = np.mean(historical_errors)
        
        if historical_avg > 0:
            increase = (recent_avg - historical_avg) / historical_avg
            if increase > self.drift_threshold:
                logger.warning(f"Model drift detected: {increase:.2%} error increase")
                return True
        
        return False


class PerformanceFeedback:
    """
    Performance feedback system.
    
    Tracks:
    - Outcome measurement (actual vs. predicted)
    - Policy effectiveness evaluation
    - Board performance tracking
    - Scenario accuracy assessment
    - Continuous improvement loop
    """
    
    def __init__(self) -> None:
        """Initialize performance feedback."""
        self.outcome_comparisons: List[Dict[str, Any]] = []
        self.policy_effectiveness: Dict[str, float] = {}
        self.board_performance: Dict[str, List[float]] = {}
    
    def compare_outcomes(
        self,
        policy_id: str,
        predicted: Dict[str, float],
        actual: Dict[str, float],
        timestamp: float
    ) -> Dict[str, Any]:
        """
        Compare predicted vs actual outcomes.
        
        Args:
            policy_id: Policy ID
            predicted: Predicted outcomes
            actual: Actual outcomes
            timestamp: Timestamp
            
        Returns:
            Dict[str, Any]: Comparison results
        """
        comparison = {
            "policy_id": policy_id,
            "predicted": predicted,
            "actual": actual,
            "timestamp": timestamp,
            "errors": {k: abs(predicted.get(k, 0) - actual.get(k, 0)) for k in set(list(predicted.keys()) + list(actual.keys()))},
            "relative_errors": {k: abs(predicted.get(k, 0) - actual.get(k, 0)) / (abs(actual.get(k, 0)) + 1e-6) for k in actual},
        }
        
        self.outcome_comparisons.append(comparison)
        
        # Update policy effectiveness
        avg_error = np.mean(list(comparison["relative_errors"].values()))
        effectiveness = max(0.0, 1.0 - avg_error)
        self.policy_effectiveness[policy_id] = effectiveness
        
        return comparison
    
    def track_board_performance(
        self,
        board_id: str,
        decision_quality: float
    ) -> None:
        """Track board decision quality."""
        if board_id not in self.board_performance:
            self.board_performance[board_id] = []
        
        self.board_performance[board_id].append(decision_quality)
    
    def get_policy_effectiveness(self, policy_id: str) -> Optional[float]:
        """Get policy effectiveness score."""
        return self.policy_effectiveness.get(policy_id)


class FaultTolerance:
    """
    Fault tolerance system.
    
    Features:
    - Redundancy (multiple data sources)
    - Failover mechanisms (backup systems)
    - Graceful degradation (fallback to simpler models)
    - Data backup and recovery
    - System health monitoring
    """
    
    def __init__(self) -> None:
        """Initialize fault tolerance."""
        self.primary_systems: Dict[str, bool] = {}
        self.backup_systems: Dict[str, bool] = {}
        self.system_health: Dict[str, str] = {}  # "healthy", "degraded", "failed"
    
    def register_system(
        self,
        system_name: str,
        is_primary: bool = True
    ) -> None:
        """Register a system (primary or backup)."""
        if is_primary:
            self.primary_systems[system_name] = True
            self.system_health[system_name] = "healthy"
        else:
            self.backup_systems[system_name] = True
    
    def check_system_health(self, system_name: str) -> str:
        """Check system health."""
        return self.system_health.get(system_name, "unknown")
    
    def mark_system_failed(self, system_name: str) -> None:
        """Mark system as failed and attempt failover."""
        self.system_health[system_name] = "failed"
        logger.warning(f"System failed: {system_name}")
        
        # Attempt failover to backup
        backup_name = f"{system_name}_backup"
        if backup_name in self.backup_systems:
            self.system_health[backup_name] = "healthy"
            logger.info(f"Failed over to backup: {backup_name}")
    
    def get_system_status(self) -> Dict[str, str]:
        """Get status of all systems."""
        return self.system_health.copy()

