"""Sensor interfaces and implementations for policy engine.

Provides standardized sensor interfaces for reading system metrics,
cloud resources, and custom metrics.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Callable
from loguru import logger

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logger.warning("psutil not available - SystemMetricsSensor will be disabled")


class BaseSensor(ABC):
    """Abstract base class for sensors.
    
    All sensors must implement read() to return metric snapshots
    matching the doctrine's metric specifications.
    """
    
    @abstractmethod
    def read(self) -> Dict[str, float]:
        """
        Read current metric snapshot.
        
        Returns:
            Dict[str, float]: Metric name -> value mapping
        """
        pass
    
    def validate(self, snapshot: Dict[str, float], expected_metrics: List[str]) -> bool:
        """
        Validate snapshot matches expected metrics.
        
        Args:
            snapshot: Metric snapshot to validate
            expected_metrics: List of expected metric names
            
        Returns:
            bool: True if snapshot is valid
        """
        missing = set(expected_metrics) - set(snapshot.keys())
        if missing:
            logger.warning(f"Missing metrics in snapshot: {missing}")
            return False
        return True
    
    def health_check(self) -> bool:
        """
        Check sensor availability.
        
        Returns:
            bool: True if sensor is healthy
        """
        try:
            self.read()
            return True
        except Exception as e:
            logger.error(f"Sensor health check failed: {e}")
            return False


class SystemMetricsSensor(BaseSensor):
    """OS-level system metrics sensor using psutil.
    
    Extracts:
    - CPU metrics: cpu_percent, cpu_count, cpu_freq
    - Memory metrics: memory_percent, memory_available, memory_used
    - Disk metrics: disk_usage_percent, disk_io_read, disk_io_write
    - Network metrics: network_bytes_sent, network_bytes_recv
    - Process metrics: process_count
    """
    
    def __init__(self, metric_mapping: Optional[Dict[str, str]] = None):
        """
        Initialize system metrics sensor.
        
        Args:
            metric_mapping: Optional mapping from doctrine metric names to psutil attributes
        """
        if not PSUTIL_AVAILABLE:
            raise ImportError("psutil is required for SystemMetricsSensor")
        
        self.metric_mapping = metric_mapping or {}
        self._last_net_io = None
        self._last_disk_io = None
    
    def read(self) -> Dict[str, float]:
        """Read system metrics."""
        snapshot = {}
        
        # CPU metrics
        snapshot['cpu_percent'] = psutil.cpu_percent(interval=0.1)
        snapshot['cpu_count'] = float(psutil.cpu_count())
        try:
            cpu_freq = psutil.cpu_freq()
            snapshot['cpu_freq_mhz'] = cpu_freq.current if cpu_freq else 0.0
        except:
            snapshot['cpu_freq_mhz'] = 0.0
        
        # Memory metrics
        mem = psutil.virtual_memory()
        snapshot['memory_percent'] = mem.percent
        snapshot['memory_available_gb'] = mem.available / (1024**3)
        snapshot['memory_used_gb'] = mem.used / (1024**3)
        
        # Disk metrics
        try:
            disk = psutil.disk_usage('/')
            snapshot['disk_usage_percent'] = disk.percent
        except:
            snapshot['disk_usage_percent'] = 0.0
        
        # Disk I/O (delta from last read)
        try:
            disk_io = psutil.disk_io_counters()
            if disk_io and self._last_disk_io:
                snapshot['disk_io_read_mb'] = (disk_io.read_bytes - self._last_disk_io.read_bytes) / (1024**2)
                snapshot['disk_io_write_mb'] = (disk_io.write_bytes - self._last_disk_io.write_bytes) / (1024**2)
            else:
                snapshot['disk_io_read_mb'] = 0.0
                snapshot['disk_io_write_mb'] = 0.0
            self._last_disk_io = disk_io
        except:
            snapshot['disk_io_read_mb'] = 0.0
            snapshot['disk_io_write_mb'] = 0.0
        
        # Network I/O (delta from last read)
        try:
            net_io = psutil.net_io_counters()
            if net_io and self._last_net_io:
                snapshot['network_bytes_sent_mb'] = (net_io.bytes_sent - self._last_net_io.bytes_sent) / (1024**2)
                snapshot['network_bytes_recv_mb'] = (net_io.bytes_recv - self._last_net_io.bytes_recv) / (1024**2)
            else:
                snapshot['network_bytes_sent_mb'] = 0.0
                snapshot['network_bytes_recv_mb'] = 0.0
            self._last_net_io = net_io
        except:
            snapshot['network_bytes_sent_mb'] = 0.0
            snapshot['network_bytes_recv_mb'] = 0.0
        
        # Process count
        try:
            snapshot['process_count'] = float(len(psutil.pids()))
        except:
            snapshot['process_count'] = 0.0
        
        # Apply metric mapping if provided
        if self.metric_mapping:
            mapped_snapshot = {}
            for doctrine_metric, system_metric in self.metric_mapping.items():
                if system_metric in snapshot:
                    mapped_snapshot[doctrine_metric] = snapshot[system_metric]
                else:
                    logger.warning(f"Metric mapping failed: {system_metric} not in snapshot")
            return mapped_snapshot
        
        return snapshot


class CloudResourceSensor(BaseSensor):
    """Abstract base for cloud provider metrics.
    
    Subclasses should implement provider-specific metric extraction.
    """
    
    def __init__(self, provider: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize cloud resource sensor.
        
        Args:
            provider: Cloud provider name ("aws", "gcp", "azure")
            config: Provider-specific configuration
        """
        self.provider = provider
        self.config = config or {}
    
    @abstractmethod
    def read(self) -> Dict[str, float]:
        """Read cloud metrics (implemented by subclasses)."""
        pass


class CustomMetricSensor(BaseSensor):
    """User-defined metric sensor with custom extraction function.
    
    Accepts a custom function that returns metric snapshots.
    """
    
    def __init__(self, extractor: Callable[[], Dict[str, float]]):
        """
        Initialize custom metric sensor.
        
        Args:
            extractor: Function that returns Dict[str, float] metric snapshot
        """
        self.extractor = extractor
    
    def read(self) -> Dict[str, float]:
        """Read metrics using custom extractor."""
        try:
            return self.extractor()
        except Exception as e:
            logger.error(f"Custom sensor extractor failed: {e}")
            return {}


class SensorRegistry:
    """Registry mapping metric extractor keys to sensor instances.
    
    Provides auto-discovery and fallback chain for missing metrics.
    """
    
    def __init__(self):
        """Initialize sensor registry."""
        self.sensors: Dict[str, BaseSensor] = {}
        self.metric_to_sensor: Dict[str, str] = {}  # metric_name -> sensor_key
    
    def register(self, sensor_key: str, sensor: BaseSensor, metrics: Optional[List[str]] = None) -> None:
        """
        Register a sensor.
        
        Args:
            sensor_key: Unique identifier for sensor
            sensor: Sensor instance
            metrics: Optional list of metric names this sensor provides
        """
        self.sensors[sensor_key] = sensor
        if metrics:
            for metric in metrics:
                self.metric_to_sensor[metric] = sensor_key
        logger.info(f"Registered sensor: {sensor_key}")
    
    def get_sensor(self, sensor_key: str) -> Optional[BaseSensor]:
        """Get sensor by key."""
        return self.sensors.get(sensor_key)
    
    def read_all(self, required_metrics: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Read metrics from all registered sensors.
        
        Args:
            required_metrics: Optional list of required metric names
            
        Returns:
            Dict[str, float]: Combined metric snapshot
        """
        snapshot = {}
        
        # Read from all sensors
        for sensor_key, sensor in self.sensors.items():
            try:
                sensor_snapshot = sensor.read()
                snapshot.update(sensor_snapshot)
            except Exception as e:
                logger.error(f"Sensor {sensor_key} read failed: {e}")
        
        # Validate required metrics
        if required_metrics:
            missing = set(required_metrics) - set(snapshot.keys())
            if missing:
                logger.warning(f"Missing required metrics: {missing}")
        
        return snapshot
    
    def auto_discover(self) -> None:
        """Auto-discover and register available sensors."""
        # Try to register SystemMetricsSensor if psutil available
        if PSUTIL_AVAILABLE:
            try:
                system_sensor = SystemMetricsSensor()
                if system_sensor.health_check():
                    self.register("system", system_sensor)
            except Exception as e:
                logger.warning(f"Failed to auto-discover system sensor: {e}")
    
    def health_check_all(self) -> Dict[str, bool]:
        """Check health of all sensors."""
        return {key: sensor.health_check() for key, sensor in self.sensors.items()}

