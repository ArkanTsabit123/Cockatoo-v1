# src/utilities/monitor.py

"""System and performance monitoring with metrics collection and alerting."""

import os
import psutil
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
from dataclasses import dataclass, field
from collections import deque
import json


class MetricType(Enum):
    """Types of metrics that can be collected."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


@dataclass
class Metric:
    """Represents a single metric data point."""
    name: str
    value: float
    type: MetricType
    timestamp: datetime = field(default_factory=datetime.now)
    tags: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "value": self.value,
            "type": self.type.value,
            "timestamp": self.timestamp.isoformat(),
            "tags": self.tags
        }


@dataclass
class MonitorConfig:
    """Configuration for monitor instances."""
    collection_interval: float = 5.0
    retention_period: int = 3600
    alert_thresholds: Dict[str, float] = field(default_factory=dict)
    enable_system_monitoring: bool = True
    enable_performance_monitoring: bool = True
    max_history_points: int = 1000


class Monitor:
    """Base class for metrics monitoring."""
    
    def __init__(self, name: str, config: Optional[MonitorConfig] = None):
        self.name = name
        self.config = config or MonitorConfig()
        self.metrics: Dict[str, deque] = {}
        self.handlers: List[Callable] = []
        self.running = False
        self._lock = threading.Lock()
        self._collection_thread: Optional[threading.Thread] = None
    
    def start(self):
        """Start the monitoring collection thread."""
        if self.running:
            return
        
        self.running = True
        self._collection_thread = threading.Thread(
            target=self._collection_loop,
            daemon=True
        )
        self._collection_thread.start()
    
    def stop(self):
        """Stop the monitoring collection thread."""
        self.running = False
        if self._collection_thread:
            self._collection_thread.join(timeout=5)
    
    def _collection_loop(self):
        while self.running:
            try:
                self.collect_metrics()
            except Exception as e:
                print(f"Error collecting metrics: {e}")
            time.sleep(self.config.collection_interval)
    
    def collect_metrics(self):
        """Collect metrics - to be overridden by subclasses."""
        pass
    
    def record_metric(self, name: str, value: float, 
                     metric_type: MetricType = MetricType.GAUGE,
                     tags: Optional[Dict[str, str]] = None):
        """Record a metric value."""
        with self._lock:
            if name not in self.metrics:
                self.metrics[name] = deque(maxlen=self.config.max_history_points)
            
            metric = Metric(
                name=name,
                value=value,
                type=metric_type,
                tags=tags or {}
            )
            
            self.metrics[name].append(metric)
            self._check_thresholds(metric)
            self._notify_handlers(metric)
    
    def _check_thresholds(self, metric: Metric):
        threshold = self.config.alert_thresholds.get(metric.name)
        if threshold and metric.value > threshold:
            self._trigger_alert(metric, threshold)
    
    def _trigger_alert(self, metric: Metric, threshold: float):
        alert = {
            "metric": metric.name,
            "value": metric.value,
            "threshold": threshold,
            "timestamp": metric.timestamp.isoformat()
        }
        print(f"ALERT: {json.dumps(alert)}")
    
    def add_handler(self, handler: Callable[[Metric], None]):
        """Add a handler function to be called for each metric."""
        self.handlers.append(handler)
    
    def _notify_handlers(self, metric: Metric):
        for handler in self.handlers:
            try:
                handler(metric)
            except Exception as e:
                print(f"Error in metric handler: {e}")
    
    def get_metric_history(self, name: str, 
                          since: Optional[datetime] = None) -> List[Metric]:
        """Get history of a specific metric."""
        if name not in self.metrics:
            return []
        
        metrics = list(self.metrics[name])
        
        if since:
            metrics = [m for m in metrics if m.timestamp >= since]
        
        return metrics
    
    def get_latest_metric(self, name: str) -> Optional[Metric]:
        """Get the latest value of a specific metric."""
        if name not in self.metrics or not self.metrics[name]:
            return None
        return self.metrics[name][-1]
    
    def get_statistics(self, name: str, 
                      period: Optional[timedelta] = None) -> Dict[str, float]:
        """Get statistics for a metric over a specified period."""
        metrics = self.get_metric_history(name)
        
        if period:
            since = datetime.now() - period
            metrics = [m for m in metrics if m.timestamp >= since]
        
        if not metrics:
            return {}
        
        values = [m.value for m in metrics]
        
        return {
            "min": min(values),
            "max": max(values),
            "avg": sum(values) / len(values),
            "count": len(values),
            "latest": values[-1]
        }


class SystemMonitor(Monitor):
    """Monitor for system-level metrics (CPU, memory, disk, network)."""
    
    def __init__(self, config: Optional[MonitorConfig] = None):
        super().__init__("system", config)
    
    def collect_metrics(self):
        cpu_percent = psutil.cpu_percent(interval=1)
        self.record_metric("cpu.usage", cpu_percent, tags={"type": "percent"})
        
        memory = psutil.virtual_memory()
        self.record_metric("memory.usage", memory.percent, tags={"type": "percent"})
        self.record_metric("memory.available", memory.available / (1024**3), 
                          tags={"unit": "GB"})
        
        disk = psutil.disk_usage('/')
        self.record_metric("disk.usage", disk.percent, tags={"type": "percent"})
        self.record_metric("disk.free", disk.free / (1024**3), tags={"unit": "GB"})
        
        net = psutil.net_io_counters()
        self.record_metric("network.bytes_sent", net.bytes_sent, 
                          metric_type=MetricType.COUNTER, tags={"unit": "bytes"})
        self.record_metric("network.bytes_recv", net.bytes_recv,
                          metric_type=MetricType.COUNTER, tags={"unit": "bytes"})
        
        process = psutil.Process()
        with process.oneshot():
            self.record_metric("process.cpu", process.cpu_percent())
            self.record_metric("process.memory", process.memory_percent())
            self.record_metric("process.threads", process.num_threads(),
                              metric_type=MetricType.GAUGE)


class PerformanceMonitor(Monitor):
    """Monitor for application performance metrics."""
    
    def __init__(self, config: Optional[MonitorConfig] = None):
        super().__init__("performance", config)
        self.timers: Dict[str, float] = {}
    
    def start_timer(self, name: str):
        """Start a timer for measuring duration."""
        self.timers[name] = time.time()
    
    def stop_timer(self, name: str, tags: Optional[Dict[str, str]] = None):
        """Stop a timer and record the duration."""
        if name not in self.timers:
            return
        
        duration = time.time() - self.timers[name]
        self.record_metric(f"timer.{name}", duration * 1000,
                          metric_type=MetricType.TIMER, tags=tags)
        del self.timers[name]
    
    def increment_counter(self, name: str, value: float = 1,
                         tags: Optional[Dict[str, str]] = None):
        """Increment a counter metric."""
        current = self.get_latest_metric(name)
        new_value = (current.value if current else 0) + value
        self.record_metric(name, new_value, 
                          metric_type=MetricType.COUNTER, tags=tags)


_monitors: Dict[str, Monitor] = {}


def get_monitor(monitor_type: str = "system", 
               config: Optional[MonitorConfig] = None) -> Monitor:
    """Get or create a monitor instance."""
    if monitor_type not in _monitors:
        if monitor_type == "system":
            _monitors[monitor_type] = SystemMonitor(config)
        elif monitor_type == "performance":
            _monitors[monitor_type] = PerformanceMonitor(config)
        else:
            _monitors[monitor_type] = Monitor(monitor_type, config)
    return _monitors[monitor_type]