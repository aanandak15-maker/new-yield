"""
Advanced Health Monitoring System for India Agricultural Intelligence Platform
Comprehensive system health monitoring with predictive analytics and alerting
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import psutil
import socket
import json
from pathlib import Path

from india_agri_platform.core.error_handling import error_handler, get_error_stats, get_system_health
from india_agri_platform.core.cache_manager import get_cache_stats
from india_agri_platform.core.task_scheduler import get_scheduler_status
from india_agri_platform.database.manager import db_manager

logger = logging.getLogger(__name__)

class HealthStatus(Enum):
    """System health status levels"""
    HEALTHY = "HEALTHY"
    WARNING = "WARNING"
    DEGRADED = "DEGRADED"
    CRITICAL = "CRITICAL"
    DOWN = "DOWN"

class ComponentStatus(Enum):
    """Component health status"""
    HEALTHY = "HEALTHY"
    WARNING = "WARNING"
    ERROR = "ERROR"
    DOWN = "DOWN"

@dataclass
class HealthMetric:
    """Individual health metric"""
    name: str
    value: Any
    unit: str
    status: ComponentStatus
    threshold_warning: Any = None
    threshold_critical: Any = None
    last_updated: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'value': self.value,
            'unit': self.unit,
            'status': self.status.value,
            'threshold_warning': self.threshold_warning,
            'threshold_critical': self.threshold_critical,
            'last_updated': self.last_updated.isoformat()
        }

@dataclass
class ComponentHealth:
    """Component health status"""
    name: str
    component_type: str
    status: ComponentStatus
    metrics: Dict[str, HealthMetric] = field(default_factory=dict)
    last_check: datetime = field(default_factory=datetime.utcnow)
    error_count: int = 0
    response_time: Optional[float] = None
    dependencies: List[str] = field(default_factory=list)
    alert_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'component_type': self.component_type,
            'status': self.status.value,
            'metrics': {k: v.to_dict() for k, v in self.metrics.items()},
            'last_check': self.last_check.isoformat(),
            'error_count': self.error_count,
            'response_time': self.response_time,
            'dependencies': self.dependencies,
            'alert_message': self.alert_message
        }

class AlertRule:
    """Alert rule definition"""

    def __init__(self, name: str, metric_name: str, condition: str,
                 threshold: Any, severity: str, cooldown_minutes: int = 5):
        self.name = name
        self.metric_name = metric_name
        self.condition = condition  # '>', '<', '>=', '<=', '==', '!='
        self.threshold = threshold
        self.severity = severity  # 'info', 'warning', 'error', 'critical'
        self.cooldown_minutes = cooldown_minutes
        self.last_triggered = None

    def should_trigger(self, metric: HealthMetric) -> bool:
        """Check if alert should trigger"""
        if self.last_triggered and \
           (datetime.utcnow() - self.last_triggered).total_seconds() < (self.cooldown_minutes * 60):
            return False

        try:
            if self.condition == '>':
                return metric.value > self.threshold
            elif self.condition == '<':
                return metric.value < self.threshold
            elif self.condition == '>=':
                return metric.value >= self.threshold
            elif self.condition == '<=':
                return metric.value <= self.threshold
            elif self.condition == '==':
                return metric.value == self.threshold
            elif self.condition == '!=':
                return metric.value != self.threshold
        except (TypeError, ValueError):
            return False

        return False

class AdvancedHealthMonitor:
    """Comprehensive health monitoring system"""

    def __init__(self):
        self.components: Dict[str, ComponentHealth] = {}
        self.alert_rules: List[AlertRule] = []
        self.monitoring_enabled = True
        self.check_interval_seconds = 60  # Check every minute
        self.history_length = 100  # Keep last 100 health reports

        # Health history
        self.health_history: List[Dict[str, Any]] = []
        self._monitoring_task = None
        self._initialized = False

        # Initialize default alert rules
        self._setup_default_alert_rules()

        logger.info("Advanced Health Monitor initialized")

    def _setup_default_alert_rules(self):
        """Setup default alert rules"""
        default_rules = [
            AlertRule("high_cpu_usage", "cpu_usage_percent", ">", 85, "warning", cooldown_minutes=5),
            AlertRule("critical_cpu_usage", "cpu_usage_percent", ">", 95, "critical", cooldown_minutes=2),

            AlertRule("high_memory_usage", "memory_usage_percent", ">", 90, "warning", cooldown_minutes=5),
            AlertRule("critical_memory_usage", "memory_usage_percent", ">", 95, "critical", cooldown_minutes=2),

            AlertRule("high_error_rate", "error_rate_per_minute", ">", 10, "warning", cooldown_minutes=10),
            AlertRule("critical_error_rate", "error_rate_per_minute", ">", 50, "critical", cooldown_minutes=5),

            AlertRule("cache_low_hit_rate", "cache_hit_rate_pct", "<", 50, "warning", cooldown_minutes=15),

            AlertRule("database_connection_issues", "db_connection_status", "==", "DOWN", "critical", cooldown_minutes=1),

            AlertRule("scheduler_high_concurrency", "active_executions", ">", 15, "warning", cooldown_minutes=10),
        ]

        self.alert_rules.extend(default_rules)

    async def initialize_monitoring(self) -> bool:
        """Initialize health monitoring"""
        if self._initialized:
            return True

        try:
            # Initialize component monitoring
            await self._initialize_components()

            # Start monitoring task
            self._monitoring_task = asyncio.create_task(self._monitoring_loop())

            self._initialized = True
            logger.info("Health monitoring initialized successfully")
            return True

        except Exception as e:
            error_handler.handle_error(e, {"component": "health_monitor", "operation": "initialization"})
            return False

    async def _initialize_components(self):
        """Initialize all component monitors"""
        # Register core components for monitoring
        components = [
            {
                "name": "system",
                "type": "infrastructure",
                "check_func": self._check_system_health,
                "dependencies": []
            },
            {
                "name": "cache",
                "type": "storage",
                "check_func": self._check_cache_health,
                "dependencies": ["redis"]
            },
            {
                "name": "database",
                "type": "storage",
                "check_func": self._check_database_health,
                "dependencies": ["postgresql"]
            },
            {
                "name": "scheduler",
                "type": "processing",
                "check_func": self._check_scheduler_health,
                "dependencies": []
            },
            {
                "name": "error_handler",
                "type": "monitoring",
                "check_func": self._check_error_handler_health,
                "dependencies": []
            },
            {
                "name": "gearth_engine",
                "type": "external",
                "check_func": self._check_gee_health,
                "dependencies": ["google_earth_engine"]
            },
            {
                "name": "iot_sensors",
                "type": "external",
                "check_func": self._check_iot_health,
                "dependencies": ["sensor_network"]
            }
        ]

        for comp_data in components:
            component = ComponentHealth(
                name=comp_data["name"],
                component_type=comp_data["type"],
                dependencies=comp_data["dependencies"],
                status=ComponentStatus.HEALTHY
            )
            self.components[comp_data["name"]] = component

            # Perform initial health check
            await self._check_component_health(comp_data["name"], comp_data["check_func"])

        logger.info(f"Initialized monitoring for {len(components)} components")

    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring_enabled:
            try:
                await self._perform_health_checks()

                # Process alerts
                await self._process_alerts()

                # Store health report
                await self._store_health_report()

            except Exception as e:
                logger.error(f"Health monitoring loop error: {e}")

            await asyncio.sleep(self.check_interval_seconds)

    async def _perform_health_checks(self):
        """Perform all component health checks"""
        checks = [
            ("system", self._check_system_health),
            ("cache", self._check_cache_health),
            ("database", self._check_database_health),
            ("scheduler", self._check_scheduler_health),
            ("error_handler", self._check_error_handler_health),
            ("gearth_engine", self._check_gee_health),
            ("iot_sensors", self._check_iot_health)
        ]

        for component_name, check_func in checks:
            await self._check_component_health(component_name, check_func)

    async def _check_component_health(self, component_name: str, check_func: Callable):
        """Check health of a specific component"""
        component = self.components.get(component_name)
        if not component:
            return

        start_time = datetime.utcnow()

        try:
            # Perform health check
            health_result = await check_func()

            # Update component status
            component.status = health_result["status"]
            component.last_check = datetime.utcnow()
            component.response_time = (datetime.utcnow() - start_time).total_seconds()

            # Update metrics
            for metric_name, metric_data in health_result.get("metrics", {}).items():
                metric = HealthMetric(
                    name=metric_name,
                    value=metric_data["value"],
                    unit=metric_data["unit"],
                    status=metric_data["status"],
                    threshold_warning=metric_data.get("threshold_warning"),
                    threshold_critical=metric_data.get("threshold_critical")
                )
                component.metrics[metric_name] = metric

            # Clear error count on successful checks
            if component.status in [ComponentStatus.HEALTHY, ComponentStatus.WARNING]:
                component.error_count = 0
                component.alert_message = None

        except Exception as e:
            component.status = ComponentStatus.ERROR
            component.error_count += 1
            component.response_time = (datetime.utcnow() - start_time).total_seconds()
            component.alert_message = str(e)

            # Log the error
            error_handler.handle_error(e, {
                "component": component_name,
                "operation": "health_check"
            })

    # Component Health Check Functions
    async def _check_system_health(self) -> Dict[str, Any]:
        """Check system resource health"""
        metrics = {}

        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_status = ComponentStatus.HEALTHY
            if cpu_percent > 85:
                cpu_status = ComponentStatus.WARNING
            elif cpu_percent > 95:
                cpu_status = ComponentStatus.ERROR

            metrics["cpu_usage_percent"] = {
                "value": round(cpu_percent, 2),
                "unit": "percent",
                "status": cpu_status,
                "threshold_warning": 85,
                "threshold_critical": 95
            }

            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_status = ComponentStatus.HEALTHY
            if memory_percent > 90:
                memory_status = ComponentStatus.WARNING
            elif memory_percent > 95:
                memory_status = ComponentStatus.ERROR

            metrics["memory_usage_percent"] = {
                "value": round(memory_percent, 2),
                "unit": "percent",
                "status": memory_status,
                "threshold_warning": 90,
                "threshold_critical": 95
            }

            metrics["memory_used_gb"] = {
                "value": round(memory.used / (1024**3), 2),
                "unit": "GB",
                "status": ComponentStatus.HEALTHY
            }

            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent
            disk_status = ComponentStatus.HEALTHY
            if disk_percent > 85:
                disk_status = ComponentStatus.WARNING
            elif disk_percent > 95:
                disk_status = ComponentStatus.ERROR

            metrics["disk_usage_percent"] = {
                "value": round(disk_percent, 2),
                "unit": "percent",
                "status": disk_status,
                "threshold_warning": 85,
                "threshold_critical": 95
            }

            # Network connections
            network_connections = len(psutil.net_connections())
            metrics["network_connections"] = {
                "value": network_connections,
                "unit": "connections",
                "status": ComponentStatus.HEALTHY
            }

            overall_status = ComponentStatus.HEALTHY
            if any(m.get("status") == ComponentStatus.ERROR for m in metrics.values()):
                overall_status = ComponentStatus.ERROR
            elif any(m.get("status") == ComponentStatus.WARNING for m in metrics.values()):
                overall_status = ComponentStatus.WARNING

            return {
                "status": overall_status,
                "metrics": metrics
            }

        except Exception as e:
            logger.error(f"System health check failed: {e}")
            return {
                "status": ComponentStatus.ERROR,
                "metrics": {},
                "error": str(e)
            }

    async def _check_cache_health(self) -> Dict[str, Any]:
        """Check cache system health"""
        try:
            cache_stats = await get_cache_stats()
            metrics = {}

            # Cache hit rate
            overall_hit_rate = cache_stats.get('performance', {}).get('overall_hit_rate', 0) * 100
            hit_rate_status = ComponentStatus.HEALTHY
            if overall_hit_rate < 70:
                hit_rate_status = ComponentStatus.WARNING
            elif overall_hit_rate < 50:
                hit_rate_status = ComponentStatus.ERROR

            metrics["cache_hit_rate_pct"] = {
                "value": round(overall_hit_rate, 2),
                "unit": "percent",
                "status": hit_rate_status,
                "threshold_warning": 70,
                "threshold_critical": 50
            }

            # L1 cache size
            l1_entries = cache_stats.get('l1_cache', {}).get('entries', 0)
            metrics["l1_cache_entries"] = {
                "value": l1_entries,
                "unit": "entries",
                "status": ComponentStatus.HEALTHY
            }

            # Redis connection status
            redis_connected = cache_stats.get('l2_cache', {}).get('connected', False)
            redis_status = ComponentStatus.HEALTHY if redis_connected else ComponentStatus.WARNING

            metrics["redis_connection_status"] = {
                "value": "UP" if redis_connected else "DOWN",
                "unit": "status",
                "status": redis_status
            }

            # Determine overall status
            overall_status = ComponentStatus.HEALTHY
            if any(m.get("status") == ComponentStatus.ERROR for m in metrics.values()):
                overall_status = ComponentStatus.ERROR
            elif any(m.get("status") == ComponentStatus.WARNING for m in metrics.values()):
                overall_status = ComponentStatus.WARNING

            return {
                "status": overall_status,
                "metrics": metrics
            }

        except Exception as e:
            logger.error(f"Cache health check failed: {e}")
            return {
                "status": ComponentStatus.WARNING,
                "metrics": {},
                "error": str(e)
            }

    async def _check_database_health(self) -> Dict[str, Any]:
        """Check database health"""
        try:
            # Simple connection test
            connection_status = "UP"
            response_time = None

            start_time = datetime.utcnow()
            db_available = await db_manager.health_check()
            response_time = (datetime.utcnow() - start_time).total_seconds()

            if not db_available:
                connection_status = "DOWN"

            metrics = {}

            status = ComponentStatus.HEALTHY if connection_status == "UP" else ComponentStatus.ERROR

            metrics["db_connection_status"] = {
                "value": connection_status,
                "unit": "status",
                "status": status
            }

            if response_time:
                metrics["db_response_time_ms"] = {
                    "value": round(response_time * 1000, 2),
                    "unit": "ms",
                    "status": ComponentStatus.HEALTHY
                }

            return {
                "status": status,
                "metrics": metrics
            }

        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return {
                "status": ComponentStatus.ERROR,
                "metrics": {
                    "db_connection_status": {
                        "value": "ERROR",
                        "unit": "status",
                        "status": ComponentStatus.ERROR
                    }
                },
                "error": str(e)
            }

    async def _check_scheduler_health(self) -> Dict[str, Any]:
        """Check task scheduler health"""
        try:
            scheduler_stats = get_scheduler_status()
            metrics = {}

            # Active executions
            active_executions = scheduler_stats.get('active_executions', 0)
            executions_status = ComponentStatus.HEALTHY
            if active_executions > 15:
                executions_status = ComponentStatus.WARNING
            elif active_executions > 25:
                executions_status = ComponentStatus.ERROR

            metrics["active_executions"] = {
                "value": active_executions,
                "unit": "tasks",
                "status": executions_status,
                "threshold_warning": 15,
                "threshold_critical": 25
            }

            # Total tasks
            total_tasks = scheduler_stats.get('total_tasks', 0)
            metrics["total_scheduled_tasks"] = {
                "value": total_tasks,
                "unit": "tasks",
                "status": ComponentStatus.HEALTHY
            }

            # Scheduler running status
            scheduler_running = scheduler_stats.get('scheduler_running', False)
            scheduler_status = ComponentStatus.HEALTHY if scheduler_running else ComponentStatus.ERROR

            metrics["scheduler_status"] = {
                "value": "RUNNING" if scheduler_running else "STOPPED",
                "unit": "status",
                "status": scheduler_status
            }

            overall_status = ComponentStatus.HEALTHY
            if any(m.get("status") == ComponentStatus.ERROR for m in metrics.values()):
                overall_status = ComponentStatus.ERROR
            elif any(m.get("status") == ComponentStatus.WARNING for m in metrics.values()):
                overall_status = ComponentStatus.WARNING

            return {
                "status": overall_status,
                "metrics": metrics
            }

        except Exception as e:
            logger.error(f"Scheduler health check failed: {e}")
            return {
                "status": ComponentStatus.ERROR,
                "metrics": {},
                "error": str(e)
            }

    async def _check_error_handler_health(self) -> Dict[str, Any]:
        """Check error handling system health"""
        try:
            error_stats = get_error_stats()
            metrics = {}

            # Error rate calculation
            total_errors = error_stats.get('total_errors', 0)
            recent_errors = error_stats.get('recent_errors', [])

            # Calculate error rate per minute
            error_rate = 0
            if recent_errors:
                # Look at errors in last 1 minute
                one_minute_ago = datetime.utcnow() - timedelta(minutes=1)
                recent_minute_errors = [
                    err for err in recent_errors
                    if datetime.fromisoformat(err['timestamp']) > one_minute_ago
                ]
                error_rate = len(recent_minute_errors)

            error_rate_status = ComponentStatus.HEALTHY
            if error_rate > 10:
                error_rate_status = ComponentStatus.WARNING
            elif error_rate > 50:
                error_rate_status = ComponentStatus.ERROR

            metrics["error_rate_per_minute"] = {
                "value": error_rate,
                "unit": "errors/min",
                "status": error_rate_status,
                "threshold_warning": 10,
                "threshold_critical": 50
            }

            metrics["total_error_count"] = {
                "value": total_errors,
                "unit": "errors",
                "status": ComponentStatus.HEALTHY
            }

            return {
                "status": ComponentStatus.HEALTHY,  # Error handler is always "healthy" - it handles errors
                "metrics": metrics
            }

        except Exception as e:
            logger.error(f"Error handler health check failed: {e}")
            return {
                "status": ComponentStatus.WARNING,
                "metrics": {},
                "error": str(e)
            }

    async def _check_gee_health(self) -> Dict[str, Any]:
        """Check Google Earth Engine health"""
        try:
            # Simple connectivity check
            # This would normally make a lightweight GEE API call
            metrics = {}

            # Placeholder - in real implementation, test actual GEE connectivity
            gee_status = ComponentStatus.HEALTHY  # Assume healthy for now

            metrics["gee_connectivity"] = {
                "value": "AVAILABLE",
                "unit": "status",
                "status": gee_status
            }

            return {
                "status": gee_status,
                "metrics": metrics
            }

        except Exception as e:
            logger.error(f"GEE health check failed: {e}")
            return {
                "status": ComponentStatus.WARNING,
                "metrics": {},
                "error": str(e)
            }

    async def _check_iot_health(self) -> Dict[str, Any]:
        """Check IoT sensors health"""
        try:
            metrics = {}

            # Placeholder - in real implementation, check sensor connectivity
            iot_status = ComponentStatus.HEALTHY  # Assume healthy for now

            metrics["sensor_network_status"] = {
                "value": "CONNECTED",
                "unit": "status",
                "status": iot_status
            }

            return {
                "status": iot_status,
                "metrics": metrics
            }

        except Exception as e:
            logger.error(f"IoT health check failed: {e}")
            return {
                "status": ComponentStatus.WARNING,
                "metrics": {},
                "error": str(e)
            }

    async def _process_alerts(self):
        """Process alert rules and generate alerts"""
        alerts_triggered = []

        for component in self.components.values():
            for metric in component.metrics.values():
                for rule in self.alert_rules:
                    if rule.metric_name == metric.name:
                        if rule.should_trigger(metric):
                            alert = {
                                'timestamp': datetime.utcnow(),
                                'rule_name': rule.name,
                                'component': component.name,
                                'metric': metric.name,
                                'value': metric.value,
                                'threshold': rule.threshold,
                                'condition': rule.condition,
                                'severity': rule.severity,
                                'message': f"{component.name} {metric.name} {rule.condition} {rule.threshold} ({metric.value})"
                            }

                            alerts_triggered.append(alert)
                            rule.last_triggered = datetime.utcnow()

                            # Log alert
                            logger.warning(f"ALERT {rule.severity.upper()}: {alert['message']}")

                            # In production, this would send emails/SMS/push notifications

        return alerts_triggered

    async def _store_health_report(self):
        """Store health report in history"""
        health_report = {
            'timestamp': datetime.utcnow(),
            'overall_health': self.get_overall_health().value,
            'components': {name: comp.to_dict() for name, comp in self.components.items()}
        }

        self.health_history.append(health_report)

        # Keep only last N reports
        if len(self.health_history) > self.history_length:
            self.health_history = self.health_history[-self.history_length:]

    def get_overall_health(self) -> HealthStatus:
        """Calculate overall system health"""
        component_statuses = [comp.status for comp in self.components.values()]

        # System is CRITICAL if any component is DOWN/ERROR
        if any(status == ComponentStatus.DOWN or status == ComponentStatus.ERROR for status in component_statuses):
            return HealthStatus.CRITICAL

        # System is DEGRADED if multiple components are WARNING
        warning_count = sum(1 for status in component_statuses if status == ComponentStatus.WARNING)
        if warning_count > 2:
            return HealthStatus.DEGRADED

        # System is WARNING if any component is WARNING
        if warning_count > 0:
            return HealthStatus.WARNING

        # All components healthy
        return HealthStatus.HEALTHY

    def get_detailed_health_report(self) -> Dict[str, Any]:
        """Get detailed health report"""
        return {
            'overall_health': self.get_overall_health().value,
            'timestamp': datetime.utcnow().isoformat(),
            'components': {name: comp.to_dict() for name, comp in self.components.items()},
            'alert_rules': len(self.alert_rules),
            'monitoring_enabled': self.monitoring_enabled,
            'check_interval_seconds': self.check_interval_seconds,
            'history_length': len(self.health_history),
            'system_info': self._get_system_info()
        }

    def _get_system_info(self) -> Dict[str, Any]:
        """Get basic system information"""
        return {
            'hostname': socket.gethostname(),
            'python_version': '.'.join(map(str, __import__('sys').version_info[:3])),
            'platform': __import__('platform').platform(),
            'process_id': __import__('os').getpid(),
            'uptime_seconds': int((datetime.utcnow() - datetime.fromtimestamp(__import__('psutil').boot_time())).total_seconds())
        }

    def get_health_trends(self, hours: int = 24) -> Dict[str, Any]:
        """Get health trends over time"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)

        recent_reports = [
            report for report in self.health_history
            if datetime.fromisoformat(report['timestamp']) > cutoff_time
        ]

        if not recent_reports:
            return {'error': 'No recent health reports available'}

        # Analyze trends
        health_states = [report['overall_health'] for report in recent_reports]
        component_healths = {}

        for report in recent_reports:
            for comp_name, comp_data in report['components'].items():
                if comp_name not in component_healths:
                    component_healths[comp_name] = []
                component_healths[comp_name].append(comp_data['status'])

        # Calculate stability
        health_changes = len(set(health_states))
        avg_health_score = self._calculate_average_health_score(health_states)

        return {
            'timeframe_hours': hours,
            'reports_count': len(recent_reports),
            'health_states': health_states,
            'health_changes': health_changes,
            'average_health_score': avg_health_score,
            'component_stability': {
                name: len(set(statuses)) for name, statuses in component_healths.items()
            }
        }

    def _calculate_average_health_score(self, health_states: List[str]) -> float:
        """Calculate average health score (0-100)"""
        score_mapping = {
            'HEALTHY': 100,
            'WARNING': 75,
            'DEGRADED': 50,
            'CRITICAL': 25,
            'DOWN': 0
        }

        scores = [score_mapping.get(state, 50) for state in health_states]
        return sum(scores) / len(scores) if scores else 50

# Global health monitor instance
health_monitor = AdvancedHealthMonitor()

# Convenience functions
async def initialize_health_monitoring() -> bool:
    """Initialize health monitoring system"""
    return await health_monitor.initialize_monitoring()

async def get_health_status() -> Dict[str, Any]:
    """Get current health status"""
    return health_monitor.get_detailed_health_report()

async def get_health_trends(hours: int = 24) -> Dict[str, Any]:
    """Get health trends over specified time period"""
    return health_monitor.get_health_trends(hours)
