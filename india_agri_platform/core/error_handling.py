"""
Comprehensive Error Handling and Logging System
Production-ready error management for agricultural intelligence platform
"""

import logging
import logging.handlers
import json
import sys
import traceback
from datetime import datetime
from typing import Dict, Any, Optional, List
from enum import Enum
import os
from pathlib import Path

class ErrorSeverity(Enum):
    """Error severity levels"""
    CRITICAL = "CRITICAL"
    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"
    DEBUG = "DEBUG"

class ErrorCategory(Enum):
    """Error categories for better classification"""
    API_ERROR = "API_ERROR"
    DATABASE_ERROR = "DATABASE_ERROR"
    MODEL_ERROR = "MODEL_ERROR"
    DATA_ERROR = "DATA_ERROR"
    EXTERNAL_API_ERROR = "EXTERNAL_API_ERROR"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    IOT_ERROR = "IOT_ERROR"
    CACHE_ERROR = "CACHE_ERROR"
    CONFIGURATION_ERROR = "CONFIGURATION_ERROR"
    SYSTEM_ERROR = "SYSTEM_ERROR"

class AgriculturalError(Exception):
    """Base exception class for agricultural platform errors"""

    def __init__(self, message: str, category: ErrorCategory,
                 severity: ErrorSeverity = ErrorSeverity.ERROR,
                 details: Optional[Dict[str, Any]] = None,
                 recovery_action: Optional[str] = None):
        super().__init__(message)
        self.message = message
        self.category = category
        self.severity = severity
        self.details = details or {}
        self.recovery_action = recovery_action
        self.timestamp = datetime.utcnow()
        self.traceback = traceback.format_exc()

class APIValidationError(AgriculturalError):
    """API input validation errors"""
    def __init__(self, message: str, field: str = None, value: Any = None):
        super().__init__(
            message=message,
            category=ErrorCategory.VALIDATION_ERROR,
            severity=ErrorSeverity.WARNING,
            details={"field": field, "value": str(value)[:100] if value else None},
            recovery_action="Check input parameters and API documentation"
        )

class DatabaseError(AgriculturalError):
    """Database operation errors"""
    def __init__(self, message: str, operation: str = None, table: str = None):
        super().__init__(
            message=message,
            category=ErrorCategory.DATABASE_ERROR,
            severity=ErrorSeverity.ERROR,
            details={"operation": operation, "table": table},
            recovery_action="Check database connectivity and table schema"
        )

class ModelError(AgriculturalError):
    """Machine learning model errors"""
    def __init__(self, message: str, model_name: str = None, input_shape: tuple = None):
        super().__init__(
            message=message,
            category=ErrorCategory.MODEL_ERROR,
            severity=ErrorSeverity.ERROR,
            details={"model_name": model_name, "input_shape": input_shape},
            recovery_action="Check model file integrity and input data format"
        )

class ExternalAPIError(AgriculturalError):
    """External API (weather, satellite, etc.) errors"""
    def __init__(self, message: str, api_name: str, status_code: int = None):
        super().__init__(
            message=message,
            category=ErrorCategory.EXTERNAL_API_ERROR,
            severity=ErrorSeverity.WARNING,
            details={"api_name": api_name, "status_code": status_code},
            recovery_action="Check API credentials and network connectivity"
        )

class IoTError(AgriculturalError):
    """IoT sensor and device errors"""
    def __init__(self, message: str, sensor_id: str, device_type: str = None):
        super().__init__(
            message=message,
            category=ErrorCategory.IOT_ERROR,
            severity=ErrorSeverity.ERROR,
            details={"sensor_id": sensor_id, "device_type": device_type},
            recovery_action="Check sensor connectivity and battery levels"
        )

class CacheError(AgriculturalError):
    """Cache system errors"""
    def __init__(self, message: str, operation: str = None, key: str = None):
        super().__init__(
            message=message,
            category=ErrorCategory.CACHE_ERROR,
            severity=ErrorSeverity.WARNING,
            details={"operation": operation, "key": key},
            recovery_action="Check Redis connection and verify cache configuration"
        )

class DataError(AgriculturalError):
    """Data processing and validation errors"""
    def __init__(self, message: str, data_type: str = None, validation_rule: str = None):
        super().__init__(
            message=message,
            category=ErrorCategory.DATA_ERROR,
            severity=ErrorSeverity.ERROR,
            details={"data_type": data_type, "validation_rule": validation_rule},
            recovery_action="Validate data format, check for missing values, and ensure data consistency"
        )

class ErrorHandler:
    """Centralized error handling and logging management"""

    def __init__(self):
        self.log_directory = Path("logs")
        self.log_directory.mkdir(exist_ok=True)

        # Initialize loggers
        self._setup_loggers()

        # Error tracking
        self.error_counts = {}
        self.recent_errors = []

    def _setup_loggers(self):
        """Setup comprehensive logging configuration"""

        # Main application logger
        self.app_logger = logging.getLogger('agri_platform')
        self.app_logger.setLevel(logging.DEBUG)

        # Error-specific logger
        self.error_logger = logging.getLogger('agri_errors')
        self.error_logger.setLevel(logging.DEBUG)

        # Database logger
        self.db_logger = logging.getLogger('agri_database')
        self.db_logger.setLevel(logging.DEBUG)

        # API logger
        self.api_logger = logging.getLogger('agri_api')
        self.api_logger.setLevel(logging.DEBUG)

        # IoT logger
        self.iot_logger = logging.getLogger('agri_iot')
        self.iot_logger.setLevel(logging.DEBUG)

        # External API logger
        self.external_logger = logging.getLogger('agri_external')
        self.external_logger.setLevel(logging.DEBUG)

        # Don't propagate to root logger
        for logger in [self.app_logger, self.error_logger, self.db_logger,
                      self.api_logger, self.iot_logger, self.external_logger]:
            logger.propagate = False

        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )

        json_formatter = logging.Formatter(
            '{"timestamp": "%(asctime)s", "logger": "%(name)s", "level": "%(levelname)s", '
            '"function": "%(funcName)s", "line": %(lineno)d, "message": "%(message)s"}'
        )

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(detailed_formatter)

        # File handlers
        file_handlers = self._create_file_handlers(detailed_formatter, json_formatter)

        # Add handlers to loggers
        self._configure_logger_handlers(console_handler, file_handlers)

    def _create_file_handlers(self, detailed_formatter, json_formatter):
        """Create file handlers for different log types"""
        handlers = {}

        # General application log (rotating)
        handlers['app_file'] = logging.handlers.RotatingFileHandler(
            self.log_directory / "application.log",
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        handlers['app_file'].setLevel(logging.DEBUG)
        handlers['app_file'].setFormatter(detailed_formatter)

        # Error log (separate file)
        handlers['error_file'] = logging.handlers.RotatingFileHandler(
            self.log_directory / "errors.log",
            maxBytes=5*1024*1024,   # 5MB
            backupCount=10
        )
        handlers['error_file'].setLevel(logging.ERROR)
        handlers['error_file'].setFormatter(detailed_formatter)

        # JSON structured log
        handlers['json_file'] = logging.handlers.RotatingFileHandler(
            self.log_directory / "structured.log",
            maxBytes=20*1024*1024,  # 20MB
            backupCount=3
        )
        handlers['json_file'].setLevel(logging.INFO)
        handlers['json_file'].setFormatter(json_formatter)

        # Separate logs for different components
        component_logs = ['database', 'api', 'iot', 'external']

        for component in component_logs:
            handlers[f'{component}_file'] = logging.handlers.RotatingFileHandler(
                self.log_directory / f"{component}.log",
                maxBytes=5*1024*1024,
                backupCount=3
            )
            handlers[f'{component}_file'].setLevel(logging.DEBUG)
            handlers[f'{component}_file'].setFormatter(detailed_formatter)

        return handlers

    def _configure_logger_handlers(self, console_handler, file_handlers):
        """Configure handlers for each logger"""

        # App logger - all handlers
        self.app_logger.addHandler(console_handler)
        self.app_logger.addHandler(file_handlers['app_file'])
        self.app_logger.addHandler(file_handlers['json_file'])

        # Error logger - error file and console for errors
        error_console = logging.StreamHandler(sys.stderr)
        error_console.setLevel(logging.ERROR)
        error_console.setFormatter(logging.Formatter('%(asctime)s - ERROR - %(message)s'))

        self.error_logger.addHandler(error_console)
        self.error_logger.addHandler(file_handlers['error_file'])
        self.error_logger.addHandler(file_handlers['json_file'])

        # Database logger
        self.db_logger.addHandler(console_handler)
        self.db_logger.addHandler(file_handlers['app_file'])
        self.db_logger.addHandler(file_handlers['database_file'])
        self.db_logger.addHandler(file_handlers['json_file'])

        # API logger
        self.api_logger.addHandler(console_handler)
        self.api_logger.addHandler(file_handlers['app_file'])
        self.api_logger.addHandler(file_handlers['api_file'])
        self.api_logger.addHandler(file_handlers['json_file'])

        # IoT logger
        self.iot_logger.addHandler(console_handler)
        self.iot_logger.addHandler(file_handlers['app_file'])
        self.iot_logger.addHandler(file_handlers['iot_file'])
        self.iot_logger.addHandler(file_handlers['json_file'])

        # External API logger
        self.external_logger.addHandler(console_handler)
        self.external_logger.addHandler(file_handlers['app_file'])
        self.external_logger.addHandler(file_handlers['external_file'])
        self.external_logger.addHandler(file_handlers['json_file'])

    def handle_error(self, error: Exception, context: Optional[Dict[str, Any]] = None) -> str:
        """Central error handling method"""
        error_id = f"ERR_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{id(error)}"

        # Categorize error
        error_info = self._categorize_error(error)

        # Log error with metadata
        self.error_logger.error(f"[{error_id}] {error_info['message']}", extra={
            'error_id': error_id,
            'category': error_info['category'].value,
            'severity': error_info['severity'].value,
            'traceback': traceback.format_exc(),
            'context': context or {}
        })

        # Update error counts
        self._update_error_counts(error_info['category'])

        # Store recent error
        self._store_recent_error(error_id, error_info)

        # Get recovery action
        recovery_action = self._get_recovery_action(error_info, context)

        # Log recovery action
        self.app_logger.info(f"[{error_id}] Recovery action: {recovery_action}")

        return error_id

    def _categorize_error(self, error: Exception) -> Dict[str, Any]:
        """Categorize error type and severity"""
        if isinstance(error, AgriculturalError):
            return {
                'category': error.category,
                'severity': error.severity,
                'message': error.message,
                'details': error.details,
                'recovery_action': error.recovery_action
            }

        # Categorize based on exception type and message
        error_type = type(error).__name__
        error_message = str(error).lower()

        if any(term in error_message for term in ['database', 'sqlite', 'postgresql', 'connection']):
            category = ErrorCategory.DATABASE_ERROR
            severity = ErrorSeverity.ERROR
        elif any(term in error_message for term in ['api', 'http', 'request', 'response']):
            category = ErrorCategory.EXTERNAL_API_ERROR
            severity = ErrorSeverity.WARNING
        elif any(term in error_message for term in ['model', 'prediction', 'ml']):
            category = ErrorCategory.MODEL_ERROR
            severity = ErrorSeverity.ERROR
        elif any(term in error_message for term in ['sensor', 'iot', 'device']):
            category = ErrorCategory.IOT_ERROR
            severity = ErrorSeverity.ERROR
        elif any(term in error_message for term in ['validation', 'input']):
            category = ErrorCategory.VALIDATION_ERROR
            severity = ErrorSeverity.WARNING
        else:
            category = ErrorCategory.SYSTEM_ERROR
            severity = ErrorSeverity.ERROR

        return {
            'category': category,
            'severity': severity,
            'message': str(error),
            'details': {'exception_type': error_type},
            'recovery_action': None
        }

    def _update_error_counts(self, category: ErrorCategory):
        """Update error count statistics"""
        category_key = category.value
        self.error_counts[category_key] = self.error_counts.get(category_key, 0) + 1

    def _store_recent_error(self, error_id: str, error_info: Dict[str, Any]):
        """Store recent error for monitoring"""
        recent_error = {
            'id': error_id,
            'timestamp': datetime.utcnow(),
            'category': error_info['category'].value,
            'severity': error_info['severity'].value,
            'message': error_info['message'][:200]  # Truncate long messages
        }

        self.recent_errors.append(recent_error)

        # Keep only last 100 errors
        if len(self.recent_errors) > 100:
            self.recent_errors.pop(0)

    def _get_recovery_action(self, error_info: Dict[str, Any], context: Optional[Dict[str, Any]]) -> str:
        """Get recovery action for error"""
        if error_info.get('recovery_action'):
            return error_info['recovery_action']

        category = error_info['category']

        recovery_actions = {
            ErrorCategory.DATABASE_ERROR: "Check database connection, verify table schema, and ensure sufficient disk space",
            ErrorCategory.EXTERNAL_API_ERROR: "Verify API credentials, check network connectivity, and review API rate limits",
            ErrorCategory.MODEL_ERROR: "Verify model file exists, check input data format, and ensure required dependencies are installed",
            ErrorCategory.IOT_ERROR: "Check sensor power, verify wireless connectivity, and inspect hardware integrity",
            ErrorCategory.VALIDATION_ERROR: "Review input parameters according to API documentation",
            ErrorCategory.DATA_ERROR: "Validate data format, check for missing values, and ensure data consistency",
            ErrorCategory.CACHE_ERROR: "Check Redis connection, verify cache configuration, and clear corrupted cache entries",
            ErrorCategory.CONFIGURATION_ERROR: "Review configuration files, check environment variables, and validate settings"
        }

        return recovery_actions.get(category, "Contact system administrator for assistance")

    def log_system_event(self, event_type: str, message: str, details: Optional[Dict[str, Any]] = None):
        """Log system events for monitoring"""
        self.app_logger.info(f"SYSTEM_EVENT [{event_type}]: {message}", extra={
            'event_type': event_type,
            'details': details or {}
        })

    def log_performance_metric(self, metric_name: str, value: float, unit: str = "ms"):
        """Log performance metrics"""
        self.app_logger.debug(f"PERFORMANCE [{metric_name}]: {value} {unit}", extra={
            'metric_type': 'performance',
            'metric_name': metric_name,
            'value': value,
            'unit': unit
        })

    def log_api_request(self, method: str, endpoint: str, status_code: int, duration: float):
        """Log API request details"""
        log_level = logging.INFO if status_code < 400 else logging.WARNING if status_code < 500 else logging.ERROR

        self.api_logger.log(log_level,
                          f"API_REQUEST {method} {endpoint} -> {status_code} ({duration:.2f}ms)",
                          extra={
                              'method': method,
                              'endpoint': endpoint,
                              'status_code': status_code,
                              'duration_ms': duration,
                              'request_type': 'api'
                          })

    def log_database_operation(self, operation: str, table: str, duration: float, success: bool):
        """Log database operation details"""
        status = "SUCCESS" if success else "FAILED"
        level = logging.INFO if success else logging.WARNING

        self.db_logger.log(level,
                          f"DB_OPERATION {operation} on {table} -> {status} ({duration:.3f}ms)",
                          extra={
                              'operation': operation,
                              'table': table,
                              'duration_ms': duration,
                              'success': success,
                              'operation_type': 'database'
                          })

    def log_external_api_call(self, api_name: str, endpoint: str, status_code: int, duration: float):
        """Log external API call details"""
        status = "SUCCESS" if status_code < 400 else f"ERROR_{status_code}"
        level = logging.INFO if status_code < 400 else logging.WARNING

        self.external_logger.log(level,
                                f"EXTERNAL_API {api_name} {endpoint} -> {status} ({duration:.3f}ms)",
                                extra={
                                    'api_name': api_name,
                                    'endpoint': endpoint,
                                    'status_code': status_code,
                                    'duration_ms': duration,
                                    'call_type': 'external_api'
                                })

    def log_iot_event(self, sensor_id: str, event_type: str, message: str, details: Optional[Dict[str, Any]] = None):
        """Log IoT sensor events"""
        self.iot_logger.info(f"IOT_EVENT {sensor_id} [{event_type}]: {message}", extra={
            'sensor_id': sensor_id,
            'event_type': event_type,
            'details': details or {},
            'iot_event': True
        })

    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics for monitoring"""
        return {
            'total_errors': sum(self.error_counts.values()),
            'error_counts_by_category': self.error_counts.copy(),
            'recent_errors': self.recent_errors[-10:],  # Last 10 errors
            'timestamp': datetime.utcnow().isoformat()
        }

    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health status"""
        critical_errors = self.error_counts.get('CRITICAL', 0)
        error_count = self.error_counts.get('ERROR', 0)
        warning_count = self.error_counts.get('WARNING', 0)

        if critical_errors > 0:
            health_status = "CRITICAL"
        elif error_count > 10:
            health_status = "DEGRADED"
        elif warning_count > 50:
            health_status = "WARNING"
        else:
            health_status = "HEALTHY"

        return {
            'status': health_status,
            'error_counts': self.error_counts,
            'recent_error_count': len([e for e in self.recent_errors
                                     if (datetime.utcnow() - e['timestamp']).seconds < 3600]),  # Last hour
            'last_error': self.recent_errors[-1] if self.recent_errors else None,
            'timestamp': datetime.utcnow().isoformat()
        }

# Global error handler instance
error_handler = ErrorHandler()

# Convenience functions for global use
def log_error(error: Exception, context: Optional[Dict[str, Any]] = None) -> str:
    """Log an error and return error ID"""
    return error_handler.handle_error(error, context)

def log_system_event(event_type: str, message: str, details: Optional[Dict[str, Any]] = None):
    """Log system events"""
    error_handler.log_system_event(event_type, message, details)

def log_performance(metric_name: str, value: float, unit: str = "ms"):
    """Log performance metrics"""
    error_handler.log_performance_metric(metric_name, value, unit)

def log_api_request(method: str, endpoint: str, status_code: int, duration: float):
    """Log API requests"""
    error_handler.log_api_request(method, endpoint, status_code, duration)

def log_db_operation(operation: str, table: str, duration: float, success: bool):
    """Log database operations"""
    error_handler.log_database_operation(operation, table, duration, success)

def get_error_stats() -> Dict[str, Any]:
    """Get error statistics"""
    return error_handler.get_error_statistics()

def get_system_health() -> Dict[str, Any]:
    """Get system health status"""
    return error_handler.get_system_health()
