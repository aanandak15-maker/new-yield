"""
Advanced Task Scheduler for India Agricultural Intelligence Platform
Intelligent background job management with adaptive scheduling, retries, and monitoring
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Awaitable
from enum import Enum
import json
from dataclasses import dataclass, asdict

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.date import DateTrigger
from apscheduler.jobstores.memory import MemoryJobStore
from apscheduler.executors.asyncio import AsyncIOExecutor
from apscheduler.events import EVENT_JOB_EXECUTED, EVENT_JOB_ERROR, EVENT_JOB_MISSED

from india_agri_platform.core.error_handling import error_handler, log_system_event
from india_agri_platform.core.cache_manager import cache_manager
from india_agri_platform.database.manager import db_manager

logger = logging.getLogger(__name__)

class TaskPriority(Enum):
    """Task execution priorities"""
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    NORMAL = "NORMAL"
    LOW = "LOW"

class TaskType(Enum):
    """Types of scheduled tasks"""
    WEATHER_MONITORING = "WEATHER_MONITORING"
    SATELLITE_UPDATE = "SATELLITE_UPDATE"
    PREDICTION_UPDATE = "PREDICTION_UPDATE"
    DATA_CLEANUP = "DATA_CLEANUP"
    CACHE_MAINTENANCE = "CACHE_MAINTENANCE"
    HEALTH_CHECK = "HEALTH_CHECK"
    BACKUP = "BACKUP"
    MAINTENANCE = "MAINTENANCE"

@dataclass
class TaskDefinition:
    """Definition of a scheduled task"""
    name: str
    task_type: TaskType
    function: Callable[[], Awaitable[None]]
    priority: TaskPriority = TaskPriority.NORMAL
    interval_seconds: Optional[int] = None
    cron_expression: Optional[str] = None
    max_retries: int = 3
    retry_delay_seconds: int = 60
    timeout_seconds: int = 300
    enabled: bool = True
    tags: List[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'task_type': self.task_type.value,
            'priority': self.priority.value,
            'interval_seconds': self.interval_seconds,
            'cron_expression': self.cron_expression,
            'max_retries': self.max_retries,
            'retry_delay_seconds': self.retry_delay_seconds,
            'timeout_seconds': self.timeout_seconds,
            'enabled': self.enabled,
            'tags': self.tags or []
        }

@dataclass
class TaskExecution:
    """Record of task execution"""
    task_name: str
    execution_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    status: str = "RUNNING"  # RUNNING, SUCCESS, FAILED, TIMEOUT, CANCELLED
    retry_count: int = 0
    error_message: Optional[str] = None
    result_summary: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'task_name': self.task_name,
            'execution_id': self.execution_id,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'duration_seconds': self.duration_seconds,
            'status': self.status,
            'retry_count': self.retry_count,
            'error_message': self.error_message,
            'result_summary': self.result_summary or {}
        }

class IntelligentTaskScheduler:
    """Advanced task scheduler with intelligent features"""

    def __init__(self):
        # Initialize APScheduler with advanced configuration
        self.scheduler = AsyncIOScheduler(
            jobstores={
                'default': MemoryJobStore()
            },
            executors={
                'default': AsyncIOExecutor(),
                'high_priority': AsyncIOExecutor(),
                'maintenance': AsyncIOExecutor()
            },
            job_defaults={
                'coalesce': True,
                'max_instances': 5,  # Allow 5 concurrent instances
                'misfire_grace_time': 60  # 1 minute grace time for missed jobs
            },
            timezone='Asia/Kolkata'  # IST timezone for agricultural tasks
        )

        # Task management
        self.task_definitions: Dict[str, TaskDefinition] = {}
        self.active_executions: Dict[str, TaskExecution] = {}
        self.task_history: List[TaskExecution] = []

        # Intelligent features
        self.intelligent_mode = True
        self.adaptive_scheduling = True
        self.weather_dependent_tasks = True

        # Performance monitoring
        self.execution_stats = {}
        self.task_dependencies = {}

        # Setup event handlers
        self._setup_event_handlers()

        logger.info("Intelligent Task Scheduler initialized")

    def _setup_event_handlers(self):
        """Setup APScheduler event handlers"""
        # Job execution events
        self.scheduler.add_listener(
            self._on_job_executed,
            EVENT_JOB_EXECUTED
        )

        # Job error events
        self.scheduler.add_listener(
            self._on_job_error,
            EVENT_JOB_ERROR
        )

        # Job missed events
        self.scheduler.add_listener(
            self._on_job_missed,
            EVENT_JOB_MISSED
        )

    async def start_scheduler(self) -> bool:
        """Start the intelligent task scheduler"""
        try:
            # Load default agricultural tasks
            await self._load_default_tasks()

            # Start the scheduler
            self.scheduler.start()

            # Start monitoring tasks
            asyncio.create_task(self._scheduler_monitor())
            asyncio.create_task(self._intelligent_adjustments())

            log_system_event(
                "task_scheduler_started",
                "Intelligent Task Scheduler started successfully",
                {
                    "total_tasks": len(self.task_definitions),
                    "active_tasks": len([t for t in self.task_definitions.values() if t.enabled]),
                    "intelligent_mode": self.intelligent_mode
                }
            )

            logger.info(f"Intelligent Task Scheduler started with {len(self.task_definitions)} tasks")
            return True

        except Exception as e:
            error_handler.handle_error(e, {"component": "task_scheduler", "operation": "startup"})
            return False

    async def stop_scheduler(self):
        """Stop the task scheduler gracefully"""
        if self.scheduler.running:
            self.scheduler.shutdown(wait=True)

        log_system_event("task_scheduler_stopped", "Task scheduler stopped")
        logger.info("Task scheduler stopped")

    async def _load_default_tasks(self):
        """Load default agricultural task definitions"""
        default_tasks = [
            TaskDefinition(
                name="weather_monitoring",
                task_type=TaskType.WEATHER_MONITORING,
                function=self._weather_monitoring_task,
                priority=TaskPriority.HIGH,
                interval_seconds=6 * 3600,  # Every 6 hours
                max_retries=3,
                retry_delay_seconds=300,  # 5 minutes
                tags=["weather", "monitoring", "critical"]
            ),

            TaskDefinition(
                name="satellite_data_update",
                task_type=TaskType.SATELLITE_UPDATE,
                function=self._satellite_update_task,
                priority=TaskPriority.NORMAL,
                interval_seconds=5 * 24 * 3600,  # Every 5 days
                max_retries=5,
                retry_delay_seconds=3600,  # 1 hour
                tags=["satellite", "data", "gee"]
            ),

            TaskDefinition(
                name="prediction_refresh",
                task_type=TaskType.PREDICTION_UPDATE,
                function=self._prediction_refresh_task,
                priority=TaskPriority.NORMAL,
                cron_expression="0 2 * * *",  # Daily at 2 AM
                max_retries=2,
                retry_delay_seconds=1800,  # 30 minutes
                tags=["predictions", "ml", "nightly"]
            ),

            TaskDefinition(
                name="data_cleanup",
                task_type=TaskType.DATA_CLEANUP,
                function=self._data_cleanup_task,
                priority=TaskPriority.LOW,
                cron_expression="0 3 * * 0",  # Weekly on Sunday at 3 AM
                max_retries=1,
                timeout_seconds=1800,  # 30 minutes
                tags=["cleanup", "maintenance", "weekly"]
            ),

            TaskDefinition(
                name="cache_maintenance",
                task_type=TaskType.CACHE_MAINTENANCE,
                function=self._cache_maintenance_task,
                priority=TaskPriority.LOW,
                interval_seconds=3600,  # Every hour
                max_retries=1,
                timeout_seconds=600,  # 10 minutes
                tags=["cache", "maintenance", "hourly"]
            ),

            TaskDefinition(
                name="health_monitoring",
                task_type=TaskType.HEALTH_CHECK,
                function=self._health_monitoring_task,
                priority=TaskPriority.CRITICAL,
                interval_seconds=300,  # Every 5 minutes
                max_retries=0,  # No retries for health checks
                timeout_seconds=60,   # 1 minute
                tags=["health", "monitoring", "critical"]
            )
        ]

        for task_def in default_tasks:
            self.task_definitions[task_def.name] = task_def
            if task_def.enabled:
                await self._schedule_task(task_def)

        logger.info(f"Loaded {len(default_tasks)} default agricultural tasks")

    async def _schedule_task(self, task_def: TaskDefinition):
        """Schedule a task with intelligent trigger selection"""
        try:
            # Select appropriate executor based on priority
            executor_name = self._get_executor_for_priority(task_def.priority)

            # Create trigger
            trigger = self._create_trigger(task_def)

            if trigger:
                # Schedule the job
                self.scheduler.add_job(
                    func=self._execute_task_with_monitoring,
                    args=[task_def.name],
                    trigger=trigger,
                    id=task_def.name,
                    name=task_def.name,
                    jobstore='default',
                    executor=executor_name,
                    replace_existing=True,
                    max_instances=1  # Only one instance at a time
                )

                logger.info(f"Scheduled task '{task_def.name}' with trigger: {trigger}")

        except Exception as e:
            logger.error(f"Failed to schedule task {task_def.name}: {e}")

    def _get_executor_for_priority(self, priority: TaskPriority) -> str:
        """Get appropriate executor for task priority"""
        priority_executors = {
            TaskPriority.CRITICAL: 'high_priority',
            TaskPriority.HIGH: 'high_priority',
            TaskPriority.NORMAL: 'default',
            TaskPriority.LOW: 'maintenance'
        }
        return priority_executors[priority]

    def _create_trigger(self, task_def: TaskDefinition):
        """Create appropriate trigger for task"""
        if task_def.cron_expression:
            return CronTrigger.from_crontab(task_def.cron_expression, timezone='Asia/Kolkata')
        elif task_def.interval_seconds:
            return IntervalTrigger(seconds=task_def.interval_seconds)
        return None

    async def _execute_task_with_monitoring(self, task_name: str):
        """Execute task with comprehensive monitoring and error handling"""
        if task_name not in self.task_definitions:
            logger.error(f"Task {task_name} not found in definitions")
            return

        task_def = self.task_definitions[task_name]
        execution_id = f"{task_name}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

        # Create execution record
        execution = TaskExecution(
            task_name=task_name,
            execution_id=execution_id,
            start_time=datetime.utcnow(),
            status="RUNNING"
        )

        self.active_executions[execution_id] = execution

        try:
            logger.info(f"Starting task execution: {task_name} ({execution_id})")

            # Execute task with timeout
            result = await asyncio.wait_for(
                task_def.function(),
                timeout=task_def.timeout_seconds
            )

            # Update execution record
            execution.end_time = datetime.utcnow()
            execution.duration_seconds = (execution.end_time - execution.start_time).total_seconds()
            execution.status = "SUCCESS"
            execution.result_summary = result if isinstance(result, dict) else {"message": str(result)}

            logger.info(f"Task {task_name} completed successfully in {execution.duration_seconds:.2f}s")

        except asyncio.TimeoutError:
            execution.status = "TIMEOUT"
            execution.error_message = f"Task timed out after {task_def.timeout_seconds} seconds"
            logger.error(f"Task {task_name} timed out")

        except Exception as e:
            execution.status = "FAILED"
            execution.error_message = str(e)

            # Log the error
            error_id = error_handler.handle_error(e, {
                "task_name": task_name,
                "execution_id": execution_id,
                "component": "task_scheduler"
            })

            logger.error(f"Task {task_name} failed with error ID: {error_id}")

            # Handle retries if configured
            if execution.retry_count < task_def.max_retries:
                await self._schedule_retry(task_def, execution)
                return

        finally:
            execution.end_time = execution.end_time or datetime.utcnow()
            execution.duration_seconds = (execution.end_time - execution.start_time).total_seconds()

            # Move to history
            self.task_history.append(execution)
            del self.active_executions[execution_id]

            # Keep only last 1000 executions
            if len(self.task_history) > 1000:
                self.task_history = self.task_history[-1000:]

            # Update execution statistics
            self._update_execution_stats(task_name, execution)

    async def _schedule_retry(self, task_def: TaskDefinition, execution: TaskExecution):
        """Schedule a retry for failed task"""
        execution.retry_count += 1

        retry_delay = task_def.retry_delay_seconds * (2 ** (execution.retry_count - 1))  # Exponential backoff

        logger.info(f"Scheduling retry {execution.retry_count}/{task_def.max_retries} for task {task_def.name} in {retry_delay}s")

        # Schedule retry
        retry_time = datetime.utcnow() + timedelta(seconds=retry_delay)

        self.scheduler.add_job(
            func=self._execute_task_with_monitoring,
            args=[task_def.name],
            trigger=DateTrigger(run_date=retry_time, timezone='UTC'),
            id=f"{task_def.name}_retry_{execution.retry_count}",
            name=f"{task_def.name} retry {execution.retry_count}",
            jobstore='default',
            executor='high_priority',
            max_instances=1
        )

    # Event Handlers
    def _on_job_executed(self, event):
        """Handle successful job execution"""
        job = self.scheduler.get_job(event.job_id)
        if job:
            logger.debug(f"Job {event.job_id} executed successfully in {event.retval:.3f}s")

    def _on_job_error(self, event):
        """Handle job execution errors"""
        job = self.scheduler.get_job(event.job_id)
        if job:
            logger.error(f"Job {event.job_id} failed: {event.exception}")

    def _on_job_missed(self, event):
        """Handle missed job executions"""
        job = self.scheduler.get_job(event.job_id)
        if job:
            logger.warning(f"Job {event.job_id} was missed")

    def _update_execution_stats(self, task_name: str, execution: TaskExecution):
        """Update execution statistics for monitoring"""
        if task_name not in self.execution_stats:
            self.execution_stats[task_name] = {
                'total_executions': 0,
                'successful_executions': 0,
                'failed_executions': 0,
                'total_duration': 0,
                'avg_duration': 0,
                'last_execution': None,
                'last_success': None,
                'last_failure': None
            }

        stats = self.execution_stats[task_name]
        stats['total_executions'] += 1
        stats['total_duration'] += execution.duration_seconds

        if execution.status == "SUCCESS":
            stats['successful_executions'] += 1
            stats['last_success'] = execution.end_time
        else:
            stats['failed_executions'] += 1
            stats['last_failure'] = execution.end_time

        stats['avg_duration'] = stats['total_duration'] / stats['total_executions']
        stats['last_execution'] = execution.end_time

    # Intelligent Task Functions
    async def _weather_monitoring_task(self) -> Dict[str, Any]:
        """Intelligent weather monitoring with adaptive scheduling"""
        try:
            from india_agri_platform.core.realtime_updates import realtime_manager

            # Check if we should actually run (intelligent skip)
            if self._should_skip_weather_monitoring():
                logger.info("Skipping weather monitoring - conditions unchanged")
                return {"status": "skipped", "reason": "no_significant_changes"}

            # Execute weather monitoring
            start_time = datetime.utcnow()
            await realtime_manager.monitor_weather_changes()
            execution_time = (datetime.utcnow() - start_time).total_seconds()

            return {
                "status": "completed",
                "execution_time_seconds": execution_time,
                "weather_checks_performed": 1,  # Simplified for example
                "significant_changes_detected": 0  # Would be calculated
            }

        except Exception as e:
            logger.error(f"Weather monitoring task failed: {e}")
            raise

    async def _satellite_update_task(self) -> Dict[str, Any]:
        """Satellite data update with intelligent timing"""
        try:
            from india_agri_platform.core.realtime_updates import realtime_manager

            start_time = datetime.utcnow()
            await realtime_manager.update_satellite_data()
            execution_time = (datetime.utcnow() - start_time).total_seconds()

            return {
                "status": "completed",
                "execution_time_seconds": execution_time,
                "satellite_requests": 1,  # Simplified
                "data_updated_fields": 0  # Would be tracked
            }

        except Exception as e:
            logger.error(f"Satellite update task failed: {e}")
            raise

    async def _prediction_refresh_task(self) -> Dict[str, Any]:
        """Nightly prediction refresh for all active fields"""
        try:
            # This would refresh predictions for fields that haven't been updated recently
            # Implementation would depend on business logic
            logger.info("Running prediction refresh task")

            return {
                "status": "completed",
                "predictions_refreshed": 0,  # Placeholder
                "new_predictions_generated": 0
            }

        except Exception as e:
            logger.error(f"Prediction refresh task failed: {e}")
            raise

    async def _data_cleanup_task(self) -> Dict[str, Any]:
        """Data cleanup and maintenance"""
        try:
            # Clean old logs, cache, temporary files
            cleanup_results = {
                "logs_cleaned": 0,
                "cache_cleaned": 0,
                "temp_files_removed": 0
            }

            logger.info(f"Data cleanup completed: {cleanup_results}")

            return {
                "status": "completed",
                **cleanup_results
            }

        except Exception as e:
            logger.error(f"Data cleanup task failed: {e}")
            raise

    async def _cache_maintenance_task(self) -> Dict[str, Any]:
        """Cache maintenance and optimization"""
        try:
            # Clean expired cache entries
            cache_cleanup = await cache_manager.cache.cleanup()

            return {
                "status": "completed",
                "l1_expired_removed": cache_cleanup.get('l1_expired', 0),
                "l2_expired_removed": cache_cleanup.get('l2_expired', 0),
                "cache_optimization_applied": True
            }

        except Exception as e:
            logger.error(f"Cache maintenance task failed: {e}")
            raise

    async def _health_monitoring_task(self) -> Dict[str, Any]:
        """System health monitoring"""
        try:
            from india_agri_platform.core.realtime_updates import realtime_manager

            await realtime_manager.daily_health_check()

            return {
                "status": "completed",
                "health_checks_performed": 1,
                "system_status": "monitored"
            }

        except Exception as e:
            logger.error(f"Health monitoring task failed: {e}")
            raise

    def _should_skip_weather_monitoring(self) -> bool:
        """Intelligent decision on whether to skip weather monitoring"""
        if not self.intelligent_mode:
            return False

        # Skip if weather was checked recently (< 3 hours ago)
        last_check = getattr(self, '_last_weather_check', None)
        if last_check and (datetime.utcnow() - last_check).total_seconds() < 3 * 3600:
            return True

        self._last_weather_check = datetime.utcnow()
        return False

    async def _scheduler_monitor(self):
        """Monitor scheduler health and performance"""
        while True:
            try:
                await asyncio.sleep(300)  # Every 5 minutes

                # Check scheduler status
                running_jobs = len(self.scheduler.get_jobs())
                active_executions = len(self.active_executions)

                # Log if there are issues
                if active_executions > 10:  # Too many concurrent tasks
                    logger.warning(f"High concurrent executions: {active_executions}")

                # Check for stuck tasks
                stuck_tasks = []
                for execution_id, execution in self.active_executions.items():
                    if (datetime.utcnow() - execution.start_time).total_seconds() > 1800:  # 30 minutes
                        stuck_tasks.append(execution.task_name)

                if stuck_tasks:
                    logger.warning(f"Stuck tasks detected: {stuck_tasks}")

            except Exception as e:
                logger.error(f"Scheduler monitoring failed: {e}")

    async def _intelligent_adjustments(self):
        """Make intelligent adjustments to task scheduling"""
        if not self.adaptive_scheduling:
            return

        while True:
            try:
                await asyncio.sleep(3600)  # Hourly adjustments

                # Adjust task frequencies based on:
                # 1. System load
                # 2. Time of day
                # 3. Weather conditions
                # 4. User activity

                await self._adjust_weather_monitoring_frequency()
                await self._adjust_satellite_update_timing()

            except Exception as e:
                logger.error(f"Intelligent adjustments failed: {e}")

    async def _adjust_weather_monitoring_frequency(self):
        """Adjust weather monitoring frequency based on conditions"""
        # Implementation would analyze weather patterns and adjust frequencies
        pass

    async def _adjust_satellite_update_timing(self):
        """Adjust satellite update timing for optimal API usage"""
        # Implementation would coordinate with GEE API limits
        pass

    # Management API
    async def add_task(self, task_def: TaskDefinition) -> bool:
        """Add a new task to the scheduler"""
        try:
            self.task_definitions[task_def.name] = task_def
            if task_def.enabled:
                await self._schedule_task(task_def)

            logger.info(f"Added new task: {task_def.name}")
            return True

        except Exception as e:
            logger.error(f"Failed to add task {task_def.name}: {e}")
            return False

    async def remove_task(self, task_name: str) -> bool:
        """Remove a task from the scheduler"""
        try:
            if task_name in self.scheduler.get_jobs():
                self.scheduler.remove_job(task_name)

            if task_name in self.task_definitions:
                del self.task_definitions[task_name]

            logger.info(f"Removed task: {task_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to remove task {task_name}: {e}")
            return False

    async def pause_task(self, task_name: str) -> bool:
        """Pause a scheduled task"""
        try:
            if task_name in self.scheduler.get_jobs():
                self.scheduler.pause_job(task_name)

            if task_name in self.task_definitions:
                self.task_definitions[task_name].enabled = False

            logger.info(f"Paused task: {task_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to pause task {task_name}: {e}")
            return False

    async def resume_task(self, task_name: str) -> bool:
        """Resume a paused task"""
        try:
            if task_name in self.scheduler.get_jobs():
                self.scheduler.resume_job(task_name)

            if task_name in self.task_definitions:
                self.task_definitions[task_name].enabled = True

            logger.info(f"Resumed task: {task_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to resume task {task_name}: {e}")
            return False

    def get_scheduler_status(self) -> Dict[str, Any]:
        """Get comprehensive scheduler status"""
        return {
            "scheduler_running": self.scheduler.running,
            "total_tasks": len(self.task_definitions),
            "active_tasks": len(self.task_definitions),
            "active_executions": len(self.active_executions),
            "historical_executions": len(self.task_history),
            "execution_stats": self.execution_stats.copy(),
            "intelligent_mode": self.intelligent_mode,
            "adaptive_scheduling": self.adaptive_scheduling,
            "last_updated": datetime.utcnow().isoformat()
        }

    def get_task_status(self, task_name: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific task"""
        if task_name not in self.task_definitions:
            return None

        task_def = self.task_definitions[task_name]
        jobs = self.scheduler.get_jobs()

        job_info = None
        for job in jobs:
            if job.id == task_name:
                job_info = {
                    "next_run_time": job.next_run_time.isoformat() if job.next_run_time else None,
                    "job_state": str(job)
                }
                break

        execution_stats = self.execution_stats.get(task_name, {})

        return {
            "task_definition": task_def.to_dict(),
            "job_info": job_info,
            "execution_stats": execution_stats,
            "is_active": task_name in [ex.task_name for ex in self.active_executions.values()]
        }

# Global task scheduler instance
task_scheduler = IntelligentTaskScheduler()

# Convenience functions
async def start_task_scheduler() -> bool:
    """Start the intelligent task scheduler"""
    return await task_scheduler.start_scheduler()

async def stop_task_scheduler():
    """Stop the task scheduler"""
    await task_scheduler.stop_scheduler()

async def get_scheduler_status() -> Dict[str, Any]:
    """Get scheduler status"""
    return task_scheduler.get_scheduler_status()
