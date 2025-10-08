"""
Real-time Update System for India Agricultural Intelligence Platform
Automatically updates predictions based on weather changes and satellite data
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import logging
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.jobstores.memory import MemoryJobStore
from apscheduler.executors.asyncio import AsyncIOExecutor

from india_agri_platform.core.streamlined_predictor import streamlined_predictor
from india_agri_platform.database.manager import db_manager
from india_agri_platform.core.gee_integration import gee_client, satellite_processor

logger = logging.getLogger(__name__)

class RealTimeUpdateManager:
    """Manages real-time updates for agricultural predictions"""

    def __init__(self):
        # Initialize scheduler
        self.scheduler = AsyncIOScheduler(
            jobstores={
                'default': MemoryJobStore()
            },
            executors={
                'default': AsyncIOExecutor()
            },
            job_defaults={
                'coalesce': True,
                'max_instances': 3,
                'misfire_grace_time': 30
            }
        )

        self.predictor = streamlined_predictor
        self.db = db_manager
        self.gee_client = gee_client
        self.satellite_processor = satellite_processor

        # Update tracking
        self.last_weather_check = {}
        self.last_satellite_update = {}
        self.active_jobs = {}

        logger.info("Real-time Update Manager initialized")

    async def start_scheduler(self):
        """Start the background job scheduler"""
        try:
            # Weather monitoring job - every 6 hours
            self.scheduler.add_job(
                func=self.monitor_weather_changes,
                trigger=IntervalTrigger(hours=6),
                id='weather_monitor',
                name='Weather Change Monitor',
                replace_existing=True
            )

            # Satellite data update job - every 5 days
            self.scheduler.add_job(
                func=self.update_satellite_data,
                trigger=IntervalTrigger(days=5),
                id='satellite_update',
                name='Satellite Data Update',
                replace_existing=True
            )

            # Daily health check job
            self.scheduler.add_job(
                func=self.daily_health_check,
                trigger=IntervalTrigger(hours=24),
                id='daily_health_check',
                name='Daily Health Check',
                replace_existing=True
            )

            # Start the scheduler
            self.scheduler.start()
            logger.info("Real-time update scheduler started successfully")

            # Keep the scheduler running
            while True:
                await asyncio.sleep(60)  # Check every minute
                if not self.scheduler.running:
                    logger.warning("Scheduler stopped, restarting...")
                    self.scheduler.start()

        except Exception as e:
            logger.error(f"Failed to start scheduler: {e}")
            raise

    async def stop_scheduler(self):
        """Stop the background job scheduler"""
        if self.scheduler.running:
            self.scheduler.shutdown(wait=True)
            logger.info("Real-time update scheduler stopped")

    async def monitor_weather_changes(self):
        """Monitor weather changes and update predictions accordingly"""
        try:
            logger.info("Starting weather change monitoring")

            # Get all active fields
            active_fields = await self.get_active_fields()

            if not active_fields:
                logger.info("No active fields found for weather monitoring")
                return

            logger.info(f"Monitoring weather for {len(active_fields)} active fields")

            update_count = 0
            alert_count = 0

            for field in active_fields:
                try:
                    # Check for significant weather changes
                    weather_change = await self.detect_weather_change(field)

                    if weather_change['significant']:
                        logger.info(f"Significant weather change detected for field {field['id']}")

                        # Recalculate prediction
                        new_prediction = await self.predict_yield_for_field(field)

                        if new_prediction:
                            # Save updated prediction
                            await self.save_updated_prediction(
                                field, new_prediction, 'weather_change', weather_change
                            )

                            # Send alert to farmer
                            await self.send_weather_alert(field, weather_change, new_prediction)

                            update_count += 1
                            alert_count += 1

                    # Update last check timestamp
                    self.last_weather_check[field['id']] = datetime.utcnow()

                except Exception as e:
                    logger.error(f"Error processing field {field['id']}: {e}")
                    continue

            logger.info(f"Weather monitoring complete: {update_count} predictions updated, {alert_count} alerts sent")

        except Exception as e:
            logger.error(f"Weather monitoring failed: {e}")

    async def update_satellite_data(self):
        """Update satellite data for all active fields"""
        try:
            logger.info("Starting satellite data update")

            # Get all active fields
            active_fields = await self.get_active_fields()

            if not active_fields:
                logger.info("No active fields found for satellite update")
                return

            logger.info(f"Updating satellite data for {len(active_fields)} active fields")

            update_count = 0

            for field in active_fields:
                try:
                    # Update satellite data for this field
                    satellite_updated = await self.update_field_satellite_data(field)

                    if satellite_updated:
                        # Check for vegetation changes
                        vegetation_change = await self.detect_vegetation_change(field)

                        if vegetation_change['significant']:
                            logger.info(f"Significant vegetation change detected for field {field['id']}")

                            # Recalculate prediction
                            new_prediction = await self.predict_yield_for_field(field)

                            if new_prediction:
                                # Save updated prediction
                                await self.save_updated_prediction(
                                    field, new_prediction, 'satellite_update', vegetation_change
                                )

                                # Send alert
                                await self.send_satellite_alert(field, vegetation_change, new_prediction)

                                update_count += 1

                    # Update last satellite update timestamp
                    self.last_satellite_update[field['id']] = datetime.utcnow()

                except Exception as e:
                    logger.error(f"Error updating satellite data for field {field['id']}: {e}")
                    continue

            logger.info(f"Satellite update complete: {update_count} predictions updated")

        except Exception as e:
            logger.error(f"Satellite update failed: {e}")

    async def daily_health_check(self):
        """Perform daily system health check"""
        try:
            logger.info("Performing daily health check")

            # Check database connectivity
            health = self.db.health_check()

            # Log system statistics
            stats = {
                'total_predictions': health.get('total_predictions', 0),
                'total_fields': health.get('total_fields', 0),
                'total_satellite_data': health.get('total_satellite_data', 0),
                'active_weather_monitors': len(self.last_weather_check),
                'active_satellite_updates': len(self.last_satellite_update),
                'scheduler_running': self.scheduler.running
            }

            logger.info(f"Daily health check: {stats}")

            # Check for any issues
            if not health.get('database_connected'):
                logger.error("Database connectivity issue detected")
            elif not self.scheduler.running:
                logger.warning("Scheduler not running, attempting restart")
                try:
                    self.scheduler.start()
                except Exception as e:
                    logger.error(f"Failed to restart scheduler: {e}")

        except Exception as e:
            logger.error(f"Daily health check failed: {e}")

    async def get_active_fields(self) -> List[Dict[str, Any]]:
        """Get all active fields that need monitoring"""
        try:
            # This would typically come from database
            # For now, return mock data - replace with actual DB call
            return [
                {
                    'id': 1,
                    'farmer_id': 1,
                    'name': 'North Field',
                    'centroid_lat': 30.9010,
                    'centroid_lng': 75.8573,
                    'area_hectares': 5.2,
                    'crop_type': 'wheat',
                    'sowing_date': '2024-11-15',
                    'irrigation_type': 'canal'
                }
            ]
        except Exception as e:
            logger.error(f"Failed to get active fields: {e}")
            return []

    async def detect_weather_change(self, field: Dict[str, Any]) -> Dict[str, Any]:
        """Detect significant weather changes for a field"""
        try:
            # Get current weather
            current_weather = await self.get_current_weather(field['centroid_lat'], field['centroid_lng'])

            # Get baseline weather (from last prediction or historical average)
            baseline_weather = await self.get_baseline_weather(field)

            # Calculate changes
            temp_change = abs(current_weather.get('temperature_c', 25) - baseline_weather.get('temperature_c', 25))
            rainfall_change = abs(current_weather.get('rainfall_mm', 0) - baseline_weather.get('rainfall_mm', 0))

            # Define significance thresholds
            significant = (
                temp_change > 5.0 or  # 5°C temperature change
                rainfall_change > 20  # 20mm rainfall change
            )

            return {
                'significant': significant,
                'temperature_change': temp_change,
                'rainfall_change': rainfall_change,
                'current_weather': current_weather,
                'baseline_weather': baseline_weather,
                'change_type': 'extreme_weather' if significant else 'normal'
            }

        except Exception as e:
            logger.error(f"Weather change detection failed for field {field['id']}: {e}")
            return {'significant': False, 'error': str(e)}

    async def update_field_satellite_data(self, field: Dict[str, Any]) -> bool:
        """Update satellite data for a specific field"""
        try:
            # Calculate date range (last 30 days)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)

            # Fetch satellite data
            satellite_data = self.gee_client.get_multi_band_data(
                latitude=field['centroid_lat'],
                longitude=field['centroid_lng'],
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d')
            )

            if satellite_data and satellite_data.get('data_source') != 'fallback_simulated':
                # Cache satellite data
                success = self.db.cache_satellite_data({
                    'field_id': field['id'],
                    'location_lat': field['centroid_lat'],
                    'location_lng': field['centroid_lng'],
                    'date': end_date.strftime('%Y-%m-%d'),
                    'ndvi': satellite_data.get('ndvi'),
                    'soil_moisture_percent': satellite_data.get('soil_moisture_percent'),
                    'land_surface_temp_c': satellite_data.get('land_surface_temp_c'),
                    'data_source': satellite_data.get('data_source')
                })

                return success

            return False

        except Exception as e:
            logger.error(f"Satellite data update failed for field {field['id']}: {e}")
            return False

    async def detect_vegetation_change(self, field: Dict[str, Any]) -> Dict[str, Any]:
        """Detect significant vegetation changes"""
        try:
            # Get recent satellite history
            satellite_history = self.db.get_satellite_history(field['id'], days=30)

            if len(satellite_history) < 2:
                return {'significant': False, 'reason': 'insufficient_data'}

            # Calculate NDVI trend
            ndvi_values = [data['ndvi'] for data in satellite_history if data['ndvi'] is not None]

            if len(ndvi_values) < 2:
                return {'significant': False, 'reason': 'insufficient_ndvi_data'}

            # Simple trend detection
            recent_avg = sum(ndvi_values[-3:]) / min(3, len(ndvi_values))
            older_avg = sum(ndvi_values[:-3]) / max(1, len(ndvi_values) - 3)

            change_percent = ((recent_avg - older_avg) / older_avg) * 100 if older_avg > 0 else 0

            significant = abs(change_percent) > 15  # 15% change threshold

            return {
                'significant': significant,
                'change_percent': change_percent,
                'recent_ndvi_avg': recent_avg,
                'older_ndvi_avg': older_avg,
                'change_type': 'vegetation_stress' if change_percent < -15 else 'vegetation_growth' if change_percent > 15 else 'stable'
            }

        except Exception as e:
            logger.error(f"Vegetation change detection failed for field {field['id']}: {e}")
            return {'significant': False, 'error': str(e)}

    async def predict_yield_for_field(self, field: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generate new yield prediction for a field"""
        try:
            result = self.predictor.predict_yield_streamlined(
                crop_name=field['crop_type'],
                sowing_date=field['sowing_date'],
                latitude=field['centroid_lat'],
                longitude=field['centroid_lng']
            )

            if 'error' in result:
                logger.error(f"Prediction failed for field {field['id']}: {result['error']}")
                return None

            return result

        except Exception as e:
            logger.error(f"Yield prediction failed for field {field['id']}: {e}")
            return None

    async def save_updated_prediction(self, field: Dict[str, Any], prediction: Dict[str, Any],
                                    trigger_reason: str, trigger_data: Dict[str, Any]):
        """Save updated prediction to database"""
        try:
            prediction_data = {
                'farmer_id': field['farmer_id'],
                'field_id': field['id'],
                'crop_type': field['crop_type'],
                'sowing_date': field['sowing_date'],
                'predicted_yield_quintal_ha': prediction['prediction']['expected_yield_quintal_ha'],
                'confidence_interval': prediction['prediction'].get('confidence_interval'),
                'confidence_level': prediction['prediction'].get('accuracy_level', 'medium'),
                'growth_stage': prediction['prediction'].get('growth_stage'),
                'trigger_reason': trigger_reason,
                'trigger_metadata': trigger_data,
                'insights': prediction.get('insights'),
                'recommendations': prediction.get('insights', {}).get('optimization_opportunities', []),
                'risk_assessment': prediction.get('insights', {}).get('risk_assessment')
            }

            prediction_id = self.db.save_prediction(prediction_data)

            if prediction_id:
                logger.info(f"Saved updated prediction {prediction_id} for field {field['id']} (trigger: {trigger_reason})")
                return prediction_id
            else:
                logger.error(f"Failed to save prediction for field {field['id']}")
                return None

        except Exception as e:
            logger.error(f"Failed to save updated prediction for field {field['id']}: {e}")
            return None

    async def send_weather_alert(self, field: Dict[str, Any], weather_change: Dict[str, Any],
                               prediction: Dict[str, Any]):
        """Send weather change alert to farmer"""
        try:
            # Create alert message
            temp_change = weather_change['temperature_change']
            rainfall_change = weather_change['rainfall_change']

            if temp_change > 5:
                alert_type = 'extreme_heat' if weather_change['current_weather']['temperature_c'] > 35 else 'temperature_change'
                title = "Weather Alert: Temperature Change"
                message = f"Temperature has changed by {temp_change:.1f}°C. Updated yield prediction: {prediction['prediction']['expected_yield_quintal_ha']:.1f} q/ha"
            elif rainfall_change > 20:
                alert_type = 'heavy_rain' if rainfall_change > 0 else 'drought_conditions'
                title = "Weather Alert: Rainfall Change"
                message = f"Rainfall has changed by {rainfall_change:.1f}mm. Updated yield prediction: {prediction['prediction']['expected_yield_quintal_ha']:.1f} q/ha"
            else:
                return  # No alert needed

            # Save alert to database
            alert_id = self.db.create_alert(
                farmer_id=field['farmer_id'],
                alert_type=alert_type,
                severity='medium',
                title=title,
                message=message,
                field_id=field['id']
            )

            if alert_id:
                logger.info(f"Created weather alert {alert_id} for farmer {field['farmer_id']}")

        except Exception as e:
            logger.error(f"Failed to send weather alert for field {field['id']}: {e}")

    async def send_satellite_alert(self, field: Dict[str, Any], vegetation_change: Dict[str, Any],
                                 prediction: Dict[str, Any]):
        """Send satellite/vegetation change alert to farmer"""
        try:
            change_percent = vegetation_change['change_percent']

            if change_percent < -15:
                alert_type = 'vegetation_stress'
                title = "Field Alert: Vegetation Stress Detected"
                message = f"Vegetation health decreased by {abs(change_percent):.1f}%. Updated yield prediction: {prediction['prediction']['expected_yield_quintal_ha']:.1f} q/ha"
            elif change_percent > 15:
                alert_type = 'vegetation_growth'
                title = "Field Update: Healthy Vegetation Growth"
                message = f"Vegetation health improved by {change_percent:.1f}%. Updated yield prediction: {prediction['prediction']['expected_yield_quintal_ha']:.1f} q/ha"
            else:
                return  # No alert needed

            # Save alert to database
            alert_id = self.db.create_alert(
                farmer_id=field['farmer_id'],
                alert_type=alert_type,
                severity='low' if change_percent > 0 else 'high',
                title=title,
                message=message,
                field_id=field['id']
            )

            if alert_id:
                logger.info(f"Created satellite alert {alert_id} for farmer {field['farmer_id']}")

        except Exception as e:
            logger.error(f"Failed to send satellite alert for field {field['id']}: {e}")

    async def get_current_weather(self, lat: float, lng: float) -> Dict[str, Any]:
        """Get current weather for location"""
        try:
            weather_data = self.predictor._fetch_weather_data(lat, lng, "2024-01-01")
            return weather_data.get('current', {})
        except Exception as e:
            logger.error(f"Failed to get current weather: {e}")
            return {}

    async def get_baseline_weather(self, field: Dict[str, Any]) -> Dict[str, Any]:
        """Get baseline weather for field (from historical data or averages)"""
        try:
            # For now, return seasonal averages
            # In production, this would come from historical weather data
            month = datetime.now().month

            # Punjab seasonal weather baselines
            baselines = {
                11: {'temperature_c': 22, 'rainfall_mm': 15},  # November
                12: {'temperature_c': 18, 'rainfall_mm': 20},  # December
                1: {'temperature_c': 16, 'rainfall_mm': 25},   # January
                2: {'temperature_c': 20, 'rainfall_mm': 30},   # February
                3: {'temperature_c': 25, 'rainfall_mm': 15},   # March
                4: {'temperature_c': 32, 'rainfall_mm': 10},   # April
            }

            return baselines.get(month, {'temperature_c': 25, 'rainfall_mm': 20})

        except Exception as e:
            logger.error(f"Failed to get baseline weather: {e}")
            return {'temperature_c': 25, 'rainfall_mm': 20}

    async def get_system_status(self) -> Dict[str, Any]:
        """Get real-time update system status"""
        return {
            'scheduler_running': self.scheduler.running,
            'active_jobs': len(self.scheduler.get_jobs()),
            'weather_monitors_active': len(self.last_weather_check),
            'satellite_updates_active': len(self.last_satellite_update),
            'last_weather_check': self.last_weather_check,
            'last_satellite_update': self.last_satellite_update
        }

# Global real-time update manager instance
realtime_manager = RealTimeUpdateManager()

# Convenience functions for external use
async def start_realtime_updates():
    """Start the real-time update system"""
    await realtime_manager.start_scheduler()

async def stop_realtime_updates():
    """Stop the real-time update system"""
    await realtime_manager.stop_scheduler()

async def get_realtime_status():
    """Get real-time system status"""
    return await realtime_manager.get_system_status()
