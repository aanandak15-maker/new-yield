"""
IoT Sensor Integration System for India Agricultural Intelligence Platform
Real-time field monitoring with smart sensor data processing and alerts
"""

import asyncio
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import json
import logging
from dataclasses import dataclass, asdict
import numpy as np

from india_agri_platform.database.manager import db_manager
from india_agri_platform.core.streamlined_predictor import streamlined_predictor
from india_agri_platform.core.realtime_updates import realtime_manager

logger = logging.getLogger(__name__)

@dataclass
class SensorReading:
    """Standardized sensor reading data structure"""
    sensor_id: str
    sensor_type: str
    field_id: str
    farmer_id: str
    timestamp: datetime
    value: float
    unit: str
    metadata: Dict[str, Any] = None

    def to_dict(self):
        return asdict(self)

@dataclass
class SensorThreshold:
    """Sensor threshold configuration for alerts"""
    sensor_type: str
    parameter: str
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    critical_min: Optional[float] = None
    critical_max: Optional[float] = None
    tolerance_percent: float = 10.0

class IoTSensorManager:
    """Manages IoT sensor integration, data processing, and intelligent alerts"""

    def __init__(self):
        self.predictor = streamlined_predictor
        self.db = db_manager
        self.realtime_manager = realtime_manager

        # Sensor thresholds for different crops and parameters
        self.thresholds = self._load_sensor_thresholds()

        # Active sensor tracking
        self.active_sensors = {}
        self.last_sensor_reading = {}

        logger.info("IoT Sensor Manager initialized")

    def _load_sensor_thresholds(self) -> Dict[str, Dict[str, SensorThreshold]]:
        """Load sensor thresholds for different crops and parameters"""
        return {
            'wheat': {
                'soil_moisture_percent': SensorThreshold(
                    sensor_type='soil_moisture',
                    parameter='soil_moisture_percent',
                    min_value=20.0,     # Minimum optimal moisture
                    max_value=60.0,     # Maximum optimal moisture
                    critical_min=10.0,  # Critical low (wilting point)
                    critical_max=80.0   # Critical high (waterlogging)
                ),
                'soil_temperature_c': SensorThreshold(
                    sensor_type='soil_temperature',
                    parameter='soil_temperature_c',
                    min_value=10.0,
                    max_value=35.0,
                    critical_min=5.0,
                    critical_max=45.0
                ),
                'soil_ph': SensorThreshold(
                    sensor_type='soil_ph',
                    parameter='soil_ph',
                    min_value=6.0,
                    max_value=8.0,
                    critical_min=5.0,
                    critical_max=9.0
                )
            },
            'rice': {
                'soil_moisture_percent': SensorThreshold(
                    sensor_type='soil_moisture',
                    parameter='soil_moisture_percent',
                    min_value=50.0,     # Rice needs more water
                    max_value=90.0,
                    critical_min=30.0,
                    critical_max=95.0
                )
            }
        }

    async def register_sensor(self, sensor_data: Dict[str, Any]) -> Dict[str, Any]:
        """Register a new IoT sensor with the platform"""
        try:
            required_fields = ['sensor_id', 'sensor_type', 'field_id', 'farmer_id', 'location_lat', 'location_lng']
            for field in required_fields:
                if field not in sensor_data:
                    raise ValueError(f"Missing required field: {field}")

            sensor_id = sensor_data['sensor_id']

            # Check if sensor already exists
            if await self._sensor_exists(sensor_id):
                return {
                    'success': False,
                    'message': f'Sensor {sensor_id} already registered',
                    'sensor_id': sensor_id
                }

            # Save sensor to database
            sensor_record = {
                'sensor_id': sensor_id,
                'sensor_type': sensor_data['sensor_type'],
                'field_id': sensor_data['field_id'],
                'farmer_id': sensor_data['farmer_id'],
                'location_lat': sensor_data['location_lat'],
                'location_lng': sensor_data['location_lng'],
                'installation_date': datetime.utcnow(),
                'status': 'active',
                'last_reading': None,
                'battery_level': sensor_data.get('battery_level', 100),
                'firmware_version': sensor_data.get('firmware_version', '1.0.0'),
                'configuration': sensor_data.get('configuration', {})
            }

            success = self.db.register_sensor(sensor_record)

            if success:
                # Update active sensors cache
                self.active_sensors[sensor_id] = sensor_record

                logger.info(f"Successfully registered sensor {sensor_id} for field {sensor_data['field_id']}")

                return {
                    'success': True,
                    'message': f'Sensor {sensor_id} registered successfully',
                    'sensor_id': sensor_id,
                    'field_id': sensor_data['field_id']
                }
            else:
                raise Exception("Database save failed")

        except Exception as e:
            logger.error(f"Failed to register sensor: {e}")
            return {
                'success': False,
                'message': f'Registration failed: {str(e)}',
                'sensor_id': sensor_data.get('sensor_id')
            }

    async def process_sensor_data(self, sensor_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process incoming sensor data and trigger intelligent responses"""
        try:
            # Validate sensor data
            sensor_reading = await self._validate_sensor_data(sensor_data)

            if not sensor_reading:
                return {
                    'success': False,
                    'message': 'Invalid sensor data'
                }

            # Save reading to database
            reading_id = await self._save_sensor_reading(sensor_reading)

            if not reading_id:
                return {
                    'success': False,
                    'message': 'Failed to save sensor reading'
                }

            # Update sensor status
            await self._update_sensor_status(sensor_reading)

            # Check for alerts
            alerts = await self._check_sensor_alerts(sensor_reading)

            # Update field conditions and predictions
            field_updates = await self._update_field_conditions(sensor_reading)

            # Process alerts and field updates
            await self._process_alerts_and_updates(sensor_reading, alerts, field_updates)

            logger.info(f"Processed sensor data for {sensor_reading.sensor_id}: {sensor_reading.value} {sensor_reading.unit}")

            return {
                'success': True,
                'message': 'Sensor data processed successfully',
                'sensor_id': sensor_reading.sensor_id,
                'reading_id': reading_id,
                'alerts_triggered': len(alerts),
                'field_updated': field_updates['updated'],
                'recommendations': field_updates.get('recommendations', [])
            }

        except Exception as e:
            logger.error(f"Failed to process sensor data: {e}")
            return {
                'success': False,
                'message': f'Processing failed: {str(e)}'
            }

    async def _validate_sensor_data(self, data: Dict[str, Any]) -> Optional[SensorReading]:
        """Validate incoming sensor data and create SensorReading object"""
        try:
            required_fields = ['sensor_id', 'sensor_type', 'value', 'unit']

            for field in required_fields:
                if field not in data:
                    logger.warning(f"Missing required field: {field}")
                    return None

            sensor_id = data['sensor_id']

            # Verify sensor exists and is active
            sensor_info = await self._get_sensor_info(sensor_id)
            if not sensor_info or sensor_info.get('status') != 'active':
                logger.warning(f"Sensor {sensor_id} not found or inactive")
                return None

            # Create sensor reading
            reading = SensorReading(
                sensor_id=sensor_id,
                sensor_type=data['sensor_type'],
                field_id=sensor_info['field_id'],
                farmer_id=sensor_info['farmer_id'],
                timestamp=datetime.utcnow(),
                value=float(data['value']),
                unit=data['unit'],
                metadata=data.get('metadata', {})
            )

            # Basic data quality checks
            if not self._is_sensor_value_reasonable(reading):
                logger.warning(f"Unreasonable sensor value: {reading.value} {reading.unit} for {reading.sensor_type}")
                return None

            return reading

        except Exception as e:
            logger.error(f"Data validation failed: {e}")
            return None

    def _is_sensor_value_reasonable(self, reading: SensorReading) -> bool:
        """Check if sensor reading values are within reasonable ranges"""
        reasonable_ranges = {
            'soil_moisture_percent': (0, 100),
            'soil_temperature_c': (-10, 60),
            'air_temperature_c': (-20, 55),
            'humidity_percent': (0, 100),
            'soil_ph': (0, 14),
            'soil_ec_dsm': (0, 10),  # Electrical conductivity
            'rainfall_mm': (0, 500),  # Per hour
        }

        range_limits = reasonable_ranges.get(reading.sensor_type)
        if not range_limits:
            return True  # Unknown sensor type, allow

        min_val, max_val = range_limits
        return min_val <= reading.value <= max_val

    async def _save_sensor_reading(self, reading: SensorReading) -> Optional[str]:
        """Save sensor reading to database"""
        try:
            reading_data = {
                'sensor_id': reading.sensor_id,
                'sensor_type': reading.sensor_type,
                'field_id': reading.field_id,
                'farmer_id': reading.farmer_id,
                'timestamp': reading.timestamp,
                'value': reading.value,
                'unit': reading.unit,
                'metadata': reading.metadata or {}
            }

            reading_id = self.db.save_sensor_reading(reading_data)

            if reading_id:
                # Update cache
                self.last_sensor_reading[reading.sensor_id] = reading
                return reading_id

            return None

        except Exception as e:
            logger.error(f"Failed to save sensor reading: {e}")
            return None

    async def _update_sensor_status(self, reading: SensorReading):
        """Update sensor last reading timestamp and status"""
        try:
            update_data = {
                'sensor_id': reading.sensor_id,
                'last_reading': reading.timestamp,
                'status': 'active'
            }

            # Check battery level if provided in metadata
            if reading.metadata and 'battery_level' in reading.metadata:
                battery = reading.metadata['battery_level']
                if battery < 10:
                    update_data['status'] = 'low_battery'
                elif battery < 20:
                    update_data['battery_level'] = battery

            self.db.update_sensor_status(update_data)

        except Exception as e:
            logger.error(f"Failed to update sensor status: {e}")

    async def _check_sensor_alerts(self, reading: SensorReading) -> List[Dict[str, Any]]:
        """Check sensor readings against thresholds and generate alerts"""
        alerts = []

        try:
            # Get crop type for field
            field_info = await self._get_field_info(reading.field_id)
            crop_type = field_info.get('crop_type', 'unknown') if field_info else 'unknown'

            # Get thresholds for crop and sensor type
            thresholds = self.thresholds.get(crop_type, {}).get(reading.sensor_type)

            if not thresholds:
                return alerts  # No thresholds defined

            # Check thresholds
            value = reading.value
            threshold = thresholds

            alert_info = self._evaluate_threshold(value, threshold, reading)

            if alert_info:
                alert = {
                    'sensor_id': reading.sensor_id,
                    'field_id': reading.field_id,
                    'farmer_id': reading.farmer_id,
                    'alert_type': alert_info['type'],
                    'severity': alert_info['severity'],
                    'message': alert_info['message'],
                    'current_value': value,
                    'threshold_info': alert_info['threshold_info'],
                    'recommendation': alert_info['recommendation'],
                    'sensor_type': reading.sensor_type,
                    'timestamp': reading.timestamp
                }
                alerts.append(alert)

            return alerts

        except Exception as e:
            logger.error(f"Alert checking failed: {e}")
            return []

    def _evaluate_threshold(self, value: float, threshold: SensorThreshold,
                          reading: SensorReading) -> Optional[Dict[str, Any]]:
        """Evaluate sensor value against threshold and return alert info"""
        info = {}

        # Critical thresholds (highest priority)
        if threshold.critical_min is not None and value <= threshold.critical_min:
            info = {
                'type': f'{reading.sensor_type}_critical_low',
                'severity': 'critical',
                'message': f"CRITICAL: {reading.sensor_type.replace('_', ' ').title()} critically low at {value}",
                'threshold_info': f"Critical minimum: {threshold.critical_min}",
                'recommendation': self._get_recommendation('critical_low', reading.sensor_type)
            }
        elif threshold.critical_max is not None and value >= threshold.critical_max:
            info = {
                'type': f'{reading.sensor_type}_critical_high',
                'severity': 'critical',
                'message': f"CRITICAL: {reading.sensor_type.replace('_', ' ').title()} critically high at {value}",
                'threshold_info': f"Critical maximum: {threshold.critical_max}",
                'recommendation': self._get_recommendation('critical_high', reading.sensor_type)
            }
        # Warning thresholds
        elif threshold.min_value is not None and value < threshold.min_value:
            info = {
                'type': f'{reading.sensor_type}_low',
                'severity': 'warning',
                'message': f"WARNING: {reading.sensor_type.replace('_', ' ').title()} below optimal at {value}",
                'threshold_info': f"Optimal minimum: {threshold.min_value}",
                'recommendation': self._get_recommendation('low', reading.sensor_type)
            }
        elif threshold.max_value is not None and value > threshold.max_value:
            info = {
                'type': f'{reading.sensor_type}_high',
                'severity': 'warning',
                'message': f"WARNING: {reading.sensor_type.replace('_', ' ').title()} above optimal at {value}",
                'threshold_info': f"Optimal maximum: {threshold.max_value}",
                'recommendation': self._get_recommendation('high', reading.sensor_type)
            }

        return info if info else None

    def _get_recommendation(self, alert_type: str, sensor_type: str) -> str:
        """Generate actionable recommendations based on alert type and sensor"""
        recommendations = {
            ('critical_low', 'soil_moisture_percent'): "URGENT: Irrigate immediately to prevent crop stress. Apply 25-30mm water.",
            ('low', 'soil_moisture_percent'): "Irrigate soon. Soil moisture should be maintained between 20-60%.",
            ('high', 'soil_moisture_percent'): "Reduce irrigation frequency. High moisture increases disease risk.",
            ('critical_high', 'soil_moisture_percent'): "CRITICAL: Stop irrigation immediately. Risk of waterlogging and root rot.",

            ('critical_low', 'soil_temperature_c'): "Cover crops or use mulch to retain soil warmth. Consider row covers.",
            ('low', 'soil_temperature_c'): "Monitor closely. Cold stress can delay germination and growth.",
            ('high', 'soil_temperature_c'): "Provide shade or mulch. High temperatures can stress crops.",
            ('critical_high', 'soil_temperature_c'): "URGENT: Implement cooling measures. Risk of heat stress damage.",
        }

        return recommendations.get((alert_type, sensor_type), f"Monitor {sensor_type.replace('_', ' ')} closely and consult agricultural expert.")

    async def _update_field_conditions(self, reading: SensorReading) -> Dict[str, Any]:
        """Update field conditions based on sensor data and generate recommendations"""
        try:
            result = {
                'updated': False,
                'recommendations': [],
                'yield_impact': None
            }

            # Get recent sensor history (last 24 hours)
            sensor_history = await self._get_sensor_history(reading.sensor_id, hours=24)

            if len(sensor_history) < 2:
                return result  # Not enough history

            # Analyze trends and patterns
            analysis = self._analyze_sensor_trends(sensor_history, reading)

            if analysis['significant_change']:
                # Update field status
                field_update = {
                    'field_id': reading.field_id,
                    'last_sensor_update': reading.timestamp,
                    'current_conditions': {
                        reading.sensor_type: {
                            'value': reading.value,
                            'trend': analysis['trend'],
                            'status': analysis['status']
                        }
                    }
                }

                # Save field conditions
                success = self.db.update_field_conditions(field_update)
                result['updated'] = success

                # Generate recommendations
                result['recommendations'] = self._generate_field_recommendations(analysis, reading)

                # Estimate yield impact
                result['yield_impact'] = self._estimate_yield_impact(analysis, reading)

            return result

        except Exception as e:
            logger.error(f"Field condition update failed: {e}")
            return {'updated': False, 'recommendations': [], 'yield_impact': None}

    def _analyze_sensor_trends(self, history: List[Dict], current_reading: SensorReading) -> Dict[str, Any]:
        """Analyze sensor trends to detect significant changes"""
        try:
            values = [h['value'] for h in history] + [current_reading.value]
            times = [h['timestamp'] for h in history] + [current_reading.timestamp]

            if len(values) < 3:
                return {'significant_change': False, 'trend': 'stable', 'status': 'normal'}

            # Calculate simple trend (recent vs historical average)
            recent_avg = np.mean(values[-3:])  # Last 3 readings
            historical_avg = np.mean(values[:-3]) if len(values) > 3 else recent_avg

            if historical_avg == 0:
                change_percent = 0
            else:
                change_percent = ((recent_avg - historical_avg) / historical_avg) * 100

            # Determine significance thresholds
            threshold = self._get_change_threshold(current_reading.sensor_type)

            significant_change = abs(change_percent) > threshold

            # Determine trend direction
            if change_percent > threshold:
                trend = 'increasing'
                status = 'high' if recent_avg > self._get_optimal_range(current_reading)[1] else 'rising'
            elif change_percent < -threshold:
                trend = 'decreasing'
                status = 'low' if recent_avg < self._get_optimal_range(current_reading)[0] else 'falling'
            else:
                trend = 'stable'
                status = 'normal'

            return {
                'significant_change': significant_change,
                'trend': trend,
                'status': status,
                'change_percent': change_percent,
                'recent_avg': recent_avg,
                'historical_avg': historical_avg
            }

        except Exception as e:
            logger.error(f"Trend analysis failed: {e}")
            return {'significant_change': False, 'trend': 'stable', 'status': 'normal'}

    def _get_change_threshold(self, sensor_type: str) -> float:
        """Get significance threshold for sensor type changes"""
        thresholds = {
            'soil_moisture_percent': 15.0,  # 15% change in moisture
            'soil_temperature_c': 5.0,     # 5°C change in temperature
            'soil_ph': 0.5,                # 0.5 pH unit change
            'air_temperature_c': 3.0,      # 3°C change
            'humidity_percent': 10.0,      # 10% humidity change
        }
        return thresholds.get(sensor_type, 10.0)  # Default 10%

    def _get_optimal_range(self, reading: SensorReading) -> Tuple[float, float]:
        """Get optimal range for sensor type"""
        # This could be made dynamic based on crop type and growth stage
        ranges = {
            'soil_moisture_percent': (20, 60),
            'soil_temperature_c': (15, 30),
            'soil_ph': (6.0, 8.0),
        }
        return ranges.get(reading.sensor_type, (0, 100))

    def _generate_field_recommendations(self, analysis: Dict, reading: SensorReading) -> List[str]:
        """Generate actionable recommendations based on sensor analysis"""
        recommendations = []

        if analysis['status'] == 'high':
            if reading.sensor_type == 'soil_moisture_percent':
                recommendations.append("Reduce irrigation frequency to prevent waterlogging")
                recommendations.append("Improve drainage if water accumulation is observed")
            elif reading.sensor_type == 'soil_temperature_c':
                recommendations.append("Consider mulching to cool soil temperature")

        elif analysis['status'] == 'low':
            if reading.sensor_type == 'soil_moisture_percent':
                recommendations.append("Increase irrigation to maintain optimal soil moisture")
                recommendations.append("Consider drip irrigation for better water efficiency")
            elif reading.sensor_type == 'soil_temperature_c':
                recommendations.append("Apply mulch to retain soil warmth")

        return recommendations

    def _estimate_yield_impact(self, analysis: Dict, reading: SensorReading) -> Optional[Dict[str, Any]]:
        """Estimate potential yield impact based on sensor trends"""
        try:
            # Simple impact estimation (could be made more sophisticated)
            impact_factors = {
                'soil_moisture_percent': {
                    'high': -0.08,  # 8% yield reduction for high moisture
                    'low': -0.15,   # 15% yield reduction for low moisture
                },
                'soil_temperature_c': {
                    'high': -0.12,
                    'low': -0.06,
                }
            }

            impacts = impact_factors.get(reading.sensor_type, {})
            impact_pct = impacts.get(analysis['status'], 0)

            if impact_pct != 0:
                return {
                    'estimated_change_percent': impact_pct,
                    'severity': 'high' if abs(impact_pct) > 0.1 else 'moderate',
                    'description': f"Estimated {abs(impact_pct)*100:.1f}% yield {'increase' if impact_pct > 0 else 'reduction'}"
                }

            return None

        except Exception as e:
            logger.error(f"Yield impact estimation failed: {e}")
            return None

    async def _process_alerts_and_updates(self, reading: SensorReading,
                                        alerts: List[Dict], field_updates: Dict):
        """Process alerts and field updates, including triggering predictions"""
        try:
            # Save alerts to database
            for alert in alerts:
                alert_id = self.db.create_alert(alert)
                if alert_id:
                    logger.info(f"Created sensor alert {alert_id} for farmer {alert['farmer_id']}")

            # If field conditions changed significantly, update predictions
            if field_updates.get('updated', False) and field_updates.get('yield_impact'):
                await self._trigger_prediction_update(reading.field_id, field_updates)

        except Exception as e:
            logger.error(f"Alert and update processing failed: {e}")

    async def _trigger_prediction_update(self, field_id: str, field_updates: Dict):
        """Trigger prediction update based on sensor changes"""
        try:
            # Get field information
            field_info = await self._get_field_info(field_id)
            if not field_info:
                return

            # Check if conditions warrant prediction update
            yield_impact = field_updates.get('yield_impact', {})
            if yield_impact and abs(yield_impact.get('estimated_change_percent', 0)) > 0.05:  # 5% change threshold

                # Generate new prediction
                prediction = await self.predictor.predict_yield_streamlined(
                    crop_name=field_info['crop_type'],
                    sowing_date=field_info['sowing_date'],
                    latitude=field_info['centroid_lat'],
                    longitude=field_info['centroid_lng']
                )

                if prediction and 'prediction' in prediction:
                    # Save updated prediction
                    prediction_data = {
                        'farmer_id': field_info['farmer_id'],
                        'field_id': field_id,
                        'crop_type': field_info['crop_type'],
                        'sowing_date': field_info['sowing_date'],
                        'predicted_yield_quintal_ha': prediction['prediction']['expected_yield_quintal_ha'],
                        'confidence_interval': prediction['prediction'].get('confidence_interval'),
                        'confidence_level': prediction['prediction'].get('accuracy_level', 'high'),
                        'trigger_reason': 'sensor_data_change',
                        'trigger_metadata': {
                            'sensor_change': field_updates,
                            'yield_impact': yield_impact
                        },
                        'insights': prediction.get('insights', {}),
                    }

                    prediction_id = self.db.save_prediction(prediction_data)

                    if prediction_id:
                        logger.info(f"Sensor-triggered prediction update saved for field {field_id}")

                        # Send notification to farmer
                        await self._send_sensor_prediction_alert(field_info, prediction, yield_impact)

        except Exception as e:
            logger.error(f"Prediction update trigger failed: {e}")

    async def _send_sensor_prediction_alert(self, field_info: Dict, prediction: Dict, yield_impact: Dict):
        """Send sensor-triggered prediction update alert"""
        try:
            yield_change = yield_impact.get('estimated_change_percent', 0)
            direction = "increase" if yield_change > 0 else "reduction"

            alert = {
                'farmer_id': field_info['farmer_id'],
                'alert_type': 'prediction_updated',
                'severity': 'medium',
                'title': "Prediction Updated - Field Conditions Changed",
                'message': f"Field conditions monitoring detected significant changes. Updated yield prediction: {prediction['prediction']['expected_yield_quintal_ha']:.1f} q/ha ({abs(yield_change)*100:.1f}% {direction} from previous estimate)",
                'field_id': field_info['id']
            }

            alert_id = self.db.create_alert(alert)
            if alert_id:
                logger.info(f"Sent sensor prediction alert to farmer {field_info['farmer_id']}")

        except Exception as e:
            logger.error(f"Sensor prediction alert failed: {e}")

    # Database helper methods
    async def _sensor_exists(self, sensor_id: str) -> bool:
        """Check if sensor exists in database"""
        return self.db.get_sensor_by_id(sensor_id) is not None

    async def _get_sensor_info(self, sensor_id: str) -> Optional[Dict[str, Any]]:
        """Get sensor information from database"""
        return self.db.get_sensor_by_id(sensor_id)

    async def _get_field_info(self, field_id: str) -> Optional[Dict[str, Any]]:
        """Get field information from database"""
        return self.db.get_field_by_id(field_id)

    async def _get_sensor_history(self, sensor_id: str, hours: int = 24) -> List[Dict[str, Any]]:
        """Get sensor reading history"""
        return self.db.get_sensor_readings(sensor_id, hours=hours)

    async def get_sensor_dashboard(self, farmer_id: str) -> Dict[str, Any]:
        """Get comprehensive sensor dashboard for farmer"""
        try:
            # Get all farmer's sensors
            sensors = self.db.get_farmer_sensors(farmer_id)

            # Get recent readings for each sensor
            sensor_data = {}
            for sensor in sensors:
                readings = await self._get_sensor_history(sensor['sensor_id'], hours=168)  # Last 7 days
                if readings:
                    sensor_data[sensor['sensor_id']] = {
                        'info': sensor,
                        'readings': readings,
                        'latest': readings[-1] if readings else None,
                        'trend_analysis': self._analyze_sensor_trends(readings, None) if len(readings) > 1 else None
                    }

            return {
                'total_sensors': len(sensors),
                'active_sensors': len([s for s in sensors if s['status'] == 'active']),
                'sensor_data': sensor_data,
                'alerts_summary': self.db.get_farmer_sensor_alerts(farmer_id, days=7)
            }

        except Exception as e:
            logger.error(f"Dashboard generation failed: {e}")
            return {'error': str(e)}

# Global IoT sensor manager instance
iot_sensor_manager = IoTSensorManager()

# Convenience functions for external use
async def register_iot_sensor(sensor_data: Dict[str, Any]) -> Dict[str, Any]:
    """Register a new IoT sensor"""
    return await iot_sensor_manager.register_sensor(sensor_data)

async def process_iot_sensor_data(sensor_data: Dict[str, Any]) -> Dict[str, Any]:
    """Process IoT sensor data"""
    return await iot_sensor_manager.process_sensor_data(sensor_data)

async def get_sensor_dashboard(farmer_id: str) -> Dict[str, Any]:
    """Get farmer's sensor dashboard"""
    return await iot_sensor_manager.get_sensor_dashboard(farmer_id)
