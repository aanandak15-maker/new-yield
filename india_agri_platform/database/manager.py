"""
Database Manager for India Agricultural Intelligence Platform
Handles all database operations with PostgreSQL + PostGIS
"""

import os
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text, func, and_, or_
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError
from geoalchemy2 import WKTElement
from geoalchemy2.functions import ST_Area, ST_Centroid, ST_X, ST_Y
import logging
import json

from .models import (
    Base, Farmer, Field, Prediction, SatelliteData, WeatherData,
    Sensor, SensorReading, Alert, SystemLog, APICache
)

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Comprehensive database manager for agricultural data"""

    def __init__(self, connection_string: Optional[str] = None):
        """Initialize database connection"""
        if connection_string is None:
            # Default to environment variable or local PostgreSQL
            connection_string = os.getenv(
                'DATABASE_URL',
                'postgresql://postgres:password@localhost:5432/agri_platform'
            )

        try:
            self.engine = create_engine(
                connection_string,
                pool_pre_ping=True,
                pool_recycle=300,
                echo=False  # Set to True for debugging
            )

            # Create session factory
            self.SessionLocal = sessionmaker(
                autocommit=False,
                autoflush=False,
                bind=self.engine
            )

            logger.info("Database connection established successfully")

        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            raise

    def create_tables(self) -> bool:
        """Create all database tables"""
        try:
            Base.metadata.create_all(bind=self.engine)
            logger.info("All database tables created successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to create tables: {e}")
            return False

    def get_session(self) -> Session:
        """Get database session"""
        return self.SessionLocal()

    def close_session(self, session: Session):
        """Close database session"""
        try:
            session.close()
        except Exception as e:
            logger.error(f"Error closing session: {e}")

    # Farmer Operations
    def create_farmer(self, name: str, phone: Optional[str] = None,
                     email: Optional[str] = None, location_lat: Optional[float] = None,
                     location_lng: Optional[float] = None) -> Optional[int]:
        """Create new farmer"""
        session = self.get_session()
        try:
            farmer = Farmer(
                name=name,
                phone=phone,
                email=email,
                location_lat=location_lat,
                location_lng=location_lng
            )
            session.add(farmer)
            session.commit()
            logger.info(f"Created farmer: {farmer.id}")
            return farmer.id
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Failed to create farmer: {e}")
            return None
        finally:
            self.close_session(session)

    def get_farmer(self, farmer_id: int) -> Optional[Dict[str, Any]]:
        """Get farmer by ID"""
        session = self.get_session()
        try:
            farmer = session.query(Farmer).filter(Farmer.id == farmer_id).first()
            if farmer:
                return {
                    'id': farmer.id,
                    'name': farmer.name,
                    'phone': farmer.phone,
                    'email': farmer.email,
                    'location_lat': farmer.location_lat,
                    'location_lng': farmer.location_lng,
                    'created_at': farmer.created_at.isoformat() if farmer.created_at else None
                }
            return None
        except SQLAlchemyError as e:
            logger.error(f"Failed to get farmer {farmer_id}: {e}")
            return None
        finally:
            self.close_session(session)

    # Field Operations
    def create_field(self, farmer_id: int, name: str, centroid_lat: float,
                    centroid_lng: float, area_hectares: Optional[float] = None,
                    boundary_wkt: Optional[str] = None) -> Optional[int]:
        """Create new field"""
        session = self.get_session()
        try:
            field = Field(
                farmer_id=farmer_id,
                name=name,
                centroid_lat=centroid_lat,
                centroid_lng=centroid_lng,
                area_hectares=area_hectares
            )

            if boundary_wkt:
                field.boundary = WKTElement(boundary_wkt, srid=4326)

            session.add(field)
            session.commit()
            logger.info(f"Created field: {field.id}")
            return field.id
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Failed to create field: {e}")
            return None
        finally:
            self.close_session(session)

    def get_field(self, field_id: int) -> Optional[Dict[str, Any]]:
        """Get field by ID"""
        session = self.get_session()
        try:
            field = session.query(Field).filter(Field.id == field_id).first()
            if field:
                return {
                    'id': field.id,
                    'farmer_id': field.farmer_id,
                    'name': field.name,
                    'centroid_lat': field.centroid_lat,
                    'centroid_lng': field.centroid_lng,
                    'area_hectares': field.area_hectares,
                    'soil_type': field.soil_type,
                    'irrigation_type': field.irrigation_type,
                    'irrigation_coverage_percent': field.irrigation_coverage_percent,
                    'created_at': field.created_at.isoformat() if field.created_at else None
                }
            return None
        except SQLAlchemyError as e:
            logger.error(f"Failed to get field {field_id}: {e}")
            return None
        finally:
            self.close_session(session)

    def get_farmer_fields(self, farmer_id: int) -> List[Dict[str, Any]]:
        """Get all fields for a farmer"""
        session = self.get_session()
        try:
            fields = session.query(Field).filter(
                and_(Field.farmer_id == farmer_id, Field.is_active == True)
            ).all()

            return [{
                'id': field.id,
                'name': field.name,
                'centroid_lat': field.centroid_lat,
                'centroid_lng': field.centroid_lng,
                'area_hectares': field.area_hectares,
                'soil_type': field.soil_type,
                'irrigation_type': field.irrigation_type,
                'created_at': field.created_at.isoformat() if field.created_at else None
            } for field in fields]
        except SQLAlchemyError as e:
            logger.error(f"Failed to get fields for farmer {farmer_id}: {e}")
            return []
        finally:
            self.close_session(session)

    # Prediction Operations
    def save_prediction(self, prediction_data: Dict[str, Any]) -> Optional[int]:
        """Save prediction result"""
        session = self.get_session()
        try:
            prediction = Prediction(
                farmer_id=prediction_data['farmer_id'],
                field_id=prediction_data['field_id'],
                crop_type=prediction_data['crop_type'],
                variety_name=prediction_data.get('variety_name'),
                sowing_date=datetime.strptime(prediction_data['sowing_date'], '%Y-%m-%d').date(),
                predicted_yield_quintal_ha=prediction_data['predicted_yield_quintal_ha'],
                confidence_interval=prediction_data.get('confidence_interval'),
                confidence_level=prediction_data.get('confidence_level', 'medium'),
                growth_stage=prediction_data.get('growth_stage'),
                days_since_sowing=prediction_data.get('days_since_sowing'),
                estimated_harvest_days=prediction_data.get('estimated_harvest_days'),
                model_version=prediction_data.get('model_version'),
                prediction_method=prediction_data.get('prediction_method', 'streamlined_predictor'),
                weather_data=prediction_data.get('weather_data'),
                satellite_data=prediction_data.get('satellite_data'),
                soil_data=prediction_data.get('soil_data'),
                trigger_reason=prediction_data.get('trigger_reason', 'manual'),
                trigger_metadata=prediction_data.get('trigger_metadata'),
                insights=prediction_data.get('insights'),
                recommendations=prediction_data.get('recommendations'),
                risk_assessment=prediction_data.get('risk_assessment')
            )

            session.add(prediction)
            session.commit()
            logger.info(f"Saved prediction: {prediction.id}")
            return prediction.id
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Failed to save prediction: {e}")
            return None
        finally:
            self.close_session(session)

    def get_prediction(self, prediction_id: int) -> Optional[Dict[str, Any]]:
        """Get prediction by ID"""
        session = self.get_session()
        try:
            prediction = session.query(Prediction).filter(Prediction.id == prediction_id).first()
            if prediction:
                return {
                    'id': prediction.id,
                    'farmer_id': prediction.farmer_id,
                    'field_id': prediction.field_id,
                    'crop_type': prediction.crop_type,
                    'variety_name': prediction.variety_name,
                    'sowing_date': prediction.sowing_date.isoformat() if prediction.sowing_date else None,
                    'predicted_yield_quintal_ha': prediction.predicted_yield_quintal_ha,
                    'confidence_interval': prediction.confidence_interval,
                    'confidence_level': prediction.confidence_level,
                    'growth_stage': prediction.growth_stage,
                    'days_since_sowing': prediction.days_since_sowing,
                    'estimated_harvest_days': prediction.estimated_harvest_days,
                    'insights': prediction.insights,
                    'recommendations': prediction.recommendations,
                    'risk_assessment': prediction.risk_assessment,
                    'created_at': prediction.created_at.isoformat() if prediction.created_at else None
                }
            return None
        except SQLAlchemyError as e:
            logger.error(f"Failed to get prediction {prediction_id}: {e}")
            return None
        finally:
            self.close_session(session)

    def get_field_predictions(self, field_id: int, limit: int = 50) -> List[Dict[str, Any]]:
        """Get prediction history for a field"""
        session = self.get_session()
        try:
            predictions = session.query(Prediction).filter(
                Prediction.field_id == field_id
            ).order_by(Prediction.created_at.desc()).limit(limit).all()

            return [{
                'id': pred.id,
                'crop_type': pred.crop_type,
                'predicted_yield_quintal_ha': pred.predicted_yield_quintal_ha,
                'confidence_interval': pred.confidence_interval,
                'growth_stage': pred.growth_stage,
                'created_at': pred.created_at.isoformat() if pred.created_at else None
            } for pred in predictions]
        except SQLAlchemyError as e:
            logger.error(f"Failed to get predictions for field {field_id}: {e}")
            return []
        finally:
            self.close_session(session)

    # Satellite Data Operations
    def cache_satellite_data(self, satellite_data: Dict[str, Any]) -> bool:
        """Cache satellite data in database"""
        session = self.get_session()
        try:
            # Check if data already exists for this location/date
            existing = session.query(SatelliteData).filter(
                and_(
                    SatelliteData.location_lat == satellite_data['location_lat'],
                    SatelliteData.location_lng == satellite_data['location_lng'],
                    SatelliteData.date == datetime.strptime(satellite_data['date'], '%Y-%m-%d').date()
                )
            ).first()

            if existing:
                # Update existing record
                for key, value in satellite_data.items():
                    if hasattr(existing, key) and key not in ['location_lat', 'location_lng', 'date']:
                        setattr(existing, key, value)
                existing.processing_date = datetime.utcnow()
            else:
                # Create new record
                data_record = SatelliteData(
                    field_id=satellite_data.get('field_id'),
                    location_lat=satellite_data['location_lat'],
                    location_lng=satellite_data['location_lng'],
                    date=datetime.strptime(satellite_data['date'], '%Y-%m-%d').date(),
                    ndvi=satellite_data.get('ndvi'),
                    evi=satellite_data.get('evi'),
                    savi=satellite_data.get('savi'),
                    ndwi=satellite_data.get('ndwi'),
                    land_surface_temp_c=satellite_data.get('land_surface_temp_c'),
                    soil_moisture_percent=satellite_data.get('soil_moisture_percent'),
                    cloud_cover_percent=satellite_data.get('cloud_cover_percent'),
                    data_quality_score=satellite_data.get('data_quality_score'),
                    data_source=satellite_data.get('data_source', 'google_earth_engine'),
                    satellite=satellite_data.get('satellite'),
                    resolution_meters=satellite_data.get('resolution_meters')
                )
                session.add(data_record)

            session.commit()
            return True
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Failed to cache satellite data: {e}")
            return False
        finally:
            self.close_session(session)

    def get_satellite_history(self, field_id: int, days: int = 30) -> List[Dict[str, Any]]:
        """Get satellite data history for a field"""
        session = self.get_session()
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            data = session.query(SatelliteData).filter(
                and_(
                    SatelliteData.field_id == field_id,
                    SatelliteData.created_at >= cutoff_date
                )
            ).order_by(SatelliteData.date).all()

            return [{
                'date': data_point.date.isoformat(),
                'ndvi': data_point.ndvi,
                'soil_moisture_percent': data_point.soil_moisture_percent,
                'land_surface_temp_c': data_point.land_surface_temp_c,
                'data_source': data_point.data_source
            } for data_point in data]
        except SQLAlchemyError as e:
            logger.error(f"Failed to get satellite history for field {field_id}: {e}")
            return []
        finally:
            self.close_session(session)

    # Weather Data Operations
    def save_weather_data(self, weather_data: Dict[str, Any]) -> bool:
        """Save weather data to database"""
        session = self.get_session()
        try:
            # Check for existing record
            existing = session.query(WeatherData).filter(
                and_(
                    WeatherData.location_lat == weather_data['location_lat'],
                    WeatherData.location_lng == weather_data['location_lng'],
                    WeatherData.date == datetime.strptime(weather_data['date'], '%Y-%m-%d').date()
                )
            ).first()

            if existing:
                # Update existing
                for key, value in weather_data.items():
                    if hasattr(existing, key) and key not in ['location_lat', 'location_lng', 'date']:
                        setattr(existing, key, value)
            else:
                # Create new
                data_record = WeatherData(
                    field_id=weather_data.get('field_id'),
                    location_lat=weather_data['location_lat'],
                    location_lng=weather_data['location_lng'],
                    date=datetime.strptime(weather_data['date'], '%Y-%m-%d').date(),
                    temperature_c=weather_data.get('temperature_c'),
                    temp_min_c=weather_data.get('temp_min_c'),
                    temp_max_c=weather_data.get('temp_max_c'),
                    rainfall_mm=weather_data.get('rainfall_mm'),
                    humidity_percent=weather_data.get('humidity_percent'),
                    wind_speed_kmph=weather_data.get('wind_speed_kmph'),
                    weather_main=weather_data.get('weather_main'),
                    weather_description=weather_data.get('weather_description'),
                    data_source=weather_data.get('data_source', 'openweathermap'),
                    is_forecast=weather_data.get('is_forecast', False)
                )
                session.add(data_record)

            session.commit()
            return True
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Failed to save weather data: {e}")
            return False
        finally:
            self.close_session(session)

    # Sensor Operations
    def register_sensor(self, field_id: int, sensor_id: str, name: str,
                       sensor_type: str, location_lat: Optional[float] = None,
                       location_lng: Optional[float] = None) -> Optional[int]:
        """Register a new sensor"""
        session = self.get_session()
        try:
            sensor = Sensor(
                field_id=field_id,
                sensor_id=sensor_id,
                name=name,
                sensor_type=sensor_type,
                location_lat=location_lat,
                location_lng=location_lng
            )
            session.add(sensor)
            session.commit()
            logger.info(f"Registered sensor: {sensor.id}")
            return sensor.id
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Failed to register sensor: {e}")
            return None
        finally:
            self.close_session(session)

    def save_sensor_reading(self, sensor_id: int, value: float, unit: str,
                           timestamp: Optional[datetime] = None) -> bool:
        """Save sensor reading"""
        session = self.get_session()
        try:
            reading = SensorReading(
                sensor_id=sensor_id,
                value=value,
                unit=unit,
                timestamp=timestamp or datetime.utcnow()
            )
            session.add(reading)

            # Update sensor last reading time
            sensor = session.query(Sensor).filter(Sensor.id == sensor_id).first()
            if sensor:
                sensor.last_reading_at = reading.timestamp

            session.commit()
            return True
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Failed to save sensor reading: {e}")
            return False
        finally:
            self.close_session(session)

    # Alert Operations
    def create_alert(self, farmer_id: int, alert_type: str, severity: str,
                    title: str, message: str, field_id: Optional[int] = None,
                    sensor_id: Optional[int] = None) -> Optional[int]:
        """Create a new alert"""
        session = self.get_session()
        try:
            alert = Alert(
                farmer_id=farmer_id,
                field_id=field_id,
                sensor_id=sensor_id,
                alert_type=alert_type,
                severity=severity,
                title=title,
                message=message
            )
            session.add(alert)
            session.commit()
            logger.info(f"Created alert: {alert.id}")
            return alert.id
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Failed to create alert: {e}")
            return None
        finally:
            self.close_session(session)

    # Cache Operations
    def get_cached_data(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get data from cache"""
        session = self.get_session()
        try:
            cache_entry = session.query(APICache).filter(
                and_(
                    APICache.cache_key == cache_key,
                    APICache.expires_at > datetime.utcnow()
                )
            ).first()

            if cache_entry:
                # Update access count
                cache_entry.access_count += 1
                cache_entry.last_accessed = datetime.utcnow()
                session.commit()

                return {
                    'data': cache_entry.data,
                    'data_type': cache_entry.data_type,
                    'created_at': cache_entry.created_at.isoformat(),
                    'expires_at': cache_entry.expires_at.isoformat()
                }
            return None
        except SQLAlchemyError as e:
            logger.error(f"Failed to get cached data: {e}")
            return None
        finally:
            self.close_session(session)

    def set_cached_data(self, cache_key: str, data: Dict[str, Any],
                       data_type: str, ttl_hours: int = 24) -> bool:
        """Store data in cache"""
        session = self.get_session()
        try:
            expires_at = datetime.utcnow() + timedelta(hours=ttl_hours)

            # Check if key exists
            existing = session.query(APICache).filter(APICache.cache_key == cache_key).first()

            if existing:
                # Update existing
                existing.data = data
                existing.expires_at = expires_at
                existing.access_count = 0
                existing.last_accessed = datetime.utcnow()
            else:
                # Create new
                cache_entry = APICache(
                    cache_key=cache_key,
                    data=data,
                    data_type=data_type,
                    expires_at=expires_at
                )
                session.add(cache_entry)

            session.commit()
            return True
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Failed to cache data: {e}")
            return False
        finally:
            self.close_session(session)

    # Analytics and Statistics
    def get_prediction_count(self) -> int:
        """Get total number of predictions"""
        session = self.get_session()
        try:
            return session.query(func.count(Prediction.id)).scalar()
        except SQLAlchemyError as e:
            logger.error(f"Failed to get prediction count: {e}")
            return 0
        finally:
            self.close_session(session)

    def get_active_field_count(self) -> int:
        """Get number of active fields"""
        session = self.get_session()
        try:
            return session.query(func.count(Field.id)).filter(Field.is_active == True).scalar()
        except SQLAlchemyError as e:
            logger.error(f"Failed to get field count: {e}")
            return 0
        finally:
            self.close_session(session)

    def get_satellite_data_count(self) -> int:
        """Get total satellite data points"""
        session = self.get_session()
        try:
            return session.query(func.count(SatelliteData.id)).scalar()
        except SQLAlchemyError as e:
            logger.error(f"Failed to get satellite data count: {e}")
            return 0
        finally:
            self.close_session(session)

    # System Health and Monitoring
    def log_system_event(self, level: str, module: str, message: str,
                        user_id: Optional[int] = None, extra_data: Optional[Dict] = None):
        """Log system event"""
        session = self.get_session()
        try:
            log_entry = SystemLog(
                level=level,
                module=module,
                message=message,
                user_id=user_id,
                extra_data=extra_data
            )
            session.add(log_entry)
            session.commit()
        except SQLAlchemyError as e:
            logger.error(f"Failed to log system event: {e}")
        finally:
            self.close_session(session)

    def health_check(self) -> Dict[str, Any]:
        """Database health check"""
        health_status = {
            'database_connected': False,
            'tables_exist': False,
            'total_predictions': 0,
            'total_fields': 0,
            'total_satellite_data': 0,
            'last_prediction': None,
            'status': 'unhealthy'
        }

        session = self.get_session()
        try:
            # Test connection
            session.execute(text("SELECT 1"))
            health_status['database_connected'] = True

            # Check tables exist
            table_names = self.engine.table_names()
            required_tables = ['farmers', 'fields', 'predictions', 'satellite_data', 'weather_data']
            health_status['tables_exist'] = all(table in table_names for table in required_tables)

            # Get statistics
            health_status['total_predictions'] = session.query(func.count(Prediction.id)).scalar()
            health_status['total_fields'] = session.query(func.count(Field.id)).filter(Field.is_active == True).scalar()
            health_status['total_satellite_data'] = session.query(func.count(SatelliteData.id)).scalar()

            # Get last prediction
            last_pred = session.query(Prediction).order_by(Prediction.created_at.desc()).first()
            if last_pred:
                health_status['last_prediction'] = last_pred.created_at.isoformat()

            health_status['status'] = 'healthy'

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            health_status['error'] = str(e)
        finally:
            self.close_session(session)

        return health_status

# Global database manager instance
db_manager = DatabaseManager()
