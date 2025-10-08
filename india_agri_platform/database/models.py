"""
Database Models for India Agricultural Intelligence Platform
PostgreSQL with PostGIS for spatial data support
"""

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, JSON, Date, Boolean, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from geoalchemy2 import Geometry
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

Base = declarative_base()

class Farmer(Base):
    """Farmer profile and basic information"""
    __tablename__ = 'farmers'

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), nullable=False)
    phone = Column(String(15), unique=True, nullable=True)
    email = Column(String(100), unique=True, nullable=True)
    location_lat = Column(Float, nullable=True)
    location_lng = Column(Float, nullable=True)
    location_address = Column(Text, nullable=True)
    preferred_language = Column(String(10), default='en')
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = Column(Boolean, default=True)

    # Relationships
    fields = relationship("Field", back_populates="farmer")
    predictions = relationship("Prediction", back_populates="farmer")

    def __repr__(self):
        return f"<Farmer(id={self.id}, name='{self.name}', phone='{self.phone}')>"

class Field(Base):
    """Agricultural field with GPS boundaries"""
    __tablename__ = 'fields'

    id = Column(Integer, primary_key=True, autoincrement=True)
    farmer_id = Column(Integer, ForeignKey('farmers.id'), nullable=False)
    name = Column(String(100), nullable=False)
    boundary = Column(Geometry('POLYGON'), nullable=True)  # GPS boundary coordinates
    centroid_lat = Column(Float, nullable=True)
    centroid_lng = Column(Float, nullable=True)
    area_hectares = Column(Float, nullable=True)
    soil_type = Column(String(50), nullable=True)
    irrigation_type = Column(String(50), nullable=True)  # canal, tubewell, rainwater, etc.
    irrigation_coverage_percent = Column(Float, default=80.0)
    crop_rotation_history = Column(JSON, nullable=True)  # Previous crops
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = Column(Boolean, default=True)

    # Relationships
    farmer = relationship("Farmer", back_populates="fields")
    predictions = relationship("Prediction", back_populates="field")
    sensors = relationship("Sensor", back_populates="field")
    satellite_data = relationship("SatelliteData", back_populates="field")

    def __repr__(self):
        return f"<Field(id={self.id}, name='{self.name}', area={self.area_hectares}ha)>"

class Prediction(Base):
    """Yield prediction results and metadata"""
    __tablename__ = 'predictions'

    id = Column(Integer, primary_key=True, autoincrement=True)
    farmer_id = Column(Integer, ForeignKey('farmers.id'), nullable=False)
    field_id = Column(Integer, ForeignKey('fields.id'), nullable=False)
    crop_type = Column(String(50), nullable=False)
    variety_name = Column(String(100), nullable=True)
    sowing_date = Column(Date, nullable=False)
    predicted_yield_quintal_ha = Column(Float, nullable=False)
    confidence_interval = Column(JSON, nullable=True)  # {"low": 40.5, "high": 52.3}
    confidence_level = Column(String(20), default='medium')  # low, medium, high
    growth_stage = Column(String(50), nullable=True)
    days_since_sowing = Column(Integer, nullable=True)
    estimated_harvest_days = Column(Integer, nullable=True)

    # Model metadata
    model_version = Column(String(50), nullable=True)
    prediction_method = Column(String(100), default='streamlined_predictor')
    accuracy_score = Column(Float, nullable=True)

    # Environmental data used
    weather_data = Column(JSON, nullable=True)  # Weather conditions
    satellite_data = Column(JSON, nullable=True)  # Satellite indices
    soil_data = Column(JSON, nullable=True)  # Soil parameters

    # Trigger information
    trigger_reason = Column(String(100), default='manual')  # manual, weather_change, satellite_update, scheduled
    trigger_metadata = Column(JSON, nullable=True)

    # Results and insights
    insights = Column(JSON, nullable=True)  # Agricultural insights
    recommendations = Column(JSON, nullable=True)  # Actionable recommendations
    risk_assessment = Column(JSON, nullable=True)  # Risk factors and levels

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    farmer = relationship("Farmer", back_populates="predictions")
    field = relationship("Field", back_populates="predictions")

    def __repr__(self):
        return f"<Prediction(id={self.id}, field={self.field_id}, yield={self.predicted_yield_quintal_ha}q/ha)>"

class SatelliteData(Base):
    """Satellite imagery and vegetation indices"""
    __tablename__ = 'satellite_data'

    id = Column(Integer, primary_key=True, autoincrement=True)
    field_id = Column(Integer, ForeignKey('fields.id'), nullable=True)
    location_lat = Column(Float, nullable=False)
    location_lng = Column(Float, nullable=False)
    date = Column(Date, nullable=False)

    # Vegetation indices
    ndvi = Column(Float, nullable=True)  # Normalized Difference Vegetation Index
    evi = Column(Float, nullable=True)   # Enhanced Vegetation Index
    savi = Column(Float, nullable=True)  # Soil Adjusted Vegetation Index
    ndwi = Column(Float, nullable=True)  # Normalized Difference Water Index

    # Surface data
    land_surface_temp_c = Column(Float, nullable=True)
    soil_moisture_percent = Column(Float, nullable=True)

    # Data quality
    cloud_cover_percent = Column(Float, nullable=True)
    data_quality_score = Column(Float, nullable=True)  # 0-1 scale

    # Source information
    data_source = Column(String(50), default='google_earth_engine')
    satellite = Column(String(50), nullable=True)  # Landsat, Sentinel, MODIS
    resolution_meters = Column(Float, nullable=True)

    # Processing metadata
    processing_date = Column(DateTime, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    field = relationship("Field", back_populates="satellite_data")

    def __repr__(self):
        return f"<SatelliteData(id={self.id}, date={self.date}, ndvi={self.ndvi})>"

class WeatherData(Base):
    """Weather observations and forecasts"""
    __tablename__ = 'weather_data'

    id = Column(Integer, primary_key=True, autoincrement=True)
    field_id = Column(Integer, ForeignKey('fields.id'), nullable=True)
    location_lat = Column(Float, nullable=False)
    location_lng = Column(Float, nullable=False)
    date = Column(Date, nullable=False)
    observation_time = Column(DateTime, nullable=True)

    # Temperature
    temperature_c = Column(Float, nullable=True)
    temp_min_c = Column(Float, nullable=True)
    temp_max_c = Column(Float, nullable=True)
    feels_like_c = Column(Float, nullable=True)

    # Precipitation
    rainfall_mm = Column(Float, nullable=True)
    snowfall_mm = Column(Float, nullable=True)
    precipitation_probability = Column(Float, nullable=True)

    # Humidity and pressure
    humidity_percent = Column(Float, nullable=True)
    pressure_hpa = Column(Float, nullable=True)
    dew_point_c = Column(Float, nullable=True)

    # Wind
    wind_speed_kmph = Column(Float, nullable=True)
    wind_direction_degrees = Column(Float, nullable=True)
    wind_gust_kmph = Column(Float, nullable=True)

    # Solar radiation
    solar_radiation_mj_m2 = Column(Float, nullable=True)
    uv_index = Column(Float, nullable=True)

    # Weather conditions
    weather_main = Column(String(50), nullable=True)  # Rain, Clear, Clouds
    weather_description = Column(String(100), nullable=True)
    weather_icon = Column(String(10), nullable=True)

    # Forecast data
    is_forecast = Column(Boolean, default=False)
    forecast_horizon_hours = Column(Integer, nullable=True)

    # Source and quality
    data_source = Column(String(50), default='openweathermap')
    data_quality_score = Column(Float, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<WeatherData(id={self.id}, date={self.date}, temp={self.temperature_c}°C, rain={self.rainfall_mm}mm)>"

class Sensor(Base):
    """IoT sensor devices and metadata"""
    __tablename__ = 'sensors'

    id = Column(Integer, primary_key=True, autoincrement=True)
    field_id = Column(Integer, ForeignKey('fields.id'), nullable=False)
    sensor_id = Column(String(100), unique=True, nullable=False)  # Unique hardware ID
    name = Column(String(100), nullable=False)
    sensor_type = Column(String(50), nullable=False)  # soil_moisture, temperature, ph, etc.
    manufacturer = Column(String(100), nullable=True)
    model = Column(String(100), nullable=True)

    # Location within field
    location_lat = Column(Float, nullable=True)
    location_lng = Column(Float, nullable=True)
    installation_depth_cm = Column(Float, nullable=True)  # For soil sensors

    # Sensor specifications
    measurement_unit = Column(String(20), nullable=True)
    measurement_range_min = Column(Float, nullable=True)
    measurement_range_max = Column(Float, nullable=True)
    accuracy_percent = Column(Float, nullable=True)

    # Operational status
    is_active = Column(Boolean, default=True)
    last_reading_at = Column(DateTime, nullable=True)
    battery_level_percent = Column(Float, nullable=True)
    signal_strength_percent = Column(Float, nullable=True)

    # Calibration and maintenance
    calibration_date = Column(DateTime, nullable=True)
    next_calibration_date = Column(DateTime, nullable=True)
    firmware_version = Column(String(20), nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    field = relationship("Field", back_populates="sensors")
    readings = relationship("SensorReading", back_populates="sensor")

    def __repr__(self):
        return f"<Sensor(id={self.id}, type='{self.sensor_type}', active={self.is_active})>"

class SensorReading(Base):
    """Individual sensor measurements"""
    __tablename__ = 'sensor_readings'

    id = Column(Integer, primary_key=True, autoincrement=True)
    sensor_id = Column(Integer, ForeignKey('sensors.id'), nullable=False)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)

    # Measurement value
    value = Column(Float, nullable=False)
    unit = Column(String(20), nullable=False)

    # Quality indicators
    quality_score = Column(Float, nullable=True)  # 0-1 scale
    is_calibrated = Column(Boolean, default=True)
    is_valid = Column(Boolean, default=True)

    # Environmental context
    temperature_c = Column(Float, nullable=True)  # Ambient temperature
    battery_voltage = Column(Float, nullable=True)

    # Processing metadata
    processed_at = Column(DateTime, default=datetime.utcnow)
    alert_triggered = Column(Boolean, default=False)
    alert_type = Column(String(50), nullable=True)

    # Relationships
    sensor = relationship("Sensor", back_populates="readings")

    def __repr__(self):
        return f"<SensorReading(id={self.id}, sensor={self.sensor_id}, value={self.value}{self.unit})>"

class Alert(Base):
    """System alerts and notifications"""
    __tablename__ = 'alerts'

    id = Column(Integer, primary_key=True, autoincrement=True)
    farmer_id = Column(Integer, ForeignKey('farmers.id'), nullable=False)
    field_id = Column(Integer, ForeignKey('fields.id'), nullable=True)
    sensor_id = Column(Integer, ForeignKey('sensors.id'), nullable=True)

    # Alert details
    alert_type = Column(String(50), nullable=False)  # weather, sensor, prediction, disease
    severity = Column(String(20), default='medium')  # low, medium, high, critical
    title = Column(String(200), nullable=False)
    message = Column(Text, nullable=False)

    # Trigger information
    trigger_source = Column(String(100), nullable=True)  # sensor, weather_api, satellite
    trigger_value = Column(Float, nullable=True)
    threshold_value = Column(Float, nullable=True)

    # Status and actions
    is_active = Column(Boolean, default=True)
    is_acknowledged = Column(Boolean, default=False)
    acknowledged_at = Column(DateTime, nullable=True)
    acknowledged_by = Column(String(100), nullable=True)

    # Recommended actions
    recommended_actions = Column(JSON, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f"<Alert(id={self.id}, type='{self.alert_type}', severity='{self.severity}')>"

class SystemLog(Base):
    """System activity and error logging"""
    __tablename__ = 'system_logs'

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow)

    # Log details
    level = Column(String(20), nullable=False)  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    module = Column(String(100), nullable=False)
    function = Column(String(100), nullable=True)
    message = Column(Text, nullable=False)

    # Context information
    user_id = Column(Integer, nullable=True)
    request_id = Column(String(100), nullable=True)
    ip_address = Column(String(45), nullable=True)

    # Additional data
    extra_data = Column(JSON, nullable=True)

    def __repr__(self):
        return f"<SystemLog(id={self.id}, level='{self.level}', module='{self.module}')>"

class APICache(Base):
    """Cache for API responses to reduce external calls"""
    __tablename__ = 'api_cache'

    id = Column(Integer, primary_key=True, autoincrement=True)
    cache_key = Column(String(200), unique=True, nullable=False)
    data = Column(JSON, nullable=False)
    data_type = Column(String(50), nullable=False)  # weather, satellite, soil
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=False)

    # Usage tracking
    access_count = Column(Integer, default=0)
    last_accessed = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<APICache(id={self.id}, key='{self.cache_key}', type='{self.data_type}')>"

# Index definitions for performance
from sqlalchemy import Index

# Prediction indexes
Index('idx_prediction_field_date', Prediction.field_id, Prediction.created_at)
Index('idx_prediction_farmer', Prediction.farmer_id)
Index('idx_prediction_crop', Prediction.crop_type)

# Satellite data indexes
Index('idx_satellite_location_date', SatelliteData.location_lat, SatelliteData.location_lng, SatelliteData.date)
Index('idx_satellite_field_date', SatelliteData.field_id, SatelliteData.date)

# Weather data indexes
Index('idx_weather_location_date', WeatherData.location_lat, WeatherData.location_lng, WeatherData.date)
Index('idx_weather_field_date', WeatherData.field_id, WeatherData.date)

# Sensor indexes
Index('idx_sensor_field', Sensor.field_id)
Index('idx_sensor_reading_sensor_time', SensorReading.sensor_id, SensorReading.timestamp)

# Alert indexes
Index('idx_alert_farmer_active', Alert.farmer_id, Alert.is_active)
Index('idx_alert_field', Alert.field_id)

class SoilAnalysis(Base):
    """Soil intelligence analysis with multiple vegetation indices - cost-conscious GEE usage"""
    __tablename__ = 'soil_analyses'

    id = Column(Integer, primary_key=True, autoincrement=True)
    field_id = Column(Integer, ForeignKey('fields.id'), nullable=False)
    farmer_id = Column(Integer, ForeignKey('farmers.id'), nullable=False)

    # Field boundary (stored permanently for reuse)
    boundary_geojson = Column(Text, nullable=False)  # GeoJSON field boundary
    boundary_centroid_lat = Column(Float, nullable=True)
    boundary_centroid_lng = Column(Float, nullable=True)
    field_area_hectares = Column(Float, nullable=True)

    # Single expensive GEE processing metadata
    gee_analysis_timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)
    gee_processing_cost = Column(String(50), default='high_single_call')
    gee_project_used = Column(String(100), default='named-tome-472312-m3')

    # Satellite snapshot for visual reference
    satellite_snapshot_url = Column(String(300), nullable=True)  # S3 URL to field image
    satellite_snapshot_date = Column(Date, nullable=True)
    satellite_source = Column(String(50), default='modis')  # MODIS, Landsat, Sentinel

    # Vegetation Health Indices (NDVI, MSAVI2, NDRE, NDWI, NDMI, SOC_VIS, RSM, RVI)
    ndvi_value = Column(Float, nullable=False)  # Normalized Difference Vegetation Index
    ndvi_interpretation = Column(String(100), nullable=True)

    msavi2_value = Column(Float, nullable=False)  # Modified Soil Adjusted Vegetation Index
    msavi2_interpretation = Column(String(100), nullable=True)

    ndre_value = Column(Float, nullable=False)  # Normalized Difference Red Edge
    ndre_interpretation = Column(String(100), nullable=True)

    ndwi_value = Column(Float, nullable=False)  # Normalized Difference Water Index
    ndwi_interpretation = Column(String(100), nullable=True)

    ndmi_value = Column(Float, nullable=False)  # Normalized Difference Moisture Index
    ndmi_interpretation = Column(String(100), nullable=True)

    soc_vis_value = Column(Float, nullable=False)  # Soil Organic Carbon Visible
    soc_vis_interpretation = Column(String(100), nullable=True)

    rsm_value = Column(Float, nullable=False)  # Radar Soil Moisture
    rsm_interpretation = Column(String(100), nullable=True)

    rvi_value = Column(Float, nullable=False)  # Radar Vegetation Index
    rvi_interpretation = Column(String(100), nullable=True)

    # Calculated Soil Health Scores
    soil_health_score = Column(Float, nullable=True)  # Overall health score 0-100
    vegetation_vitality_score = Column(Float, nullable=True)  # Plant health 0-100
    moisture_reservoir_score = Column(Float, nullable=True)  # Water holding capacity 0-100
    organic_matter_score = Column(Float, nullable=True)  # SOC content 0-100
    soil_health_grade = Column(String(5), nullable=True)  # A+, A, A-, B+, etc.

    # Cost-effectiveness tracking
    reuse_count = Column(Integer, default=0)  # How many times data was accessed
    last_reused_at = Column(DateTime, nullable=True)
    value_generated_inr = Column(Float, default=0.0)  # Economic value from reuse

    # Analysis metadata
    analysis_status = Column(String(20), default='completed')  # pending, processing, completed, failed
    data_quality_score = Column(Float, nullable=True)  # 0-1 scale for analysis quality
    geolocation_accuracy = Column(Float, nullable=True)  # GPS accuracy in meters

    # Intelligence Applications Tracking
    used_for_predictions = Column(Boolean, default=False)
    used_for_recommendations = Column(Boolean, default=False)
    used_for_alerts = Column(Boolean, default=False)
    fed_to_ml_training = Column(Boolean, default=False)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    field = relationship("Field", back_populates="soil_analyses")
    farmer = relationship("Farmer", back_populates="soil_analyses")

    def __repr__(self):
        return f"<SoilAnalysis(id={self.id}, field={self.field_id}, health_score={self.soil_health_score})>"

class SoilIntelligenceCache(Base):
    """Caching layer for soil intelligence data to maximize reuse"""
    __tablename__ = 'soil_intelligence_cache'

    id = Column(Integer, primary_key=True, autoincrement=True)
    field_id = Column(Integer, ForeignKey('fields.id'), nullable=False)

    # Cache metadata
    cache_key = Column(String(200), nullable=False)  # unique identifier
    cache_type = Column(String(50), nullable=False)  # health_scores, recommendations, predictions

    # Cached data
    cached_data = Column(JSON, nullable=False)  # JSON response data
    calculated_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=False)  # TTL for cache

    # Usage tracking
    access_count = Column(Integer, default=0)
    last_accessed_at = Column(DateTime, nullable=True)

    # Cost tracking
    original_gee_cost_inr = Column(Float, nullable=True)  # Cost of initial GEE call
    revenue_generated_inr = Column(Float, default=0.0)  # Revenue from cache hits

    # Relationships
    field = relationship("Field", back_populates="soil_cache")

class SoilIntelligenceReport(Base):
    """Generated reports from soil intelligence analysis"""
    __tablename__ = 'soil_intelligence_reports'

    id = Column(Integer, primary_key=True, autoincrement=True)
    field_id = Column(Integer, ForeignKey('fields.id'), nullable=False)
    farmer_id = Column(Integer, ForeignKey('farmers.id'), nullable=False)
    soil_analysis_id = Column(Integer, ForeignKey('soil_analyses.id'), nullable=False)

    # Report metadata
    report_type = Column(String(50), nullable=False)  # health_score, recommendations, alerts
    report_title = Column(String(200), nullable=False)
    report_summary = Column(Text, nullable=False)
    report_generated_at = Column(DateTime, default=datetime.utcnow)

    # Report content
    soil_health_score = Column(Float, nullable=True)
    soil_health_grade = Column(String(5), nullable=True)
    recommendations = Column(JSON, nullable=True)  # Actionable recommendations
    risk_assessments = Column(JSON, nullable=True)  # Potential risks identified
    crop_suitability_matrix = Column(JSON, nullable=True)  # Crop compatibility scores

    # Generated insights
    key_insights = Column(JSON, nullable=True)
    urgent_actions = Column(JSON, nullable=True)
    seasonal_timeline = Column(JSON, nullable=True)

    # Usage tracking
    viewed_at = Column(DateTime, nullable=True)
    downloaded_at = Column(DateTime, nullable=True)
    shared_with = Column(String(200), nullable=True)  # Track sharing with cooperatives/banks

    # Relationships
    field = relationship("Field", back_populates="soil_reports")
    farmer = relationship("Farmer", back_populates="soil_reports")
    soil_analysis = relationship("SoilAnalysis", back_populates="reports")

    def __repr__(self):
        return f"<SoilIntelligenceReport(id={self.id}, type='{self.report_type}', field={self.field_id})>"

# Add soil intelligence relationships to existing models
Farmer.soil_analyses = relationship("SoilAnalysis", back_populates="farmer")
Farmer.soil_reports = relationship("SoilIntelligenceReport", back_populates="farmer")

Field.soil_analyses = relationship("SoilAnalysis", back_populates="field")
Field.soil_cache = relationship("SoilIntelligenceCache", back_populates="field")
Field.soil_reports = relationship("SoilIntelligenceReport", back_populates="field")

SoilAnalysis.reports = relationship("SoilIntelligenceReport", back_populates="soil_analysis")

# Soil intelligence indexes for performance
Index('idx_soil_field', SoilAnalysis.field_id)
Index('idx_soil_farmer', SoilAnalysis.farmer_id)
Index('idx_soil_health_score', SoilAnalysis.soil_health_score)
Index('idx_soil_analysis_timestamp', SoilAnalysis.gee_analysis_timestamp)
Index('idx_soil_reuse_count', SoilAnalysis.reuse_count)

# Cache indexes
Index('idx_soil_cache_field', SoilIntelligenceCache.field_id)
Index('idx_soil_cache_key', SoilIntelligenceCache.cache_key)
Index('idx_soil_cache_expires', SoilIntelligenceCache.cache_key, SoilIntelligenceCache.expires_at)

# Report indexes
Index('idx_soil_report_field', SoilIntelligenceReport.field_id)
Index('idx_soil_report_farmer', SoilIntelligenceReport.farmer_id)
Index('idx_soil_report_type', SoilIntelligenceReport.report_type)

class Consultation(Base):
    """Gemini agricultural consultation records for learning feedback"""
    __tablename__ = 'consultations'

    id = Column(Integer, primary_key=True, autoincrement=True)
    consultation_id = Column(String(100), unique=True, nullable=False)  # gai_YYYYMMDD_HHMMSS

    # Farmer context
    farmer_id = Column(Integer, nullable=True)
    farmer_name = Column(String(100), nullable=True)
    farmer_experience_years = Column(Integer, nullable=True)
    farmer_location = Column(String(200), nullable=True)

    # Consultation context
    request_timestamp = Column(DateTime, default=datetime.utcnow)
    consultation_text = Column(Text, nullable=False)
    processing_time_seconds = Column(Float, nullable=True)
    response_length = Column(Integer, nullable=True)

    # Field & crop context
    field_id = Column(Integer, nullable=True)
    crop_type = Column(String(50), nullable=True)
    crop_age_days = Column(Integer, nullable=True)
    field_area_ha = Column(Float, nullable=True)
    soil_ph = Column(Float, nullable=True)
    irrigation_method = Column(String(50), nullable=True)

    # Intelligence quality
    intelligence_confidence = Column(Float, nullable=True)  # AI confidence 0-1
    scientific_foundation = Column(String(100), default='ICAR-recommended')
    recommendation_categories = Column(JSON, nullable=True)  # ["fert_nutrition", "irrigation", "pest_control"]

    # Data utilization tracking
    satellite_data_used = Column(Boolean, default=False)
    weather_data_used = Column(Boolean, default=False)
    regional_knowledge_used = Column(Boolean, default=False)
    crop_knowledge_used = Column(Boolean, default=True)
    farmer_profile_used = Column(Boolean, default=False)

    # Cost tracking
    consultation_cost_usd = Column(Float, default=0.0003)
    expected_value_generation_inr = Column(Float, nullable=True)  # Economic value from advice

    # Learning & analytics
    gemini_model_version = Column(String(50), default='gemini-2.0-flash-exp')
    api_version = Column(String(20), default='2.0_gemini_unified')

    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    feedback = relationship("ConsultationFeedback", back_populates="consultation", uselist=False)

    def __repr__(self):
        return f"<Consultation(id={self.id}, farmer='{self.farmer_name}', crop='{self.crop_type}')>"

class ConsultationFeedback(Base):
    """Farmer feedback on consultations for continuous learning"""
    __tablename__ = 'consultation_feedback'

    id = Column(Integer, primary_key=True, autoincrement=True)
    consultation_id = Column(String(100), ForeignKey('consultations.consultation_id'), unique=True, nullable=False)

    # Rating & satisfaction
    overall_rating = Column(Float, nullable=True)  # 1-5 scale
    helpfulness_rating = Column(Float, nullable=True)  # 1-5 scale
    accuracy_rating = Column(Float, nullable=True)  # 1-5 scale

    # Detailed feedback
    feedback_text = Column(Text, nullable=True)
    implementation_intent = Column(Float, nullable=True)  # 0-1 scale (likelihood to implement)
    actual_implementation = Column(Boolean, nullable=True)  # Whether advice was actually followed

    # Impact assessment (collected post-season)
    yield_improvement_observed = Column(Float, nullable=True)  # Percentage improvement
    cost_savings_inr = Column(Float, nullable=True)
    labor_savings_hours = Column(Float, nullable=True)
    satisfaction_comments = Column(Text, nullable=True)

    # Learning categories (farmer feedback for AI improvement)
    helpful_aspects = Column(JSON, nullable=True)  # ["specific_recommendations", "timely_advice"]
    improved_aspects = Column(JSON, nullable=True)  # ["technical_detail", "local_context"]

    # Demographic correlation data
    farmer_experience_level = Column(String(20), nullable=True)
    farmer_farm_size_category = Column(String(20), nullable=True)  # small, medium, large
    farmer_budget_constraint = Column(String(20), nullable=True)

    # Metadata
    feedback_submitted_at = Column(DateTime, default=datetime.utcnow)
    feedback_channel = Column(String(20), default='api')  # api, whatsapp, web, sms
    feedback_source = Column(String(50), nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    consultation = relationship("Consultation", back_populates="feedback")

    def __repr__(self):
        return f"<ConsultationFeedback(id={self.id}, consultation='{self.consultation_id}', rating={self.overall_rating})>"

class ConsultationAnalytics(Base):
    """Analytics and performance metrics for consultation system"""
    __tablename__ = 'consultation_analytics'

    id = Column(Integer, primary_key=True, autoincrement=True)

    # Time period
    analysis_period_start = Column(DateTime, nullable=False)
    analysis_period_end = Column(DateTime, nullable=False)
    analysis_type = Column(String(50), nullable=False)  # daily, weekly, monthly, crop_season

    # Usage statistics
    total_consultations = Column(Integer, default=0)
    unique_farmers_served = Column(Integer, default=0)
    unique_crops_covered = Column(Integer, default=0)
    states_reached = Column(JSON, nullable=True)  # ["Punjab", "Haryana", "Delhi"]

    # Performance metrics
    average_response_time_seconds = Column(Float, nullable=True)
    average_consultation_quality_score = Column(Float, nullable=True)
    farmer_satisfaction_rate = Column(Float, nullable=True)

    # Economic impact
    total_cost_usd = Column(Float, default=0.0)
    estimated_value_generated_inr = Column(Float, nullable=True)
    roi_ratio = Column(Float, nullable=True)  # Value generated / cost invested

    # Crop-specific insights
    crop_consultation_distribution = Column(JSON, nullable=True)
    most_helpful_advice_types = Column(JSON, nullable=True)

    # Regional performance
    top_performing_states = Column(JSON, nullable=True)
    regional_adoption_rates = Column(JSON, nullable=True)

    # Generated insights
    key_success_patterns = Column(JSON, nullable=True)
    improvement_recommendations = Column(JSON, nullable=True)
    farmer_segmentation_insights = Column(JSON, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<ConsultationAnalytics(id={self.id}, period_start={self.analysis_period_start.date()}, consultations={self.total_consultations})>"

class LearningPattern(Base):
    """Learned patterns from consultation feedback for adaptive AI"""
    __tablename__ = 'learning_patterns'

    id = Column(Integer, primary_key=True, autoincrement=True)

    # Pattern identification
    pattern_type = Column(String(50), nullable=False)  # recommendation_type, regional_preference, farmer_behavior
    pattern_category = Column(String(50), nullable=False)  # crop_specific, regional, general
    pattern_name = Column(String(100), nullable=False)

    # Pattern conditions (when to apply)
    applicability_conditions = Column(JSON, nullable=False)  # {"crop": "rice", "region": "ncr", "soil_ph": {"min": 6.5, "max": 7.5}}

    # Pattern action/recommendation
    recommended_action = Column(JSON, nullable=False)  # {"type": "fertilizer", "quantity": "50kg_urea", "timing": "immediate"}

    # Learning confidence & validation
    confidence_score = Column(Float, default=0.5)  # 0-1 scale, based on feedback
    validation_count = Column(Integer, default=1)
    success_rate = Column(Float, default=0.8)
    last_validated_at = Column(DateTime, default=datetime.utcnow)

    # Adoption tracking
    times_applied = Column(Integer, default=0)
    positive_feedbacks = Column(Integer, default=0)
    negative_feedbacks = Column(Integer, default=0)

    # Metadata
    discovered_from_consultation_id = Column(String(100), nullable=True)
    pattern_source = Column(String(50), nullable=True)  # farmer_feedback, expert_validation, historical_analysis

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f"<LearningPattern(id={self.id}, type='{self.pattern_type}', confidence={self.confidence_score})>"

# Consultation relationships to existing models
Farmer.consultations = relationship("Consultation", back_populates="farmer")

# Index definitions for consultation learning system
Index('idx_consultation_id', Consultation.consultation_id)
Index('idx_consultation_farmer', Consultation.farmer_id)
Index('idx_consultation_crop', Consultation.crop_type)
Index('idx_consultation_timestamp', Consultation.request_timestamp)

Index('idx_feedback_consultation', ConsultationFeedback.consultation_id)
Index('idx_feedback_rating', ConsultationFeedback.overall_rating)
Index('idx_feedback_submitted', ConsultationFeedback.feedback_submitted_at)

Index('idx_analytics_period', ConsultationAnalytics.analysis_period_start, ConsultationAnalytics.analysis_period_end)
Index('idx_analytics_type', ConsultationAnalytics.analysis_type)

Index('idx_learning_confidence', LearningPattern.confidence_score)
Index('idx_learning_category', LearningPattern.pattern_category)
Index('idx_learning_validation', LearningPattern.last_validated_at)

# Cache indexes
Index('idx_cache_key_expires', APICache.cache_key, APICache.expires_at)
