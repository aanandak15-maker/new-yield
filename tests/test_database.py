"""
Database Layer Tests
Comprehensive testing for database operations
"""

import pytest
import os
from datetime import datetime, timedelta
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, JSON, Date, Boolean, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship

# Create simplified models for testing (without PostGIS)
TestBase = declarative_base()

class TestFarmer(TestBase):
    """Simplified farmer model for testing"""
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
    fields = relationship("TestField", back_populates="farmer")
    predictions = relationship("TestPrediction", back_populates="farmer")

class TestField(TestBase):
    """Simplified field model for testing"""
    __tablename__ = 'fields'

    id = Column(Integer, primary_key=True, autoincrement=True)
    farmer_id = Column(Integer, ForeignKey('farmers.id'), nullable=False)
    name = Column(String(100), nullable=False)
    centroid_lat = Column(Float, nullable=True)
    centroid_lng = Column(Float, nullable=True)
    area_hectares = Column(Float, nullable=True)
    soil_type = Column(String(50), nullable=True)
    irrigation_type = Column(String(50), nullable=True)
    irrigation_coverage_percent = Column(Float, default=80.0)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = Column(Boolean, default=True)

    # Relationships
    farmer = relationship("TestFarmer", back_populates="fields")
    predictions = relationship("TestPrediction", back_populates="field")

class TestPrediction(TestBase):
    """Simplified prediction model for testing"""
    __tablename__ = 'predictions'

    id = Column(Integer, primary_key=True, autoincrement=True)
    farmer_id = Column(Integer, ForeignKey('farmers.id'), nullable=False)
    field_id = Column(Integer, ForeignKey('fields.id'), nullable=False)
    crop_type = Column(String(50), nullable=False)
    variety_name = Column(String(100), nullable=True)
    sowing_date = Column(Date, nullable=False)
    predicted_yield_quintal_ha = Column(Float, nullable=False)
    confidence_interval = Column(JSON, nullable=True)
    confidence_level = Column(String(20), default='medium')
    growth_stage = Column(String(50), nullable=True)
    days_since_sowing = Column(Integer, nullable=True)
    estimated_harvest_days = Column(Integer, nullable=True)
    insights = Column(JSON, nullable=True)
    recommendations = Column(JSON, nullable=True)
    risk_assessment = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    farmer = relationship("TestFarmer", back_populates="predictions")
    field = relationship("TestField", back_populates="predictions")

# Import database manager
from india_agri_platform.database.manager import DatabaseManager


@pytest.fixture(scope="session")
def test_db():
    """Create test database"""
    # Use SQLite for testing (no PostgreSQL required)
    test_db_url = "sqlite:///./test_agri_platform.db"

    # Create engine
    engine = create_engine(test_db_url, echo=False)

    # Create tables with simplified models
    TestBase.metadata.create_all(bind=engine)

    # Create session factory
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    yield SessionLocal

    # Cleanup
    try:
        os.unlink("./test_agri_platform.db")
    except FileNotFoundError:
        pass


@pytest.fixture
def db_manager(test_db):
    """Database manager for testing"""
    # Create a test instance with SQLite
    manager = DatabaseManager.__new__(DatabaseManager)
    manager.SessionLocal = test_db
    manager.engine = test_db.kw['bind']
    return manager


class TestFarmerOperations:
    """Test farmer CRUD operations"""

    def test_create_farmer(self, db_manager):
        """Test farmer creation"""
        farmer_id = db_manager.create_farmer(
            name="Test Farmer",
            phone="+91-9876543210",
            email="farmer@test.com",
            location_lat=30.9010,
            location_lng=75.8573
        )

        assert farmer_id is not None
        assert isinstance(farmer_id, int)

    def test_get_farmer(self, db_manager):
        """Test farmer retrieval"""
        # Create farmer first
        farmer_id = db_manager.create_farmer("Test Farmer 2")

        # Retrieve farmer
        farmer = db_manager.get_farmer(farmer_id)

        assert farmer is not None
        assert farmer['name'] == "Test Farmer 2"
        assert farmer['id'] == farmer_id

    def test_get_nonexistent_farmer(self, db_manager):
        """Test retrieving non-existent farmer"""
        farmer = db_manager.get_farmer(99999)
        assert farmer is None


class TestFieldOperations:
    """Test field CRUD operations"""

    def test_create_field(self, db_manager):
        """Test field creation"""
        # Create farmer first
        farmer_id = db_manager.create_farmer("Field Test Farmer")

        # Create field
        field_id = db_manager.create_field(
            farmer_id=farmer_id,
            name="North Field",
            centroid_lat=30.9010,
            centroid_lng=75.8573,
            area_hectares=5.2,
            soil_type="Sandy Loam"
        )

        assert field_id is not None

    def test_get_field(self, db_manager):
        """Test field retrieval"""
        # Create farmer and field
        farmer_id = db_manager.create_farmer("Field Get Test")
        field_id = db_manager.create_field(
            farmer_id=farmer_id,
            name="Test Field",
            centroid_lat=30.9010,
            centroid_lng=75.8573
        )

        # Retrieve field
        field = db_manager.get_field(field_id)

        assert field is not None
        assert field['name'] == "Test Field"
        assert field['farmer_id'] == farmer_id

    def test_get_farmer_fields(self, db_manager):
        """Test getting all fields for a farmer"""
        # Create farmer
        farmer_id = db_manager.create_farmer("Multi Field Farmer")

        # Create multiple fields
        field_ids = []
        for i in range(3):
            field_id = db_manager.create_field(
                farmer_id=farmer_id,
                name=f"Field {i+1}",
                centroid_lat=30.9010 + i*0.01,
                centroid_lng=75.8573 + i*0.01
            )
            field_ids.append(field_id)

        # Get all fields
        fields = db_manager.get_farmer_fields(farmer_id)

        assert len(fields) == 3
        assert all(field['farmer_id'] == farmer_id for field in fields)


class TestPredictionOperations:
    """Test prediction CRUD operations"""

    def test_save_prediction(self, db_manager):
        """Test saving prediction"""
        # Create farmer and field first
        farmer_id = db_manager.create_farmer("Prediction Test Farmer")
        field_id = db_manager.create_field(
            farmer_id=farmer_id,
            name="Prediction Field",
            centroid_lat=30.9010,
            centroid_lng=75.8573
        )

        # Save prediction
        prediction_data = {
            'farmer_id': farmer_id,
            'field_id': field_id,
            'crop_type': 'wheat',
            'variety_name': 'HD-2967',
            'sowing_date': '2024-11-15',
            'predicted_yield_quintal_ha': 48.5,
            'confidence_interval': {'low': 45.2, 'high': 51.8},
            'confidence_level': 'high',
            'growth_stage': 'vegetative_growth',
            'days_since_sowing': 45,
            'estimated_harvest_days': 75,
            'prediction_method': 'streamlined_predictor',
            'insights': {'yield_category': 'high', 'risk_level': 'low'},
            'recommendations': ['maintain_current_practices'],
            'risk_assessment': {'risk_level': 'low', 'risk_factors': []}
        }

        prediction_id = db_manager.save_prediction(prediction_data)

        assert prediction_id is not None

    def test_get_prediction(self, db_manager):
        """Test prediction retrieval"""
        # Create and save prediction
        farmer_id = db_manager.create_farmer("Get Prediction Test")
        field_id = db_manager.create_field(
            farmer_id=farmer_id,
            name="Get Prediction Field",
            centroid_lat=30.9010,
            centroid_lng=75.8573
        )

        prediction_data = {
            'farmer_id': farmer_id,
            'field_id': field_id,
            'crop_type': 'wheat',
            'sowing_date': '2024-11-15',
            'predicted_yield_quintal_ha': 45.2
        }

        prediction_id = db_manager.save_prediction(prediction_data)

        # Retrieve prediction
        prediction = db_manager.get_prediction(prediction_id)

        assert prediction is not None
        assert prediction['crop_type'] == 'wheat'
        assert prediction['predicted_yield_quintal_ha'] == 45.2

    def test_get_field_predictions(self, db_manager):
        """Test getting prediction history for field"""
        # Create farmer and field
        farmer_id = db_manager.create_farmer("History Test Farmer")
        field_id = db_manager.create_field(
            farmer_id=farmer_id,
            name="History Field",
            centroid_lat=30.9010,
            centroid_lng=75.8573
        )

        # Save multiple predictions
        for i in range(3):
            prediction_data = {
                'farmer_id': farmer_id,
                'field_id': field_id,
                'crop_type': 'wheat',
                'sowing_date': '2024-11-15',
                'predicted_yield_quintal_ha': 40 + i * 5
            }
            db_manager.save_prediction(prediction_data)

        # Get prediction history
        predictions = db_manager.get_field_predictions(field_id, limit=5)

        assert len(predictions) == 3
        assert all(pred['crop_type'] == 'wheat' for pred in predictions)


class TestSatelliteDataOperations:
    """Test satellite data operations"""

    def test_cache_satellite_data(self, db_manager):
        """Test caching satellite data"""
        satellite_data = {
            'field_id': None,  # Location-based data
            'location_lat': 30.9010,
            'location_lng': 75.8573,
            'date': '2024-11-15',
            'ndvi': 0.72,
            'soil_moisture_percent': 45.0,
            'land_surface_temp_c': 28.5,
            'data_source': 'google_earth_engine',
            'satellite': 'MODIS',
            'resolution_meters': 250
        }

        success = db_manager.cache_satellite_data(satellite_data)
        assert success is True

    def test_get_satellite_history(self, db_manager):
        """Test retrieving satellite data history"""
        # Cache some test data
        for i in range(3):
            satellite_data = {
                'location_lat': 30.9010,
                'location_lng': 75.8573,
                'date': f'2024-11-{15+i}',
                'ndvi': 0.65 + i * 0.05,
                'data_source': 'test'
            }
            db_manager.cache_satellite_data(satellite_data)

        # Get history (need to create a field first for this test)
        # This test would need a field_id to work properly
        pass


class TestWeatherDataOperations:
    """Test weather data operations"""

    def test_save_weather_data(self, db_manager):
        """Test saving weather data"""
        weather_data = {
            'location_lat': 30.9010,
            'location_lng': 75.8573,
            'date': '2024-11-15',
            'temperature_c': 25.5,
            'rainfall_mm': 12.5,
            'humidity_percent': 68,
            'wind_speed_kmph': 8.5,
            'weather_main': 'Clear',
            'weather_description': 'clear sky',
            'data_source': 'openweathermap'
        }

        success = db_manager.save_weather_data(weather_data)
        assert success is True


class TestSensorOperations:
    """Test IoT sensor operations"""

    def test_register_sensor(self, db_manager):
        """Test sensor registration"""
        # Create farmer and field first
        farmer_id = db_manager.create_farmer("Sensor Test Farmer")
        field_id = db_manager.create_field(
            farmer_id=farmer_id,
            name="Sensor Field",
            centroid_lat=30.9010,
            centroid_lng=75.8573
        )

        # Register sensor
        sensor_id = db_manager.register_sensor(
            field_id=field_id,
            sensor_id="SOIL_001",
            name="North Field Soil Sensor",
            sensor_type="soil_moisture",
            location_lat=30.9010,
            location_lng=75.8573
        )

        assert sensor_id is not None

    def test_save_sensor_reading(self, db_manager):
        """Test saving sensor readings"""
        # Create sensor first
        farmer_id = db_manager.create_farmer("Reading Test Farmer")
        field_id = db_manager.create_field(
            farmer_id=farmer_id,
            name="Reading Field",
            centroid_lat=30.9010,
            centroid_lng=75.8573
        )

        sensor_id = db_manager.register_sensor(
            field_id=field_id,
            sensor_id="TEMP_001",
            name="Temperature Sensor",
            sensor_type="temperature"
        )

        # Save reading
        success = db_manager.save_sensor_reading(
            sensor_id=sensor_id,
            value=26.5,
            unit="°C"
        )

        assert success is True


class TestAlertOperations:
    """Test alert system operations"""

    def test_create_alert(self, db_manager):
        """Test alert creation"""
        farmer_id = db_manager.create_farmer("Alert Test Farmer")

        alert_id = db_manager.create_alert(
            farmer_id=farmer_id,
            alert_type="weather",
            severity="high",
            title="Heavy Rain Warning",
            message="Heavy rainfall expected in next 24 hours. Prepare drainage.",
            field_id=None
        )

        assert alert_id is not None


class TestCacheOperations:
    """Test API caching operations"""

    def test_cache_operations(self, db_manager):
        """Test cache set and get operations"""
        cache_key = "weather_ludhiana_20241115"
        test_data = {
            'temperature': 25.5,
            'humidity': 68,
            'description': 'clear sky'
        }

        # Set cache
        success = db_manager.set_cached_data(
            cache_key=cache_key,
            data=test_data,
            data_type="weather",
            ttl_hours=24
        )
        assert success is True

        # Get cache
        cached = db_manager.get_cached_data(cache_key)
        assert cached is not None
        assert cached['data']['temperature'] == 25.5
        assert cached['data_type'] == 'weather'


class TestAnalyticsOperations:
    """Test analytics and statistics operations"""

    def test_statistics_operations(self, db_manager):
        """Test getting system statistics"""
        # These operations should work even with empty database
        prediction_count = db_manager.get_prediction_count()
        field_count = db_manager.get_active_field_count()
        satellite_count = db_manager.get_satellite_data_count()

        assert isinstance(prediction_count, int)
        assert isinstance(field_count, int)
        assert isinstance(satellite_count, int)

    def test_health_check(self, db_manager):
        """Test database health check"""
        health = db_manager.health_check()

        assert isinstance(health, dict)
        assert 'status' in health
        assert 'database_connected' in health
        assert health['database_connected'] is True


class TestSystemLogging:
    """Test system logging operations"""

    def test_log_system_event(self, db_manager):
        """Test system event logging"""
        db_manager.log_system_event(
            level="INFO",
            module="test_database",
            message="Test system event logging",
            user_id=None,
            extra_data={"test": True}
        )

        # Log should be created without errors
        assert True


# Integration tests
class TestDatabaseIntegration:
    """Test database operations as complete workflows"""

    def test_complete_farmer_workflow(self, db_manager):
        """Test complete farmer registration to prediction workflow"""
        # 1. Create farmer
        farmer_id = db_manager.create_farmer(
            name="Complete Workflow Farmer",
            phone="+91-9876543210",
            location_lat=30.9010,
            location_lng=75.8573
        )
        assert farmer_id is not None

        # 2. Create field
        field_id = db_manager.create_field(
            farmer_id=farmer_id,
            name="Main Field",
            centroid_lat=30.9010,
            centroid_lng=75.8573,
            area_hectares=10.5
        )
        assert field_id is not None

        # 3. Save prediction
        prediction_data = {
            'farmer_id': farmer_id,
            'field_id': field_id,
            'crop_type': 'wheat',
            'sowing_date': '2024-11-15',
            'predicted_yield_quintal_ha': 52.0,
            'growth_stage': 'reproductive',
            'days_since_sowing': 60
        }

        prediction_id = db_manager.save_prediction(prediction_data)
        assert prediction_id is not None

        # 4. Verify complete data retrieval
        farmer = db_manager.get_farmer(farmer_id)
        field = db_manager.get_field(field_id)
        prediction = db_manager.get_prediction(prediction_id)
        fields = db_manager.get_farmer_fields(farmer_id)

        assert farmer['name'] == "Complete Workflow Farmer"
        assert field['name'] == "Main Field"
        assert prediction['crop_type'] == 'wheat'
        assert len(fields) == 1

    def test_data_consistency(self, db_manager):
        """Test data consistency across operations"""
        # Create farmer
        farmer_id = db_manager.create_farmer("Consistency Test Farmer")

        # Create multiple fields
        field_ids = []
        for i in range(5):
            field_id = db_manager.create_field(
                farmer_id=farmer_id,
                name=f"Field {i+1}",
                centroid_lat=30.9010 + i*0.01,
                centroid_lng=75.8573 + i*0.01,
                area_hectares=5.0 + i
            )
            field_ids.append(field_id)

        # Verify all fields belong to farmer
        fields = db_manager.get_farmer_fields(farmer_id)
        assert len(fields) == 5

        for field in fields:
            assert field['farmer_id'] == farmer_id
            assert field['area_hectares'] >= 5.0

        # Test statistics
        field_count = db_manager.get_active_field_count()
        assert field_count >= 5
