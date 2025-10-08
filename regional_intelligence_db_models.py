#!/usr/bin/env python3
"""
Regional Intelligence Database Models - Phase 2 Week 3
Database schema for regional environmental intelligence storage
"""

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, Boolean, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from datetime import datetime
from typing import List

Base = declarative_base()

class AgriculturalRegion(Base):
    """Agricultural regions with geographic boundaries and characteristics"""

    __tablename__ = 'agricultural_regions'

    id = Column(String(50), primary_key=True)  # e.g., "ncr_gangetic_plain"
    name = Column(String(100), nullable=False)  # e.g., "NCR Gangetic Plain"
    region_type = Column(String(50))  # "district", "taluk", "state", "custom"

    # Geographic bounds
    min_longitude = Column(Float, nullable=False)
    min_latitude = Column(Float, nullable=False)
    max_longitude = Column(Float, nullable=False)
    max_latitude = Column(Float, nullable=False)

    # Agricultural characteristics
    primary_crops = Column(Text)  # JSON array: ["rice", "wheat", "sugarcane"]
    irrigation_type = Column(String(50))  # "canal_perennial", "well_irrigation", etc.
    climate_zone = Column(String(50))  # "subtropical", "gangetic_plain", etc.
    soil_type = Column(String(50))  # "alluvial", "black_cotton", "red_soil", etc.
    average_rainfall_mm = Column(Float)
    average_temperature_celsius = Column(Float)

    # Operational data
    active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    vegetation_analyses = relationship("VegetationHealthAnalysis", back_populates="region")
    pest_analyses = relationship("PestRiskAnalysis", back_populates="region")

    def __repr__(self):
        return f"<AgriculturalRegion(id='{self.id}', name='{self.name}')>"

class VegetationHealthAnalysis(Base):
    """Regional vegetation health analysis results"""

    __tablename__ = 'vegetation_health_analysis'

    id = Column(Integer, primary_key=True)
    region_id = Column(String(50), ForeignKey('agricultural_regions.id'), nullable=False)
    analysis_timestamp = Column(DateTime, default=datetime.utcnow)

    # NDVI and health metrics
    current_ndvi_average = Column(Float)
    ndvi_trend_30days = Column(String(20))
    health_score = Column(Float)  # 0-100

    # Stress indicators
    water_stress_detected = Column(Boolean, default=False)
    nutrient_stress_detected = Column(Boolean, default=False)
    disease_stress_detected = Column(Boolean, default=False)

    # Critical zones as JSON
    critical_zones = Column(Text)  # JSON array of critical areas

    # Health insights
    overall_health_trend = Column(String(100))
    predicted_7day_change = Column(String(50))
    comparison_to_last_season = Column(String(100))

    # Processing metadata
    gee_request_cost = Column(Float)  # Cost tracking for optimization
    processing_time_seconds = Column(Float)
    sentinel2_image_count = Column(Integer)
    analysis_status = Column(String(20), default="completed")

    # Relationships
    region = relationship("AgriculturalRegion", back_populates="vegetation_analyses")

class PestRiskAnalysis(Base):
    """Pest infestation cluster analysis and risk assessment"""

    __tablename__ = 'pest_risk_analysis'

    id = Column(Integer, primary_key=True)
    region_id = Column(String(50), ForeignKey('agricultural_regions.id'), nullable=False)
    analysis_timestamp = Column(DateTime, default=datetime.utcnow)

    # Overall risk assessment
    overall_risk_level = Column(String(20))  # "low", "moderate", "high", "critical"
    high_risk_zones = Column(Integer)

    # Pest types detected (JSON)
    pest_types_detected = Column(Text)  # JSON array: ["brown_planthopper", "stem_borer"]

    # Cluster data as JSON
    cluster_locations = Column(Text)  # JSON array of pest clusters

    # Risk assessment details
    immediate_treatment_needed = Column(Boolean, default=False)
    surveillance_zones_established = Column(Integer)
    preventive_measures_recommended = Column(Text)  # JSON array

    # Economic impact
    potential_loss_percentage = Column(Float)
    affected_acres = Column(Float)

class EnvironmentalStressMapping(Base):
    """Environmental stress patterns and factors"""

    __tablename__ = 'environmental_stress_mapping'

    id = Column(Integer, primary_key=True)
    region_id = Column(String(50), nullable=False)  # Will add FK constraint after region table exists
    analysis_timestamp = Column(DateTime, default=datetime.utcnow)

    # Water stress data
    water_stress_total_area_sqkm = Column(Float)
    water_stress_critical_sqkm = Column(Float)
    water_stress_high_sqkm = Column(Float)
    water_stress_moderate_sqkm = Column(Float)
    water_stress_causes = Column(Text)  # JSON array
    water_stress_recommendations = Column(Text)  # JSON array

    # Nutrient stress data
    nitrogen_transfer_zones = Column(Integer)
    phosphorus_deficit_estimated = Column(Boolean, default=False)
    potassium_imbalance_detected = Column(Boolean, default=False)
    micro_nutrient_deficiencies = Column(Text)  # JSON array

    # Soil health indicators
    organic_matter_decline_zones = Column(Integer)
    salinity_stress_areas_sqkm = Column(Float)
    erosion_risk_zones_sqkm = Column(Float)
    compaction_problem_areas_sqkm = Column(Float)

    # Climate stress patterns
    heat_wave_impact_zones = Column(Integer)
    frost_risk_areas = Column(Integer)
    heavy_rainfall_stress_zones = Column(Integer)
    drought_vulnerability_index = Column(Float)

class RegionalIntelligenceCache(Base):
    """Cached regional intelligence results to prevent redundant GEE processing"""

    __tablename__ = 'regional_intelligence_cache'

    id = Column(Integer, primary_key=True)

    # Cache identification
    region_id = Column(String(50), nullable=False)
    analysis_type = Column(String(50), nullable=False)  # "vegetation_health", "pest_risk", etc.
    cache_key = Column(String(100), unique=True)  # Combined region_type identifier

    # Cache metadata
    analysis_timestamp = Column(DateTime, nullable=False)
    cached_at = Column(DateTime, default=datetime.utcnow)
    cache_age_hours = Column(Float)
    is_valid = Column(Boolean, default=True)

    # Cached data
    analysis_results = Column(Text, nullable=False)  # JSON results

    # Cost and processing tracking
    processing_cost_usd = Column(Float)
    processing_time_seconds = Column(Float)

class RegionalRecommendation(Base):
    """Regional-level recommendations generated from intelligence analysis"""

    __tablename__ = 'regional_recommendations'

    id = Column(Integer, primary_key=True)
    region_id = Column(String(50), nullable=False)
    generated_at = Column(DateTime, default=datetime.utcnow)

    # Recommendation details
    recommendation_type = Column(String(50))  # "pest_control", "irrigation", "fertilizer", etc.
    priority_level = Column(String(20))  # "critical", "high", "medium", "low"
    title = Column(String(200), nullable=False)
    description = Column(Text, nullable=False)

    # Implementation details
    affected_area_sqkm = Column(Float)
    implementation_cost_estimate = Column(Float)
    time_to_implement_weeks = Column(Integer)
    expected_benefits = Column(Text)  # JSON: economic impact, yield improvement, etc.

    # Status tracking
    status = Column(String(20), default="generated")  # "generated", "approved", "implemented", "monitored"
    approved_at = Column(DateTime)
    implemented_at = Column(DateTime)

class IntelligenceAnalysisTriggers(Base):
    """Triggers for automatic intelligence analysis based on conditions"""

    __tablename__ = 'intelligence_analysis_triggers'

    id = Column(Integer, primary_key=True)
    trigger_type = Column(String(50), nullable=False)  # "time_based", "weather_condition", "vegetation_anomaly"
    trigger_name = Column(String(100), nullable=False)

    # Target regions
    region_ids = Column(Text)  # JSON array of region IDs

    # Trigger conditions
    cron_schedule = Column(String(50))  # Cron expression for time-based triggers
    vegetation_ndvi_threshold = Column(Float)
    weather_condition = Column(String(100))

    # Execution settings
    priority = Column(String(20), default="normal")  # "high", "normal", "low"
    enabled = Column(Boolean, default=True)
    last_triggered = Column(DateTime)
    trigger_count = Column(Integer, default=0)

    # Notification settings
    notify_on_trigger = Column(Boolean, default=True)
    notification_channels = Column(Text)  # JSON: email, sms, webhook URLs

# Database engine and session setup
def create_regional_intelligence_db(engine_url: str = None):
    """Create database tables for regional intelligence"""

    if engine_url is None:
        engine_url = "postgresql://postgres:password@localhost:5432/plant_saathi"

    engine = create_engine(engine_url, echo=True)
    Base.metadata.create_all(engine)

    # Create session factory
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    return engine, SessionLocal

def populate_sample_regions(session):
    """Populate database with agricultural regions"""

    regions_data = [
        {
            "id": "punjab_rice_belt",
            "name": "Punjab Rice Belt",
            "region_type": "agricultural_zone",
            "min_longitude": 74.0, "min_latitude": 29.5,
            "max_longitude": 77.0, "max_latitude": 32.5,
            "primary_crops": '["rice", "wheat"]',
            "irrigation_type": "canal_perennial",
            "climate_zone": "subtropical",
            "soil_type": "alluvial",
            "average_rainfall_mm": 650.0,
            "average_temperature_celsius": 23.8
        },
        {
            "id": "ncr_gangetic_plain",
            "name": "NCR Gangetic Plain",
            "region_type": "metropolitan_agricultural",
            "min_longitude": 76.8, "min_latitude": 28.2,
            "max_longitude": 78.2, "max_latitude": 29.2,
            "primary_crops": '["rice", "wheat", "sugarcane"]',
            "irrigation_type": "yamuna_canal",
            "climate_zone": "semi_arid",
            "soil_type": "gangetic_alluvial",
            "average_rainfall_mm": 714.0,
            "average_temperature_celsius": 25.5
        },
        {
            "id": "maharashtra_cotton_belt",
            "name": "Maharashtra Cotton Belt",
            "region_type": "commodity_zone",
            "min_longitude": 73.5, "min_latitude": 18.0,
            "max_longitude": 81.0, "max_latitude": 22.0,
            "primary_crops": '["cotton", "sugarcane", "soybean"]',
            "irrigation_type": "well_irrigation",
            "climate_zone": "deccan_plateau",
            "soil_type": "black_cotton",
            "average_rainfall_mm": 625.0,
            "average_temperature_celsius": 27.2
        }
    ]

    for region_data in regions_data:
        region = AgriculturalRegion(**region_data)
        session.add(region)

    session.commit()
    print(f"✅ Populated {len(regions_data)} sample agricultural regions")

if __name__ == "__main__":
    # Create database and populate sample data
    print("🏗️ REGIONAL INTELLIGENCE DATABASE SETUP")
    print("=" * 50)

    try:
        engine, SessionLocal = create_regional_intelligence_db()

        # Populate sample regions
        with SessionLocal() as session:
            populate_sample_regions(session)

        print("✅ Regional intelligence database setup complete")
        print("✅ Agricultural regions defined for environmental analysis")
        print("✅ Intelligence storage schema ready for Phase 2")

        # Database schema summary
        tables_created = [
            "agricultural_regions",
            "vegetation_health_analysis",
            "pest_risk_analysis",
            "environmental_stress_mapping",
            "regional_intelligence_cache",
            "regional_recommendations",
            "intelligence_analysis_triggers"
        ]

        print("\n📊 TABLES CREATED:")
        for table in tables_created:
            print(f"   ✅ {table}")

    except Exception as e:
        print(f"❌ Database setup failed: {e}")
        print("💡 Make sure PostgreSQL is running and accessible")
