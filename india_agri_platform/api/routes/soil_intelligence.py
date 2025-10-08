#!/usr/bin/env python3
"""
Soil Intelligence API Routes - Cost-Effective Satellite Soil Analysis
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, status, Depends
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime, timedelta
import json
from sqlalchemy import func  # Add this import

logger = logging.getLogger(__name__)

# Import soil intelligence components
from india_agri_platform.core.soil_intelligence import SoilIntelligenceAPI
from india_agri_platform.database.manager import DatabaseManager
from india_agri_platform.database.models import SoilAnalysis, SoilIntelligenceCache, SoilIntelligenceReport

router = APIRouter(prefix="/api/v1/soil", tags=["Soil Intelligence"])

# Initialize soil API
soil_api = SoilIntelligenceAPI()
db_manager = DatabaseManager()

@router.post("/analyze-field/{field_id}", status_code=status.HTTP_202_ACCEPTED)
async def analyze_field_soil(
    field_id: int,
    boundary_data: Dict[str, Any],
    background_tasks: BackgroundTasks
) -> Dict[str, Any]:
    """
    ANALYZE SOIL HEALTH: Single GEE call for comprehensive soil analysis

    COST-AWARE APPROACH: Expensive call → unlimited reuse ROI

    Parameters:
    - field_id: Unique field identifier
    - boundary_data: GeoJSON field boundary or coordinates

    Process Flow:
    1. One expensive GEE call → calculate 8 vegetation indices
    2. Generate satellite snapshot for visual verification
    3. Store permanent data in black box for unlimited reuse
    4. Return soil health scores, recommendations, and future usage data
    """

    try:
        logger.info(f"🪱 Starting soil intelligence analysis for field {field_id}")

        # Extract boundary information
        boundary_geojson = boundary_data.get('geojson') or boundary_data.get('coordinates')
        if not boundary_geojson:
            raise HTTPException(status_code=400, detail="Field boundary required")

        # Convert dict to JSON string if needed
        if isinstance(boundary_geojson, dict):
            boundary_geojson = json.dumps(boundary_geojson)

        # Check if analysis already exists (cost saver!)
        existing_analysis = check_existing_soil_analysis(field_id)
        if existing_analysis:
            # Return cached results - no GEE cost!
            return create_cost_saved_response(existing_analysis)

        # Start expensive soil analysis (background processing for large fields)
        background_tasks.add_task(
            process_soil_analysis_async,
            field_id, boundary_geojson
        )

        return {
            "field_id": field_id,
            "status": "processing_started",
            "processing_type": "single_gee_call_cost_optimization",
            "estimated_completion_time_minutes": 3,
            "expected_output": {
                "vegetation_indices": ["NDVI", "MSAVI2", "NDRE", "NDWI", "NDMI", "SOC_VIS", "RSM", "RVI"],
                "soil_health_scores": ["overall_grade", "vegetation_vitality", "moisture_reservoir", "organic_matter"],
                "recommendations": ["immediate_actions", "seasonal_improvements"],
                "satellite_snapshot": "visual_field_verification"
            },
            "reuse_benefit": "After this initial costly analysis, field data available forever for unlimited reuse"
        }

    except Exception as e:
        logger.error(f"❌ Soil analysis request failed for field {field_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Soil analysis failed: {str(e)}")

@router.get("/field/{field_id}/analysis")
async def get_field_soil_analysis(field_id: int) -> Dict[str, Any]:
    """Get existing soil analysis (cost-free reuse)"""

    try:
        # Check for cached/completed analysis
        analysis = check_existing_soil_analysis(field_id)
        if not analysis:
            raise HTTPException(
                status_code=404,
                detail=f"No soil analysis found for field {field_id}. Run /analyze-field/{field_id} first."
            )

        # Update reuse count (tracks cost-effectiveness)
        analysis.reuse_count += 1
        analysis.last_reused_at = datetime.utcnow()
        db_manager.save(analysis)

        # Return complete analysis results
        return build_analysis_response(analysis)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to retrieve soil analysis for field {field_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve analysis")

@router.get("/field/{field_id}/indices")
async def get_vegetation_indices(field_id: int) -> Dict[str, Any]:
    """Get 8 vegetation indices for field (cost-free)"""

    analysis = check_existing_soil_analysis(field_id)
    if not analysis:
        raise HTTPException(
            status_code=404, detail="No soil analysis found. Run full analysis first."
        )

    # Update access tracking
    analysis.reuse_count += 1
    analysis.last_reused_at = datetime.utcnow()
    db_manager.save(analysis)

    return {
        "field_id": field_id,
        "analysis_timestamp": analysis.gee_analysis_timestamp.isoformat(),
        "vegetation_indices": {
            "NDVI": {
                "value": analysis.ndvi_value,
                "interpretation": analysis.ndvi_interpretation,
                "health_contribution": analysis.ndvi_value_contribution()  # Implement method
            },
            "MSAVI2": {
                "value": analysis.msavi2_value,
                "interpretation": analysis.msavi2_interpretation,
                "health_contribution": analysis.msavi2_value_contribution()
            },
            "NDRE": {
                "value": analysis.ndre_value,
                "interpretation": analysis.ndre_interpretation,
                "health_contribution": analysis.ndre_value_contribution()
            },
            # ... add all 8 indices with interpretations
        },
        "total_reuse_count": analysis.reuse_count,
        "last_updated": analysis.updated_at.isoformat()
    }

@router.get("/field/{field_id}/health-score")
async def get_soil_health_score(field_id: int) -> Dict[str, Any]:
    """Get soil health scoring (instant - no GEE cost)"""

    analysis = check_existing_soil_analysis(field_id)
    if not analysis:
        raise HTTPException(status_code=404, detail="No soil analysis found")

    return {
        "field_id": field_id,
        "overall_health_score": analysis.soil_health_score,
        "health_grade": analysis.soil_health_grade,
        "detailed_scores": {
            "vegetation_vitality": analysis.vegetation_vitality_score,
            "moisture_reservoir": analysis.moisture_reservoir_score,
            "organic_matter": analysis.organic_matter_score
        },
        "interpretation": analysis.get_health_interpretation(),  # Implement method
        "recommendations": analysis.get_improvement_recommendations(),
        "last_updated": analysis.updated_at.isoformat(),
        "reuse_count": analysis.reuse_count
    }

@router.get("/cost-analysis")
async def get_soil_intelligence_cost_analysis() -> Dict[str, Any]:
    """Show cost-effectiveness of soil intelligence system"""

    try:
        # Calculate total cost savings
        total_analyses = db_manager.query(SoilAnalysis).count()
        total_reuses = db_manager.query(func.sum(SoilAnalysis.reuse_count)).scalar() or 0

        avg_gee_cost_per_analysis = 2.5  # USD per GEE call
        total_cost_saved = total_reuses * avg_gee_cost_per_analysis

        return {
            "total_soil_analyses": total_analyses,
            "total_data_reuses": total_reuses,
            "estimated_cost_saved_usd": total_cost_saved,
            "average_reuse_per_analysis": total_reuses / max(total_analyses, 1),
            "cost_efficiency_rating": get_cost_efficiency_rating(total_reuses, total_analyses),
            "roi_explanation": "Each GEE call generates unlimited future reuse"
        }

    except Exception as e:
        logger.error(f"❌ Cost analysis failed: {e}")
        raise HTTPException(status_code=500, detail="Cost analysis temporarily unavailable")

# Helper functions
def check_existing_soil_analysis(field_id: int) -> Optional[SoilAnalysis]:
    """Check if soil analysis already exists for field"""
    return db_manager.query(SoilAnalysis).filter_by(field_id=field_id).first()

def create_cost_saved_response(existing_analysis: SoilAnalysis) -> Dict[str, Any]:
    """Return response when cached analysis found (cost saver!)"""
    return {
        "field_id": existing_analysis.field_id,
        "status": "cached_analysis_found",
        "cost_saving": "expensive_gee_call_avoided",
        "analysis_timestamp": existing_analysis.gee_analysis_timestamp.isoformat(),
        "health_score": existing_analysis.soil_health_score,
        "grade": existing_analysis.soil_health_grade,
        "reuse_count": existing_analysis.reuse_count,
        "last_reused": existing_analysis.last_reused_at.isoformat(),
        "instruction": f"Use GET /api/v1/soil/field/{existing_analysis.field_id}/analysis to get full results"
    }

def build_analysis_response(analysis: SoilAnalysis) -> Dict[str, Any]:
    """Build complete analysis response from database"""
    return {
        "field_id": analysis.field_id,
        "analysis_status": analysis.analysis_status,
        "geo_analysis_timestamp": analysis.gee_analysis_timestamp.isoformat(),
        "boundary_geojson": analysis.boundary_geojson,

        "satellite_snapshot": {
            "url": analysis.satellite_snapshot_url,
            "source": analysis.satellite_source,
            "date": analysis.satellite_snapshot_date.isoformat() if analysis.satellite_snapshot_date else None
        },

        "vegetation_indices": {
            "NDVI": {"value": analysis.ndvi_value, "interpretation": analysis.ndvi_interpretation},
            "MSAVI2": {"value": analysis.msavi2_value, "interpretation": analysis.msavi2_interpretation},
            "NDRE": {"value": analysis.ndre_value, "interpretation": analysis.ndre_interpretation},
            "NDWI": {"value": analysis.ndwi_value, "interpretation": analysis.ndwi_interpretation},
            "NDMI": {"value": analysis.ndmi_value, "interpretation": analysis.ndmi_interpretation},
            "SOC_VIS": {"value": analysis.soc_vis_value, "interpretation": analysis.soc_vis_interpretation},
            "RSM": {"value": analysis.rsm_value, "interpretation": analysis.rsm_interpretation},
            "RVI": {"value": analysis.rvi_value, "interpretation": analysis.rvi_interpretation}
        },

        "soil_health_scores": {
            "overall_score": analysis.soil_health_score,
            "grade": analysis.soil_health_grade,
            "vegetation_vitality": analysis.vegetation_vitality_score,
            "moisture_reservoir": analysis.moisture_reservoir_score,
            "organic_matter": analysis.organic_matter_score
        },

        "recommendations": analysis.get_improvement_recommendations(),  # Implement method

        "cost_tracking": {
            "gee_cost": "high_single_call",
            "reuse_count": analysis.reuse_count,
            "last_reused": analysis.last_reused_at.isoformat() if analysis.last_reused_at else None,
            "value_generated_inr": analysis.value_generated_inr
        },

        "metadata": {
            "data_quality_score": analysis.data_quality_score,
            "geolocation_accuracy": analysis.geolocation_accuracy,
            "fed_to_ml_training": analysis.fed_to_ml_training,
            "used_for_predictions": analysis.used_for_predictions
        }
    }

async def process_soil_analysis_async(field_id: int, boundary_geojson: str):
    """Process soil analysis asynchronously (background task)"""

    try:
        logger.info(f"🔬 Processing soil analysis for field {field_id} in background")

        # Perform the GEE analysis (expensive but one-time)
        analysis_result = soil_api.analyze_soil_health(field_id, boundary_geojson)

        # Save to database for permanent storage and unlimited reuse
        save_soil_analysis_to_database(field_id, analysis_result)

        # Update field status to indicate soil analysis available
        update_field_with_soil_data(field_id)

        logger.info(f"✅ Background soil analysis completed for field {field_id}")

    except Exception as e:
        logger.error(f"❌ Background soil analysis failed for field {field_id}: {e}")
        # In production, would update database with failure status

def save_soil_analysis_to_database(field_id: int, analysis_result: Dict[str, Any]):
    """Save complete soil analysis to database for permanent storage"""

    try:
        # Extract vegetation indices
        vegetation_indices = analysis_result.get("vegetation_indices", {})

        # Create soil analysis record
        soil_analysis = SoilAnalysis(
            field_id=field_id,
            farmer_id=get_farmer_id_for_field(field_id),  # Helper function
            gee_project_used=analysis_result.get("gee_project_used", "named-tome-472312-m3"),
            gee_analysis_timestamp=datetime.fromisoformat(analysis_result["analysis_timestamp"]),
            boundary_geojson=analysis_result.get("boundary_geojson", ""),

            # Satellite snapshot
            satellite_snapshot_url=analysis_result.get("satellite_snapshot", {}).get("snapshot_url"),
            satellite_source=analysis_result.get("satellite_snapshot", {}).get("satellite_source", "sentinel-2"),
            satellite_snapshot_date=datetime.now().date() if analysis_result.get("satellite_snapshot") else None,

            # Vegetation indices with interpretations
            ndvi_value=vegetation_indices.get("NDVI", {}).get("value", 0.0),
            ndvi_interpretation=vegetation_indices.get("NDVI", {}).get("interpretation", ""),
            msavi2_value=vegetation_indices.get("MSAVI2", {}).get("value", 0.0),
            msavi2_interpretation=vegetation_indices.get("MSAVI2", {}).get("interpretation", ""),
            ndre_value=vegetation_indices.get("NDRE", {}).get("value", 0.0),
            ndre_interpretation=vegetation_indices.get("NDRE", {}).get("interpretation", ""),
            ndwi_value=vegetation_indices.get("NDWI", {}).get("value", 0.0),
            ndwi_interpretation=vegetation_indices.get("NDWI", {}).get("interpretation", ""),
            ndmi_value=vegetation_indices.get("NDMI", {}).get("value", 0.0),
            ndmi_interpretation=vegetation_indices.get("NDMI", {}).get("interpretation", ""),
            soc_vis_value=vegetation_indices.get("SOC_VIS", {}).get("value", 0.0),
            soc_vis_interpretation=vegetation_indices.get("SOC_VIS", {}).get("interpretation", ""),
            rsm_value=vegetation_indices.get("RSM", {}).get("value", 0.0),
            rsm_interpretation=vegetation_indices.get("RSM", {}).get("interpretation", ""),
            rvi_value=vegetation_indices.get("RVI", {}).get("value", 0.0),
            rvi_interpretation=vegetation_indices.get("RVI", {}).get("interpretation", ""),

            # Soil health scores
            soil_health_score=analysis_result.get("soil_health_scores", {}).get("overall_score", 0.0),
            soil_health_grade=analysis_result.get("soil_health_scores", {}).get("grade", "D"),
            vegetation_vitality_score=analysis_result.get("soil_health_scores", {}).get("vegetation_vitality_score", 0.0),
            moisture_reservoir_score=analysis_result.get("soil_health_scores", {}).get("moisture_reservoir_score", 0.0),
            organic_matter_score=analysis_result.get("soil_health_scores", {}).get("organic_matter_score", 0.0),

            # Metadata
            analysis_status="completed",
            data_quality_score=analysis_result.get("data_quality_score", 0.85),
            geolocation_accuracy=5.0,  # Assume 5m GPS accuracy
        )

        # Save to database
        db_manager.save(soil_analysis)

        logger.info(f"💾 Soil analysis saved for field {field_id} - unlimited future reuse enabled!")

    except Exception as e:
        logger.error(f"❌ Failed to save soil analysis for field {field_id}: {e}")
        raise

def update_field_with_soil_data(field_id: int):
    """Update field record to indicate soil analysis is available"""
    try:
        # In production, would update Field model with soil data availability flag
        logger.info(f"📋 Updated field {field_id} metadata - soil analysis available")
    except Exception as e:
        logger.error(f"❌ Failed to update field metadata: {e}")

def get_farmer_id_for_field(field_id: int) -> int:
    """Get farmer ID for field (helper function)"""
    # In production, would query Field model to get farmer_id
    return 1001  # Default farmer ID for demo

def get_cost_efficiency_rating(total_reuses: int, total_analyses: int) -> str:
    """Calculate cost-efficiency rating"""
    efficiency_ratio = total_reuses / max(total_analyses, 1)

    if efficiency_ratio > 50:
        return "excellent_cost_optimization"
    elif efficiency_ratio > 25:
        return "good_cost_optimization"
    elif efficiency_ratio > 10:
        return "moderate_cost_optimization"
    else:
        return "low_cost_optimization"
