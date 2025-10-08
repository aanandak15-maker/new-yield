"""
India Agricultural Intelligence Platform API
Unified Multi-Crop Yield Prediction Service
"""

import sys
from pathlib import Path
from fastapi import FastAPI, HTTPException, Query, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, List, Any
import logging
import uvicorn
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import platform components
from india_agri_platform.core.multi_crop_predictor import (
    get_multi_crop_predictor, predict_yield, get_platform_info
)
from india_agri_platform.core.error_handling import error_handler

# Import Firebase and Railway integrations (MVP Fix)
try:
    from firebase_config import firebase_manager
    from railway_config import railway_db
    FIREBASE_AVAILABLE = True
    RAILWAY_AVAILABLE = True
    logging.info("✅ Firebase and Railway integrations loaded successfully")
except ImportError as e:
    logging.warning(f"⚠️ Firebase/Railway integrations not available: {e}")
    FIREBASE_AVAILABLE = False
    RAILWAY_AVAILABLE = False

# Import ethical agricultural orchestrator
try:
    from ethical_agricultural_orchestrator import EthicalAgriculturalOrchestrator
    ethical_orchestrator = EthicalAgriculturalOrchestrator()
    ETHICAL_ORCHESTRATOR_AVAILABLE = True
    logging.info("✅ Ethical Agricultural Orchestrator loaded")
except ImportError as e:
    logging.warning(f"⚠️ Ethical Orchestrator not available: {e}")
    ETHICAL_ORCHESTRATOR_AVAILABLE = False

# Import unified 3-crop prediction API (Phase 3)
try:
    from india_agri_platform.api.unified_crop_api import unified_crop_router
    UNIFIED_CROP_API_AVAILABLE = True
    logging.info("✅ Unified 3-Crop Prediction API loaded")
except ImportError as e:
    logging.warning(f"⚠️ Unified Crop API not available: {e}")
    UNIFIED_CROP_API_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# FastAPI app with CORS for Firebase integration
app = FastAPI(
    title="India Agricultural Intelligence Platform API",
    description="Unified Multi-Crop Yield Prediction Service for Indian Farmers",
    version="2.0.0",
    contact={
        "name": "India Agricultural Intelligence Platform",
        "url": "https://github.com/Kevinbose/Crop-Yield-Prediction",
        "email": "agritech@platform.com"
    }
)

# Add CORS middleware for Firebase integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include unified crop prediction API (Phase 3)
if UNIFIED_CROP_API_AVAILABLE:
    app.include_router(unified_crop_router)
    logger.info("✅ Unified 3-Crop Prediction API integrated")

# Include advanced yield prediction API with trained models (Production Ready)
from india_agri_platform.api.routes.yield_prediction import router as yield_router
app.include_router(yield_router, prefix="/api/v1", tags=["Yield Prediction"])
logger.info("✅ Advanced Yield Prediction API with trained models integrated")

# Initialize multi-crop predictor
try:
    multi_crop_predictor = get_multi_crop_predictor()
    logger.info("✅ Multi-Crop Agricultural Platform initialized successfully")
except Exception as e:
    logger.error(f"❌ Platform initialization failed: {e}")
    raise RuntimeError("Failed to initialize agricultural platform")

# Pydantic models for API request/response
class PredictionRequest(BaseModel):
    crop: Optional[str] = Query(None, description="Crop type (rice, wheat, cotton) or None for auto-detection")
    latitude: float = Query(..., description="GPS latitude coordinate")
    longitude: float = Query(..., description="GPS longitude coordinate")
    variety_name: Optional[str] = Query(None, description="Specific crop variety")
    temperature_celsius: Optional[float] = Query(None, description="Current temperature in Celsius")
    rainfall_mm: Optional[float] = Query(None, description="Annual rainfall in mm")
    humidity_percent: Optional[float] = Query(None, description="Humidity percentage")
    irrigation_coverage: Optional[float] = Query(None, description="Irrigation coverage (0-1)")
    soil_ph: Optional[float] = Query(None, description="Soil pH level")
    area_hectares: Optional[float] = Query(None, description="Farm area in hectares")
    season: Optional[str] = Query(None, description="Growing season (kharif, rabi, summer)")

class PredictionResponse(BaseModel):
    crop: str
    variety: Optional[str]
    state: str
    predicted_yield_quintal_ha: float
    unit: str
    confidence_level: str
    insights: Optional[Dict[str, Any]] = None
    regional_context: Optional[str] = None
    crop_rotation_suggestions: Optional[List[str]] = None
    alternative_crops: Optional[List[Dict[str, Any]]] = None
    seasonal_intelligence: Optional[str] = None
    timestamp: str
    prediction_method: str

class PlatformInfo(BaseModel):
    platform_name: str
    version: str
    capabilities: Dict[str, Any]
    api_endpoints: List[str]
    supported_parameters: List[str]

# MVP Consultation Models (Firebase + Railway Integration)
class FarmerRegistration(BaseModel):
    phone_number: str
    display_name: Optional[str] = None
    location: Optional[str] = None
    farm_size: Optional[float] = None
    district: Optional[str] = None
    state: Optional[str] = None
    pincode: Optional[str] = None
    preferred_crops: Optional[List[str]] = None

class ConsultationRequest(BaseModel):
    farmer_query: str
    crop_type: Optional[str] = None
    location: Optional[str] = None
    district: Optional[str] = None
    state: Optional[str] = None
    phone_number: Optional[str] = None
    firebase_token: Optional[str] = None

class ConsultationResponse(BaseModel):
    consultation_id: str
    farmer_query: str
    solution_recommended: str
    ethical_tier_applied: str
    confidence_score: float
    safety_instructions: List[str]
    prevention_education: Dict[str, Any]
    response_timestamp: str
    timestamp: str
    platform_version: str

class FeedbackRequest(BaseModel):
    consultation_id: str
    farmer_uid: str
    feedback_text: Optional[str] = None
    rating: Optional[int] = Query(None, ge=1, le=5, description="Rating 1-5")
    helpful_boolean: Optional[bool] = None

class FarmerProfileResponse(BaseModel):
    firebase_uid: str
    phone_number: str
    display_name: Optional[str]
    consultation_count: int
    last_consultation: Optional[str]
    subscription_tier: str
    created_at: str

# API Routes
@app.get("/", tags=["Platform"])
async def root():
    """Platform information and API documentation"""
    return {
        "message": "🎯 India Agricultural Intelligence Platform API",
        "version": "2.0 - Multi-Crop Enterprise",
        "description": "Unified agricultural yield prediction for Indian farmers",
        "docs": "/docs",
        "redoc": "/redoc"
    }

@app.get("/health", tags=["Platform"])
async def health_check():
    """Platform health check"""
    return {
        "status": "healthy",
        "platform": "India Agricultural Intelligence Platform",
        "crops_available": multi_crop_predictor.get_available_crops(),
        "regional_coverage": "Major Indian agricultural regions",
        "intelligent_routing": True
    }

@app.get("/platform/info", response_model=PlatformInfo, tags=["Platform"])
async def get_platform_info():
    """Get detailed platform capabilities and features"""
    return get_platform_info()

@app.get("/crops", tags=["Crops"])
async def get_available_crops():
    """Get list of crops supported for prediction"""
    return {
        "available_crops": multi_crop_predictor.get_available_crops(),
        "total_crops": len(multi_crop_predictor.get_available_crops()),
        "auto_detection_supported": True
    }

@app.post("/predict/yield", response_model=PredictionResponse, tags=["Predictions"])
async def predict_crop_yield(request: PredictionRequest):
    """
    Unified crop yield prediction with intelligent routing.

    Make predictions by providing GPS coordinates - the platform will automatically:
    - Detect the appropriate crop for the region
    - Route to the correct crop predictor
    - Apply regional farming intelligence
    - Provide actionable insights

    Example requests:
    - Punjab rice prediction: latitude=30.5, longitude=75.5
    - Maharashtra cotton: latitude=19.5, longitude=75.5, crop="cotton"
    - Auto-detection: latitude=22.0, longitude=72.5 (Gujarat)
    """
    try:
        logger.info(f"🌾 API Request: crop={request.crop}, lat={request.latitude}, lng={request.longitude}")

        # Call unified prediction function
        result = predict_yield(
            crop=request.crop,
            latitude=request.latitude,
            longitude=request.longitude,
            variety_name=request.variety_name,
            temperature_celsius=request.temperature_celsius,
            rainfall_mm=request.rainfall_mm,
            humidity_percent=request.humidity_percent,
            irrigation_coverage=request.irrigation_coverage,
            soil_ph=request.soil_ph,
            area_hectares=request.area_hectares,
            season=request.season
        )

        # Check for API errors
        if 'error' in result:
            raise HTTPException(status_code=400, detail=result['error'])

        return PredictionResponse(**result)

    except HTTPException:
        raise
    except Exception as e:
        error_details = {
            "operation": "api_prediction",
            "crop": request.crop,
            "latitude": request.latitude,
            "longitude": request.longitude,
            "error": str(e)
        }
        error_handler.handle_error(e, error_details)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/predict/yield/location", response_model=PredictionResponse, tags=["Predictions"])
async def predict_by_location(
    latitude: float = Query(..., description="GPS latitude coordinate"),
    longitude: float = Query(..., description="GPS longitude coordinate"),
    crop: Optional[str] = Query(None, description="Specific crop to predict"),
    season: Optional[str] = Query(None, description="Growing season for auto-detection")
):
    """
    Quick location-based prediction with automatic crop detection.

    Simply provide GPS coordinates and the platform will:
    - Auto-detect the most likely crop for that region
    - Provide yield prediction with regional context
    - Include crop rotation suggestions
    - Offer alternative crop options
    """
    try:
        logger.info(f"📍 Location-based prediction: lat={latitude}, lng={longitude}, crop={crop}, season={season}")

        result = predict_yield(
            crop=crop,
            latitude=latitude,
            longitude=longitude,
            season=season
        )

        if 'error' in result:
            raise HTTPException(status_code=400, detail=result['error'])

        return PredictionResponse(**result)

    except HTTPException:
        raise
    except Exception as e:
        error_handler.handle_error(e, {
            "operation": "location_prediction",
            "latitude": latitude,
            "longitude": longitude
        })
        raise HTTPException(status_code=500, detail=f"Location prediction failed: {str(e)}")

@app.get("/regions/{crop}", tags=["Regions"])
async def get_crop_regions(crop: str):
    """Get regional information for a specific crop"""
    available_crops = multi_crop_predictor.get_available_crops()

    if crop not in available_crops:
        raise HTTPException(
            status_code=404,
            detail=f"Crop '{crop}' not found. Available crops: {available_crops}"
        )

    regions = multi_crop_predictor.get_crop_regions()
    crop_regions = regions.get(crop, {})

    return {
        "crop": crop,
        "primary_regions": crop_regions.get('primary', []),
        "secondary_regions": crop_regions.get('secondary', []),
        "regional_coverage": f"Covers {len(crop_regions.get('primary', []))} primary and {len(crop_regions.get('secondary', []))} secondary regions"
    }

# MVP CONSULTATION ENDPOINTS (Firebase + Railway Integration)
@app.post("/auth/farmer/register", tags=["MVP - Farmer Authentication"])
async def register_farmer(registration: FarmerRegistration):
    """Register a new farmer user with Firebase authentication and Railway database"""
    if not FIREBASE_AVAILABLE or not RAILWAY_AVAILABLE:
        raise HTTPException(status_code=503, detail="Authentication services temporarily unavailable")

    try:
        logger.info(f"👤 Registering farmer: {registration.phone_number}")

        # Create Firebase user
        firebase_result = firebase_manager.create_farmer_user(
            phone_number=registration.phone_number,
            display_name=registration.display_name,
            email=None  # Add email support later if needed
        )

        if not firebase_result['success']:
            raise HTTPException(status_code=400, detail=f"Firebase registration failed: {firebase_result['error']}")

        # Create database profile
        profile_data = {
            'firebase_uid': firebase_result['uid'],
            'phone_number': registration.phone_number,
            'display_name': registration.display_name,
            'location': registration.location,
            'farm_size': registration.farm_size,
            'district': registration.district,
            'state': registration.state,
            'pincode': registration.pincode,
            'preferred_crops': registration.preferred_crops
        }

        db_success = railway_db.create_farmer_profile(profile_data)

        if not db_success:
            # Rollback Firebase user if database fails
            logger.warning(f"⚠️ Rollback Firebase user {firebase_result['uid']} due to database failure")
            # In practice, would implement delete_user method

        return {
            "success": True,
            "farmer_id": firebase_result['uid'],
            "phone_number": registration.phone_number,
            "profile_created": db_success,
            "timestamp": datetime.now().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Farmer registration failed: {e}")
        raise HTTPException(status_code=500, detail=f"Registration failed: {str(e)}")

@app.post("/consultation/advice", response_model=ConsultationResponse, tags=["MVP - Agricultural Consultations"])
async def get_agricultural_advice(request: ConsultationRequest):
    """Get ethical agricultural advice using Gemini AI, saved to Firebase/Railway"""
    if not ETHICAL_ORCHESTRATOR_AVAILABLE:
        raise HTTPException(status_code=503, detail="AI consultation service temporarily unavailable")

    try:
        logger.info(f"🌾 Consultation request: {request.farmer_query[:50]}...")

        # Enrich farmer query with additional context
        enriched_query = {
            'farmer_query': request.farmer_query,
            'crop_type': request.crop_type,
            'location': request.location,
            'district': request.district,
            'state': request.state,
            'phone_number': request.phone_number,
            'firebase_token': request.firebase_token,
            'timestamp': datetime.now().isoformat(),
            'api_version': '2.0.0'
        }

        # Get farmer UID if token provided
        farmer_uid = None
        if request.firebase_token and FIREBASE_AVAILABLE:
            token_validation = firebase_manager.verify_firebase_token(request.firebase_token)
            if token_validation['valid']:
                farmer_uid = token_validation['uid']
                enriched_query['firebase_uid'] = farmer_uid

        # Orchestrate ethical agricultural advice
        consultation_result = ethical_orchestrator.orchestrate_ethical_agricultural_response(enriched_query)

        if not consultation_result or 'ethical_audit' not in consultation_result:
            raise HTTPException(status_code=500, detail="Failed to generate consultation")

        # Prepare response
        response = {
            'consultation_id': consultation_result.get('ethical_audit', {}).get('query_id', f"consult_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
            'farmer_query': request.farmer_query,
            'solution_recommended': _extract_solution_text(consultation_result),
            'ethical_tier_applied': consultation_result.get('ethical_tier_applied', 'unknown'),
            'confidence_score': consultation_result.get('confidence_score', 0.8),
            'safety_instructions': consultation_result.get('safety_instructions', {}).get('general_safety', []),
            'prevention_education': consultation_result.get('prevention_education', {}),
            'response_timestamp': datetime.now().isoformat(),
            'timestamp': datetime.now().isoformat(),
            'platform_version': '2.0.0'
        }

        # Save to databases if available
        if FIREBASE_AVAILABLE and farmer_uid:
            firebase_consultation = {
                'consultation_id': response['consultation_id'],
                'farmer_id': farmer_uid,
                'firebase_uid': farmer_uid,
                'query_text': request.farmer_query,
                'crop_type': request.crop_type,
                'solution_recommended': response['solution_recommended'][:1000],  # Limit size
                'ethical_tier_applied': response['ethical_tier_applied'],
                'confidence_score': response['confidence_score'],
                'response_status': 'completed',
                'created_at': datetime.now().isoformat()
            }
            firebase_manager.save_consultation(firebase_consultation)

        if RAILWAY_AVAILABLE:
            railway_consultation = {
                'consultation_id': response['consultation_id'],
                'farmer_id': farmer_uid or request.phone_number or 'anonymous',
                'firebase_uid': farmer_uid,
                'query_text': request.farmer_query,
                'crop_type': request.crop_type,
                'problem_description': request.farmer_query[:500],
                'solution_recommended': response['solution_recommended'][:1000],
                'ethical_tier_applied': response['ethical_tier_applied'],
                'confidence_score': response['confidence_score'],
                'response_status': 'completed'
            }
            railway_db.save_consultation(railway_consultation)

        # Log analytics
        if FIREBASE_AVAILABLE:
            firebase_manager.log_platform_analytics('consultation_completed', {
                'farmer_uid': farmer_uid or 'anonymous',
                'crop_type': request.crop_type,
                'ethical_tier': response['ethical_tier_applied'],
                'confidence_score': response['confidence_score']
            })

        logger.info(f"✅ Consultation completed: {response['consultation_id']}")
        return ConsultationResponse(**response)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Consultation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Consultation failed: {str(e)}")

@app.post("/feedback/submit", tags=["MVP - Feedback & Learning"])
async def submit_feedback(feedback: FeedbackRequest):
    """Submit farmer feedback for continuous learning"""
    if not FIREBASE_AVAILABLE or not RAILWAY_AVAILABLE:
        raise HTTPException(status_code=503, detail="Feedback service temporarily unavailable")

    try:
        logger.info(f"📝 Feedback received for consultation: {feedback.consultation_id}")

        # Save to Firebase
        firebase_feedback = {
            'consultation_id': feedback.consultation_id,
            'farmer_uid': feedback.farmer_uid,
            'feedback_text': feedback.feedback_text,
            'rating': feedback.rating,
            'helpful_boolean': feedback.helpful_boolean,
            'created_at': datetime.now().isoformat()
        }
        firebase_manager.save_feedback(firebase_feedback)

        # Save to Railway
        railway_feedback = {
            'consultation_id': feedback.consultation_id,
            'farmer_uid': feedback.farmer_uid,
            'feedback_text': feedback.feedback_text,
            'rating': feedback.rating,
            'helpful_boolean': feedback.helpful_boolean
        }
        railway_db.save_feedback(railway_feedback)

        # Log analytics
        firebase_manager.log_platform_analytics('feedback_submitted', {
            'consultation_id': feedback.consultation_id,
            'farmer_uid': feedback.farmer_uid,
            'rating': feedback.rating,
            'helpful': feedback.helpful_boolean
        })

        return {
            "success": True,
            "feedback_id": f"fb_{feedback.consultation_id}_{datetime.now().strftime('%H%M%S')}",
            "timestamp": datetime.now().isoformat(),
            "message": "Thank you for your feedback! This helps improve our agricultural recommendations."
        }

    except Exception as e:
        logger.error(f"❌ Feedback submission failed: {e}")
        raise HTTPException(status_code=500, detail=f"Feedback submission failed: {str(e)}")

@app.get("/farmer/profile/{firebase_uid}", response_model=FarmerProfileResponse, tags=["MVP - Farmer Profiles"])
async def get_farmer_profile(firebase_uid: str):
    """Get farmer profile and consultation history"""
    if not RAILWAY_AVAILABLE:
        raise HTTPException(status_code=503, detail="Profile service temporarily unavailable")

    try:
        logger.info(f"👤 Fetching profile for farmer: {firebase_uid}")

        # Get profile from Railway database
        profile = railway_db.get_farmer_profile(firebase_uid)
        if not profile:
            raise HTTPException(status_code=404, detail="Farmer profile not found")

        # Get consultation history
        consultations = railway_db.get_farmer_consultations(firebase_uid, limit=10)

        response = {
            'firebase_uid': firebase_uid,
            'phone_number': profile.get('phone_number', ''),
            'display_name': profile.get('display_name', ''),
            'consultation_count': len(consultations),
            'last_consultation': consultations[0].get('created_at') if consultations else None,
            'subscription_tier': profile.get('subscription_tier', 'free'),
            'created_at': profile.get('created_at', '')
        }

        return FarmerProfileResponse(**response)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Profile retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=f"Profile retrieval failed: {str(e)}")

@app.get("/health/mvp", tags=["Platform"])
async def mvp_health_check():
    """Comprehensive MVP health check including Firebase and Railway status"""
    firebase_status = firebase_manager.get_system_metrics() if FIREBASE_AVAILABLE else {"available": False}
    railway_status = railway_db.get_system_metrics() if RAILWAY_AVAILABLE else {"database_type": "Unavailable", "available": False}
    orchestrator_status = ETHICAL_ORCHESTRATOR_AVAILABLE

    return {
        "status": "healthy" if (firebase_status.get("available") or railway_status["available"]) else "limited",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "firebase_auth_database": firebase_status,
            "railway_postgresql": railway_status,
            "ethical_orchestrator": orchestrator_status,
            "multi_crop_predictor": True
        },
        "mvp_ready": True,
        "can_accept_consultations": ETHICAL_ORCHESTRATOR_AVAILABLE,
        "can_save_data": firebase_status.get("available") or railway_status["available"]
    }

@app.get("/stats", tags=["Platform"])
async def get_platform_stats():
    """Get platform performance and usage statistics"""
    return multi_crop_predictor.get_platform_stats()

def _extract_solution_text(consultation_result: Dict[str, Any]) -> str:
    """Extract readable solution text from consultation result"""
    try:
        # Extract from Gemini response if available
        if 'gemini_response' in consultation_result:
            response = consultation_result['gemini_response']
            if 'solution' in response:
                return response['solution'][:500]  # Limit length

        # Extract from ethical audit
        audit = consultation_result.get('ethical_audit', {})
        if 'recommendations' in audit:
            return audit['recommendations'][:500]

        # Default fallback
        return "Personalized agricultural recommendation provided. Please see safety instructions."

    except Exception:
        return "Agricultural consultation completed. See detailed response for recommendations."

# Error handlers
@app.exception_handler(500)
async def internal_server_error_handler(request, exc):
    """Handle internal server errors"""
    logger.error(f"Internal server error: {exc}")
    return HTTPException(status_code=500, detail="Internal server error")

@app.exception_handler(404)
async def not_found_handler(request, exc):
    """Handle 404 errors"""
    return HTTPException(status_code=404, detail="Endpoint not found")

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Application startup tasks"""
    logger.info("🚀 Starting India Agricultural Intelligence Platform API")
    logger.info(f"Available crops: {multi_crop_predictor.get_available_crops()}")

@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown tasks"""
    logger.info("🛑 Shutting down India Agricultural Intelligence Platform API")

# Main execution
if __name__ == "__main__":
    # Run with uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
