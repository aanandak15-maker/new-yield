"""
Multi-Crop Agricultural Intelligence Predictor
Unified API for Rice, Wheat, Cotton yield predictions with intelligent routing
"""

import sys
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import logging
from datetime import datetime
import numpy as np

# Add current directory to path for imports
sys.path.append('.')

# Lazy load individual crop predictors (fail-safe initialization)
rice_available = None
wheat_available = None
cotton_available = None

def _load_rice_predictor():
    global rice_available
    if rice_available is None:  # Not loaded yet
        try:
            from india_agri_platform.crops.rice.model import get_rice_predictor, rice_predictor
            rice_available = rice_predictor
        except (ImportError, Exception) as e:
            rice_available = False
            logging.warning(f"Rice predictor not available: {e}")
    return rice_available

def _load_wheat_predictor():
    global wheat_available
    if wheat_available is None:  # Not loaded yet
        try:
            from india_agri_platform.crops.wheat.model import wheat_predictor
            wheat_available = wheat_predictor
        except (ImportError, Exception) as e:
            wheat_available = False
            logging.warning(f"Wheat predictor not available: {e}")
    return wheat_available

def _load_cotton_predictor():
    global cotton_available
    if cotton_available is None:  # Not loaded yet
        try:
            from india_agri_platform.crops.cotton.model import get_cotton_predictor, cotton_predictor
            cotton_available = cotton_predictor
        except (ImportError, Exception) as e:
            cotton_available = False
            logging.warning(f"Cotton predictor not available: {e}")
    return cotton_available

from india_agri_platform.core.error_handling import error_handler

logger = logging.getLogger(__name__)

class FallbackPredictor:
    """Fallback predictor that works when real ML models are unavailable"""

    def predict_rice_yield(self, features: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "crop": "rice",
            "variety": "Fallback Estimate",
            "state": "Punjab",  # Default assume
            "predicted_yield_quintal_ha": 45.0 + np.random.uniform(-10, 10),  # Baseline with some variance
            "unit": "quintal per hectare",
            "confidence_level": "estimation",
            "insights": ["Limited ML functionality - using fallback estimates"],
            "timestamp": datetime.utcnow().isoformat(),
            "prediction_method": "fallback_baseline",
            "error": "ML models temporarily unavailable - using statistical estimates"
        }

    def predict_yield(self, lat: float, lng: float, adapted_params: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "crop": "wheat",
            "variety": "Fallback Estimate",
            "state": "Punjab",
            "predicted_yield_quintal_ha": 38.0 + np.random.uniform(-8, 8),
            "unit": "quintal per hectare",
            "confidence_level": "estimation",
            "insights": ["Limited ML functionality - using fallback estimates"],
            "timestamp": datetime.utcnow().isoformat(),
            "prediction_method": "fallback_baseline",
            "error": "ML models temporarily unavailable - using statistical estimates"
        }

    def predict_cotton_yield(self, features: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "crop": "cotton",
            "variety": "Fallback Estimate",
            "state": "Punjab",
            "predicted_yield_quintal_ha": 25.0 + np.random.uniform(-5, 5),
            "unit": "quintal per hectare",
            "confidence_level": "estimation",
            "insights": ["Limited ML functionality - using fallback estimates"],
            "timestamp": datetime.utcnow().isoformat(),
            "prediction_method": "fallback_baseline",
            "error": "ML models temporarily unavailable - using statistical estimates"
        }

class MultiCropPredictor:
    """
    Unified Agricultural Intelligence Platform for Multi-Crop Yield Prediction

    Intelligent routing to appropriate crop predictors based on:
    - Geographic location (coordinates)
    - Crop type specification
    - Agricultural season and regional patterns
    - Environmental conditions
    """

    def __init__(self):
        self.predictors = {}
        self.crop_regions = {}
        self.seasonal_patterns = {}

        # Initialize available predictors
        self._initialize_predictors()

        # Define crop specific regions and seasonal patterns
        self._setup_crop_intelligence()

        logger.info("🎯 Multi-Crop Agricultural Intelligence Platform initialized")
        logger.info(f"Available crops: {list(self.predictors.keys())}")

    def _initialize_predictors(self):
        """Initialize all available crop predictors with fallbacks"""
        predictors_initialized = 0

        # Load each predictor safely
        try:
            rice_pred = _load_rice_predictor()
            if rice_pred:
                self.predictors['rice'] = rice_pred
                predictors_initialized += 1
                logger.info("✅ Rice predictor loaded")
        except Exception as e:
            logger.warning(f"❌ Rice predictor initialization failed: {e}")

        try:
            wheat_pred = _load_wheat_predictor()
            if wheat_pred:
                self.predictors['wheat'] = wheat_pred
                predictors_initialized += 1
                logger.info("✅ Wheat predictor loaded")
        except Exception as e:
            logger.warning(f"❌ Wheat predictor initialization failed: {e}")

        try:
            cotton_pred = _load_cotton_predictor()
            if cotton_pred:
                self.predictors['cotton'] = cotton_pred
                predictors_initialized += 1
                logger.info("✅ Cotton predictor loaded")
        except Exception as e:
            logger.warning(f"❌ Cotton predictor initialization failed: {e}")

        if predictors_initialized == 0:
            logger.warning("❌ No crop predictors could be loaded - using fallback predictors")
            # Use fallback predictors when real ones aren't available
            fallback = FallbackPredictor()
            self.predictors = {
                'rice': fallback,
                'wheat': fallback,
                'cotton': fallback
            }
            predictors_initialized = 3  # All fallbacks available
        else:
            logger.info(f"✅ Multi-crop predictor initialized with {predictors_initialized} crops")

    def _setup_crop_intelligence(self):
        """Setup intelligent crop routing based on geography and seasons"""

        # Define primary crop regions in India (approximate latitude/longitude bounds)
        self.crop_regions = {
            'rice': {
                'primary': [
                    {'name': 'Punjab Rice Belt', 'lat_min': 29.5, 'lat_max': 32.5, 'lng_min': 73.8, 'lng_max': 76.9, 'states': ['PUNJAB']},
                    {'name': 'UP Rice Belt', 'lat_min': 25.0, 'lat_max': 28.0, 'lng_min': 80.0, 'lng_max': 85.0, 'states': ['UTTAR_PRADESH']},
                    {'name': 'Bihar Rice Bowl', 'lat_min': 24.2, 'lat_max': 27.3, 'lng_min': 83.0, 'lng_max': 88.2, 'states': ['BIHAR']},
                    {'name': 'West Bengal Delta', 'lat_min': 21.5, 'lat_max': 25.0, 'lng_min': 85.8, 'lng_max': 89.1, 'states': ['WEST_BENGAL']},
                    {'name': 'Andhra Pradesh', 'lat_min': 14.0, 'lat_max': 19.1, 'lng_min': 76.7, 'lng_max': 84.8, 'states': ['ANDHRA_PRADESH']},
                    {'name': 'Tamil Nadu', 'lat_min': 8.1, 'lat_max': 13.1, 'lng_min': 76.2, 'lng_max': 80.4, 'states': ['TAMIL_NADU']},
                ],
                'secondary': [
                    {'name': 'Karnataka Rice', 'lat_min': 11.5, 'lat_max': 18.5, 'lng_min': 74.0, 'lng_max': 78.6, 'states': ['KARNATAKA']},
                    {'name': 'Odisha Rice', 'lat_min': 17.8, 'lat_max': 22.1, 'lng_min': 81.4, 'lng_max': 87.6, 'states': ['ODISHA']},
                ]
            },
            'wheat': {
                'primary': [
                    {'name': 'Punjab Wheat Bowl', 'lat_min': 29.5, 'lat_max': 32.5, 'lng_min': 73.8, 'lng_max': 76.9, 'states': ['PUNJAB']},
                    {'name': 'Haryana Wheat Belt', 'lat_min': 27.5, 'lat_max': 30.9, 'lng_min': 74.4, 'lng_max': 77.6, 'states': ['HARYANA']},
                    {'name': 'UP Wheat Region', 'lat_min': 25.0, 'lat_max': 30.0, 'lng_min': 77.0, 'lng_max': 84.0, 'states': ['UTTAR_PRADESH']},
                    {'name': 'MP Wheat Zone', 'lat_min': 21.0, 'lat_max': 26.0, 'lng_min': 74.0, 'lng_max': 82.0, 'states': ['MADHYA_PRADESH']},
                ],
                'secondary': [
                    {'name': 'Rajasthan Wheat', 'lat_min': 24.0, 'lat_max': 30.0, 'lng_min': 69.0, 'lng_max': 78.0, 'states': ['RAJASTHAN']},
                    {'name': 'Bihar Wheat', 'lat_min': 24.0, 'lat_max': 27.0, 'lng_min': 83.0, 'lng_max': 88.0, 'states': ['BIHAR']},
                ]
            },
            'cotton': {
                'primary': [
                    {'name': 'Maharashtra Cotton', 'lat_min': 15.6, 'lat_max': 22.0, 'lng_min': 72.6, 'lng_max': 80.9, 'states': ['MAHARASHTRA']},
                    {'name': 'Gujarat Cotton', 'lat_min': 20.1, 'lat_max': 24.7, 'lng_min': 68.1, 'lng_max': 74.5, 'states': ['GUJARAT']},
                    {'name': 'Punjab Cotton', 'lat_min': 29.5, 'lat_max': 32.5, 'lng_min': 73.8, 'lng_max': 76.9, 'states': ['PUNJAB']},
                    {'name': 'Haryana Cotton', 'lat_min': 27.5, 'lat_max': 30.9, 'lng_min': 74.4, 'lng_max': 77.6, 'states': ['HARYANA']},
                ],
                'secondary': [
                    {'name': 'Andhra Pradesh Cotton', 'lat_min': 14.0, 'lat_max': 19.0, 'lng_min': 76.7, 'lng_max': 84.8, 'states': ['ANDHRA_PRADESH']},
                    {'name': 'Tamil Nadu Cotton', 'lat_min': 8.1, 'lat_max': 13.1, 'lng_min': 76.2, 'lng_max': 80.4, 'states': ['TAMIL_NADU']},
                ]
            }
        }

        # Define seasonal patterns for intelligent crop detection
        self.seasonal_patterns = {
            'kharif': ['rice', 'cotton'],  # June-October planting
            'rabi': ['wheat'],              # November-March planting
            'summer': ['rice', 'cotton']    # March-May planting (limited)
        }

    def predict_yield(self, crop: Optional[str] = None, location: Optional[str] = None,
                     latitude: Optional[float] = None, longitude: Optional[float] = None,
                     **kwargs) -> Dict[str, Any]:
        """
        Unified crop yield prediction with intelligent routing

        Args:
            crop: Specific crop to predict ('rice', 'wheat', 'cotton') or None for auto-detection
            location: Location string for context
            latitude: GPS latitude
            longitude: GPS longitude
            **kwargs: Additional prediction parameters

        Returns:
            Dictionary containing prediction results and insights
        """

        try:
            logger.info(f"🌾 Processing prediction request: crop={crop}, lat={latitude}, lng={longitude}")

            # Determine crop if not specified
            if not crop:
                crop = self._detect_crop_from_location(latitude, longitude, kwargs.get('season'))
                if not crop:
                    return {
                        "error": "Could not determine crop type from location. Please specify crop parameter.",
                        "available_crops": list(self.predictors.keys()),
                        "suggestion": "Try specifying crop='rice', 'wheat', or 'cotton'"
                    }

            # Validate crop availability
            if crop not in self.predictors:
                available_crops = list(self.predictors.keys())
                return {
                    "error": f"Crop '{crop}' not available. Available crops: {available_crops}",
                    "available_crops": available_crops
                }

            # Prepare prediction parameters
            prediction_params = self._prepare_prediction_params(crop, latitude, longitude, kwargs)

            # Route to appropriate crop predictor
            predictor = self.predictors[crop]

            # Make prediction based on crop type
            if crop == 'rice':
                if hasattr(predictor, 'predict_rice_yield'):
                    result = predictor.predict_rice_yield(prediction_params)
                else:
                    return {"error": f"Invalid rice predictor method"}
            elif crop == 'wheat':
                if hasattr(predictor, 'predict_yield'):
                    # For wheat, we may need to adapt parameters
                    adapted_params = self._adapt_wheat_parameters(prediction_params)
                    result = predictor.predict_yield(prediction_params['latitude'], prediction_params['longitude'], adapted_params)
                else:
                    return {"error": f"Invalid wheat predictor method"}
            elif crop == 'cotton':
                if hasattr(predictor, 'predict_cotton_yield'):
                    result = predictor.predict_cotton_yield(prediction_params)
                else:
                    return {"error": f"Invalid cotton predictor method"}
            else:
                return {"error": f"Unsupported crop: {crop}"}

            # Add multi-crop intelligence
            if 'error' not in result:
                result = self._enhance_with_multi_crop_intelligence(result, crop, latitude, longitude)

            return result

        except Exception as e:
            error_result = {
                "error": f"Multi-crop prediction failed: {str(e)}",
                "crop_requested": crop,
                "location": f"{latitude},{longitude}" if latitude and longitude else None
            }
            error_handler.handle_error(e, {"operation": "multi_crop_prediction", "crop": crop})
            return error_result

    def _detect_crop_from_location(self, latitude: Optional[float], longitude: Optional[float],
                                 season: Optional[str] = None) -> Optional[str]:
        """Intelligent crop detection based on location and season"""

        if not latitude or not longitude:
            return None

        # Determine season if available, otherwise use default logic
        season = season.lower() if season else 'kharif'  # Default to kharif (most common)

        # Score each crop based on regional prevalence and seasonal patterns
        crop_scores = {}

        for crop_name, regions in self.crop_regions.items():
            if crop_name not in self.predictors:
                continue

            score = 0

            # Primary regions get higher score
            for region in regions.get('primary', []):
                if (region['lat_min'] <= latitude <= region['lat_max'] and
                    region['lng_min'] <= longitude <= region['lng_max']):
                    score += 3  # High weight for primary region

            # Secondary regions get medium score
            for region in regions.get('secondary', []):
                if (region['lat_min'] <= latitude <= region['lat_max'] and
                    region['lng_min'] <= longitude <= region['lng_max']):
                    score += 1  # Medium weight for secondary region

            # Seasonal weighting
            if season in self.seasonal_patterns and crop_name in self.seasonal_patterns[season]:
                score += 2  # Bonus for season appropriateness

            crop_scores[crop_name] = score

        # Return crop with highest score (must be > 0)
        if crop_scores:
            best_crop = max(crop_scores.items(), key=lambda x: x[1])
            if best_crop[1] > 0:
                logger.info(f"🎯 Auto-detected crop: {best_crop[0]} (score: {best_crop[1]})")
                return best_crop[0]

        return None

    def _prepare_prediction_params(self, crop: str, latitude: float, longitude: float,
                                 kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare standardized parameters for crop prediction"""

        # Base parameters required by all predictors
        params = {
            'latitude': latitude,
            'longitude': longitude,
            'sowing_date': kwargs.get('sowing_date', datetime.now().strftime('%Y-%m-%d')),
        }

        # Add crop-specific parameters
        if crop == 'rice':
            params.update({
                'variety_name': kwargs.get('variety_name'),
                'temperature_celsius': kwargs.get('temperature_celsius', 28),
                'rainfall_mm': kwargs.get('rainfall_mm', 800),
                'humidity_percent': kwargs.get('humidity_percent', 70),
                'irrigation_coverage': kwargs.get('irrigation_coverage', 0.7),
                'soil_ph': kwargs.get('soil_ph', 7.0)
            })
        elif crop == 'wheat':
            params.update({
                'temperature_celsius': kwargs.get('temperature_celsius', 20),
                'rainfall_mm': kwargs.get('rainfall_mm', 300),
                'humidity_percent': kwargs.get('humidity_percent', 60),
                'ndvi': kwargs.get('ndvi', 0.6),
                'irrigation_coverage': kwargs.get('irrigation_coverage', 0.8),
                'soil_ph': kwargs.get('soil_ph', 7.2)
            })
        elif crop == 'cotton':
            params.update({
                'variety_name': kwargs.get('variety_name', 'F-1861'),
                'temperature_celsius': kwargs.get('temperature_celsius', 30),
                'rainfall_mm': kwargs.get('rainfall_mm', 600),
                'humidity_percent': kwargs.get('humidity_percent', 65),
                'irrigation_coverage': kwargs.get('irrigation_coverage', 0.8),
                'soil_ph': kwargs.get('soil_ph', 7.5),
                'area_hectares': kwargs.get('area_hectares', 2.0)
            })

        return params

    def _adapt_wheat_parameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Adapt parameters for wheat predictor format"""
        return {
            'temperature_celsius': params.get('temperature_celsius', 20),
            'rainfall_mm': params.get('rainfall_mm', 300),
            'humidity_percent': params.get('humidity_percent', 60),
            'ndvi': params.get('ndvi', 0.6),
            'irrigation_coverage': params.get('irrigation_coverage', 0.8),
            'soil_ph': params.get('soil_ph', 7.2)
        }

    def _enhance_with_multi_crop_intelligence(self, result: Dict[str, Any], crop: str,
                                            latitude: float, longitude: float) -> Dict[str, Any]:
        """Add multi-crop agricultural intelligence to results"""

        # Add regional context
        region_info = self._get_region_context(crop, latitude, longitude)
        if region_info:
            result['regional_context'] = region_info

        # Add crop rotation insights
        rotation_advice = self._get_crop_rotation_advice(crop, latitude, longitude)
        if rotation_advice:
            result['crop_rotation_suggestions'] = rotation_advice

        # Add alternative crops
        alternatives = self._suggest_alternative_crops(crop, latitude, longitude)
        if alternatives:
            result['alternative_crops'] = alternatives

        # Add season intelligence
        season_info = self._get_season_intelligence(crop, latitude, longitude)
        if season_info:
            result['seasonal_intelligence'] = season_info

        return result

    def _get_region_context(self, crop: str, latitude: float, longitude: float) -> Optional[str]:
        """Get regional farming context for the location"""
        for region_type in ['primary', 'secondary']:
            for region in self.crop_regions.get(crop, {}).get(region_type, []):
                if (region['lat_min'] <= latitude <= region['lat_max'] and
                    region['lng_min'] <= longitude <= region['lng_max']):
                    return f"{region['name']} - {region_type.title()} {crop.title()} growing region in {', '.join(region['states'])}"

        return None

    def _get_crop_rotation_advice(self, crop: str, latitude: float, longitude: float) -> List[str]:
        """Suggest beneficial crop rotations"""
        rotation_suggestions = []

        if crop == 'rice':
            rotation_suggestions.extend([
                "Rice-Wheat rotation: Excellent for Punjab/Haryana (maintains soil fertility)",
                "Rice-Pulse rotation: Good for Bihar (fixes nitrogen, breaks pest cycle)",
                "Rice-Oilseed rotation: Recommended for UP (diversifies income, improves soil health)"
            ])
        elif crop == 'wheat':
            rotation_suggestions.extend([
                "Wheat-Rice rotation: Traditional and profitable in North India",
                "Wheat-Maize rotation: Good for diversification and pest management",
                "Wheat-Pulse rotation: Sustainable for soil health and nutrition"
            ])
        elif crop == 'cotton':
            rotation_suggestions.extend([
                "Cotton-Soybean rotation: Breaks pest cycle, improves soil fertility",
                "Cotton-Groundnut rotation: Good for deep soil aeration",
                "Cotton-Pulse rotation: Reduces bollworm infestation risks"
            ])

        return rotation_suggestions[:3]  # Return top 3 suggestions

    def _suggest_alternative_crops(self, crop: str, latitude: float, longitude: float) -> List[Dict[str, Any]]:
        """Suggest alternative crops for the region"""
        alternatives = []

        # Location-based alternatives (example logic)
        if crop == 'rice':
            alternatives.extend([
                {"crop": "maize", "reason": "Higher drought tolerance, less water requirement"},
                {"crop": "wheat", "reason": "Rabi season alternative with good profitability"}
            ])
        elif crop == 'wheat':
            alternatives.extend([
                {"crop": "rice", "reason": "Kharif season alternative, higher yields"},
                {"crop": "maize", "reason": "Similar conditions, potentially higher returns"}
            ])
        elif crop == 'cotton':
            alternatives.extend([
                {"crop": "rice", "reason": "Better irrigation suitability in wet regions"},
                {"crop": "wheat", "reason": "Rabi alternative with more stable pricing"}
            ])

        return alternatives[:2]

    def _get_season_intelligence(self, crop: str, latitude: float, longitude: float) -> Optional[str]:
        """Provide seasonal farming intelligence"""
        if latitude > 20:  # Northern India
            if crop == 'rice':
                return "Kharif season recommended - June-July sowing for optimal monsoon utilization"
            elif crop == 'wheat':
                return "Rabi season - November-December sowing for winter crop cycle"
            elif crop == 'cotton':
                return "Kharif season preferred - May-June sowing avoids excessive heat stress"
        else:  # Southern India
            if crop == 'rice':
                return "Multiple seasons possible - Kharif (June-Sep) and Rabi (Oct-Jan)"
            elif crop == 'wheat':
                return "Winter season - November-December sowing in southern regions"
            elif crop == 'cotton':
                return "Kharif season dominant - June-July sowing for rainfed cultivation"

        return None

    def get_available_crops(self) -> List[str]:
        """Get list of available crops for prediction"""
        return list(self.predictors.keys())

    def get_crop_regions(self) -> Dict[str, Any]:
        """Get crop regional information for API documentation"""
        return self.crop_regions

    def get_platform_stats(self) -> Dict[str, Any]:
        """Get platform statistics"""
        return {
            "total_crops": len(self.predictors),
            "crops_available": list(self.predictors.keys()),
            "intelligent_routing": True,
            "auto_crop_detection": True,
            "regional_coverage": "Major Indian agricultural regions",
            "seasonal_adaptability": True
        }

# Global multi-crop predictor instance
multi_crop_predictor = MultiCropPredictor()

def get_multi_crop_predictor() -> MultiCropPredictor:
    """Get global multi-crop predictor instance"""
    return multi_crop_predictor

def predict_yield(crop: Optional[str] = None, location: Optional[str] = None,
                 latitude: Optional[float] = None, longitude: Optional[float] = None,
                 **kwargs) -> Dict[str, Any]:
    """
    Convenience function for quick predictions

    Examples:
        # Auto-detect crop from location
        result = predict_yield(latitude=30.5, longitude=75.5)

        # Specific crop prediction
        result = predict_yield(crop='rice', latitude=30.5, longitude=75.5,
                              temperature_celsius=28, rainfall_mm=800)

        # Location-based prediction
        result = predict_yield(crop='cotton', latitude=19.5, longitude=75.5)
    """
    return multi_crop_predictor.predict_yield(crop, location, latitude, longitude, **kwargs)

# Platform API functions for external integration
def get_platform_info() -> Dict[str, Any]:
    """Get platform capabilities and available features"""
    return {
        "platform_name": "India Agricultural Intelligence Platform",
        "version": "2.0 - Multi-Crop Enterprise",
        "capabilities": {
            "crops_supported": multi_crop_predictor.get_available_crops(),
            "intelligent_routing": True,
            "auto_crop_detection": True,
            "regional_adaptability": True,
            "seasonal_intelligence": True,
            "multi_crop_insights": True
        },
        "api_endpoints": [
            "predict_yield(crop, latitude, longitude, **params)",
            "get_available_crops()",
            "get_platform_stats()"
        ],
        "supported_parameters": [
            "temperature_celsius", "rainfall_mm", "humidity_percent",
            "irrigation_coverage", "soil_ph", "variety_name", "season"
        ]
    }
