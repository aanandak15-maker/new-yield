"""
India Agricultural Intelligence Platform
Unified API for multi-crop yield prediction and agricultural insights
"""

from typing import Dict, List, Any, Optional, Union
from india_agri_platform.core.multi_crop_predictor import (
    MultiCropPredictor,
    get_multi_crop_predictor,
    predict_yield
)

# Optional imports - handle gracefully if not available
try:
    from india_agri_platform.core.streamlined_predictor import (
        StreamlinedPredictor
    )
    streamlined_available = True
except ImportError:
    streamlined_available = False
    StreamlinedPredictor = None
    print("⚠️ Streamlined predictor not available - continuing without it")

# Create platform instance
_platform = None

def get_platform():
    """Get the global agricultural intelligence platform instance"""
    global _platform
    if _platform is None:
        _platform = AgriculturalIntelligencePlatform()
    return _platform

class AgriculturalIntelligencePlatform:
    """
    Unified Agricultural Intelligence Platform

    Provides high-level API for crop prediction and agricultural insights
    """

    def __init__(self):
        self.multi_crop_predictor = get_multi_crop_predictor()
        # streamlined_predictor is optional and handled by multi_crop_predictor
        print("🇮🇳 Agricultural Intelligence Platform initialized")

    def predict_yield(self, crop: Optional[str] = None, location: str = None,
                     latitude: Optional[float] = None, longitude: Optional[float] = None,
                     **kwargs) -> Dict[str, Any]:
        """
        Unified crop yield prediction

        Args:
            crop: Crop type ('rice', 'wheat', 'cotton') or None for auto-detection
            location: Location name
            latitude: GPS latitude
            longitude: GPS longitude
            **kwargs: Additional prediction parameters

        Returns:
            Prediction results with insights
        """

        # Use multi-crop predictor for intelligent routing
        return self.multi_crop_predictor.predict_yield(
            crop=crop, location=location,
            latitude=latitude, longitude=longitude,
            **kwargs
        )

    def predict_rice_yield(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Rice-specific yield prediction"""
        return self.multi_crop_predictor.predict_yield(crop='rice', **features)

    def predict_wheat_yield(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Wheat-specific yield prediction"""
        return self.multi_crop_predictor.predict_yield(crop='wheat', **features)

    def predict_cotton_yield(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Cotton-specific yield prediction"""
        return self.multi_crop_predictor.predict_yield(crop='cotton', **features)

    def get_available_crops(self) -> List[str]:
        """Get list of crops available for prediction"""
        return self.multi_crop_predictor.get_available_crops()

    def get_platform_stats(self) -> Dict[str, Any]:
        """Get platform statistics and capabilities"""
        return self.multi_crop_predictor.get_platform_stats()

    def get_crop_regions(self) -> Dict[str, Any]:
        """Get regional crop information"""
        return self.multi_crop_predictor.get_crop_regions()

# Convenience functions for direct access
def predict_crop_yield(crop: str, latitude: float, longitude: float, **kwargs) -> Dict[str, Any]:
    """Convenience function for crop yield prediction"""
    platform = get_platform()
    return platform.predict_yield(crop=crop, latitude=latitude, longitude=longitude, **kwargs)

# Global platform instance
platform = get_platform()

# Export key functions for external use
__all__ = [
    'AgriculturalIntelligencePlatform',
    'get_platform',
    'platform',
    'predict_yield',
    'predict_crop_yield'
]
