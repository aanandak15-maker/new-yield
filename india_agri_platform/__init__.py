"""
India Agricultural Intelligence Platform
Main package initialization with core modules
"""

# Core platform modules (required)
try:
    from .platform import (
        AgriculturalIntelligencePlatform,
        get_platform,
        platform,
        predict_yield,
        predict_crop_yield
    )
    __all__ = [
        'AgriculturalIntelligencePlatform',
        'get_platform',
        'platform',
        'predict_yield',
        'predict_crop_yield'
    ]
except ImportError as e:
    print(f"⚠️ Core platform initialization warning: {e}")
    __all__ = []

print("🇮🇳 India Agricultural Intelligence Platform core initialized")
