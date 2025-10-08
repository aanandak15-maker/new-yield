"""
Crop Configuration System for Multi-Crop Platform
"""

import json
from typing import Dict, Any, Optional, List
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class CropConfig:
    """Crop-specific configuration management"""

    def __init__(self, crop_name: str):
        self.crop_name = crop_name
        self.config = self._load_config()
        self._ensure_default_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load crop configuration from file"""
        config_file = Path(f"india_agri_platform/crops/{self.crop_name}/config.json")
        if config_file.exists():
            with open(config_file, 'r') as f:
                return json.load(f)
        return {}

    def _save_config(self):
        """Save crop configuration to file"""
        config_dir = Path(f"india_agri_platform/crops/{self.crop_name}")
        config_dir.mkdir(parents=True, exist_ok=True)
        config_file = config_dir / 'config.json'

        with open(config_file, 'w') as f:
            json.dump(self.config, f, indent=2)

    def _ensure_default_config(self):
        """Ensure default configuration values exist"""
        defaults = {
            "crop_name": self.crop_name,
            "scientific_name": "",
            "growth_duration_days": 120,
            "water_requirement_mm_per_day": 6.0,
            "growth_stages": {
                "initial": {"duration_days": 30, "kc": 0.3, "water_mm_day": 2.5},
                "development": {"duration_days": 40, "kc": 0.7, "water_mm_day": 4.0},
                "mid_season": {"duration_days": 40, "kc": 1.15, "water_mm_day": 6.5},
                "late_season": {"duration_days": 30, "kc": 0.6, "water_mm_day": 3.0}
            },
            "disease_susceptibility": {
                "rust": 0.5,
                "blight": 0.5,
                "mildew": 0.5,
                "aphids": 0.5,
                "termites": 0.5
            },
            "climate_sensitivity": {
                "temperature_optimal_c": 25,
                "temperature_range_c": [15, 35],
                "humidity_optimal_percent": 60,
                "rainfall_optimal_mm": 800
            },
            "soil_requirements": {
                "ph_range": [6.0, 7.5],
                "texture_preference": ["loam", "sandy_loam"],
                "organic_matter_min_percent": 1.0
            },
            "yield_parameters": {
                "max_yield_quintal_ha": 50,
                "yield_stability_index": 0.8,
                "response_to_fertilizer": 0.7
            },
            "season": "rabi",  # rabi, kharif, or zaid
            "planting_months": [10, 11, 12],  # October to December
            "harvest_months": [3, 4, 5],      # March to May
            "varieties": [],
            "feature_weights": {
                "ndvi": 0.25,
                "rainfall": 0.20,
                "temperature": 0.15,
                "soil_ph": 0.10,
                "irrigation_coverage": 0.10,
                "variety_score": 0.20
            }
        }

        # Update config with defaults for missing keys
        for key, value in defaults.items():
            if key not in self.config:
                self.config[key] = value

        # Save updated config
        self._save_config()

    def get_growth_stage(self, days_after_sowing: int) -> str:
        """Get current growth stage based on days after sowing"""
        stages = self.config.get('growth_stages', {})
        cumulative_days = 0

        for stage_name, stage_info in stages.items():
            stage_duration = stage_info.get('duration_days', 30)
            if days_after_sowing <= cumulative_days + stage_duration:
                return stage_name
            cumulative_days += stage_duration

        return 'late_season'  # Default to late season

    def get_water_requirement(self, growth_stage: str = None) -> float:
        """Get water requirement for growth stage"""
        if growth_stage is None:
            return self.config.get('water_requirement_mm_per_day', 6.0)

        stages = self.config.get('growth_stages', {})
        if growth_stage in stages:
            return stages[growth_stage].get('water_mm_day', 6.0)

        return self.config.get('water_requirement_mm_per_day', 6.0)

    def get_crop_coefficient(self, growth_stage: str) -> float:
        """Get crop coefficient (Kc) for growth stage"""
        stages = self.config.get('growth_stages', {})
        if growth_stage in stages:
            return stages[growth_stage].get('kc', 1.0)
        return 1.0

    def get_disease_risk(self, disease_type: str) -> float:
        """Get disease susceptibility score (0-1, higher = more susceptible)"""
        diseases = self.config.get('disease_susceptibility', {})
        return diseases.get(disease_type, 0.5)

    def get_climate_suitability(self, temperature_c: float, humidity_percent: float,
                              rainfall_mm: float) -> float:
        """Calculate climate suitability score (0-1)"""
        climate = self.config.get('climate_sensitivity', {})

        # Temperature suitability
        temp_optimal = climate.get('temperature_optimal_c', 25)
        temp_range = climate.get('temperature_range_c', [15, 35])
        temp_score = 1.0 if temp_range[0] <= temperature_c <= temp_range[1] else 0.5

        # Humidity suitability
        humidity_optimal = climate.get('humidity_optimal_percent', 60)
        humidity_score = 1.0 - abs(humidity_percent - humidity_optimal) / 100.0
        humidity_score = max(0.1, min(1.0, humidity_score))

        # Rainfall suitability (simplified)
        rainfall_optimal = climate.get('rainfall_optimal_mm', 800)
        rainfall_score = 1.0 - abs(rainfall_mm - rainfall_optimal) / rainfall_optimal
        rainfall_score = max(0.1, min(1.0, rainfall_score))

        # Combined score
        return (temp_score * 0.4 + humidity_score * 0.3 + rainfall_score * 0.3)

    def get_soil_suitability(self, ph: float, texture: str, organic_matter_percent: float) -> float:
        """Calculate soil suitability score (0-1)"""
        soil_req = self.config.get('soil_requirements', {})

        # pH suitability
        ph_range = soil_req.get('ph_range', [6.0, 7.5])
        ph_score = 1.0 if ph_range[0] <= ph <= ph_range[1] else 0.5

        # Texture suitability
        preferred_textures = soil_req.get('texture_preference', ['loam'])
        texture_score = 1.0 if texture.lower() in [t.lower() for t in preferred_textures] else 0.7

        # Organic matter suitability
        om_min = soil_req.get('organic_matter_min_percent', 1.0)
        om_score = 1.0 if organic_matter_percent >= om_min else (organic_matter_percent / om_min)

        return (ph_score * 0.4 + texture_score * 0.4 + om_score * 0.2)

    def get_feature_weight(self, feature_name: str) -> float:
        """Get feature weight for model"""
        weights = self.config.get('feature_weights', {})
        return weights.get(feature_name, 0.1)

    def update_config(self, updates: Dict[str, Any]):
        """Update crop configuration"""
        self.config.update(updates)
        self._save_config()
        logger.info(f"Updated configuration for crop: {self.crop_name}")

    def get_varieties(self) -> List[str]:
        """Get list of available varieties"""
        return self.config.get('varieties', [])

    def add_variety(self, variety_name: str, characteristics: Dict[str, Any]):
        """Add a new variety to the crop"""
        varieties = self.config.get('varieties', [])
        if variety_name not in varieties:
            varieties.append(variety_name)
            self.config['varieties'] = varieties

            # Store variety characteristics
            if 'variety_characteristics' not in self.config:
                self.config['variety_characteristics'] = {}
            self.config['variety_characteristics'][variety_name] = characteristics

            self._save_config()
            logger.info(f"Added variety {variety_name} to crop {self.crop_name}")

    def get_season_info(self) -> Dict[str, Any]:
        """Get seasonal information for the crop"""
        return {
            'season': self.config.get('season', 'rabi'),
            'planting_months': self.config.get('planting_months', []),
            'harvest_months': self.config.get('harvest_months', [])
        }

# Pre-configured crop configurations
DEFAULT_CROP_CONFIGS = {
    "wheat": {
        "scientific_name": "Triticum aestivum",
        "growth_duration_days": 155,
        "water_requirement_mm_per_day": 6.5,
        "season": "rabi",
        "planting_months": [10, 11, 12],
        "harvest_months": [3, 4, 5],
        "yield_parameters": {
            "max_yield_quintal_ha": 55,
            "yield_stability_index": 0.85
        }
    },

    "rice": {
        "scientific_name": "Oryza sativa",
        "growth_duration_days": 140,
        "water_requirement_mm_per_day": 8.0,
        "season": "kharif",
        "planting_months": [6, 7, 8],
        "harvest_months": [10, 11, 12],
        "yield_parameters": {
            "max_yield_quintal_ha": 60,
            "yield_stability_index": 0.75
        }
    },

    "maize": {
        "scientific_name": "Zea mays",
        "growth_duration_days": 120,
        "water_requirement_mm_per_day": 7.0,
        "season": "kharif",
        "planting_months": [6, 7],
        "harvest_months": [9, 10, 11],
        "yield_parameters": {
            "max_yield_quintal_ha": 80,
            "yield_stability_index": 0.70
        }
    },

    "cotton": {
        "scientific_name": "Gossypium spp.",
        "growth_duration_days": 180,
        "water_requirement_mm_per_day": 6.0,
        "season": "kharif",
        "planting_months": [5, 6],
        "harvest_months": [11, 12, 1],
        "yield_parameters": {
            "max_yield_quintal_ha": 25,
            "yield_stability_index": 0.65
        }
    },

    "sugarcane": {
        "scientific_name": "Saccharum officinarum",
        "growth_duration_days": 365,
        "water_requirement_mm_per_day": 7.5,
        "season": "annual",
        "planting_months": [2, 3, 9, 10],
        "harvest_months": [11, 12, 1, 2, 3, 4],
        "yield_parameters": {
            "max_yield_quintal_ha": 800,
            "yield_stability_index": 0.80
        }
    }
}

def create_crop_config(crop_name: str) -> CropConfig:
    """Factory function to create crop configuration"""
    config = CropConfig(crop_name)

    # Apply default configuration if available
    if crop_name in DEFAULT_CROP_CONFIGS:
        config.update_config(DEFAULT_CROP_CONFIGS[crop_name])

    return config

def get_available_crops() -> List[str]:
    """Get list of available crops"""
    crops_dir = Path("india_agri_platform/crops")
    if crops_dir.exists():
        return [d.name for d in crops_dir.iterdir() if d.is_dir()]
    return []
