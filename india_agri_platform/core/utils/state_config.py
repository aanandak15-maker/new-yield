"""
State Configuration System for Multi-State Platform
"""

import json
from typing import Dict, Any, Optional, List
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class StateConfig:
    """State-specific configuration management"""

    def __init__(self, state_name: str):
        self.state_name = state_name
        self.config = self._load_config()
        self._ensure_default_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load state configuration from file"""
        config_file = Path(f"india_agri_platform/states/{self.state_name}/config.json")
        if config_file.exists():
            with open(config_file, 'r') as f:
                return json.load(f)
        return {}

    def _save_config(self):
        """Save state configuration to file"""
        config_dir = Path(f"india_agri_platform/states/{self.state_name}")
        config_dir.mkdir(parents=True, exist_ok=True)
        config_file = config_dir / 'config.json'

        with open(config_file, 'w') as f:
            json.dump(self.config, f, indent=2)

    def _ensure_default_config(self):
        """Ensure default configuration values exist"""
        defaults = {
            "state_name": self.state_name,
            "full_name": self.state_name.replace('_', ' ').title(),
            "total_area_hectares": 1000000,
            "agricultural_area_hectares": 600000,
            "districts": [],
            "agro_climatic_zones": [],
            "major_crops": [],
            "irrigation": {
                "coverage_percent": 70,
                "methods": ["canal", "tubewell", "rainfed"],
                "efficiency": 0.65
            },
            "soil_types": ["alluvial", "sandy_loam", "loam"],
            "climate": {
                "temperature_range_c": [15, 45],
                "annual_rainfall_mm": 800,
                "humidity_percent": 60
            },
            "crop_calendar": {
                "rabi_season": {"start_month": 10, "end_month": 4},
                "kharif_season": {"start_month": 6, "end_month": 10},
                "zaid_season": {"start_month": 3, "end_month": 6}
            },
            "government_policies": [],
            "market_centers": [],
            "extension_services": []
        }

        # Update config with defaults for missing keys
        for key, value in defaults.items():
            if key not in self.config:
                self.config[key] = value

        # Save updated config
        self._save_config()

    def get_districts(self) -> List[str]:
        """Get list of districts in the state"""
        return self.config.get('districts', [])

    def get_agro_climatic_zones(self) -> List[str]:
        """Get agro-climatic zones in the state"""
        return self.config.get('agro_climatic_zones', [])

    def get_major_crops(self) -> List[str]:
        """Get major crops grown in the state"""
        return self.config.get('major_crops', [])

    def get_irrigation_info(self) -> Dict[str, Any]:
        """Get irrigation information for the state"""
        return self.config.get('irrigation', {})

    def get_climate_info(self) -> Dict[str, Any]:
        """Get climate information for the state"""
        return self.config.get('climate', {})

    def get_crop_calendar(self, season: str = None) -> Dict[str, Any]:
        """Get crop calendar information"""
        calendar = self.config.get('crop_calendar', {})
        if season:
            return calendar.get(season, {})
        return calendar

    def is_crop_suitable(self, crop_name: str) -> bool:
        """Check if a crop is suitable for this state"""
        major_crops = self.get_major_crops()
        return crop_name.lower() in [c.lower() for c in major_crops]

    def get_season_for_month(self, month: int) -> str:
        """Get the cropping season for a given month"""
        calendar = self.get_crop_calendar()

        for season, period in calendar.items():
            start_month = period.get('start_month', 1)
            end_month = period.get('end_month', 12)

            if start_month <= end_month:
                # Normal season (e.g., June-October)
                if start_month <= month <= end_month:
                    return season
            else:
                # Wrapped season (e.g., October-April)
                if month >= start_month or month <= end_month:
                    return season

        return "off_season"

    def get_soil_suitability_score(self, soil_type: str) -> float:
        """Get soil suitability score for the state (0-1)"""
        soil_types = self.config.get('soil_types', [])
        if soil_type.lower() in [s.lower() for s in soil_types]:
            return 1.0
        return 0.5  # Moderately suitable

    def get_climate_suitability_score(self, temperature_c: float,
                                    rainfall_mm: float) -> float:
        """Get climate suitability score for the state (0-1)"""
        climate = self.get_climate_info()

        temp_range = climate.get('temperature_range_c', [15, 45])
        temp_score = 1.0 if temp_range[0] <= temperature_c <= temp_range[1] else 0.5

        annual_rainfall = climate.get('annual_rainfall_mm', 800)
        rainfall_score = 1.0 - abs(rainfall_mm - annual_rainfall) / annual_rainfall
        rainfall_score = max(0.1, min(1.0, rainfall_score))

        return (temp_score * 0.6 + rainfall_score * 0.4)

    def add_district(self, district_name: str, district_info: Dict[str, Any] = None):
        """Add a district to the state"""
        districts = self.config.get('districts', [])
        if district_name not in districts:
            districts.append(district_name)
            self.config['districts'] = districts

            # Store district information
            if district_info:
                if 'district_info' not in self.config:
                    self.config['district_info'] = {}
                self.config['district_info'][district_name] = district_info

            self._save_config()
            logger.info(f"Added district {district_name} to state {self.state_name}")

    def add_crop(self, crop_name: str, crop_info: Dict[str, Any] = None):
        """Add a crop to the state's major crops"""
        major_crops = self.config.get('major_crops', [])
        if crop_name not in major_crops:
            major_crops.append(crop_name)
            self.config['major_crops'] = major_crops

            # Store crop information
            if crop_info:
                if 'crop_info' not in self.config:
                    self.config['crop_info'] = {}
                self.config['crop_info'][crop_name] = crop_info

            self._save_config()
            logger.info(f"Added crop {crop_name} to state {self.state_name}")

    def update_config(self, updates: Dict[str, Any]):
        """Update state configuration"""
        self.config.update(updates)
        self._save_config()
        logger.info(f"Updated configuration for state: {self.state_name}")

# Pre-configured state configurations
DEFAULT_STATE_CONFIGS = {
    "punjab": {
        "full_name": "Punjab",
        "total_area_hectares": 5036000,
        "agricultural_area_hectares": 4200000,
        "major_crops": ["wheat", "rice", "cotton", "maize", "sugarcane"],
        "irrigation": {
            "coverage_percent": 98,
            "methods": ["canal", "tubewell"],
            "efficiency": 0.75
        },
        "soil_types": ["alluvial", "sandy_loam"],
        "climate": {
            "temperature_range_c": [5, 45],
            "annual_rainfall_mm": 600,
            "humidity_percent": 65
        },
        "agro_climatic_zones": ["North-Western Zone", "Central Zone", "South-Western Zone", "North-Eastern Zone"]
    },

    "haryana": {
        "full_name": "Haryana",
        "total_area_hectares": 4421000,
        "agricultural_area_hectares": 3500000,
        "major_crops": ["wheat", "rice", "cotton", "mustard", "sugarcane"],
        "irrigation": {
            "coverage_percent": 85,
            "methods": ["canal", "tubewell", "rainfed"],
            "efficiency": 0.70
        },
        "soil_types": ["alluvial", "sandy_loam", "loam"],
        "climate": {
            "temperature_range_c": [5, 45],
            "annual_rainfall_mm": 500,
            "humidity_percent": 60
        },
        "agro_climatic_zones": ["North-Eastern Zone", "Central Zone", "South-Western Zone"]
    },

    "uttar_pradesh": {
        "full_name": "Uttar Pradesh",
        "total_area_hectares": 24092800,  # 24 million hectares
        "agricultural_area_hectares": 17000000,
        "major_crops": ["wheat", "rice", "sugarcane", "potato", "mustard"],
        "irrigation": {
            "coverage_percent": 75,
            "methods": ["canal", "tubewell", "rainfed"],
            "efficiency": 0.65
        },
        "soil_types": ["alluvial", "sandy_loam", "loam", "clay"],
        "climate": {
            "temperature_range_c": [5, 48],
            "annual_rainfall_mm": 1000,
            "humidity_percent": 65
        },
        "agro_climatic_zones": ["Western Plain", "Central Plain", "Eastern Plain", "Bundelkhand", "Tarai"]
    },

    "bihar": {
        "full_name": "Bihar",
        "total_area_hectares": 9416300,
        "agricultural_area_hectares": 5800000,
        "major_crops": ["rice", "wheat", "maize", "pulses", "sugarcane"],
        "irrigation": {
            "coverage_percent": 60,
            "methods": ["canal", "tubewell", "rainfed"],
            "efficiency": 0.60
        },
        "soil_types": ["alluvial", "clay", "sandy_loam"],
        "climate": {
            "temperature_range_c": [8, 45],
            "annual_rainfall_mm": 1200,
            "humidity_percent": 70
        },
        "agro_climatic_zones": ["North Bihar", "South Bihar"]
    },

    "madhya_pradesh": {
        "full_name": "Madhya Pradesh",
        "total_area_hectares": 30835000,
        "agricultural_area_hectares": 15000000,
        "major_crops": ["soybean", "wheat", "maize", "pulses", "cotton"],
        "irrigation": {
            "coverage_percent": 45,
            "methods": ["canal", "tubewell", "rainfed"],
            "efficiency": 0.55
        },
        "soil_types": ["black_soil", "alluvial", "red_soil"],
        "climate": {
            "temperature_range_c": [10, 48],
            "annual_rainfall_mm": 1200,
            "humidity_percent": 60
        },
        "agro_climatic_zones": ["Malwa", "Nimar", "Bundelkhand", "Baghelkhand", "Chhattisgarh Plain"]
    },

    "jharkhand": {
        "full_name": "Jharkhand",
        "total_area_hectares": 7971600,
        "agricultural_area_hectares": 1800000,
        "major_crops": ["rice", "maize", "wheat", "pulses", "oilseeds"],
        "irrigation": {
            "coverage_percent": 25,
            "methods": ["rainfed", "canal", "tubewell"],
            "efficiency": 0.50
        },
        "soil_types": ["red_soil", "laterite", "alluvial"],
        "climate": {
            "temperature_range_c": [5, 45],
            "annual_rainfall_mm": 1400,
            "humidity_percent": 75
        },
        "agro_climatic_zones": ["North Chhotanagpur", "South Chhotanagpur", "Santhal Pargana"]
    }
}

def create_state_config(state_name: str) -> StateConfig:
    """Factory function to create state configuration"""
    config = StateConfig(state_name)

    # Apply default configuration if available
    if state_name in DEFAULT_STATE_CONFIGS:
        config.update_config(DEFAULT_STATE_CONFIGS[state_name])

    return config

def get_available_states() -> List[str]:
    """Get list of available states"""
    states_dir = Path("india_agri_platform/states")
    if states_dir.exists():
        return [d.name for d in states_dir.iterdir() if d.is_dir()]
    return []
