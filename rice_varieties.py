"""
Rice Variety Characteristics Database
Comprehensive rice variety profiles for multi-state yield modeling
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from pathlib import Path

class RiceVarietyManager:
    """Manages rice variety characteristics and profiles across India"""

    def __init__(self):
        # Major rice varieties by region
        self.variety_database = self._load_rice_varieties()

    def _load_rice_varieties(self) -> Dict[str, Dict[str, Any]]:
        """Load comprehensive rice variety database"""
        return {
            # Punjab Rice Varieties
            "PR121": {
                "name": "PR121",
                "state": "PUNJAB",
                "maturity_days": 120,
                "yield_potential_quintal_ha": 85,
                "grain_type": "long_grain",
                "aroma": "non_aromatic",
                "disease_resistance": {
                    "blast": "moderate",
                    "bacterial_blight": "high",
                    "brown_plant_hopper": "moderate"
                },
                "water_requirement_mm": 1350,
                "soil_adaptability": ["clay", "clay_loam"],
                "climate_zone": "north_india_irrigated",
                "market_preference": "high",
                "created_year": 2015
            },
            "PR126": {
                "name": "PR126",
                "state": "PUNJAB",
                "maturity_days": 125,
                "yield_potential_quintal_ha": 82,
                "grain_type": "long_grain",
                "aroma": "non_aromatic",
                "disease_resistance": {
                    "blast": "high",
                    "bacterial_blight": "moderate",
                    "brown_plant_hopper": "high"
                },
                "water_requirement_mm": 1300,
                "soil_adaptability": ["clay", "clay_loam"],
                "climate_zone": "north_india_irrigated",
                "market_preference": "high",
                "created_year": 2018
            },
            "PUSA-44": {
                "name": "PUSA-44",
                "state": "PUNJAB",
                "maturity_days": 140,
                "yield_potential_quintal_ha": 78,
                "grain_type": "long_grain",
                "aroma": "non_aromatic",
                "disease_resistance": {
                    "blast": "moderate",
                    "bacterial_blight": "high",
                    "brown_plant_hopper": "low"
                },
                "water_requirement_mm": 1400,
                "soil_adaptability": ["clay", "silty_clay"],
                "climate_zone": "north_india_irrigated",
                "market_preference": "high",
                "created_year": 2010
            },
            "HKV-18": {
                "name": "HKV-18",
                "state": "HARYANA",
                "maturity_days": 125,
                "yield_potential_quintal_ha": 75,
                "grain_type": "long_grain",
                "aroma": "non_aromatic",
                "disease_resistance": {
                    "blast": "moderate",
                    "bacterial_blight": "high",
                    "brown_plant_hopper": "moderate"
                },
                "water_requirement_mm": 1280,
                "soil_adaptability": ["clay_loam", "loam"],
                "climate_zone": "north_india_irrigated",
                "market_preference": "high",
                "created_year": 2014
            },
            # UP Rice Varieties
            "NDR-359": {
                "name": "NDR-359",
                "state": "UTTAR PRADESH",
                "maturity_days": 140,
                "yield_potential_quintal_ha": 65,
                "grain_type": "medium_grain",
                "aroma": "non_aromatic",
                "disease_resistance": {
                    "blast": "low",
                    "bacterial_blight": "moderate",
                    "brown_plant_hopper": "low"
                },
                "water_requirement_mm": 1200,
                "soil_adaptability": ["clay", "clay_loam"],
                "climate_zone": "central_india_mixed",
                "market_preference": "medium",
                "created_year": 2012
            },
            "MDS-204": {
                "name": "MDS-204",
                "state": "UTTAR PRADESH",
                "maturity_days": 125,
                "yield_potential_quintal_ha": 55,
                "grain_type": "medium_grain",
                "aroma": "non_aromatic",
                "disease_resistance": {
                    "blast": "moderate",
                    "bacterial_blight": "low",
                    "brown_plant_hopper": "low"
                },
                "water_requirement_mm": 1150,
                "soil_adaptability": ["silty_clay", "clay_loam"],
                "climate_zone": "central_india_rainfed",
                "market_preference": "medium",
                "created_year": 2008
            },
            # Bihar Rice Varieties
            "RAJENDRA-93": {
                "name": "RAJENDRA-93",
                "state": "BIHAR",
                "maturity_days": 135,
                "yield_potential_quintal_ha": 45,
                "grain_type": "medium_grain",
                "aroma": "non_aromatic",
                "disease_resistance": {
                    "blast": "moderate",
                    "bacterial_blight": "moderate",
                    "brown_plant_hopper": "low"
                },
                "water_requirement_mm": 950,
                "soil_adaptability": ["clay", "silty_clay"],
                "climate_zone": "east_india_rainfed",
                "market_preference": "medium",
                "created_year": 2005
            },
            "RAJENDRA-101": {
                "name": "RAJENDRA-101",
                "state": "BIHAR",
                "maturity_days": 120,
                "yield_potential_quintal_ha": 50,
                "grain_type": "long_grain",
                "aroma": "non_aromatic",
                "disease_resistance": {
                    "blast": "high",
                    "bacterial_blight": "moderate",
                    "brown_plant_hopper": "low"
                },
                "water_requirement_mm": 1000,
                "soil_adaptability": ["clay", "clay_loam"],
                "climate_zone": "east_india_mixed",
                "market_preference": "medium",
                "created_year": 2010
            },
            # West Bengal Rice Varieties
            "SWARNA": {
                "name": "SWARNA",
                "state": "WEST BENGAL",
                "maturity_days": 145,
                "yield_potential_quintal_ha": 55,
                "grain_type": "medium_grain",
                "aroma": "non_aromatic",
                "disease_resistance": {
                    "blast": "moderate",
                    "bacterial_blight": "moderate",
                    "brown_plant_hopper": "low"
                },
                "water_requirement_mm": 1100,
                "soil_adaptability": ["clay", "silty_clay"],
                "climate_zone": "east_india_irrigated",
                "market_preference": "high",
                "created_year": 2002
            },
            "KRISHNA-HAMSARA": {
                "name": "KRISHNA-HAMSARA",
                "state": "WEST BENGAL",
                "maturity_days": 130,
                "yield_potential_quintal_ha": 48,
                "grain_type": "medium_grain",
                "aroma": "aromatic",
                "disease_resistance": {
                    "blast": "moderate",
                    "bacterial_blight": "low",
                    "brown_plant_hopper": "moderate"
                },
                "water_requirement_mm": 1050,
                "soil_adaptability": ["clay", "clay_loam"],
                "climate_zone": "east_india_irrigated",
                "market_preference": "high",
                "created_year": 2008
            },
            # Andhra Pradesh Rice Varieties
            "MTU-1001": {
                "name": "MTU-1001",
                "state": "ANDHRA PRADESH",
                "maturity_days": 125,
                "yield_potential_quintal_ha": 68,
                "grain_type": "long_grain",
                "aroma": "non_aromatic",
                "disease_resistance": {
                    "blast": "high",
                    "bacterial_blight": "moderate",
                    "brown_plant_hopper": "high"
                },
                "water_requirement_mm": 1250,
                "soil_adaptability": ["clay", "clay_loam"],
                "climate_zone": "south_india_irrigated",
                "market_preference": "high",
                "created_year": 2015
            },
            "MTU-7029": {
                "name": "MTU-7029",
                "state": "ANDHRA PRADESH",
                "maturity_days": 120,
                "yield_potential_quintal_ha": 72,
                "grain_type": "long_grain",
                "aroma": "non_aromatic",
                "disease_resistance": {
                    "blast": "moderate",
                    "bacterial_blight": "high",
                    "brown_plant_hopper": "moderate"
                },
                "water_requirement_mm": 1220,
                "soil_adaptability": ["clay", "clay_loam"],
                "climate_zone": "south_india_irrigated",
                "market_preference": "high",
                "created_year": 2018
            },
            # Tamil Nadu Rice Varieties
            "ASD-18": {
                "name": "ASD-18",
                "state": "TAMIL NADU",
                "maturity_days": 115,
                "yield_potential_quintal_ha": 55,
                "grain_type": "medium_grain",
                "aroma": "non_aromatic",
                "disease_resistance": {
                    "blast": "moderate",
                    "bacterial_blight": "moderate",
                    "brown_plant_hopper": "low"
                },
                "water_requirement_mm": 1150,
                "soil_adaptability": ["clay", "silty_clay"],
                "climate_zone": "south_india_irrigated",
                "market_preference": "medium",
                "created_year": 2012
            },
            "IR-64": {
                "name": "IR-64",
                "state": "TAMIL NADU",
                "maturity_days": 110,
                "yield_potential_quintal_ha": 60,
                "grain_type": "long_grain",
                "aroma": "non_aromatic",
                "disease_resistance": {
                    "blast": "moderate",
                    "bacterial_blight": "high",
                    "brown_plant_hopper": "moderate"
                },
                "water_requirement_mm": 1180,
                "soil_adaptability": ["clay", "clay_loam"],
                "climate_zone": "south_india_irrigated",
                "market_preference": "high",
                "created_year": 1985
            }
        }

    def get_variety_info(self, variety_code: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific rice variety"""
        return self.variety_database.get(variety_code)

    def get_varieties_for_state(self, state: str) -> List[str]:
        """Get all rice varieties available in a specific state"""
        state_upper = state.upper()
        return [v for v, info in self.variety_database.items()
                if info['state'] == state_upper]

    def get_varieties_for_climate_zone(self, zone: str) -> List[str]:
        """Get rice varieties suitable for a climate zone"""
        return [v for v, info in self.variety_database.items()
                if info['climate_zone'] == zone]

    def calculate_variety_yield_potential(self, variety_code: str,
                                         environmental_factors: Dict[str, Any]) -> float:
        """Calculate yield potential for a variety under specific conditions"""

        variety = self.get_variety_info(variety_code)
        if not variety:
            return 35.0  # Default average

        base_yield = variety['yield_potential_quintal_ha']

        # Adjust for environmental factors
        adjustment_factors = []

        # Irrigation adjustment
        irrigation_coverage = environmental_factors.get('irrigation_coverage', 0.7)
        irrigation_adjustment = 0.8 + (irrigation_coverage * 0.4)  # 0.8 to 1.2
        adjustment_factors.append(irrigation_adjustment)

        # Temperature adjustment (optimal 25-30°C)
        temperature = environmental_factors.get('temperature_celsius', 25)
        temp_adjustment = 1.0
        if temperature < 20 or temperature > 35:
            temp_adjustment = 0.9
        elif 20 <= temperature <= 30:
            temp_adjustment = 1.0
        else:
            temp_adjustment = 0.95
        adjustment_factors.append(temp_adjustment)

        # Disease adjustment based on resistance
        disease_risk = environmental_factors.get('disease_pressure', 0.5)
        blast_resistance = variety['disease_resistance'].get('blast', 'moderate')
        resistance_multiplier = {'low': 0.9, 'moderate': 0.95, 'high': 1.0}
        disease_adjustment = resistance_multiplier.get(blast_resistance, 0.95)
        disease_adjustment = 1.0 - (disease_risk * (1.0 - disease_adjustment))
        adjustment_factors.append(disease_adjustment)

        # Apply all adjustments
        final_yield = base_yield
        for factor in adjustment_factors:
            final_yield *= factor

        return round(final_yield, 2)

    def get_disease_resistance_score(self, variety_code: str, disease_type: str) -> float:
        """Get disease resistance score (0-1 scale)"""
        variety = self.get_variety_info(variety_code)
        if not variety:
            return 0.5  # Default moderate resistance

        resistance_map = {
            'low': 0.3,
            'moderate': 0.6,
            'high': 0.9
        }

        resistance_level = variety['disease_resistance'].get(disease_type, 'moderate')
        return resistance_map.get(resistance_level, 0.5)

    def get_maturity_period(self, variety_code: str) -> int:
        """Get maturity period in days"""
        variety = self.get_variety_info(variety_code)
        return variety.get('maturity_days', 120) if variety else 120

    def get_optimal_conditions(self, variety_code: str) -> Dict[str, Any]:
        """Get optimal growing conditions for a variety"""
        variety = self.get_variety_info(variety_code)
        if not variety:
            return {
                'temperature_range': [20, 30],
                'water_requirement_mm': 1200,
                'soil_types': ['clay', 'clay_loam'],
                'climate_zone': 'mixed'
            }

        return {
            'temperature_range': [20, 35],  # General rice range
            'water_requirement_mm': variety['water_requirement_mm'],
            'soil_types': variety['soil_adaptability'],
            'climate_zone': variety['climate_zone']
        }

    def create_variety_dataset_for_state(self, state: str) -> pd.DataFrame:
        """Create a comprehensive dataset of rice varieties for a state"""
        state_varieties = self.get_varieties_for_state(state)

        if not state_varieties:
            # Return general varieties
            state_varieties = ['PR121', 'PR126', 'PUSA-44']

        variety_data = []
        for variety in state_varieties:
            info = self.get_variety_info(variety)
            if info:
                variety_data.append(info)

        return pd.DataFrame(variety_data)

    def recommend_varieties(self, conditions: Dict[str, Any],
                           top_n: int = 5) -> List[Dict[str, Any]]:
        """Recommend best rice varieties for given conditions"""

        recommendations = []

        for variety_code, info in self.variety_database.items():
            score = self._calculate_variety_score(variety_code, conditions)
            recommendations.append({
                'variety': variety_code,
                'name': info['name'],
                'state': info['state'],
                'yield_potential': info['yield_potential_quintal_ha'],
                'maturity_days': info['maturity_days'],
                'suitability_score': score,
                'climate_zone': info['climate_zone']
            })

        # Sort by suitability score
        recommendations.sort(key=lambda x: x['suitability_score'], reverse=True)

        return recommendations[:top_n]

    def _calculate_variety_score(self, variety_code: str, conditions: Dict[str, Any]) -> float:
        """Calculate overall suitability score for a variety under given conditions"""

        variety = self.get_variety_info(variety_code)
        if not variety:
            return 0.0

        scores = []

        # Irrigation compatibility (0-1)
        available_irrigation = conditions.get('irrigation_coverage', 0.5)
        required_irrigation = variety['water_requirement_mm'] / 1350  # Normalize
        irrigation_score = 1.0 - abs(available_irrigation - min(required_irrigation, 1.0))
        scores.append(irrigation_score)

        # Temperature compatibility (0-1)
        temperature = conditions.get('temperature_celsius', 25)
        temp_score = 1.0 if 20 <= temperature <= 30 else 0.8
        scores.append(temp_score)

        # Yield potential (normalized 0-1)
        yield_score = variety['yield_potential_quintal_ha'] / 85.0  # Max known yield
        scores.append(yield_score)

        # Recency bonus (newer varieties preferred)
        recency_score = min(variety['created_year'] / 2020, 1.0)
        scores.append(recency_score * 0.3)  # 30% weight

        # Average of all scores
        return sum(scores) / len(scores)

# Global rice variety manager instance
rice_variety_manager = RiceVarietyManager()

def main():
    """Test and demonstrate rice variety functionality"""
    print("🌾 RICE VARIETY MANAGEMENT SYSTEM")
    print("=" * 50)

    # Test basic functionality
    print("\n📊 Testing Rice Variety Database:")

    # Get Punjab varieties
    punjab_varieties = rice_variety_manager.get_varieties_for_state('punjab')
    print(f"Punjab rice varieties: {punjab_varieties}")

    # Test specific variety
    pr121_info = rice_variety_manager.get_variety_info('PR121')
    print(f"\nPR121 variety details: {pr121_info}")

    # Test yield potential calculation
    test_conditions = {
        'irrigation_coverage': 0.9,
        'temperature_celsius': 28,
        'disease_pressure': 0.3
    }

    yield_potential = rice_variety_manager.calculate_variety_yield_potential('PR121', test_conditions)
    print(f"PR121 yield potential in optimal conditions: {yield_potential} q/ha")

    # Test variety recommendations
    recommendations = rice_variety_manager.recommend_varieties(test_conditions, top_n=3)
    print(f"\nTop 3 recommended varieties:")
    for rec in recommendations:
        print(f"• {rec['variety']}: Score {rec['suitability_score']:.2f}, Yield {rec['yield_potential']} q/ha")

    # Create variety dataset
    punjab_df = rice_variety_manager.create_variety_dataset_for_state('punjab')
    print(f"\nPunjab rice varieties dataset: {len(punjab_df)} varieties")

    print("\n✅ Rice variety management system ready!")

if __name__ == "__main__":
    main()
