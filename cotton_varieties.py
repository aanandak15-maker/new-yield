"""
Cotton Variety Characteristics Database
Comprehensive cotton variety profiles for multi-state yield modeling
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from pathlib import Path

class CottonVarietyManager:
    """Manages cotton variety characteristics and profiles across India"""

    def __init__(self):
        # Major cotton varieties by region
        self.variety_database = self._load_cotton_varieties()

    def _load_cotton_varieties(self) -> Dict[str, Dict[str, Any]]:
        """Load comprehensive cotton variety database"""
        return {
            # Punjab Cotton Varieties (BT Focus)
            "F-1861": {
                "name": "F-1861",
                "state": "PUNJAB",
                "maturity_days": 170,
                "yield_potential_quintal_ha": 35,
                "fiber_type": "long_staple",
                "fiber_length_mm": 28,
                "bt_trait": "cry1ac",
                "pest_resistance": {
                    "bollworm": "high",
                    "whitefly": "moderate",
                    "jassids": "moderate"
                },
                "water_requirement_mm": 650,
                "soil_adaptability": ["clay_loam", "silty_clay"],
                "climate_zone": "north_india_irrigated",
                "market_preference": "high",
                "created_year": 2008,
                "boll_opening_schedule": "progressive"
            },
            "CIM-600": {
                "name": "CIM-600",
                "state": "PUNJAB",
                "maturity_days": 165,
                "yield_potential_quintal_ha": 32,
                "fiber_type": "long_staple",
                "fiber_length_mm": 29,
                "bt_trait": "cry1ac_cry2ab",
                "pest_resistance": {
                    "bollworm": "very_high",
                    "whitefly": "high",
                    "jassids": "high"
                },
                "water_requirement_mm": 620,
                "soil_adaptability": ["clay", "clay_loam"],
                "climate_zone": "north_india_irrigated",
                "market_preference": "high",
                "created_year": 2012,
                "boll_opening_schedule": "progressive"
            },
            "SLH-317": {
                "name": "SLH-317",
                "state": "PUNJAB",
                "maturity_days": 175,
                "yield_potential_quintal_ha": 30,
                "fiber_type": "long_staple",
                "fiber_length_mm": 27,
                "bt_trait": "cry1ac",
                "pest_resistance": {
                    "bollworm": "high",
                    "whitefly": "low",
                    "jassids": "moderate"
                },
                "water_requirement_mm": 680,
                "soil_adaptability": ["loam", "silty_loam"],
                "climate_zone": "north_india_irrigated",
                "market_preference": "medium",
                "created_year": 2005,
                "boll_opening_schedule": "progressive"
            },
            # Haryana Cotton Varieties
            "H-1098": {
                "name": "H-1098",
                "state": "HARYANA",
                "maturity_days": 170,
                "yield_potential_quintal_ha": 28,
                "fiber_type": "medium_staple",
                "fiber_length_mm": 25,
                "bt_trait": "cry1ac",
                "pest_resistance": {
                    "bollworm": "high",
                    "whitefly": "moderate",
                    "jassids": "low"
                },
                "water_requirement_mm": 630,
                "soil_adaptability": ["clay_loam", "sandy_loam"],
                "climate_zone": "north_india_irrigated",
                "market_preference": "medium",
                "created_year": 2010,
                "boll_opening_schedule": "progressive"
            },
            # Maharashtra Cotton Varieties (Major Producer)
            "AKH-081": {
                "name": "AKH-081",
                "state": "MAHARASHTRA",
                "maturity_days": 165,
                "yield_potential_quintal_ha": 30,
                "fiber_type": "medium_staple",
                "fiber_length_mm": 26,
                "bt_trait": "cry1ac_cry2ab",
                "pest_resistance": {
                    "bollworm": "very_high",
                    "whitefly": "high",
                    "jassids": "high"
                },
                "water_requirement_mm": 580,
                "soil_adaptability": ["black_cotton", "deep_black"],
                "climate_zone": "central_india_rainfed",
                "market_preference": "high",
                "created_year": 2015,
                "boll_opening_schedule": "simultaneous"
            },
            "Jayadhar": {
                "name": "Jayadhar",
                "state": "MAHARASHTRA",
                "maturity_days": 160,
                "yield_potential_quintal_ha": 32,
                "fiber_type": "medium_staple",
                "fiber_length_mm": 27,
                "bt_trait": "cry1ac",
                "pest_resistance": {
                    "bollworm": "high",
                    "whitefly": "moderate",
                    "jassids": "moderate"
                },
                "water_requirement_mm": 600,
                "soil_adaptability": ["black_cotton", "red_soil"],
                "climate_zone": "central_india_mixed",
                "market_preference": "high",
                "created_year": 2013,
                "boll_opening_schedule": "simultaneous"
            },
            "Suraj": {
                "name": "Suraj",
                "state": "MAHARASHTRA",
                "maturity_days": 155,
                "yield_potential_quintal_ha": 28,
                "fiber_type": "medium_staple",
                "fiber_length_mm": 25,
                "bt_trait": "cry1ac_cry2ab",
                "pest_resistance": {
                    "bollworm": "very_high",
                    "whitefly": "high",
                    "jassids": "low"
                },
                "water_requirement_mm": 620,
                "soil_adaptability": ["black_cotton", "alluvial"],
                "climate_zone": "central_india_irrigated",
                "market_preference": "medium",
                "created_year": 2018,
                "boll_opening_schedule": "simultaneous"
            },
            # Gujarat Cotton Varieties
            "G-27": {
                "name": "G-27",
                "state": "GUJARAT",
                "maturity_days": 165,
                "yield_potential_quintal_ha": 31,
                "fiber_type": "long_staple",
                "fiber_length_mm": 28,
                "bt_trait": "cry1ac",
                "pest_resistance": {
                    "bollworm": "high",
                    "whitefly": "low",
                    "jassids": "moderate"
                },
                "water_requirement_mm": 650,
                "soil_adaptability": ["black_cotton", "alluvial"],
                "climate_zone": "west_india_irrigated",
                "market_preference": "high",
                "created_year": 2009,
                "boll_opening_schedule": "progressive"
            },
            "GC-1": {
                "name": "GC-1",
                "state": "GUJARAT",
                "maturity_days": 170,
                "yield_potential_quintal_ha": 29,
                "fiber_type": "long_staple",
                "fiber_length_mm": 29,
                "bt_trait": "cry1ac_cry2ab",
                "pest_resistance": {
                    "bollworm": "very_high",
                    "whitefly": "moderate",
                    "jassids": "high"
                },
                "water_requirement_mm": 680,
                "soil_adaptability": ["medium_black", "heavy_black"],
                "climate_zone": "west_india_mixed",
                "market_preference": "high",
                "created_year": 2014,
                "boll_opening_schedule": "progressive"
            },
            "KD-208": {
                "name": "KD-208",
                "state": "GUJARAT",
                "maturity_days": 175,
                "yield_potential_quintal_ha": 27,
                "fiber_type": "medium_staple",
                "fiber_length_mm": 26,
                "bt_trait": "cry1ac",
                "pest_resistance": {
                    "bollworm": "high",
                    "whitefly": "low",
                    "jassids": "moderate"
                },
                "water_requirement_mm": 660,
                "soil_adaptability": ["black_cotton", "saline_soil"],
                "climate_zone": "west_india_irrigated",
                "market_preference": "medium",
                "created_year": 2007,
                "boll_opening_schedule": "progressive"
            },
            # Andhra Pradesh Cotton Varieties
            "S-6": {
                "name": "S-6",
                "state": "ANDHRA PRADESH",
                "maturity_days": 160,
                "yield_potential_quintal_ha": 26,
                "fiber_type": "medium_staple",
                "fiber_length_mm": 24,
                "bt_trait": "cry1ac",
                "pest_resistance": {
                    "bollworm": "moderate",
                    "whitefly": "high",
                    "jassids": "moderate"
                },
                "water_requirement_mm": 580,
                "soil_adaptability": ["red_soil", "black_soil"],
                "climate_zone": "south_india_mixed",
                "market_preference": "medium",
                "created_year": 2006,
                "boll_opening_schedule": "simultaneous"
            },
            "MCU-5": {
                "name": "MCU-5",
                "state": "ANDHRA PRADESH",
                "maturity_days": 165,
                "yield_potential_quintal_ha": 28,
                "fiber_type": "medium_staple",
                "fiber_length_mm": 25,
                "bt_trait": "cry1ac_cry2ab",
                "pest_resistance": {
                    "bollworm": "high",
                    "whitefly": "moderate",
                    "jassids": "low"
                },
                "water_requirement_mm": 620,
                "soil_adaptability": ["vertisol", "alfisol"],
                "climate_zone": "south_india_irrigated",
                "market_preference": "medium",
                "created_year": 2011,
                "boll_opening_schedule": "simultaneous"
            },
            "NDLH-1936": {
                "name": "NDLH-1936",
                "state": "ANDHRA PRADESH",
                "maturity_days": 155,
                "yield_potential_quintal_ha": 30,
                "fiber_type": "medium_staple",
                "fiber_length_mm": 26,
                "bt_trait": "cry1ac",
                "pest_resistance": {
                    "bollworm": "high",
                    "whitefly": "moderate",
                    "jassids": "high"
                },
                "water_requirement_mm": 600,
                "soil_adaptability": ["black_soil", "red_soil"],
                "climate_zone": "south_india_mixed",
                "market_preference": "high",
                "created_year": 2020,
                "boll_opening_schedule": "simultaneous"
            },
            # Tamil Nadu Cotton Varieties
            "Surabhi": {
                "name": "Surabhi",
                "state": "TAMIL NADU",
                "maturity_days": 180,
                "yield_potential_quintal_ha": 35,
                "fiber_type": "extra_long_staple",
                "fiber_length_mm": 32,
                "bt_trait": "cry1ac_cry2ab",
                "pest_resistance": {
                    "bollworm": "very_high",
                    "whitefly": "high",
                    "jassids": "high"
                },
                "water_requirement_mm": 700,
                "soil_adaptability": ["black_soil", "red_soil"],
                "climate_zone": "south_india_irrigated",
                "market_preference": "premium",
                "created_year": 2016,
                "boll_opening_schedule": "progressive"
            },
            "CO-14": {
                "name": "CO-14",
                "state": "TAMIL NADU",
                "maturity_days": 175,
                "yield_potential_quintal_ha": 32,
                "fiber_type": "long_staple",
                "fiber_length_mm": 30,
                "bt_trait": "cry1ac",
                "pest_resistance": {
                    "bollworm": "high",
                    "whitefly": "moderate",
                    "jassids": "moderate"
                },
                "water_requirement_mm": 680,
                "soil_adaptability": ["black_cotton", "alluvial"],
                "climate_zone": "south_india_irrigated",
                "market_preference": "high",
                "created_year": 2010,
                "boll_opening_schedule": "progressive"
            },
            # Karnataka Cotton Varieties
            "DCH-32": {
                "name": "DCH-32",
                "state": "KARNATAKA",
                "maturity_days": 160,
                "yield_potential_quintal_ha": 28,
                "fiber_type": "medium_staple",
                "fiber_length_mm": 25,
                "bt_trait": "cry1ac_cry2ab",
                "pest_resistance": {
                    "bollworm": "very_high",
                    "whitefly": "high",
                    "jassids": "moderate"
                },
                "water_requirement_mm": 590,
                "soil_adaptability": ["red_soil", "laterite"],
                "climate_zone": "south_india_rainfed",
                "market_preference": "medium",
                "created_year": 2019,
                "boll_opening_schedule": "simultaneous"
            }
        }

    def get_variety_info(self, variety_code: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific cotton variety"""
        return self.variety_database.get(variety_code)

    def get_varieties_for_state(self, state: str) -> List[str]:
        """Get all cotton varieties available in a specific state"""
        state_upper = state.upper()
        return [v for v, info in self.variety_database.items()
                if info['state'] == state_upper]

    def get_varieties_for_climate_zone(self, zone: str) -> List[str]:
        """Get cotton varieties suitable for a climate zone"""
        return [v for v, info in self.variety_database.items()
                if info['climate_zone'] == zone]

    def calculate_variety_yield_potential(self, variety_code: str,
                                         environmental_factors: Dict[str, Any]) -> float:
        """Calculate yield potential for a variety under specific conditions"""

        variety = self.get_variety_info(variety_code)
        if not variety:
            return 15.0  # Default average cotton yield

        base_yield = variety['yield_potential_quintal_ha']

        # Adjust for environmental factors
        adjustment_factors = []

        # BT trait effectiveness (against bollworm)
        bollworm_pressure = environmental_factors.get('bollworm_pressure', 0.5)
        bt_trait = variety.get('bt_trait', '')
        if 'cry1ac' in bt_trait:
            bt_effectiveness = 0.9 if 'cry2ab' in bt_trait else 0.8  # Dual trait better
            pest_adjustment = 1.0 - (bollworm_pressure * (1.0 - bt_effectiveness))
            adjustment_factors.append(pest_adjustment)

        # Irrigation adjustment
        irrigation_coverage = environmental_factors.get('irrigation_coverage', 0.5)
        irrigation_adjustment = 0.85 + (irrigation_coverage * 0.3)  # 0.85 to 1.15
        adjustment_factors.append(irrigation_adjustment)

        # Temperature adjustment (optimal 25-35°C for cotton)
        temperature = environmental_factors.get('temperature_celsius', 28)
        temp_adjustment = 1.0
        if temperature < 20 or temperature > 40:
            temp_adjustment = 0.9
        elif 25 <= temperature <= 35:
            temp_adjustment = 1.0
        else:
            temp_adjustment = 0.95
        adjustment_factors.append(temp_adjustment)

        # Disease adjustment based on resistance
        disease_risk = environmental_factors.get('disease_risk', 0.3)
        disease_adjustment = 1.0 - (disease_risk * 0.1)  # Minor impact
        adjustment_factors.append(disease_adjustment)

        # Apply all adjustments
        final_yield = base_yield
        for factor in adjustment_factors:
            final_yield *= factor

        return round(final_yield, 2)

    def get_pest_resistance_score(self, variety_code: str, pest_type: str) -> float:
        """Get pest resistance score (0-1 scale)"""
        variety = self.get_variety_info(variety_code)
        if not variety:
            return 0.5  # Default moderate resistance

        resistance_map = {
            'low': 0.3,
            'moderate': 0.6,
            'high': 0.8,
            'very_high': 0.9
        }

        resistance_level = variety['pest_resistance'].get(pest_type, 'moderate')
        return resistance_map.get(resistance_level, 0.5)

    def get_bt_trait_info(self, variety_code: str) -> Dict[str, Any]:
        """Get BT trait information for a variety"""
        variety = self.get_variety_info(variety_code)
        if not variety:
            return {"bt_trait": "none", "effectiveness": 0.0}

        bt_trait = variety.get('bt_trait', 'none')
        if bt_trait == 'none':
            effectiveness = 0.0
        elif 'cry1ac_cry2ab' in bt_trait:
            effectiveness = 0.95  # Dual gene very effective
        elif 'cry1ac' in bt_trait:
            effectiveness = 0.85  # Single gene good
        else:
            effectiveness = 0.0

        return {
            "bt_trait": bt_trait,
            "effectiveness": effectiveness,
            "multi_gene": 'cry2ab' in bt_trait
        }

    def get_maturity_period(self, variety_code: str) -> int:
        """Get maturity period in days"""
        variety = self.get_variety_info(variety_code)
        return variety.get('maturity_days', 165) if variety else 165

    def get_optimal_conditions(self, variety_code: str) -> Dict[str, Any]:
        """Get optimal growing conditions for a variety"""
        variety = self.get_variety_info(variety_code)
        if not variety:
            return {
                'temperature_range': [25, 35],
                'water_requirement_mm': 650,
                'soil_types': ['black_cotton', 'clay_loam'],
                'climate_zone': 'irrigated'
            }

        return {
            'temperature_range': [25, 35],  # General cotton range
            'water_requirement_mm': variety['water_requirement_mm'],
            'soil_types': variety['soil_adaptability'],
            'climate_zone': variety['climate_zone']
        }

    def create_variety_dataset_for_state(self, state: str) -> pd.DataFrame:
        """Create a comprehensive dataset of cotton varieties for a state"""
        state_varieties = self.get_varieties_for_state(state)

        if not state_varieties:
            # Return general varieties
            state_varieties = ['F-1861', 'CIM-600', 'Surabhi']

        variety_data = []
        for variety in state_varieties:
            info = self.get_variety_info(variety)
            if info:
                variety_data.append(info)

        return pd.DataFrame(variety_data)

    def recommend_varieties(self, conditions: Dict[str, Any],
                           top_n: int = 5) -> List[Dict[str, Any]]:
        """Recommend best cotton varieties for given conditions"""

        recommendations = []

        for variety_code, info in self.variety_database.items():
            score = self._calculate_variety_score(variety_code, conditions)
            recommendations.append({
                'variety': variety_code,
                'name': info['name'],
                'state': info['state'],
                'yield_potential': info['yield_potential_quintal_ha'],
                'maturity_days': info['maturity_days'],
                'bt_trait': info['bt_trait'],
                'fiber_quality': f"{info['fiber_type']} ({info['fiber_length_mm']}mm)",
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

        # BT effectiveness for pest pressure
        bollworm_pressure = conditions.get('bollworm_pressure', 0.5)
        bt_info = self.get_bt_trait_info(variety_code)
        bt_score = bt_info['effectiveness'] * (1.0 - bollworm_pressure) + (bt_info['effectiveness'] * bollworm_pressure)
        scores.append(bt_score)

        # Irrigation compatibility (0-1)
        irrigation_available = conditions.get('irrigation_coverage', 0.5)
        water_req = variety['water_requirement_mm'] / 800  # Normalize
        irrigation_score = 1.0 - abs(irrigation_available - water_req)
        scores.append(irrigation_score)

        # Yield potential (normalized 0-1)
        yield_score = variety['yield_potential_quintal_ha'] / 35.0  # Max known yield
        scores.append(yield_score)

        # Fiber quality score (market preference)
        fiber_score = {'extra_long_staple': 1.0, 'long_staple': 0.9, 'medium_staple': 0.7}.get(
            variety['fiber_type'], 0.6)
        scores.append(fiber_score * 0.5)  # 50% weight

        # Recency bonus (newer varieties preferred)
        recency_score = min(variety['created_year'] / 2023, 1.0)
        scores.append(recency_score * 0.3)  # 30% weight

        # Average of all scores
        return sum(scores) / len(scores)

# Global cotton variety manager instance
cotton_variety_manager = CottonVarietyManager()

def main():
    """Test and demonstrate cotton variety functionality"""
    print("👔 COTTON VARIETY MANAGEMENT SYSTEM")
    print("=" * 50)

    # Test basic functionality
    print("\n📊 Testing Cotton Variety Database:")

    # Get Punjab varieties
    punjab_varieties = cotton_variety_manager.get_varieties_for_state('punjab')
    print(f"Punjab cotton varieties: {punjab_varieties}")

    # Test specific variety
    fim_1861_info = cotton_variety_manager.get_variety_info('F-1861')
    print(f"\nF-1861 variety details: {fim_1861_info}")

    # Test yield potential calculation
    test_conditions = {
        'irrigation_coverage': 0.9,
        'temperature_celsius': 30,
        'bollworm_pressure': 0.7,
        'disease_risk': 0.2
    }

    yield_potential = cotton_variety_manager.calculate_variety_yield_potential('F-1861', test_conditions)
    print(f"F-1861 yield potential in optimal conditions: {yield_potential} q/ha")

    # Test BT trait info
    bt_info = cotton_variety_manager.get_bt_trait_info('CIM-600')
    print(f"CIM-600 BT information: {bt_info}")

    # Test variety recommendations
    recommendations = cotton_variety_manager.recommend_varieties(test_conditions, top_n=3)
    print(f"\nTop 3 recommended varieties:")
    for rec in recommendations:
        print(f"   • {rec['variety']}: Score {rec['suitability_score']:.2f}, Yield {rec['yield_potential']} q/ha, BT: {rec['bt_trait']}")

    # Create variety dataset
    punjab_df = cotton_variety_manager.create_variety_dataset_for_state('punjab')
    print(f"\nPunjab cotton varieties dataset: {len(punjab_df)} varieties")

    print("\n✅ Cotton variety management system ready!")

if __name__ == "__main__":
    main()
