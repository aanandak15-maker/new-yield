"""
Maize Variety Characteristics Database
Comprehensive maize/corn hybrid variety profiles for multi-state yield modeling
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from pathlib import Path

class MaizeVarietyManager:
    """Manages maize variety characteristics and profiles across India"""

    def __init__(self):
        # Major maize varieties by region
        self.variety_database = self._load_maize_varieties()

    def _load_maize_varieties(self) -> Dict[str, Dict[str, Any]]:
        """Load comprehensive maize variety database"""
        return {
            # Karnataka Maize Hybrids
            "DKC-9108": {
                "name": "DKC-9108",
                "state": "KARNATAKA",
                "maturity_days": 105,
                "yield_potential_quintal_ha": 75,
                "hybrid_type": "single_cross",
                "grain_type": "dent_corn",
                "pest_resistance": {
                    "corn_borer": "high",
                    "fall_armyworm": "moderate",
                    "stem_borer": "high"
                },
                "water_requirement_mm": 550,
                "soil_adaptability": ["sandy_loam", "loam", "clay_loam"],
                "climate_zone": "south_india_rainfed",
                "market_preference": "processing_industry",
                "qr_code_compliant": True,
                "created_year": 2018,
                "starch_content": 68,
                "protein_content": 8.5,
                "oil_content": 4.2
            },
            "DKC-9142": {
                "name": "DKC-9142",
                "state": "KARNATAKA",
                "maturity_days": 110,
                "yield_potential_quintal_ha": 72,
                "hybrid_type": "single_cross",
                "grain_type": "dent_corn",
                "pest_resistance": {
                    "corn_borer": "very_high",
                    "fall_armyworm": "high",
                    "stem_borer": "high"
                },
                "water_requirement_mm": 580,
                "soil_adaptability": ["loam", "clay_loam"],
                "climate_zone": "south_india_irrigated",
                "market_preference": "fresh_corn",
                "qr_code_compliant": True,
                "created_year": 2020,
                "starch_content": 67,
                "protein_content": 9.0,
                "oil_content": 4.5
            },
            "30V92": {
                "name": "30V92",
                "state": "KARNATAKA",
                "maturity_days": 115,
                "yield_potential_quintal_ha": 70,
                "hybrid_type": "single_cross",
                "grain_type": "dent_corn",
                "pest_resistance": {
                    "corn_borer": "high",
                    "fall_armyworm": "moderate",
                    "stem_borer": "moderate"
                },
                "water_requirement_mm": 520,
                "soil_adaptability": ["sandy_loam", "loam"],
                "climate_zone": "south_india_rainfed",
                "market_preference": "dual_purpose",
                "qr_code_compliant": True,
                "created_year": 2014,
                "starch_content": 65,
                "protein_content": 8.8,
                "oil_content": 4.0
            },
            # Maharashtra Maize Hybrids
            "30F53": {
                "name": "30F53",
                "state": "MAHARASHTRA",
                "maturity_days": 95,
                "yield_potential_quintal_ha": 68,
                "hybrid_type": "single_cross",
                "grain_type": "dent_corn",
                "pest_resistance": {
                    "corn_borer": "very_high",
                    "fall_armyworm": "high",
                    "stem_borer": "high"
                },
                "water_requirement_mm": 500,
                "soil_adaptability": ["black_cotton", "deep_black"],
                "climate_zone": "central_india_rainfed",
                "market_preference": "food_grain",
                "qr_code_compliant": True,
                "created_year": 2016,
                "starch_content": 66,
                "protein_content": 9.2,
                "oil_content": 4.3
            },
            "NK-6240": {
                "name": "NK-6240",
                "state": "MAHARASHTRA",
                "maturity_days": 100,
                "yield_potential_quintal_ha": 65,
                "hybrid_type": "single_cross",
                "grain_type": "dent_corn",
                "pest_resistance": {
                    "corn_borer": "high",
                    "fall_armyworm": "moderate",
                    "stem_borer": "high"
                },
                "water_requirement_mm": 480,
                "soil_adaptability": ["black_cotton", "red_soil"],
                "climate_zone": "central_india_mixed",
                "market_preference": "food_grain",
                "qr_code_compliant": True,
                "created_year": 2015,
                "starch_content": 64,
                "protein_content": 9.5,
                "oil_content": 4.1
            },
            # Andhra Pradesh Maize Hybrids
            "PAC-740": {
                "name": "PAC-740",
                "state": "ANDHRA_PRADESH",
                "maturity_days": 120,
                "yield_potential_quintal_ha": 60,
                "hybrid_type": "single_cross",
                "grain_type": "dent_corn",
                "pest_resistance": {
                    "corn_borer": "moderate",
                    "fall_armyworm": "moderate",
                    "stem_borer": "moderate"
                },
                "water_requirement_mm": 550,
                "soil_adaptability": ["red_soil", "black_soil"],
                "climate_zone": "south_india_mixed",
                "market_preference": "mixed_use",
                "qr_code_compliant": True,
                "created_year": 2013,
                "starch_content": 63,
                "protein_content": 8.7,
                "oil_content": 3.8
            },
            # Bihar Maize Hybrids
            "HQPM-1": {
                "name": "HQPM-1",
                "state": "BIHAR",
                "maturity_days": 125,
                "yield_potential_quintal_ha": 55,
                "hybrid_type": "quality_protein_maize",
                "grain_type": "qpm_dent_corn",
                "pest_resistance": {
                    "corn_borer": "moderate",
                    "fall_armyworm": "low",
                    "stem_borer": "low"
                },
                "water_requirement_mm": 450,
                "soil_adaptability": ["clay", "silty_clay"],
                "climate_zone": "east_india_rainfed",
                "market_preference": "nutrition_focus",
                "qr_code_compliant": False,
                "created_year": 2008,
                "protein_content": 12.0,  # Higher protein
                "tryptophan_content": 0.08,
                "lysine_content": 0.35
            },
            # Uttar Pradesh Maize Hybrids
            "30V92": {
                "name": "30V92",
                "state": "UTTAR_PRADESH",
                "maturity_days": 130,
                "yield_potential_quintal_ha": 52,
                "hybrid_type": "single_cross",
                "grain_type": "dent_corn",
                "pest_resistance": {
                    "corn_borer": "low",
                    "fall_armyworm": "low",
                    "stem_borer": "moderate"
                },
                "water_requirement_mm": 480,
                "soil_adaptability": ["alluvial", "loam"],
                "climate_zone": "north_india_irrigated",
                "market_preference": "food_security",
                "qr_code_compliant": True,
                "created_year": 2012,
                "starch_content": 64,
                "protein_content": 8.9,
                "oil_content": 4.0
            },
            # Madhya Pradesh Maize Hybrids
            "PAC-740": {
                "name": "PAC-740",
                "state": "MADHYA_PRADESH",
                "maturity_days": 118,
                "yield_potential_quintal_ha": 58,
                "hybrid_type": "single_cross",
                "grain_type": "dent_corn",
                "pest_resistance": {
                    "corn_borer": "moderate",
                    "fall_armyworm": "low",
                    "stem_borer": "moderate"
                },
                "water_requirement_mm": 470,
                "soil_adaptability": ["black_soil", "red_soil"],
                "climate_zone": "central_india_rainfed",
                "market_preference": "local_markets",
                "qr_code_compliant": True,
                "created_year": 2010,
                "starch_content": 63,
                "protein_content": 8.8,
                "oil_content": 4.0
            },
            # Tamil Nadu Maize Hybrids (Baby Corn Special)
            "CP-818": {
                "name": "CP-818",
                "state": "TAMIL_NADU",
                "maturity_days": 65,
                "yield_potential_quintal_ha": 45,
                "hybrid_type": "baby_corn_special",
                "grain_type": "sweet_corn",
                "pest_resistance": {
                    "corn_borer": "high",
                    "fall_armyworm": "high",
                    "stem_borer": "high"
                },
                "water_requirement_mm": 600,
                "soil_adaptability": ["sandy_loam", "loam"],
                "climate_zone": "south_india_irrigated",
                "market_preference": "vegetable_market",
                "qr_code_compliant": True,
                "created_year": 2021,
                "sugar_content": 8.5,
                "baby_corn_yield": 800,  # kg/ha
                "baby_corn_days": 25
            },
            "Kaveri Gold": {
                "name": "Kaveri Gold",
                "state": "TAMIL_NADU",
                "maturity_days": 90,
                "yield_potential_quintal_ha": 55,
                "hybrid_type": "sweet_corn",
                "grain_type": "sweet_corn",
                "pest_resistance": {
                    "corn_borer": "very_high",
                    "fall_armyworm": "moderate",
                    "stem_borer": "high"
                },
                "water_requirement_mm": 650,
                "soil_adaptability": ["loam", "silt_loam"],
                "climate_zone": "south_india_irrigated",
                "market_preference": "export_quality",
                "qr_code_compliant": True,
                "created_year": 2019,
                "sugar_content": 12.5,
                "vitamin_a_content": 15
            },
            # Gujarat Maize Hybrids (Salt Tolerant)
            "Ganga-5": {
                "name": "Ganga-5",
                "state": "GUJARAT",
                "maturity_days": 105,
                "yield_potential_quintal_ha": 45,
                "hybrid_type": "salt_tolerant",
                "grain_type": "dent_corn",
                "pest_resistance": {
                    "corn_borer": "moderate",
                    "fall_armyworm": "low",
                    "stem_borer": "moderate"
                },
                "water_requirement_mm": 550,
                "soil_adaptability": ["saline_soil", "sodic_soil"],
                "climate_zone": "west_india_coastal",
                "market_preference": "coastal_areas",
                "salt_tolerance_ec": 8.0,
                "qr_code_compliant": True,
                "created_year": 2017
            }
        }

    def get_variety_info(self, variety_code: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific maize variety"""
        return self.variety_database.get(variety_code)

    def get_varieties_for_state(self, state: str) -> List[str]:
        """Get all maize varieties available in a specific state"""
        state_upper = state.upper()
        return [v for v, info in self.variety_database.items()
                if info['state'] == state_upper]

    def get_varieties_for_climate_zone(self, zone: str) -> List[str]:
        """Get maize varieties suitable for a climate zone"""
        return [v for v, info in self.variety_database.items()
                if info['climate_zone'] == zone]

    def calculate_variety_yield_potential(self, variety_code: str,
                                         environmental_factors: Dict[str, Any]) -> float:
        """Calculate yield potential for a variety under specific conditions"""

        variety = self.get_variety_info(variety_code)
        if not variety:
            return 25.0  # Default average maize yield

        base_yield = variety['yield_potential_quintal_ha']

        # Adjust for environmental factors
        adjustment_factors = []

        # Heat stress adjustment (maize is highly temperature-sensitive)
        temperature = environmental_factors.get('temperature_celsius', 28)
        if temperature > 35:
            heat_stress = (temperature - 35) * 0.03  # 3% yield loss per degree above 35°C
            heat_adjustment = 1.0 - min(heat_stress, 0.5)  # Max 50% loss
            adjustment_factors.append(heat_adjustment)

        # Pest pressure adjustment
        corn_borer_pressure = environmental_factors.get('corn_borer_pressure', 0.4)
        borer_resistance = self.get_pest_resistance_score(variety_code, 'corn_borer')
        resistance_map = {'low': 0.3, 'moderate': 0.6, 'high': 0.8, 'very_high': 0.9}
        pest_adjustment = 1.0 - (corn_borer_pressure * (1.0 - resistance_map.get(borer_resistance, 0.5)))
        adjustment_factors.append(pest_adjustment)

        # Irrigation adjustment (maize needs regular water)
        irrigation_coverage = environmental_factors.get('irrigation_coverage', 0.5)
        irrigation_adjustment = 0.75 + (irrigation_coverage * 0.5)  # 0.75 to 1.25
        adjustment_factors.append(irrigation_adjustment)

        # Hybrid premium
        hybrid_premium = 1.15 if variety.get('hybrid_type') != 'open_pollinated' else 1.0
        adjustment_factors.append(hybrid_premium)

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

        resistance_level = variety['pest_resistance'].get(pest_type, 'moderate')
        resistance_map = {
            'low': 0.3,
            'moderate': 0.6,
            'high': 0.8,
            'very_high': 0.9
        }
        return resistance_map.get(resistance_level, 0.5)

    def get_maturity_period(self, variety_code: str) -> int:
        """Get maturity period in days"""
        variety = self.get_variety_info(variety_code)
        return variety.get('maturity_days', 105) if variety else 105

    def get_optimal_conditions(self, variety_code: str) -> Dict[str, Any]:
        """Get optimal growing conditions for a variety"""
        variety = self.get_variety_info(variety_code)
        if not variety:
            return {
                'temperature_range': [20, 35],
                'water_requirement_mm': 550,
                'soil_types': ['sandy_loam', 'loam'],
                'climate_zone': 'rainfed'
            }

        return {
            'temperature_range': [15, 35],  # Maize temperature range
            'water_requirement_mm': variety['water_requirement_mm'],
            'soil_types': variety['soil_adaptability'],
            'climate_zone': variety['climate_zone']
        }

    def create_variety_dataset_for_state(self, state: str) -> pd.DataFrame:
        """Create a comprehensive dataset of maize varieties for a state"""
        state_varieties = self.get_varieties_for_state(state)

        if not state_varieties:
            # Return general varieties
            state_varieties = ['DKC-9108', 'PAC-740', '30V92']

        variety_data = []
        for variety in state_varieties:
            info = self.get_variety_info(variety)
            if info:
                variety_data.append(info)

        return pd.DataFrame(variety_data)

    def recommend_varieties(self, conditions: Dict[str, Any],
                           top_n: int = 5) -> List[Dict[str, Any]]:
        """Recommend best maize varieties for given conditions"""

        recommendations = []

        for variety_code, info in self.variety_database.items():
            score = self._calculate_variety_score(variety_code, conditions)
            recommendations.append({
                'variety': variety_code,
                'name': info['name'],
                'state': info['state'],
                'yield_potential': info['yield_potential_quintal_ha'],
                'maturity_days': info['maturity_days'],
                'hybrid_type': info['hybrid_type'],
                'grain_type': info['grain_type'],
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

        # Temperature suitability (critical for maize)
        temperature = conditions.get('temperature_celsius', 28)
        if 20 <= temperature <= 35:
            temp_score = 1.0
        elif temperature < 20 or temperature > 40:
            temp_score = 0.7
        else:
            temp_score = 0.9
        scores.append(temp_score)

        # Pest resistance score
        corn_borer_pressure = conditions.get('corn_borer_pressure', 0.4)
        pest_score = 1.0 - (corn_borer_pressure * 0.5)  # Penalize high pest areas
        scores.append(pest_score)

        # Hybrid advantage
        is_hybrid = 1.2 if variety.get('hybrid_type') != 'open_pollinated' else 0.8
        scores.append(is_hybrid)

        # Yield potential
        yield_score = variety['yield_potential_quintal_ha'] / 75.0  # Max known yield
        scores.append(yield_score)

        # Recency bonus (newer varieties preferred)
        recency_score = min(variety['created_year'] / 2023, 1.0)
        scores.append(recency_score * 0.5)  # 50% weight

        # Average of all scores
        return sum(scores) / len(scores)

# Global maize variety manager instance
maize_variety_manager = MaizeVarietyManager()

def main():
    """Test and demonstrate maize variety functionality"""
    print("🌽 MAIZE VARIETY MANAGEMENT SYSTEM")
    print("=" * 50)

    # Test basic functionality
    print("\n📊 Testing Maize Variety Database:")

    # Get Karnataka varieties
    karnataka_varieties = maize_variety_manager.get_varieties_for_state('karnataka')
    print(f"Karnataka maize varieties: {karnataka_varieties}")

    # Test specific variety
    dkc_9108_info = maize_variety_manager.get_variety_info('DKC-9108')
    print(f"\nDKC-9108 variety details: {dkc_9108_info}")

    # Test yield potential calculation
    test_conditions = {
        'temperature_celsius': 32,
        'corn_borer_pressure': 0.6,
        'irrigation_coverage': 0.7
    }

    yield_potential = maize_variety_manager.calculate_variety_yield_potential('DKC-9108', test_conditions)
    print(f"DKC-9108 yield potential in optimal conditions: {yield_potential} q/ha")

    # Test pest resistance
    pest_score = maize_variety_manager.get_pest_resistance_score('DKC-9108', 'corn_borer')
    print(f"DKC-9108 corn borer resistance: {pest_score}")

    # Test variety recommendations
    recommendations = maize_variety_manager.recommend_varieties(test_conditions, top_n=3)
    print(f"\nTop 3 recommended varieties:")
    for rec in recommendations:
        print(f"   • {rec['variety']}: Score {rec['suitability_score']:.2f}, Yield {rec['yield_potential']} q/ha, Type: {rec['hybrid_type']}")

    # Create variety dataset
    karnataka_df = maize_variety_manager.create_variety_dataset_for_state('karnataka')
    print(f"\nKarnataka maize varieties dataset: {len(karnataka_df)} varieties")

    print("\n✅ Maize variety management system ready!")

if __name__ == "__main__":
    main()
