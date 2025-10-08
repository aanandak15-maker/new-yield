"""
Cotton Variety Database and Management System
Provides Indian cotton varieties with specifications and recommendations
"""

from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)

class CottonVarietyManager:
    """Manages Indian cotton varieties with yield potential and traits"""

    def __init__(self):
        self.varieties = self._load_varieties()

    def _load_varieties(self) -> Dict[str, Dict[str, Any]]:
        """Load Indian cotton variety database"""

        # Major Indian cotton varieties with their characteristics
        varieties = {
            # BT Cotton Varieties
            'F-1861': {
                'name': 'F-1861',
                'breed': 'hirsutum',
                'yield_potential_quintal_ha': 25,
                'maturity_days': 165,
                'bt_trait': 'cry1ac',
                'fiber_type': 'medium_staple',
                'fiber_length_mm': 27,
                'fiber_strength_g_tex': 28,
                'boll_opening_schedule': 'progressive',
                'disease_resistance': {
                    'bollworm': 'high',
                    'jassid': 'medium',
                    'aphid': 'medium'
                },
                'preferred_states': ['PUNJAB', 'HARYANA', 'MAHARASHTRA'],
                'water_requirement_mm': 900,
                'seed_cost_per_kg': 850,
                'market_premium': True
            },

            'RCH-659': {
                'name': 'RCH-659',
                'breed': 'hirsutum',
                'yield_potential_quintal_ha': 28,
                'maturity_days': 170,
                'bt_trait': 'cry1ac_cry2ab',
                'fiber_type': 'medium_staple',
                'fiber_length_mm': 26,
                'fiber_strength_g_tex': 27,
                'boll_opening_schedule': 'progressive',
                'disease_resistance': {
                    'bollworm': 'very_high',
                    'jassid': 'medium',
                    'whitefly': 'high'
                },
                'preferred_states': ['GUJARAT', 'MAHARASHTRA'],
                'water_requirement_mm': 950,
                'seed_cost_per_kg': 920,
                'market_premium': True
            },

            'NCS-207': {
                'name': 'NCS-207',
                'breed': 'barbadense',
                'yield_potential_quintal_ha': 22,
                'maturity_days': 175,
                'bt_trait': 'cry1ac',
                'fiber_type': 'extra_long_staple',
                'fiber_length_mm': 32,
                'fiber_strength_g_tex': 32,
                'boll_opening_schedule': 'simultaneous',
                'disease_resistance': {
                    'bollworm': 'high',
                    'fungal_diseases': 'high',
                    'bacterial_blight': 'medium'
                },
                'preferred_states': ['TAMIL_NADU', 'ANDHRA_PRADESH'],
                'water_requirement_mm': 800,
                    'seed_cost_per_kg': 1200,
                'market_premium': True
            },

            # Non-BT Varieties
            'Surabhi': {
                'name': 'Surabhi',
                'breed': 'barbadense',
                'yield_potential_quintal_ha': 24,
                'maturity_days': 180,
                'bt_trait': 'none',
                'fiber_type': 'extra_long_staple',
                'fiber_length_mm': 31,
                'fiber_strength_g_tex': 34,
                'boll_opening_schedule': 'progressive',
                'disease_resistance': {
                    'bacterial_blight': 'high',
                    'fungal_diseases': 'medium',
                    'bollworm': 'low'
                },
                'preferred_states': ['TAMIL_NADU', 'KARNATAKA'],
                'water_requirement_mm': 850,
                'seed_cost_per_kg': 950,
                'market_premium': True
            },

            'MCU-5': {
                'name': 'MCU-5',
                'breed': 'hirsutum',
                'yield_potential_quintal_ha': 20,
                'maturity_days': 160,
                'bt_trait': 'none',
                'fiber_type': 'medium_staple',
                'fiber_length_mm': 25,
                'fiber_strength_g_tex': 26,
                'boll_opening_schedule': 'progressive',
                'disease_resistance': {
                    'jassid': 'medium',
                    'aphid': 'medium',
                    'bollworm': 'low'
                },
                'preferred_states': ['TN', 'KA', 'AP'],
                'water_requirement_mm': 750,
                'seed_cost_per_kg': 400,
                'market_premium': False
            },

            'PAU-114': {
                'name': 'PAU-114',
                'breed': 'hirsutum',
                'yield_potential_quintal_ha': 18,
                'maturity_days': 155,
                'bt_trait': 'none',
                'fiber_type': 'medium_staple',
                'fiber_length_mm': 26,
                'fiber_strength_g_tex': 27,
                'boll_opening_schedule': 'simultaneous',
                'disease_resistance': {
                    'bollworm': 'medium',
                    'jassid': 'low',
                    'aphid': 'medium'
                },
                'preferred_states': ['PUNJAB', 'HARYANA'],
                'water_requirement_mm': 700,
                'seed_cost_per_kg': 350,
                'market_premium': False
            },

            'RCH-2': {
                'name': 'RCH-2',
                'breed': 'hirsutum',
                'yield_potential_quintal_ha': 26,
                'maturity_days': 168,
                'bt_trait': 'cry1ac_cry2ab',
                'fiber_type': 'medium_staple',
                'fiber_length_mm': 28,
                'fiber_strength_g_tex': 29,
                'boll_opening_schedule': 'progressive',
                'disease_resistance': {
                    'bollworm': 'very_high',
                    'whitefly': 'high',
                    'fungal_diseases': 'medium'
                },
                'preferred_states': ['MAHARASHTRA', 'GUJARAT', 'ANDHRA_PRADESH'],
                'water_requirement_mm': 1000,
                'seed_cost_per_kg': 1100,
                'market_premium': True
            },

            'SHR-6': {
                'name': 'SHR-6',
                'breed': 'hirsutum',
                'yield_potential_quintal_ha': 23,
                'maturity_days': 175,
                'bt_trait': 'cry1ac',
                'fiber_type': 'medium_staple',
                'fiber_length_mm': 27,
                'fiber_strength_g_tex': 28,
                'boll_opening_schedule': 'progressive',
                'disease_resistance': {
                    'bollworm': 'high',
                    'jassid': 'medium',
                    'grey_mildew': 'low'
                },
                'preferred_states': ['GUJARAT', 'MAHARASHTRA'],
                'water_requirement_mm': 900,
                'seed_cost_per_kg': 800,
                'market_premium': True
            },

            'AKH-081': {
                'name': 'AKH-081',
                'breed': 'hirsutum',
                'yield_potential_quintal_ha': 27,
                'maturity_days': 172,
                'bt_trait': 'cry1ac_eee',
                'fiber_type': 'medium_staple',
                'fiber_length_mm': 28,
                'fiber_strength_g_tex': 30,
                'boll_opening_schedule': 'progressive',
                'disease_resistance': {
                    'bollworm': 'very_high',
                    'whitefly': 'very_high',
                    'jassid': 'high'
                },
                'preferred_states': ['MAHARASHTRA', 'GUJARAT', 'KARNATAKA'],
                'water_requirement_mm': 1100,
                'seed_cost_per_kg': 1300,
                'market_premium': True
            }
        }

        logger.info(f"Loaded {len(varieties)} Indian cotton varieties")
        return varieties

    def get_variety_info(self, variety_name: str) -> Optional[Dict[str, Any]]:
        """Get complete information for a specific variety"""

        if variety_name in self.varieties:
            return self.varieties[variety_name].copy()

        # Try case-insensitive search
        for name, info in self.varieties.items():
            if name.lower() == variety_name.lower():
                return info.copy()

        logger.warning(f"Cotton variety '{variety_name}' not found in database")
        return None

    def get_varieties_for_state(self, state: str) -> List[Dict[str, Any]]:
        """Get recommended varieties for a specific state"""

        state_varieties = []
        for variety_info in self.varieties.values():
            if state.upper() in variety_info.get('preferred_states', []):
                state_varieties.append(variety_info.copy())

        return state_varieties

    def recommend_varieties(self, conditions: Dict[str, Any], top_n: int = 3) -> List[Dict[str, Any]]:
        """
        Recommend varieties based on growing conditions

        Args:
            conditions: Dictionary containing:
                - irrigation_coverage (0-1)
                - temperature_celsius
                - bollworm_pressure (0-1)
                - state: State name
        """

        recommendations = []

        for variety_info in self.varieties.values():
            score = self._calculate_variety_score(variety_info, conditions)
            recommendation = variety_info.copy()
            recommendation['suitability_score'] = score
            recommendations.append(recommendation)

        # Sort by suitability score
        recommendations.sort(key=lambda x: x['suitability_score'], reverse=True)

        return recommendations[:top_n]

    def _calculate_variety_score(self, variety_info: Dict[str, Any],
                                conditions: Dict[str, Any]) -> float:
        """Calculate suitability score for a variety given conditions"""

        score = 0.0

        # State preference (high weight)
        state = conditions.get('state', '').upper()
        if state in variety_info.get('preferred_states', []):
            score += 30

        # Irrigation compatibility
        irrigation = conditions.get('irrigation_coverage', 0.5)
        water_req = variety_info.get('water_requirement_mm', 900)
        irrigation_index = irrigation * 1000  # Convert to mm equivalent

        if irrigation_index >= water_req * 0.8:
            score += 25  # Adequate water
        elif irrigation_index >= water_req * 0.6:
            score += 15  # Moderate deficit
        else:
            score += 5   # Water stress

        # Pest pressure compatibility
        bollworm_pressure = conditions.get('bollworm_pressure', 0.5)
        bt_trait = variety_info.get('bt_trait', 'none')

        if bt_trait != 'none':
            # BT varieties perform better under pest pressure
            score += 20 + (bollworm_pressure * 10)
        else:
            # Non-BT varieties: penalty for high pest pressure
            score += max(0, 15 - (bollworm_pressure * 20))

        # Temperature suitability (preference for tropical varieties)
        temperature = conditions.get('temperature_celsius', 28)
        if 25 <= temperature <= 32:
            score += 15
        elif 20 <= temperature <= 35:
            score += 10

        # Maturity season alignment (assuming kharif)
        maturity = variety_info.get('maturity_days', 165)
        if 150 <= maturity <= 180:
            score += 10

        return score

    def get_all_varieties(self) -> List[str]:
        """Get list of all available variety names"""
        return list(self.varieties.keys())

    def get_variety_characteristics(self, variety_name: str) -> Optional[Dict[str, Any]]:
        """Get key characteristics for a variety"""

        info = self.get_variety_info(variety_name)
        if not info:
            return None

        return {
            'yield_potential': info.get('yield_potential_quintal_ha'),
            'maturity': info.get('maturity_days'),
            'fiber_quality': info.get('fiber_type'),
            'bt_technology': info.get('bt_trait') != 'none',
            'disease_resistance': info.get('disease_resistance', {}),
            'water_requirement': info.get('water_requirement_mm'),
            'cost_category': 'premium' if info.get('market_premium') else 'standard'
        }

    def get_bt_varieties(self) -> List[str]:
        """Get list of BT cotton varieties"""
        return [name for name, info in self.varieties.items()
                if info.get('bt_trait', 'none') != 'none']

    def get_premium_varieties(self) -> List[str]:
        """Get list of premium (high market value) varieties"""
        return [name for name, info in self.varieties.items()
                if info.get('market_premium', False)]

# Global cotton variety manager instance
cotton_variety_manager = CottonVarietyManager()

def get_cotton_variety_manager() -> CottonVarietyManager:
    """Get global cotton variety manager instance"""
    return cotton_variety_manager
