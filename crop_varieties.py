"""
Punjab Wheat Varieties Database
SIH 2025 - Agricultural Enhancement Module

This module contains detailed information about wheat varieties grown in Punjab,
including yield potential, disease resistance, and agronomic characteristics.
Data sourced from Punjab Agricultural University and agricultural research.
"""

import pandas as pd
import numpy as np

# Punjab Wheat Varieties Database
PUNJAB_WHEAT_VARIETIES = {
    # High-yielding varieties (Primary choice for farmers)
    'HD_3086': {
        'name': 'HD 3086',
        'yield_potential_quintal_ha': 55,  # Quintal per hectare
        'maturity_days': 155,  # Days to maturity
        'disease_resistance': {
            'yellow_rust': 0.9,    # 0-1 scale (1 = highly resistant)
            'brown_rust': 0.8,
            'black_rust': 0.7,
            'powdery_mildew': 0.8,
            'aphids': 0.6,
            'termites': 0.7
        },
        'drought_tolerance': 0.7,    # 0-1 scale
        'heat_tolerance': 0.8,
        'cold_tolerance': 0.9,
        'grain_quality': 0.8,        # Milling and baking quality
        'plant_height_cm': 95,
        'release_year': 2016,
        'suitable_zones': ['North-Western', 'Central', 'North-Eastern']
    },

    'PBW_725': {
        'name': 'PBW 725',
        'yield_potential_quintal_ha': 52,
        'maturity_days': 150,
        'disease_resistance': {
            'yellow_rust': 0.8,
            'brown_rust': 0.9,
            'black_rust': 0.6,
            'powdery_mildew': 0.7,
            'aphids': 0.8,
            'termites': 0.5
        },
        'drought_tolerance': 0.8,
        'heat_tolerance': 0.7,
        'cold_tolerance': 0.8,
        'grain_quality': 0.9,
        'plant_height_cm': 90,
        'release_year': 2018,
        'suitable_zones': ['Central', 'South-Western']
    },

    'HD_2967': {
        'name': 'HD 2967',
        'yield_potential_quintal_ha': 50,
        'maturity_days': 145,
        'disease_resistance': {
            'yellow_rust': 0.7,
            'brown_rust': 0.8,
            'black_rust': 0.9,
            'powdery_mildew': 0.6,
            'aphids': 0.7,
            'termites': 0.8
        },
        'drought_tolerance': 0.9,    # Excellent drought tolerance
        'heat_tolerance': 0.6,
        'cold_tolerance': 0.7,
        'grain_quality': 0.7,
        'plant_height_cm': 85,
        'release_year': 2015,
        'suitable_zones': ['South-Western', 'North-Western']
    },

    'DBW_187': {
        'name': 'DBW 187',
        'yield_potential_quintal_ha': 48,
        'maturity_days': 140,
        'disease_resistance': {
            'yellow_rust': 0.9,
            'brown_rust': 0.9,
            'black_rust': 0.8,
            'powdery_mildew': 0.9,
            'aphids': 0.8,
            'termites': 0.6
        },
        'drought_tolerance': 0.6,
        'heat_tolerance': 0.8,
        'cold_tolerance': 0.9,
        'grain_quality': 0.8,
        'plant_height_cm': 88,
        'release_year': 2019,
        'suitable_zones': ['Central', 'North-Eastern']
    },

    'WH_1105': {
        'name': 'WH 1105',
        'yield_potential_quintal_ha': 45,
        'maturity_days': 135,  # Early maturing
        'disease_resistance': {
            'yellow_rust': 0.6,
            'brown_rust': 0.7,
            'black_rust': 0.5,
            'powdery_mildew': 0.8,
            'aphids': 0.9,
            'termites': 0.7
        },
        'drought_tolerance': 0.8,
        'heat_tolerance': 0.7,
        'cold_tolerance': 0.6,
        'grain_quality': 0.6,
        'plant_height_cm': 82,
        'release_year': 2017,
        'suitable_zones': ['North-Western', 'South-Western']
    },

    'PBW_752': {
        'name': 'PBW 752',
        'yield_potential_quintal_ha': 53,
        'maturity_days': 152,
        'disease_resistance': {
            'yellow_rust': 0.8,
            'brown_rust': 0.7,
            'black_rust': 0.8,
            'powdery_mildew': 0.9,
            'aphids': 0.6,
            'termites': 0.8
        },
        'drought_tolerance': 0.7,
        'heat_tolerance': 0.9,
        'cold_tolerance': 0.8,
        'grain_quality': 0.9,
        'plant_height_cm': 92,
        'release_year': 2020,
        'suitable_zones': ['Central', 'North-Eastern']
    },

    'HD_3059': {
        'name': 'HD 3059',
        'yield_potential_quintal_ha': 47,
        'maturity_days': 148,
        'disease_resistance': {
            'yellow_rust': 0.7,
            'brown_rust': 0.8,
            'black_rust': 0.9,
            'powdery_mildew': 0.7,
            'aphids': 0.8,
            'termites': 0.9
        },
        'drought_tolerance': 0.8,
        'heat_tolerance': 0.6,
        'cold_tolerance': 0.8,
        'grain_quality': 0.7,
        'plant_height_cm': 87,
        'release_year': 2014,
        'suitable_zones': ['North-Eastern', 'Central']
    },

    'VL_907': {
        'name': 'VL 907',
        'yield_potential_quintal_ha': 44,
        'maturity_days': 142,
        'disease_resistance': {
            'yellow_rust': 0.8,
            'brown_rust': 0.9,
            'black_rust': 0.7,
            'powdery_mildew': 0.8,
            'aphids': 0.7,
            'termites': 0.6
        },
        'drought_tolerance': 0.7,
        'heat_tolerance': 0.7,
        'cold_tolerance': 0.9,
        'grain_quality': 0.8,
        'plant_height_cm': 84,
        'release_year': 2016,
        'suitable_zones': ['North-Eastern']
    }
}

# Crop coefficient data for wheat (FAO methodology)
WHEAT_CROP_COEFFICIENTS = {
    'initial': 0.3,      # Initial stage (0-20 days)
    'development': 0.7,  # Development stage (20-90 days)
    'mid_season': 1.15,  # Mid-season stage (90-130 days)
    'late_season': 0.6   # Late season stage (130-end)
}

def get_variety_info(variety_code):
    """Get detailed information about a specific wheat variety"""
    return PUNJAB_WHEAT_VARIETIES.get(variety_code, {})

def get_varieties_for_zone(zone_name):
    """Get varieties suitable for a specific agro-climatic zone"""
    # Map full zone names to short names used in variety data
    zone_mapping = {
        'North-Western Zone': 'North-Western',
        'Central Zone': 'Central',
        'South-Western Zone': 'South-Western',
        'North-Eastern Zone': 'North-Eastern'
    }

    short_zone_name = zone_mapping.get(zone_name, zone_name)

    suitable_varieties = []
    for code, info in PUNJAB_WHEAT_VARIETIES.items():
        if short_zone_name in info['suitable_zones']:
            suitable_varieties.append(code)
    return suitable_varieties

def calculate_variety_yield_potential(variety_code, environmental_factors):
    """
    Calculate adjusted yield potential based on environmental conditions

    Parameters:
    - variety_code: Wheat variety code
    - environmental_factors: Dict with stress factors (0-1 scale)
        - drought_stress: 0-1 (1 = severe drought)
        - disease_pressure: 0-1 (1 = high disease incidence)
        - heat_stress: 0-1 (1 = severe heat)
    """
    if variety_code not in PUNJAB_WHEAT_VARIETIES:
        return 0

    variety = PUNJAB_WHEAT_VARIETIES[variety_code]
    base_yield = variety['yield_potential_quintal_ha']

    # Environmental stress adjustments
    drought_factor = 1 - (environmental_factors.get('drought_stress', 0) *
                         (1 - variety['drought_tolerance']))
    disease_factor = 1 - (environmental_factors.get('disease_pressure', 0) *
                          (1 - np.mean(list(variety['disease_resistance'].values()))))
    heat_factor = 1 - (environmental_factors.get('heat_stress', 0) *
                      (1 - variety['heat_tolerance']))

    adjusted_yield = base_yield * drought_factor * disease_factor * heat_factor
    return max(0, adjusted_yield)

def get_disease_resistance_score(variety_code, disease_type):
    """Get disease resistance score for specific disease"""
    variety = get_variety_info(variety_code)
    if not variety or 'disease_resistance' not in variety:
        return 0.5  # Default moderate resistance

    return variety['disease_resistance'].get(disease_type, 0.5)

def get_maturity_period(variety_code):
    """Get maturity period in days"""
    variety = get_variety_info(variety_code)
    return variety.get('maturity_days', 150)  # Default 150 days

def create_variety_dataset_for_punjab():
    """Create a dataset with variety information for Punjab districts"""
    variety_data = []

    # Punjab agro-climatic zones
    zones = ['North-Western Zone', 'Central Zone', 'South-Western Zone', 'North-Eastern Zone']

    for zone in zones:
        zone_varieties = get_varieties_for_zone(zone)

        for variety_code in zone_varieties:
            variety_info = get_variety_info(variety_code)

            # Create sample data for different districts in the zone
            zone_districts = {
                'North-Western Zone': ['Amritsar', 'Tarn Taran', 'Ferozepur'],
                'Central Zone': ['Ludhiana', 'Jalandhar', 'Patiala'],
                'South-Western Zone': ['Bathinda', 'Mansa', 'Sri Muktsar Sahib'],
                'North-Eastern Zone': ['Gurdaspur', 'Hoshiarpur', 'Rupnagar']
            }

            for district in zone_districts.get(zone, []):
                variety_data.append({
                    'district': district,
                    'variety_code': variety_code,
                    'variety_name': variety_info.get('name', variety_code),
                    'yield_potential_quintal_ha': variety_info.get('yield_potential_quintal_ha', 45),
                    'maturity_days': variety_info.get('maturity_days', 150),
                    'drought_tolerance': variety_info.get('drought_tolerance', 0.7),
                    'heat_tolerance': variety_info.get('heat_tolerance', 0.7),
                    'cold_tolerance': variety_info.get('cold_tolerance', 0.8),
                    'grain_quality': variety_info.get('grain_quality', 0.7),
                    'yellow_rust_resistance': variety_info.get('disease_resistance', {}).get('yellow_rust', 0.7),
                    'brown_rust_resistance': variety_info.get('disease_resistance', {}).get('brown_rust', 0.7),
                    'aphid_resistance': variety_info.get('disease_resistance', {}).get('aphids', 0.7),
                    'release_year': variety_info.get('release_year', 2015),
                    'zone': zone
                })

    df = pd.DataFrame(variety_data)
    return df

if __name__ == "__main__":
    # Test the variety database
    print("Punjab Wheat Varieties Database")
    print("=" * 40)

    # Show available varieties
    print(f"Total varieties: {len(PUNJAB_WHEAT_VARIETIES)}")

    # Show varieties by zone
    for zone in ['North-Western Zone', 'Central Zone', 'South-Western Zone', 'North-Eastern Zone']:
        varieties = get_varieties_for_zone(zone)
        print(f"{zone}: {len(varieties)} varieties - {varieties}")

    # Test yield calculation
    test_factors = {
        'drought_stress': 0.3,
        'disease_pressure': 0.2,
        'heat_stress': 0.1
    }

    print("\nYield Potential Examples:")
    for variety in ['HD_3086', 'PBW_725', 'HD_2967']:
        base_yield = PUNJAB_WHEAT_VARIETIES[variety]['yield_potential_quintal_ha']
        adjusted_yield = calculate_variety_yield_potential(variety, test_factors)
        print(".1f")

    # Create and save variety dataset
    variety_df = create_variety_dataset_for_punjab()
    variety_df.to_csv('punjab_wheat_varieties.csv', index=False)
    print(f"\nVariety dataset saved: {len(variety_df)} records")
