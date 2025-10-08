"""
Punjab Districts Configuration for Rabi Crop Yield Prediction
SIH 2025 Final Project - Auto-Fetching Crop Yield Prediction System

This module defines Punjab state districts and their agricultural characteristics
for wheat yield prediction during Rabi season (Nov-Mar).
"""

# Punjab Districts with enhanced agricultural data
PUNJAB_DISTRICTS = {
    # Major Wheat Producing Districts (North-West Punjab)
    'Amritsar': {
        'division': 'Amritsar',
        'area_km2': 2673,
        'wheat_priority': 'High',
        'soil_type': 'Alluvial',
        'irrigation_coverage': 0.95,
        'groundwater_depth': 8.5,
        'historical_yield_avg': 42.3
    },
    'Tarn Taran': {
        'division': 'Amritsar',
        'area_km2': 2414,
        'wheat_priority': 'High',
        'soil_type': 'Sandy Loam',
        'irrigation_coverage': 0.92,
        'groundwater_depth': 9.2,
        'historical_yield_avg': 41.8
    },
    'Bathinda': {
        'division': 'Bathinda',
        'area_km2': 3342,
        'wheat_priority': 'High',
        'soil_type': 'Alluvial',
        'irrigation_coverage': 0.98,
        'groundwater_depth': 6.8,
        'historical_yield_avg': 40.5
    },
    'Mansa': {
        'division': 'Bathinda',
        'area_km2': 2174,
        'wheat_priority': 'High',
        'soil_type': 'Sandy',
        'irrigation_coverage': 0.96,
        'groundwater_depth': 7.5,
        'historical_yield_avg': 38.2
    },
    'Muktsar': {
        'division': 'Ferozepur',
        'area_km2': 2641,
        'wheat_priority': 'High',
        'soil_type': 'Sandy Loam',
        'irrigation_coverage': 0.94,
        'groundwater_depth': 8.1,
        'historical_yield_avg': 37.8
    },
    'Ferozepur': {
        'division': 'Ferozepur',
        'area_km2': 5305,
        'wheat_priority': 'High',
        'soil_type': 'Alluvial',
        'irrigation_coverage': 0.97,
        'groundwater_depth': 7.2,
        'historical_yield_avg': 39.6
    },
    'Faridkot': {
        'division': 'Ferozepur',
        'area_km2': 1471,
        'wheat_priority': 'High',
        'soil_type': 'Loam',
        'irrigation_coverage': 0.95,
        'groundwater_depth': 6.9,
        'historical_yield_avg': 41.2
    },

    # Central Punjab Districts
    'Ludhiana': {
        'division': 'Ludhiana',
        'area_km2': 3746,
        'wheat_priority': 'High',
        'soil_type': 'Alluvial',
        'irrigation_coverage': 0.99,
        'groundwater_depth': 5.8,
        'historical_yield_avg': 45.8
    },
    'Jalandhar': {
        'division': 'Jalandhar',
        'area_km2': 2632,
        'wheat_priority': 'High',
        'soil_type': 'Loam',
        'irrigation_coverage': 0.96,
        'groundwater_depth': 8.3,
        'historical_yield_avg': 43.7
    },
    'Hoshiarpur': {
        'division': 'Jalandhar',
        'area_km2': 3386,
        'wheat_priority': 'Medium',
        'soil_type': 'Loam',
        'irrigation_coverage': 0.88,
        'groundwater_depth': 9.8,
        'historical_yield_avg': 38.9
    },
    'Kapurthala': {
        'division': 'Jalandhar',
        'area_km2': 1633,
        'wheat_priority': 'Medium',
        'soil_type': 'Sandy Loam',
        'irrigation_coverage': 0.91,
        'groundwater_depth': 8.7,
        'historical_yield_avg': 39.4
    },
    'Nawanshahr': {
        'division': 'Jalandhar',
        'area_km2': 1259,
        'wheat_priority': 'Medium',
        'soil_type': 'Loam',
        'irrigation_coverage': 0.89,
        'groundwater_depth': 9.1,
        'historical_yield_avg': 37.6
    },

    # South-West Punjab
    'Sangrur': {
        'division': 'Sangrur',
        'area_km2': 3622,
        'wheat_priority': 'High',
        'soil_type': 'Alluvial',
        'irrigation_coverage': 0.97,
        'groundwater_depth': 7.4,
        'historical_yield_avg': 42.1
    },
    'Barnala': {
        'division': 'Sangrur',
        'area_km2': 1482,
        'wheat_priority': 'High',
        'soil_type': 'Loam',
        'irrigation_coverage': 0.94,
        'groundwater_depth': 8.2,
        'historical_yield_avg': 40.3
    },
    'Patiala': {
        'division': 'Patiala',
        'area_km2': 3322,
        'wheat_priority': 'High',
        'soil_type': 'Alluvial',
        'irrigation_coverage': 0.98,
        'groundwater_depth': 6.5,
        'historical_yield_avg': 41.9
    },
    'Fatehgarh Sahib': {
        'division': 'Patiala',
        'area_km2': 1180,
        'wheat_priority': 'Medium',
        'soil_type': 'Loam',
        'irrigation_coverage': 0.92,
        'groundwater_depth': 7.8,
        'historical_yield_avg': 39.7
    },

    # North-East Punjab (Sub-mountain)
    'Gurdaspur': {
        'division': 'Gurdaspur',
        'area_km2': 3556,
        'wheat_priority': 'Medium',
        'soil_type': 'Loam',
        'irrigation_coverage': 0.85,
        'groundwater_depth': 10.2,
        'historical_yield_avg': 36.8
    },
    'Pathankot': {
        'division': 'Gurdaspur',
        'area_km2': 929,
        'wheat_priority': 'Medium',
        'soil_type': 'Sandy Loam',
        'irrigation_coverage': 0.82,
        'groundwater_depth': 11.5,
        'historical_yield_avg': 35.2
    },
    'Rupnagar': {
        'division': 'Rupnagar',
        'area_km2': 1400,
        'wheat_priority': 'Medium',
        'soil_type': 'Loam',
        'irrigation_coverage': 0.87,
        'groundwater_depth': 9.5,
        'historical_yield_avg': 34.1
    },
    'Mohali': {
        'division': 'Rupnagar',
        'area_km2': 1108,
        'wheat_priority': 'Medium',
        'soil_type': 'Loam',
        'irrigation_coverage': 0.91,
        'groundwater_depth': 8.9,
        'historical_yield_avg': 33.8
    },
    'Fazilka': {
        'division': 'Ferozepur',
        'area_km2': 3016,
        'wheat_priority': 'High',
        'soil_type': 'Sandy',
        'irrigation_coverage': 0.93,
        'groundwater_depth': 7.9,
        'historical_yield_avg': 38.9
    },
    'Sri Muktsar Sahib': {
        'division': 'Ferozepur',
        'area_km2': 1596,
        'wheat_priority': 'High',
        'soil_type': 'Sandy Loam',
        'irrigation_coverage': 0.95,
        'groundwater_depth': 8.4,
        'historical_yield_avg': 37.3
    }
}

# Agro-climatic zones in Punjab for wheat cultivation
AGRO_CLIMATIC_ZONES = {
    'North-Western Zone': {
        'districts': ['Amritsar', 'Tarn Taran', 'Ferozepur', 'Fazilka', 'Sri Muktsar Sahib'],
        'characteristics': 'Semi-arid, alluvial soils, canal irrigation dominant',
        'wheat_varieties': ['PBW 725', 'HD 3086', 'WH 1105']
    },
    'Central Zone': {
        'districts': ['Ludhiana', 'Jalandhar', 'Bathinda', 'Mansa', 'Sangrur', 'Barnala'],
        'characteristics': 'Sub-tropical, fertile alluvial soils, mixed irrigation',
        'wheat_varieties': ['PBW 752', 'HD 2967', 'DBW 187']
    },
    'South-Western Zone': {
        'districts': ['Muktsar', 'Faridkot', 'Patiala', 'Fatehgarh Sahib'],
        'characteristics': 'Arid to semi-arid, sandy loam soils, groundwater dependent',
        'wheat_varieties': ['PBW 677', 'HD 3059', 'WH 1124']
    },
    'North-Eastern Zone': {
        'districts': ['Gurdaspur', 'Pathankot', 'Hoshiarpur', 'Kapurthala', 'Nawanshahr', 'Rupnagar', 'Mohali'],
        'characteristics': 'Sub-mountain, loamy soils, rainfall dependent',
        'wheat_varieties': ['PBW 723', 'HD 3059', 'VL 907']
    }
}

# Rabi Season Timeline (Wheat crop cycle in Punjab)
RABI_SEASON = {
    'sowing_start': '11-01',  # November 1st
    'sowing_end': '11-30',    # November 30th
    'peak_vegetation': '01-15',  # January 15th
    'flowering': '02-15',     # February 15th
    'grain_filling': '03-01', # March 1st
    'harvest_start': '04-01', # April 1st
    'harvest_end': '04-30'    # April 30th
}

# Data collection parameters for auto-fetching
DATA_SOURCES = {
    'yield_data': {
        'source': 'Punjab Agriculture Department',
        'url': 'https://agri.punjab.gov.in/',
        'frequency': 'monthly',
        'format': 'CSV/Excel'
    },
    'weather_data': {
        'source': 'India Meteorological Department (IMD)',
        'url': 'https://mausam.imd.gov.in/',
        'frequency': 'daily',
        'parameters': ['temperature', 'rainfall', 'humidity', 'wind_speed']
    },
    'satellite_data': {
        'source': 'Google Earth Engine',
        'datasets': ['MODIS/061/MCD15A3H', 'LANDSAT/LC08/C01/T1'],
        'frequency': '16-day',
        'indices': ['NDVI', 'FPAR', 'LAI']
    },
    'soil_data': {
        'source': 'Punjab Remote Sensing Centre',
        'parameters': ['pH', 'organic_carbon', 'texture', 'salinity']
    }
}

def get_district_list():
    """Get list of all Punjab districts"""
    return list(PUNJAB_DISTRICTS.keys())

def get_high_priority_districts():
    """Get districts with high wheat production priority"""
    return [district for district, info in PUNJAB_DISTRICTS.items()
            if info['wheat_priority'] == 'High']

def get_zone_districts(zone_name):
    """Get districts belonging to a specific agro-climatic zone"""
    if zone_name in AGRO_CLIMATIC_ZONES:
        return AGRO_CLIMATIC_ZONES[zone_name]['districts']
    return []

def get_district_info(district_name):
    """Get detailed information about a specific district"""
    return PUNJAB_DISTRICTS.get(district_name, {})

if __name__ == "__main__":
    # Print district information
    print("Punjab Districts for Rabi Crop Yield Prediction")
    print("=" * 50)
    print(f"Total Districts: {len(PUNJAB_DISTRICTS)}")
    print(f"High Priority Wheat Districts: {len(get_high_priority_districts())}")

    print("\nAgro-Climatic Zones:")
    for zone, info in AGRO_CLIMATIC_ZONES.items():
        print(f"- {zone}: {len(info['districts'])} districts")

    print("\nRabi Season Timeline:")
    for stage, date in RABI_SEASON.items():
        print(f"- {stage.replace('_', ' ').title()}: {date}")
