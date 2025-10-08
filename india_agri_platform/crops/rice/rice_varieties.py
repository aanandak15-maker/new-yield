"""
Indian Agricultural Intelligence Platform - Rice Varieties Module
Production-ready rice crop variety mapping and configuration
"""

# Indian Rice Varieties by Region and State
RICE_VARIETIES = {
    # Punjab Varieties (High-yield, basmati specialty)
    "PB": {
        "basmati": ["Pusa Basmati 1121", "Punjab Basmati 1", "Pusa Basmati 1509", "CSR 30"],
        "regular": ["PAU 201", "Punjab Mehak 1", "PR 126", "PR 127", "PR 128"],
        "aromatic": ["Punjab Mehak 2", "Basmati 386", "Tarori Basmati"],
        "hybrid": ["Pusa RH-10", "Arize 6444", "Arize 6129"]
    },

    # Haryana Varieties (Basmati and regular rice)
    "HR": {
        "basmati": ["Pusa Basmati 1121", "Pusa Basmati 1", "HBC 19", "HM 362", "HM 490"],
        "regular": ["HK 164", "HK 188", "HK 160", "HK 198", "HK 263"],
        "aromatic": ["Sugandi Dhan", "Basmati 386", "Dehradun Basmati"],
        "hybrid": ["PAC 801", "RAC 462", "Shatabdi 5"]
    },

    # Gujarat Varieties (Salt-tolerant varieties)
    "GJ": {
        "regular": ["Gurjari", "GAUF 201", "Garud", "NAZ", "GH-12"],
        "salt_tolerant": ["CSR 36", "CSR 43", "Punjab Mehak 1", "GR-11"],
        "hybrid": ["Arize 6129", "Arize 6444", "Pusa RH-10"],
        "early": ["IR-36", "Ankur"]
    },

    # Andhra Pradesh Varieties (High-yield indica)
    "AP": {
        "regular": ["MTU 1001", "MBN 042", "Nellore Masuri", "Gururatan", "Samba Masuri"],
        "hybrid": ["APHR 2", "PMC 21", "US-312", "PA 6444", "PHB 71"],
        "extra_early": ["ZH 11", "TN 1", "IR 64", "MTU 7029"],
        "stress_tolerant": ["Swarna", "Sabita", "Swarna-Sub1", "CR Dhan 800"]
    },

    # Uttar Pradesh Varieties
    "UP": {
        "basmati": ["Pusa Basmati 1121", "Pusa Basmati 1509", "IPSB-9"],
        "regular": ["Pant Dhan 4", "Pant Dhan 12", "NDR 1403"],
        "hybrid": ["US 312", "Arize 6444", "Arize 6129"],
        "coarse": ["Pant Ragi Dhan", "Narendra LGR Dhan", "Sarju 52"]
    },

    # Madhya Pradesh Varieties
    "MP": {
        "regular": ["ICow", "ICSR 11014", "ICSV 12114", "Pant Dhan 19"],
        "hybrid": ["IPAH 501", "SH-22", "Arize 6444", "SUDHA"],
        "stress_tolerant": ["Pant Dhan 4", "Pant Dhan 12", "IR 36"],
        "traditional": ["Kalamasalathu", "Vasistha Dhan", "Rathu Heenati"]
    },

    # Karnataka Varieties
    "KA": {
        "regular": ["Jaya", "MTU 1010", "Tunga", "Abhilasha", "HMT"],
        "hybrid": ["Arize 6444", "Arize 6129", "KRH 2", "KRH 4"],
        "aromatic": ["Basmati 370", "Sugandi Dahn", "Khoda"]
    },

    # Tamil Nadu Varieties
    "TN": {
        "regular": ["ADT 39", "ADT 43", "ADK 47", "CO 51", "Pusa 44"],
        "hybrid": ["Arize 6444", "Arize 6129", "CORH 3"],
        "coastal": ["VKR 3", "TRY 3", "Dhivara"]
    },

    # Maharashtra Varieties
    "MH": {
        "regular": ["Indrayani", "Amogh", "GNR 3", "Suraj", "Hansdad"],
        "hybrid": ["Arize 6444", "Arize 6129", "Pusa RH 10"],
        "salt_tolerant": ["CSR 36", "CSR 43", "GR-11"]
    },

    # West Bengal Varieties
    "WB": {
        "regular": ["Swarna", "MTU 7029", "Miniket", "Manashi", "Radhunipagal"],
        "hybrid": ["Arize 6444", "Arize 6129", "SUDHA"],
        "traditional": ["Kalobo", "Lal Dhan", "Rupali", "Charnok"]
    },

    # Bihar Varieties
    "BR": {
        "regular": ["MTU 1010", "Rajendra Bhagavati", "Bina Dhan", "Rajendra Sweta"],
        "hybrid": ["Arize 6444", "Arize 6129", "PHB 71"],
        "disease_resistant": ["TN 1", "CR Dhan 700", "Rajendra Castor"]
    },

    # Odisha Varieties
    "OR": {
        "regular": ["CR Dhan 503", "CR Dhan 500", "CR Dhan 201", "CR Dhan 401"],
        "stress_tolerant": ["Swarna-Sub1", "Lalit Dhan", "Naveen"],
        "salinity_tolerant": ["CSR 36", "CSR 43", "Lutikir Dhan"]
    },

    # Rajasthan Varieties
    "RJ": {
        "regular": ["Basmati 370", "Sugandi Dhan", "RTLP 1", "RTLP 2"],
        "water_efficient": ["CR 238", "CR 326", "CR 335"],
        "arid_tolerant": ["ARA-1", "ARA-2", "ARA-3"]
    },

    # Telangana Varieties
    "TL": {
        "regular": ["MTU 1001", "Vijetha", "Samaribili Dhan", "Swetha"],
        "hybrid": ["APHR 2", "RNHP 8", "PA 6129"],
        "pest_resistant": ["MTU 1026", "JGL 19661", "Pusa RH 10"]
    },

    # Chhattisgarh Varieties (New State)
    "CG": {
        "regular": ["Danteswari", "Indra Dhan", "KS 498", "Ruchi"],
        "stress_tolerant": ["CR Dhan 601", "MTU 7029", "Pant Dhan 12"]
    },

    # Jharkhand Varieties (New State)
    "JH": {
        "regular": ["RH 10", "Kavit", "Pratikhsha", "Sahabhagi Dhan"],
        "aromatic": ["Badshabhog", "Kalajhara", "Katyaina Dhan"],
        "traditional": ["Agra Dhan", "Bhairuly Dhan"]
    },

    # Uttarakhand Varieties (New State)
    "UT": {
        "regular": ["Pant Dhan 4", "Pant Dhan 12", "VL Dhan 24"],
        "basmati": ["HBC 19", "Pusa Sugandh 5", "Type 3 Dhan"],
        "hill_rice": ["Lalat Dhan", "Garhkadi"]
    }
}

# Rice Variety Map (for quick lookup by state)
RICE_VARIETY_MAP = {
    # Basmati varieties (premium)
    "basmati": ["Punjab", "Haryana", "Uttar Pradesh", "Rajasthan"],
    "premium": ["Punjab Basmati 1", "Pusa Basmati 1121", "Pusa Basmati 1509"],

    # Regular varieties (high yield)
    "regular": ["Swarna", "MTU 1010", "PAU 201"],
    "high_yield": ["CR Dhan", "Arize 6444", "SAHOD UTTAR"],

    # Stress-tolerant varieties
    "stress_tolerant": ["Swarna-Sub1", "Sabita", "Sahbhagi Dhan"],
    "salt_tolerant": ["CSR 36", "CSR 43", "GR-11"],

    # Hybrid varieties (maximum yield)
    "hybrid": ["Arize 6444", "Arize 6129", "Pusa RH-10"],
    "early_maturing": ["MTU 7029", "TN 1", "IR 36"],

    # Traditional/aromatic varieties
    "aromatic": ["Basmati 370", "Kalobo", "Badshabhog"],
    "traditional": ["Handi Dhan", "Rathu Heenati", "Kalamasalathu"]
}

# Regional Classification for Rice Production in India
RICE_ECOZONES = {
    "north_india": ["PB", "HR", "UP", "UT", "HP", "JK", "DL"],  # Northern states
    "west_india": ["GJ", "MH", "MP", "RJ"],  # Western states
    "east_india": ["WB", "BR", "OR", "JH"],  # Eastern states
    "south_india": ["AP", "TL", "TN", "KA", "KL", "PY"],  # Southern states
    "central_india": ["MP", "CG", "JH", "UP_part"]  # Central states
}

# Irrigation-based rice variety recommendations
IRRIGATION_TYPES = {
    "irrigated": {
        "varieties": ["Pusa Basmati 1121", "MTU 1010", "CR Dhan 503"],
        "yield_potential": "high",
        "water_requirement": "high"
    },
    "rainfed": {
        "varieties": ["Sabita", "Sahbhagi Dhan", "Danteswari"],
        "yield_potential": "medium",
        "water_requirement": "low-to-medium"
    },
    "water_saving": {
        "varieties": ["Swarna-Sub1", "MTU 7029", "FL 478"],
        "yield_potential": "high",
        "water_requirement": "low"
    }
}

# Climate-based rice variety recommendations
CLIMATE_ZONES = {
    "tropical": {
        "states": ["TL", "AP", "TN", "KA", "KL"],
        "varieties": ["MTU 1001", "Swarna", "IR 64"],
        "temperature_range": "25-35°C"
    },
    "subtropical": {
        "states": ["WB", "BR", "OR", "JH", "CG", "MP"],
        "varieties": ["RH 10", "Manashi", "Radhunipagal"],
        "temperature_range": "20-30°C"
    },
    "temperate": {
        "states": ["UT", "HP", "JK"],
        "varieties": ["Pant Dhan 4", "VL Dhan 24"],
        "temperature_range": "15-25°C"
    },
    "arid_hot": {
        "states": ["RJ", "GJ", "MH"],
        "varieties": ["Gurjari", "CSR 36", "GAUF 201"],
        "temperature_range": "30-40°C"
    },
    "basin_rice": {
        "states": ["PB", "HR", "UP"],
        "varieties": ["Pusa Basmati 1121", "Punjab Mehak 1", "HBC 19"],
        "temperature_range": "20-35°C"
    }
}

# Quality classification for rice varieties
QUALITY_CLASSES = {
    "premium_basmati": {
        "varieties": ["Pusa Basmati 1121", "Punjab Basmati 1", "Pusa Basmati 1509"],
        "grain_length": ">7.5mm",
        "aroma_level": "high",
        "market_price": "premium"
    },
    "standard_basmati": {
        "varieties": ["HBC 19", "HM 362", "Basmati 386"],
        "grain_length": "6.6-7.5mm",
        "aroma_level": "medium",
        "market_price": "standard"
    },
    "regular_long_grain": {
        "varieties": ["Swarna", "MTU 1010", "PAU 201"],
        "grain_length": "6.1-7.0mm",
        "aroma_level": "low",
        "market_price": "standard"
    },
    "regular_medium_grain": {
        "varieties": ["ADT 39", "CR Dhan 503", "Indrayani"],
        "grain_length": "5.6-6.5mm",
        "aroma_level": "low",
        "market_price": "commodity"
    },
    "coarse_short_grain": {
        "varieties": ["Radhunipagal", "Kalobo", "Rupali"],
        "grain_length": "<6.0mm",
        "aroma_level": "none",
        "market_price": "commodity"
    }
}

def get_rice_varieties_by_state(state_code: str) -> dict:
    """Get all rice varieties available in a specific state"""
    return RICE_VARIETIES.get(state_code.upper(), {})

def get_rice_varieties_by_type(variety_type: str) -> list:
    """Get varieties by type (basmati, regular, hybrid, aromatic, stress_tolerant, salt_tolerant)"""
    variety_lists = []
    for state_code, varieties in RICE_VARIETIES.items():
        if variety_type in varieties:
            variety_lists.extend(varieties[variety_type])

    # Remove duplicates and return
    return list(set(variety_lists))

def get_rice_varieties_by_ecozone(ecozone: str) -> dict:
    """Get rice varieties by ecozone (north_india, west_india, etc.)"""
    states_in_zone = RICE_ECOZONES.get(ecozone, [])

    varieties_in_zone = {}
    for state in states_in_zone:
        varieties_in_zone[state] = get_rice_varieties_by_state(state)

    return varieties_in_zone

def recommend_rice_varieties(conditions: dict) -> list:
    """
    Recommend rice varieties based on environmental conditions

    Parameters:
    conditions (dict): Dictionary containing conditions like
        - location: 'GJ' (state code)
        - irrigation: 'irrigated'/'rainfed'/'water_saving'
        - climate: 'tropical'/'subtropical'/'temperate'/'arid_hot'
        - soil_type: 'sandy'/'clay'/'loam'/'alkaline'/'saline'
        - farming_type: 'conventional'/'organic'/'sustainable'
        - yield_goal: 'high'/'medium'/'sustainable'

    Returns:
    list: Recommended rice varieties
    """
    recommended_varieties = []

    # Location-based varieties
    if 'location' in conditions:
        state_varieties = get_rice_varieties_by_state(conditions['location'])
        for variety_type, varieties in state_varieties.items():
            recommended_varieties.extend(varieties[:3])  # Top 3 per type

    # Irrigation-based recommendations
    if 'irrigation' in conditions:
        irrigation_type = conditions['irrigation']
        if irrigation_type in IRRIGATION_TYPES:
            irrigation_varieties = IRRIGATION_TYPES[irrigation_type]['varieties'][:5]
            recommended_varieties.extend(irrigation_varieties)

    # Climate-based recommendations
    if 'climate' in conditions:
        climate_type = conditions['climate']
        if climate_type in CLIMATE_ZONES:
            climate_varieties = CLIMATE_ZONES[climate_type]['varieties'][:5]
            recommended_varieties.extend(climate_varieties)

    # Remove duplicates while maintaining order
    seen = set()
    unique_varieties = []
    for variety in recommended_varieties:
        if variety not in seen:
            unique_varieties.append(variety)
            seen.add(variety)

    return unique_varieties[:10]  # Limit to top 10 recommendations

def get_variety_characteristics(variety_name: str) -> dict:
    """Get detailed characteristics of a specific rice variety"""
    # This would be a more comprehensive lookup in a production system
    characteristics = {
        "yield_potential": "varies",
        "maturity_days": "varies",
        "grain_quality": "varies",
        "disease_resistance": "varies",
        "water_requirement": "varies",
        "climate_adaptability": "varies"
    }

    # Add specific characteristics based on variety name
    return characteristics

# Export main components
__all__ = [
    'RICE_VARIETIES',
    'RICE_VARIETY_MAP',
    'RICE_ECOZONES',
    'IRRIGATION_TYPES',
    'CLIMATE_ZONES',
    'QUALITY_CLASSES',
    'get_rice_varieties_by_state',
    'get_rice_varieties_by_type',
    'get_rice_varieties_by_ecozone',
    'recommend_rice_varieties',
    'get_variety_characteristics'
]
