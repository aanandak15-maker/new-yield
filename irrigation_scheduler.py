"""
Weather-Based Irrigation Scheduling System
SIH 2025 - Agricultural Enhancement Module

This module implements weather-based irrigation scheduling for different crops,
considering rainfall, evapotranspiration, and soil moisture requirements.
Based on FAO crop water requirements methodology.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

# Crop water requirements (mm/day) - FAO data
CROP_WATER_REQUIREMENTS = {
    'wheat': {
        'initial': 2.5,      # Initial stage (0-30 days)
        'development': 4.0,  # Development stage (30-90 days)
        'mid_season': 6.5,   # Mid-season stage (90-120 days)
        'late_season': 3.0   # Late season stage (120-end)
    },
    'rice': {
        'initial': 4.0,
        'development': 6.0,
        'mid_season': 8.0,
        'late_season': 5.0
    },
    'maize': {
        'initial': 3.0,
        'development': 5.0,
        'mid_season': 7.0,
        'late_season': 4.0
    },
    'cotton': {
        'initial': 2.0,
        'development': 4.0,
        'mid_season': 6.0,
        'late_season': 3.0
    },
    'sugarcane': {
        'initial': 3.5,
        'development': 5.5,
        'mid_season': 7.5,
        'late_season': 4.5
    }
}

# Crop coefficients (Kc) - FAO methodology
CROP_COEFFICIENTS = {
    'wheat': {
        'initial': 0.3,
        'development': 0.7,
        'mid_season': 1.15,
        'late_season': 0.6
    },
    'rice': {
        'initial': 0.5,
        'development': 0.8,
        'mid_season': 1.2,
        'late_season': 0.8
    },
    'maize': {
        'initial': 0.3,
        'development': 0.8,
        'mid_season': 1.2,
        'late_season': 0.6
    },
    'cotton': {
        'initial': 0.4,
        'development': 0.7,
        'mid_season': 1.1,
        'late_season': 0.7
    },
    'sugarcane': {
        'initial': 0.4,
        'development': 0.7,
        'mid_season': 1.2,
        'late_season': 0.8
    }
}

# Punjab soil characteristics
PUNJAB_SOIL_DATA = {
    'sandy': {
        'field_capacity': 0.15,    # m³/m³
        'wilting_point': 0.05,
        'bulk_density': 1.5,       # g/cm³
        'available_water': 0.10
    },
    'sandy_loam': {
        'field_capacity': 0.20,
        'wilting_point': 0.08,
        'bulk_density': 1.4,
        'available_water': 0.12
    },
    'loam': {
        'field_capacity': 0.25,
        'wilting_point': 0.10,
        'bulk_density': 1.3,
        'available_water': 0.15
    },
    'alluvial': {
        'field_capacity': 0.22,
        'wilting_point': 0.09,
        'bulk_density': 1.35,
        'available_water': 0.13
    }
}

class IrrigationScheduler:
    """Weather-based irrigation scheduling system"""

    def __init__(self):
        self.crop = 'wheat'  # Default crop
        self.soil_type = 'sandy_loam'
        self.root_depth = 1.2  # meters for wheat

    def set_crop(self, crop_name: str):
        """Set the crop for irrigation scheduling"""
        if crop_name.lower() in CROP_WATER_REQUIREMENTS:
            self.crop = crop_name.lower()
        else:
            print(f"Warning: Crop '{crop_name}' not found. Using wheat as default.")
            self.crop = 'wheat'

    def set_soil_type(self, soil_type: str):
        """Set the soil type for calculations"""
        if soil_type.lower() in PUNJAB_SOIL_DATA:
            self.soil_type = soil_type.lower()
        else:
            print(f"Warning: Soil type '{soil_type}' not found. Using sandy_loam as default.")
            self.soil_type = 'sandy_loam'

    def calculate_et0(self, temperature_celsius: float, humidity_percent: float,
                     wind_speed_kmph: float, solar_radiation_mj_m2_day: Optional[float] = None) -> float:
        """
        Calculate reference evapotranspiration (ET0) using FAO Penman-Monteith method
        Simplified version for demonstration

        Parameters:
        - temperature_celsius: Mean daily temperature (°C)
        - humidity_percent: Relative humidity (%)
        - wind_speed_kmph: Wind speed (km/h)
        - solar_radiation_mj_m2_day: Solar radiation (MJ/m²/day) - optional

        Returns:
        - ET0 in mm/day
        """
        # Simplified Hargreaves equation for ET0 calculation
        # ET0 = 0.0023 * (T_mean + 17.8) * (T_max - T_min)^0.5 * Ra

        # Estimate T_max and T_min from mean temperature (simplified)
        t_mean = temperature_celsius
        t_max = t_mean + 5  # Rough estimate
        t_min = t_mean - 5  # Rough estimate

        # Solar radiation estimate (MJ/m²/day) - simplified
        if solar_radiation_mj_m2_day is None:
            # Rough estimate based on temperature
            solar_radiation_mj_m2_day = 15 + (t_mean * 0.1)

        # Hargreaves equation
        et0 = 0.0023 * (t_mean + 17.8) * ((t_max - t_min) ** 0.5) * solar_radiation_mj_m2_day

        # Adjust for humidity and wind (simplified factors)
        humidity_factor = 1 + (humidity_percent - 60) * 0.002  # Humidity adjustment
        wind_factor = 1 + (wind_speed_kmph - 2) * 0.01         # Wind adjustment

        et0_adjusted = et0 * humidity_factor * wind_factor

        return max(0, et0_adjusted)

    def calculate_crop_water_requirement(self, et0: float, growth_stage: str) -> float:
        """
        Calculate crop water requirement using crop coefficient approach

        Parameters:
        - et0: Reference evapotranspiration (mm/day)
        - growth_stage: Crop growth stage ('initial', 'development', 'mid_season', 'late_season')

        Returns:
        - Crop evapotranspiration (ETc) in mm/day
        """
        kc = CROP_COEFFICIENTS[self.crop].get(growth_stage, 1.0)
        etc = et0 * kc
        return etc

    def calculate_irrigation_requirement(self, etc: float, rainfall_mm: float,
                                       soil_moisture_depletion: float = 0.5) -> Dict[str, float]:
        """
        Calculate irrigation requirement considering rainfall and soil moisture

        Parameters:
        - etc: Crop evapotranspiration (mm/day)
        - rainfall_mm: Rainfall in mm
        - soil_moisture_depletion: Current soil moisture depletion (0-1)

        Returns:
        - Dictionary with irrigation scheduling information
        """
        # Get soil parameters
        soil_params = PUNJAB_SOIL_DATA[self.soil_type]
        taw = soil_params['available_water'] * self.root_depth * 1000  # Total available water (mm)

        # Readily available water (typically 50-60% of TAW)
        raw = taw * 0.5

        # Net irrigation requirement
        net_irrigation = max(0, etc - rainfall_mm)

        # Gross irrigation requirement (accounting for efficiency)
        irrigation_efficiency = 0.75  # Typical Punjab canal irrigation efficiency
        gross_irrigation = net_irrigation / irrigation_efficiency

        # Check if irrigation is needed
        current_depletion = soil_moisture_depletion * taw
        irrigation_needed = current_depletion >= raw

        return {
            'net_irrigation_mm': round(net_irrigation, 1),
            'gross_irrigation_mm': round(gross_irrigation, 1),
            'irrigation_needed': irrigation_needed,
            'soil_moisture_threshold': round(raw, 1),
            'current_depletion': round(current_depletion, 1),
            'total_available_water': round(taw, 1),
            'irrigation_efficiency': irrigation_efficiency
        }

    def get_irrigation_schedule(self, weather_data: pd.DataFrame,
                               crop_stage: str = 'mid_season') -> pd.DataFrame:
        """
        Generate irrigation schedule for a period based on weather data

        Parameters:
        - weather_data: DataFrame with weather columns
        - crop_stage: Current crop growth stage

        Returns:
        - DataFrame with irrigation scheduling information
        """
        required_columns = ['temperature_celsius', 'humidity_percent',
                          'wind_speed_kmph', 'rainfall_mm']

        missing_columns = [col for col in required_columns if col not in weather_data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Calculate irrigation requirements for each day
        irrigation_schedule = []

        for idx, row in weather_data.iterrows():
            # Calculate ET0
            et0 = self.calculate_et0(
                temperature_celsius=row['temperature_celsius'],
                humidity_percent=row['humidity_percent'],
                wind_speed_kmph=row['wind_speed_kmph']
            )

            # Calculate crop water requirement
            etc = self.calculate_crop_water_requirement(et0, crop_stage)

            # Calculate irrigation requirement
            irrigation_info = self.calculate_irrigation_requirement(
                etc=etc,
                rainfall_mm=row['rainfall_mm']
            )

            # Add date and results
            schedule_entry = {
                'date': row.get('date', idx),
                'et0_mm': round(et0, 2),
                'etc_mm': round(etc, 2),
                'rainfall_mm': row['rainfall_mm'],
                **irrigation_info
            }

            irrigation_schedule.append(schedule_entry)

        return pd.DataFrame(irrigation_schedule)

    def optimize_irrigation_frequency(self, weather_forecast: pd.DataFrame,
                                    crop_stage: str = 'mid_season') -> Dict[str, any]:
        """
        Optimize irrigation frequency based on weather forecast

        Parameters:
        - weather_forecast: DataFrame with forecasted weather
        - crop_stage: Current crop growth stage

        Returns:
        - Dictionary with irrigation optimization recommendations
        """
        # Calculate cumulative water requirements
        schedule = self.get_irrigation_schedule(weather_forecast, crop_stage)

        # Analyze irrigation needs
        irrigation_days = schedule[schedule['irrigation_needed']].shape[0]
        total_irrigation = schedule['gross_irrigation_mm'].sum()

        # Calculate optimal frequency
        total_days = len(schedule)
        irrigation_frequency = total_days / max(irrigation_days, 1)

        # Calculate water productivity
        rainfall_effectiveness = schedule['rainfall_mm'].sum() * 0.8  # 80% effectiveness
        total_water_input = total_irrigation + rainfall_effectiveness

        return {
            'recommended_frequency_days': round(irrigation_frequency, 1),
            'total_irrigation_mm': round(total_irrigation, 1),
            'effective_rainfall_mm': round(rainfall_effectiveness, 1),
            'total_water_input_mm': round(total_water_input, 1),
            'irrigation_days': irrigation_days,
            'total_days': total_days,
            'water_productivity_index': round(total_irrigation / max(irrigation_days, 1), 2)
        }

def create_punjab_irrigation_dataset(districts_data: Dict) -> pd.DataFrame:
    """
    Create irrigation dataset for Punjab districts

    Parameters:
    - districts_data: Dictionary with district information

    Returns:
    - DataFrame with irrigation parameters for each district
    """
    irrigation_data = []

    for district, info in districts_data.items():
        soil_type = info.get('soil_type', 'sandy_loam')
        irrigation_coverage = info.get('irrigation_coverage', 0.9)
        groundwater_depth = info.get('groundwater_depth', 8.0)

        # Determine irrigation method based on groundwater depth
        if groundwater_depth < 5:
            irrigation_method = 'tube_well'
            efficiency = 0.75
        elif groundwater_depth < 8:
            irrigation_method = 'canal'
            efficiency = 0.65
        else:
            irrigation_method = 'rainfed'
            efficiency = 0.4

        irrigation_data.append({
            'district': district,
            'soil_type': soil_type,
            'irrigation_coverage': irrigation_coverage,
            'groundwater_depth_m': groundwater_depth,
            'irrigation_method': irrigation_method,
            'irrigation_efficiency': efficiency,
            'crop_water_requirement_mm_day': 6.5,  # For wheat mid-season
            'irrigation_interval_days': 7,  # Typical interval
            'soil_moisture_capacity_mm': 120,  # Available water capacity
            'critical_moisture_threshold': 0.5  # 50% depletion threshold
        })

    return pd.DataFrame(irrigation_data)

if __name__ == "__main__":
    # Test the irrigation scheduler
    print("Punjab Irrigation Scheduling System")
    print("=" * 40)

    scheduler = IrrigationScheduler()
    scheduler.set_crop('wheat')
    scheduler.set_soil_type('sandy_loam')

    # Test ET0 calculation
    et0 = scheduler.calculate_et0(
        temperature_celsius=25,
        humidity_percent=60,
        wind_speed_kmph=5
    )
    print(f"Reference ET0: {et0:.2f} mm/day")

    # Test crop water requirement
    etc = scheduler.calculate_crop_water_requirement(et0, 'mid_season')
    print(f"Crop ET (wheat, mid-season): {etc:.2f} mm/day")

    # Test irrigation requirement
    irrigation = scheduler.calculate_irrigation_requirement(etc, 5.0, 0.3)
    print(f"Net irrigation needed: {irrigation['net_irrigation_mm']} mm")
    print(f"Gross irrigation needed: {irrigation['gross_irrigation_mm']} mm")
    print(f"Irrigation needed: {irrigation['irrigation_needed']}")

    # Create Punjab irrigation dataset
    from punjab_districts import PUNJAB_DISTRICTS
    irrigation_df = create_punjab_irrigation_dataset(PUNJAB_DISTRICTS)
    irrigation_df.to_csv('punjab_irrigation_data.csv', index=False)
    print(f"\nIrrigation dataset saved: {len(irrigation_df)} records")

    print("\nIrrigation scheduling system ready for integration!")
