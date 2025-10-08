"""
Auto-Fetching Data Collection System for Punjab Rabi Crop Yield Prediction
SIH 2025 Final Project

This module provides automated data collection from multiple sources:
- Punjab Government Agriculture Department (yield data)
- India Meteorological Department (weather data)
- Google Earth Engine (satellite vegetation indices)
- Punjab Remote Sensing Centre (soil data)

WARNING: This implementation uses realistic data simulation for demonstration.
Production deployment requires actual API integrations with government sources.
"""

import os
import pandas as pd
import numpy as np
import requests
import json
import logging
from datetime import datetime, timedelta
import time
from typing import Dict, List, Optional, Tuple
from punjab_districts import PUNJAB_DISTRICTS, get_district_list, RABI_SEASON

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('punjab_data_fetcher.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class PunjabDataFetcher:
    """Main class for fetching Punjab agricultural data"""

    def __init__(self, data_dir="punjab_data"):
        self.data_dir = data_dir
        self.ensure_data_directory()
        self.session = requests.Session()

        # API endpoints (these would be actual government API endpoints)
        self.api_endpoints = {
            'yield_data': 'https://api.agri.punjab.gov.in/yield-data',
            'weather_data': 'https://api.imd.gov.in/weather-data',
            'soil_data': 'https://api.prsc.punjab.gov.in/soil-data'
        }

    def ensure_data_directory(self):
        """Create data directory structure"""
        subdirs = ['yield', 'weather', 'satellite', 'soil', 'processed']
        for subdir in subdirs:
            os.makedirs(os.path.join(self.data_dir, subdir), exist_ok=True)

    def fetch_yield_data(self, year=None, districts=None):
        """
        Fetch wheat yield data from Punjab Agriculture Department
        In real implementation, this would connect to government APIs
        """
        if year is None:
            year = datetime.now().year

        if districts is None:
            districts = get_district_list()

        print(f"Fetching yield data for {year} from {len(districts)} districts...")

        # Simulate API call - in real implementation, this would be actual API calls
        yield_data = []

        for district in districts:
            # Generate realistic yield data based on district characteristics
            base_yield = self._get_base_yield_for_district(district)
            monthly_yields = self._generate_monthly_yields(base_yield, year)

            for month, yield_value in monthly_yields.items():
                yield_data.append({
                    'district': district,
                    'year': year,
                    'month': month,
                    'wheat_yield_quintal_per_hectare': yield_value,
                    'area_hectares': PUNJAB_DISTRICTS[district]['area_km2'] * 100,  # Convert to hectares
                    'data_source': 'Punjab_Agriculture_Department'
                })

        df = pd.DataFrame(yield_data)
        output_file = os.path.join(self.data_dir, 'yield', f'yield_data_{year}.csv')
        df.to_csv(output_file, index=False)
        print(f"Yield data saved to {output_file}")
        return df

    def fetch_weather_data(self, start_date=None, end_date=None, districts=None):
        """
        Fetch weather data from IMD for Punjab districts
        """
        if start_date is None:
            start_date = f"{datetime.now().year}-11-01"  # Rabi season start
        if end_date is None:
            end_date = f"{datetime.now().year}-04-30"    # Rabi season end

        if districts is None:
            districts = get_district_list()

        print(f"Fetching weather data from {start_date} to {end_date}...")

        weather_data = []

        # District-wise weather stations (approximate coordinates)
        weather_stations = self._get_weather_stations()

        print(f"Found {len(weather_stations)} weather stations")

        for district in districts:
            if district in weather_stations:
                print(f"Fetching data for {district}...")
                station_data = self._fetch_station_weather(
                    weather_stations[district],
                    start_date,
                    end_date
                )
                print(f"Generated {len(station_data)} data points for {district}")
                # Add district to each data point
                for data_point in station_data:
                    data_point['district'] = district
                weather_data.extend(station_data)

        print(f"Total weather data points: {len(weather_data)}")

        df = pd.DataFrame(weather_data)
        output_file = os.path.join(self.data_dir, 'weather', f'weather_data_{start_date[:4]}_{end_date[:4]}.csv')
        df.to_csv(output_file, index=False)
        print(f"Weather data saved to {output_file}")
        return df

    def fetch_satellite_data(self, year=None, districts=None):
        """
        Fetch satellite vegetation indices from Google Earth Engine
        This would require GEE authentication and processing
        """
        if year is None:
            year = datetime.now().year

        if districts is None:
            districts = get_district_list()

        print(f"Fetching satellite data for {year}...")

        # In real implementation, this would use GEE Python API
        satellite_data = []

        for district in districts:
            # Generate satellite indices data
            monthly_data = self._generate_satellite_data(district, year)

            for data_point in monthly_data:
                data_point['district'] = district
                data_point['year'] = year
                satellite_data.append(data_point)

        df = pd.DataFrame(satellite_data)
        output_file = os.path.join(self.data_dir, 'satellite', f'satellite_data_{year}.csv')
        df.to_csv(output_file, index=False)
        print(f"Satellite data saved to {output_file}")
        return df

    def fetch_soil_data(self, districts=None):
        """
        Fetch soil data from Punjab Remote Sensing Centre
        """
        if districts is None:
            districts = get_district_list()

        print("Fetching soil data...")

        soil_data = []

        for district in districts:
            soil_info = self._get_soil_data_for_district(district)
            soil_info['district'] = district
            soil_data.append(soil_info)

        df = pd.DataFrame(soil_data)
        output_file = os.path.join(self.data_dir, 'soil', 'soil_data.csv')
        df.to_csv(output_file, index=False)
        print(f"Soil data saved to {output_file}")
        return df

    def fetch_variety_data(self, districts=None):
        """
        Fetch wheat variety data for Punjab districts
        """
        if districts is None:
            districts = get_district_list()

        print("Fetching wheat variety data...")

        try:
            from crop_varieties import create_variety_dataset_for_punjab
            variety_df = create_variety_dataset_for_punjab()
            output_file = os.path.join(self.data_dir, 'varieties', 'variety_data.csv')
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            variety_df.to_csv(output_file, index=False)
            print(f"Variety data saved to {output_file}")
            return variety_df
        except ImportError:
            print("Warning: crop_varieties module not found. Skipping variety data.")
            return pd.DataFrame()

    def fetch_irrigation_data(self, districts=None):
        """
        Fetch irrigation scheduling data for Punjab districts
        """
        if districts is None:
            districts = get_district_list()

        print("Fetching irrigation data...")

        try:
            from irrigation_scheduler import create_punjab_irrigation_dataset
            from punjab_districts import PUNJAB_DISTRICTS
            irrigation_df = create_punjab_irrigation_dataset(PUNJAB_DISTRICTS)
            output_file = os.path.join(self.data_dir, 'irrigation', 'irrigation_data.csv')
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            irrigation_df.to_csv(output_file, index=False)
            print(f"Irrigation data saved to {output_file}")
            return irrigation_df
        except ImportError:
            print("Warning: irrigation_scheduler module not found. Skipping irrigation data.")
            return pd.DataFrame()

    def create_training_dataset(self, year=None):
        """
        Combine all data sources to create training dataset
        """
        if year is None:
            year = datetime.now().year - 1  # Use previous year for training

        print(f"Creating training dataset for {year}...")

        # Fetch all data types
        yield_df = self.fetch_yield_data(year)
        weather_df = self.fetch_weather_data(f"{year}-11-01", f"{year+1}-04-30")
        satellite_df = self.fetch_satellite_data(year)
        soil_df = self.fetch_soil_data()
        variety_df = self.fetch_variety_data()
        irrigation_df = self.fetch_irrigation_data()

        # Merge datasets
        merged_df = self._merge_datasets(yield_df, weather_df, satellite_df, soil_df,
                                       variety_df, irrigation_df)

        # Save processed dataset
        output_file = os.path.join(self.data_dir, 'processed', f'training_data_{year}.csv')
        merged_df.to_csv(output_file, index=False)
        print(f"Training dataset saved to {output_file}")

        return merged_df

    def _get_base_yield_for_district(self, district):
        """Get base yield for district based on historical averages"""
        # Realistic yield ranges for Punjab districts (quintal per hectare)
        yield_ranges = {
            'Ludhiana': 45, 'Jalandhar': 42, 'Amritsar': 40, 'Bathinda': 38,
            'Sangrur': 41, 'Patiala': 39, 'Ferozepur': 37, 'Muktsar': 35,
            'Faridkot': 36, 'Mansa': 34, 'Barnala': 38, 'Tarn Taran': 39,
            'Hoshiarpur': 35, 'Kapurthala': 36, 'Nawanshahr': 34,
            'Gurdaspur': 33, 'Pathankot': 32, 'Rupnagar': 31, 'Mohali': 30,
            'Fazilka': 36, 'Sri Muktsar Sahib': 35, 'Fatehgarh Sahib': 37
        }
        return yield_ranges.get(district, 35)

    def _generate_monthly_yields(self, base_yield, year):
        """Generate monthly yield data for Rabi season"""
        months = [11, 12, 1, 2, 3, 4]  # Rabi season months
        yields = {}

        for month in months:
            # Add seasonal variation and some randomness
            if month in [11, 12]:  # Sowing period
                multiplier = 0.1
            elif month == 1:  # Early growth
                multiplier = 0.3
            elif month == 2:  # Peak vegetation
                multiplier = 0.7
            elif month == 3:  # Grain filling
                multiplier = 0.9
            else:  # Harvest
                multiplier = 1.0

            # Add some random variation
            variation = np.random.normal(0, 0.05)  # 5% variation
            yields[month] = base_yield * multiplier * (1 + variation)

        return yields

    def _get_weather_stations(self):
        """Get weather station coordinates for districts"""
        # Approximate coordinates for weather stations in Punjab
        # In real implementation, these would be actual IMD station coordinates
        return {
            'Amritsar': {'lat': 31.63, 'lon': 74.87, 'station_id': 'AMR001'},
            'Tarn Taran': {'lat': 31.45, 'lon': 74.93, 'station_id': 'TTR001'},
            'Bathinda': {'lat': 30.21, 'lon': 74.95, 'station_id': 'BTH001'},
            'Mansa': {'lat': 29.98, 'lon': 75.38, 'station_id': 'MSA001'},
            'Muktsar': {'lat': 30.48, 'lon': 74.52, 'station_id': 'MKT001'},
            'Ferozepur': {'lat': 30.92, 'lon': 74.61, 'station_id': 'FZP001'},
            'Faridkot': {'lat': 30.67, 'lon': 74.76, 'station_id': 'FKT001'},
            'Ludhiana': {'lat': 30.90, 'lon': 75.85, 'station_id': 'LDH001'},
            'Jalandhar': {'lat': 31.33, 'lon': 75.58, 'station_id': 'JLD001'},
            'Hoshiarpur': {'lat': 31.53, 'lon': 75.91, 'station_id': 'HSP001'},
            'Kapurthala': {'lat': 31.38, 'lon': 75.38, 'station_id': 'KPT001'},
            'Nawanshahr': {'lat': 31.13, 'lon': 76.12, 'station_id': 'NSW001'},
            'Sangrur': {'lat': 30.25, 'lon': 75.84, 'station_id': 'SGR001'},
            'Barnala': {'lat': 30.38, 'lon': 75.55, 'station_id': 'BNL001'},
            'Patiala': {'lat': 30.34, 'lon': 76.39, 'station_id': 'PTL001'},
            'Fatehgarh Sahib': {'lat': 30.65, 'lon': 76.40, 'station_id': 'FGS001'},
            'Gurdaspur': {'lat': 32.04, 'lon': 75.41, 'station_id': 'GDP001'},
            'Pathankot': {'lat': 32.27, 'lon': 75.65, 'station_id': 'PTK001'},
            'Rupnagar': {'lat': 30.97, 'lon': 76.53, 'station_id': 'RPN001'},
            'Mohali': {'lat': 30.68, 'lon': 76.72, 'station_id': 'MHL001'},
            'Fazilka': {'lat': 30.40, 'lon': 74.02, 'station_id': 'FZK001'},
            'Sri Muktsar Sahib': {'lat': 30.48, 'lon': 74.52, 'station_id': 'SMS001'}
        }

    def _fetch_station_weather(self, station, start_date, end_date):
        """Fetch weather data for a specific station"""
        # In real implementation, this would call IMD API
        # For now, generate realistic weather data

        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        weather_data = []

        for date in date_range:
            # Generate realistic Punjab weather patterns
            month = date.month

            if month in [11, 12, 1]:  # Winter
                temp = np.random.normal(15, 3)  # 15°C average
                rainfall = np.random.exponential(0.5) if np.random.random() < 0.1 else 0
            elif month == 2:  # Peak winter
                temp = np.random.normal(12, 2)
                rainfall = np.random.exponential(0.3) if np.random.random() < 0.05 else 0
            elif month == 3:  # Spring
                temp = np.random.normal(20, 4)
                rainfall = np.random.exponential(0.2) if np.random.random() < 0.03 else 0
            else:  # April
                temp = np.random.normal(25, 5)
                rainfall = np.random.exponential(0.1) if np.random.random() < 0.02 else 0

            weather_data.append({
                'date': date.strftime('%Y-%m-%d'),
                'temperature_celsius': round(temp, 1),
                'rainfall_mm': round(rainfall, 1),
                'humidity_percent': np.random.normal(60, 10),
                'wind_speed_kmph': np.random.normal(5, 2),
                'station_id': station['station_id']
            })

        return weather_data

    def _generate_satellite_data(self, district, year):
        """Generate satellite vegetation indices data"""
        months = [11, 12, 1, 2, 3, 4]
        satellite_data = []

        for month in months:
            # NDVI follows crop growth pattern
            if month in [11, 12]:
                ndvi = np.random.normal(0.2, 0.05)
                fpar = np.random.normal(0.15, 0.03)
            elif month == 1:
                ndvi = np.random.normal(0.4, 0.08)
                fpar = np.random.normal(0.35, 0.05)
            elif month == 2:
                ndvi = np.random.normal(0.6, 0.1)
                fpar = np.random.normal(0.55, 0.08)
            elif month == 3:
                ndvi = np.random.normal(0.5, 0.08)
                fpar = np.random.normal(0.45, 0.06)
            else:
                ndvi = np.random.normal(0.3, 0.05)
                fpar = np.random.normal(0.25, 0.04)

            satellite_data.append({
                'month': month,
                'ndvi': round(max(0, min(1, ndvi)), 3),
                'fpar': round(max(0, min(1, fpar)), 3),
                'lai': round(ndvi * 3, 2),  # LAI approximation
                'data_source': 'MODIS_GEE'
            })

        return satellite_data

    def _get_soil_data_for_district(self, district):
        """Get soil characteristics for district"""
        # Use actual district data from configuration
        district_info = PUNJAB_DISTRICTS.get(district, {})

        # Punjab soil types vary by region
        if district in ['Amritsar', 'Tarn Taran', 'Ferozepur']:
            soil_ph = np.random.normal(7.8, 0.3)  # Slightly alkaline
            organic_carbon = np.random.normal(0.4, 0.1)
        elif district in ['Ludhiana', 'Jalandhar', 'Patiala']:
            soil_ph = np.random.normal(7.5, 0.2)  # Near neutral
            organic_carbon = np.random.normal(0.5, 0.1)
        else:
            soil_ph = np.random.normal(8.0, 0.4)  # More alkaline
            organic_carbon = np.random.normal(0.3, 0.1)

        return {
            'soil_ph': round(soil_ph, 1),
            'organic_carbon_percent': round(organic_carbon, 2),
            'soil_texture': district_info.get('soil_type', 'Sandy Loam'),
            'salinity_dsm': round(np.random.normal(0.5, 0.2), 2),
            'irrigation_coverage': district_info.get('irrigation_coverage', 0.9),
            'groundwater_depth': district_info.get('groundwater_depth', 8.0),
            'data_source': 'Punjab_Remote_Sensing_Centre'
        }

    def _merge_datasets(self, yield_df, weather_df, satellite_df, soil_df, variety_df=None, irrigation_df=None):
        """Merge all data sources into training dataset"""
        # Aggregate weather data to monthly level
        weather_df['date'] = pd.to_datetime(weather_df['date'])
        weather_df['year'] = weather_df['date'].dt.year
        weather_df['month'] = weather_df['date'].dt.month

        weather_monthly = weather_df.groupby(['district', 'year', 'month']).agg({
            'temperature_celsius': 'mean',
            'rainfall_mm': 'sum',
            'humidity_percent': 'mean',
            'wind_speed_kmph': 'mean'
        }).reset_index()

        # Start with base merge
        merged = yield_df.merge(
            weather_monthly,
            on=['district', 'year', 'month'],
            how='left'
        ).merge(
            satellite_df,
            on=['district', 'year', 'month'],
            how='left'
        ).merge(
            soil_df,
            on=['district'],
            how='left'
        )

        # Add variety data if available
        if variety_df is not None and not variety_df.empty:
            merged = merged.merge(
                variety_df,
                on=['district'],
                how='left'
            )

        # Add irrigation data if available
        if irrigation_df is not None and not irrigation_df.empty:
            merged = merged.merge(
                irrigation_df,
                on=['district'],
                how='left'
            )

        return merged

def main():
    """Main function to demonstrate data fetching"""
    fetcher = PunjabDataFetcher()

    print("Punjab Rabi Crop Yield Data Fetching System")
    print("=" * 50)

    # Create training dataset for 2023
    training_data = fetcher.create_training_dataset(2023)

    print(f"\nTraining dataset shape: {training_data.shape}")
    print(f"Columns: {list(training_data.columns)}")
    print(f"Districts covered: {training_data['district'].nunique()}")

    return training_data

if __name__ == "__main__":
    main()
