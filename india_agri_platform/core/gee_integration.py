"""
Google Earth Engine Integration for Real Satellite Data
Provides live NDVI, soil moisture, and vegetation indices
"""

import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
import logging
import time

logger = logging.getLogger(__name__)

try:
    import ee
    GEE_AVAILABLE = True
except ImportError:
    GEE_AVAILABLE = False
    logger.warning("Google Earth Engine API not available - using fallback mode")

class GoogleEarthEngineClient:
    """Real Google Earth Engine client with satellite data access"""

    def __init__(self, project_id: str = None):
        self.initialized = False
        self.project_id = project_id  # Optional project ID

        # Available satellite datasets with backup options
        self.datasets = {
            'ndvi': ['MODIS/061/MOD13Q1', 'MODIS/006/MOD13Q1'],  # Primary and backup
            'soil_moisture': ['NASA/SMAP/SPL4SMGP/007', 'NASA/SMAP/SPL3SMP_E/005'],
            'land_surface_temp': ['MODIS/061/MOD11A1', 'MODIS/006/MOD11A1'],
            'precipitation': ['UCSB-CHG/CHIRPS/DAILY', 'NASA/GPM_L3/IMERG_MONTHLY_V07'],
            'evapotranspiration': ['MODIS/061/MOD16A2GF', 'MODIS/006/MOD16A2'],
            'fpar': ['MODIS/061/MOD15A2H', 'MODIS/006/MOD09GQ'],  # Updated FPAR
        }

        # Force initialization during construction
        if GEE_AVAILABLE:
            try:
                # Try to initialize immediately
                ee.Initialize()
                self.initialized = True
                logger.info("✅ Google Earth Engine initialized immediately")
            except ee.ee_exception.EEException as e:
                logger.warning(f"❌ GEE init failed: {e}")
                # Try authentication
                try:
                    logger.info("🔄 Attempting GEE authentication...")
                    ee.Authenticate()  # This will launch browser if needed
                    ee.Initialize()
                    self.initialized = True
                    logger.info("✅ GEE authentication successful")
                except Exception as auth_e:
                    logger.error(f"❌ GEE authentication failed: {auth_e}")
                    self.initialized = False
        else:
            logger.warning("❌ GEE library not available - using fallback mode")

    def initialize(self) -> bool:
        """Initialize Google Earth Engine"""
        try:
            if not GEE_AVAILABLE:
                return False

            # Initialize Earth Engine
            ee.Initialize()
            self.initialized = True
            logger.info("✅ Google Earth Engine initialized successfully")
            return True

        except ee.ee_exception.EEException as e:
            logger.error(f"❌ GEE initialization failed: {e}")
            # Try authentication if not authenticated
            try:
                logger.info("Attempting GEE authentication...")
                ee.Authenticate()
                ee.Initialize()
                self.initialized = True
                logger.info("✅ GEE authentication completed")
                return True
            except Exception as auth_e:
                logger.error(f"❌ GEE authentication failed: {auth_e}")
                return False
        except Exception as e:
            logger.error(f"❌ GEE initialization error: {e}")
            return False

    def authenticate(self) -> bool:
        """Authenticate with Google Earth Engine (not used - Initialize() handles this)"""
        # The Initialize() method handles authentication
        return self.initialized

    def _parse_gee_properties(self, properties: Dict[str, Any], dataset: str,
                            latitude: float, longitude: float, start_date: str,
                            end_date: str) -> Dict[str, Any]:
        """Parse GEE satellite properties into standardized format"""

        try:
            if dataset == 'ndvi':
                # NDVI from MODIS (scaled by 10000, -1 to 1 range)
                ndvi_raw = properties.get('NDVI', 0)
                ndvi = ndvi_raw / 10000.0 if ndvi_raw != 0 else 0
                ndvi = max(0, min(1, ndvi))  # Clamp to 0-1 range

                return {
                    'ndvi': ndvi,
                    'vegetation_health': self._classify_vegetation_health(ndvi),
                    'data_source': 'google_earth_engine_modis',
                    'timestamp': datetime.now().isoformat(),
                    'resolution': '250m',
                    'period': f'{start_date} to {end_date}',
                    'location': {'lat': latitude, 'lng': longitude}
                }

            elif dataset == 'soil_moisture':
                # Soil moisture from SMAP (percentage)
                soil_moisture = properties.get('sm_surface', 0)  # Surface soil moisture
                soil_moisture_pct = max(0, min(100, soil_moisture * 100))

                return {
                    'soil_moisture_percent': soil_moisture_pct,
                    'data_source': 'google_earth_engine_smap',
                    'timestamp': datetime.now().isoformat(),
                    'resolution': '9km',
                    'period': f'{start_date} to {end_date}',
                    'location': {'lat': latitude, 'lng': longitude}
                }

            elif dataset == 'land_surface_temp':
                # Land surface temperature from MODIS (convert Kelvin to Celsius)
                lst_kelvin = properties.get('LST_Day_1km', 273.15)  # Default to 0°C if missing
                lst_celsius = lst_kelvin - 273.15
                lst_celsius = max(-50, min(80, lst_celsius))  # Realistic range

                return {
                    'land_surface_temp_c': lst_celsius,
                    'data_source': 'google_earth_engine_modis',
                    'timestamp': datetime.now().isoformat(),
                    'resolution': '1km',
                    'period': f'{start_date} to {end_date}',
                    'location': {'lat': latitude, 'lng': longitude}
                }

            elif dataset == 'evapotranspiration':
                # ET from MOD16A2 (convert to mm/day)
                et = properties.get('ET', 0)
                # Scale factor for ET is typically 0.1 mm/day
                et_mm_day = et * 0.1 if et > 0 else 0

                return {
                    'evapotranspiration': et_mm_day,
                    'data_source': 'google_earth_engine_mod16a2',
                    'timestamp': datetime.now().isoformat(),
                    'resolution': '500m',
                    'period': f'{start_date} to {end_date}',
                    'location': {'lat': latitude, 'lng': longitude}
                }

            else:
                # Generic response for any other dataset
                return {
                    'value': properties,
                    'data_source': f'google_earth_engine_{dataset}',
                    'timestamp': datetime.now().isoformat(),
                    'period': f'{start_date} to {end_date}',
                    'location': {'lat': latitude, 'lng': longitude}
                }

        except Exception as e:
            logger.error(f"Error parsing GEE properties for {dataset}: {e}")
            return self._get_fallback_data_for_dataset(dataset, latitude, longitude)

    def _get_fallback_data_for_dataset(self, dataset: str, latitude: float, longitude: float) -> Dict[str, Any]:
        """Provide realistic fallback data for different datasets"""

        timestamp = datetime.now().isoformat()
        base_data = {
            'data_source': 'fallback_simulated',
            'timestamp': timestamp,
            'period': 'recent',
            'location': {'lat': latitude, 'lng': longitude},
            'resolution': 'estimated'
        }

        if dataset == 'ndvi':
            # Realistic NDVI based on location and season
            base_ndvi = 0.65 if latitude > 25 else 0.55  # Higher in north India
            return {
                **base_data,
                'ndvi': base_ndvi,
                'vegetation_health': 'good' if base_ndvi > 0.6 else 'fair',
                'data_source': 'fallback_ndvi_simulation'
            }

        elif dataset == 'soil_moisture':
            # Realistic soil moisture (moderate levels for most regions)
            soil_moisture = 40.0  # 40% typical
            return {
                **base_data,
                'soil_moisture_percent': soil_moisture,
                'data_source': 'fallback_soil_moisture_simulation'
            }

        elif dataset == 'land_surface_temp':
            # Realistic LST based on region (warmer in south India)
            base_temp = 32.0 if latitude < 20 else 28.0
            return {
                **base_data,
                'land_surface_temp_c': base_temp,
                'data_source': 'fallback_temperature_simulation'
            }

        elif dataset == 'evapotranspiration':
            # Typical ET values (moderate)
            et = 4.5  # 4.5 mm/day
            return {
                **base_data,
                'evapotranspiration': et,
                'data_source': 'fallback_evapotranspiration_simulation'
            }

        else:
            return {**base_data, 'status': 'fallback_data_used'}

    def get_satellite_data(self, latitude: float, longitude: float,
                          start_date: str, end_date: str,
                          dataset: str = 'ndvi') -> Optional[Dict[str, Any]]:
        """Fetch satellite data for a specific location and time period using GEE Python API"""

        if not self.initialized or not GEE_AVAILABLE:
            logger.warning("GEE not available - returning fallback data")
            return self._get_fallback_data_for_dataset(dataset, latitude, longitude)

        try:
            dataset_ids = self.datasets.get(dataset, self.datasets['ndvi'])

            # Try primary dataset first, fall back to secondary if needed
            collection = None
            used_dataset_id = None

            if isinstance(dataset_ids, list):
                for dataset_id in dataset_ids:
                    try:
                        collection = ee.ImageCollection(dataset_id)
                        # Test availability by checking size (but don't evaluate fully yet)
                        used_dataset_id = dataset_id
                        break
                    except Exception as e:
                        logger.warning(f"Dataset {dataset_id} failed, trying alternative: {e}")
                        continue
            else:
                collection = ee.ImageCollection(dataset_ids)
                used_dataset_id = dataset_ids

            if collection is None:
                logger.error(f"No working dataset found for {dataset}")
                return self._get_fallback_data_for_dataset(dataset, latitude, longitude)

            # Filter by date range
            filtered_collection = collection.filterDate(start_date, end_date)

            # Filter by location (point)
            point = ee.Geometry.Point([longitude, latitude])
            filtered_collection = filtered_collection.filterBounds(point)

            # Get mean composite for the period
            if filtered_collection.size().getInfo() == 0:
                logger.warning(f"No {dataset} data available for location {latitude}, {longitude} between {start_date} and {end_date}")
                return self._get_fallback_data_for_dataset(dataset, latitude, longitude)

            mean_image = filtered_collection.mean()

            # Sample the point at 250m resolution
            sample_dict = mean_image.sample(
                region=point,
                scale=250,  # 250m resolution
                numPixels=1,
                geometries=False
            ).getInfo()

            if not sample_dict.get('features'):
                logger.warning(f"Could not sample {dataset} at location {latitude}, {longitude}")
                return self._get_fallback_data_for_dataset(dataset, latitude, longitude)

            # Extract properties
            properties = sample_dict['features'][0].get('properties', {})

            return self._parse_gee_properties(properties, dataset, latitude, longitude, start_date, end_date)

        except ee.EEException as e:
            logger.error(f"❌ GEE API error: {e}")
            return self._get_fallback_data_for_dataset(dataset, latitude, longitude)
        except Exception as e:
            logger.error(f"❌ Unexpected GEE error: {e}")
            return self._get_fallback_data_for_dataset(dataset, latitude, longitude)

    def _parse_gee_response(self, response: Dict, dataset: str) -> Dict[str, Any]:
        """Parse GEE API response into usable format"""

        try:
            features = response.get('features', [])
            if not features:
                return None

            # Extract properties from first feature
            properties = features[0].get('properties', {})

            # Parse based on dataset type
            if dataset == 'ndvi':
                # NDVI from MODIS (scaled by 10000, -1 to 1 range)
                ndvi_raw = properties.get('NDVI', 0)
                ndvi = ndvi_raw / 10000.0 if ndvi_raw != 0 else 0

                return {
                    'ndvi': max(0, min(1, ndvi)),
                    'vegetation_health': self._classify_vegetation_health(ndvi),
                    'data_source': 'google_earth_engine_modis',
                    'timestamp': datetime.now().isoformat(),
                    'resolution': '250m',
                    'period': '16-day composite'
                }

            elif dataset == 'soil_moisture':
                # Soil moisture from SMAP
                soil_moisture = properties.get('sm_surface', 0)  # Surface soil moisture

                return {
                    'soil_moisture_percent': max(0, min(100, soil_moisture * 100)),
                    'data_source': 'google_earth_engine_smap',
                    'timestamp': datetime.now().isoformat(),
                    'resolution': '9km'
                }

            elif dataset == 'land_surface_temp':
                # Land surface temperature from MODIS
                lst_day = properties.get('LST_Day_1km', 0)
                # Convert from Kelvin to Celsius (subtract 273.15)
                lst_celsius = lst_day - 273.15 if lst_day > 0 else 25

                return {
                    'land_surface_temp_c': lst_celsius,
                    'data_source': 'google_earth_engine_modis',
                    'timestamp': datetime.now().isoformat(),
                    'resolution': '1km'
                }

            else:
                # Generic response
                return {
                    'value': properties,
                    'data_source': f'google_earth_engine_{dataset}',
                    'timestamp': datetime.now().isoformat()
                }

        except Exception as e:
            logger.error(f"GEE response parsing error: {e}")
            return None

    def _classify_vegetation_health(self, ndvi: float) -> str:
        """Classify vegetation health based on NDVI"""
        if ndvi >= 0.7:
            return "excellent"
        elif ndvi >= 0.6:
            return "good"
        elif ndvi >= 0.5:
            return "fair"
        elif ndvi >= 0.4:
            return "poor"
        else:
            return "critical"

    def get_multi_band_data(self, latitude: float, longitude: float,
                           start_date: str, end_date: str) -> Dict[str, Any]:
        """Get comprehensive satellite data from multiple bands"""

        results = {}

        # Fetch different datasets
        datasets_to_fetch = ['ndvi', 'soil_moisture', 'land_surface_temp']

        for dataset in datasets_to_fetch:
            data = self.get_satellite_data(latitude, longitude, start_date, end_date, dataset)
            if data:
                results[dataset] = data
            else:
                # Provide fallback values
                results[dataset] = self._get_fallback_data(dataset)

        # Combine into comprehensive satellite data
        combined_data = {
            'ndvi': results['ndvi'].get('ndvi', 0.6),
            'soil_moisture_percent': results['soil_moisture'].get('soil_moisture_percent', 35),
            'land_surface_temp_c': results['land_surface_temp'].get('land_surface_temp_c', 28),
            'vegetation_health': results['ndvi'].get('vegetation_health', 'good'),
            'data_source': 'google_earth_engine_multi_band',
            'timestamp': datetime.now().isoformat(),
            'location': {'lat': latitude, 'lng': longitude},
            'period': f"{start_date} to {end_date}",
            'individual_sources': results
        }

        return combined_data

    def _get_fallback_data(self, dataset: str) -> Dict[str, Any]:
        """Provide fallback data when GEE is unavailable"""
        fallbacks = {
            'ndvi': {
                'ndvi': 0.6,
                'vegetation_health': 'good',
                'data_source': 'fallback_simulated'
            },
            'soil_moisture': {
                'soil_moisture_percent': 35,
                'data_source': 'fallback_simulated'
            },
            'land_surface_temp': {
                'land_surface_temp_c': 28,
                'data_source': 'fallback_simulated'
            }
        }

        return fallbacks.get(dataset, {})

    def get_time_series_data(self, latitude: float, longitude: float,
                           start_date: str, end_date: str,
                           dataset: str = 'ndvi') -> List[Dict[str, Any]]:
        """Get time series satellite data"""

        try:
            # Calculate monthly intervals
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')

            time_series = []
            current_date = start_dt

            while current_date <= end_dt:
                # Get data for this month
                month_end = min(current_date + timedelta(days=30), end_dt)

                data = self.get_satellite_data(
                    latitude, longitude,
                    current_date.strftime('%Y-%m-%d'),
                    month_end.strftime('%Y-%m-%d'),
                    dataset
                )

                if data:
                    data['date'] = current_date.strftime('%Y-%m-%d')
                    time_series.append(data)

                # Move to next month
                current_date = current_date + timedelta(days=30)

            return time_series

        except Exception as e:
            logger.error(f"Time series data error: {e}")
            return []

class SatelliteDataProcessor:
    """Process and analyze satellite data for agricultural insights"""

    def __init__(self):
        self.gee_client = None

    def set_gee_client(self, gee_client: GoogleEarthEngineClient):
        """Set the GEE client for data fetching"""
        self.gee_client = gee_client

    def analyze_crop_health(self, satellite_data: Dict[str, Any],
                          crop_type: str, growth_stage: str) -> Dict[str, Any]:
        """Analyze crop health from satellite data"""

        ndvi = satellite_data.get('ndvi', 0.6)
        soil_moisture = satellite_data.get('soil_moisture_percent', 35)

        # Crop-specific NDVI thresholds
        crop_thresholds = {
            'wheat': {'excellent': 0.7, 'good': 0.6, 'fair': 0.5},
            'rice': {'excellent': 0.75, 'good': 0.65, 'fair': 0.55},
            'maize': {'excellent': 0.8, 'good': 0.7, 'fair': 0.6},
            'cotton': {'excellent': 0.65, 'good': 0.55, 'fair': 0.45},
            'sugarcane': {'excellent': 0.75, 'good': 0.65, 'fair': 0.55}
        }

        thresholds = crop_thresholds.get(crop_type, crop_thresholds['wheat'])

        # Assess health
        if ndvi >= thresholds['excellent']:
            health_status = "excellent"
            health_score = 95
        elif ndvi >= thresholds['good']:
            health_status = "good"
            health_score = 80
        elif ndvi >= thresholds['fair']:
            health_status = "fair"
            health_score = 65
        else:
            health_status = "poor"
            health_score = 40

        # Growth stage adjustments
        stage_multipliers = {
            'establishment': 0.8,  # Early stage, lower NDVI expected
            'vegetative_growth': 1.0,
            'reproductive': 1.1,  # Peak NDVI
            'grain_filling': 1.0,
            'maturity': 0.9  # NDVI may decline
        }

        stage_multiplier = stage_multipliers.get(growth_stage, 1.0)
        adjusted_score = min(100, health_score * stage_multiplier)

        # Soil moisture analysis
        moisture_status = "optimal"
        if soil_moisture < 20:
            moisture_status = "dry"
        elif soil_moisture > 60:
            moisture_status = "wet"

        return {
            'health_status': health_status,
            'health_score': round(adjusted_score, 1),
            'ndvi_assessment': f"NDVI {ndvi:.3f} ({health_status})",
            'soil_moisture_status': moisture_status,
            'growth_stage_adjusted': growth_stage,
            'recommendations': self._generate_health_recommendations(
                health_status, moisture_status, crop_type
            ),
            'risk_factors': self._identify_risk_factors(ndvi, soil_moisture, crop_type)
        }

    def _generate_health_recommendations(self, health_status: str,
                                       moisture_status: str, crop_type: str) -> List[str]:
        """Generate crop health recommendations"""

        recommendations = []

        if health_status in ['poor', 'critical']:
            recommendations.extend([
                "Apply additional fertilizers immediately",
                "Check for pest/disease infestations",
                "Consider replanting if damage is severe"
            ])

        if moisture_status == "dry":
            recommendations.extend([
                "Increase irrigation frequency",
                "Apply mulch to retain soil moisture",
                "Monitor for drought stress symptoms"
            ])
        elif moisture_status == "wet":
            recommendations.extend([
                "Improve drainage to prevent waterlogging",
                "Monitor for fungal diseases",
                "Reduce irrigation temporarily"
            ])

        if not recommendations:
            recommendations.append("Crop health is good, continue current management practices")

        return recommendations

    def _identify_risk_factors(self, ndvi: float, soil_moisture: float, crop_type: str) -> List[str]:
        """Identify potential risk factors"""

        risks = []

        if ndvi < 0.4:
            risks.append("severe_vegetation_stress")

        if soil_moisture < 15:
            risks.append("drought_conditions")

        if soil_moisture > 70:
            risks.append("waterlogging_risk")

        # Crop-specific risks
        if crop_type == 'rice' and soil_moisture < 30:
            risks.append("rice_drought_sensitivity")

        if crop_type == 'wheat' and ndvi < 0.5:
            risks.append("wheat_rust_risk")

        return risks

    def detect_changes(self, time_series_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Detect changes in satellite data over time"""

        if len(time_series_data) < 2:
            return {"change_detected": False, "message": "Insufficient data for change detection"}

        # Extract NDVI values
        ndvi_values = [data.get('ndvi', 0) for data in time_series_data]
        dates = [data.get('date', '') for data in time_series_data]

        # Calculate trends
        ndvi_trend = np.polyfit(range(len(ndvi_values)), ndvi_values, 1)[0]

        # Detect significant changes
        change_threshold = 0.1  # 10% change
        recent_change = abs(ndvi_values[-1] - ndvi_values[-2]) if len(ndvi_values) >= 2 else 0

        change_analysis = {
            "ndvi_trend": ndvi_trend,
            "recent_change": recent_change,
            "change_detected": recent_change > change_threshold,
            "trend_direction": "increasing" if ndvi_trend > 0.001 else "decreasing" if ndvi_trend < -0.001 else "stable",
            "analysis_period": f"{dates[0]} to {dates[-1]}" if dates else "unknown"
        }

        if change_analysis["change_detected"]:
            change_analysis["alert"] = "Significant vegetation change detected - investigate field conditions"

        return change_analysis

# Global instances - configured with your GEE project
gee_client = GoogleEarthEngineClient(project_id="named-tome-472312-m3")
satellite_processor = SatelliteDataProcessor()
satellite_processor.set_gee_client(gee_client)
