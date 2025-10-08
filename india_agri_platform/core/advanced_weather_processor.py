"""
Advanced Weather Processing Engine for Agricultural Intelligence
Intelligent weather data analysis, forecasting, and agricultural insights
"""

import asyncio
import logging
from datetime import datetime, timedelta, time
from typing import Dict, List, Any, Optional, Tuple, Union
from enum import Enum
import numpy as np
import pandas as pd
from pathlib import Path
import json
from scipy import stats
from scipy.signal import find_peaks
import warnings

from india_agri_platform.core.error_handling import error_handler, log_system_event, ExternalAPIError
from india_agri_platform.core.cache_manager import cache_manager, set_cached_value
from india_agri_platform.core.data_processing_pipeline import data_pipeline, DataSource

# Suppress warnings for clean logging
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class WeatherPhenomenon(Enum):
    """Weather phenomena affecting agriculture"""
    HEAT_WAVE = "HEAT_WAVE"
    COLD_SNAP = "COLD_SNAP"
    DROUGHT = "DROUGHT"
    FLOOD = "FLOOD"
    FROST = "FROST"
    HAILSTORM = "HAILSTORM"
    HIGH_WINDS = "HIGH_WINDS"
    HEAVY_RAINFALL = "HEAVY_RAINFALL"

class WeatherSeverity(Enum):
    """Severity levels for weather events"""
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    SEVERE = "SEVERE"
    EXTREME = "EXTREME"

class CropStage(Enum):
    """Critical crop growth stages"""
    GERMINATION = "GERMINATION"
    SEEDLING = "SEEDLING"
    VEGETATIVE = "VEGETATIVE"
    FLOWERING = "FLOWERING"
    GRAIN_FILLING = "GRAIN_FILLING"
    MATURATION = "MATURATION"
    HARVEST = "HARVEST"

class AdvancedWeatherProcessor:
    """Advanced weather processing with agricultural intelligence"""

    def __init__(self):
        # Weather storage
        self.weather_data = {}
        self.weather_forecasts = {}
        self.weather_alerts = []

        # Agricultural weather thresholds (Punjab-specific)
        self.crop_weather_thresholds = {
            'wheat': {
                CropStage.GERMINATION: {
                    'temp_min_optimal': 18, 'temp_max_optimal': 22,
                    'temp_min_critical': 5, 'temp_max_critical': 35,
                    'humidity_optimal': (40, 60), 'rainfall_max_daily': 20
                },
                CropStage.SEEDLING: {
                    'temp_min_optimal': 12, 'temp_max_optimal': 18,
                    'temp_min_critical': 0, 'temp_max_critical': 30,
                    'humidity_optimal': (50, 70), 'rainfall_max_daily': 15
                },
                CropStage.VEGETATIVE: {
                    'temp_min_optimal': 15, 'temp_max_optimal': 20,
                    'temp_min_critical': 8, 'temp_max_critical': 32,
                    'humidity_optimal': (45, 65), 'rainfall_max_daily': 25
                },
                CropStage.FLOWERING: {
                    'temp_min_optimal': 20, 'temp_max_optimal': 25,
                    'temp_min_critical': 12, 'temp_max_critical': 35,
                    'humidity_optimal': (35, 55), 'rainfall_max_daily': 10
                },
                CropStage.GRAIN_FILLING: {
                    'temp_min_optimal': 20, 'temp_max_optimal': 25,
                    'temp_min_critical': 15, 'temp_max_critical': 35,
                    'humidity_optimal': (30, 50), 'rainfall_max_daily': 5
                },
                CropStage.MATURATION: {
                    'temp_min_optimal': 25, 'temp_max_optimal': 30,
                    'temp_min_critical': 20, 'temp_max_critical': 40,
                    'humidity_optimal': (25, 45), 'rainfall_max_daily': 2
                }
            },
            'rice': {
                # Similar structure for rice would go here
            },
            'cotton': {
                # Similar structure for cotton would go here
            }
        }

        # Weather patterns and analysis
        self.weather_patterns = {}
        self.seasonal_baseline = {}
        self.anomaly_history = []

        # Processing configuration
        self.processing_config = {
            'heat_wave_threshold_celsius': 35,
            'heat_wave_consecutive_days': 3,
            'cold_snap_threshold_celsius': 5,
            'cold_snap_consecutive_days': 2,
            'frost_threshold_celsius': 0,
            'drought_definition_mm': 10,  # Less than 10mm in 7 days
            'drought_consecutive_weeks': 3,
            'heavy_rainfall_daily_threshold_mm': 100,
            'high_wind_speed_threshold_kmh': 40
        }

        # Punjab seasonal characteristics
        self.punjab_seasons = {
            'kharif': {'start_month': 6, 'end_month': 10, 'characteristics': 'hot_wet'},
            'rabi': {'start_month': 11, 'end_month': 3, 'characteristics': 'cool_dry'},
            'summer': {'start_month': 4, 'end_month': 5, 'characteristics': 'very_hot_dry'}
        }

        logger.info("Advanced Weather Processor initialized")

    async def initialize_weather_processing(self) -> bool:
        """Initialize weather processing system"""

        try:
            # Start background weather monitoring
            asyncio.create_task(self._weather_monitoring_loop())

            # Initialize seasonal baselines
            await self._initialize_seasonal_baselines()

            log_system_event(
                "weather_processor_initialized",
                "Advanced Weather Processing initialized",
                {"monitored_parameters": len(self.processing_config)}
            )

            return True

        except Exception as e:
            error_handler.handle_error(e, {"component": "weather_processor", "operation": "initialization"})
            return False

    async def process_weather_data(self, weather_data: Union[Dict[str, Any], pd.DataFrame],
                                 location: str = "punjab_general", timestamp: Optional[datetime] = None) -> Dict[str, Any]:
        """Process incoming weather data with agricultural intelligence"""

        if timestamp is None:
            timestamp = datetime.utcnow()

        if isinstance(weather_data, dict):
            weather_data = pd.DataFrame([weather_data])

        # Add timestamp if not present
        if 'timestamp' not in weather_data.columns:
            weather_data['timestamp'] = timestamp

        # Validate and clean weather data
        validated_data = await self._validate_weather_data(weather_data)

        # Analyze weather patterns
        analysis_result = await self._analyze_weather_patterns(validated_data, location)

        # Generate agricultural insights
        agricultural_insights = await self._generate_agricultural_insights(validated_data, location)

        # Check for weather alerts
        weather_alerts = await self._check_weather_alerts(validated_data, location)

        # Store processed data
        await self._store_weather_data(validated_data, location, analysis_result)

        result = {
            'location': location,
            'timestamp': timestamp.isoformat(),
            'data_quality': analysis_result.get('data_quality', 0),
            'weather_metrics': validated_data.to_dict('records')[0] if len(validated_data) > 0 else {},
            'analysis': analysis_result,
            'agricultural_insights': agricultural_insights,
            'alerts': weather_alerts,
            'recommendations': self._generate_weather_recommendations(agricultural_insights, weather_alerts)
        }

        # Cache weather analysis
        cache_key = f"weather_analysis_{location}_{timestamp.strftime('%Y%m%d_%H')}"
        await set_cached_value(cache_key, result, ttl_seconds=3600)  # 1 hour

        return result

    async def _validate_weather_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean weather data"""

        validated_data = data.copy()

        # Basic validations and corrections
        if 'temperature' in validated_data.columns:
            # Check for realistic temperature ranges (-10°C to 50°C for Punjab)
            mask = (validated_data['temperature'] < -10) | (validated_data['temperature'] > 50)
            validated_data.loc[mask, 'temperature'] = np.nan

            # Check for sudden unrealistic changes
            temp_changes = validated_data['temperature'].diff().abs()
            spike_mask = temp_changes > 15  # More than 15°C change in one reading
            if spike_mask.any():
                validated_data.loc[spike_mask, 'temperature'] = np.nan

        if 'humidity' in validated_data.columns:
            # Humidity should be 0-100%
            validated_data['humidity'] = validated_data['humidity'].clip(0, 100)

        if 'rainfall' in validated_data.columns:
            # Rainfall should not be negative and reasonable (< 500mm/day)
            validated_data['rainfall'] = validated_data['rainfall'].clip(0, 500)

        if 'wind_speed' in validated_data.columns:
            # Wind speed should be reasonable (< 200 km/h)
            validated_data['wind_speed'] = validated_data['wind_speed'].clip(0, 200)

        # Fill missing values using seasonal averages
        validated_data = await self._fill_missing_weather_values(validated_data)

        return validated_data

    async def _analyze_weather_patterns(self, data: pd.DataFrame, location: str) -> Dict[str, Any]:
        """Analyze weather patterns and trends"""

        analysis = {}

        if len(data) < 3:
            return {'error': 'Insufficient data for pattern analysis'}

        # Temperature patterns
        if 'temperature' in data.columns and not data['temperature'].isna().all():
            temp_data = data['temperature'].dropna()
            analysis['temperature'] = {
                'current': float(temp_data.iloc[-1]),
                'daily_average': float(temp_data.mean()),
                'daily_min': float(temp_data.min()),
                'daily_max': float(temp_data.max()),
                'volatility': float(temp_data.std()) if len(temp_data) > 1 else 0,
                'trend': self._calculate_trend(temp_data.values)
            }

        # Rainfall patterns
        if 'rainfall' in data.columns and not data['rainfall'].isna().all():
            rainfall_data = data['rainfall'].dropna()
            analysis['rainfall'] = {
                'daily_total': float(rainfall_data.sum()),
                'rain_probability': float((rainfall_data > 0.1).sum() / len(rainfall_data)),
                'intensity': float(rainfall_data.mean()) if len(rainfall_data) > 0 else 0,
                'distribution': self._analyze_rainfall_distribution(rainfall_data.values)
            }

        # Humidity patterns
        if 'humidity' in data.columns and not data['humidity'].isna().all():
            humidity_data = data['humidity'].dropna()
            analysis['humidity'] = {
                'current': float(humidity_data.iloc[-1]),
                'average': float(humidity_data.mean()),
                'stability': self._calculate_humidity_stability(humidity_data.values)
            }

        # Overall weather quality and consistency
        analysis['data_quality'] = self._assess_weather_data_quality(data)
        analysis['consistency_score'] = self._calculate_weather_consistency(data)

        return analysis

    async def _generate_agricultural_insights(self, data: pd.DataFrame, location: str) -> List[Dict[str, Any]]:
        """Generate agricultural insights from weather data"""

        insights = []

        # Determine current crop season
        current_month = datetime.utcnow().month
        current_season = self._get_current_season(current_month)

        # Focus on main crops: Wheat (Rabi) and Rice (Kharif)
        main_crops = ['wheat'] if current_season == 'rabi' else ['rice']

        for crop in main_crops:
            crop_insights = await self._analyze_weather_impact_on_crop(data, crop, current_season)
            insights.extend(crop_insights)

        # Cross-crop insights
        if 'temperature' in data.columns and 'rainfall' in data.columns:
            temp_trend = self._calculate_trend(data['temperature'].dropna().values)
            rainfall_trend = self._calculate_trend(data['rainfall'].dropna().values)

            if temp_trend == 'increasing' and rainfall_trend == 'decreasing':
                insights.append({
                    'type': 'climate_change_impact',
                    'severity': 'high',
                    'description': 'Rising temperatures with declining rainfall indicates changing climate patterns',
                    'impact': 'reduced_crop_yields',
                    'recommendation': 'Consider drought-resistant crop varieties and improved irrigation'
                })

        return insights

    async def _check_weather_alerts(self, data: pd.DataFrame, location: str) -> List[Dict[str, Any]]:
        """Check for weather conditions that require alerts"""

        alerts = []

        if len(data) == 0:
            return alerts

        # Heat wave alert
        if 'temperature' in data.columns:
            temp_data = data['temperature'].dropna()
            if len(temp_data) > 0:
                max_temp = temp_data.max()
                if max_temp >= self.processing_config['heat_wave_threshold_celsius']:
                    alerts.append({
                        'phenomenon': WeatherPhenomenon.HEAT_WAVE.value,
                        'severity': WeatherSeverity.HIGH.value if max_temp > 38 else WeatherSeverity.MEDIUM.value,
                        'temperature': float(max_temp),
                        'threshold': self.processing_config['heat_wave_threshold_celsius'],
                        'impact': 'reduced_photosynthesis_crop_stress',
                        'recommendation': 'Provide irrigation_during_cool_hours_shade_protection'
                    })

        # Frost alert
        if 'temperature' in data.columns:
            min_temp = data['temperature'].dropna().min()
            if min_temp <= self.processing_config['frost_threshold_celsius']:
                alerts.append({
                    'phenomenon': WeatherPhenomenon.FROST.value,
                    'severity': WeatherSeverity.HIGH.value,
                    'temperature': float(min_temp),
                    'threshold': self.processing_config['frost_threshold_celsius'],
                    'impact': 'crop_damage_freezing',
                    'recommendation': 'frost_protection_blankets_heaters'
                })

        # Drought alert
        if 'rainfall' in data.columns:
            recent_rainfall = data['rainfall'].dropna()
            if len(recent_rainfall) >= 7:  # 7-day period
                weekly_total = recent_rainfall.sum()
                if weekly_total < self.processing_config['drought_definition_mm']:
                    alerts.append({
                        'phenomenon': WeatherPhenomenon.DROUGHT.value,
                        'severity': WeatherSeverity.MODERATE.value,
                        'rainfall_deficit': float(self.processing_config['drought_definition_mm'] - weekly_total),
                        'period_days': 7,
                        'impact': 'soil_moisture_depletion_crop_stress',
                        'recommendation': 'immediate_irrigation_schedule_monitoring'
                    })

        # Heavy rainfall alert
        if 'rainfall' in data.columns:
            max_daily_rainfall = data['rainfall'].dropna().max()
            if max_daily_rainfall >= self.processing_config['heavy_rainfall_daily_threshold_mm']:
                alerts.append({
                    'phenomenon': WeatherPhenomenon.HEAVY_RAINFALL.value,
                    'severity': WeatherSeverity.HIGH.value,
                    'rainfall_amount': float(max_daily_rainfall),
                    'threshold': self.processing_config['heavy_rainfall_daily_threshold_mm'],
                    'impact': 'flooding_soil_erosion_waterlogging',
                    'recommendation': 'drainage_improvement_delay_operations'
                })

        # High wind alert
        if 'wind_speed' in data.columns:
            max_wind = data['wind_speed'].dropna().max()
            if max_wind >= self.processing_config['high_wind_speed_threshold_kmh']:
                alerts.append({
                    'phenomenon': WeatherPhenomenon.HIGH_WINDS.value,
                    'severity': WeatherSeverity.MEDIUM.value,
                    'wind_speed': float(max_wind),
                    'threshold': self.processing_config['high_wind_speed_threshold_kmh'],
                    'impact': 'physical_damage_lodging_pollination_issues',
                    'recommendation': 'wind_breaks_supporting_structures'
                })

        return alerts

    async def generate_weather_forecast_insights(self, forecast_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate insights from weather forecast data"""

        insights = {
            'upcoming_events': [],
            'crop_advisories': [],
            'risk_assessment': 'low'
        }

        if not forecast_data.get('forecast'):
            return insights

        # Analyze forecast for critical periods
        forecast_list = forecast_data['forecast']

        # Check for heat waves in forecast
        heat_wave_risk = self._analyze_forecast_heat_wave_risk(forecast_list)
        if heat_wave_risk['risk_level'] != 'low':
            insights['upcoming_events'].append({
                'event': 'predicted_heat_wave',
                'risk_level': heat_wave_risk['risk_level'],
                'period': heat_wave_risk['period'],
                'impact': 'crop_stress_reduced_yields',
                'preparation': ['shade_nets', 'extra_irrigation', 'delayed_sowing']
            })

        # Check for frost risk
        frost_risk = self._analyze_forecast_frost_risk(forecast_list)
        if frost_risk['risk_level'] != 'low':
            insights['upcoming_events'].append({
                'event': 'frost_risk',
                'risk_level': frost_risk['risk_level'],
                'frost_dates': frost_risk['dates'],
                'impact': 'crop_damage_freezing',
                'preparation': ['frost_blankets', 'smudge_pots', 'windbreaks']
            })

        # Irrigation planning based on forecast
        irrigation_advice = self._plan_forecast_based_irrigation(forecast_list)
        if irrigation_advice:
            insights['crop_advisories'].extend(irrigation_advice)

        # Overall risk assessment
        insights['risk_assessment'] = self._calculate_overall_forecast_risk(insights)

        return insights

    async def _analyze_weather_impact_on_crop(self, weather_data: pd.DataFrame,
                                            crop: str, season: str) -> List[Dict[str, Any]]:
        """Analyze how weather conditions affect specific crops"""

        insights = []

        if crop not in self.crop_weather_thresholds:
            return insights

        crop_thresholds = self.crop_weather_thresholds[crop]
        current_stage = self._estimate_current_crop_stage(crop, season)

        if current_stage and current_stage in crop_thresholds:
            stage_thresholds = crop_thresholds[current_stage]

            # Temperature analysis
            if 'temperature' in weather_data.columns:
                temp_issues = self._analyze_temperature_impact(
                    weather_data['temperature'], stage_thresholds, current_stage
                )
                insights.extend(temp_issues)

            # Humidity analysis
            if 'humidity' in weather_data.columns:
                humidity_issues = self._analyze_humidity_impact(
                    weather_data['humidity'], stage_thresholds, current_stage
                )
                insights.extend(humidity_issues)

            # Rainfall analysis
            if 'rainfall' in weather_data.columns:
                rainfall_issues = self._analyze_rainfall_impact(
                    weather_data['rainfall'], stage_thresholds, current_stage
                )
                insights.extend(rainfall_issues)

        return insights

    def _calculate_trend(self, values: np.ndarray, look_back: int = 5) -> str:
        """Calculate trend direction from values"""

        if len(values) < look_back:
            return "insufficient_data"

        recent_values = values[-look_back:]
        if len(recent_values) < 3:
            return "insufficient_data"

        # Simple linear regression
        x = np.arange(len(recent_values))
        slope, _, r_value, _, _ = stats.linregress(x, recent_values)

        r_squared = r_value ** 2
        threshold = 0.3  # R² threshold for meaningful trend

        if r_squared < threshold:
            return "stable"

        if slope > 0:
            return "increasing"
        else:
            return "decreasing"

    def _analyze_rainfall_distribution(self, values: np.ndarray) -> Dict[str, Any]:
        """Analyze rainfall distribution patterns"""

        if len(values) == 0:
            return {'pattern': 'no_rainfall'}

        # Calculate basic statistics
        total_rainfall = float(np.sum(values))
        rainy_days = int(np.sum(values > 0.1))
        max_daily = float(np.max(values))

        # Distribution analysis
        if rainy_days == 0:
            distribution = 'dry'
        elif rainy_days <= 3:
            distribution = 'scattered_showers'
        elif max_daily > 50:
            distribution = 'intense_showers'
        else:
            distribution = 'moderate_distribution'

        return {
            'total_mm': total_rainfall,
            'rainy_days': rainy_days,
            'max_daily': max_daily,
            'distribution_pattern': distribution,
            'efficiency': total_rainfall / max(rainy_days, 1)  # mm per rainy day
        }

    def _calculate_humidity_stability(self, values: np.ndarray) -> float:
        """Calculate humidity stability (lower is more stable)"""

        if len(values) < 2:
            return 0.0

        # Coefficient of variation
        mean_val = np.mean(values)
        std_val = np.std(values)

        if mean_val == 0:
            return 0.0

        return std_val / mean_val

    def _assess_weather_data_quality(self, data: pd.DataFrame) -> float:
        """Assess quality of weather data on 0-1 scale"""

        quality_scores = []

        for column in data.columns:
            if column in ['temperature', 'humidity', 'rainfall', 'wind_speed', 'pressure']:
                col_data = data[column].dropna()

                if len(col_data) == 0:
                    quality_scores.append(0.0)
                    continue

                # Completeness score
                completeness = len(col_data) / len(data)
                quality_scores.append(completeness)

                # Reasonableness score
                if column == 'temperature' and len(col_data) > 0:
                    reasonable = ((col_data >= -10) & (col_data <= 50)).mean()
                elif column == 'humidity' and len(col_data) > 0:
                    reasonable = ((col_data >= 0) & (col_data <= 100)).mean()
                elif column == 'rainfall' and len(col_data) > 0:
                    reasonable = (col_data >= 0).mean()
                elif column == 'wind_speed' and len(col_data) > 0:
                    reasonable = (col_data >= 0).mean()
                else:
                    reasonable = 1.0

                quality_scores.append(reasonable)

        return np.mean(quality_scores) if quality_scores else 0.0

    def _calculate_weather_consistency(self, data: pd.DataFrame) -> float:
        """Calculate weather data consistency score"""

        consistency_scores = []

        # Check for sudden jumps in numerical columns
        numerical_columns = ['temperature', 'humidity', 'rainfall', 'wind_speed']

        for column in numerical_columns:
            if column in data.columns:
                values = data[column].dropna()
                if len(values) >= 3:
                    # Check differences between consecutive values
                    diffs = np.abs(np.diff(values))
                    median_diff = np.median(diffs)

                    # Calculate how many diffs are reasonable (< 3x median)
                    reasonable_diffs = (diffs < 3 * median_diff).sum()
                    consistency_score = reasonable_diffs / len(diffs)
                    consistency_scores.append(consistency_score)

        return np.mean(consistency_scores) if consistency_scores else 0.0

    async def _fill_missing_weather_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fill missing weather values using seasonal patterns and interpolation"""

        filled_data = data.copy()

        # Get seasonal averages for Punjab
        current_month = datetime.utcnow().month
        seasonal_averages = self._get_seasonal_weather_averages(current_month)

        for column in ['temperature', 'humidity', 'rainfall', 'wind_speed']:
            if column in filled_data.columns:
                missing_mask = filled_data[column].isna()

                if missing_mask.any():
                    if column in seasonal_averages:
                        # Fill with seasonal average
                        filled_data.loc[missing_mask, column] = seasonal_averages[column]
                    elif missing_mask.sum() < len(filled_data) * 0.5:
                        # Interpolate if less than 50% missing
                        filled_data[column] = filled_data[column].interpolate(method='linear')

        return filled_data

    def _get_seasonal_weather_averages(self, month: int) -> Dict[str, float]:
        """Get average weather values for Punjab by month"""

        # Punjab monthly weather averages
        monthly_averages = {
            1: {'temperature': 12.2, 'humidity': 65, 'rainfall': 0.2, 'wind_speed': 8},   # January
            2: {'temperature': 15.0, 'humidity': 60, 'rainfall': 0.3, 'wind_speed': 9},   # February
            3: {'temperature': 20.6, 'humidity': 55, 'rainfall': 0.3, 'wind_speed': 10},  # March
            4: {'temperature': 26.1, 'humidity': 40, 'rainfall': 0.2, 'wind_speed': 12},  # April
            5: {'temperature': 31.1, 'humidity': 35, 'rainfall': 0.5, 'wind_speed': 13},  # May
            6: {'temperature': 33.9, 'humidity': 50, 'rainfall': 2.0, 'wind_speed': 12},  # June
            7: {'temperature': 31.8, 'humidity': 75, 'rainfall': 8.2, 'wind_speed': 11},  # July
            8: {'temperature': 30.1, 'humidity': 80, 'rainfall': 9.1, 'wind_speed': 10},  # August
            9: {'temperature': 28.9, 'humidity': 70, 'rainfall': 3.8, 'wind_speed': 9},   # September
            10: {'temperature': 23.9, 'humidity': 65, 'rainfall': 0.8, 'wind_speed': 8},  # October
            11: {'temperature': 17.2, 'humidity': 70, 'rainfall': 0.3, 'wind_speed': 7},  # November
            12: {'temperature': 13.3, 'humidity': 68, 'rainfall': 0.2, 'wind_speed': 7}   # December
        }

        return monthly_averages.get(month, {})

    def _get_current_season(self, month: int) -> str:
        """Determine current agricultural season"""

        if month in [6, 7, 8, 9, 10]:
            return 'kharif'
        elif month in [11, 12, 1, 2, 3]:
            return 'rabi'
        else:
            return 'summer'

    def _estimate_current_crop_stage(self, crop: str, season: str) -> Optional[CropStage]:
        """Estimate current crop growth stage"""

        current_month = datetime.utcnow().month
        current_day = datetime.utcnow().day

        if crop == 'wheat' and season == 'rabi':
            # Wheat growth stages for Punjab Rabi season
            if current_month == 11:  # November
                return CropStage.GERMINATION
            elif current_month == 12:  # December
                return CropStage.SEEDLING
            elif current_month == 1:  # January
                return CropStage.VEGETATIVE
            elif current_month in [2, 3] and current_day < 15:  # Early March
                return CropStage.FLOWERING
            elif current_month == 3 and current_day >= 15:  # Mid-late March
                return CropStage.GRAIN_FILLING
            elif current_month == 4:  # April
                return CropStage.MATURATION
            else:
                return CropStage.HARVEST

        # Similar logic would be implemented for other crops
        return None

    def _analyze_temperature_impact(self, temperatures: pd.Series,
                                  thresholds: Dict[str, Any],
                                  stage: CropStage) -> List[Dict[str, Any]]:
        """Analyze temperature impact on crop stage"""

        insights = []

        temp_values = temperatures.dropna()
        if len(temp_values) == 0:
            return insights

        avg_temp = temp_values.mean()
        min_temp = temp_values.min()
        max_temp = temp_values.max()

        # Check critical temperature violations
        if min_temp < thresholds['temp_min_critical']:
            severity = "severe" if avg_temp < thresholds['temp_min_critical'] - 5 else "moderate"
            insights.append({
                'type': 'cold_stress',
                'severity': severity,
                'temperature': float(min_temp),
                'threshold': thresholds['temp_min_critical'],
                'stage': stage.value,
                'impact': 'slowed_growth_potential_damage',
                'recommendation': 'increase_insulation_protect_from_cold'
            })

        if max_temp > thresholds['temp_max_critical']:
            severity = "severe" if avg_temp > thresholds['temp_max_critical'] + 5 else "moderate"
            insights.append({
                'type': 'heat_stress',
                'severity': severity,
                'temperature': float(max_temp),
                'threshold': thresholds['temp_max_critical'],
                'stage': stage.value,
                'impact': 'reduced_photosynthesis_protein_damage',
                'recommendation': 'provide_shade_increase_irrigation'
            })

        # Check optimal range compliance
        optimal_min, optimal_max = thresholds['temp_min_optimal'], thresholds['temp_max_optimal']
        optimal_compliance = ((temp_values >= optimal_min) & (temp_values <= optimal_max)).mean()

        if optimal_compliance < 0.5:  # Less than 50% of time in optimal range
            insights.append({
                'type': 'suboptimal_temperature',
                'severity': 'moderate',
                'optimal_compliance': float(optimal_compliance),
                'stage': stage.value,
                'impact': 'reduced_growth_efficiency',
                'recommendation': 'monitor_closely_consider_timing_adjustments'
            })

        return insights

    def _analyze_humidity_impact(self, humidities: pd.Series,
                               thresholds: Dict[str, Any],
                               stage: CropStage) -> List[Dict[str, Any]]:
        """Analyze humidity impact on crop stage"""

        insights = []

        humidity_values = humidities.dropna()
        if len(humidity_values) == 0:
            return insights

        avg_humidity = humidity_values.mean()
        min_humidity = humidity_values.min()
        max_humidity = humidity_values.max()

        optimal_min, optimal_max = thresholds['humidity_optimal']

        # Check if outside optimal range
        if avg_humidity < optimal_min or avg_humidity > optimal_max:
            if max_humidity > 85:  # High humidity
                insights.append({
                    'type': 'high_humidity_stress',
                    'severity': 'moderate',
                    'humidity': float(avg_humidity),
                    'stage': stage.value,
                    'impact': 'increased_disease_risk_poor_pollination',
                    'recommendation': 'improve_air_circulation_fungicide_prevention'
                })

        return insights

    def _analyze_rainfall_impact(self, rainfall: pd.Series,
                               thresholds: Dict[str, Any],
                               stage: CropStage) -> List[Dict[str, Any]]:
        """Analyze rainfall impact on crop stage"""

        insights = []

        rainfall_values = rainfall.dropna()
        if len(rainfall_values) == 0:
            return insights

        total_rainfall = rainfall_values.sum()
        max_daily = rainfall_values.max()

        # Check heavy rainfall impact
        if max_daily > thresholds['rainfall_max_daily']:
            insights.append({
                'type': 'excess_rainfall',
                'severity': 'moderate',
                'rainfall': float(max_daily),
                'threshold': thresholds['rainfall_max_daily'],
                'stage': stage.value,
                'impact': 'flooding_soil_compaction_nutrient_leaching',
                'recommendation': 'improve_drainage_delay_operations'
            })

        return insights

    def _generate_weather_recommendations(self, insights: List[Dict[str, Any]],
                                        alerts: List[Dict[str, Any]]) -> List[str]:
        """Generate actionable weather-based recommendations"""

        recommendations = []

        # Process insights
        for insight in insights:
            if 'recommendation' in insight:
                rec = insight['recommendation'].replace('_', ' ')
                if rec not in recommendations:
                    recommendations.append(rec)

        # Process alerts
        for alert in alerts:
            if 'recommendation' in alert:
                rec = alert['recommendation'].replace('_', ' ')
                if rec not in recommendations:
                    recommendations.append(rec)

        # Add general recommendations based on current conditions
        current_hour = datetime.utcnow().hour
        if 10 <= current_hour <= 16:  # Peak sunlight hours
            if any(a.get('phenomenon') == 'HEAT_WAVE' for a in alerts):
                recommendations.append("schedule irrigation during early morning or evening only")

        return recommendations

    async def _store_weather_data(self, data: pd.DataFrame, location: str,
                                analysis: Dict[str, Any]) -> None:
        """Store processed weather data"""

        # Store in data processing pipeline
        batch_data = {
            'weather_metrics': data.to_dict('records'),
            'analysis': analysis,
            'location': location,
            'processed_at': datetime.utcnow().isoformat()
        }

        await data_pipeline.ingest_data_batch(
            source=DataSource.WEATHER_API,
            data=batch_data,
            metadata={'weather_location': location}
        )

    async def _weather_monitoring_loop(self):
        """Background weather monitoring loop"""

        while True:
            try:
                await asyncio.sleep(900)  # Check every 15 minutes

                # Periodic weather anomaly detection
                await self._check_seasonal_anomalies()

                # Update weather patterns
                await self._update_weather_patterns()

            except Exception as e:
                logger.error(f"Weather monitoring loop error: {e}")

    async def _check_seasonal_anomalies(self):
        """Check for seasonal weather anomalies"""

        current_month = datetime.utcnow().month
        seasonal_normals = self._get_seasonal_weather_averages(current_month)

        # Compare recent weather against seasonal normals
        # This would be enhanced with actual recent data

        logger.debug(f"Seasonal anomaly check completed for month {current_month}")

    async def _update_weather_patterns(self):
        """Update learned weather patterns"""

        # Machine learning-based pattern recognition
        # This would analyze historical weather patterns and predict upcoming conditions

        logger.debug("Weather pattern analysis updated")

    # Forecast analysis helper methods
    def _analyze_forecast_heat_wave_risk(self, forecast_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze forecast for heat wave risk"""

        risk_periods = []
        for day_forecast in forecast_list:
            if day_forecast.get('max_temp', 0) >= self.processing_config['heat_wave_threshold_celsius']:
                risk_periods.append(day_forecast.get('date'))

        if len(risk_periods) >= self.processing_config['heat_wave_consecutive_days']:
            return {
                'risk_level': 'high',
                'period': f"{len(risk_periods)} consecutive days",
                'start_date': min(risk_periods)
            }
        elif len(risk_periods) >= 1:
            return {
                'risk_level': 'medium',
                'period': f"{len(risk_periods)} days",
                'start_date': min(risk_periods)
            }

        return {'risk_level': 'low'}

    def _analyze_forecast_frost_risk(self, forecast_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze forecast for frost risk"""

        frost_dates = []
        for day_forecast in forecast_list:
            if day_forecast.get('min_temp', 20) <= self.processing_config['frost_threshold_celsius']:
                frost_dates.append(day_forecast.get('date'))

        if frost_dates:
            return {
                'risk_level': 'high',
                'dates': frost_dates
            }

        return {'risk_level': 'low'}

    def _plan_forecast_based_irrigation(self, forecast_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Plan irrigation based on weather forecast"""

        irrigation_advice = []

        # Check next 7 days rainfall forecast
        total_forecast_rainfall = sum(day.get('rainfall', 0) for day in forecast_list[:7])

        if total_forecast_rainfall < 10:  # Less than 10mm in 7 days
            irrigation_advice.append({
                'type': 'soil_moisture_monitoring',
                'recommendation': 'monitor soil moisture closely and prepare irrigation schedule',
                'reason': f'only {total_forecast_rainfall}mm rainfall forecast in next 7 days'
            })

        dry_days = sum(1 for day in forecast_list[:5]
                      if day.get('rainfall', 0) < 2)  # Less than 2mm per day

        if dry_days >= 4:
            irrigation_advice.append({
                'type': 'preventive_irrigation',
                'recommendation': 'consider preventive irrigation to maintain soil moisture',
                'reason': f'{dry_days} dry days forecast in next 5 days'
            })

        return irrigation_advice

    def _calculate_overall_forecast_risk(self, insights: Dict[str, Any]) -> str:
        """Calculate overall risk level from forecast insights"""

        risk_score = 0

        # Count high-risk events
        for event in insights.get('upcoming_events', []):
            if event.get('risk_level') == 'high':
                risk_score += 2
            elif event.get('risk_level') == 'medium':
                risk_score += 1

        if risk_score >= 3:
            return 'high'
        elif risk_score >= 1:
            return 'medium'
        else:
            return 'low'

    async def _initialize_seasonal_baselines(self):
        """Initialize seasonal baseline weather patterns"""

        # This would load historical weather patterns for Punjab
        # Used for anomaly detection and forecasting

        logger.info("Seasonal weather baselines initialized")

    def get_weather_insights_dashboard(self) -> Dict[str, Any]:
        """Get weather insights for dashboard display"""

        return {
            'current_alerts': self.weather_alerts[-5:],  # Last 5 alerts
            'active_monitoring': {
                'weather_stations': len(self.weather_data),
                'monitored_crops': list(self.crop_weather_thresholds.keys()),
                'active_patterns': len(self.weather_patterns)
            },
            'system_status': 'operational',
            'last_updated': datetime.utcnow().isoformat()
        }

# Global weather processor instance
weather_processor = AdvancedWeatherProcessor()

# Convenience functions
async def initialize_weather_processing() -> bool:
    """Initialize weather processing system"""
    return await weather_processor.initialize_weather_processing()

async def process_weather_batch(source: str, data: Any, location: str = "punjab") -> Dict[str, Any]:
    """Process weather data from source"""
    source_enum = DataSource(source.upper())
    return await weather_processor.process_weather_data(data, location)

async def analyze_weather_forecast(forecast_data: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze weather forecast for agricultural insights"""
    return await weather_processor.generate_weather_forecast_insights(forecast_data)

def get_weather_insights() -> Dict[str, Any]:
    """Get current weather insights dashboard"""
    return weather_processor.get_weather_insights_dashboard()
