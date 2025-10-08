"""
India Agricultural Intelligence Platform
Main orchestrator for multi-crop, multi-state yield prediction
"""

from .config import config_manager
from .utils.model_registry import model_registry
from .utils.crop_config import create_crop_config, get_available_crops
from .utils.state_config import create_state_config, get_available_states
from typing import Dict, Any, Optional, List, Tuple
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class IndiaAgriPlatform:
    """Main orchestrator for the India Agricultural Intelligence Platform"""

    def __init__(self):
        self.config = config_manager
        self.model_registry = model_registry
        self.crop_configs = {}
        self.state_configs = {}
        logger.info("India Agricultural Intelligence Platform initialized")

    def get_available_crops(self) -> List[str]:
        """Get list of available crops"""
        return get_available_crops()

    def get_available_states(self) -> List[str]:
        """Get list of available states"""
        return get_available_states()

    def get_crop_config(self, crop_name: str):
        """Get crop configuration"""
        if crop_name not in self.crop_configs:
            self.crop_configs[crop_name] = create_crop_config(crop_name)
        return self.crop_configs[crop_name]

    def get_state_config(self, state_name: str):
        """Get state configuration"""
        if state_name not in self.state_configs:
            self.state_configs[state_name] = create_state_config(state_name)
        return self.state_configs[state_name]

    def predict_yield(self, crop_name: str, state_name: str,
                     features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict crop yield for given parameters

        Args:
            crop_name: Name of the crop (e.g., 'wheat', 'rice')
            state_name: Name of the state (e.g., 'punjab', 'haryana')
            features: Dictionary of input features

        Returns:
            Dictionary with prediction results
        """
        try:
            # Get model for crop/state combination
            model = self.model_registry.get_model_for_crop_state(crop_name, state_name)
            if not model:
                return {
                    "error": f"No model found for crop '{crop_name}' in state '{state_name}'",
                    "available_crops": self.get_available_crops(),
                    "available_states": self.get_available_states()
                }

            # Prepare features for prediction
            processed_features = self._prepare_features(crop_name, state_name, features)

            # Make prediction
            if hasattr(model, 'predict'):
                prediction = model.predict(processed_features)
                yield_value = float(prediction[0]) if hasattr(prediction, '__len__') else float(prediction)
            else:
                return {"error": "Invalid model format"}

            # Get confidence interval (simplified)
            confidence_range = self._calculate_confidence_interval(yield_value, model)

            # Get additional insights
            insights = self._generate_insights(crop_name, state_name, features, yield_value)

            return {
                "crop": crop_name,
                "state": state_name,
                "predicted_yield_quintal_ha": round(yield_value, 2),
                "confidence_interval": confidence_range,
                "unit": "quintal per hectare",
                "insights": insights,
                "model_version": model.get('metadata', {}).get('version', 'unknown'),
                "prediction_timestamp": pd.Timestamp.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {"error": f"Prediction failed: {str(e)}"}

    def _prepare_features(self, crop_name: str, state_name: str,
                         features: Dict[str, Any]) -> pd.DataFrame:
        """Prepare and validate features for prediction"""
        # Get crop and state configurations
        crop_config = self.get_crop_config(crop_name)
        state_config = self.get_state_config(state_name)

        # Validate required features
        required_features = [
            'temperature_celsius', 'rainfall_mm', 'humidity_percent',
            'ndvi', 'soil_ph', 'irrigation_coverage'
        ]

        missing_features = [f for f in required_features if f not in features]
        if missing_features:
            # Try to provide defaults or estimates
            features = self._fill_missing_features(features, crop_name, state_name, missing_features)

        # Add crop-specific features
        features['crop_water_requirement'] = crop_config.get_water_requirement()
        features['disease_risk_rust'] = crop_config.get_disease_risk('rust')
        features['variety_score'] = features.get('variety_score', 0.8)  # Default good variety

        # Add state-specific features
        irrigation_info = state_config.get_irrigation_info()
        features['irrigation_efficiency'] = irrigation_info.get('efficiency', 0.65)

        # Convert to DataFrame
        df = pd.DataFrame([features])

        # Apply feature engineering
        df = self._apply_feature_engineering(df, crop_name, state_name)

        return df

    def _fill_missing_features(self, features: Dict[str, Any], crop_name: str,
                             state_name: str, missing_features: List[str]) -> Dict[str, Any]:
        """Fill missing features with reasonable defaults"""
        filled_features = features.copy()

        # Get state climate info for defaults
        state_config = self.get_state_config(state_name)
        climate_info = state_config.get_climate_info()

        defaults = {
            'temperature_celsius': climate_info.get('temperature_range_c', [25, 35])[0] + 10,  # Mid-range
            'rainfall_mm': climate_info.get('annual_rainfall_mm', 800) / 12,  # Monthly average
            'humidity_percent': climate_info.get('humidity_percent', 60),
            'ndvi': 0.6,  # Moderate vegetation
            'soil_ph': 7.0,  # Neutral pH
            'irrigation_coverage': state_config.get_irrigation_info().get('coverage_percent', 70) / 100
        }

        for feature in missing_features:
            if feature in defaults:
                filled_features[feature] = defaults[feature]
                logger.info(f"Filled missing feature '{feature}' with default value: {defaults[feature]}")

        return filled_features

    def _apply_feature_engineering(self, df: pd.DataFrame,
                                 crop_name: str, state_name: str) -> pd.DataFrame:
        """Apply feature engineering transformations"""
        # Temperature-humidity index
        df['temp_humidity_index'] = df['temperature_celsius'] * df['humidity_percent'] / 100

        # Irrigation effectiveness
        df['effective_irrigation'] = df['irrigation_coverage'] * df['irrigation_efficiency']

        # Soil fertility index (simplified)
        df['soil_fertility_index'] = (df['soil_ph'] - 6.5).abs() * -0.1 + 1.0  # Closer to 6.5 is better

        # Climate stress factor
        optimal_temp = 25  # General optimal temperature
        df['temperature_stress'] = ((df['temperature_celsius'] - optimal_temp) / 10).abs()

        return df

    def _calculate_confidence_interval(self, prediction: float, model: Dict[str, Any]) -> str:
        """Calculate prediction confidence interval"""
        # Simplified confidence interval based on model performance
        model_performance = model.get('performance', {})
        mae = model_performance.get('mae', 2.0)  # Default MAE

        lower_bound = max(0, prediction - (mae * 1.96))  # 95% confidence
        upper_bound = prediction + (mae * 1.96)

        return f"{lower_bound:.1f} - {upper_bound:.1f}"

    def _generate_insights(self, crop_name: str, state_name: str,
                          features: Dict[str, Any], prediction: float) -> Dict[str, Any]:
        """Generate actionable insights from prediction"""
        insights = {}

        # Get configurations
        crop_config = self.get_crop_config(crop_name)
        state_config = self.get_state_config(state_name)

        # Yield comparison with historical average
        historical_avg = crop_config.config.get('yield_parameters', {}).get('max_yield_quintal_ha', 40)
        yield_ratio = prediction / historical_avg

        if yield_ratio > 1.1:
            insights['yield_status'] = "Above average - Excellent growing conditions"
        elif yield_ratio > 0.9:
            insights['yield_status'] = "Average yield - Standard conditions"
        else:
            insights['yield_status'] = "Below average - Check stress factors"

        # Irrigation recommendations
        irrigation_coverage = features.get('irrigation_coverage', 0)
        if irrigation_coverage < 0.7:
            insights['irrigation'] = "Consider improving irrigation coverage"
        else:
            insights['irrigation'] = "Irrigation coverage is adequate"

        # Disease risk assessment
        disease_risk = crop_config.get_disease_risk('rust')
        humidity = features.get('humidity_percent', 60)
        if humidity > 70 and disease_risk > 0.5:
            insights['disease'] = "High disease risk - Monitor for rust infections"

        # Climate suitability
        temperature = features.get('temperature_celsius', 25)
        climate_score = state_config.get_climate_suitability_score(temperature, features.get('rainfall_mm', 800))
        if climate_score < 0.7:
            insights['climate'] = "Climate conditions may stress crop growth"

        return insights

    def get_crop_recommendations(self, state_name: str, conditions: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get crop recommendations for given conditions"""
        recommendations = []

        available_crops = self.get_available_crops()
        state_config = self.get_state_config(state_name)

        for crop in available_crops:
            if state_config.is_crop_suitable(crop):
                crop_config = self.get_crop_config(crop)

                # Calculate suitability scores
                climate_score = crop_config.get_climate_suitability(
                    conditions.get('temperature', 25),
                    conditions.get('humidity', 60),
                    conditions.get('rainfall', 800)
                )

                soil_score = crop_config.get_soil_suitability(
                    conditions.get('ph', 7.0),
                    conditions.get('soil_texture', 'loam'),
                    conditions.get('organic_matter', 1.0)
                )

                overall_score = (climate_score * 0.6 + soil_score * 0.4)

                if overall_score > 0.6:  # Only recommend suitable crops
                    recommendations.append({
                        "crop": crop,
                        "suitability_score": round(overall_score, 2),
                        "expected_yield_range": f"{crop_config.config.get('yield_parameters', {}).get('max_yield_quintal_ha', 40) * 0.7:.0f}-{crop_config.config.get('yield_parameters', {}).get('max_yield_quintal_ha', 40):.0f}",
                        "season": crop_config.get_season_info()['season'],
                        "risk_factors": self._assess_risks(crop, conditions)
                    })

        # Sort by suitability score
        recommendations.sort(key=lambda x: x['suitability_score'], reverse=True)
        return recommendations[:5]  # Top 5 recommendations

    def _assess_risks(self, crop_name: str, conditions: Dict[str, Any]) -> List[str]:
        """Assess risks for crop under given conditions"""
        risks = []
        crop_config = self.get_crop_config(crop_name)

        # Temperature risk
        temp = conditions.get('temperature', 25)
        if temp > 35 or temp < 10:
            risks.append("Temperature stress")

        # Humidity/disease risk
        humidity = conditions.get('humidity', 60)
        if humidity > 80:
            risks.append("High disease pressure")

        # Drought risk
        rainfall = conditions.get('rainfall', 800)
        if rainfall < 500:
            risks.append("Drought risk")

        # Soil pH risk
        ph = conditions.get('ph', 7.0)
        if ph < 5.5 or ph > 8.5:
            risks.append("Soil pH stress")

        return risks if risks else ["Low risk"]

    def get_model_performance_summary(self) -> Dict[str, Any]:
        """Get summary of all available models"""
        return self.model_registry.list_available_models()

    def validate_crop_state_compatibility(self, crop_name: str, state_name: str) -> Dict[str, Any]:
        """Validate if crop is suitable for state"""
        state_config = self.get_state_config(state_name)

        is_suitable = state_config.is_crop_suitable(crop_name)
        major_crops = state_config.get_major_crops()

        return {
            "compatible": is_suitable,
            "state_major_crops": major_crops,
            "recommendation": "Suitable" if is_suitable else "Not typically grown - check local conditions"
        }

# Global platform instance
platform = IndiaAgriPlatform()
