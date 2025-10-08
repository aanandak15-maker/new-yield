"""
Rice Yield Predictor for India Agricultural Intelligence Platform
Multi-state rice yield modeling with variety-specific predictions
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging
import pickle
from datetime import datetime

# Add platform imports
sys.path.append('.')
from india_agri_platform.core.error_handling import error_handler, ModelError
from india_agri_platform.core.cache_manager import cache_manager
from india_agri_platform import platform
import rice_varieties

logger = logging.getLogger(__name__)

class RiceYieldPredictor:
    """Multi-state rice yield predictor with variety-specific modeling"""

    def __init__(self, model_dir="rice_models"):
        self.model_dir = Path("india_agri_platform/models") / model_dir
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # State-specific models
        self.state_models = {}

        # Rice variety manager
        self.rice_variety_manager = rice_varieties.rice_variety_manager

        # Load models
        self._load_or_create_models()

    def _load_or_create_models(self):
        """Load existing models or create new ones if needed"""

        # Major rice states
        rice_states = [
            'PUNJAB', 'HARYANA', 'UTTAR PRADESH', 'BIHAR',
            'WEST BENGAL', 'ANDHRA PRADESH', 'TAMIL NADU',
            'ODISHA', 'KARNATAKA', 'ASSAM'
        ]

        for state in rice_states:
            model_path = self.model_dir / f"rice_model_{state.lower()}.pkl"
            if model_path.exists():
                try:
                    with open(model_path, 'rb') as f:
                        self.state_models[state] = pickle.load(f)
                    logger.info(f"Loaded existing rice model for {state}")
                except Exception as e:
                    logger.error(f"Failed to load {state} model: {e}")
                    self.state_models[state] = self._create_baseline_model()
            else:
                self.state_models[state] = self._create_baseline_model()
                logger.info(f"Created baseline rice model for {state}")

    def _create_baseline_model(self):
        """Create a baseline Random Forest model for rice prediction"""
        try:
            from sklearn.ensemble import RandomForestRegressor
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=15,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            )
            return model
        except ImportError:
            logger.warning("scikit-learn not available, using fallback")
            return None

    def train_state_model(self, state: str, training_data: pd.DataFrame) -> bool:
        """Train model for a specific state"""

        try:
            logger.info(f"Training rice model for {state} with {len(training_data)} records")

            # Prepare features and target
            features = [
                'yield_quintal_ha', 'Area', 'irrigation_coverage',
                'temperature_celsius', 'rainfall_mm', 'humidity_percent',
                'soil_ph', 'ndvi'
            ]

            # Filter available features
            available_features = [f for f in features if f in training_data.columns]
            if 'yield_quintal_ha' not in training_data.columns:
                logger.error("Training data missing yield_quintal_ha column")
                return False

            X = training_data[available_features].copy()
            y = training_data['yield_quintal_ha']

            # Handle missing values
            X = X.fillna({
                'irrigation_coverage': 0.5,
                'temperature_celsius': 25,
                'rainfall_mm': 800,
                'humidity_percent': 65,
                'soil_ph': 7.0,
                'ndvi': 0.6
            })

            # Create and train model
            model = self._create_baseline_model()
            if model is None:
                return False

            model.fit(X.drop('yield_quintal_ha', axis=1), y)

            # Save model
            model_path = self.model_dir / f"rice_model_{state.lower()}.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)

            self.state_models[state] = model
            logger.info(f"Successfully trained rice model for {state}")

            return True

        except Exception as e:
            error_handler.handle_error(e, {
                "operation": "model_training",
                "state": state,
                "training_records": len(training_data) if 'training_data' in locals() else 0
            })
            return False

    def predict_rice_yield(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict rice yield for given features

        Args:
            features: Dictionary containing prediction features
                Required: sowing_date, latitude, longitude
                Optional: variety_name, temperature, rainfall, etc.
        """

        try:
            # Extract required parameters
            sowing_date = features.get('sowing_date')
            latitude = features.get('latitude')
            longitude = features.get('longitude')

            if not all([sowing_date, latitude, longitude]):
                raise ValueError("Missing required parameters: sowing_date, latitude, longitude")

            # Determine state from coordinates
            state = self._determine_state_from_coordinates(latitude, longitude)
            if not state:
                return {"error": "Unable to determine state from coordinates"}

            # Get variety information
            variety_name = features.get('variety_name')
            if variety_name:
                variety_info = self.rice_variety_manager.get_variety_info(variety_name)
            else:
                # Select best variety for state and conditions
                conditions = self._extract_conditions_from_features(features, state)
                recommendations = self.rice_variety_manager.recommend_varieties(conditions, top_n=1)
                variety_name = recommendations[0]['variety'] if recommendations else 'PR121'
                variety_info = self.rice_variety_manager.get_variety_info(variety_name)

            # Prepare features for prediction
            processed_features = self._prepare_rice_features(
                features, state, variety_name, variety_info
            )

            # Make prediction using state-specific model
            model = self.state_models.get(state)
            if model is None:
                return {"error": f"No trained model available for state: {state}"}

            prediction = self._make_rice_prediction(model, processed_features, variety_info)

            # Generate insights and recommendations
            insights = self._generate_rice_insights(prediction, features, state, variety_info)

            return {
                "crop": "rice",
                "variety": variety_name,
                "state": state,
                "predicted_yield_quintal_ha": prediction,
                "unit": "quintal per hectare",
                "confidence_level": "medium",  # Could be improved with validation data
                "insights": insights,
                "timestamp": datetime.utcnow().isoformat(),
                "prediction_method": "multi_state_rice_model"
            }

        except Exception as e:
            error_handler.handle_error(e, {
                "operation": "rice_prediction",
                "features_provided": list(features.keys()),
                "error_details": str(e)
            })
            return {"error": f"Rice prediction failed: {str(e)}"}

    def _determine_state_from_coordinates(self, lat: float, lng: float) -> Optional[str]:
        """Determine Indian state from latitude/longitude coordinates"""

        # Simplified state boundaries for major rice-growing regions
        state_boundaries = {
            'PUNJAB': {'lat_min': 29.5, 'lat_max': 32.5, 'lng_min': 73.8, 'lng_max': 76.9},
            'HARYANA': {'lat_min': 27.5, 'lat_max': 30.9, 'lng_min': 74.4, 'lng_max': 77.6},
            'UTTAR PRADESH': {'lat_min': 23.8, 'lat_max': 30.4, 'lng_min': 77.0, 'lng_max': 84.6},
            'BIHAR': {'lat_min': 24.2, 'lat_max': 27.3, 'lng_min': 83.0, 'lng_max': 88.2},
            'WEST BENGAL': {'lat_min': 21.5, 'lat_max': 27.3, 'lng_min': 85.8, 'lng_max': 89.1},
            'ANDHRA PRADESH': {'lat_min': 12.9, 'lat_max': 19.1, 'lng_min': 76.7, 'lng_max': 84.8},
            'TAMIL NADU': {'lat_min': 8.1, 'lat_max': 13.1, 'lng_min': 76.2, 'lng_max': 80.4},
            'ODISHA': {'lat_min': 17.8, 'lat_max': 22.1, 'lng_min': 81.4, 'lng_max': 87.6},
            'KARNATAKA': {'lat_min': 11.5, 'lat_max': 18.5, 'lng_min': 74.0, 'lng_max': 78.6},
            'ASSAM': {'lat_min': 24.0, 'lat_max': 28.0, 'lng_min': 89.7, 'lng_max': 96.1}
        }

        for state, bounds in state_boundaries.items():
            if (bounds['lat_min'] <= lat <= bounds['lat_max'] and
                bounds['lng_min'] <= lng <= bounds['lng_max']):
                return state

        return None

    def _extract_conditions_from_features(self, features: Dict[str, Any],
                                        state: str) -> Dict[str, Any]:
        """Extract growing conditions from feature dictionary"""

        return {
            'irrigation_coverage': features.get('irrigation_coverage', 0.5),
            'temperature_celsius': features.get('temperature_celsius', 25),
            'disease_pressure': features.get('disease_risk', 0.5),
            'state': state,
            'season': features.get('season', 'kharif')
        }

    def _prepare_rice_features(self, features: Dict[str, Any], state: str,
                              variety: str, variety_info: Dict[str, Any]) -> pd.DataFrame:
        """Prepare feature vector for rice yield prediction"""

        # Base features (similar to streamlined predictor)
        feature_dict = {
            'temperature_celsius': features.get('temperature_celsius', 25),
            'rainfall_mm': features.get('rainfall_mm', 800),
            'humidity_percent': features.get('humidity_percent', 65),
            'ndvi': features.get('ndvi', 0.6),
            'soil_ph': features.get('soil_ph', 7.0),
            'irrigation_coverage': features.get('irrigation_coverage', 0.5),
            'area_hectares': features.get('area_hectares', 1.0)
        }

        # State-specific adjustments
        state_adjustments = self._get_state_adjustments(state)
        for key, multiplier in state_adjustments.items():
            if key in feature_dict:
                feature_dict[key] *= multiplier

        # Variety-specific adjustments
        if variety_info:
            variety_multipliers = self._get_variety_adjustments(variety_info)
            for key, multiplier in variety_multipliers.items():
                if key in feature_dict:
                    feature_dict[key] *= multiplier

        # Add variety-derived features
        feature_dict['variety_yield_potential'] = variety_info.get('yield_potential_quintal_ha', 50) / 100
        feature_dict['variety_maturity_factor'] = (variety_info.get('maturity_days', 120) - 100) / 50  # Normalize
        feature_dict['variety_water_requirement'] = variety_info.get('water_requirement_mm', 1200) / 1500

        return pd.DataFrame([feature_dict])

    def _get_state_adjustments(self, state: str) -> Dict[str, float]:
        """Get state-specific feature adjustments"""

        adjustments = {
            'PUNJAB': {'irrigation_coverage': 1.2, 'temperature_celsius': 1.05},
            'HARYANA': {'irrigation_coverage': 1.1, 'temperature_celsius': 1.02},
            'UTTAR PRADESH': {'irrigation_coverage': 0.8, 'rainfall_mm': 1.1},
            'BIHAR': {'irrigation_coverage': 0.6, 'rainfall_mm': 1.2, 'humidity_percent': 1.1},
            'WEST BENGAL': {'irrigation_coverage': 0.7, 'rainfall_mm': 1.3, 'humidity_percent': 1.15},
            'ANDHRA PRADESH': {'irrigation_coverage': 0.9, 'temperature_celsius': 1.1},
            'TAMIL NADU': {'irrigation_coverage': 0.85, 'humidity_percent': 0.9},
            'ODISHA': {'irrigation_coverage': 0.65, 'rainfall_mm': 1.25},
            'KARNATAKA': {'irrigation_coverage': 0.4, 'rainfall_mm': 0.9},
            'ASSAM': {'irrigation_coverage': 0.3, 'rainfall_mm': 1.4, 'humidity_percent': 1.2}
        }

        return adjustments.get(state, {})

    def _get_variety_adjustments(self, variety_info: Dict[str, Any]) -> Dict[str, float]:
        """Get variety-specific feature adjustments"""

        adjustments = {}

        # Water requirement adjustment
        water_req = variety_info.get('water_requirement_mm', 1200)
        if water_req > 1300:
            adjustments['irrigation_coverage'] = 0.9  # Needs more water
        elif water_req < 1100:
            adjustments['irrigation_coverage'] = 1.1  # More water-efficient

        # Disease resistance adjustment
        blast_resistance = variety_info.get('disease_resistance', {}).get('blast', 'moderate')
        if blast_resistance == 'high':
            adjustments['irrigation_coverage'] = adjustments.get('irrigation_coverage', 1.0) * 1.05  # Can handle more water
        elif blast_resistance == 'low':
            adjustments['irrigation_coverage'] = adjustments.get('irrigation_coverage', 1.0) * 0.95  # More careful with water

        return adjustments

    def _make_rice_prediction(self, model, features: pd.DataFrame,
                            variety_info: Dict[str, Any]) -> float:
        """Make prediction using trained model and variety data"""

        try:
            # Model prediction
            base_prediction = model.predict(features)[0]

            # Apply variety-specific yield potential
            variety_yield_potential = variety_info.get('yield_potential_quintal_ha', 50)

            # Blend model prediction with variety potential
            # Give 70% weight to model, 30% to variety potential
            final_prediction = (base_prediction * 0.7) + (variety_yield_potential * 0.3)

            return round(final_prediction, 2)

        except Exception as e:
            logger.warning(f"Model prediction failed, using variety potential: {e}")
            return variety_info.get('yield_potential_quintal_ha', 35)

    def _generate_rice_insights(self, prediction: float, features: Dict[str, Any],
                              state: str, variety_info: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive rice cultivation insights"""

        insights = {
            "yield_analysis": self._analyze_yield_level(prediction, state),
            "irrigation_recommendations": self._get_rice_irrigation_advice(features, variety_info),
            "disease_risk_assessment": self._assess_rice_disease_risks(features, variety_info),
            "variety_performance": self._evaluate_variety_performance(variety_info),
            "state_specific_advice": self._get_state_rice_advice(state, prediction)
        }

        return insights

    def _analyze_yield_level(self, prediction: float, state: str) -> str:
        """Analyze predicted yield level compared to state averages"""

        # State averages (based on historical data)
        state_averages = {
            'PUNJAB': 38, 'HARYANA': 36, 'UTTAR PRADESH': 22,
            'BIHAR': 16, 'WEST BENGAL': 25, 'ANDHRA PRADESH': 29,
            'TAMIL NADU': 37, 'ODISHA': 17, 'KARNATAKA': 26, 'ASSAM': 17
        }

        avg_yield = state_averages.get(state, 25)

        if prediction > avg_yield * 1.2:
            return ".1f"
        elif prediction > avg_yield * 0.9:
            return ".1f"
        else:
            return ".1f"

    def _get_rice_irrigation_advice(self, features: Dict[str, Any],
                                   variety_info: Dict[str, Any]) -> str:
        """Generate irrigation recommendations for rice"""

        irrigation = features.get('irrigation_coverage', 0.5)
        water_req = variety_info.get('water_requirement_mm', 1200)

        if irrigation < 0.7 and water_req > 1200:
            return "Increase irrigation - variety needs continuous flooding during critical growth stages"
        elif irrigation > 0.9:
            return "Monitor water levels - avoid waterlogging which can cause disease"
        else:
            return "Maintain current irrigation schedule - adequate water management observed"

    def _assess_rice_disease_risks(self, features: Dict[str, Any],
                                  variety_info: Dict[str, Any]) -> str:
        """Assess disease risks for rice cultivation"""

        humidity = features.get('humidity_percent', 65)
        temperature = features.get('temperature_celsius', 25)

        risks = []
        if humidity > 80:
            risks.append("high humidity increases fungal disease risk")
        if temperature > 30:
            risks.append("heat stress can trigger bacterial diseases")
        if variety_info.get('disease_resistance', {}).get('blast') == 'low':
            risks.append("variety has low blast resistance - monitor for early symptoms")

        if risks:
            return "; ".join(risks).capitalize()
        else:
            return "Disease pressure appears low under current conditions"

    def _evaluate_variety_performance(self, variety_info: Dict[str, Any]) -> str:
        """Evaluate variety performance characteristics"""

        variety_name = variety_info.get('name', 'Unknown')
        yield_potential = variety_info.get('yield_potential_quintal_ha', 0)
        maturity = variety_info.get('maturity_days', 120)

        if yield_potential >= 75:
            return f"{variety_name} is a high-yielding variety with excellent yield potential"
        elif yield_potential >= 60:
            return f"{variety_name} offers good yield potential with balanced characteristics"
        else:
            return f"{variety_name} is suitable for specific conditions with moderate yield potential"

    def _get_state_rice_advice(self, state: str, prediction: float) -> str:
        """Get state-specific rice cultivation advice"""

        advice = {
            'PUNJAB': "Focus on timely transplanting and efficient water management in canal-irrigated fields",
            'BIHAR': "Rice in Bihar benefits from improved seed quality and proper spacing during transplanting",
            'UTTAR PRADESH': "Diversified sowing windows help manage heat stress in eastern UP rice cultivation",
            'WEST BENGAL': "Coastal areas require salt-tolerant varieties and proper drainage management"
        }

        return advice.get(state, f"Standard rice cultivation practices recommended for {state}")

    def create_rice_model_report(self, state: str) -> Dict[str, Any]:
        """Create comprehensive model performance report"""

        # This would analyze training data and create detailed reports
        # For now, return basic info
        return {
            'state': state,
            'model_exists': state in self.state_models,
            'variety_count': len(self.rice_variety_manager.get_varieties_for_state(state)),
            'last_updated': datetime.utcnow().isoformat()
        }

# Global rice predictor instance
rice_predictor = RiceYieldPredictor()

def get_rice_predictor() -> RiceYieldPredictor:
    """Get global rice predictor instance"""
    return rice_predictor
