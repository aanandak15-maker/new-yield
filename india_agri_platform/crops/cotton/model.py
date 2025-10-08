"""
Cotton Yield Predictor for India Agricultural Intelligence Platform
Multi-state cotton yield modeling with variety-specific predictions
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
from india_agri_platform.crops.cotton.cotton_varieties import cotton_variety_manager

logger = logging.getLogger(__name__)

class CottonYieldPredictor:
    """Multi-state cotton yield predictor with variety-specific modeling"""

    def __init__(self, model_dir="cotton_models"):
        self.model_dir = Path("india_agri_platform/models") / model_dir
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # State-specific models
        self.state_models = {}

        # Cotton variety manager (already imported at top level)

        # Load models
        self._load_or_create_models()

    def _load_or_create_models(self):
        """Load existing models or create new ones if needed"""

        # Major cotton states
        cotton_states = [
            'PUNJAB', 'HARYANA', 'MAHARASHTRA', 'GUJARAT',
            'ANDHRA_PRADESH', 'KARNATKA', 'TAMIL_NADU'
        ]

        for state in cotton_states:
            model_path = self.model_dir / f"cotton_model_{state.lower()}.pkl"
            if model_path.exists():
                try:
                    with open(model_path, 'rb') as f:
                        self.state_models[state] = pickle.load(f)
                    logger.info(f"Loaded existing cotton model for {state}")
                except Exception as e:
                    logger.error(f"Failed to load {state} model: {e}")
                    self.state_models[state] = self._create_baseline_model()
            else:
                self.state_models[state] = self._create_baseline_model()
                logger.info(f"Created baseline cotton model for {state}")

    def _create_baseline_model(self):
        """Create a baseline Random Forest model for cotton prediction"""
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
            logger.info(f"Training cotton model for {state} with {len(training_data)} records")

            # Prepare features and target
            features = [
                'yield_quintal_ha', 'Area', 'bollworm_pressure',
                'temperature_stress', 'soil_suitability', 'irrigation_timing'
            ]

            # Filter available features
            available_features = [f for f in features if f in training_data.columns]
            if 'yield_quintal_ha' not in available_features:
                logger.error("Training data missing yield_quintal_ha column")
                return False

            X = training_data[available_features].copy()
            y = training_data['yield_quintal_ha']

            # Handle missing values
            X = X.fillna({
                'Area': X['Area'].median() if 'Area' in X.columns else 1.0,
                'bollworm_pressure': 0.6,
                'temperature_stress': 0.3,
                'soil_suitability': 0.8,
                'irrigation_timing': 0.5
            })

            # Create and train model
            model = self._create_baseline_model()
            if model is None:
                return False

            model.fit(X.drop('yield_quintal_ha', axis=1), y)

            # Save model
            model_path = self.model_dir / f"cotton_model_{state.lower()}.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)

            self.state_models[state] = model
            logger.info(f"Successfully trained cotton model for {state}")

            return True

        except Exception as e:
            error_handler.handle_error(e, {
                "operation": "cotton_model_training",
                "state": state,
                "training_records": len(training_data) if 'training_data' in locals() else 0
            })
            return False

    def predict_cotton_yield(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict cotton yield for given features

        Args:
            features: Dictionary containing prediction features
                Required: sowing_date, latitude, longitude
                Optional: variety_name, temperature, rainfall, pest_pressure, etc.
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

            # Validate cotton-growing state
            cotton_states = ['PUNJAB', 'HARYANA', 'MAHARASHTRA', 'GUJARAT', 'ANDHRA_PRADESH', 'KARNATKA', 'TAMIL_NADU']
            if state not in cotton_states:
                return {"error": f"Cotton models not available for {state}. Cotton is primarily grown in major cotton states."}

            # Get variety information
            variety_name = features.get('variety_name')
            if variety_name:
                variety_info = self.cotton_variety_manager.get_variety_info(variety_name)
            else:
                # Select best variety for state and conditions
                conditions = self._extract_conditions_from_features(features, state)
                recommendations = self.cotton_variety_manager.recommend_varieties(conditions, top_n=1)
                variety_name = recommendations[0]['variety'] if recommendations else 'F-1861'
                variety_info = self.cotton_variety_manager.get_variety_info(variety_name)

            # Prepare features for prediction
            processed_features = self._prepare_cotton_features(
                features, state, variety_name, variety_info
            )

            # Make prediction using state-specific model
            model = self.state_models.get(state)
            if model is None:
                return {"error": f"No trained model available for state: {state}"}

            prediction = self._make_cotton_prediction(model, processed_features, variety_info)

            # Generate insights and recommendations
            insights = self._generate_cotton_insights(prediction, features, state, variety_info)

            return {
                "crop": "cotton",
                "variety": variety_name,
                "state": state,
                "predicted_yield_quintal_ha": prediction,
                "unit": "quintal per hectare",
                "confidence_level": "medium",  # Could be improved with validation data
                "insights": insights,
                "timestamp": datetime.utcnow().isoformat(),
                "prediction_method": "multi_state_cotton_model"
            }

        except Exception as e:
            error_handler.handle_error(e, {
                "operation": "cotton_prediction",
                "features_provided": list(features.keys()),
                "error_details": str(e)
            })
            return {"error": f"Cotton prediction failed: {str(e)}"}

    def _determine_state_from_coordinates(self, lat: float, lng: float) -> Optional[str]:
        """Determine Indian state from latitude/longitude coordinates"""

        # Major cotton growing regions
        cotton_regions = {
            'PUNJAB': {'lat_min': 29.5, 'lat_max': 32.5, 'lng_min': 73.8, 'lng_max': 76.9},
            'HARYANA': {'lat_min': 27.5, 'lat_max': 30.9, 'lng_min': 74.4, 'lng_max': 77.6},
            'MAHARASHTRA': {'lat_min': 15.6, 'lat_max': 22.0, 'lng_min': 72.6, 'lng_max': 80.9},
            'GUJARAT': {'lat_min': 20.1, 'lat_max': 24.7, 'lng_min': 68.1, 'lng_max': 74.5},
            'ANDHRA_PRADESH': {'lat_min': 12.9, 'lat_max': 19.1, 'lng_min': 76.7, 'lng_max': 84.8},
            'KARNATKA': {'lat_min': 11.5, 'lat_max': 18.5, 'lng_min': 74.0, 'lng_max': 78.6},
            'TAMIL_NADU': {'lat_min': 8.1, 'lat_max': 13.1, 'lng_min': 76.2, 'lng_max': 80.4}
        }

        for state, bounds in cotton_regions.items():
            if (bounds['lat_min'] <= lat <= bounds['lat_max'] and
                bounds['lng_min'] <= lng <= bounds['lng_max']):
                return state

        return None

    def _extract_conditions_from_features(self, features: Dict[str, Any],
                                        state: str) -> Dict[str, Any]:
        """Extract growing conditions from feature dictionary"""

        return {
            'irrigation_coverage': features.get('irrigation_coverage', 0.5),
            'temperature_celsius': features.get('temperature_celsius', 28),
            'bollworm_pressure': features.get('pest_pressure', 0.5),
            'state': state
        }

    def _prepare_cotton_features(self, features: Dict[str, Any], state: str,
                               variety: str, variety_info: Dict[str, Any]) -> pd.DataFrame:
        """Prepare feature vector for cotton yield prediction"""

        # Base features
        feature_dict = {
            'temperature_celsius': features.get('temperature_celsius', 28),
            'rainfall_mm': features.get('rainfall_mm', 600),
            'humidity_percent': features.get('humidity_percent', 60),
            'area_hectares': features.get('area_hectares', 1.0),
            'soil_ph': features.get('soil_ph', 7.5)
        }

        # Add cotton-specific features
        feature_dict.update(self._add_cotton_specific_features(state))

        # Variety-derived features
        if variety_info:
            feature_dict['variety_yield_potential'] = variety_info.get('yield_potential_quintal_ha', 25) / 100
            feature_dict['variety_maturity_factor'] = (variety_info.get('maturity_days', 165) - 150) / 50  # Normalize

            # BT trait effectiveness
            bt_trait = variety_info.get('bt_trait', 'none')
            if 'cry1ac' in bt_trait:
                feature_dict['bt_effectiveness'] = 0.9 if 'cry2ab' in bt_trait else 0.8

        return pd.DataFrame([feature_dict])

    def _add_cotton_specific_features(self, state: str) -> Dict[str, float]:
        """Add state-specific cotton features"""

        # State-specific pest pressures and conditions
        features = {
            'bollworm_pressure': {
                'PUNJAB': 0.7, 'HARYANA': 0.6, 'MAHARASHTRA': 0.9,
                'GUJARAT': 0.8, 'ANDHRA_PRADESH': 0.7, 'KARNATKA': 0.6, 'TAMIL_NADU': 0.8
            },
            'whitefly_risk': {
                'PUNJAB': 0.3, 'HARYANA': 0.4, 'MAHARASHTRA': 0.7,
                'GUJARAT': 0.6, 'ANDHRA_PRADESH': 0.8, 'KARNATKA': 0.5, 'TAMIL_NADU': 0.9
            },
            'soil_suitability': {
                'PUNJAB': 0.9, 'HARYANA': 0.8, 'MAHARASHTRA': 0.95,
                'GUJARAT': 0.85, 'ANDHRA_PRADESH': 0.75, 'KARNATKA': 0.7, 'TAMIL_NADU': 0.8
            }
        }

        return {
            'bollworm_pressure': features['bollworm_pressure'].get(state, 0.6),
            'whitefly_risk': features['whitefly_risk'].get(state, 0.4),
            'soil_suitability': features['soil_suitability'].get(state, 0.8)
        }

    def _make_cotton_prediction(self, model, features: pd.DataFrame,
                              variety_info: Dict[str, Any]) -> float:
        """Make prediction using trained model and variety data"""

        try:
            # Model prediction
            base_prediction = model.predict(features)[0]

            # Apply variety-specific yield potential
            variety_yield_potential = variety_info.get('yield_potential_quintal_ha', 25)

            # Apply BT cotton premium (varieties with BT technology perform better)
            bt_multiplier = 1.1 if variety_info.get('bt_trait') != 'none' else 1.0

            # Blend model prediction with variety potential
            final_prediction = (base_prediction * 0.7) + (variety_yield_potential * 0.3) * bt_multiplier

            # Ensure prediction is reasonable for cotton (5-40 q/ha typically)
            final_prediction = np.clip(final_prediction, 5.0, 40.0)

            return round(final_prediction, 2)

        except Exception as e:
            logger.warning(f"Model prediction failed, using variety potential: {e}")
            return variety_info.get('yield_potential_quintal_ha', 25)

    def _generate_cotton_insights(self, prediction: float, features: Dict[str, Any],
                                state: str, variety_info: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive cotton cultivation insights"""

        insights = {
            "yield_analysis": self._analyze_cotton_yield_level(prediction, state),
            "pest_management": self._get_cotton_pest_advice(features, variety_info),
            "irrigation_schedule": self._get_cotton_irrigation_advice(features, state),
            "variety_performance": self._evaluate_cotton_variety(variety_info),
            "market_recommendations": self._get_cotton_market_advice(prediction, variety_info),
            "harvesting_guidance": self._get_cotton_harvest_advice(variety_info)
        }

        return insights

    def _analyze_cotton_yield_level(self, prediction: float, state: str) -> str:
        """Analyze predicted yield level compared to state averages"""

        # State averages (cotton yields in q/ha)
        state_averages = {
            'PUNJAB': 12, 'HARYANA': 11, 'MAHARASHTRA': 18,
            'GUJARAT': 15, 'ANDHRA_PRADESH': 14, 'KARNATKA': 13, 'TAMIL_NADU': 19
        }

        avg_yield = state_averages.get(state, 15)

        if prediction > avg_yield * 1.25:
            return ".1f"
        elif prediction > avg_yield * 0.9:
            return ".1f"
        else:
            return ".1f"

    def _get_cotton_pest_advice(self, features: Dict[str, Any],
                              variety_info: Dict[str, Any]) -> str:
        """Generate pest management recommendations for cotton"""

        bt_trait = variety_info.get('bt_trait', 'none')
        bollworm_pressure = features.get('bollworm_pressure', 0.6)

        if bt_trait != 'none':
            if bollworm_pressure > 0.7:
                return "High bollworm pressure detected - ensure proper BT cotton cultivation practices"
            else:
                return "BT cotton variety provides protection against bollworm - monitor for secondary pests like whiteflies"
        else:
            return "Non-BT variety - intensive pest monitoring required, consider integrated pest management"

    def _get_cotton_irrigation_advice(self, features: Dict[str, Any], state: str) -> str:
        """Generate irrigation recommendations for cotton"""

        irrigation_coverage = features.get('irrigation_coverage', 0.5)

        if irrigation_coverage < 0.6:
            if state in ['MAHARASHTRA', 'KARNATKA']:
                return "Cotton in dry regions needs supplemental irrigation - critical at flowering and boll development"
            else:
                return "Maintain adequate soil moisture - cotton vulnerable to water stress during boll formation"
        elif irrigation_coverage > 0.9:
            return "Excessive irrigation may cause nutrient leaching and increased disease risk - monitor soil moisture"
        else:
            return "Current irrigation adequate - maintain consistent moisture during reproductive stages"

    def _evaluate_cotton_variety(self, variety_info: Dict[str, Any]) -> str:
        """Evaluate cotton variety performance characteristics"""

        variety_name = variety_info.get('name', 'Unknown')
        yield_potential = variety_info.get('yield_potential_quintal_ha', 0)
        bt_trait = variety_info.get('bt_trait', 'none')
        fiber_quality = variety_info.get('fiber_type', 'unknown')

        assessment = f"{variety_name} variety"
        if yield_potential >= 30:
            assessment = "High-yielding " + assessment
        elif yield_potential >= 25:
            assessment = "Medium-yielding " + assessment
        else:
            assessment = "Standard-yielding " + assessment

        if bt_trait != 'none':
            assessment += " with BT technology for bollworm protection"
        else:
            assessment += " - consider pest management practices"

        if fiber_quality == 'extra_long_staple':
            assessment += "; premium fiber quality for export markets"

        return assessment

    def _get_cotton_market_advice(self, prediction: float, variety_info: Dict[str, Any]) -> str:
        """Generate market recommendations for cotton"""

        fiber_quality = variety_info.get('fiber_type', 'medium_staple')

        if prediction > 25:
            if fiber_quality == 'extra_long_staple':
                return "High yield with premium fiber - position for export market with quality certifications"
            else:
                return "Good yield potential - focus on ginneries with quality testing facilities"
        elif prediction > 15:
            return "Moderate yield potential - consider contract farming with minimum support price protection"
        else:
            return "Lower yield scenario - focus on risk management and diversified income sources"

    def _get_cotton_harvest_advice(self, variety_info: Dict[str, Any]) -> str:
        """Generate harvesting guidance for cotton"""

        maturity = variety_info.get('maturity_days', 165)
        boll_schedule = variety_info.get('boll_opening_schedule', 'progressive')

        if boll_schedule == 'progressive':
            return f"Progressive boll opening expected - plan for multiple pickings over {maturity} days"
        else:
            return f"Simultaneous boll opening - coordinate harvesting for efficiency after {maturity} days"

    def create_cotton_model_report(self, state: str) -> Dict[str, Any]:
        """Create comprehensive model performance report"""

        # This would analyze training data and create detailed reports
        return {
            'state': state,
            'model_exists': state in self.state_models,
            'variety_count': len(self.cotton_variety_manager.get_varieties_for_state(state)),
            'last_updated': datetime.utcnow().isoformat(),
            'cotton_specific': True
        }

# Global cotton predictor instance
cotton_predictor = CottonYieldPredictor()

def get_cotton_predictor() -> CottonYieldPredictor:
    """Get global cotton predictor instance"""
    return cotton_predictor
