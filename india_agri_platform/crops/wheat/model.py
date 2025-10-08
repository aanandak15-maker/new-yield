"""
Advanced Wheat Yield Prediction Model for Punjab
Production-ready ML model with ensemble techniques and feature engineering
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression
import xgboost as xgb
import joblib
from datetime import datetime
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from scipy import stats

from india_agri_platform.models.model_management import model_manager, ModelType

logger = logging.getLogger(__name__)

class AdvancedWheatModel:
    """Advanced wheat yield prediction model with ensemble techniques"""

    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_selector = SelectKBest(score_func=f_regression, k=15)
        self.poly_features = PolynomialFeatures(degree=2, interaction_only=True)

        # Ensemble models
        self.models = {
            'random_forest': RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=150,
                learning_rate=0.1,
                max_depth=8,
                min_samples_split=5,
                min_samples_leaf=3,
                random_state=42,
                subsample=0.8
            ),
            'xgboost': xgb.XGBRegressor(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=8,
                min_child_weight=3,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1
            )
        }

        self.ensemble_weights = {
            'random_forest': 0.4,
            'gradient_boosting': 0.3,
            'xgboost': 0.3
        }

        # Model performance tracking
        self.performance_metrics = {}
        self.feature_importance = {}
        self.training_history = []

        # Punjab-specific features
        self.punjab_districts = [
            'Amritsar', 'Barnala', 'Bathinda', 'Faridkot', 'Fatehgarh Sahib',
            'Fazilka', 'Ferozepur', 'Gurdaspur', 'Hoshiarpur', 'Jalandhar',
            'Kapurthala', 'Ludhiana', 'Mansa', 'Moga', 'Muktsar', 'Nawanshahr',
            'Pathankot', 'Patiala', 'Rupnagar', 'Sahibzada Ajit Singh Nagar',
            'Sangrur', 'Shahid Bhagat Singh Nagar', 'Sri Muktsar Sahib', 'Tarn Taran'
        ]

        # Wheat-specific growth parameters
        self.wheat_params = {
            'optimal_temp_range': (15, 25),  # Celsius
            'optimal_rainfall': 400,  # mm annual
            'sowing_month': 'November',
            'harvest_month': 'April',
            'growing_season_days': 150,
            'critical_stages': {
                'germination': {'days': 10, 'temp_range': (18, 22)},
                'tillering': {'days': 30, 'temp_range': (12, 18)},
                'stem_elongation': {'days': 25, 'temp_range': (15, 20)},
                'grain_filling': {'days': 35, 'temp_range': (20, 25)},
                'maturation': {'days': 20, 'temp_range': (25, 30)}
            }
        }

        logger.info("Advanced Wheat Model initialized")

    def engineer_wheat_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Advanced feature engineering for wheat yield prediction"""

        df = data.copy()

        # Temporal features (wheat growth cycle specific)
        if 'date' in df.columns:
            df['day_of_year'] = pd.to_datetime(df['date']).dt.dayofyear
            df['season_progress'] = df['day_of_year'] / 365

            # Wheat specific temporal windows
            df['wheat_sowing_period'] = ((df['day_of_year'] >= 300) & (df['day_of_year'] <= 320)).astype(int)  # Nov-Dec
            df['wheat_tillering_period'] = ((df['day_of_year'] >= 320) & (df['day_of_year'] <= 340)).astype(int)  # Dec-Jan
            df['wheat_flowering_period'] = ((df['day_of_year'] >= 75) & (df['day_of_year'] <= 90)).astype(int)  # Mar
            df['wheat_harvest_period'] = ((df['day_of_year'] >= 90) & (df['day_of_year'] <= 110)).astype(int)  # Apr

        # Weather stress indices
        if 'temperature' in df.columns:
            optimal_min, optimal_max = self.wheat_params['optimal_temp_range']
            df['temp_stress'] = np.where(
                df['temperature'].between(optimal_min, optimal_max), 0,
                np.abs(df['temperature'] - np.mean([optimal_min, optimal_max]))
            )

            # Heat stress (wheat sensitive)
            df['heat_stress_index'] = np.where(df['temperature'] > 35, 1,
                                             np.where(df['temperature'] > 30, 0.5, 0))

            # Cold stress (wheat sensitive during certain stages)
            df['cold_stress_index'] = np.where(df['temperature'] < 5, 1,
                                             np.where(df['temperature'] < 10, 0.5, 0))

        # Rainfall quality metrics
        if 'rainfall' in df.columns:
            df['rainfall_distribution'] = df['rainfall'].rolling(window=7).std()  # Weekly variability
            df['extreme_rainfall'] = (df['rainfall'] > 50).astype(int)  # Heavy rain events

            # Water availability index
            df['water_availability_index'] = np.clip(df['rainfall'] / self.wheat_params['optimal_rainfall'], 0, 2)

        # Soil moisture stress (if available)
        if 'soil_moisture' in df.columns:
            df['soil_moisture_stress'] = np.where(df['soil_moisture'] < 30, (30 - df['soil_moisture']) / 30, 0)

        # NDVI growth indicators
        if 'ndvi' in df.columns:
            # Growth rate (change over time)
            df['ndvi_growth_rate'] = df['ndvi'].pct_change(periods=7)  # Weekly growth rate

            # Vegetation health categories
            df['vegetation_stress'] = np.where(df['ndvi'] < 0.3, 1,
                                             np.where(df['ndvi'] < 0.5, 0.5, 0))

            # Growth stage estimation based on NDVI
            df['estimated_growth_stage'] = np.where(df['ndvi'] < 0.2, 'emergence',
                                                   np.where(df['ndvi'] < 0.4, 'vegetative',
                                                           np.where(df['ndvi'] < 0.7, 'reproductive', 'ripening')))

        # Geographical features for Punjab
        if 'district' in df.columns:
            # Distance from optimal wheat growing regions
            # Bathinda, Ludhiana, Jalandhar are considered best for wheat
            optimal_districts = ['Bathinda', 'Ludhiana', 'Jalandhar']
            df['optimal_region'] = df['district'].isin(optimal_districts).astype(int)

            # Irrigation facility index (Punjab districts vary in irrigation access)
            irrigation_access = {
                'Bathinda': 0.95, 'Ludhiana': 0.92, 'Jalandhar': 0.88, 'Amritsar': 0.85,
                'Patiala': 0.80, 'Sangrur': 0.78, 'Ferozepur': 0.82, 'Faridkot': 0.86,
                'Mansa': 0.75, 'Barnala': 0.77, 'Sri Muktsar Sahib': 0.79, 'Moga': 0.81,
                'Shahid Bhagat Singh Nagar': 0.73, 'Tarn Taran': 0.74, 'Rupnagar': 0.71,
                'Sahibzada Ajit Singh Nagar': 0.76, 'Fatehgarh Sahib': 0.72, 'Kapurthala': 0.78,
                'Pathankot': 0.70, 'Gurdaspur': 0.69, 'Hoshiarpur': 0.75, 'Nawanshahr': 0.74,
                'Muktsar': 0.80, 'Fazilka': 0.82
            }
            df['irrigation_access_index'] = df['district'].map(irrigation_access).fillna(0.7)

        # Climate zone factors
        df['climate_zone'] = 'subtropical'  # Punjab is subtropical

        # Inter-feature interactions
        if all(col in df.columns for col in ['temperature', 'rainfall', 'ndvi']):
            # Weather-yield interaction
            df['temp_rainfall_interaction'] = df['temperature'] * df['rainfall'] / 100

            # Vegetation-weather stress
            df['weather_vegetation_stress'] = df['temp_stress'] * df['vegetation_stress']

        # Rolling statistics (7-day windows)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col != 'yield':  # Don't compute rolling stats for target
                df[f'{col}_rolling_mean_7d'] = df[col].rolling(window=7, min_periods=1).mean()
                df[f'{col}_rolling_std_7d'] = df[col].rolling(window=7, min_periods=1).std().fillna(0)

        return df

    def train_ensemble(self, X: pd.DataFrame, y: pd.Series, validation_size: float = 0.2) -> Dict[str, Any]:
        """Train ensemble model with cross-validation"""

        logger.info(f"Training wheat yield ensemble model with {len(X)} samples")

        # Split data
        split_idx = int(len(X) * (1 - validation_size))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        performaces = {}
        predictions = {}

        for model_name, model in self.models.items():
            try:
                logger.info(f"Training {model_name}")
                start_time = datetime.utcnow()

                # Train model
                model.fit(X_train, y_train)

                # Make predictions
                train_pred = model.predict(X_train)
                val_pred = model.predict(X_val)

                # Calculate metrics
                train_r2 = r2_score(y_train, train_pred)
                val_r2 = r2_score(y_val, val_pred)
                val_mae = mean_absolute_error(y_val, val_pred)
                val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))

                performaces[model_name] = {
                    'train_r2': train_r2,
                    'validation_r2': val_r2,
                    'validation_mae': val_mae,
                    'validation_rmse': val_rmse,
                    'training_time': (datetime.utcnow() - start_time).total_seconds()
                }

                predictions[model_name] = val_pred

                # Feature importance (for tree-based models)
                if hasattr(model, 'feature_importances_'):
                    self.feature_importance[model_name] = dict(zip(X.columns, model.feature_importances_))

                logger.info(f"{model_name} trained - Val R²: {val_r2:.4f}, MAE: {val_mae:.2f}")

            except Exception as e:
                logger.error(f"Failed to train {model_name}: {e}")
                performaces[model_name] = {'error': str(e)}

        # Compute ensemble predictions
        ensemble_pred = self._compute_ensemble_prediction(predictions, self.ensemble_weights)

        if ensemble_pred is not None:
            ensemble_r2 = r2_score(y_val, ensemble_pred)
            ensemble_mae = mean_absolute_error(y_val, ensemble_pred)
            ensemble_rmse = np.sqrt(mean_squared_error(y_val, ensemble_pred))

            performaces['ensemble'] = {
                'validation_r2': ensemble_r2,
                'validation_mae': ensemble_mae,
                'validation_rmse': ensemble_rmse
            }

            logger.info(f"Ensemble - Val R²: {ensemble_r2:.4f}, MAE: {ensemble_mae:.2f}")

        # Store training history
        self.training_history.append({
            'timestamp': datetime.utcnow(),
            'data_size': len(X),
            'performances': performaces
        })

        self.performance_metrics = performaces

        return performaces

    def _compute_ensemble_prediction(self, predictions: Dict[str, np.ndarray],
                                   weights: Dict[str, float]) -> Optional[np.ndarray]:
        """Compute weighted ensemble prediction"""

        if not predictions:
            return None

        prediction_arrays = [pred for pred in predictions.values() if len(pred) > 0]

        if not prediction_arrays:
            return None

        # Ensure all predictions have same length
        min_length = min(len(pred) for pred in prediction_arrays)
        prediction_arrays = [pred[:min_length] for pred in prediction_arrays]

        # Weighted average
        weighted_sum = np.zeros(min_length)
        total_weight = 0

        for model_name, pred in predictions.items():
            if model_name in weights and len(pred) >= min_length:
                weight = weights[model_name]
                weighted_sum += pred[:min_length] * weight
                total_weight += weight

        if total_weight > 0:
            return weighted_sum / total_weight

        return None

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make ensemble prediction"""

        predictions = {}

        for model_name, model in self.models.items():
            try:
                pred = model.predict(X)
                predictions[model_name] = pred
            except Exception as e:
                logger.warning(f"Prediction failed for {model_name}: {e}")

        ensemble_pred = self._compute_ensemble_prediction(predictions, self.ensemble_weights)

        if ensemble_pred is None:
            # Fallback to best performing model
            best_model = self._get_best_model()
            if best_model:
                ensemble_pred = predictions.get(best_model, np.zeros(len(X)))

        return ensemble_pred

    def _get_best_model(self) -> Optional[str]:
        """Get best performing model based on recent validation"""

        if not self.performance_metrics:
            return None

        best_model = None
        best_score = -float('inf')

        for model_name, metrics in self.performance_metrics.items():
            if model_name != 'ensemble' and 'validation_r2' in metrics:
                score = metrics['validation_r2']
                if score > best_score:
                    best_score = score
                    best_model = model_name

        return best_model

    def cross_validate(self, X: pd.DataFrame, y: pd.Series, cv_folds: int = 5) -> Dict[str, Any]:
        """Perform time series cross-validation"""

        tscv = TimeSeriesSplit(n_splits=cv_folds)

        cv_results = {}

        for model_name, model in self.models.items():
            try:
                # Cross-validation scores
                r2_scores = cross_val_score(model, X, y, cv=tscv, scoring='r2')
                mae_scores = cross_val_score(model, X, y, cv=tscv, scoring='neg_mean_absolute_error')
                rmse_scores = cross_val_score(model, X, y, cv=tscv, scoring='neg_root_mean_squared_error')

                cv_results[model_name] = {
                    'r2_mean': r2_scores.mean(),
                    'r2_std': r2_scores.std(),
                    'mae_mean': -mae_scores.mean(),  # Negate because sklearn returns negative MAE
                    'mae_std': mae_scores.std(),
                    'rmse_mean': -rmse_scores.mean(),
                    'rmse_std': rmse_scores.std()
                }

            except Exception as e:
                logger.error(f"Cross-validation failed for {model_name}: {e}")
                cv_results[model_name] = {'error': str(e)}

        return cv_results

    def get_feature_importance(self, model_name: str = None) -> Dict[str, float]:
        """Get feature importance for a specific model or ensemble"""

        if model_name and model_name in self.feature_importance:
            return self.feature_importance[model_name]

        # Compute ensemble feature importance
        if not self.feature_importance:
            return {}

        ensemble_importance = {}
        total_weight = sum(self.ensemble_weights.values())

        for model_name, importance in self.feature_importance.items():
            weight = self.ensemble_weights.get(model_name, 0)
            for feature, score in importance.items():
                ensemble_importance[feature] = ensemble_importance.get(feature, 0) + (score * weight)

        if total_weight > 0:
            ensemble_importance = {k: v / total_weight for k, v in ensemble_importance.items()}

        return dict(sorted(ensemble_importance.items(), key=lambda x: x[1], reverse=True))

    def get_seasonal_yield_patterns(self, historical_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze seasonal yield patterns specific to Punjab wheat"""

        patterns = {}

        if 'yield' in historical_data.columns and 'district' in historical_data.columns:

            # District-wise yield trends
            district_yields = historical_data.groupby('district')['yield'].agg(['mean', 'std', 'min', 'max'])
            patterns['district_performance'] = district_yields.to_dict('index')

            # Year-over-year yield trends
            if 'year' in historical_data.columns:
                yearly_trends = historical_data.groupby('year')['yield'].agg(['mean', 'std', 'count'])
                patterns['yearly_trends'] = yearly_trends.to_dict('index')

                # Calculate trend
                years = yearly_trends.index.values
                yields = yearly_trends['mean'].values

                if len(years) >= 3:
                    slope, _, r_value, _, _ = stats.linregress(years, yields)
                    patterns['yield_trend'] = {
                        'slope': slope,
                        'r_squared': r_value ** 2,
                        'direction': 'increasing' if slope > 0 else 'decreasing',
                        'annual_improvement': slope
                    }

        # Punjab-specific insights
        patterns['punjab_insights'] = {
            'top_performing_districts': ['Bathinda', 'Ludhiana', 'Jalandhar'],
            'optimal_sowing_window': 'November 15 - December 15',
            'critical_rainfall_months': ['January', 'February', 'March'],
            'yield_forecasting_period': 'Early March for reliable predictions',
            'climate_change_impacts': 'Increasing temperatures affecting grain filling stage'
        }

        return patterns

    def predict_yield_with_uncertainty(self, X: pd.DataFrame, n_simulations: int = 100) -> Dict[str, Any]:
        """Predict yield with uncertainty estimation using ensemble variance"""

        predictions = []

        # Generate predictions from individual models
        for model_name, model in self.models.items():
            try:
                pred = model.predict(X)
                predictions.extend([pred[0]] * int(self.ensemble_weights[model_name] * n_simulations))
            except Exception:
                continue

        if not predictions:
            return {
                'prediction': 0.0,
                'confidence_interval': (0.0, 0.0),
                'uncertainty_score': 1.0
            }

        # Calculate statistics
        prediction_mean = np.mean(predictions)
        prediction_std = np.std(predictions)
        confidence_interval = (
            prediction_mean - 1.96 * prediction_std,
            prediction_mean + 1.96 * prediction_std
        )

        # Uncertainty score (coefficient of variation)
        uncertainty_score = prediction_std / abs(prediction_mean) if prediction_mean != 0 else 1.0

        return {
            'prediction': prediction_mean,
            'confidence_interval': confidence_interval,
            'uncertainty_score': uncertainty_score,
            'prediction_range': prediction_std * 2,  # 95% range
            'confidence_level': max(0, 1 - uncertainty_score)  # Higher is better
        }

    def save_model(self, filepath: Path):
        """Save trained model to disk"""

        model_data = {
            'models': self.models,
            'ensemble_weights': self.ensemble_weights,
            'scaler': self.scaler,
            'feature_selector': self.feature_selector,
            'performance_metrics': self.performance_metrics,
            'feature_importance': self.feature_importance,
            'training_history': self.training_history,
            'wheat_params': self.wheat_params,
            'punjab_districts': self.punjab_districts,
            'poly_features': self.poly_features
        }

        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model_data, filepath)
        logger.info(f"Wheat model saved to {filepath}")

    def load_model(self, filepath: Path) -> bool:
        """Load trained model from disk"""

        try:
            if not filepath.exists():
                logger.warning(f"Model file not found: {filepath}")
                return False

            model_data = joblib.load(filepath)

            self.models = model_data['models']
            self.ensemble_weights = model_data['ensemble_weights']
            self.scaler = model_data['scaler']
            self.feature_selector = model_data['feature_selector']
            self.performance_metrics = model_data.get('performance_metrics', {})
            self.feature_importance = model_data.get('feature_importance', {})
            self.training_history = model_data.get('training_history', [])
            self.wheat_params = model_data.get('wheat_params', self.wheat_params)
            self.punjab_districts = model_data.get('punjab_districts', self.punjab_districts)
            self.poly_features = model_data.get('poly_features', self.poly_features)

            logger.info(f"Wheat model loaded from {filepath}")
            return True

        except Exception as e:
            logger.error(f"Failed to load wheat model: {e}")
            return False

    def get_model_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""

        report = {
            'model_type': 'wheat_yield_ensemble',
            'ensemble_composition': list(self.models.keys()),
            'ensemble_weights': self.ensemble_weights,
            'performance_metrics': self.performance_metrics,
            'feature_importance': self.get_feature_importance(),
            'training_history': len(self.training_history)
        }

        # Add Punjab-specific insights
        report['punjab_specific'] = {
            'supported_districts': len(self.punjab_districts),
            'optimal_regions': ['Bathinda', 'Ludhiana', 'Jalandhar'],
            'seasonal_factors': {
                'sowing_season': 'Rabi (Winter)',
                'sowing_months': ['November', 'December'],
                'harvest_season': 'Summer',
                'harvest_months': ['April', 'May']
            }
        }

        return report

    async def register_with_model_manager(self, model_id: str) -> bool:
        """Register model with the advanced model management system"""

        try:
            # Prepare input/output feature definitions
            input_features = [
                'temperature', 'humidity', 'rainfall', 'wind_speed', 'ndvi', 'ndwi',
                'soil_ph', 'soil_nitrogen', 'soil_phosphorus', 'soil_potassium',
                'district', 'season_progress', 'irrigation_access_index',
                'temp_stress', 'heat_stress_index', 'water_availability_index'
            ]

            output_features = ['yield']

            # Use best performance metrics
            best_metrics = self._get_best_performance_metrics()

            success = await model_manager.register_model(
                model=self,
                model_type=ModelType.ENSEMBLE,
                input_features=input_features,
                output_features=output_features,
                hyperparams=self.ensemble_weights,
                performance_metrics={
                    'accuracy': best_metrics.get('validation_r2', 0),
                    'r2_score': best_metrics.get('validation_r2', 0),
                    'mae': best_metrics.get('validation_mae', 0),
                    'rmse': best_metrics.get('validation_rmse', 0)
                },
                framework="ensemble_sklearn_xgboost"
            )

            if success:
                logger.info(f"Wheat model registered with ID: {model_id}")

                # Auto-deploy if performance is good
                if best_metrics.get('validation_r2', 0) > 0.75:
                    await model_manager.deploy_model(model_id)

            return success

        except Exception as e:
            logger.error(f"Failed to register wheat model with manager: {e}")
            return False

    def _get_best_performance_metrics(self) -> Dict[str, float]:
        """Get best performance metrics from ensemble components"""

        best_r2 = 0
        best_metrics = {}

        for model_name, metrics in self.performance_metrics.items():
            if 'validation_r2' in metrics:
                r2_score = metrics['validation_r2']
                if r2_score > best_r2:
                    best_r2 = r2_score
                    best_metrics = metrics

        return best_metrics or {'validation_r2': 0, 'validation_mae': 0, 'validation_rmse': 0}

# Factory function for creating trained wheat model
def create_trained_wheat_model() -> AdvancedWheatModel:
    """Factory function to create and return a trained wheat model"""

    model = AdvancedWheatModel()

    # TODO: In production, this would load from trained model files
    # For now, return initialized model

    logger.info("Created Advanced Wheat Yield Prediction Model")
    return model

# Global wheat predictor instance (for multi_crop_predictor compatibility)
wheat_predictor = create_trained_wheat_model()
