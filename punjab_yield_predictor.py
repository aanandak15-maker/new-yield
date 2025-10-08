"""
Punjab Rabi Crop Yield Prediction Model
SIH 2025 Final Project

This module implements a machine learning model for predicting wheat yield
in Punjab districts using satellite, weather, and soil data.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV
import joblib
from datetime import datetime
from data_fetcher import PunjabDataFetcher

# ===========================
# Ensure output folder exists
# ===========================
os.makedirs("punjab_plots", exist_ok=True)
os.makedirs("punjab_models", exist_ok=True)

class PunjabYieldPredictor:
    """Machine learning model for Punjab wheat yield prediction"""

    def __init__(self, model_dir="punjab_models"):
        self.model_dir = model_dir
        self.scaler = StandardScaler()
        self.poly = PolynomialFeatures(degree=2, include_bias=False)
        self.models = {}
        self.best_model = None
        self.feature_names = None

        # Enhanced Punjab-specific features for wheat yield prediction
        self.features = [
            'temperature_celsius',     # Temperature affects wheat growth stages
            'rainfall_mm',            # Rainfall crucial for Rabi season
            'humidity_percent',       # Humidity affects disease pressure
            'wind_speed_kmph',        # Wind affects evapotranspiration
            'ndvi',                   # Vegetation index from satellite
            'fpar',                   # Fraction of photosynthetically active radiation
            'lai',                    # Leaf area index
            'soil_ph',               # Soil pH affects nutrient availability
            'organic_carbon_percent', # Soil organic matter
            'salinity_dsm',          # Soil salinity (important in Punjab)
            'irrigation_coverage',   # Irrigation access (critical for Punjab)
            'groundwater_depth',     # Groundwater availability
            # Agricultural variety features
            'yield_potential_quintal_ha',  # Variety genetic potential
            'drought_tolerance',     # Variety drought tolerance
            'heat_tolerance',        # Variety heat tolerance
            'yellow_rust_resistance', # Disease resistance scores
            'brown_rust_resistance',
            'aphid_resistance',
            'maturity_days',         # Crop maturity period
            # Irrigation scheduling features
            'irrigation_efficiency', # System efficiency
            'crop_water_requirement_mm_day',  # Daily water needs
            'soil_moisture_capacity_mm'       # Available soil water
        ]

        self.target = 'wheat_yield_quintal_per_hectare'

    def load_training_data(self, year=None):
        """Load training data for Punjab"""
        if year is None:
            year = datetime.now().year - 1

        data_file = f"punjab_data/processed/training_data_{year}.csv"

        if not os.path.exists(data_file):
            print(f"Training data not found. Generating data for {year}...")
            fetcher = PunjabDataFetcher()
            df = fetcher.create_training_dataset(year)
        else:
            df = pd.read_csv(data_file)

        print(f"Loaded training data: {df.shape[0]} samples, {df.shape[1]} features")
        return df

    def preprocess_data(self, df):
        """Preprocess the training data"""
        # Remove rows with missing values
        df_clean = df.dropna(subset=self.features + [self.target])

        # Extract features and target
        X = df_clean[self.features]
        y = df_clean[self.target]

        print(f"After preprocessing: {X.shape[0]} samples")
        print(f"Features: {list(X.columns)}")

        return X, y, df_clean

    def train_models(self, X, y):
        """Train multiple models and select the best one"""
        print("Training models...")

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Model 1: Ridge Regression with Polynomial Features
        print("Training Ridge Regression model...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_train_poly = self.poly.fit_transform(X_train_scaled)

        ridge_model = Ridge(alpha=0.1, random_state=42)
        ridge_model.fit(X_train_poly, y_train)

        # Evaluate Ridge model
        X_test_scaled = self.scaler.transform(X_test)
        X_test_poly = self.poly.transform(X_test_scaled)
        ridge_pred = ridge_model.predict(X_test_poly)
        ridge_r2 = r2_score(y_test, ridge_pred)
        ridge_mae = mean_absolute_error(y_test, ridge_pred)

        print(".4f")
        print(".2f")

        # Model 2: Random Forest
        print("Training Random Forest model...")
        rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        rf_model.fit(X_train, y_train)
        rf_pred = rf_model.predict(X_test)
        rf_r2 = r2_score(y_test, rf_pred)
        rf_mae = mean_absolute_error(y_test, rf_pred)

        print(".4f")
        print(".2f")

        # Model 3: Gradient Boosting
        print("Training Gradient Boosting model...")
        gb_model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        gb_model.fit(X_train, y_train)
        gb_pred = gb_model.predict(X_test)
        gb_r2 = r2_score(y_test, gb_pred)
        gb_mae = mean_absolute_error(y_test, gb_pred)

        print(".4f")
        print(".2f")

        # Store models and their performance
        self.models = {
            'ridge': {
                'model': ridge_model,
                'r2': ridge_r2,
                'mae': ridge_mae,
                'predictions': ridge_pred
            },
            'random_forest': {
                'model': rf_model,
                'r2': rf_r2,
                'mae': rf_mae,
                'predictions': rf_pred
            },
            'gradient_boosting': {
                'model': gb_model,
                'r2': gb_r2,
                'mae': gb_mae,
                'predictions': gb_pred
            }
        }

        # Select best model based on R² score
        best_model_name = max(self.models.keys(), key=lambda k: self.models[k]['r2'])
        self.best_model = self.models[best_model_name]['model']

        print(f"\nBest model: {best_model_name} (R² = {self.models[best_model_name]['r2']:.4f})")

        # Store feature names for later use
        self.feature_names = list(X.columns)

        return X_train, X_test, y_train, y_test

    def save_model(self, model_name="punjab_yield_model"):
        """Save the trained model"""
        if self.best_model is None:
            print("No model trained yet!")
            return

        # For sklearn models that expect feature names, we need to handle this
        # by creating a wrapper that converts to numpy arrays
        model_data = {
            'model': self.best_model,
            'scaler': self.scaler,
            'poly_features': self.poly,
            'feature_names': self.feature_names,
            'models_performance': {k: v['r2'] for k, v in self.models.items()},
            'trained_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'is_ridge_model': hasattr(self.best_model, 'alpha')
        }

        model_file = os.path.join(self.model_dir, f"{model_name}.pkl")
        joblib.dump(model_data, model_file)
        print(f"Model saved to {model_file}")

    def load_model(self, model_name="punjab_yield_model"):
        """Load a trained model"""
        model_file = os.path.join(self.model_dir, f"{model_name}.pkl")

        if not os.path.exists(model_file):
            print(f"Model file {model_file} not found!")
            return False

        model_data = joblib.load(model_file)
        self.best_model = model_data['model']
        self.scaler = model_data['scaler']
        self.poly = model_data['poly_features']
        self.feature_names = model_data['feature_names']

        print(f"Model loaded from {model_file}")
        print(f"Trained on: {model_data['trained_date']}")
        print(f"Performance: {model_data['models_performance']}")

        return True

    def predict_yield(self, input_data):
        """
        Predict wheat yield for new data
        input_data should be a DataFrame with the required features
        """
        if self.best_model is None:
            print("No model loaded! Please train or load a model first.")
            return None

        # Ensure input has all required features
        missing_features = set(self.feature_names) - set(input_data.columns)
        if missing_features:
            print(f"Missing features: {missing_features}")
            return None

        # Create a clean DataFrame with only the required features in the correct order
        X_pred = input_data[self.feature_names].copy()

        # Convert to numpy array to avoid feature name issues
        X_pred_array = X_pred.values

        print(f"Predicting with {X_pred_array.shape[1]} features")

        # Check if it's a Ridge model (needs scaling and polynomial features)
        if hasattr(self.best_model, 'alpha'):  # Ridge model
            X_pred_scaled = self.scaler.transform(X_pred_array)
            X_pred_poly = self.poly.transform(X_pred_scaled)
            predictions = self.best_model.predict(X_pred_poly)
        else:  # Tree-based models
            predictions = self.best_model.predict(X_pred_array)

        return predictions

    def create_visualizations(self, X_train, X_test, y_train, y_test, df):
        """Create comprehensive visualizations for the model"""
        print("Creating visualizations...")

        # 1. Model Performance Comparison
        plt.figure(figsize=(12, 6))
        model_names = list(self.models.keys())
        r2_scores = [self.models[m]['r2'] for m in model_names]
        mae_scores = [self.models[m]['mae'] for m in model_names]

        x = np.arange(len(model_names))
        width = 0.35

        fig, ax1 = plt.subplots(figsize=(12, 6))
        ax1.bar(x - width/2, r2_scores, width, label='R² Score', color='skyblue')
        ax1.set_xlabel('Models')
        ax1.set_ylabel('R² Score', color='skyblue')
        ax1.set_title('Model Performance Comparison - Punjab Wheat Yield Prediction')
        ax1.set_xticks(x)
        ax1.set_xticklabels([m.replace('_', ' ').title() for m in model_names])

        ax2 = ax1.twinx()
        ax2.bar(x + width/2, mae_scores, width, label='MAE', color='lightcoral')
        ax2.set_ylabel('Mean Absolute Error (quintal/ha)', color='lightcoral')

        plt.tight_layout()
        plt.savefig('punjab_plots/01_model_comparison.png', bbox_inches='tight')
        plt.show()

        # 2. Predicted vs Actual for Best Model
        best_model_name = max(self.models.keys(), key=lambda k: self.models[k]['r2'])
        y_pred = self.models[best_model_name]['predictions']

        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.6, color='blue')
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
        plt.xlabel('Actual Wheat Yield (quintal/ha)')
        plt.ylabel('Predicted Wheat Yield (quintal/ha)')
        plt.title(f'Predicted vs Actual Wheat Yield - {best_model_name.replace("_", " ").title()}')
        plt.grid(True, alpha=0.3)
        plt.savefig('punjab_plots/02_predicted_vs_actual.png', bbox_inches='tight')
        plt.show()

        # 3. Feature Importance (for Random Forest)
        if 'random_forest' in self.models:
            rf_model = self.models['random_forest']['model']
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': rf_model.feature_importances_
            }).sort_values('importance', ascending=True)

            plt.figure(figsize=(10, 8))
            plt.barh(importance_df['feature'], importance_df['importance'], color='green')
            plt.xlabel('Feature Importance')
            plt.ylabel('Features')
            plt.title('Feature Importance - Random Forest Model')
            plt.tight_layout()
            plt.savefig('punjab_plots/03_feature_importance.png', bbox_inches='tight')
            plt.show()

        # 4. Yield Distribution by District
        plt.figure(figsize=(14, 8))
        district_yields = df.groupby('district')[self.target].mean().sort_values(ascending=False)
        district_yields.plot(kind='bar', color='orange')
        plt.xlabel('District')
        plt.ylabel('Average Wheat Yield (quintal/ha)')
        plt.title('Average Wheat Yield by District - Punjab 2023')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('punjab_plots/04_district_yields.png', bbox_inches='tight')
        plt.show()

        # 5. Correlation Heatmap
        plt.figure(figsize=(12, 10))
        correlation_matrix = df[self.features + [self.target]].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', square=True)
        plt.title('Feature Correlation Matrix - Punjab Wheat Yield')
        plt.tight_layout()
        plt.savefig('punjab_plots/05_correlation_heatmap.png', bbox_inches='tight')
        plt.show()

        # 6. Monthly Yield Trends
        plt.figure(figsize=(12, 6))
        monthly_yields = df.groupby('month')[self.target].mean()
        monthly_yields.plot(kind='line', marker='o', linewidth=2, color='purple')
        plt.xlabel('Month')
        plt.ylabel('Average Wheat Yield (quintal/ha)')
        plt.title('Monthly Wheat Yield Trends - Punjab Rabi Season')
        plt.xticks(range(1, 13))
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('punjab_plots/06_monthly_trends.png', bbox_inches='tight')
        plt.show()

    def generate_report(self):
        """Generate a comprehensive report of the model performance"""
        if not self.models:
            print("No models trained yet!")
            return

        report = f"""
Punjab Rabi Crop Yield Prediction Model Report
{'='*50}

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

MODEL PERFORMANCE SUMMARY
{'-'*25}

"""

        for model_name, model_data in self.models.items():
            report += f"""
{model_name.replace('_', ' ').title()}:
  R² Score: {model_data['r2']:.4f}
  Mean Absolute Error: {model_data['mae']:.2f} quintal/ha
  RMSE: {np.sqrt(mean_squared_error([0]*len(model_data['predictions']), model_data['predictions'] - model_data['predictions'])):.2f} quintal/ha
"""

        best_model = max(self.models.keys(), key=lambda k: self.models[k]['r2'])
        report += f"""

BEST MODEL: {best_model.replace('_', ' ').title()}
R² Score: {self.models[best_model]['r2']:.4f}

FEATURES USED ({len(self.feature_names)}):
{chr(10).join(f"  - {f}" for f in self.feature_names)}

DATA SOURCES:
  - Punjab Agriculture Department (yield data)
  - India Meteorological Department (weather data)
  - Google Earth Engine (satellite vegetation indices)
  - Punjab Remote Sensing Centre (soil data)

TRAINING PERIOD: Rabi Season (November-April)
GEOGRAPHICAL COVERAGE: 22 districts of Punjab

MODEL LIMITATIONS:
  - Trained on historical data (2023)
  - Performance may vary with extreme weather events
  - Satellite data quality depends on cloud cover
  - Soil data represents average conditions

RECOMMENDATIONS:
  - Update model annually with new yield data
  - Incorporate real-time weather forecasts for predictions
  - Validate model performance across different agro-climatic zones
  - Consider district-specific model tuning for better accuracy
"""

        # Save report
        report_file = os.path.join(self.model_dir, 'model_report.txt')
        with open(report_file, 'w') as f:
            f.write(report)

        print(f"Model report saved to {report_file}")
        print(report)

def main():
    """Main function to train and evaluate the Punjab yield prediction model"""
    print("Punjab Rabi Crop Yield Prediction System")
    print("=" * 50)

    # Initialize predictor
    predictor = PunjabYieldPredictor()

    # Load and preprocess training data
    df = predictor.load_training_data(2023)
    X, y, df_clean = predictor.preprocess_data(df)

    # Train models
    X_train, X_test, y_train, y_test = predictor.train_models(X, y)

    # Save the best model
    predictor.save_model()

    # Create visualizations
    predictor.create_visualizations(X_train, X_test, y_train, y_test, df_clean)

    # Generate report
    predictor.generate_report()

    print("\nModel training completed successfully!")
    print("Check the 'punjab_plots' folder for visualizations")
    print("Check the 'punjab_models' folder for saved models")

if __name__ == "__main__":
    main()
