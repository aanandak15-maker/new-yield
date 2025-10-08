"""
Comprehensive Accuracy Testing for India Agricultural Intelligence Platform
"""

import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add platform to path
sys.path.append('india_agri_platform')

def load_and_prepare_data():
    """Load and prepare training data"""
    print("Loading datasets...")

    # Load agriyield dataset (primary ML data)
    import zipfile
    with zipfile.ZipFile('agriyield-2025.zip', 'r') as zip_ref:
        with zip_ref.open('train.csv') as f:
            df = pd.read_csv(f)

    print(f"Loaded {len(df)} records from agriyield dataset")

    # Select features for training
    features = ['temperature', 'humidity', 'rainfall', 'ndvi', 'soil_ph', 'sand_pct', 'organic_matter']
    target = 'yield'

    # Clean data
    df_clean = df.dropna(subset=features + [target])
    df_clean = df_clean[features + [target]]

    print(f"After cleaning: {len(df_clean)} records")
    print(f"Features: {features}")

    return df_clean, features, target

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """Train and evaluate multiple ML models"""
    print("\nTraining and evaluating models...")

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'Linear Regression': LinearRegression()
    }

    results = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")

        # Train model
        model.fit(X_train_scaled, y_train)

        # Make predictions
        y_pred = model.predict(X_test_scaled)

        # Calculate metrics
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

        # Cross-validation
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()

        results[name] = {
            'r2': r2,
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'cv_mean': cv_mean,
            'cv_std': cv_std,
            'model': model,
            'scaler': scaler
        }

        print(f"R²: {r2:.4f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"MAE: {mae:.2f}")
        print(f"MAPE: {mape:.2f}%")
        print(f"CV R²: {cv_mean:.3f} ± {cv_std:.3f}")

    return results

def find_best_model(results):
    """Find the best performing model"""
    best_model_name = max(results.keys(), key=lambda x: results[x]['r2'])
    best_metrics = results[best_model_name]

    print(f"\n🏆 BEST MODEL: {best_model_name}")
    print("-" * 50)
    print(f"R²: {best_metrics['r2']:.4f}")
    print(f"RMSE: {best_metrics['rmse']:.2f}")
    print(f"MAE: {best_metrics['mae']:.2f}")
    print(f"MAPE: {best_metrics['mape']:.2f}%")
    print(f"CV R²: {best_metrics['cv_mean']:.3f} ± {best_metrics['cv_std']:.3f}")

    # Accuracy interpretation
    r2 = best_metrics['r2']
    if r2 > 0.9:
        level = "EXCELLENT (>90%)"
    elif r2 > 0.8:
        level = "VERY GOOD (80-90%)"
    elif r2 > 0.7:
        level = "GOOD (70-80%)"
    elif r2 > 0.6:
        level = "FAIR (60-70%)"
    else:
        level = "NEEDS IMPROVEMENT (<60%)"

    print(f"\nACCURACY LEVEL: {level}")

    return best_model_name, best_metrics, level

def test_real_world_predictions(best_model_name, best_metrics):
    """Test predictions with real-world scenarios"""
    print("\n🌍 TESTING REAL-WORLD PREDICTIONS")
    print("-" * 50)

    # Test scenarios
    scenarios = [
        {"name": "Punjab Wheat (Optimal)", "temp": 25, "humidity": 60, "rainfall": 50, "ndvi": 0.65, "ph": 7.0, "sand": 40, "organic": 2.5},
        {"name": "Punjab Rice (Monsoon)", "temp": 28, "humidity": 75, "rainfall": 150, "ndvi": 0.70, "ph": 6.8, "sand": 35, "organic": 2.0},
        {"name": "Haryana Cotton (Summer)", "temp": 32, "humidity": 55, "rainfall": 30, "ndvi": 0.60, "ph": 7.2, "sand": 45, "organic": 1.8},
        {"name": "UP Sugarcane (Irrigated)", "temp": 30, "humidity": 65, "rainfall": 80, "ndvi": 0.75, "ph": 7.1, "sand": 30, "organic": 3.0}
    ]

    model = best_metrics['model']
    scaler = best_metrics['scaler']

    feature_order = ['temperature', 'humidity', 'rainfall', 'ndvi', 'soil_ph', 'sand_pct', 'organic_matter']

    print(f"{'Scenario':<30}{'Predicted (q/ha)':>20}")
    print("-" * 80)

    for scenario in scenarios:
        # Prepare features
        features = [
            scenario['temp'], scenario['humidity'], scenario['rainfall'],
            scenario['ndvi'], scenario['ph'], scenario['sand'], scenario['organic']
        ]

        # Scale and predict
        features_scaled = scaler.transform([features])
        prediction = model.predict(features_scaled)[0]

        print(f"{scenario['name']:<30}{prediction:>20.2f}")

    print("-" * 80)

def generate_comprehensive_report(results, best_model_name, best_metrics, accuracy_level):
    """Generate comprehensive accuracy report"""
    print("\n📊 COMPREHENSIVE ACCURACY REPORT")
    print("=" * 80)

    # Summary
    print("MODEL PERFORMANCE SUMMARY:")
    print("-" * 80)
    print(f"{'Model':<22}{'R²':>10}{'RMSE':>12}{'MAE':>12}{'MAPE %':>10}{'CV R² (±)':>15}")
    print("-" * 80)

    for model_name, metrics in results.items():
        print(f"{model_name:<22}{metrics['r2']:.4f:>10}{metrics['rmse']:.2f:>12}{metrics['mae']:.2f:>12}{metrics['mape']:.2f:>10}{(str(f"{metrics['cv_mean']:.3f} ± {metrics['cv_std']:.3f}")):>15}")

    print("-" * 80)

    # Best model details
    print("\n🏆 BEST MODEL ANALYSIS:")
    print(f"Algorithm: {best_model_name}")
    print(f"R²: {best_metrics['r2']:.1f}")
    print(f"RMSE: {best_metrics['rmse']:.2f} quintal/ha")
    print(f"MAE: {best_metrics['mae']:.2f} quintal/ha")
    print(f"MAPE: {best_metrics['mape']:.2f}%")
    print(f"Cross-validation: {best_metrics['cv_mean']:.3f} ± {best_metrics['cv_std']:.3f}")

    # Production readiness
    print("\n🚀 PRODUCTION READINESS:")
    print("-" * 50)

    r2 = best_metrics['r2']
    if r2 > 0.85:
        print("✅ EXCELLENT: Ready for production deployment")
        print("✅ High accuracy suitable for commercial use")
        print("✅ Proceed with mobile app development")
    elif r2 > 0.75:
        print("⚠️ GOOD: Suitable for production with monitoring")
        print("⚠️ Implement continuous model updates")
        print("⚠️ Add farmer feedback validation")
    else:
        print("❌ NEEDS IMPROVEMENT: Additional development required")
        print("❌ Collect more training data")
        print("❌ Implement ensemble methods")

    # Technical recommendations
    print("\n🔧 TECHNICAL RECOMMENDATIONS:")
    print("• Implement real-time weather API integration")
    print("• Add spatial data from satellite imagery")
    print("• Develop mobile app for field-level predictions")
    print("• Create farmer feedback loop for model improvement")
    print("• Implement automated model retraining pipeline")

    # Save detailed results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"accuracy_test_results_{timestamp}.txt"

    with open(report_file, 'w') as f:
        f.write("INDIA AGRICULTURAL INTELLIGENCE PLATFORM - ACCURACY TEST RESULTS\n")
        f.write("=" * 80 + "\n")
        f.write(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Best Model: {best_model_name}\n")
        f.write(f"Accuracy Level: {accuracy_level}\n")
        f.write("=" * 80 + "\n")

        for model_name, metrics in results.items():
            f.write(f"\n{model_name}:\n")
            f.write(f"  R²: {metrics['r2']:.4f}\n")
            f.write(f"  RMSE: {metrics['rmse']:.2f}\n")
            f.write(f"  MAE: {metrics['mae']:.2f}\n")
            f.write(f"  MAPE: {metrics['mape']:.2f}%\n")
            f.write(f"  CV R²: {metrics['cv_mean']:.3f} ± {metrics['cv_std']:.3f}\n")

    print(f"\n📄 Detailed report saved to: {report_file}")

    return {
        'best_model': best_model_name,
        'accuracy': best_metrics['r2'],
        'level': accuracy_level,
        'production_ready': r2 > 0.8
    }

def main():
    """Main testing function"""
    print("🧪 INDIA AGRICULTURAL INTELLIGENCE PLATFORM - ACCURACY TESTING")
    print("=" * 80)
    print("Comprehensive evaluation of multi-crop, multi-state yield prediction models")
    print("=" * 80)

    start_time = datetime.now()

    try:
        # Phase 1: Data Loading and Preparation
        print("\n📊 PHASE 1: DATA LOADING AND PREPARATION")
        df, features, target = load_and_prepare_data()

        # Phase 2: Train-Test Split
        print("\n🔄 PHASE 2: MODEL TRAINING AND EVALUATION")
        X = df[features]
        y = df[target]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        print(f"Training set: {len(X_train)} samples")
        print(f"Testing set: {len(X_test)} samples")

        # Phase 3: Model Training and Evaluation
        results = train_and_evaluate_models(X_train, X_test, y_train, y_test)

        # Phase 4: Find Best Model
        best_model_name, best_metrics, accuracy_level = find_best_model(results)

        # Phase 5: Real-World Testing
        test_real_world_predictions(best_model_name, best_metrics)

        # Phase 6: Generate Report
        final_results = generate_comprehensive_report(results, best_model_name, best_metrics, accuracy_level)

        # Phase 7: Final Summary
        end_time = datetime.now()
        duration = end_time - start_time

        print("⏱️ TESTING COMPLETED")
        print(f"Duration: {duration}")
        print("=" * 80)

        if final_results['production_ready']:
            print("🎉 SUCCESS: Platform achieves production-ready accuracy!")
            print("The multi-crop, multi-state agricultural intelligence platform")
            print("is validated and ready for commercial deployment.")
        else:
            print("⚠️ CAUTION: Platform needs accuracy improvements")
            print("Additional data collection and model refinement required.")

        print("=" * 80)

    except Exception as e:
        print(f"❌ Testing failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
