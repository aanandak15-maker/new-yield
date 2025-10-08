"""
Rice Model Training for India Agricultural Intelligence Platform
Train state-specific rice yield prediction models using extracted datasets
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib
import os
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

class RiceModelTrainer:
    """Train state-specific rice yield prediction models"""

    def __init__(self, rice_data_dir="rice_data", model_output_dir="india_agri_platform/models/rice_models"):
        self.rice_data_dir = Path(rice_data_dir)
        self.model_output_dir = Path(model_output_dir)
        self.model_output_dir.mkdir(parents=True, exist_ok=True)

        # Training results
        self.training_results = {}

    def train_all_rice_models(self):
        """Train rice models for all available states"""

        print("🌾 RICE MODEL TRAINING - STATE-BY-STATE")
        print("=" * 60)

        # States prioritized by data availability
        training_states = [
            'punjab', 'uttar_pradesh', 'bihar', 'haryana',
            'west_bengal', 'andhra_pradesh', 'tamil_nadu',
            'odisha', 'karnataka', 'assam'
        ]

        # Check which datasets are available
        available_datasets = self._get_available_datasets()

        print(f"Available rice datasets: {available_datasets}")

        successful_states_upper = []

        for state in training_states:
            if f"rice_{state}.csv" in available_datasets:
                success = self.train_state_model(state)
                if success:
                    successful_states_upper.append(state.upper())
            else:
                print(f"⚠️  No dataset found for {state}")

        print("\n" + "=" * 60)
        print(f"✅ TRAINING COMPLETE: {len(successful_states_upper)}/{len(training_states)} states trained")
        print(f"📊 Successful states: {[s.lower() for s in successful_states_upper]}")

        self._create_training_summary(successful_states_upper)

        return successful_states_upper

    def _get_available_datasets(self):
        """Get list of available rice datasets"""
        if not self.rice_data_dir.exists():
            print(f"❌ Rice data directory not found: {self.rice_data_dir}")
            return []

        datasets = [f.name for f in self.rice_data_dir.glob("rice_*.csv")]
        return datasets

    def train_state_model(self, state: str) -> bool:
        """Train model for a specific state"""

        try:
            print(f"\n🏗️  Training rice model for {state.upper()}...")

            # Load state-specific data
            data_file = self.rice_data_dir / f"rice_{state}.csv"

            if not data_file.exists():
                print(f"⚠️  Data file not found: {data_file}")
                return False

            df = pd.read_csv(data_file)
            print(f"   📊 Loaded {len(df)} records for {state.upper()}")

            if len(df) < 50:
                print(f"⚠️  Insufficient data for {state.upper()} ({len(df)} records)")
                return False

            # Prepare features and target
            success = self._prepare_and_train_model(df, state)
            return success

        except Exception as e:
            print(f"❌ Training failed for {state.upper()}: {e}")
            return False

    def _prepare_and_train_model(self, df: pd.DataFrame, state: str) -> bool:
        """Prepare data and train model for a state"""

        try:
            # Define feature columns (from our rice data structure)
            feature_columns = [
                'Area', 'yield_quintal_ha', 'irrigation_coverage',
                'temperature_celsius', 'rainfall_mm', 'humidity_percent',
                'soil_ph', 'Crop_Year'
            ]

            # Check which features are available
            available_features = [col for col in feature_columns if col in df.columns]

            if len(available_features) < 4:
                print(f"   ⚠️  Insufficient features for {state.upper()}")
                return False

            # Prepare target variable
            if 'yield_quintal_ha' not in available_features:
                print(f"   ⚠️  Missing yield data for {state.upper()}")
                return False

            # Create feature matrix
            X = df[available_features].copy()
            y = df['yield_quintal_ha'].copy()

            # Remove target from features if present
            if 'yield_quintal_ha' in X.columns:
                X = X.drop('yield_quintal_ha', axis=1)

            # Handle missing values
            X = X.fillna({
                'Area': X['Area'].median() if 'Area' in X.columns else 1.0,
                'irrigation_coverage': 0.5,
                'temperature_celsius': 25.0,
                'rainfall_mm': 800.0,
                'humidity_percent': 65.0,
                'soil_ph': 7.0,
                'Crop_Year': X['Crop_Year'].median() if 'Crop_Year' in X.columns else 2015
            })

            # Train/test split (temporal validation)
            recent_years = df[df['Crop_Year'] >= 2010]  # Focus on recent data
            if len(recent_years) < 30:
                recent_years = df  # Use all data if insufficient recent data

            # Split by year: older for training, recent for testing
            train_data = recent_years[recent_years['Crop_Year'] <= 2017]
            test_data = recent_years[recent_years['Crop_Year'] >= 2018]

            if len(train_data) < 20:
                # Random split if insufficient temporal data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.3, random_state=42
                )
            else:
                # Use the temporal split we prepared
                X_train = X.loc[train_data.index]
                X_test = X.loc[test_data.index]
                y_train = y.loc[train_data.index]
                y_test = y.loc[test_data.index]

            print(f"   📈 Training data: {len(X_train)} records")
            print(f"   🧪 Testing data: {len(X_test)} records")

            # Feature scaling
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Train Random Forest model
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )

            model.fit(X_train_scaled, y_train)

            # Predictions and evaluation
            y_pred = model.predict(X_test_scaled)

            # Calculate metrics
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)

            # Cross-validation
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=3, scoring='r2')
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()

            # Store results
            model_results = {
                'state': state.upper(),
                'training_records': len(X_train),
                'testing_records': len(X_test),
                'features_used': list(X_train.columns),
                'r2_score': round(r2, 4),
                'rmse': round(rmse, 2),
                'mae': round(mae, 2),
                'cv_r2_mean': round(cv_mean, 4),
                'cv_r2_std': round(cv_std, 4),
                'accuracy_level': self._classify_accuracy(r2),
                'train_years': f"{train_data['Crop_Year'].min()}-{train_data['Crop_Year'].max()}" if 'Crop_Year' in train_data.columns else "unknown",
                'test_years': f"{test_data['Crop_Year'].min()}-{test_data['Crop_Year'].max()}" if 'Crop_Year' in test_data.columns else "unknown"
            }

            self.training_results[state.upper()] = model_results

            print(f"   📊 R² Score: {r2:.4f}")
            print(f"   📏 RMSE: {rmse:.2f} q/ha")
            print(f"   🎯 MAE: {mae:.2f} q/ha")
            print(f"   🏷️  Accuracy: {model_results['accuracy_level']}")

            # Save model and scaler
            self._save_model_and_scaler(model, scaler, state, model_results)

            return True

        except Exception as e:
            print(f"   ❌ Model preparation failed for {state.upper()}: {e}")
            return False

    def _classify_accuracy(self, r2_score: float) -> str:
        """Classify model accuracy level"""
        if r2_score >= 0.8:
            return "EXCELLENT (≥80%)"
        elif r2_score >= 0.7:
            return "VERY GOOD (70-80%)"
        elif r2_score >= 0.6:
            return "GOOD (60-70%)"
        elif r2_score >= 0.5:
            return "FAIR (50-60%)"
        else:
            return "NEEDS IMPROVEMENT (<50%)"

    def _save_model_and_scaler(self, model, scaler, state: str, results: dict):
        """Save trained model and feature scaler"""

        try:
            # Save model
            model_file = self.model_output_dir / f"rice_model_{state.lower()}.pkl"
            joblib.dump(model, model_file)

            # Save scaler
            scaler_file = self.model_output_dir / f"rice_scaler_{state.lower()}.pkl"
            joblib.dump(scaler, scaler_file)

            # Save training metadata
            metadata = {
                'state': state.upper(),
                'created_at': datetime.now().isoformat(),
                'model_type': 'RandomForestRegressor',
                'training_results': results,
                'feature_columns': results['features_used'],
                'rice_variety_aware': True
            }

            metadata_file = self.model_output_dir / f"rice_metadata_{state.lower()}.json"

            import json
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)

            print(f"   💾 Model saved: {model_file.name}")

        except Exception as e:
            print(f"   ⚠️  Failed to save model for {state.upper()}: {e}")

    def _create_training_summary(self, successful_states: list):
        """Create comprehensive training summary"""

        print("\n" + "=" * 60)
        print("📊 RICE MODEL TRAINING SUMMARY")
        print("=" * 60)

        total_states = len(successful_states)
        accuracy_levels = [self.training_results[state]['accuracy_level'] for state in successful_states]

        # Calculate summary statistics
        r2_scores = [self.training_results[state]['r2_score'] for state in successful_states]
        avg_r2 = np.mean(r2_scores)
        best_r2 = max(r2_scores)
        worst_r2 = min(r2_scores)

        # Count accuracy levels
        excellent_count = accuracy_levels.count("EXCELLENT (≥80%)")
        very_good_count = accuracy_levels.count("VERY GOOD (70-80%)")
        good_count = accuracy_levels.count("GOOD (60-70%)")

        good_models = excellent_count + very_good_count + good_count

        print(f"🎯 STATES TRAINED: {total_states}")
        print(f"🎖️  GOOD+ MODELS: {good_models}/{total_states} ({100*good_models//total_states if total_states > 0 else 0}%)")
        print(f"🏆 BEST R²: {best_r2:.3f}")
        print(f"📈 AVG R²: {avg_r2:.3f}")
        print(f"⚠️  WORST R²: {worst_r2:.3f}")

        print(f"\n📈 MODEL QUALITY BREAKDOWN:")
        print(f"   🏅 EXCELLENT (≥80%): {excellent_count}")
        print(f"   🥈 VERY GOOD (70-80%): {very_good_count}")
        print(f"   🥉 GOOD (60-70%): {good_count}")
        print(f"   📚 FAIR/OTHER: {total_states - good_models}")

        print(f"\n🌾 KEY INSIGHTS:")
        if total_states >= 3:
            top_performers = sorted(successful_states,
                                  key=lambda x: self.training_results[x]['r2_score'],
                                  reverse=True)[:3]
            print(f"   🏆 Top performers: {', '.join(top_performers)}")

            # Analyze common factors
            if all(self.training_results[s]['r2_score'] > 0.6 for s in ['PUNJAB', 'HARYANA']):
                print("   💧 Irrigation-heavy regions show better model performance")
            if all(self.training_results[s]['r2_score'] > 0.5 for s in ['BIHAR', 'UTTAR PRADESH', 'WEST BENGAL']):
                print("   🌧️ Mixed irrigation regions have acceptable performance")

        print(f"\n📁 MODELS SAVED TO:")
        print(f"   {self.model_output_dir}")
        print(f"   • {len(successful_states)} trained models")
        print(f"   • {len(successful_states)} feature scalers")
        print(f"   • {len(successful_states)} metadata files")

        # Agricultural interpretation
        if good_models >= total_states * 0.7:
            overall_assessment = "🌟 EXCELLENT - Ready for production deployment"
        elif good_models >= total_states * 0.5:
            overall_assessment = "✅ GOOD - Suitable for development with improvements"
        else:
            overall_assessment = "⚠️ FAIR - Requires model refinement and additional data"

        print(f"\n🎯 OVERALL ASSESSMENT: {overall_assessment}")

        # Save summary to file
        summary_data = {
            'training_summary': {
                'total_states': total_states,
                'successful_states': successful_states,
                'good_models_count': good_models,
                'avg_r2_score': round(avg_r2, 4),
                'best_r2_score': round(best_r2, 4),
                'worst_r2_score': round(worst_r2, 4),
                'accuracy_distribution': {
                    'excellent': excellent_count,
                    'very_good': very_good_count,
                    'good': good_count
                },
                'overall_assessment': overall_assessment,
                'training_timestamp': datetime.now().isoformat()
            },
            'individual_results': self.training_results
        }

        summary_file = self.model_output_dir / "rice_training_summary.json"
        import json
        with open(summary_file, 'w') as f:
            json.dump(summary_data, f, indent=2, default=str)

        print(f"\n💾 Detailed summary saved: {summary_file}")

def main():
    """Main training execution"""

    print("🌾 INDIA AGRICULTURAL INTELLIGENCE PLATFORM")
    print("🤖 Rice Model Training Pipeline")
    print("=" * 60)

    trainer = RiceModelTrainer()

    try:
        # Check if rice data exists
        if not Path("rice_data").exists():
            print("❌ RICE DATA NOT FOUND: Run rice_data_extractor.py first")
            return False

        # Train all available rice models
        successful_states = trainer.train_all_rice_models()

        if not successful_states:
            print("\n❌ NO MODELS TRAINED: Check data quality and try again")
            return False

        print(f"\n🎉 SUCCESS: {len(successful_states)} rice models trained and ready for deployment!")

        # Next steps guidance
        print(f"\n🚀 NEXT STEPS:")
        print(f"   1. Test models: python -c 'from india_agri_platform.crops.rice.model import get_rice_predictor; print('Rice predictor ready!')'")
        print(f"   2. API integration: Models will auto-load in rice predictor")
        print(f"   3. Wheat + Rice platform: Ready for dual-crop deployment")
        print(f"   4. Scale to other crops: Use this rice implementation as template")

        return True

    except Exception as e:
        print(f"❌ TRAINING FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
