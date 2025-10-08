"""
Simple Accuracy Test for India Agricultural Intelligence Platform
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import zipfile

def main():
    print("🧪 INDIA AGRICULTURAL INTELLIGENCE PLATFORM - ACCURACY TEST")
    print("=" * 70)

    try:
        # Load data
        print("Loading training data...")
        with zipfile.ZipFile('agriyield-2025.zip', 'r') as z:
            with z.open('train.csv') as f:
                df = pd.read_csv(f)

        print(f"Loaded {len(df)} records")

        # Prepare features
        features = ['temperature', 'humidity', 'rainfall', 'ndvi', 'soil_ph', 'sand_pct', 'organic_matter']
        target = 'yield'

        # Clean data
        df_clean = df.dropna(subset=features + [target])
        print(f"After cleaning: {len(df_clean)} records")

        # Split data
        X = df_clean[features]
        y = df_clean[target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train model
        print("Training Random Forest model...")
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_train_scaled, y_train)

        # Make predictions
        y_pred = model.predict(X_test_scaled)

        # Calculate metrics
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)

        # Cross-validation
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()

        print("\n📊 MODEL PERFORMANCE RESULTS")
        print("-" * 40)
        print(".4f")
        print(".2f")
        print(".2f")
        print(".3f")

        # Accuracy interpretation
        if r2 > 0.9:
            accuracy_level = "EXCELLENT (>90%)"
        elif r2 > 0.8:
            accuracy_level = "VERY GOOD (80-90%)"
        elif r2 > 0.7:
            accuracy_level = "GOOD (70-80%)"
        elif r2 > 0.6:
            accuracy_level = "FAIR (60-70%)"
        else:
            accuracy_level = "NEEDS IMPROVEMENT (<60%)"

        print(f"\nACCURACY LEVEL: {accuracy_level}")

        # Test real scenarios
        print("\n🌍 REAL-WORLD PREDICTIONS")
        print("-" * 40)

        scenarios = [
            ("Punjab Wheat", [25, 60, 50, 0.65, 7.0, 40, 2.5]),
            ("Punjab Rice", [28, 75, 150, 0.70, 6.8, 35, 2.0]),
            ("Haryana Cotton", [32, 55, 30, 0.60, 7.2, 45, 1.8])
        ]

        for name, features in scenarios:
            features_scaled = scaler.transform([features])
            prediction = model.predict(features_scaled)[0]
            print(".1f")

        print("\n" + "=" * 70)

        if r2 > 0.8:
            print("🎉 SUCCESS: Platform achieves production-ready accuracy!")
            print("✅ Ready for commercial deployment")
        else:
            print("⚠️ CAUTION: Platform needs accuracy improvements")
            print("⚠️ Additional data collection recommended")

        print("=" * 70)

        # Save results
        with open("accuracy_test_results.txt", "w") as f:
            f.write("INDIA AGRICULTURAL INTELLIGENCE PLATFORM - ACCURACY TEST RESULTS\n")
            f.write(f"R² Score: {r2:.4f}\n")
            f.write(f"RMSE: {rmse:.2f}\n")
            f.write(f"MAE: {mae:.2f}\n")
            f.write(f"Accuracy Level: {accuracy_level}\n")

        print("📄 Results saved to accuracy_test_results.txt")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
