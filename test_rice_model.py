#!/usr/bin/env python3
"""
Test Rice Ensemble Model Predictions
"""

import pickle
import numpy as np

def load_and_test_rice_model():
    """Load trained model and make test predictions"""

    print("🧪 Testing Rice Ensemble Model...")
    print("=" * 30)

    # Load model
    with open('models/advanced_models/rice_ensemble_model_v1.pkl', 'rb') as f:
        model_data = pickle.load(f)

    print("✅ Model loaded successfully")

    # Create test inputs
    test_cases = [
        {
            "description": "Standard NCR conditions",
            "features": np.array([1.0, 75, 120, 6.8, 60])  # area, days, rainfall, ph, fertilizer
        },
        {
            "description": "Early season Punjab",
            "features": np.array([2.5, 45, 80, 7.2, 40])
        },
        {
            "description": "High rainfall Tamil Nadu",
            "features": np.array([1.5, 90, 200, 5.8, 80])
        }
    ]

    feature_names = model_data['feature_columns']
    weights = model_data['weights']

    for i, test_case in enumerate(test_cases):
        print(f"\nTest Case {i+1}: {test_case['description']}")

        # Individual model predictions
        rf_pred = model_data['models']['random_forest'].predict(test_case['features'].reshape(1, -1))[0]
        xgb_pred = model_data['models']['xgboost'].predict(test_case['features'].reshape(1, -1))[0]
        cat_pred = model_data['models']['catboost'].predict(test_case['features'].reshape(1, -1))[0]

        # Ensemble prediction
        ensemble_pred = (
            weights['random_forest'] * rf_pred +
            weights['xgboost'] * xgb_pred +
            weights['catboost'] * cat_pred
        )

        print(f"Individual Predictions:")
        print(f"  Random Forest: {rf_pred:.1f} q/ha")
        print(f"  XGBoost: {xgb_pred:.1f} q/ha")
        print(f"  CatBoost: {cat_pred:.1f} q/ha")
        print(f"Ensemble Prediction: {ensemble_pred:.1f} q/ha")

    print(f"\n✅ Model testing completed!")
    print(f"🎯 Plant Saathi AI rice model operating at {model_data['performance']['accuracy_percentage']:.1f}% accuracy")

if __name__ == "__main__":
    load_and_test_rice_model()
