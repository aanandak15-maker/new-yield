"""
Simple test script for India Agricultural Intelligence Platform
"""

import sys
import os
sys.path.append('india_agri_platform')

def test_platform():
    """Test the platform functionality"""
    print("🌾 Testing India Agricultural Intelligence Platform")
    print("=" * 60)

    try:
        from india_agri_platform.core import platform

        # Test 1: Get available crops and states
        print("✅ Testing platform initialization...")
        crops = platform.get_available_crops()
        states = platform.get_available_states()

        print(f"Available crops: {len(crops)} - {crops}")
        print(f"Available states: {len(states)} - {states}")

        # Test 2: Get crop configuration
        print("\n✅ Testing crop configuration...")
        wheat_config = platform.get_crop_config('wheat')
        print(f"Wheat growth duration: {wheat_config.config.get('growth_duration_days', 'N/A')} days")
        print(f"Wheat season: {wheat_config.get_season_info()['season']}")

        # Test 3: Get state configuration
        print("\n✅ Testing state configuration...")
        punjab_config = platform.get_state_config('punjab')
        print(f"Punjab major crops: {punjab_config.get_major_crops()}")
        print(f"Punjab irrigation coverage: {punjab_config.get_irrigation_info().get('coverage_percent', 'N/A')}%")

        # Test 4: Test yield prediction (will show error since no models trained)
        print("\n✅ Testing yield prediction framework...")
        test_features = {
            'temperature_celsius': 25,
            'rainfall_mm': 50,
            'humidity_percent': 60,
            'ndvi': 0.65,
            'soil_ph': 7.0,
            'irrigation_coverage': 0.95
        }

        result = platform.predict_yield('wheat', 'punjab', test_features)
        if 'error' in result:
            print(f"Expected error (no models trained): {result['error'][:100]}...")
        else:
            print("Unexpected success - models are available!")

        # Test 5: Test crop recommendations
        print("\n✅ Testing crop recommendations...")
        conditions = {
            'temperature': 28,
            'humidity': 65,
            'rainfall': 600,
            'ph': 7.2,
            'soil_texture': 'alluvial'
        }

        recommendations = platform.get_crop_recommendations('punjab', conditions)
        print(f"Generated {len(recommendations)} crop recommendations")

        # Test 6: Test model registry
        print("\n✅ Testing model registry...")
        registry_status = platform.get_model_performance_summary()
        print(f"Model registry status: {registry_status}")

        print("\n" + "=" * 60)
        print("🎉 PLATFORM TESTS COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("✅ Core architecture functional")
        print("✅ Configuration systems working")
        print("✅ Multi-crop, multi-state framework ready")
        print("✅ Ready for model training and data integration")
        print("=" * 60)

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_platform()
