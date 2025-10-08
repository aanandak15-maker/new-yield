"""
Streamlined Yield Predictor Demo
Shows how the system works with minimal user input and maximum automation
"""

import sys
sys.path.append('india_agri_platform')

from india_agri_platform.core.streamlined_predictor import streamlined_predictor
import json

def demo_streamlined_prediction():
    """Demonstrate streamlined prediction with various scenarios"""

    print("🌾 INDIA AGRICULTURAL INTELLIGENCE PLATFORM")
    print("🚀 STREAMLINED YIELD PREDICTOR DEMO")
    print("=" * 80)
    print("Minimal User Input → Maximum Automation → Actionable Insights")
    print("=" * 80)

    # Test scenarios representing different farmer situations
    scenarios = [
        {
            "name": "Punjab Wheat Farmer",
            "crop": "wheat",
            "sowing_date": "2024-11-15",
            "latitude": 30.9010,  # Ludhiana, Punjab
            "longitude": 75.8573,
            "variety": "HD-2967"
        },
        {
            "name": "Haryana Rice Farmer",
            "crop": "rice",
            "sowing_date": "2024-06-20",
            "latitude": 29.0588,  # Rohtak, Haryana
            "longitude": 76.0856,
            "variety": None
        },
        {
            "name": "UP Sugarcane Farmer",
            "crop": "sugarcane",
            "sowing_date": "2024-02-15",
            "latitude": 26.8467,  # Lucknow, UP
            "longitude": 80.9462,
            "variety": "Co-0238"
        },
        {
            "name": "Bihar Maize Farmer",
            "crop": "maize",
            "sowing_date": "2024-07-10",
            "latitude": 25.5941,  # Patna, Bihar
            "longitude": 85.1376,
            "variety": None
        }
    ]

    print("\n📱 USER INPUT SIMULATION")
    print("-" * 50)
    print("Users only provide:")
    print("• 📍 Field location (GPS coordinates)")
    print("• 🌱 Crop type (dropdown selection)")
    print("• 📅 Sowing date (calendar picker)")
    print("• 🌾 Variety (optional dropdown)")
    print("\nEverything else is auto-fetched and calculated!")

    for i, scenario in enumerate(scenarios, 1):
        print(f"\n{'='*80}")
        print(f"🌾 SCENARIO {i}: {scenario['name'].upper()}")
        print('='*80)

        # Show user inputs
        print("👤 USER INPUTS:")
        print(f"   📍 Location: {scenario['latitude']:.4f}°N, {scenario['longitude']:.4f}°E")
        print(f"   🌱 Crop: {scenario['crop'].title()}")
        print(f"   📅 Sowing Date: {scenario['sowing_date']}")
        if scenario['variety']:
            print(f"   🌾 Variety: {scenario['variety']}")
        else:
            print("   🌾 Variety: Auto-selected")

        # Make prediction
        print("\n🤖 SYSTEM PROCESSING:")
        print("   🔍 Analyzing location → Determining state & district")
        print("   🌡️ Auto-fetching weather data from sowing date")
        print("   🛰️ Retrieving satellite data (NDVI, soil moisture)")
        print("   🌱 Calculating growth stage & crop health")
        print("   📊 Running ML prediction with confidence intervals")
        print("   💡 Generating actionable insights")

        result = streamlined_predictor.predict_yield_streamlined(
            crop_name=scenario['crop'],
            sowing_date=scenario['sowing_date'],
            latitude=scenario['latitude'],
            longitude=scenario['longitude'],
            variety_name=scenario['variety']
        )

        if 'error' in result:
            print(f"\n❌ PREDICTION ERROR: {result['error']}")
            continue

        # Display results
        print("\n🎯 PREDICTION RESULTS:")
        pred = result['prediction']
        print(f"   🎯 Expected Yield: {pred['expected_yield_quintal_ha']:.1f} q/ha")
        print(f"   📊 Confidence Interval: {pred['confidence_interval']}")
        print(f"   🌱 Growth Stage: {pred['growth_stage'].replace('_', ' ').title()}")
        print(f"   📅 Days Since Sowing: {pred['days_since_sowing']}")
        print(f"   🕐 Days to Harvest: {pred['estimated_harvest_days']}")

        # Display insights
        print("\n💡 ACTIONABLE INSIGHTS:")
        insights = result['insights']
        print(f"   🌱 Growth Status: {insights['growth_status']}")
        print(f"   💧 Irrigation: {insights['irrigation_advice']}")
        print(f"   🌤️ Weather: {insights['weather_alerts']}")
        print(f"   🦠 Disease Risk: {insights['disease_risk']}")
        print(f"   🚜 Harvest: {insights['harvest_readiness']}")
        print(f"   💰 Market: {insights['market_timing']}")

        # Show auto-fetched data summary
        print("\n🔄 AUTO-FETCHED DATA:")
        auto_data = result['auto_fetched_data']
        weather = auto_data['weather_summary']
        satellite = auto_data['satellite_indices']
        location = auto_data['location_analysis']

        print(f"   📍 Detected Location: {location['state'].title()}, {location.get('district', 'Unknown')}")
        print(f"   🌡️ Weather: {weather.get('average_temperature_c', 'N/A'):.1f}°C avg, {weather.get('total_rainfall_mm', 0):.0f}mm rain")
        print(f"   🛰️ Satellite: NDVI {satellite.get('ndvi', 0):.2f}, Vegetation {satellite.get('vegetation_health', 'unknown')}")

        print(f"\n⏰ Generated at: {result['timestamp'][:19].replace('T', ' ')}")

    print(f"\n{'='*80}")
    print("🎉 STREAMLINED PREDICTOR DEMO COMPLETE!")
    print('='*80)

    print("\n📊 SUMMARY:")
    print("✅ Minimal user input (4 fields max)")
    print("✅ Automatic location intelligence")
    print("✅ Real-time weather data integration")
    print("✅ Satellite imagery analysis")
    print("✅ Growth stage calculations")
    print("✅ ML-powered yield predictions")
    print("✅ Actionable farmer insights")
    print("✅ Confidence intervals for reliability")

    print("\n🎯 PERFECT FOR MOBILE APP:")
    print("• 📱 Simple farmer interface")
    print("• 🚀 Instant predictions")
    print("• 💡 Daily actionable advice")
    print("• 🌐 Works offline with cached data")
    print("• 📊 Tracks field performance over time")

    print("\n🚀 READY FOR PRODUCTION DEPLOYMENT!")
    print("The streamlined predictor transforms complex agricultural")
    print("science into simple, actionable insights for farmers!")

def demo_api_format():
    """Show the API format for integration"""

    print("\n🔌 API INTEGRATION EXAMPLE")
    print("-" * 50)

    # Sample API request
    api_request = {
        "crop": "wheat",
        "sowing_date": "2024-11-15",
        "latitude": 30.9010,
        "longitude": 75.8573,
        "variety": "HD-2967"
    }

    print("📤 API REQUEST (JSON):")
    print(json.dumps(api_request, indent=2))

    # Get sample response
    response = streamlined_predictor.predict_yield_streamlined(**api_request)

    print("\n📥 API RESPONSE (Simplified):")
    simplified_response = {
        "prediction": {
            "expected_yield_quintal_ha": response.get('prediction', {}).get('expected_yield_quintal_ha', 0),
            "confidence_interval": response.get('prediction', {}).get('confidence_interval', 'N/A'),
            "growth_stage": response.get('prediction', {}).get('growth_stage', 'unknown')
        },
        "insights": {
            "irrigation_advice": response.get('insights', {}).get('irrigation_advice', ''),
            "weather_alerts": response.get('insights', {}).get('weather_alerts', ''),
            "disease_risk": response.get('insights', {}).get('disease_risk', '')
        },
        "status": "success" if 'error' not in response else "error"
    }

    print(json.dumps(simplified_response, indent=2))

if __name__ == "__main__":
    demo_streamlined_prediction()
    demo_api_format()
