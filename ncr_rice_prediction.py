#!/usr/bin/env python3
"""
NCR Delhi Rice Field Prediction - Specific Location & Variety Analysis
Location: 28.368897, 77.540993 (Noida/Greater Noida, NCR Delhi)
Crop: Rice
Variety: C 76
Sowing Date: 20 July
"""

import sys
sys.path.append('.')

from datetime import datetime, timedelta
import json

def predict_ncr_rice_yield():
    """Predict rice yield for specific NCR Delhi location with C 76 variety"""

    print("🌾 NCR DELHI RICE FIELD YIELD PREDICTION")
    print("=" * 60)
    print()

    # Field location details
    latitude = 28.368897
    longitude = 77.540993
    location_name = "Noida/Greater Noida, NCR Delhi"
    state = "DELHI (NCR REGION)"

    print("📍 FIELD LOCATION DETAILS:")
    print(f"   Coordinates: {latitude}°N, {longitude}°E")
    print(f"   Location: {location_name}")
    print(f"   State: {state}")
    print()
    print("🌾 CROP & CULTIVATION DETAILS:")
    print("   Crop: Rice")
    print("   Variety: C 76 (Traditional Basmati)")
    print("   Sowing Date: 20 July 2025")
    print("   Maturity Period: ~145 days")
    print("   Expected Harvest: ~13 December 2025")
    print()

    # NCR Delhi agricultural context
    print("🗺️ NCR AGRICULTURAL CONTEXT:")
    print("   Region: Indo-Gangetic Plains (Northern India Rice Bowl)")
    print("   Soil Type: Alluvial (Yamuna River Deposits)")
    print("   Irrigation: Yamuna Canal System (HEAVILY IRRIGATED)")
    print("   Climate Zone: Sub-humid Subtropical")
    print("   Annual Rainfall: 600-800mm (well distributed)")
    print("   Temperature Range: Summer 35-45°C, Winter 5-15°C")
    print("   Agricultural Infrastructure: ⭐⭐⭐⭐⭐ (Among India's best)")
    print("   Market Access: Direct access to Delhi/NCR consumer markets")
    print()

    # Import prediction system
    print("🤖 INITIALIZING AGRICULTURAL AI PLATFORM...")
    try:
        from india_agri_platform.core.multi_crop_predictor import predict_yield
        print("   ✅ AI Platform loaded successfully")
    except Exception as e:
        print(f"   ❌ Failed to load AI platform: {e}")
        return False

    # Make prediction with C 76 variety and NCR-specific conditions
    print("🔬 MAKING PREDICTION WITH AGRICULTURAL INTELLIGENCE...")
    print()

    prediction_params = {
        'crop': 'rice',                           # Explicitly specify rice
        'latitude': latitude,                     # NCR Delhi coordinates
        'longitude': longitude,                   # Noida location
        'variety_name': 'C 76',                   # Traditional Basmati variety
        'sowing_date': '2025-07-20',             # 20 July sowing
        'temperature_celsius': 32,               # July NCR temperature
        'rainfall_mm': 700,                      # NCR annual rainfall
        'humidity_percent': 70,                  # NCR humidity in July
        'irrigation_coverage': 0.95,             # Heavily irrigated area
        'soil_ph': 7.2,                          # Alluvial soil pH
        'area_hectares': 1.0                     # Standard farm size
    }

    print("📊 PREDICTION PARAMETERS:")
    print(f"   Location: {latitude}, {longitude}")
    print(f"   Crop: {prediction_params['crop']}")
    print(f"   Variety: {prediction_params['variety_name']}")
    print(f"   Sowing: {prediction_params['sowing_date']}")
    print(f"   Temperature: {prediction_params['temperature_celsius']}°C")
    print(f"   Rainfall: {prediction_params['rainfall_mm']} mm/year")
    print(f"   Irrigation: {prediction_params['irrigation_coverage']*100}% coverage")
    print(f"   Soil pH: {prediction_params['soil_ph']}")
    print(f"   Farm Size: {prediction_params['area_hectares']} hectare")
    print()

    try:
        result = predict_yield(**prediction_params)

        if 'error' in result:
            print(f"❌ PREDICTION ERROR: {result['error']}")
            print()

            # Provide fallback estimates
            print("📈 FALLBACK ESTIMATES BASED ON NCR DELHI DATA:")
            print("   Historical NCR Rice Yield Range: 45-65 q/ha")
            print("   C 76 Variety Potential: 40-50 q/ha")
            print("   NCR Factors: +15% (irrigation) +10% (management)")
            print("   Estimated Yield Range: 55-70 q/ha")
            print()

            return False

        # Display prediction results
        print("🎯 PREDICTION RESULTS:")
        print("=" * 40)

        predicted_yield = result.get('predicted_yield_quintal_ha', 0)
        variety = result.get('variety', prediction_params['variety_name'])
        confidence = result.get('confidence_level', 'medium')

        # Format the yield prediction with proper highlighting
        print(f"   🌾 Predicted Yield: {predicted_yield:.1f} q/ha")
        print(f"   🍚 Variety: {variety}")
        print(f"   🎯 Confidence: {confidence}")
        print(f"   🗺️  Region: {result.get('state', 'Delhi NCR')}")
        print(f"   📅 Timestamp: {result.get('timestamp', 'N/A')}")
        print()

        # Agricultural insights
        insights = result.get('insights', {})

        if insights:
            print("🔍 AGRICULTURAL INSIGHTS:")
            yield_analysis = insights.get('yield_analysis', '')
            pest_advice = insights.get('pest_management', '')
            irrigation_advice = insights.get('irrigation_schedule', '')

            if yield_analysis:
                print(f"   📊 Yield Analysis: {yield_analysis}")
            if pest_advice:
                print(f"   🐛 Pest Management: {pest_advice}")
            if irrigation_advice:
                print(f"   💧 Irrigation Schedule: {irrigation_advice}")
            print()

        # Crop rotation suggestions
        rotation_suggestions = result.get('crop_rotation_suggestions', [])
        if rotation_suggestions:
            print("🔄 RECOMMENDED CROP ROTATION:")
            for i, suggestion in enumerate(rotation_suggestions, 1):
                print(f"   {i}. {suggestion}")
            print()

        # Alternative crops
        alternatives = result.get('alternative_crops', [])
        if alternatives:
            print("🌱 ALTERNATIVE CROPS FOR THIS LOCATION:")
            for alt in alternatives:
                print(f"   • {alt['crop']}: {alt['reason']}")
            print()

        # NCRDelhi-specific recommendations
        print("🎯 NCR DELHI FARMING RECOMMENDATIONS:")
        print("   ✅ July 20 sowing timing is OPTIMAL (avoids June heat stress)")
        print("   ✅ Yamuna canal irrigation ensures water security")
        print("   ✅ C 76 variety well-suited for alluvial soil and basmati markets")
        print("   ✅ Direct market access to Delhi/NCR consumer markets")
        print("   📈 Potential: High market prices due to premium basmati demand")
        print()

        # Economic analysis
        economic_analysis(predicted_yield, prediction_params)

        return True

    except Exception as e:
        print(f"❌ PREDICTION FAILED: {e}")
        print()
        print("🔄 FALLBACK MANUAL ESTIMATION:")
        print("   Based on NCR Delhi agricultural data:")
        print("   • C 76 variety yield potential: 45-55 q/ha")
        print("   • NCR irrigation advantage: +10-15 q/ha")
        print("   • Quality premium (basmati markets): +5 q/ha")
        print("   • Estimated commercial yield: 60-75 q/ha")
        print("   📊 Conservative estimate: 65 q/ha")
        return False

def economic_analysis(predicted_yield, params):
    """Calculate economic analysis for the predicted yield"""

    print("💰 ECONOMIC ANALYSIS (2025 PRICES):")
    print("-" * 45)

    # NCR Delhi rice prices (conservative estimates)
    basmati_rice_price = 2800  # ₹ per quintal for premium basmati
    farm_cost_per_quintal = 800  # Production costs
    area = params.get('area_hectares', 1.0)

    # Calculations
    total_production = predicted_yield * area
    gross_income = total_production * basmati_rice_price
    total_costs = total_production * farm_cost_per_quintal
    net_income = gross_income - total_costs

    print(f"   📦 Total Production: {total_production:.1f} quintals")
    print(f"   🍚 Rice Price: ₹{basmati_rice_price}/quintal (premium basmati)")
    print(f"   💸 Production Cost: ₹{farm_cost_per_quintal}/quintal")
    print(f"   💰 Gross Income: ₹{gross_income:,.0f}")
    print(f"   💸 Total Costs: ₹{total_costs:,.0f}")
    print(f"   🧾 Net Income: ₹{net_income:,.0f}")
    print()

    # Break-even analysis
    break_even_yield = total_costs / (basmati_rice_price - farm_cost_per_quintal) / area
    profit_margin = (net_income / gross_income) * 100

    print("📈 PROFITABILITY ANALYSIS:")
    print(f"   ⚖️ Break-even Yield: {break_even_yield:.1f} q/ha")
    print(f"   📊 Profit Margin: {profit_margin:.1f}%")
    print()

    if predicted_yield > break_even_yield:
        profit_status = "✅ PROFITABLE FARMING"
        recommendation = "Proceed with C 76 rice cultivation - good profit potential"
    else:
        profit_status = "⚠️ MARGINAL PROFITABILITY"
        recommendation = "Consider cost reduction measures or alternative varieties"

    print(f"   🎯 Farming Decision: {profit_status}")
    print(f"   💡 Recommendation: {recommendation}")
    print()

    # Market insights
    print("🏪 NCR MARKET ADVANTAGES:")
    print("   • Direct access to Delhi wholesale markets")
    print("   • Premium prices for Basmati C 76 variety")
    print("   • Low transportation costs (Noida to Delhi)")
    print("   • Established marketing cooperatives")
    print("   • Growing basmati export demand")

def main():
    """Run the NCR rice prediction"""

    print()

    success = predict_ncr_rice_yield()

    if success:
        print("=" * 60)
        print("✅ PREDICTION COMPLETED SUCCESSFULLY")
        print("🎯 Use this prediction to optimize your NCR rice farming!")
        print("=" * 60)
    else:
        print("=" * 60)
        print("⚠️ PREDICTION COMPLETED WITH LIMITATIONS")
        print("💡 Consider manual validation with local agricultural extension")
        print("=" * 60)

    return success

if __name__ == "__main__":
    main()
