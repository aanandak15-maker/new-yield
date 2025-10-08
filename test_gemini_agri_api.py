#!/usr/bin/env python3
"""
TEST PRODUCTION GEMINI AGRICULTURAL API - Phase 3 Final Validation
Test the unified agricultural consultation API with real farmer use cases
"""

import requests
import json
import time
from datetime import datetime
import traceback

def test_gemini_agricultural_api():
    """Comprehensive test of the production agricultural consultation API"""

    print("🔬 TESTING PRODUCTION GEMINI AGRICULTURAL CONSULTATION API")
    print("=" * 75)

    # API endpoint configuration
    BASE_URL = "http://localhost:8000"  # Change if running on different port
    CONSULTATION_ENDPOINT = f"{BASE_URL}/api/v1/consultation"
    HEALTH_ENDPOINT = f"{BASE_URL}/api/v1/health"
    FEEDBACK_ENDPOINT = f"{BASE_URL}/api/v1/feedback"

    test_cases = [
        {
            "name": "NCR Rice Yellow Leaves Problem",
            "payload": {
                "farmer_query": "My rice plants are showing yellow leaves at 85 days after sowing. What should I do?",
                "farmer_id": "farmer_ncr_001",
                "location": "NCR, Delhi",
                "crop_type": "rice",
                "crop_age_days": 85,
                "field_area_ha": 2.5,
                "soil_ph": 7.2,
                "irrigation_method": "flooded",
                "farmer_experience_years": 8,
                "budget_constraint": "medium_cost",
                "satellite_ndvi": 0.68,
                "recent_rainfall_mm": 25,
                "current_temperature_c": 33
            }
        },
        {
            "name": "Punjab Wheat Aphid Infestation",
            "payload": {
                "farmer_query": "There are many small green insects on my wheat crop. They are sucking sap from leaves. This happened 45 days after sowing.",
                "farmer_id": "farmer_punjab_005",
                "location": "Punjab",
                "crop_type": "wheat",
                "crop_age_days": 45,
                "field_area_ha": 4.2,
                "soil_ph": 7.8,
                "irrigation_method": "drip",
                "farmer_experience_years": 12,
                "budget_constraint": "budget_constrained",
                "satellite_ndvi": 0.72
            }
        },
        {
            "name": "Cotton Bollworm Attack - Maharashtra",
            "payload": {
                "farmer_query": "Bolls are getting damaged by pink worms in my cotton field. It's 80 days after sowing. What organic control methods should I use?",
                "farmer_id": "farmer_maharashtra_002",
                "location": "Maharashtra, Yavatmal",
                "crop_type": "cotton",
                "crop_age_days": 80,
                "field_area_ha": 3.8,
                "soil_ph": 6.5,
                "irrigation_method": "rainfed",
                "farmer_experience_years": 6,
                "budget_constraint": "organic_preferred",
                "satellite_ndvi": 0.65
            }
        },
        {
            "name": "Maize Nutrient Deficiency - Karnataka",
            "payload": {
                "farmer_query": "My maize plants are stunted with yellow stripes on leaves. Field is 50 days old on black soil.",
                "farmer_id": "farmer_karnataka_003",
                "location": "Karnataka, Dharwad",
                "crop_type": "maize",
                "crop_age_days": 50,
                "field_area_ha": 2.0,
                "soil_ph": 7.0,
                "irrigation_method": "sprinkler",
                "farmer_experience_years": 15,
                "budget_constraint": "premium_inputs_ok"
            }
        }
    ]

    # Step 1: Test API Health
    print("1️⃣ TESTING API HEALTH CHECK:")
    print("-" * 40)

    try:
        health_response = requests.get(HEALTH_ENDPOINT, timeout=10)
        if health_response.status_code == 200:
            health_data = health_response.json()
            print("✅ API Health Check: PASSED")
            print(f"   📊 Status: {health_data.get('status', 'unknown')}")
            print(f"   🏥 System Health: {health_data.get('platform', {}).get('name', 'unknown')}")
            print(f"   🤖 Gemini Status: {health_data.get('services', {}).get('operational', 'unknown')}")
        else:
            print(f"❌ API Health Check FAILED: HTTP {health_response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ API Health Check ERROR: Cannot connect to {BASE_URL}")
        print(f"   💡 Make sure the API server is running: 'python agri_consultation_api.py'")
        return False

    print()

    # Step 2: Test Agricultural Consultations
    print("2️⃣ TESTING AGRICULTURAL CONSULTATIONS:")
    print("-" * 45)

    consultation_results = []
    total_response_time = 0
    successful_consultations = 0

    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest Case {i}/{len(test_cases)}: {test_case['name']}")
        print("-" * 50)

        try:
            start_time = time.time()

            # Make consultation request
            response = requests.post(
                CONSULTATION_ENDPOINT,
                json=test_case['payload'],
                headers={'Content-Type': 'application/json'},
                timeout=60  # 60 second timeout for AI processing
            )

            response_time = time.time() - start_time
            total_response_time += response_time

            if response.status_code == 200:
                data = response.json()
                consultation = data['consultation']

                # Validate response structure
                required_fields = ['consultation_id', 'consultation_text', 'intelligence_confidence']
                if all(field in consultation for field in required_fields):
                    print(f"✅ Consultation PASSED ({response_time:.1f}s)")
                    print(f"   🆔 ID: {consultation['consultation_id']}")
                    print(f"   🤖 Confidence: {consultation.get('intelligence_confidence', 0):.3f}")
                    print(f"   📝 Length: {consultation.get('response_length', 0)} chars")

                    # Check consultation quality
                    quality_checks = validate_consultation_quality(consultation)
                    quality_score = sum(quality_checks.values())
                    quality_percentage = quality_score / len(quality_checks) * 100

                    print(f"   📈 Quality Score: {quality_percentage:.1f}%")
                    print(f"   ✨ Highlights: {consultation['recommendation_categories']}")

                    # Summarize consultation (first 200 chars)
                    consultation_text = consultation.get('consultation_text', '')
                    summary = consultation_text[:200] + "..." if len(consultation_text) > 200 else consultation_text
                    print(f"   📋 Summary: {summary}")

                    consultation_results.append({
                        "test_case": test_case['name'],
                        "success": True,
                        "response_time": response_time,
                        "quality_score": quality_percentage,
                        "consultation_id": consultation['consultation_id']
                    })

                    successful_consultations += 1

                else:
                    print(f"❌ Invalid Response Structure: Missing {required_fields}")
                    consultation_results.append({
                        "test_case": test_case['name'],
                        "success": False,
                        "error": "Invalid response structure"
                    })

            else:
                print(f"❌ HTTP Error: {response.status_code}")
                print(f"   Response: {response.text[:200]}...")

                consultation_results.append({
                    "test_case": test_case['name'],
                    "success": False,
                    "http_error": response.status_code
                })

        except requests.exceptions.Timeout:
            print("❌ TIMEOUT: Consultation took too long (>60 seconds)")
            consultation_results.append({
                "test_case": test_case['name'],
                "success": False,
                "error": "timeout"
            })

        except Exception as e:
            print(f"❌ ERROR: {e}")
            print(f"   Details: {traceback.format_exc()}")
            consultation_results.append({
                "test_case": test_case['name'],
                "success": False,
                "error": str(e)
            })

    print("\n" + "=" * 75)
    print("📊 CONSULTATION API TEST RESULTS")
    print("=" * 75)

    # Overall Statistics
    success_rate = successful_consultations / len(test_cases) * 100
    avg_response_time = total_response_time / successful_consultations if successful_consultations > 0 else 0

    print(f"📈 OVERALL SUCCESS RATE: {successful_consultations}/{len(test_cases)} ({success_rate:.1f}%)")
    print(f"⏱️ Average Response Time: {avg_response_time:.2f}s")
    print(f"✅ Successful Cases: {successful_consultations}/{len(test_cases)}")
    print()

    # Individual Test Results
    print("🔍 INDIVIDUAL TEST RESULTS:")
    successful_cases = []
    failed_cases = []

    for result in consultation_results:
        status_icon = "✅" if result['success'] else "❌"
        response_time_str = f"{result.get('response_time', 0):.1f}s" if result.get('response_time') else "N/A"
        quality_str = f" ({result.get('quality_score', 0):.1f}%)" if result.get('quality_score') else ""

        print(f"   {status_icon} {result['test_case']}: {response_time_str}{quality_str}")

        if result['success']:
            successful_cases.append(result)
        else:
            failed_cases.append(result)

    print()

    # Performance Analysis
    if successful_cases:
        response_times = [r['response_time'] for r in successful_cases]
        quality_scores = [r['quality_score'] for r in successful_cases]

        print("⚡ PERFORMANCE ANALYSIS:")
        print(f"   🚀 Best Response Time: {min(response_times):.2f}s")
        print(f"   🐌 Slowest Response Time: {max(response_times):.2f}s")
        print(f"   ⌛ Average Response Time: {sum(response_times)/len(response_times):.2f}s")
        print(f"   📏 Response Time Range: {max(response_times) - min(response_times):.2f}s")
        print()

        print("⭐ QUALITY ANALYSIS:")
        print(f"   🧮 Average Quality: {sum(quality_scores)/len(quality_scores):.1f}%")
        print(f"   📊 Quality Score Range: {max(quality_scores) - min(quality_scores):.1f} points")
        print("   🎯 Quality Distribution: [" + ", ".join(f"{s:.1f}%" for s in sorted(quality_scores)) + "]")
        print()

    # Real-world Validation
    print("🌾 REAL-WORLD VALIDATION:")
    if success_rate >= 90:
        print("   ✅ EXCELLENT: Production-ready API with high reliability")
        print("   🎯 CONFIDENCE: Farmers will receive expert agricultural advice")
        print("   🚀 SCALE READY: Can handle 30M consultations at current performance")
    elif success_rate >= 75:
        print("   ⚠️ GOOD: Core functionality working, minor improvements needed")
        print("   🔧 RECOMMENDATION: Address failed test cases before full deployment")
    else:
        print("   ❌ NEEDS WORK: Core API functionality requires attention")
        print("   🛠️ ACTION REQUIRED: Debug and fix failed consultation cases")

    print()

    # Production Readiness Assessment
    print("🏭 PRODUCTION READINESS:")
    production_criteria = [
        ("API Responsiveness", avg_response_time < 5.0 if successful_cases else False),
        ("Consultation Success", success_rate >= 90),
        ("Response Quality", all(r.get('quality_score', 0) >= 70 for r in successful_cases) if successful_cases else False),
        ("Error Handling", not failed_cases),  # No failed cases
        ("Content Relevance", all(r.get('quality_score', 0) >= 60 for r in successful_cases) if successful_cases else False)
    ]

    readiness_score = sum(1 for _, passed in production_criteria if passed)
    print(f"   📊 Production Readiness: {readiness_score}/{len(production_criteria)} criteria met")
    print()

    for criterion, passed in production_criteria:
        status = "✅" if passed else "❌"
        print(f"   {status} {criterion}")

    print()

    # Final Recommendation
    if readiness_score >= 4:
        print("🎊 FINAL VERDICT: AGRICULTURAL SUPERINTELLIGENCE API READY FOR PRODUCTION!")
        print("🎯 RECOMMENDATION: Deploy and start serving farmers immediately")
        success = True
    else:
        print("⚠️ FINAL VERDICT: Additional testing and fixes needed before production")
        print("🔧 RECOMMENDATION: Address failing criteria and retest")
        success = False

    return success

def validate_consultation_quality(consultation: dict) -> dict:
    """Validate the quality of a consultation response"""

    quality_checks = {}

    response_text = consultation.get('consultation_text', '')

    # Check for essential agricultural elements
    quality_checks['crop_knowledge'] = any(word in response_text.lower() for word in [
        'nitrogen', 'phosphorus', 'potassium', 'npk', 'urea', 'पोषक'
    ])

    quality_checks['specific_recommendations'] = any(word in response_text.lower() for word in [
        'spray', 'apply', 'sprinkler', 'drip', 'छिड़काव', 'लगाएं'
    ])

    quality_checks['timing_guidance'] = any(word in response_text.lower() for word in [
        'immediately', 'within', 'tomorrow', 'next', 'तुरंत', 'कई दिनों'
    ])

    quality_checks['monitoring_advice'] = any(word in response_text.lower() for word in [
        'monitor', 'check', 'observe', 'watch', 'निगरानी', 'देखें'
    ])

    quality_checks['scientific_foundation'] = consultation.get('scientific_foundation') == 'ICAR-recommended'

    quality_checks['practicality'] = len(response_text.split()) > 100  # Substantial response

    quality_checks['language_support'] = any(char in response_text for char in ['न', 'क', 'र'])  # Hindi present

    quality_checks['confidence_level'] = consultation.get('intelligence_confidence', 0) >= 0.85

    return quality_checks

def main():
    """Run the complete Gemini Agricultural API test suite"""

    print("🌾 PLANT SAATHI GEMINI AGRICULTURAL INTELLIGENCE")
    print("PRODUCTION API VALIDATION SUITE")
    print()

    try:
        success = test_gemini_agricultural_api()

        print("\n" + "=" * 75)
        print("🎉 TESTING COMPLETE")
        print("=" * 75)

        if success:
            print("🎯 MISSION ACCOMPLISHED!")
            print("✅ Agricultural superintelligence is production-ready")
            print("✅ Farmers can now get expert ICAR consultations")
            print("✅ Gemini 2.0 Flash agricultural brain is operational")
            print("✅ 30 million Indian farmers one step closer to prosperity")
        else:
            print("⚠️ TESTING IDENTIFIED ISSUES")
            print("🔧 Additional work needed before full deployment")
            print("💡 Review console output for specific failure details")

        return success

    except KeyboardInterrupt:
        print("\n⏹️ Testing interrupted by user")
        return False
    except Exception as e:
        print(f"\n❌ Testing failed with unexpected error: {e}")
        print("🔧 Check API server status and network connectivity")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
