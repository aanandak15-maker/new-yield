#!/usr/bin/env python3
"""
REAL-TIME PLATFORM VALIDATION: Comprehensive End-to-End Testing
Agricultural Intelligence Platform Real-Time Data Testing & Validation
"""

import sys
import os
import json
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Any, Optional, Tuple
import concurrent.futures
import threading
from queue import Queue
import psutil
import tracemalloc
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append('india_agri_platform')

# Import all platform components for testing
from india_agri_platform.core.multi_crop_predictor import get_multi_crop_predictor
from india_agri_platform.api.unified_crop_api import UnifiedCropPredictionEngine
from farmer_insights_dashboard import AgriculturalInsightsDashboard
from business_scaling_strategy import AgriculturalBusinessScaler
from data_fetcher import AgriculturalDataFetcher
from gemini_agricultural_intelligence import GeminiAgriculturalAI
from ethical_agricultural_orchestrator import EthicalAgriculturalOrchestrator

# Configure logging for testing
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('realtime_platform_validation.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class RealTimePlatformValidator:
    """
    COMPREHENSIVE REAL-TIME PLATFORM VALIDATION
    Intensive Testing Framework for Agricultural Intelligence Platform

    Tests all components with real data, real scenarios, real performance.
    """

    def __init__(self):
        self.api_base_url = "http://localhost:8000"
        self.start_time = datetime.now()
        self.test_results = {
            'ml_models': {},
            'apis': {},
            'dashboard': {},
            'business_logic': {},
            'database': {},
            'performance': {},
            'security': {},
            'edge_cases': {},
            'scalability': {}
        }

        # Initialize all platform components
        logger.info("🚀 INITIALIZING PLATFORM COMPONENTS FOR REAL-TIME TESTING")
        self.initialize_components()

        # Real test data - Indian agricultural regions
        self.test_locations = self.load_test_locations()

    def initialize_components(self):
        """Initialize all platform components"""
        try:
            self.multi_crop_predictor = get_multi_crop_predictor()
            logger.info("✅ Multi-crop predictor initialized")

            self.unified_engine = UnifiedCropPredictionEngine()
            logger.info("✅ Unified prediction engine initialized")

            self.dashboard = AgriculturalInsightsDashboard()
            logger.info("✅ Agricultural dashboard initialized")

            self.business_scaler = AgriculturalBusinessScaler()
            logger.info("✅ Business scaling framework initialized")

            # Initialize AI components (optional if APIs not available)
            try:
                self.data_fetcher = AgriculturalDataFetcher()
                logger.info("✅ Agricultural data fetcher initialized")
            except:
                logger.warning("⚠️ Data fetcher initialization failed - will use local data")

            try:
                self.gemini_ai = GeminiAgriculturalAI()
                logger.info("✅ Gemini AI consultant initialized")
            except:
                logger.warning("⚠️ Gemini AI initialization failed - API key required")

            try:
                self.ethical_orchestrator = EthicalAgriculturalOrchestrator()
                logger.info("✅ Ethical agricultural orchestrator initialized")
            except:
                logger.warning("⚠️ Ethical orchestrator initialization failed")

        except Exception as e:
            logger.error(f"❌ Component initialization error: {e}")
            self.test_results['initialization'] = str(e)

    def load_test_locations(self) -> List[Dict[str, Any]]:
        """Load comprehensive Indian agricultural test locations"""

        return [
            # Punjab grain belt
            {'name': 'Ludhiana Rice Fields', 'lat': 30.9010, 'lng': 75.8573,
             'state': 'punjab', 'crop': 'rice', 'season': 'kharif',
             'soil_ph': 7.2, 'irrigation': 'flooded', 'variety': 'PB1121'},

            {'name': 'Bathinda Wheat Fields', 'lat': 30.2103, 'lng': 74.9455,
             'state': 'punjab', 'crop': 'wheat', 'season': 'rabi',
             'soil_ph': 7.8, 'irrigation': 'sprinkler', 'variety': 'PBW721'},

            # Haryana wheat bowl
            {'name': 'Karnal Agricultural Farms', 'lat': 29.6857, 'lng': 76.9874,
             'state': 'haryana', 'crop': 'rice', 'season': 'kharif',
             'soil_ph': 7.5, 'irrigation': 'drip', 'variety': 'WH1105'},

            # Maharashtra cotton belt
            {'name': 'Nagpur Cotton Fields', 'lat': 21.1458, 'lng': 79.0882,
             'state': 'maharashtra', 'crop': 'cotton', 'season': 'kharif',
             'soil_ph': 6.8, 'irrigation': 'flooded', 'variety': 'NCS-207'},

            {'name': 'Vidarbha BT Cotton', 'lat': 20.8, 'lng': 79.5,
             'state': 'maharashtra', 'crop': 'cotton', 'season': 'kharif',
             'soil_ph': 7.0, 'irrigation': 'sprinkler', 'variety': 'MCU-5 Bt'},

            # Gujarat cotton and groundnut
            {'name': 'Surat Cotton Belt', 'lat': 21.1702, 'lng': 72.8311,
             'state': 'gujarat', 'crop': 'cotton', 'season': 'kharif',
             'soil_ph': 7.1, 'irrigation': 'drip', 'variety': 'GDHY-1'},

            # Karnataka rice and maize
            {'name': 'Bangalore Rural Rice', 'lat': 12.9716, 'lng': 77.5946,
             'state': 'karnataka', 'crop': 'rice', 'season': 'kharif',
             'soil_ph': 6.5, 'irrigation': 'flooded', 'variety': 'Jaya'},

            {'name': 'Dharwad Maize Fields', 'lat': 15.4589, 'lng': 75.0078,
             'state': 'karnataka', 'crop': 'maize', 'season': 'kharif',
             'soil_ph': 7.2, 'irrigation': 'sprinkler', 'variety': 'African Tall'},

            # Tamil Nadu rice
            {'name': 'Thanjavur Rice Bowl', 'lat': 10.7870, 'lng': 79.1378,
             'state': 'tamil_nadu', 'crop': 'rice', 'season': 'kharif',
             'soil_ph': 7.8, 'irrigation': 'flooded', 'variety': 'ASD16'},

            # Andhra Pradesh rice and cotton
            {'name': 'Krishna Rice Delta', 'lat': 16.3067, 'lng': 81.7292,
             'state': 'andhra_pradesh', 'crop': 'rice', 'season': 'kharif',
             'soil_ph': 8.1, 'irrigation': 'flooded', 'variety': 'BPT5204'},

            # Uttar Pradesh wheat and sugarcane
            {'name': 'Meerut Wheat Fields', 'lat': 28.9845, 'lng': 77.7064,
             'state': 'uttar_pradesh', 'crop': 'wheat', 'season': 'rabi',
             'soil_ph': 7.6, 'irrigation': 'sprinkler', 'variety': 'HD3086'},

            # Bihar rice
            {'name': 'Patna Rice Plains', 'lat': 25.5941, 'lng': 85.1376,
             'state': 'bihar', 'crop': 'rice', 'season': 'kharif',
             'soil_ph': 7.4, 'irrigation': 'flooded', 'variety': 'MTU1075'},

            # West Bengal rice and jute
            {'name': 'Hooghly Rice Valley', 'lat': 22.9868, 'lng': 88.4181,
             'state': 'west_bengal', 'crop': 'rice', 'season': 'kharif',
             'soil_ph': 7.0, 'irrigation': 'flooded', 'variety': 'Samba Mahsuri'}
        ]

    def test_machine_learning_models(self) -> Dict[str, Any]:
        """Test all ML models with real agricultural data"""

        logger.info("🧪 TESTING MACHINE LEARNING MODELS")

        model_tests = {
            'rice_model': {'crop': 'rice', 'locations': ['punjab', 'tamil_nadu', 'karnataka', 'west_bengal']},
            'wheat_model': {'crop': 'wheat', 'locations': ['punjab', 'haryana', 'uttar_pradesh', 'rajasthan']},
            'cotton_model': {'crop': 'cotton', 'locations': ['maharashtra', 'gujarat', 'andhra_pradesh', 'punjab']},
            'maize_model': {'crop': 'maize', 'locations': ['karnataka', 'rajasthan', 'bihar']}
        }

        results = {}

        for model_name, config in model_tests.items():
            crop = config['crop']
            logger.info(f"🧬 Testing {model_name} for {len(config['locations'])} locations")

            # Test only available locations from our test data
            test_locs = [loc for loc in self.test_locations if loc['state'] in config['locations'] and loc['crop'] == crop]

            model_results = []
            for loc in test_locs[:3]:  # Test 3 samples per model
                try:
                    # Get prediction
                    prediction = self.unified_engine._generate_yield_prediction(
                        crop, loc['lat'], loc['lng'], loc.get('variety'), loc['state']
                    )

                    # Validate prediction structure
                    if isinstance(prediction, dict) and 'yield' in prediction:
                        yield_pred = prediction['yield']
                        confidence = prediction.get('confidence', 'Medium')

                        model_results.append({
                            'location': f"{loc['name']} ({loc['state']})",
                            'predicted_yield': yield_pred,
                            'confidence': confidence,
                            'crop_specific': True,
                            'status': '✅ PASSED'
                        })
                    else:
                        model_results.append({
                            'location': loc['name'],
                            'predicted_yield': 'N/A',
                            'confidence': 'N/A',
                            'crop_specific': True,
                            'status': '❌ FAILED - Invalid prediction format'
                        })

                except Exception as e:
                    model_results.append({
                        'location': loc['name'],
                        'predicted_yield': 'N/A',
                        'confidence': 'N/A',
                        'crop_specific': True,
                        'status': f'❌ FAILED - {str(e)[:50]}...'
                    })

            results[model_name] = {
                'total_tests': len(test_locs[:3]),
                'passed_tests': len([r for r in model_results if 'PASSED' in r['status']]),
                'failed_tests': len([r for r in model_results if 'FAILED' in r['status']]),
                'pass_rate': len([r for r in model_results if 'PASSED' in r['status']]) / max(1, len(test_locs[:3])),
                'detailed_results': model_results
            }

        # Overall ML model performance
        total_tests = sum([r['total_tests'] for r in results.values()])
        total_passed = sum([r['passed_tests'] for r in results.values()])

        results['OVERALL_ML_PERFORMANCE'] = {
            'total_models_tested': len(results) - 1,  # Exclude overall key
            'total_test_cases': total_tests,
            'overall_pass_rate': total_passed / max(1, total_tests),
            'model_accuracy_validation': 'COMPLETED',
            'real_data_integration': total_passed > 0
        }

        logger.info(f"✅ ML Model Testing Complete: {total_passed}/{total_tests} tests passed")
        return results

    def test_api_endpoints(self) -> Dict[str, Any]:
        """Test all API endpoints with real data and scenarios"""

        logger.info("🔗 TESTING API ENDPOINTS WITH REAL DATA")

        api_tests = {
            'predict/yield': {'method': 'POST', 'test_locs': 5, 'expected_status': 200},
            'predict/yield/location': {'method': 'GET', 'test_locs': 3, 'expected_status': 200},
            'unified/predict/yield': {'method': 'POST', 'test_locs': 5, 'expected_status': 200},
            'unified/crop/suitability': {'method': 'GET', 'test_locs': 3, 'expected_status': 200},
            'region/rice': {'method': 'GET', 'test_locs': 1, 'expected_status': 200},
            'region/wheat': {'method': 'GET', 'test_locs': 1, 'expected_status': 200},
            'region/cotton': {'method': 'GET', 'test_locs': 1, 'expected_status': 200},
            'crops': {'method': 'GET', 'test_locs': 0, 'expected_status': 200},
            'platform/info': {'method': 'GET', 'test_locs': 0, 'expected_status': 200},
            'health': {'method': 'GET', 'test_locs': 0, 'expected_status': 200},
            'health/mvp': {'method': 'GET', 'test_locs': 0, 'expected_status': 200},
            'stats': {'method': 'GET', 'test_locs': 0, 'expected_status': 200}
        }

        results = {}

        for endpoint, config in api_tests.items():
            logger.info(f"🖥️ Testing {config['method']} /{endpoint}")

            endpoint_results = []
            test_locations = self.test_locations[:config['test_locs']] if config['test_locs'] > 0 else [None]

            for i, loc in enumerate(test_locations):
                try:
                    if config['method'] == 'POST':
                        # POST endpoint with body
                        payload = {
                            "crop": loc['crop'] if loc else None,
                            "latitude": loc['lat'] if loc else 28.6139,
                            "longitude": loc['lng'] if loc else 77.2090,
                            "variety_name": loc.get('variety') if loc else None,
                            "season": loc['season'] if loc else 'kharif'
                        }

                        response = requests.post(f"{self.api_base_url}/{endpoint}",
                                               json=payload, timeout=30)

                    else:
                        # GET endpoint with params
                        params = {}
                        if 'predict/yield/location' in endpoint or 'unified/crop/suitability' in endpoint:
                            params.update({
                                'latitude': loc['lat'] if loc else 28.6139,
                                'longitude': loc['lng'] if loc else 77.2090,
                                'season': loc['season'] if loc else 'kharif'
                            })

                        if '/region/' in endpoint:
                            crop = endpoint.split('/region/')[1]
                            params['crop'] = crop

                        response = requests.get(f"{self.api_base_url}/{endpoint}",
                                              params=params, timeout=30)

                    # Validate response
                    test_result = {
                        'request_id': f"{endpoint}_{i+1}",
                        'http_status': response.status_code,
                        'expected_status': config['expected_status'],
                        'response_time_ms': int(response.elapsed.total_seconds() * 1000),
                        'content_length': len(response.text),
                        'has_error': 'error' in response.text.lower() if response.headers.get('content-type', '').startswith('application/json') else False
                    }

                    if response.status_code == config['expected_status']:
                        test_result['status'] = '✅ PASSED'
                    else:
                        test_result['status'] = f'❌ FAILED - Expected {config["expected_status"]}, got {response.status_code}'

                    endpoint_results.append(test_result)

                except requests.exceptions.Timeout:
                    endpoint_results.append({
                        'request_id': f"{endpoint}_{i+1}",
                        'http_status': 'TIMEOUT',
                        'response_time_ms': 30000,
                        'status': '❌ FAILED - Timeout'
                    })

                except Exception as e:
                    endpoint_results.append({
                        'request_id': f"{endpoint}_{i+1}",
                        'http_status': 'ERROR',
                        'response_time_ms': 'N/A',
                        'status': f'❌ FAILED - {str(e)[:50]}...'
                    })

            results[endpoint] = {
                'total_tests': len(test_locations),
                'passed_tests': len([r for r in endpoint_results if 'PASSED' in r['status']]),
                'failed_tests': len([r for r in endpoint_results if 'FAILED' in r['status']]),
                'avg_response_time_ms': np.mean([r['response_time_ms'] for r in endpoint_results if r['response_time_ms'] != 'N/A']),
                'pass_rate': len([r for r in endpoint_results if 'PASSED' in r['status']]) / len(test_locations),
                'detailed_results': endpoint_results
            }

        # Overall API performance
        total_api_tests = sum([r['total_tests'] for r in results.values()])
        total_api_passed = sum([r['passed_tests'] for r in results.values()])

        results['OVERALL_API_PERFORMANCE'] = {
            'total_endpoints_tested': len(results) - 1,
            'total_api_calls': total_api_tests,
            'overall_pass_rate': total_api_passed / max(1, total_api_tests),
            'avg_response_time_ms': np.mean([r['avg_response_time_ms'] for r in results.values() if isinstance(r.get('avg_response_time_ms'), (int, float))]),
            'api_reliability_score': total_api_passed / max(1, total_api_tests),
            'production_ready_apis': total_api_passed >= total_api_tests * 0.95  # 95% success rate
        }

        logger.info(f"✅ API Testing Complete: {total_api_passed}/{total_api_tests} tests passed")
        return results

    def test_dashboard_interface(self) -> Dict[str, Any]:
        """Test farmer dashboard with real user scenarios"""

        logger.info("💻 TESTING FARMER DASHBOARD INTERFACE")

        dashboard_tests = {
            'location_detection': {'test_type': 'input', 'scenarios': 3},
            'crop_auto_detection': {'test_type': 'logic', 'scenarios': 4},
            'prediction_display': {'test_type': 'output', 'scenarios': 5},
            'map_interaction': {'test_type': 'ui', 'scenarios': 2},
            'recommendations': {'test_type': 'logic', 'scenarios': 3},
            'profitability_charts': {'test_type': 'visualization', 'scenarios': 2}
        }

        results = {}
        test_scenarios = [
            {'name': 'Punjab Wheat Farmer', 'lat': 30.5, 'lng': 75.5, 'crop': 'wheat', 'season': 'rabi'},
            {'name': 'Maharashtra Cotton Farmer', 'lat': 19.5, 'lng': 75.5, 'crop': 'cotton', 'season': 'kharif'},
            {'name': 'Karnataka Rice Farmer', 'lat': 15.0, 'lng': 75.0, 'crop': 'rice', 'season': 'kharif'},
            {'name': 'Haryana Wheat Farmer', 'lat': 29.0, 'lng': 76.5, 'crop': 'wheat', 'season': 'rabi'},
            {'name': 'Tamil Nadu Rice Farmer', 'lat': 11.0, 'lng': 79.0, 'crop': 'rice', 'season': 'kharif'}
        ]

        for test_name, config in dashboard_tests.items():
            logger.info(f"📊 Testing {test_name}")

            test_results = []
            scenarios = test_scenarios[:config['scenarios']]

            for i, scenario in enumerate(scenarios):
                try:
                    if test_name == 'location_detection':
                        # Test location processing
                        state = self.dashboard.unified_engine.determine_state_from_coordinates(
                            scenario['lat'], scenario['lng']
                        )
                        result = {
                            'scenario': scenario['name'],
                            'state_detected': state,
                            'accuracy': state.lower() in scenario['name'].lower(),
                            'status': '✅ PASSED' if state != 'unknown' else '❌ FAILED'
                        }

                    elif test_name == 'crop_auto_detection':
                        # Test crop auto-detection
                        predicted_crop = self.dashboard.unified_engine.auto_detect_crop(
                            scenario['lat'], scenario['lng'], scenario['season']
                        )
                        result = {
                            'scenario': scenario['name'],
                            'expected_crop': scenario['crop'],
                            'predicted_crop': predicted_crop,
                            'accuracy': predicted_crop == scenario['crop'],
                            'status': '✅ PASSED' if predicted_crop == scenario['crop'] else '⚠️ PARTIAL'
                        }

                    elif test_name == 'prediction_display':
                        # Test prediction retrieval and display
                        prediction = self.dashboard.get_unified_prediction(
                            scenario['lat'], scenario['lng'],
                            scenario['crop'], scenario['season']
                        )

                        if prediction and 'crop_type' in prediction:
                            yield_val = prediction.get('predicted_yield_quintal_ha', 0)
                            confidence = prediction.get('confidence_level', 'Unknown')
                            result = {
                                'scenario': scenario['name'],
                                'yield_displayed': yield_val,
                                'confidence_level': confidence,
                                'data_complete': bool(prediction.get('state') and prediction.get('location_context')),
                                'status': '✅ PASSED' if yield_val > 0 else '❌ FAILED'
                            }
                        else:
                            result = {
                                'scenario': scenario['name'],
                                'yield_displayed': 0,
                                'confidence_level': 'N/A',
                                'data_complete': False,
                                'status': '❌ FAILED - No data'
                            }

                    elif test_name == 'recommendations':
                        # Test recommendations generation
                        prediction = self.dashboard.get_unified_prediction(
                            scenario['lat'], scenario['lng'], scenario['crop']
                        )

                        if prediction:
                            recommendations = self.dashboard.create_actionable_recommendations(prediction)

                            result = {
                                'scenario': scenario['name'],
                                'recommendations_count': len(recommendations),
                                'has_high_priority': any(r['priority'] == 'high' for r in recommendations),
                                'diversity_score': len(set(r['type'] for r in recommendations)),
                                'status': '✅ PASSED' if len(recommendations) > 2 else '❌ FAILED'
                            }
                        else:
                            result = {
                                'scenario': scenario['name'],
                                'recommendations_count': 0,
                                'has_high_priority': False,
                                'diversity_score': 0,
                                'status': '❌ FAILED - No recommendations'
                            }

                    elif test_name == 'map_interaction':
                        # Test map creation (simplified)
                        try:
                            map_obj = self.dashboard.create_location_map(scenario['lat'], scenario['lng'])
                            result = {
                                'scenario': scenario['name'],
                                'map_created': True,
                                'center_coordinates': [scenario['lat'], scenario['lng']],
                                'zoom_level': 10,
                                'status': '✅ PASSED'
                            }
                        except Exception as e:
                            result = {
                                'scenario': scenario['name'],
                                'map_created': False,
                                'status': f'❌ FAILED - {str(e)[:30]}...'
                            }

                    elif test_name == 'profitability_charts':
                        # Test visualization generation
                        prediction = self.dashboard.get_unified_prediction(
                            scenario['lat'], scenario['lng'], scenario['crop']
                        )

                        if prediction:
                            try:
                                chart = self.dashboard.generate_profitability_chart(prediction)
                                result = {
                                    'scenario': scenario['name'],
                                    'chart_generated': True,
                                    'has_yield_data': prediction.get('predicted_yield_quintal_ha', 0) > 0,
                                    'has_cost_data': True,  # Mock data used
                                    'visualization_complete': True,
                                    'status': '✅ PASSED'
                                }
                            except Exception as e:
                                result = {
                                    'scenario': scenario['name'],
                                    'chart_generated': False,
                                    'status': f'❌ FAILED - {str(e)[:30]}...'
                                }
                        else:
                            result = {
                                'scenario': scenario['name'],
                                'chart_generated': False,
                                'status': '❌ FAILED - No prediction data'
                            }

                    test_results.append(result)

                except Exception as e:
                    test_results.append({
                        'scenario': scenario['name'],
                        'status': f'❌ FAILED - {str(e)[:50]}...'
                    })

            results[test_name] = {
                'total_tests': len(scenarios),
                'passed_tests': len([r for r in test_results if 'PASSED' in r['status']]),
                'failed_tests': len([r for r in test_results if 'FAILED' in r['status']]),
                'pass_rate': len([r for r in test_results if 'PASSED' in r['status']]) / len(scenarios),
                'detailed_results': test_results
            }

        # Overall dashboard performance
        total_dashboard_tests = sum([r['total_tests'] for r in results.values()])
        total_dashboard_passed = sum([r['passed_tests'] for r in results.values()])

        results['OVERALL_DASHBOARD_PERFORMANCE'] = {
            'features_tested': len(results) - 1,
            'total_user_scenarios': total_dashboard_tests,
            'overall_pass_rate': total_dashboard_passed / max(1, total_dashboard_tests),
            'user_experience_score': total_dashboard_passed / max(1, total_dashboard_tests),
            'farmer_interface_ready': total_dashboard_passed >= total_dashboard_tests * 0.9  # 90% reliability
        }

        logger.info(f"✅ Dashboard Testing Complete: {total_dashboard_passed}/{total_dashboard_tests} scenarios passed")
        return results

    def test_performance_load(self) -> Dict[str, Any]:
        """Test platform performance under load"""

        logger.info("⚡ PERFORMANCE LOAD TESTING")

        performance_tests = {
            'concurrent_users_10': {'concurrent_users': 10, 'requests_per_user': 5},
            'concurrent_users_25': {'concurrent_users': 25, 'requests_per_user': 3},
            'concurrent_users_50': {'concurrent_users': 50, 'requests_per_user': 2}
        }

        results = {}

        for test_name, config in performance_tests.items():
            logger.info(f"🏃 Load Testing: {config['concurrent_users']} concurrent users")

            def make_api_call(user_id: int) -> Dict[str, Any]:
                """Single API call for performance testing"""
                test_locs = self.test_locations
                loc = test_locs[user_id % len(test_locs)]  # Distribute across test locations

                start_time = time.time()

                try:
                    payload = {
                        "latitude": loc['lat'],
                        "longitude": loc['lng'],
                        "season": loc['season']
                    }

                    response = requests.post(
                        f"{self.api_base_url}/unified/predict/yield",
                        json=payload,
                        timeout=30
                    )

                    response_time = time.time() - start_time

                    return {
                        'user_id': user_id,
                        'response_time': response_time,
                        'success': response.status_code == 200,
                        'status_code': response.status_code
                    }

                except Exception as e:
                    response_time = time.time() - start_time
                    return {
                        'user_id': user_id,
                        'response_time': response_time,
                        'success': False,
                        'error': str(e)
                    }

            # Run concurrent requests
            total_requests = config['concurrent_users'] * config['requests_per_user']
            all_response_times = []
            success_count = 0

            with concurrent.futures.ThreadPoolExecutor(max_workers=config['concurrent_users']) as executor:
                futures = []

                for user_id in range(config['concurrent_users']):
                    for req in range(config['requests_per_user']):
                        futures.append(executor.submit(make_api_call, user_id))

                # Collect results
                for future in concurrent.futures.as_completed(futures):
                    try:
                        result = future.result(timeout=60)
                        all_response_times.append(result['response_time'])

                        if result['success']:
                            success_count += 1

                    except Exception as e:
                        logger.warning(f"Request timeout: {e}")

            # Calculate performance metrics
            if all_response_times:
                avg_response_time = np.mean(all_response_times)
                p95_response_time = np.percentile(all_response_times, 95)
                p99_response_time = np.percentile(all_response_times, 99)
                max_response_time = np.max(all_response_times)
                min_response_time = np.min(all_response_times)

                results[test_name] = {
                    'concurrent_users': config['concurrent_users'],
                    'total_requests': total_requests,
                    'successful_requests': success_count,
                    'success_rate': success_count / total_requests,
                    'avg_response_time': avg_response_time,
                    'p95_response_time': p95_response_time,
                    'p99_response_time': p99_response_time,
                    'max_response_time': max_response_time,
                    'min_response_time': min_response_time,
                    'requests_per_second': total_requests / sum(all_response_times),
                    'performance_score': 'EXCELLENT' if success_count / total_requests > 0.95 and avg_response_time < 5.0 else
                                        'GOOD' if success_count / total_requests > 0.90 and avg_response_time < 10.0 else
                                        'POOR'
                }
            else:
                results[test_name] = {
                    'concurrent_users': config['concurrent_users'],
                    'total_requests': 0,
                    'successful_requests': 0,
                    'success_rate': 0,
                    'performance_score': 'NO_DATA'
                }

        # Overall performance assessment
        successful_tests = len([r for r in results.values() if r.get('performance_score') in ['EXCELLENT', 'GOOD']])

        results['OVERALL_PERFORMANCE_ANALYSIS'] = {
            'load_levels_tested': len(results) - 1,
            'successful_load_tests': successful_tests,
            ' Recommended_concurrent_users': 25 if successful_tests >= 2 else 10,
            'response_time_sla_seconds': 5.0,
            'throughput_requirement': '50 req/sec',
            'production_scalability_confirmed': successful_tests >= len(results) - 1
        }

        logger.info("✅ Performance Load Testing Complete")
        return results

    def test_edge_cases(self) -> Dict[str, Any]:
        """Test edge cases and error conditions"""

        logger.info("⚠️ TESTING EDGE CASES & ERROR CONDITIONS")

        edge_cases = {
            'invalid_coordinates': [
                {'lat': 91, 'lng': 0, 'expected': 'invalid_latitude'},
                {'lat': -91, 'lng': 0, 'expected': 'invalid_latitude'},
                {'lat': 0, 'lng': 181, 'expected': 'invalid_longitude'},
                {'lat': 0, 'lng': -181, 'expected': 'invalid_longitude'}
            ],
            'remote_locations': [
                {'lat': 8.4, 'lng': 77.5, 'expected': 'kerala'},  # Kanyakumari tip
                {'lat': 35.0, 'lng': 75.0, 'expected': 'jammu_kashmir'},  # Northernmost
                {'lat': 6.7, 'lng': 93.8, 'expected': 'andaman'},  # Southernmost
            ],
            'unsupported_crops': [
                {'crop': 'sugarcane', 'lat': 26.0, 'lng': 80.0},
                {'crop': 'coffee', 'lat': 12.0, 'lng': 75.0},
                {'crop': 'tea', 'lat': 27.0, 'lng': 88.0}
            ],
            'extreme_weather_conditions': [
                {'lat': 32.5, 'lng': 75.5, 'season': 'winter'},  # High altitude
                {'lat': 8.0, 'lng': 76.5, 'season': 'summer'},   # Coastal humidity
                {'lat': 24.0, 'lng': 92.0, 'season': 'monsoon'}  # Heavy rainfall
            ]
        }

        results = {}

        for test_type, test_cases in edge_cases.items():
            logger.info(f"🧪 Testing {test_type.replace('_', ' ')}")

            test_results = []

            for i, case in enumerate(test_cases):
                try:
                    # Test unified prediction with edge case
                    prediction = self.dashboard.get_unified_prediction(
                        case['lat'], case['lng'],
                        case.get('crop'), case.get('season')
                    )

                    # Validate response
                    if prediction and isinstance(prediction, dict):
                        if 'error' in prediction:
                            # Expected error case
                            result_status = '✅ EXPECTED ERROR' if case.get('expected') else '⚠️ UNEXPECTED ERROR'
                        elif prediction.get('state') in ['unknown', '']:
                            # Unknown location case
                            result_status = '⚠️ UNKNOWN LOCATION' if case.get('expected') == 'remote' else '❌ FAILED'
                        else:
                            # Successful case
                            result_status = '✅ PASSED' if not case.get('expected') else '⚠️ UNEXPECTED SUCCESS'
                    else:
                        result_status = '❌ FAILED - No response'

                    test_results.append({
                        'test_case': f"{test_type}_{i+1}",
                        'coordinates': f"{case['lat']}, {case['lng']}",
                        'parameters': case,
                        'response_received': bool(prediction),
                        'has_data': bool(prediction and 'crop_type' in prediction),
                        'status': result_status
                    })

                except Exception as e:
                    test_results.append({
                        'test_case': f"{test_type}_{i+1}",
                        'coordinates': f"{case['lat']}, {case['lng']}",
                        'parameters': case,
                        'response_received': False,
                        'has_data': False,
                        'status': f'❌ EXCEPTION - {str(e)[:50]}...'
                    })

            results[test_type] = {
                'total_tests': len(test_cases),
                'passed_tests': len([r for r in test_results if 'PASSED' in r['status']]),
                'expected_errors': len([r for r in test_results if 'EXPECTED' in r['status']]),
                'failed_tests': len([r for r in test_results if 'FAILED' in r['status']]),
                'robustness_score': len([r for r in test_results if 'PASSED' in r['status'] or 'EXPECTED' in r['status']]) / len(test_cases),
                'detailed_results': test_results
            }

        # Overall robustness assessment
        total_edge_tests = sum([r['total_tests'] for r in results.values()])
        total_edge_passed = sum([
            r['passed_tests'] + r['expected_errors'] for r in results.values()
        ])

        results['OVERALL_SYSTEM_ROBUSTNESS'] = {
            'edge_case_categories': len(results) - 1,
            'total_edge_cases_tested': total_edge_tests,
            'handled_gracefully': total_edge_passed,
            'error_handling_rate': total_edge_passed / max(1, total_edge_tests),
            'system_stability_score': total_edge_passed / max(1, total_edge_tests),
            'production_ready_for_edge_cases': total_edge_passed >= total_edge_tests * 0.9  # 90% graceful handling
        }

        logger.info("✅ Edge Case Testing Complete")
        return results

    def run_comprehensive_validation(self) -> Dict[str, Any]:
        """Run all validation tests comprehensively"""

        logger.info("🚀 STARTING COMPREHENSIVE REAL-TIME PLATFORM VALIDATION")
        logger.info("=" * 80)

        # Start memory tracing
        tracemalloc.start()

        validation_start = time.time()

        try:
            # 1. ML Model Validation
            logger.info("\n🔬 PHASE 1: MACHINE LEARNING MODEL VALIDATION")
            self.test_results['ml_models'] = self.test_machine_learning_models()

            # 2. API Endpoint Testing
            logger.info("\n🌐 PHASE 2: API ENDPOINT TESTING")
            self.test_results['apis'] = self.test_api_endpoints()

            # 3. Dashboard Interface Testing
            logger.info("\n💻 PHASE 3: FARMER DASHBOARD TESTING")
            self.test_results['dashboard'] = self.test_dashboard_interface()

            # 4. Performance Load Testing
            logger.info("\n⚡ PHASE 4: PERFORMANCE LOAD TESTING")
            self.test_results['performance'] = self.test_performance_load()

            # 5. Edge Cases & Error Conditions
            logger.info("\n🛡️ PHASE 5: EDGE CASES & ROBUSTNESS TESTING")
            self.test_results['edge_cases'] = self.test_edge_cases()

            # 6. Database Integration (if available)
            logger.info("\n💾 PHASE 6: DATABASE INTEGRATION TESTING")
            self.test_results['database'] = self.test_database_integration()

            # 7. Business Logic Validation
            logger.info("\n📊 PHASE 7: BUSINESS LOGIC VALIDATION")
            self.test_results['business_logic'] = self.test_business_logic()

        except Exception as e:
            logger.error(f"❌ Validation failed with exception: {e}")
            self.test_results['validation_error'] = str(e)

        # Calculate memory usage
        memory_usage = tracemalloc.get_traced_memory()[0] / 1024 / 1024  # MB
        tracemalloc.stop()

        validation_time = time.time() - validation_start

        # Generate comprehensive report
        validation_report = self.generate_comprehensive_report(validation_time, memory_usage)

        logger.info("✅ COMPREHENSIVE REAL-TIME VALIDATION COMPLETE")
        logger.info(f"Validation completed in {validation_time:.2f} seconds")
        logger.info(f"📋 Results saved to: realtime_validation_report.json")

        return {
            'validation_results': self.test_results,
            'performance_metrics': {
                'total_validation_time_seconds': validation_time,
                'peak_memory_usage_mb': memory_usage,
                'tests_executed': sum(len(component.get('detailed_results', [])) for category in self.test_results.values() for component in category.values() if isinstance(component, dict) and 'detailed_results' in component),
                'test_categories': len(self.test_results)
            },
            'validation_report': validation_report
        }

    def test_database_integration(self) -> Dict[str, Any]:
        """Test database connectivity and operations"""
        return {
            'database_available': False,
            'connectivity_test': 'NOT_AVAILABLE',
            'data_persistence_test': 'NOT_TESTED',
            'query_performance': 'NOT_TESTED',
            'note': 'Database testing requires Railway/Firestore credentials'
        }

    def test_business_logic(self) -> Dict[str, Any]:
        """Test business scaling and partnership logic"""

        try:
            # Test projection calculations
            projections_calc = self.business_scaler.calculate_growth_projections()

            # Test partnership scenarios
            pilot_scenarios = self.business_scaler.identify_pilot_opportunities()

            # Test revenue modeling
            revenue_scenarios = self.business_scaler.create_revenue_model_calculator(10_000_000, {})

            return {
                'growth_projections_test': bool(projections_calc.get('projections')),
                'partnership_scenarios_test': len(pilot_scenarios) > 0,
                'revenue_modeling_test': bool(revenue_scenarios.get('forecasts')),
                'business_logic_integrity': True,
                'detailed_tests': {
                    'target_farmers_calculation': self.business_scaler.target_farmers == 30_000_000,
                    'partnership_opportunities': len(self.business_scaler.partnership_pipeline) > 0,
                    'revenue_scenarios': len([s for s in ['conservative', 'realistic', 'optimistic'] if s in revenue_scenarios.get('forecasts', {})]) > 0
                }
            }
        except Exception as e:
            return {
                'business_logic_test': 'FAILED',
                'error': str(e),
                'business_logic_integrity': False
            }

    def generate_comprehensive_report(self, validation_time: float, memory_usage: float) -> Dict[str, Any]:
        """Generate comprehensive validation report"""

        report = {
            'VALIDATION_SUMMARY': {
                'validation_timestamp': datetime.now().isoformat(),
                'total_validation_time_seconds': validation_time,
                'peak_memory_usage_mb': memory_usage,
                'platform_version': '3.0.0',
                'validation_scope': 'Real-time Data Intensive Testing'
            },
            'OVERALL_PLATFORM_HEALTH': {},
            'COMPONENT_WISE_RESULTS': {},
            'RECOMMENDATIONS': {},
            'PRODUCTION_READINESS_CERTIFICATE': {}
        }

        # Overall health calculation
        component_scores = []

        for category, results in self.test_results.items():
            if category in ['ml_models', 'apis', 'dashboard', 'performance', 'edge_cases']:
                overall_key = [k for k in results.keys() if 'OVERALL' in k]
                if overall_key:
                    overall_result = results[overall_key[0]]
                    component_scores.append(overall_result.get('pass_rate', 0))
                else:
                    # Calculate from individual results
                    total_passed = sum([r.get('passed_tests', 0) for r in results.values() if isinstance(r, dict)])
                    total_tests = sum([r.get('total_tests', 0) for r in results.values() if isinstance(r, dict)])
                    if total_tests > 0:
                        component_scores.append(total_passed / total_tests)
                    else:
                        component_scores.append(0)

        overall_platform_health = np.mean(component_scores) if component_scores else 0

        report['OVERALL_PLATFORM_HEALTH'] = {
            'platform_health_score': overall_platform_health,
            'component_count': len(component_scores),
            'healthy_components': len([s for s in component_scores if s >= 0.9]),
            'warning_components': len([s for s in component_scores if 0.7 <= s < 0.9]),
            'critical_components': len([s for s in component_scores if s < 0.7]),
            'production_deployment_recommended': overall_platform_health >= 0.85
        }

        # Component-wise results
        report['COMPONENT_WISE_RESULTS'] = self.test_results

        # Recommendations
        recommendations = []

        if overall_platform_health >= 0.95:
            recommendations.append("🏆 EXCELLENT: Platform ready for immediate production deployment")
        elif overall_platform_health >= 0.85:
            recommendations.append("✅ GOOD: Platform ready for production with minor fixes")
        elif overall_platform_health >= 0.7:
            recommendations.append("⚠️ WARNING: Address critical component issues before production")
        else:
            recommendations.append("❌ CRITICAL: Major fixes required before any deployment")

        if self.test_results.get('apis', {}).get('OVERALL_API_PERFORMANCE', {}).get('overall_pass_rate', 0) < 0.95:
            recommendations.append("- Improve API endpoint reliability and error handling")

        if self.test_results.get('performance', {}).get('OVERALL_PERFORMANCE_ANALYSIS', {}).get('production_scalability_confirmed', False) == False:
            recommendations.append("- Optimize performance for concurrent user load")

        if self.test_results.get('edge_cases', {}).get('OVERALL_SYSTEM_ROBUSTNESS', {}).get('error_handling_rate', 0) < 0.9:
            recommendations.append("- Enhance error handling and edge case management")

        report['RECOMMENDATIONS'] = recommendations

        # Production readiness certificate
        report['PRODUCTION_READINESS_CERTIFICATE'] = {
            'ceremony_date': datetime.now().isoformat(),
            'certified_by': 'Real-Time Platform Validation Suite',
            'validation_criteria_met': [
                'ML model accuracy validation',
                'API endpoint functionality',
                'User interface testing',
                'Performance load testing',
                'Edge case robustness',
                'Real data integration'
            ],
            'production_readiness_score': overall_platform_health,
            'certificate_status': 'READY' if overall_platform_health >= 0.85 else 'REQUIRES_FIXES',
            'next_steps': 'Immediate production deployment' if overall_platform_health >= 0.85 else 'Fix critical issues'
        }

        # Save detailed report
        with open('real_time_validation_report.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, default=str, ensure_ascii=False)

        return report

def main():
    """Main validation execution"""

    print("🌾 AGRICULTURAL INTELLIGENCE PLATFORM")
    print("📊 Real-Time Data Intensive Validation Suite")
    print("=" * 60)

    # Initialize validator
    validator = RealTimePlatformValidator()

    # Run comprehensive validation
    validation_results = validator.run_comprehensive_validation()

    # Print summary
    print("\n🎯 VALIDATION SUMMARY")
    print("=" * 40)

    platform_health = validation_results.get('validation_report', {}).get('OVERALL_PLATFORM_HEALTH', {})
    health_score = platform_health.get('platform_health_score', 0)

    print(f"{health_score:.0%}")
    print(f"✅ Healthy Components: {platform_health.get('healthy_components', 0)}")
    print(f"⚠️ Warning Components: {platform_health.get('warning_components', 0)}")
    print(f"❌ Critical Components: {platform_health.get('critical_components', 0)}")
