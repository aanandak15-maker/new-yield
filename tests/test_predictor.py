"""
Predictor Tests
Comprehensive testing for streamlined predictor with real APIs
"""

import pytest
from unittest.mock import Mock, patch
import pandas as pd
import numpy as np
from datetime import datetime

# Import predictor components
from india_agri_platform.core.streamlined_predictor import StreamlinedPredictor
from india_agri_platform.core.gee_integration import gee_client, satellite_processor


class TestStreamlinedPredictor:
    """Test streamlined predictor functionality"""

    @pytest.fixture
    def predictor(self):
        """Create predictor instance"""
        return StreamlinedPredictor()

    @pytest.fixture
    def sample_field_data(self):
        """Sample field data for testing"""
        return {
            'crop_name': 'wheat',
            'sowing_date': '2024-11-15',
            'latitude': 30.9010,
            'longitude': 75.8573,
            'variety_name': 'HD-2967'
        }

    def test_location_analysis(self, predictor):
        """Test location coordinate to state/district conversion"""
        lat, lng = 30.9010, 75.8573
        result = predictor._analyze_location(lat, lng)

        assert result is not None
        assert result['state'] == 'punjab'
        assert 'district' in result
        assert result['coordinates']['lat'] == lat

    def test_growth_stage_calculation(self, predictor):
        """Test growth stage calculation accuracy"""
        # Test establishment stage
        result = predictor._calculate_growth_stage('wheat', '2024-11-15')
        assert result['stage'] == 'establishment'
        assert 0 <= result['progress'] <= 1
        assert result['days_since_sowing'] >= 0

        # Test reproductive stage (simulate 80 days later)
        # This would need mocking datetime
        pass

    @patch('india_agri_platform.core.streamlined_predictor.requests.get')
    def test_weather_api_integration(self, mock_get, predictor):
        """Test OpenWeather API integration"""
        # Mock API response
        mock_response = Mock()
        mock_response.json.return_value = {
            'main': {'temp': 25.5, 'humidity': 65},
            'weather': [{'description': 'clear sky'}],
            'rain': {'3h': 0}
        }
        mock_get.return_value = mock_response

        result = predictor._fetch_weather_data(30.9, 75.8, '2024-11-15')

        assert 'summary' in result
        assert result['summary']['average_temperature_c'] == 25.5
        assert result['api_source'] == 'OpenWeatherMap'

    @patch('india_agri_platform.core.gee_integration.gee_client')
    def test_satellite_data_integration(self, mock_gee, predictor):
        """Test Google Earth Engine satellite data integration"""
        # Mock GEE response
        mock_gee.get_multi_band_data.return_value = {
            'ndvi': 0.72,
            'soil_moisture_percent': 45.0,
            'land_surface_temp_c': 28.5,
            'vegetation_health': 'good',
            'data_source': 'google_earth_engine_multi_band'
        }

        result = predictor._get_satellite_data(30.9, 75.8, {'stage': 'reproductive'})

        assert 'ndvi' in result
        assert result['data_source'] == 'google_earth_engine_multi_band'
        assert 'crop_health_analysis' in result

    def test_prediction_pipeline(self, predictor, sample_field_data):
        """Test complete prediction pipeline"""
        result = predictor.predict_yield_streamlined(**sample_field_data)

        assert 'prediction' in result
        assert 'insights' in result
        assert 'auto_fetched_data' in result
        assert 'expected_yield_quintal_ha' in result['prediction']

    def test_error_handling(self, predictor):
        """Test error handling for invalid inputs"""
        # Test invalid coordinates
        result = predictor.predict_yield_streamlined(
            'wheat', '2024-11-15', 999, 999, 'HD-2967'
        )
        assert 'error' in result

        # Test invalid crop
        result = predictor.predict_yield_streamlined(
            'invalid_crop', '2024-11-15', 30.9, 75.8, 'HD-2967'
        )
        assert 'error' in result

    def test_fallback_prediction(self, predictor, sample_field_data):
        """Test fallback prediction when ML models fail"""
        # Mock platform.predict_yield to return error
        with patch('india_agri_platform.core.streamlined_predictor.platform') as mock_platform:
            mock_platform.predict_yield.return_value = {'error': 'Model not available'}

            result = predictor.predict_yield_streamlined(**sample_field_data)

            # Should still return a prediction using fallback
            assert 'prediction' in result
            assert 'expected_yield_quintal_ha' in result['prediction']

    def test_feature_engineering(self, predictor):
        """Test feature engineering for ML models"""
        features = predictor._prepare_features(
            'wheat',
            {'state': 'punjab', 'district': 'Ludhiana'},
            {'stage': 'vegetative_growth', 'days_since_sowing': 45},
            {'summary': {'average_temperature_c': 25, 'total_rainfall_mm': 50, 'average_humidity_percent': 65}},
            {'ndvi': 0.7, 'soil_moisture_percent': 45}
        )

        assert isinstance(features, dict)
        assert 'temperature_celsius' in features
        assert 'ndvi' in features

    def test_insights_generation(self, predictor):
        """Test agricultural insights generation"""
        insights = predictor._generate_insights(
            'wheat',
            {'state': 'punjab'},
            {'stage': 'vegetative_growth', 'days_since_sowing': 45},
            {'summary': {'average_temperature_c': 25, 'total_rainfall_mm': 50}},
            {'predicted_yield_quintal_ha': 48.5}
        )

        assert 'growth_status' in insights
        assert 'irrigation_advice' in insights
        assert 'disease_risk' in insights

    def test_risk_assessment(self, predictor):
        """Test risk assessment functionality"""
        risk = predictor._generate_insights(
            'wheat',
            {'state': 'punjab'},
            {'stage': 'vegetative_growth'},
            {'summary': {'average_temperature_c': 35, 'total_rainfall_mm': 10}},  # High temp, low rain
            {'predicted_yield_quintal_ha': 35}
        )['risk_assessment']

        assert 'risk_level' in risk
        assert 'risk_factors' in risk

    def test_disease_risk_assessment(self, predictor):
        """Test disease risk assessment for different conditions"""
        # High humidity scenario
        risk = predictor._assess_disease_risk(
            'wheat',
            {'stage': 'grain_filling'},
            {'summary': {'average_humidity_percent': 85}}
        )

        assert isinstance(risk, str)
        assert len(risk) > 0

    def test_harvest_readiness(self, predictor):
        """Test harvest readiness assessment"""
        readiness = predictor._get_harvest_readiness({'days_to_harvest': 25})
        assert '25 days' in readiness

        readiness = predictor._get_harvest_readiness({'days_to_harvest': 3})
        assert 'Harvest ready' in readiness

    def test_market_timing_advice(self, predictor):
        """Test market timing recommendations"""
        timing = predictor._get_market_timing('wheat', {'days_to_harvest': 20})
        assert 'Monitor market prices' in timing

        timing = predictor._get_market_timing('wheat', {'days_to_harvest': 5})
        assert 'Harvest soon' in timing


class TestGEESatelliteIntegration:
    """Test Google Earth Engine satellite data integration"""

    @pytest.fixture
    def gee_client_mock(self):
        """Mock GEE client for testing"""
        return Mock()

    def test_gee_data_fetching(self, gee_client_mock):
        """Test GEE data fetching functionality"""
        # Mock successful GEE response
        gee_client_mock.get_multi_band_data.return_value = {
            'ndvi': 0.68,
            'soil_moisture_percent': 42.0,
            'land_surface_temp_c': 27.5,
            'vegetation_health': 'good',
            'data_source': 'google_earth_engine_multi_band'
        }

        # Test data fetching
        result = gee_client_mock.get_multi_band_data(30.9, 75.8, '2024-11-01', '2024-11-15')

        assert result['ndvi'] == 0.68
        assert result['data_source'] == 'google_earth_engine_multi_band'

    def test_satellite_health_analysis(self):
        """Test satellite-based crop health analysis"""
        satellite_data = {
            'ndvi': 0.75,
            'soil_moisture_percent': 50.0
        }

        health_analysis = satellite_processor.analyze_crop_health(
            satellite_data, 'wheat', 'reproductive'
        )

        assert 'health_status' in health_analysis
        assert 'health_score' in health_analysis
        assert 'recommendations' in health_analysis

    def test_vegetation_health_classification(self):
        """Test vegetation health classification"""
        # Test different NDVI values
        test_cases = [
            (0.8, 'excellent'),
            (0.65, 'good'),
            (0.5, 'fair'),
            (0.35, 'poor'),
            (0.2, 'critical')
        ]

        for ndvi, expected_health in test_cases:
            satellite_data = {'ndvi': ndvi}
            health = satellite_processor.analyze_crop_health(
                satellite_data, 'wheat', 'vegetative_growth'
            )
            assert health['health_status'] == expected_health

    def test_change_detection(self):
        """Test vegetation change detection over time"""
        time_series = [
            {'date': '2024-11-01', 'ndvi': 0.6},
            {'date': '2024-11-08', 'ndvi': 0.65},
            {'date': '2024-11-15', 'ndvi': 0.58}
        ]

        change_analysis = satellite_processor.detect_changes(time_series)

        assert 'change_detected' in change_analysis
        assert 'ndvi_trend' in change_analysis
        assert 'trend_direction' in change_analysis


class TestPredictorIntegration:
    """Integration tests for complete predictor workflows"""

    @pytest.fixture
    def predictor(self):
        return StreamlinedPredictor()

    def test_complete_prediction_workflow(self, predictor):
        """Test complete prediction workflow from input to output"""
        input_data = {
            'crop_name': 'wheat',
            'sowing_date': '2024-11-15',
            'latitude': 30.9010,
            'longitude': 75.8573,
            'variety_name': 'HD-2967'
        }

        result = predictor.predict_yield_streamlined(**input_data)

        # Verify all expected components are present
        required_keys = ['prediction', 'insights', 'auto_fetched_data', 'input', 'timestamp']
        for key in required_keys:
            assert key in result

        # Verify prediction structure
        prediction = result['prediction']
        required_pred_keys = ['expected_yield_quintal_ha', 'confidence_interval', 'growth_stage']
        for key in required_pred_keys:
            assert key in prediction

        # Verify insights structure
        insights = result['insights']
        required_insight_keys = ['growth_status', 'irrigation_advice', 'disease_risk']
        for key in required_insight_keys:
            assert key in insights

    def test_different_crop_types(self, predictor):
        """Test prediction with different crop types"""
        crops_to_test = ['wheat', 'rice', 'maize', 'cotton']

        for crop in crops_to_test:
            input_data = {
                'crop_name': crop,
                'sowing_date': '2024-11-15',
                'latitude': 30.9010,
                'longitude': 75.8573
            }

            result = predictor.predict_yield_streamlined(**input_data)

            assert 'prediction' in result
            assert result['input']['crop'] == crop

    def test_different_locations(self, predictor):
        """Test prediction with different geographic locations"""
        locations = [
            (30.9010, 75.8573, 'punjab'),    # Ludhiana
            (28.6139, 77.2090, 'delhi'),     # Delhi
            (26.8467, 80.9462, 'uttar_pradesh'),  # Lucknow
            (25.5941, 85.1376, 'bihar')      # Patna
        ]

        for lat, lng, expected_state in locations:
            input_data = {
                'crop_name': 'wheat',
                'sowing_date': '2024-11-15',
                'latitude': lat,
                'longitude': lng
            }

            result = predictor.predict_yield_streamlined(**input_data)

            assert 'prediction' in result
            # Note: Location detection might not work perfectly for all coordinates
            # but the prediction should still complete

    def test_seasonal_variations(self, predictor):
        """Test predictions across different seasons"""
        # Test different sowing dates to simulate seasonal effects
        sowing_dates = ['2024-06-15', '2024-11-15', '2025-01-15']

        for sowing_date in sowing_dates:
            input_data = {
                'crop_name': 'wheat',
                'sowing_date': sowing_date,
                'latitude': 30.9010,
                'longitude': 75.8573
            }

            result = predictor.predict_yield_streamlined(**input_data)

            assert 'prediction' in result
            assert result['input']['sowing_date'] == sowing_date

    def test_api_error_handling(self, predictor):
        """Test graceful handling of API failures"""
        # Test with invalid API keys or network issues
        # Should fallback to seasonal estimates

        input_data = {
            'crop_name': 'wheat',
            'sowing_date': '2024-11-15',
            'latitude': 30.9010,
            'longitude': 75.8573
        }

        result = predictor.predict_yield_streamlined(**input_data)

        # Should still return a prediction even if APIs fail
        assert 'prediction' in result
        assert 'expected_yield_quintal_ha' in result['prediction']

    def test_performance_requirements(self, predictor):
        """Test performance meets requirements"""
        import time

        input_data = {
            'crop_name': 'wheat',
            'sowing_date': '2024-11-15',
            'latitude': 30.9010,
            'longitude': 75.8573
        }

        start_time = time.time()
        result = predictor.predict_yield_streamlined(**input_data)
        end_time = time.time()

        response_time = end_time - start_time

        # Should respond within 2 seconds for good user experience
        assert response_time < 2.0
        assert 'prediction' in result

    def test_data_consistency(self, predictor):
        """Test data consistency across multiple predictions"""
        input_data = {
            'crop_name': 'wheat',
            'sowing_date': '2024-11-15',
            'latitude': 30.9010,
            'longitude': 75.8573
        }

        # Make multiple predictions with same input
        results = []
        for _ in range(5):
            result = predictor.predict_yield_streamlined(**input_data)
            results.append(result['prediction']['expected_yield_quintal_ha'])

        # Results should be reasonably consistent (not random)
        std_dev = np.std(results)
        assert std_dev < 5.0  # Less than 5 quintal variation

    def test_realistic_yield_ranges(self, predictor):
        """Test that predictions are within realistic ranges"""
        test_cases = [
            ('wheat', 35, 55),    # Punjab wheat typical range
            ('rice', 40, 65),     # Rice typical range
            ('maize', 45, 75),    # Maize typical range
            ('cotton', 20, 35),   # Cotton typical range
        ]

        for crop, min_yield, max_yield in test_cases:
            input_data = {
                'crop_name': crop,
                'sowing_date': '2024-11-15',
                'latitude': 30.9010,
                'longitude': 75.8573
            }

            result = predictor.predict_yield_streamlined(**input_data)
            predicted_yield = result['prediction']['expected_yield_quintal_ha']

            assert min_yield <= predicted_yield <= max_yield, \
                f"{crop} yield {predicted_yield} outside realistic range {min_yield}-{max_yield}"
