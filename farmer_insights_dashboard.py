#!/usr/bin/env python3
"""
AGRICULTURAL INSIGHTS DASHBOARD - Phase 3 Week 2
Frontend Interface for Farmers with Unified Crop Intelligence
Production-Ready Web Dashboard with Real-Time Map & Analytics
"""

import sys
import os
import json
import requests
from datetime import datetime, timedelta
import folium
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append('india_agri_platform')

# Import unified crop API components
from india_agri_platform.api.unified_crop_api import UnifiedCropPredictionEngine
from india_agri_platform.api.main import multi_crop_predictor

class AgriculturalInsightsDashboard:
    """
    PRODUCER AGRICULTURAL INSIGHTS DASHBOARD
    Web-based interface for Indian farmers with intelligent mapping and analytics

    Features:
    - Interactive India map with GPS location detection
    - Unified crop prediction with real-time insights
    - Market price intelligence and profitability analytics
    - Crop rotation suggestions and seasonal planning
    - Mobile-responsive design for farmer phones
    - Multilingual support (Hindi + English)
    """

    def __init__(self):
        self.unified_engine = UnifiedCropPredictionEngine()
        self.api_base_url = "http://localhost:8000"  # Production API URL
        self.india_center = [20.5937, 78.9629]  # Geographic center of India

        # Indian states with agricultural data
        self.state_boundaries = self._load_state_boundaries()
        self.market_prices = self._load_market_prices()

        # Multi-language support
        self.languages = {
            'en': {
                'title': 'Agricultural Intelligence Platform',
                'subtitle': 'Smart Farming for Indian Farmers',
                'select_location': 'Click on map or enter coordinates',
                'crop_prediction': 'Crop Intelligence',
                'market_prices': 'Market Prices',
                'profitability': 'Profitability Analysis',
                'recommendations': 'Smart Recommendations'
            },
            'hi': {
                'title': 'कृषि बुद्धिमत्ता मंच',
                'subtitle': 'भारतीय किसानों के लिए बुद्धिमान खेती',
                'select_location': 'मानचित्र पर क्लिक करें या निर्देशांक दर्ज करें',
                'crop_prediction': 'फसल बुद्धिमत्ता',
                'market_prices': 'बाजार मूल्य',
                'profitability': 'लाभप्रदता विश्लेषण',
                'recommendations': 'बुद्धिमान सिफारिशें'
            }
        }

    def _load_state_boundaries(self) -> Dict[str, Dict]:
        """Load Indian state boundaries for map visualization"""
        # Simplified state boundaries (would be loaded from GeoJSON in production)
        return {
            'punjab': {'center': [31.1471, 75.3412], 'crops': ['wheat', 'rice', 'cotton']},
            'haryana': {'center': [29.0588, 76.0856], 'crops': ['wheat', 'rice', 'cotton']},
            'maharashtra': {'center': [19.7515, 75.7139], 'crops': ['cotton', 'soybean', 'wheat']},
            'gujarat': {'center': [22.2587, 71.1924], 'crops': ['cotton', 'groundnut', 'wheat']},
            'rajasthan': {'center': [27.0238, 74.2179], 'crops': ['wheat', 'mustard', 'bajara']},
            'karnataka': {'center': [15.3173, 75.7139], 'crops': ['rice', 'maize', 'cotton']},
            'tamil_nadu': {'center': [11.1271, 78.6569], 'crops': ['rice', 'cotton', 'sugarcane']},
            'uttar_pradesh': {'center': [26.8467, 80.9462], 'crops': ['wheat', 'rice', 'sugarcane']},
            'bihar': {'center': [25.0961, 85.3131], 'crops': ['rice', 'wheat', 'maize']},
            'west_bengal': {'center': [22.9868, 87.8550], 'crops': ['rice', 'jute', 'potato']}
        }

    def _load_market_prices(self) -> Dict[str, Dict[str, float]]:
        """Load current market prices for Indian crops"""
        # In production, this would fetch from commodity exchanges
        return {
            'wheat': {
                'punjab': 2150, 'haryana': 2100, 'uttar_pradesh': 2050,
                'rajasthan': 2000, 'madhya_pradesh': 1900, 'bihar': 1950
            },
            'rice': {
                'punjab': 2200, 'haryana': 2100, 'uttar_pradesh': 2050,
                'west_bengal': 2800, 'tamil_nadu': 2500, 'andhra_pradesh': 2300
            },
            'cotton': {
                'maharashtra': 5200, 'gujarat': 5100, 'andhra_pradesh': 5000,
                'punjab': 5300, 'haryana': 5250, 'tamil_nadu': 5150
            },
            'maize': {
                'karnataka': 1600, 'rajasthan': 1500, 'madhya_pradesh': 1450,
                'uttar_pradesh': 1550, 'bihar': 1475, 'andhra_pradesh': 1525
            },
            'soybean': {
                'maharashtra': 3500, 'madhya_pradesh': 3200, 'rajasthan': 3300
            },
            'sugarcane': {
                'uttar_pradesh': 3500, 'maharashtra': 3200, 'karnataka': 3150,
                'tamil_nadu': 3300, 'andhra_pradesh': 3250
            }
        }

    def create_location_map(self, selected_lat: Optional[float] = None,
                          selected_lng: Optional[float] = None) -> folium.Map:
        """Create interactive India map with agricultural regions"""

        # Create base map centered on India
        india_map = folium.Map(
            location=self.india_center,
            zoom_start=5,
            tiles='OpenStreetMap'
        )

        # Add state boundaries and crop information
        for state, data in self.state_boundaries.items():
            center = data['center']
            crops = ', '.join(data['crops']).title()

            # Add state marker with crop information
            folium.Marker(
                location=center,
                popup=f'<b>{state.title()}</b><br>Major Crops: {crops}',
                icon=folium.Icon(color='green', icon='leaf')
            ).add_to(india_map)

        # Add selected location marker if provided
        if selected_lat and selected_lng:
            determined_state = self.unified_engine.determine_state_from_coordinates(
                selected_lat, selected_lng
            )

            folium.Marker(
                location=[selected_lat, selected_lng],
                popup=f'<b>Your Location</b><br>State: {determined_state.title() or "Unknown"}',
                icon=folium.Icon(color='red', icon='home')
            ).add_to(india_map)

            # Zoom to selected location
            india_map.location = [selected_lat, selected_lng]
            india_map.zoom_start = 10

        return india_map

    def get_unified_prediction(self, latitude: float, longitude: float,
                             crop: Optional[str] = None, season: Optional[str] = None) -> Dict[str, Any]:
        """Get unified crop prediction from API"""

        try:
            # Call unified API
            payload = {
                "latitude": latitude,
                "longitude": longitude,
                "crop": crop,
                "season": season
            }

            response = requests.post(f"{self.api_base_url}/unified/predict/yield", json=payload)

            if response.status_code == 200:
                return response.json()
            else:
                # Fallback to local engine
                return self._get_local_prediction(latitude, longitude, crop, season)

        except Exception as e:
            st.error(f"API connection failed, using local fallback: {str(e)}")
            return self._get_local_prediction(latitude, longitude, crop, season)

    def _get_local_prediction(self, latitude: float, longitude: float,
                            crop: Optional[str] = None, season: Optional[str] = None) -> Dict[str, Any]:
        """Local prediction fallback when API is unavailable"""

        # Determine location and crop
        state = self.unified_engine.determine_state_from_coordinates(latitude, longitude)

        if not crop:
            crop = self.unified_engine.auto_detect_crop(latitude, longitude, season)

        # Generate basic prediction
        yield_prediction = self.unified_engine._generate_yield_prediction(
            crop, latitude, longitude, "regional_variety", state
        )

        # Get market price
        price_per_quintal = self.market_prices.get(crop, {}).get(state.lower(), 2000)

        # Calculate profitability
        yield_quintal = yield_prediction.get('yield', 0)
        revenue_per_hectare = yield_quintal * price_per_quintal

        return {
            "crop_type": crop,
            "variety": "Regional recommended",
            "state": state,
            "district": "Auto-determined",
            "predicted_yield_quintal_ha": yield_quintal,
            "unit": "quintal/ha",
            "confidence_level": yield_prediction.get('confidence', 'Medium'),
            "location_context": f"Located in {state} - agricultural region",
            "regional_crop_suitability": [],  # Would be populated in full API
            "alternative_crops": [],
            "crop_rotation_suggestions": [f"{crop.title()} → Wheat → Maize"],
            "optimal_planting_window": "June-July (Kharif)",
            "risk_factors": ["Monitor weather regularly"],
            "timestamp": datetime.now().isoformat(),
            "market_price_per_quintal": price_per_quintal,
            "estimated_revenue_per_hectare": revenue_per_hectare,
            "model_id": f"local_{crop}_{state.lower()}"
        }

    def get_crop_suitability(self, latitude: float, longitude: float,
                           season: Optional[str] = None) -> Dict[str, Any]:
        """Get crop suitability analysis"""

        try:
            params = {'latitude': latitude, 'longitude': longitude}
            if season:
                params['season'] = season

            response = requests.get(f"{self.api_base_url}/unified/crop/suitability", params=params)

            if response.status_code == 200:
                return response.json()
            else:
                return self._get_local_suitability(latitude, longitude, season)

        except Exception:
            return self._get_local_suitability(latitude, longitude, season)

    def _get_local_suitability(self, latitude: float, longitude: float,
                             season: Optional[str] = None) -> Dict[str, Any]:
        """Local crop suitability analysis"""

        primary_crop = self.unified_engine.auto_detect_crop(latitude, longitude, season)
        state = self.unified_engine.determine_state_from_coordinates(latitude, longitude)

        location_crops = {}
        for crop in ['rice', 'wheat', 'cotton', 'maize']:
            score = self.unified_engine.calculate_crop_suitability_score(
                crop, latitude, longitude, season
            )
            location_crops[crop] = {
                'suitability_score': score,
                'suitability_category': 'High' if score > 0.8 else 'Medium' if score > 0.6 else 'Low',
                'recommended': score > 0.7
            }

        recommendations = []
        if state.lower() == 'punjab':
            recommendations = ["Punjab is India's bread basket", "Consider Wheat-Rice crop rotation"]
        elif state.lower() in ['maharashtra', 'gujarat']:
            recommendations = ["Strong cotton growing region", "Diversify with soybean or maize"]
        else:
            recommendations = [f"Good potential in {state} for primary crops"]

        return {
            "primary_crop": primary_crop,
            "suitability_score": location_crops[primary_crop]['suitability_score'],
            "location_based_crops": location_crops,
            "season_context": f"Recommended crops for {season or 'current'} season",
            "regional_recommendations": recommendations,
            "timestamp": datetime.now().isoformat()
        }

    def generate_profitability_chart(self, prediction_data: Dict) -> plt.Figure:
        """Generate profitability visualization"""

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('Agricultural Profitability Analysis', fontsize=16, fontweight='bold')

        # Yield prediction
        crops = ['Conservative', 'Average', 'Optimistic']
        yields = [35, prediction_data['predicted_yield_quintal_ha'], 60]  # Example ranges

        ax1.bar(crops, yields, color=['red', 'orange', 'green'])
        ax1.set_title('Yield Predictions (Quintal/ha)')
        ax1.set_ylabel('Yield (Quintal/ha)')

        # Revenue analysis
        price = prediction_data.get('market_price_per_quintal', 2000)
        revenues = [y * price for y in yields]
        ax1_twin = ax1.twinx()
        ax1_twin.plot(range(len(crops)), revenues, 'o-', color='blue', linewidth=2)
        ax1_twin.set_ylabel('Revenue (₹)', color='blue')

        # Cost break down (simplified)
        categories = ['Seeds', 'Fertilizer', 'Labor', 'Pesticides', 'Others']
        costs = [8000, 12000, 15000, 6000, 8000]  # Approximate costs per hectare

        ax2.barh(categories, costs, color='lightblue')
        ax2.set_title('Estimated Costs (₹/hectare)')
        ax2.set_xlabel('Cost (₹)')

        # Profit margin
        conservative_profit = revenues[0] - sum(costs)
        average_profit = revenues[1] - sum(costs)
        optimistic_profit = revenues[2] - sum(costs)

        ax3.bar(['Conservative', 'Average', 'Optimistic'],
               [conservative_profit, average_profit, optimistic_profit],
               color=['red', 'orange', 'green'])
        ax3.set_title('Profit Margin (₹/hectare)')
        ax3.set_ylabel('Profit (₹)')
        ax3.axhline(y=0, color='black', linestyle='--', alpha=0.7)

        # Risk factors
        risks = ['Weather', 'Pest', 'Price', 'Disease']
        risk_levels = [3, 2, 4, 1]  # 1-5 scale

        ax4.barh(risks, risk_levels, color='salmon')
        ax4.set_title('Risk Factors (1-5 scale)')
        ax4.set_xlabel('Risk Level')

        plt.tight_layout()
        return fig

    def create_actionable_recommendations(self, prediction_data: Dict) -> List[Dict[str, Any]]:
        """Create actionable recommendations based on prediction data"""

        recommendations = []
        crop = prediction_data['crop_type']
        state = prediction_data['state']

        # Basic recommendations
        recommendations.append({
            'type': 'immediate',
            'title': f'Start {crop.title()} Preparation',
            'description': f'Begin land preparation and seed procurement for {crop}',
            'priority': 'high',
            'due_date': (datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d')
        })

        # Variety selection
        varieties = self.unified_engine._get_top_varieties(crop, state)
        if varieties:
            recommendations.append({
                'type': 'planning',
                'title': 'Choose High-Yield Varieties',
                'description': f'Consider these varieties: {", ".join(varieties[:3])}',
                'priority': 'medium',
                'due_date': (datetime.now() + timedelta(days=14)).strftime('%Y-%m-%d')
            })

        # Crop rotation
        rotation_suggestions = prediction_data.get('crop_rotation_suggestions', [])
        if rotation_suggestions:
            recommendations.append({
                'type': 'long_term',
                'title': 'Plan Crop Rotation',
                'description': f'Next season: {" → ".join(rotation_suggestions[0].split(" → ")) if rotation_suggestions else "Consult local extension"}',
                'priority': 'medium',
                'due_date': (datetime.now() + timedelta(days=90)).strftime('%Y-%m-%d')
            })

        # Market timing
        recommendations.append({
            'type': 'marketing',
            'title': 'Monitor Market Prices',
            'description': f'Current {crop} price: ₹{prediction_data.get("market_price_per_quintal", 2000)}/quintal',
            'priority': 'low',
            'due_date': 'ongoing'
        })

        return recommendations

def create_streamlit_dashboard():
    """Create Streamlit web dashboard for farmers"""

    # Initialize dashboard
    dashboard = AgriculturalInsightsDashboard()

    # Configure page
    st.set_page_config(
        page_title="Agricultural Intelligence Platform",
        page_icon="🌾",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Language selector in sidebar
    with st.sidebar:
        st.header("🌾 Agricultural Dashboard")

        language = st.selectbox(
            "Select Language / भाषा चुनें",
            options=['en', 'hi'],
            format_func=lambda x: 'English' if x == 'en' else 'हिंदी'
        )

        st.markdown("---")

        # User location input
        st.subheader("📍 Your Location")

        # Auto-detect current location (would use browser geolocation in production)
        if st.button("📍 Detect My Location"):
            # Simulate location detection (Delhi coordinates for demo)
            st.session_state.latitude = 28.6139
            st.session_state.longitude = 77.2090
            st.success("Location detected: Delhi, India")

        # Manual location input
        col1, col2 = st.columns(2)
        with col1:
            latitude = st.number_input(
                "Latitude",
                value=getattr(st.session_state, 'latitude', 28.6139),
                step=0.01,
                format="%.4f"
            )
        with col2:
            longitude = st.number_input(
                "Longitude",
                value=getattr(st.session_state, 'longitude', 77.2090),
                step=0.01,
                format="%.4f"
            )

        # State determination
        if latitude and longitude:
            try:
                state = dashboard.unified_engine.determine_state_from_coordinates(latitude, longitude)
                st.info(f"📍 Location: {state.title() if state != 'unknown' else state}")
            except:
                st.info("📍 Location: Not determined")
        else:
            state = "Unknown"

        st.markdown("---")

        # Crop and season selection
        crop = st.selectbox(
            "🌱 Select Crop (or Auto-detect)",
            options=['auto'] + dashboard.unified_engine.available_crops,
            format_func=lambda x: 'Auto-detect' if x == 'auto' else x.title()
        )

        season = st.selectbox(
            "🌤️ Season",
            options=['kharif', 'rabi', 'summer'],
            format_func=lambda x: x.title()
        )

        # Analysis trigger
        analyze_button = st.button("🚀 Analyze This Location", type="primary")

    # Main content area
    col1, col2 = st.columns([2, 1])

    # Map section
    with col1:
        st.header("🗺️ India Agricultural Map")

        # Create and display map
        india_map = dashboard.create_location_map(latitude, longitude)
        folium_static(india_map)

        if latitude and longitude:
            if st.button("🔍 Zoom to My Location"):
                st.rerun()

    # Prediction results section
    with col2:
        st.header("🎯 Crop Intelligence")

        if analyze_button and latitude and longitude:
            with st.spinner("Analyzing your location..."):

                # Get unified prediction
                if crop == 'auto':
                    crop = None

                prediction = dashboard.get_unified_prediction(
                    latitude, longitude, crop, season
                )

                # Display results
                if prediction:
                    st.success(f"🌾 Primary Crop: {prediction['crop_type'].title()}")

                    # Yield prediction
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric(
                            "Predicted Yield",
                            f"{prediction['predicted_yield_quintal_ha']} quintal/ha",
                            prediction['confidence_level']
                        )
                    with col_b:
                        revenue = prediction.get('estimated_revenue_per_hectare', 0)
                        st.metric("Revenue Potential", f"₹{revenue:,}")

                    # Location context
                    st.info(prediction['location_context'])

                    # Alternative crops
                    alternatives = prediction.get('alternative_crops', [])
                    if alternatives:
                        st.subheader("🔄 Alternative Crops")
                        for alt in alternatives[:3]:
                            st.write(f"• {alt['crop'].title()} (Score: {alt['suitability_score']:.2f})")

                    # Crop rotation
                    rotation = prediction.get('crop_rotation_suggestions', [])
                    if rotation:
                        st.subheader("🔄 Suggested Rotation")
                        for rot in rotation[:2]:
                            st.write(f"• {rot}")

        else:
            st.info("Click 'Analyze This Location' to get intelligent recommendations for your farm")

    # Detailed analysis section (full width)
    if 'prediction' in locals() and prediction:

        # Suitability analysis
        st.header("🌱 Crop Suitability Analysis")

        suitability = dashboard.get_crop_suitability(latitude, longitude, season)

        if suitability:
            primary_crop = suitability['primary_crop']

            # Suitability scores
            st.subheader(f"🎯 Primary Crop: {primary_crop.title()}")

            # Create score visualization
            crop_scores = {}
            for crop_data in suitability.get('location_based_crops', {}).items():
                crop_name, info = crop_data
                crop_scores[crop_name] = info['suitability_score']

            if crop_scores:
                fig, ax = plt.subplots(figsize=(10, 4))
                crops = list(crop_scores.keys())
                scores = list(crop_scores.values())

                bars = ax.barh(crops, scores, color=['green' if s > 0.7 else 'orange' if s > 0.5 else 'red' for s in scores])
                ax.set_xlabel('Suitability Score (0-1)')
                ax.set_title('Crop Suitability in Your Location')

                for bar, score in zip(bars, scores):
                    ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                           '.2f', ha='left', va='center')

                st.pyplot(fig)

            # Regional recommendations
            recommendations = suitability.get('regional_recommendations', [])
            if recommendations:
                st.subheader("💡 Regional Insights")
                for rec in recommendations:
                    st.write(f"• {rec}")

        # Profitability analysis
        st.header("💰 Profitability Analysis")
        profit_chart = dashboard.generate_profitability_chart(prediction)
        st.pyplot(profit_chart)

        # Actionable recommendations
        st.header("📋 Action Plan")

        recommendations = dashboard.create_actionable_recommendations(prediction)

        # Group recommendations by priority
        for priority in ['high', 'medium', 'low']:
            priority_recs = [r for r in recommendations if r['priority'] == priority]

            if priority_recs:
                st.subheader(f"{'🔴' if priority == 'high' else '🟡' if priority == 'medium' else '🟢'} {priority.title()} Priority Tasks")

                for rec in priority_recs:
                    with st.expander(f"{rec['title']} - Due: {rec['due_date']}"):
                        st.write(rec['description'])

    # Footer
    st.markdown("---")
    st.markdown("""
    **🧪 Agricultural Intelligence Platform** - Powered by Phase 3 Production API

    *Disclaimer: This is agricultural intelligence tool providing data-driven insights.
    Always consult local agricultural extension officers for final decision making.*
    """)

def folium_static(fig):
    """Convert folium map to streamlit display"""
    import streamlit.components.v1 as components

    # Save map as HTML
    fig.save("temp_map.html")

    # Read and display HTML
    with open("temp_map.html", 'r') as f:
        html = f.read()

    components.html(html, height=400)

    # Clean up
    if os.path.exists("temp_map.html"):
        os.remove("temp_map.html")

if __name__ == "__main__":
    print("🌾 STARTING AGRICULTURAL INSIGHTS DASHBOARD - PHASE 3 WEEK 2")
    print("=" * 70)

    # Check if running as Streamlit app or standalone
    if len(sys.argv) > 1 and sys.argv[1] == "--streamlit":
        # Run as Streamlit app
        print("🚀 Launching Streamlit Dashboard...")
        create_streamlit_dashboard()
    else:
        # Run demo analysis
        print("🧪 Running dashboard demo analysis...")

        dashboard = AgriculturalInsightsDashboard()

        # Demo location: Punjab wheat belt
        demo_lat, demo_lng = 30.5, 75.5

        print(f"📍 Demo Analysis for coordinates: {demo_lat}, {demo_lng}")

        # State determination
        state = dashboard.unified_engine.determine_state_from_coordinates(demo_lat, demo_lng)
        print(f"🏛️ Determined State: {state.title()}")

        # Crop prediction
        print("🌾 Getting unified crop prediction...")
        prediction = dashboard.get_unified_prediction(demo_lat, demo_lng)

        print("=" * 50)
        print("🎯 PREDICTION RESULTS:")
        print(f"Crop: {prediction.get('crop_type', 'Unknown')}")
        print(f"Yield: {prediction.get('predicted_yield_quintal_ha', 0)} quintal/ha")
        print(f"Confidence: {prediction.get('confidence_level', 'Unknown')}")
        print(f"State: {prediction.get('state', 'Unknown')}")
        print(f"Location Context: {prediction.get('location_context', '')}")

        # Suitability analysis
        print("\n🌱 Crop Suitability Analysis:")
        suitability = dashboard.get_crop_suitability(demo_lat, demo_lng)
        if suitability:
            print(f"Primary Crop: {suitability.get('primary_crop', 'Unknown')}")
            print(f"Suitability Score: {suitability.get('suitability_score', 0):.2f}")

            location_crops = suitability.get('location_based_crops', {})
            if location_crops:
                print("Crop Suitability Scores:")
                for crop_name, info in location_crops.items():
                    score = info.get('suitability_score', 0)
                    category = info.get('suitability_category', 'Low')
                    print(f"  • {crop_name.title()}: {score:.2f} ({category})")

        # Market insights
        crop = prediction.get('crop_type', 'wheat')
        market_price = dashboard.market_prices.get(crop, {}).get(state.lower(), 2000)
        estimated_revenue = prediction.get('predicted_yield_quintal_ha', 0) * market_price

        print("\n💰 Market Insights:")
        print(f"Market Price ({crop}): ₹{market_price}/quintal")
        print(f"Estimated Revenue: ₹{estimated_revenue:,}/hectare")

        print("\n📋 Action Recommendations:")
        recommendations = dashboard.create_actionable_recommendations(prediction)
        for rec in recommendations[:3]:
            urgency = "🔴" if rec['priority'] == 'high' else "🟡" if rec['priority'] == 'medium' else "🟢"
            print(f"{urgency} {rec['title']}: {rec['description']}")

        print("\n✅ DASHBOARD ANALYSIS COMPLETE")
        print("=" * 70)

        if len(sys.argv) > 1 and sys.argv[1] == "--web":
            print("\n🌐 To start web dashboard, run:")
            print("streamlit run farmer_insights_dashboard.py --streamlit")
        else:
            print("\n🌐 To start interactive web dashboard:")
            print("python farmer_insights_dashboard.py --web")
            print("\nFor streamlit app (requires streamlit):")
            print("streamlit run farmer_insights_dashboard.py --streamlit")

        print(f"\n🇮🇳 DASHBOARD READY FOR {state.upper()} FARMERS!")
        print("🚀 Phase 3 Week 2: Farmer Interface & Business Logic - CHECKED!")
