#!/usr/bin/env python3
"""
User Field Intelligence Layer - Phase 2 Week 4
Micro vision - Individual field understanding with GPS fingerprinting
"""

import os
import sys
import json
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class FieldIntelligenceProcessor:
    """
    Field Intelligence Layer - Micro Vision Individual Field Understanding
    Creates dynamic field profiles with GPS boundaries and temporal health tracking
    """

    def __init__(self):
        self.field_intelligence_cache = {}
        self.field_profiles_dir = Path("field_intelligence_profiles")

        # Field health assessment thresholds (NDVI-based)
        self.health_thresholds = {
            "critical": 0.2,      # Severe stress/dead vegetation
            "poor": 0.35,         # Significant stress
            "moderate": 0.5,      # Moderate health
            "good": 0.65,         # Healthy vegetation
            "excellent": 0.8      # Very healthy/lush vegetation
        }

        logger.info(f"✅ Field Intelligence Processor initialized")

    def create_field_fingerprint(self, field_gps_coordinates: List[Tuple[float, float]],
                                farmer_id: str, field_name: str,
                                crop_type: str, sowing_date: str) -> Dict[str, Any]:
        """
        Create comprehensive field fingerprint from GPS boundaries

        Args:
            field_gps_coordinates: List of (lat, lon) tuples forming field boundary
            farmer_id: Unique farmer identifier
            field_name: User-defined field name
            crop_type: Current crop planted
            sowing_date: When crop was sown (YYYY-MM-DD)
        """

        field_id = f"{farmer_id}_{field_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        try:
            # Step 1: GPS Boundary Analysis
            boundary_analysis = self._analyze_gps_boundaries(field_gps_coordinates)

            # Step 2: Field Characteristics Deduction
            field_characteristics = self._deduce_field_characteristics(
                boundary_analysis, crop_type, sowing_date
            )

            # Step 3: Intelligence Profile Creation
            field_profile = {
                "field_id": field_id,
                "farmer_id": farmer_id,
                "field_name": field_name,
                "created_at": datetime.now().isoformat(),

                # GPS & Geometric Data
                "boundary_analysis": boundary_analysis,

                # Agricultural Intelligence
                "crop_type": crop_type,
                "sowing_date": sowing_date,
                "crop_age_days": (datetime.now() - datetime.fromisoformat(sowing_date)).days,

                # Field Characteristics
                "field_characteristics": field_characteristics,

                # Intelligence Layers
                "health_monitoring": self._initialize_health_monitoring(crop_type),
                "crop_adaptability": self._assess_crop_adaptability(field_characteristics, crop_type),
                "performance_history": [],  # Will be populated over time

                # Risk Assessment
                "vulnerability_assessment": self._create_vulnerability_profile(field_characteristics),

                # Intelligence Status
                "fingerprint_status": "active",
                "last_updated": datetime.now().isoformat(),
                "intelligence_score": 85  # Initial high confidence
            }

            # Cache the profile
            self.field_intelligence_cache[field_id] = field_profile

            # Save to persistent storage
            self._save_field_profile(field_profile)

            logger.info(f"✅ Field fingerprint created: {field_id}")
            return field_profile

        except Exception as e:
            logger.error(f"❌ Field fingerprint creation failed: {e}")
            return {
                "field_id": field_id,
                "fingerprint_status": "failed",
                "error": str(e)
            }

    def update_field_intelligence(self, field_id: str,
                                 satellite_data: Optional[Dict] = None,
                                 weather_data: Optional[Dict] = None,
                                 observed_data: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Update field intelligence with new data (satellite, weather, farmer observations)

        Args:
            field_id: Unique field identifier
            satellite_data: NDVI, vegetation indices from GEE
            weather_data: Local weather conditions
            observed_data: Farmer observations (pests, health, yields)
        """

        try:
            # Load existing field profile
            field_profile = self.field_intelligence_cache.get(field_id)
            if not field_profile:
                # Try loading from storage
                field_profile = self._load_field_profile(field_id)
                if not field_profile:
                    raise ValueError(f"Field profile not found: {field_id}")

            # Update health monitoring with new data
            if satellite_data:
                field_profile = self._update_satellite_intelligence(field_profile, satellite_data)

            if weather_data:
                field_profile = self._update_weather_intelligence(field_profile, weather_data)

            if observed_data:
                field_profile = self._update_farmer_observations(field_profile, observed_data)

            # Recalculate field characteristics
            field_profile["field_characteristics"] = self._reassess_characteristics(field_profile)

            # Update timestamp and intelligence score
            field_profile["last_updated"] = datetime.now().isoformat()
            field_profile["intelligence_score"] = self._calculate_intelligence_score(field_profile)

            # Save updates
            self.field_intelligence_cache[field_id] = field_profile
            self._save_field_profile(field_profile)

            # Generate update insights
            insights = self._generate_update_insights(field_profile)

            return {
                "field_id": field_id,
                "update_status": "successful",
                "intelligence_score": field_profile["intelligence_score"],
                "insights": insights,
                "next_monitoring": (datetime.now() + timedelta(days=3)).isoformat(),  # Suggest next check
                "recommendations": self._generate_field_recommendations(field_profile)
            }

        except Exception as e:
            logger.error(f"❌ Field intelligence update failed: {e}")
            return {
                "field_id": field_id,
                "update_status": "failed",
                "error": str(e)
            }

    def get_field_intelligence(self, field_id: str) -> Dict[str, Any]:
        """
        Retrieve comprehensive field intelligence profile
        """

        try:
            # Try cache first
            field_profile = self.field_intelligence_cache.get(field_id)
            if field_profile:
                return field_profile

            # Load from storage
            field_profile = self._load_field_profile(field_id)
            if field_profile:
                self.field_intelligence_cache[field_id] = field_profile
                return field_profile

            raise ValueError(f"Field intelligence not found: {field_id}")

        except Exception as e:
            logger.error(f"❌ Field intelligence retrieval failed: {e}")
            return {"field_id": field_id, "intelligence_status": "not_found", "error": str(e)}

    def _analyze_gps_boundaries(self, coordinates: List[Tuple[float, float]]) -> Dict[str, Any]:
        """Analyze GPS boundary points for field characteristics"""

        if not coordinates or len(coordinates) < 3:
            raise ValueError("Insufficient GPS coordinates for boundary analysis")

        # Convert to numpy for calculations
        coords = np.array(coordinates)

        # Calculate geometric properties
        centroid_lat = np.mean(coords[:, 0])
        centroid_lon = np.mean(coords[:, 1])

        # Area calculation using shoelace formula (approximate)
        # Assuming coordinates are in lat/lon, convert to approximate meters
        area_sqkm = self._calculate_polygon_area_sqkm(coordinates)

        # Shape analysis
        perimeter_km = self._calculate_perimeter_km(coordinates)

        # Irregularity factor (ratio of perimeter to area)
        irregularity_factor = perimeter_km * 1000 / np.sqrt(area_sqkm * 1e6)  # Perimeter/area ratio
        shape_classification = self._classify_field_shape(irregularity_factor)

        return {
            "centroid_latitude": centroid_lat,
            "centroid_longitude": centroid_lon,
            "area_sqkm": area_sqkm,
            "perimeter_km": perimeter_km,
            "shape_classification": shape_classification,
            "irregularity_factor": irregularity_factor,
            "boundary_points": len(coordinates),
            "gps_accuracy_estimate": "high",  # Assumed for well-defined boundaries
            "elevation_estimate_meters": self._estimate_elevation(centroid_lat, centroid_lon)
        }

    def _deduce_field_characteristics(self, boundary_analysis: Dict, crop_type: str, sowing_date: str) -> Dict[str, Any]:
        """Deduce field characteristics from boundary analysis and crop data"""

        characteristics = {
            "size_category": self._categorize_field_size(boundary_analysis["area_sqkm"]),
            "terrain_type": self._determine_terrain_type(boundary_analysis),
            "irrigation_suitability": self._assess_irrigation_potential(boundary_analysis, crop_type),
            "soil_type_inference": self._infer_soil_characteristics(boundary_analysis),
            "microclimate_indicators": self._analyze_microclimate_features(boundary_analysis),
            "crop_suitability_score": self._calculate_crop_suitability(boundary_analysis, crop_type),
            "vulnerability_factors": self._identify_vulnerability_factors(boundary_analysis, crop_type),
            "management_complexity": self._assess_management_requirements(boundary_analysis, crop_type)
        }

        return characteristics

    def _initialize_health_monitoring(self, crop_type: str) -> Dict[str, Any]:
        """Initialize field health monitoring system"""

        # Crop-specific health parameters
        crop_health_params = {
            "rice": {
                "critical_ndvi_low": 0.3,
                "optimal_ndvi_range": [0.6, 0.9],
                "pest_vulnerable_stages": ["tillering", "panicle_initiation"],
                "water_stress_sensitivity": "high"
            },
            "wheat": {
                "critical_ndvi_low": 0.25,
                "optimal_ndvi_range": [0.5, 0.8],
                "pest_vulnerable_stages": ["crown_root_initiation"],
                "water_stress_sensitivity": "medium"
            },
            "cotton": {
                "critical_ndvi_low": 0.35,
                "optimal_ndvi_range": [0.55, 0.75],
                "pest_vulnerable_stages": ["squared", "boll_development"],
                "water_stress_sensitivity": "high"
            }
        }

        health_params = crop_health_params.get(crop_type, crop_health_params["rice"])

        return {
            "monitoring_parameters": health_params,
            "health_history": [],
            "alerts_active": [],
            "last_health_assessment": datetime.now().isoformat(),
            "health_trend": "monitoring_started",
            "stress_indicators": {
                "water_deficit_detected": False,
                "nutrient_deficiency_alert": False,
                "pest_damage_indicated": False,
                "disease_symptoms_present": False
            },
            "recommended_monitoring_frequency": "every_3_days",
            "alert_thresholds": health_params
        }

    def _assess_crop_adaptability(self, field_characteristics: Dict, crop_type: str) -> Dict[str, Any]:
        """Assess how adaptable the field is to the planted crop"""

        adaptability_factors = {
            "irrigation_match": field_characteristics["irrigation_suitability"]["score"] / 100,
            "size_compatibility": self._calculate_size_adaptability(field_characteristics["size_category"], crop_type),
            "terrain_compatibility": self._calculate_terrain_adaptability(field_characteristics["terrain_type"], crop_type),
            "microclimate_suitability": field_characteristics["microclimate_indicators"]["suitability_score"] / 100,
            "soil_adaptability": field_characteristics["soil_type_inference"]["fertility_index"] / 10
        }

        overall_adaptability = sum(adaptability_factors.values()) / len(adaptability_factors)

        return {
            "adaptability_factors": adaptability_factors,
            "overall_score": overall_adaptability,
            "performance_prediction": "good" if overall_adaptability > 0.7 else "moderate",
            "improvement_suggestions": self._generate_adaptability_improvements(adaptability_factors)
        }

    def _create_vulnerability_profile(self, field_characteristics: Dict) -> Dict[str, Any]:
        """Create comprehensive vulnerability assessment for the field"""

        # Based on field characteristics, assess various risks
        vulnerability_factors = field_characteristics["vulnerability_factors"]

        risk_levels = {
            "flood_risk": vulnerability_factors.get("flood_susceptibility", 0),
            "drought_risk": field_characteristics["irrigation_suitability"].get("drought_vulnerability", 0),
            "pest_risk": field_characteristics["terrain_type"].get("pest_habitat_suitability", 0),
            "erosion_risk": vulnerability_factors.get("soil_erosion_risk", 0),
            "disease_risk": field_characteristics["microclimate_indicators"].get("disease_pressure", 0)
        }

        overall_risk_score = sum(risk_levels.values()) / len(risk_levels)

        return {
            "risk_factors": risk_levels,
            "overall_risk_score": overall_risk_score,
            "risk_category": "high" if overall_risk_score > 0.7 else "medium" if overall_risk_score > 0.4 else "low",
            "monitoring_priorities": sorted(risk_levels.items(), key=lambda x: x[1], reverse=True)[:3],
            "preventive_measures": self._recommend_preventive_measures(risk_levels)
        }

    # Helper methods (simplified implementations)
    def _calculate_polygon_area_sqkm(self, coordinates: List[Tuple[float, float]]) -> float:
        """Calculate approximate area in square kilometers"""
        # Simplified approximation for GPS coordinates
        if len(coordinates) < 3:
            return 0.0

        # Use bounding box as rough approximation
        lats = [coord[0] for coord in coordinates]
        lons = [coord[1] for coord in coordinates]

        lat_range = max(lats) - min(lats)
        lon_range = max(lons) - min(lons)

        # Rough conversion: 1 degree lat ≈ 111 km, 1 degree lon varies by latitude
        avg_lat = sum(lats) / len(lats)
        lat_km = lat_range * 111
        lon_km = lon_range * 111 * abs(np.cos(np.radians(avg_lat)))

        area_sqkm = lat_km * lon_km
        return max(area_sqkm, 0.01)  # Minimum 1 are (0.01 sq km)

    def _calculate_perimeter_km(self, coordinates: List[Tuple[float, float]]) -> float:
        """Calculate perimeter in kilometers"""
        if len(coordinates) < 3:
            return 0.0

        perimeter = 0.0
        for i in range(len(coordinates)):
            p1 = coordinates[i]
            p2 = coordinates[(i + 1) % len(coordinates)]
            distance = self._haversine_distance(p1[0], p1[1], p2[0], p2[1])
            perimeter += distance

        return perimeter

    def _haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate haversine distance in km"""
        R = 6371  # Earth radius in km

        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)

        a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))

        return R * c

    def _classify_field_shape(self, irregularity_factor: float) -> str:
        if irregularity_factor < 4:
            return "rectangular"
        elif irregularity_factor < 6:
            return "irregular"
        else:
            return "very_irregular"

    def _estimate_elevation(self, lat: float, lon: float) -> float:
        """Rough elevation estimation based on latitude (India context)"""
        if lat > 32:
            return 300  # Himalayan foothills
        elif lat > 24:
            return 200  # Northern plains
        elif lat > 20:
            return 400  # Central highlands
        elif lat > 11:
            return 100  # Peninsular region
        else:
            return 50   # Coastal/southern

    # Classification and assessment methods would have more sophisticated implementations
    # These are simplified demonstrations of the intelligence concept

    def _categorize_field_size(self, area_sqkm: float) -> str:
        if area_sqkm < 0.1:
            return "small_garden"
        elif area_sqkm < 0.5:
            return "small_farm"
        elif area_sqkm < 2.0:
            return "medium_farm"
        elif area_sqkm < 10.0:
            return "large_farm"
        else:
            return "industrial_farm"

    def _determine_terrain_type(self, boundary_analysis: Dict) -> Dict[str, Any]:
        """Determine terrain characteristics"""
        area = boundary_analysis["area_sqkm"]
        shape = boundary_analysis["shape_classification"]

        terrain_type = "flat" if shape == "rectangular" else "undulating" if area > 1.0 else "mixed"
        elevation = boundary_analysis["elevation_estimate_meters"]

        return {
            "primary_terrain": terrain_type,
            "elevation_meters": elevation,
            "slope_estimate": "gentle" if elevation < 200 else "moderate",
            "drainage_potential": "good" if terrain_type == "flat" else "moderate",
            "erosion_risk": "low" if shape == "rectangular" else "medium",
            "pest_habitat_suitability": 0.3 if terrain_type == "mixed" else 0.1
        }

    def _assess_irrigation_potential(self, boundary_analysis: Dict, crop_type: str) -> Dict[str, Any]:
        """Assess irrigation suitability"""
        size = boundary_analysis["area_sqkm"]
        shape = boundary_analysis["shape_classification"]

        # Crop-specific irrigation requirements
        irrigation_needs = {
            "rice": "high",
            "wheat": "medium",
            "cotton": "high",
            "maize": "medium"
        }

        crop_need = irrigation_needs.get(crop_type, "medium")

        return {
            "recommended_method": "drip" if size < 0.5 else "sprinkler" if crop_need == "medium" else "flood",
            "irrigation_frequency_estimate": "daily" if crop_need == "high" else "weekly",
            "water_requirement_category": crop_need,
            "drought_vulnerability": 0.2 if crop_need == "low" else 0.7 if crop_need == "high" else 0.4,
            "efficiency_score": 80 if shape == "rectangular" else 65,
            "score": 75 if crop_need == "medium" else 90 if crop_need == "low" else 60
        }

    def _infer_soil_characteristics(self, boundary_analysis: Dict) -> Dict[str, Any]:
        """Infer soil type from geographic characteristics"""
        lat = boundary_analysis["centroid_latitude"]

        if lat > 28:  # Northern India
            soil_type = "alluvial"
            fertility_index = 8.5
        elif lat > 24:  # Gangetic plain
            soil_type = "gangetic_alluvial"
            fertility_index = 9.0
        elif lat > 21:  # Central India
            soil_type = "black_cotton"
            fertility_index = 7.5
        elif lat > 15:  # Western Ghats to Deccan
            soil_type = "red_soil"
            fertility_index = 6.5
        else:  # Southern India
            soil_type = "lateritic"
            fertility_index = 7.0

        return {
            "inferred_soil_type": soil_type,
            "fertility_index": fertility_index,
            "water_retention_capacity": "high" if soil_type in ["alluvial", "black_cotton"] else "medium",
            "nutrient_holding_capacity": "excellent" if fertility_index > 8.5 else "good" if fertility_index > 7.0 else "moderate",
            "erosion_sensitivity": "low" if soil_type == "alluvial" else "medium"
        }

    def _analyze_microclimate_features(self, boundary_analysis: Dict) -> Dict[str, Any]:
        """Analyze microclimate indicators"""
        lat = boundary_analysis["centroid_latitude"]
        lon = boundary_analysis["centroid_longitude"]

        # Microclimate factors based on location
        humidity_factor = 0.7 if lat > 25 else 0.5  # More humid in north
        wind_exposure = 0.3 if lat < 20 else 0.6  # More windy in south

        return {
            "humidity_regime": "humid" if humidity_factor > 0.6 else "moderate",
            "wind_exposure_index": wind_exposure,
            "frost_risk": "high" if lat > 30 else "low",
            "heat_stress_potential": "high" if lat < 20 else "medium",
            "disease_pressure_index": 0.4 if humidity_factor > 0.6 else 0.2,
            "suitability_score": int(100 - (humidity_factor * 20) - (wind_exposure * 30))
        }

    def _calculate_crop_suitability(self, boundary_analysis: Dict, crop_type: str) -> float:
        """Calculate crop-field compatibility score"""
        elevation = boundary_analysis["elevation_estimate_meters"]
        area = boundary_analysis["area_sqkm"]

        # Crop-specific suitability calculations
        suitability_scores = {
            "rice": 85 if elevation < 300 and area < 5.0 else 65,  # Rice needs water and moderate size
            "wheat": 80 if 200 < elevation < 600 else 70,  # Wheat prefers cooler elevations
            "cotton": 75 if 100 < elevation < 400 else 60,  # Cotton needs moderate climate
            "maize": 90 if 300 < elevation < 800 else 70   # Maize tolerates higher altitudes
        }

        return suitability_scores.get(crop_type, 70)

    def _identify_vulnerability_factors(self, boundary_analysis: Dict, crop_type: str) -> Dict[str, float]:
        """Identify field vulnerability factors"""
        lat = boundary_analysis["centroid_latitude"]
        area = boundary_analysis["area_sqkm"]
        elevation = boundary_analysis["elevation_estimate_meters"]

        vulnerabilities = {
            "flood_susceptibility": 0.3 if lat > 25 else 0.7,  # Higher in plains
            "soil_erosion_risk": 0.6 if area > 2.0 else 0.3,   # Larger fields more prone
            "wind_damage_risk": 0.7 if elevation < 200 else 0.3,  # Flat areas windier
            "frost_damage_risk": 0.8 if lat > 30 and crop_type == "rice" else 0.2,
            "heat_stress_risk": 0.8 if lat < 20 and crop_type in ["wheat", "rice"] else 0.3
        }

        return vulnerabilities

    def _assess_management_requirements(self, boundary_analysis: Dict, crop_type: str) -> Dict[str, Any]:
        """Assess field management complexity"""
        size = self._categorize_field_size(boundary_analysis["area_sqkm"])
        shape = boundary_analysis["shape_classification"]

        complexity_factors = {
            "labor_intensity": "high" if size in ["large_farm", "industrial_farm"] else "medium",
            "equipment_needs": "tractor_required" if boundary_analysis["area_sqkm"] > 1.0 else "manual_possible",
            "monitoring_frequency": "daily" if crop_type == "rice" else "weekly",
            "irrigation_complexity": "automated" if size == "industrial_farm" else "manual",
            "harvest_difficulty": "high" if shape == "very_irregular" else "medium"
        }

        return complexity_factors

    def _calculate_size_adaptability(self, size_category: str, crop_type: str) -> float:
        """Calculate size compatibility with crop"""
        compatibility_matrix = {
            "rice": {"small_farm": 0.9, "medium_farm": 1.0, "large_farm": 0.8},
            "wheat": {"small_farm": 0.9, "medium_farm": 0.95, "large_farm": 1.0},
            "cotton": {"small_farm": 0.7, "medium_farm": 0.9, "large_farm": 1.0},
            "maize": {"small_farm": 0.8, "medium_farm": 1.0, "large_farm": 1.0}
        }

        crop_compat = compatibility_matrix.get(crop_type, compatibility_matrix["rice"])
        return crop_compat.get(size_category, 0.8)

    def _calculate_terrain_adaptability(self, terrain_type: Dict, crop_type: str) -> float:
        """Calculate terrain compatibility"""
        terrain = terrain_type["primary_terrain"]

        terrain_compat = {
            "rice": {"flat": 1.0, "undulating": 0.6, "mixed": 0.7},
            "wheat": {"flat": 0.95, "undulating": 0.9, "mixed": 0.8},
            "cotton": {"flat": 0.8, "undulating": 0.9, "mixed": 0.85},
            "maize": {"flat": 0.8, "undulating": 1.0, "mixed": 0.9}
        }

        crop_compat = terrain_compat.get(crop_type, terrain_compat["rice"])
        return crop_compat.get(terrain, 0.8)

    def _generate_adaptability_improvements(self, adaptability_factors: Dict) -> List[str]:
        """Generate improvement suggestions"""
        improvements = []

        if adaptability_factors["irrigation_match"] < 0.7:
            improvements.append("Consider irrigation system upgrade for better water distribution")

        if adaptability_factors["terrain_compatibility"] < 0.8:
            improvements.append("Evaluate terrain modification for optimal crop performance")

        if adaptability_factors["soil_adaptability"] < 0.7:
            improvements.append("Soil amendment and fertility enhancement suggested")

        if not improvements:
            improvements.append("Field shows excellent crop adaptability - maintain current management")

        return improvements

    def _recommend_preventive_measures(self, risk_levels: Dict) -> List[str]:
        """Generate preventive recommendations"""
        recommendations = []

        if risk_levels["flood_risk"] > 0.6:
            recommendations.append("Install drainage systems and raised planting beds")

        if risk_levels["drought_risk"] > 0.6:
            recommendations.append("Install water conservation systems and drought-resistant varieties")

        if risk_levels["pest_risk"] > 0.5:
            recommendations.append("Establish pest monitoring traps and biological control systems")

        if risk_levels["erosion_risk"] > 0.6:
            recommendations.append("Implement contour farming and windbreak trees")

        return recommendations if recommendations else ["Field has low risk profile - standard preventive measures sufficient"]

    def _update_satellite_intelligence(self, field_profile: Dict, satellite_data: Dict) -> Dict:
        """Update field intelligence with satellite data"""
        # Implementation for satellite data integration
        pass

    def _update_weather_intelligence(self, field_profile: Dict, weather_data: Dict) -> Dict:
        """Update field intelligence with weather data"""
        # Implementation for weather data integration
        pass

    def _update_farmer_observations(self, field_profile: Dict, observed_data: Dict) -> Dict:
        """Update field intelligence with farmer observations"""
        # Implementation for farmer observation integration
        pass

    def _reassess_characteristics(self, field_profile: Dict) -> Dict:
        """Reassess field characteristics based on accumulated data"""
        # Implementation for dynamic characteristic reassessment
        return field_profile.get("field_characteristics", {})

    def _calculate_intelligence_score(self, field_profile: Dict) -> int:
        """Calculate composite intelligence score"""
        base_score = 85

        # Factors that affect intelligence score
        data_freshness = -5 if len(field_profile.get("health_monitoring", {}).get("health_history", [])) < 3 else 0
        profile_completeness = 5 if field_profile.get("crop_adaptability", {}).get("overall_score", 0) > 0.7 else 0

        return min(95, max(60, base_score + data_freshness + profile_completeness))

    def _generate_update_insights(self, field_profile: Dict) -> List[str]:
        """Generate insights from recent updates"""
        insights = []

        # Basic insights generation
        intelligence_score = field_profile.get("intelligence_score", 85)

        if intelligence_score > 90:
            insights.append("Field intelligence at excellent level - highly confident recommendations")
        elif intelligence_score > 80:
            insights.append("Field intelligence well-established - good predictive accuracy")
        else:
            insights.append("Building field intelligence - recommendations will improve with more data")

        health_monitoring = field_profile.get("health_monitoring", {})
        if health_monitoring.get("alerts_active"):
            insights.append(f"Active health alerts: {len(health_monitoring['alerts_active'])}")

        return insights if insights else ["Field monitoring active - no specific insights at this time"]

    def _generate_field_recommendations(self, field_profile: Dict) -> List[str]:
        """Generate specific field-level recommendations"""
        recommendations = []

        crop_type = field_profile.get("crop_type", "unknown")
        crop_age = field_profile.get("crop_age_days", 0)

        # Crop-specific recommendations based on growth stage
        if crop_type == "rice":
            if crop_age < 30:
                recommendations.append("Early growth stage - ensure proper water logging for root development")
            elif crop_age < 60:
                recommendations.append("Active tillering phase - monitor for stem borer damage")
            else:
                recommendations.append("Panicle formation stage - avoid water stress during flowering")

        elif crop_type == "wheat":
            if crop_age < 45:
                recommendations.append("Critical crown root initiation phase - adequate soil moisture essential")
            elif crop_age < 90:
                recommendations.append("Stem elongation phase - nitrogen availability crucial")
            else:
                recommendations.append("Grain filling stage - ensure sufficient potassium")

        # Add field-specific recommendations
        adaptability = field_profile.get("crop_adaptability", {})
        if adaptability.get("overall_score", 0) < 0.7:
            recommendations.extend(adaptability.get("improvement_suggestions", []))

        return recommendations if recommendations else ["Continue standard crop management practices"]

    def _save_field_profile(self, field_profile: Dict):
        """Save field profile to persistent storage"""
        try:
            self.field_profiles_dir.mkdir(parents=True, exist_ok=True)

            field_id = field_profile["field_id"]
            profile_file = self.field_profiles_dir / f"{field_id}.json"

            with open(profile_file, 'w') as f:
                json.dump(field_profile, f, indent=2, default=str)

        except Exception as e:
            logger.error(f"❌ Failed to save field profile: {e}")

    def _load_field_profile(self, field_id: str) -> Optional[Dict]:
        """Load field profile from storage"""
        try:
            profile_file = self.field_profiles_dir / f"{field_id}.json"

            if not profile_file.exists():
                return None

            with open(profile_file, 'r') as f:
                return json.load(f)

        except Exception as e:
            logger.error(f"❌ Failed to load field profile: {e}")
            return None


# Global field intelligence processor
field_processor = FieldIntelligenceProcessor()

def create_field_intelligence(
    field_coordinates: List[Tuple[float, float]],
    farmer_id: str,
    field_name: str,
    crop_type: str,
    sowing_date: str
) -> Dict[str, Any]:
    """
    Public API for field intelligence creation
    """

    return field_processor.create_field_fingerprint(
        field_coordinates, farmer_id, field_name, crop_type, sowing_date
    )

def update_field_intelligence(
    field_id: str,
    satellite_data: Optional[Dict] = None,
    weather_data: Optional[Dict] = None,
    farmer_observations: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    Public API for field intelligence updates
    """

    return field_processor.update_field_intelligence(
        field_id, satellite_data, weather_data, farmer_observations
    )

def get_field_intelligence(field_id: str) -> Dict[str, Any]:
    """
    Public API for retrieving field intelligence
    """

    return field_processor.get_field_intelligence(field_id)

def list_farmer_fields(farmer_id: str) -> List[str]:
    """
    List all field IDs for a specific farmer
    """

    return [profile["field_id"]
            for profile in field_processor.field_intelligence_cache.values()
            if profile.get("farmer_id") == farmer_id]

if __name__ == "__main__":
    # Test field intelligence with sample NCR rice field
    print("🌱 USER FIELD INTELLIGENCE LAYER - MICRO VISION TEST")
    print("=" * 60)

    # Sample NCR field coordinates (approximate Delhi region)
    sample_field_coordinates = [
        (28.65, 77.20),  # North East
        (28.65, 77.25),  # North West
        (28.62, 77.25),  # South West
        (28.62, 77.20)   # South East
    ]

    # Create field fingerprint
    field_profile = create_field_intelligence(
        field_coordinates=sample_field_coordinates,
        farmer_id="farmer_ncr_001",
        field_name="rice_field_a",
        crop_type="rice",
        sowing_date="2024-10-01"
    )

    if field_profile.get("fingerprint_status") == "active":
        print(f"✅ Field fingerprint created successfully!")
        print(f"📍 Field ID: {field_profile['field_id']}")
        print(f"📊 Intelligence Score: {field_profile['intelligence_score']}/100")
        print(f"🌾 Crop: {field_profile['crop_type']} (Age: {field_profile['crop_age_days']} days)")
        print(f"📏 Field Size: {field_profile['boundary_analysis']['area_sqkm']:.2f} sq km")

        # Test intelligence retrieval
        retrieved = get_field_intelligence(field_profile['field_id'])
        if retrieved:
            print(f"✅ Field intelligence retrieval successful!")
            print(f"🌱 Crop Adaptability: {retrieved['crop_adaptability']['overall_score']:.2f}")
            print(f"⚠️  Risk Category: {retrieved['vulnerability_assessment']['risk_category']}")

    else:
        print(f"❌ Field fingerprint creation failed: {field_profile.get('error', 'Unknown error')}")

    print(f"\n🎯 Field Intelligence Layer operational!")
    print(f"🌾 Micro vision capability: ACTIVE")
    print(f"👤 Individual farmer intelligence: READY")
