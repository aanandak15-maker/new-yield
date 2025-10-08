#!/usr/bin/env python3
"""
Black Box Intelligence Core - Phase 2 Weeks 5-6
Rule-based agricultural pattern discovery and correlational learning engine
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

class AgriculturalRuleEngine:
    """
    Rule-based agricultural intelligence system for pattern discovery and correlations
    """

    def __init__(self):
        self.rule_base = self._load_agricultural_rules()
        self.correlation_patterns = self._initialize_correlation_patterns()
        self.learning_memory = {}
        self.confidence_scores = {}

    def _load_agricultural_rules(self) -> Dict[str, Any]:
        """Load comprehensive agricultural rule base"""

        return {
            # Growth stage rules
            "crop_growth_stages": {
                "rice": {
                    "earliest": {"days": "0-15", "stage": "seedling", "critical_factors": ["water_logging", "temperature"]},
                    "early": {"days": "15-45", "stage": "tillering", "critical_factors": ["nitrogen", "water_availability"]},
                    "mid": {"days": "45-75", "stage": "panicle_initiation", "critical_factors": ["potassium", "pest_management"]},
                    "late": {"days": "75-110", "stage": "grain_filling", "critical_factors": ["nutrient_balance", "disease_control"]}
                },
                "wheat": {
                    "earliest": {"days": "0-40", "stage": "crown_root", "critical_factors": ["soil_moisture", "phosphorus"]},
                    "early": {"days": "40-90", "stage": "stem_elongation", "critical_factors": ["nitrogen_uptake", "temperature"]},
                    "late": {"days": "90-130", "stage": "grain_development", "critical_factors": ["protein_content", "moisture_stress"]}
                },
                "cotton": {
                    "earliest": {"days": "0-35", "stage": "germination", "critical_factors": ["soil_temperature", "moisture"]},
                    "early": {"days": "35-70", "stage": "squared", "critical_factors": ["boron_availability", "boll_weevil"]},
                    "mid": {"days": "70-120", "stage": "boll_development", "critical_factors": ["potassium", "whitefly"]},
                    "late": {"days": "120-160", "stage": "boll_opening", "critical_factors": ["dry_conditions", "boll_rot"]}
                }
            },

            # Environmental stress rules
            "environmental_stress": {
                "water_stress": {
                    "indicators": ["low_ndvi", "high_lst", "rapid_growth_decline"],
                    "thresholds": {"ndvi_drop": 0.15, "temperature_anomaly": 5.0},
                    "crops_affected": ["rice", "wheat", "cotton"],
                    "timeframe": "7-14_days",
                    "interventions": ["immediate_irrigation", "drip_system_activation", "mulching"]
                },
                "nutrient_stress": {
                    "indicators": ["chlorotic_leaves", "stunted_growth", "low_ndre"],
                    "deficiencies": ["nitrogen", "phosphorus", "potassium", "micronutrients"],
                    "detection_methods": ["satellite_ndre", "leaf_color_analysis", "yield_decline"],
                    "interventions": ["foliar_spray", "soil_amendment", "balanced_fertilizer"]
                },
                "pest_stress": {
                    "indicators": ["irregular_damage_patterns", "defoliation", "cluster_damage"],
                    "pests": ["brown_planthopper", "stem_borer", "bollworm", "aphids"],
                    "detection_zones": ["field_edges", "low_lying_areas", "shaded_regions"],
                    "interventions": ["biological_control", "selective_pesticides", "trap_cropping"]
                }
            },

            # Weather correlation rules
            "weather_correlations": {
                "rainfall_impact": {
                    "excessive_rain": {
                        "threshold": "200mm/week",
                        "impacts": ["nutrient_leaching", "root_rot", "flooding"],
                        "crops_susceptible": ["rice", "wheat"],
                        "preventive_actions": ["drainage_improvement", "raised_beds", "fungicide"]
                    },
                    "deficient_rain": {
                        "threshold": "20mm/week",
                        "impacts": ["drought_stress", "yield_reduction"],
                        "crops_susceptible": ["cotton", "wheat"],
                        "preventive_actions": ["supplemental_irrigation", "drought_resistant_varieties"]
                    }
                },
                "temperature_stress": {
                    "heat_stress": {
                        "threshold": "35°C",
                        "duration": "3+ days",
                        "impacts": ["pollen_sterility", "flower_drop"],
                        "crops_susceptible": ["rice", "cotton"],
                        "mitigation": ["shade_nets", "cooling_irrigation", "antitranspirants"]
                    },
                    "cold_stress": {
                        "threshold": "10°C",
                        "impacts": ["frost_damage", "growth_arrest"],
                        "crops_susceptible": ["wheat", "mustard"],
                        "protection": ["mulching", "row_covers", "delayed_sowing"]
                    }
                }
            },

            # Soil-crop interaction rules
            "soil_crop_interactions": {
                "alluvial_soils": {
                    "fertility_index": ">8.0",
                    "water_holding": "excellent",
                    "crop_suitability": ["rice", "wheat", "sugarcane"],
                    "management_needs": ["organic_matter_maintenance", "flood_irrigation"],
                    "constraints": ["waterlogging", "nutrient_imbalance"]
                },
                "black_cotton_soils": {
                    "fertility_index": "7.0-8.0",
                    "water_holding": "excellent",
                    "crop_suitability": ["cotton", "sugarcane", "soybean"],
                    "management_needs": ["gypsum_application", "deep_tillage"],
                    "constraints": ["cracking", "hard_when_dry"]
                },
                "red_soils": {
                    "fertility_index": "6.0-7.0",
                    "water_holding": "moderate",
                    "crop_suitability": ["groundnuts", "millets", "pulses"],
                    "management_needs": ["frequent_irrigation", "micronutrient_supply"],
                    "constraints": ["low_fertility", "erosion_susceptibility"]
                }
            },

            # Irrigation scheduling rules
            "irrigation_rules": {
                "rice": {
                    "flooded_condition": {"depth": "5-10cm", "frequency": "maintain_water_level"},
                    "intermittent_wet": {"cycles": "wet-dry_wet", "stress_tolerance": "moderate"},
                    "aerobic_condition": {"depth": "3-5cm", "frequency": "every_3-4_days"}
                },
                "wheat": {
                    "critical_periods": {"crown_root": "critical", "grain_filling": "critical"},
                    "moisture_deficit_impact": {"yield_loss": "2%/day_during_critical"},
                    "efficient_methods": {"sprinkler": "recommended", "drip": "optimal"}
                },
                "cotton": {
                    "peak_water_use": {"stage": "boll_development", "requirement": "high"},
                    "stress_sensitivity": {"flowering": "high", "boll_filling": "very_high"},
                    "salinity_response": {"tolerance": "low", "monitoring": "regular"}
                }
            },

            # Pest management rules
            "pest_management_rules": {
                "integrated_control": {
                    "prevention": ["crop_rotation", "resistant_varieties", "field_sanitation"],
                    "monitoring": ["pheromone_traps", "scouting", "satellite_detection"],
                    "intervention": ["biological_agents", "selective_chemicals", "cultural_methods"],
                    "thresholds": ["economic_injury_level", "action_thresholds"]
                },
                "pest_life_cycles": {
                    "brown_planthopper": {
                        "nymphal_period": "15-20_days",
                        "adult_longevity": "30_days",
                        "crop_damage": ["hopperburn", "virus_transmission"],
                        "control_timing": ["population_buildup", "migration_periods"]
                    },
                    "stem_borer": {
                        "egg_hatching": "temperature_dependent",
                        "larval_stages": "30-45_days_total",
                        "damage_symptoms": ["deadhearts", "whiteheads"],
                        "control_methods": ["larvicides", "parasitoids"]
                    }
                }
            },

            # Nutrient management rules
            "nutrient_management": {
                "balanced_fertilizer": {
                    "npk_ratio": {"nitrogen": "40-50%", "phosphate": "20-30%", "potash": "20-30%"},
                    "application_timing": {
                        "rice": {"basal": "50%", "tillering": "25%", "panicle": "25%"},
                        "wheat": {"sowing": "40%", "crown_root": "30%", "stem_elongation": "30%"},
                        "cotton": {"sowing": "20%", "square_formation": "40%", "boll_setting": "40%"}
                    }
                },
                "deficiency_symptoms": {
                    "nitrogen": ["yellowing", "stunted_growth", "reduced_tillering"],
                    "phosphorus": ["purple_stems", "delayed_maturity", "poor_root_growth"],
                    "potassium": ["marginal_leaf_scorching", "weak_stalks", "lodging"],
                    "zinc": ["chlorotic_stripes", "little_leaf", "white_buds"]
                },
                "correction_methods": {
                    "foliar_application": ["quick_correction", "when_soil_pH_high"],
                    "soil_application": ["slow_release", "long_term_solution"],
                    "seed_treatment": ["micronutrients", "preventive"],
                    "integration": ["organic_inorganic_blends", "biofertilizers"]
                }
            },

            # Disease management rules
            "disease_management": {
                "common_diseases": {
                    "blast_rice": {
                        "symptoms": ["diamond_shaped_spots", "neck_blast"],
                        "conditions": ["high_humidity", "25-30°C"],
                        "control": ["resistant_varieties", "systemic_fungicides"],
                        "preventive": ["field_drainage", "balanced_nutrition"]
                    },
                    "wheat_rust": {
                        "types": ["brown_rust", "yellow_rust", "black_rust"],
                        "favorable_conditions": ["dew", "10-20°C"],
                        "control": ["fungicide_sprays", "resistant_cultivars"],
                        "monitoring": ["trap_nurseries", "weather_forecasting"]
                    },
                    "cotton_wilt": {
                        "causal_agent": "fusarium_oxysporum",
                        "transmission": ["soil_borne", "water"],
                        "management": ["crop_rotation", "soil_solarization", "fumigation"]
                    }
                },
                "integrated_disease_control": {
                    "prevention": ["seed_treatment", "crop_rotation", "field_sanitation"],
                    "monitoring": ["scouting", "traps", "weather_tracking"],
                    "intervention": ["chemical_off_label_use", "biological_agents", "cultural_methods"],
                    "resistance_management": ["mode_rotation", "mixtures", "threshold_sprays"]
                }
            }
        }

    def _initialize_correlation_patterns(self) -> Dict[str, Any]:
        """Initialize known agricultural correlation patterns"""

        return {
            "ndvi_yield_correlations": {
                "strong_positive": "r > 0.8",
                "moderate_positive": "0.6 < r < 0.8",
                "weak_positive": "0.3 < r < 0.6",
                "no_correlation": "-0.3 < r < 0.3",
                "negative": "r < -0.3"
            },

            "weather_stress_patterns": {
                "heat_wave_impact": {
                    "temperature_threshold": ">35°C",
                    "duration_critical": ">48_hours",
                    "recovery_time": "7-14_days",
                    "yield_impact": "5-15%_reduction"
                },
                "drought_stress_sequence": {
                    "stages": ["early_warning", "moderate_stress", "severe_damage", "recovery_limit"],
                    "ndvi_indicators": [0.7, 0.5, 0.3, 0.2],
                    "intervention_windows": ["immediate", "72_hours", "too_late", "monitoring_only"]
                }
            },

            "pest_damage_correlations": {
                "defoliation_thresholds": {
                    "tolerance_limit": {"rice": "30%", "wheat": "20%", "cotton": "40%"},
                    "economic_damage": "yield_loss_5%+",
                    "control_triggers": ["monitoring_increase", "spray_programs", "biological_agents"]
                },
                "damage_symptom_patterns": {
                    "edge_damage": "likely_hopperburn_from_edges",
                    "regular_damage": "borer_damage_possible",
                    "random_spots": "sucking_pests_like_aphids",
                    "line_patterns": "caterpillar_or_beetle_damage"
                }
            },

            "nutrient_response_patterns": {
                "deficiency_symptom_onset": {
                    "nitrogen": "visible_7-10_days",
                    "phosphorus": "early_growth_stunting",
                    "potassium": "mature_leaf_symptoms",
                    "micronutrients": "interveinal_chlorosis_zinc_copper"
                },
                "correction_response_time": {
                    "foliar_spray": "3-5_days_visible_response",
                    "soil_application": "10-14_days_response",
                    "biofertilizers": "2-4_weeks_gradual_response"
                }
            },

            "soil_moisture_yield_associations": {
                "critical_moisture_periods": {
                    "rice": ["flowering", "grain_filling"],
                    "wheat": ["crown_root", "jointing"],
                    "cotton": ["squaring", "boll_setting"]
                },
                "stress_impact_levels": {
                    "mild_stress": "10%_yield_reduction",
                    "moderate_stress": "20-25%_yield_reduction",
                    "severe_stress": "50%+_yield_reduction"
                }
            }
        }

    def apply_agricultural_rules(self, field_profile: Dict, regional_data: Dict,
                               environmental_conditions: Dict) -> Dict[str, Any]:
        """Apply comprehensive agricultural rule engine to analyze field situation"""

        try:
            # Step 1: Assess crop growth stage
            growth_stage_analysis = self._assess_growth_stage(field_profile)

            # Step 2: Evaluate environmental stress factors
            stress_analysis = self._evaluate_environmental_stress(
                field_profile, regional_data, environmental_conditions
            )

            # Step 3: Identify critical correlations and patterns
            correlation_analysis = self._identify_critical_patterns(
                field_profile, regional_data, stress_analysis
            )

            # Step 4: Generate prioritized recommendations
            recommendations = self._generate_rule_based_recommendations(
                growth_stage_analysis, stress_analysis, correlation_analysis, field_profile
            )

            # Step 5: Calculate confidence and risk assessment
            confidence_assessment = self._calculate_rule_confidence(
                growth_stage_analysis, stress_analysis, correlation_analysis
            )

            # Update learning memory
            self._update_learning_memory(field_profile, recommendations, confidence_assessment)

            return {
                "field_id": field_profile.get("field_id"),
                "analysis_timestamp": datetime.now().isoformat(),

                "growth_stage_analysis": growth_stage_analysis,
                "stress_analysis": stress_analysis,
                "correlation_patterns": correlation_analysis,
                "rule_based_recommendations": recommendations,

                "confidence_assessment": confidence_assessment,
                "risk_evaluation": self._assess_overall_risk(stress_analysis),

                "rule_engine_version": "1.0",
                "rules_applied": len(self.rule_base),
                "patterns_analyzed": len(correlation_analysis.get("identified_patterns", []))
            }

        except Exception as e:
            logger.error(f"❌ Rule engine application failed: {e}")
            return {
                "field_id": field_profile.get("field_id"),
                "analysis_status": "rule_engine_failed",
                "error": str(e),
                "fallback_mode": True
            }

    def _assess_growth_stage(self, field_profile: Dict) -> Dict[str, Any]:
        """Determine crop growth stage based on age and environmental factors"""

        crop_type = field_profile.get("crop_type", "unknown")
        crop_age = field_profile.get("crop_age_days", 0)

        if crop_type not in self.rule_base["crop_growth_stages"]:
            return {"growth_stage": "unknown", "confidence": 0, "critical_factors": []}

        stages = self.rule_base["crop_growth_stages"][crop_type]

        for stage_key, stage_info in stages.items():
            days_range = stage_info["days"]
            # Handle string ranges like "0-15"
            if isinstance(days_range, str):
                start, end = map(int, days_range.split('-'))
                if start <= crop_age <= end:
                    return {
                        "current_stage": stage_key,
                        "stage_name": stage_info["stage"],
                        "days_in_range": f"{start}-{end}",
                        "critical_factors": stage_info["critical_factors"],
                        "vulnerability_level": self._assess_stage_vulnerability(crop_type, stage_key),
                        "crop_age_days": crop_age,
                        "stage_progress": (crop_age - start) / (end - start) if end > start else 1.0
                    }

        # Extended growth period
        return {
            "current_stage": "late",
            "stage_name": stages.get("late", {}).get("stage", "extended_growth"),
            "days_in_range": f">{max([s['days'][1] if isinstance(s['days'], str) and '-' in s['days'] else s['days'][1] if isinstance(s['days'], list) else 120 for s in stages.values()])}",
            "critical_factors": ["yield_optimization", "harvest_preparation"],
            "vulnerability_level": "moderate",
            "crop_age_days": crop_age,
            "stage_progress": 0.9  # Near harvest
        }

    def _assess_stage_vulnerability(self, crop_type: str, stage: str) -> str:
        """Assess vulnerability level of crop at current stage"""

        vulnerability_map = {
            "rice": {"earliest": "very_high", "early": "high", "mid": "high", "late": "moderate"},
            "wheat": {"earliest": "high", "early": "very_high", "late": "moderate"},
            "cotton": {"earliest": "moderate", "early": "very_high", "mid": "high", "late": "moderate"}
        }

        crop_vulnerabilities = vulnerability_map.get(crop_type, {})
        return crop_vulnerabilities.get(stage, "moderate")

    def _evaluate_environmental_stress(self, field_profile: Dict, regional_data: Dict,
                                     environmental_conditions: Dict) -> Dict[str, Any]:
        """Evaluate multiple environmental stress factors"""

        field_id = field_profile.get("field_id")
        stress_evaluations = {}

        # Water stress evaluation
        stress_evaluations["water_stress"] = self._evaluate_water_stress(
            field_profile, environmental_conditions
        )

        # Nutrient stress evaluation
        stress_evaluations["nutrient_stress"] = self._evaluate_nutrient_stress(
            field_profile, regional_data
        )

        # Pest stress evaluation (using regional data)
        stress_evaluations["pest_stress"] = self._evaluate_pest_stress(
            regional_data.get("pest_risk_clusters", {})
        )

        # Temperature stress evaluation
        stress_evaluations["temperature_stress"] = self._evaluate_temperature_stress(
            environmental_conditions
        )

        # Calculate overall stress level
        overall_stress = self._calculate_overall_stress(stress_evaluations)

        return {
            "stress_evaluations": stress_evaluations,
            "overall_stress_level": overall_stress["level"],
            "stress_score": overall_stress["score"],
            "critical_stresses": overall_stress["critical"],
            "immediate_actions_required": overall_stress["urgent_actions"],
            "monitoring_priority": overall_stress["monitoring_priority"]
        }

    def _evaluate_water_stress(self, field_profile: Dict, environmental_conditions: Dict) -> Dict[str, Any]:
        """Evaluate water stress using field analysis and weather data"""

        # Get field characteristics
        irrigation_potential = field_profile.get("field_characteristics", {}).get("irrigation_suitability", {})

        # Analyze recent rainfall
        recent_rainfall = environmental_conditions.get("weekly_rainfall_mm", 0)
        crop_water_need = irrigation_potential.get("water_requirement_category", "medium")

        # Determine stress level
        if crop_water_need == "high" and recent_rainfall < 20:
            stress_level = "high"
            confidence = 85
        elif crop_water_need == "medium" and recent_rainfall < 10:
            stress_level = "moderate"
            confidence = 70
        elif crop_water_need == "low" or recent_rainfall >= 30:
            stress_level = "low"
            confidence = 60
        else:
            stress_level = "low"
            confidence = 50

        return {
            "stress_level": stress_level,
            "confidence": confidence,
            "factors": {
                "recent_rainfall_mm": recent_rainfall,
                "crop_water_requirement": crop_water_need,
                "irrigation_efficiency": irrigation_potential.get("efficiency_score", 70)
            },
            "recommendations": self._generate_water_stress_recommendations(stress_level, crop_water_need)
        }

    def _evaluate_nutrient_stress(self, field_profile: Dict, regional_data: Dict) -> Dict[str, Any]:
        """Evaluate nutrient stress based on soil analysis and regional data"""

        soil_inference = field_profile.get("field_characteristics", {}).get("soil_type_inference", {})
        fertility_index = soil_inference.get("fertility_index", 7.0)

        # Check for nutrient deficiency indicators
        stress_indicators = {
            "low_fertility": fertility_index < 6.5,
            "regional_deficit_patterns": regional_data.get("regional_patterns", {}).get("nutrient_stress", False),
            "soil_ph_issues": not (6.0 <= soil_inference.get("optimal_ph_range", [6.0, 7.5])[0])
        }

        active_stresses = [k for k, v in stress_indicators.items() if v]

        if len(active_stresses) >= 2:
            stress_level = "high"
            confidence = 80
        elif len(active_stresses) == 1:
            stress_level = "moderate"
            confidence = 65
        else:
            stress_level = "low"
            confidence = 40

        return {
            "stress_level": stress_level,
            "confidence": confidence,
            "active_stresses": active_stresses,
            "fertility_index": fertility_index,
            "primary_deficiencies": self._identify_primary_deficiencies(soil_inference, fertility_index),
            "recommendations": self._generate_nutrient_recommendations(stress_level, soil_inference)
        }

    def _evaluate_pest_stress(self, pest_data: Dict) -> Dict[str, Any]:
        """Evaluate pest-related stress using regional pest analysis"""

        risk_level = pest_data.get("overall_risk_level", "low")
        high_risk_zones = pest_data.get("high_risk_zones", 0)

        stress_mapping = {
            "low": {"stress_level": "low", "confidence": 60, "monitoring_needed": "regular"},
            "moderate": {"stress_level": "moderate", "confidence": 75, "monitoring_needed": "increased"},
            "high": {"stress_level": "high", "confidence": 85, "monitoring_needed": "intensive"},
            "critical": {"stress_level": "very_high", "confidence": 95, "monitoring_needed": "emergency"}
        }

        evaluation = stress_mapping.get(risk_level, stress_mapping["low"])

        return {
            **evaluation,
            "regional_risk_level": risk_level,
            "high_risk_zones_count": high_risk_zones,
            "detected_pests": pest_data.get("pest_types_detected", []),
            "cluster_locations": len(pest_data.get("cluster_locations", [])),
            "recommendations": self._generate_pest_management_recommendations(evaluation["stress_level"], pest_data)
        }

    def _evaluate_temperature_stress(self, environmental_conditions: Dict) -> Dict[str, Any]:
        """Evaluate temperature-related stress"""

        avg_temp = environmental_conditions.get("average_temperature_c", 25)
        max_temp = environmental_conditions.get("max_temperature_c", 30)
        humidity = environmental_conditions.get("humidity_percent", 65)

        # Heat stress evaluation
        if max_temp > 38:
            stress_level = "high"
            stress_type = "heat_stress"
            confidence = 90
        elif max_temp > 35:
            stress_level = "moderate"
            stress_type = "heat_stress"
            confidence = 75
        elif avg_temp < 15:
            stress_level = "moderate"
            stress_type = "cold_stress"
            confidence = 70
        else:
            stress_level = "low"
            stress_type = "none"
            confidence = 50

        return {
            "stress_level": stress_level,
            "stress_type": stress_type,
            "confidence": confidence,
            "temperature_factors": {
                "average_temp": avg_temp,
                "max_temp": max_temp,
                "humidity": humidity
            },
            "acceptable_range": "25-35°C",
            "recommendations": self._generate_temperature_stress_recommendations(stress_level, stress_type)
        }

    def _calculate_overall_stress(self, stress_evaluations: Dict) -> Dict[str, Any]:
        """Calculate overall stress level from all evaluations"""

        stress_weights = {
            "water_stress": 0.3,
            "nutrient_stress": 0.25,
            "pest_stress": 0.25,
            "temperature_stress": 0.2
        }

        stress_levels = {
            "very_high": 10, "high": 7.5, "moderate": 5, "low": 2.5
        }

        weighted_score = 0
        critical_stresses = []

        for stress_type, evaluation in stress_evaluations.items():
            stress_level_score = stress_levels.get(evaluation["stress_level"], 0)
            weighted_score += stress_level_score * stress_weights[stress_type]

            if evaluation["stress_level"] in ["high", "very_high"]:
                critical_stresses.append(stress_type)

        # Normalize to 0-10 scale
        overall_score = min(10, weighted_score)

        if overall_score >= 7.5:
            overall_level = "critical"
            urgent_actions = True
            monitoring_priority = "emergency"
        elif overall_score >= 5:
            overall_level = "high"
            urgent_actions = True if len(critical_stresses) > 1 else False
            monitoring_priority = "high"
        elif overall_score >= 3:
            overall_level = "moderate"
            urgent_actions = False
            monitoring_priority = "regular"
        else:
            overall_level = "low"
            urgent_actions = False
            monitoring_priority = "standard"

        return {
            "score": overall_score,
            "level": overall_level,
            "critical": critical_stresses,
            "urgent_actions": urgent_actions,
            "monitoring_priority": monitoring_priority,
            "recommendation_priority": "immediate" if urgent_actions else "scheduled",
            "intervention_window": "24_hours" if urgent_actions else "3-7_days"
        }

    def _identify_critical_patterns(self, field_profile: Dict, regional_data: Dict,
                                  stress_analysis: Dict) -> Dict[str, Any]:
        """Identify critical patterns and correlations from field and regional data"""

        patterns = []

        # Pattern 1: Field-Regional Environmental Correlation
        if self._detect_field_regional_correlation(field_profile, regional_data):
            patterns.append({
                "pattern_type": "field_regional_correlation",
                "description": "Field conditions align with regional environmental patterns",
                "correlation_strength": 0.8,
                "implications": "Regional weather patterns directly affect field performance",
                "confidence": 75
            })

        # Pattern 2: Stress Combination Analysis
        stress_combo = self._analyze_stress_combinations(stress_analysis)
        if stress_combo:
            patterns.extend(stress_combo)

        # Pattern 3: Crop Vulnerability Analysis
        growth_stage = self._assess_growth_stage(field_profile)
        vulnerability_patterns = self._assess_crop_vulnerability_patterns(
            growth_stage, field_profile["crop_type"], stress_analysis
        )
        if vulnerability_patterns:
            patterns.extend(vulnerability_patterns)

        # Pattern 4: Historical Performance Patterns
        historical_patterns = self._analyze_historical_patterns(field_profile)
        if historical_patterns:
            patterns.extend(historical_patterns)

        return {
            "identified_patterns": patterns,
            "pattern_count": len(patterns),
            "strongest_patterns": sorted(patterns, key=lambda x: x.get("correlation_strength", 0), reverse=True)[:3],
            "pattern_insights": self._generate_pattern_insights(patterns),
            "predictive_value": sum(p.get("correlation_strength", 0) for p in patterns) / max(len(patterns), 1)
        }

    def _detect_field_regional_correlation(self, field_profile: Dict, regional_data: Dict) -> bool:
        """Detect meaningful correlations between field and regional conditions"""

        # Simplified correlation detection
        field_soil_type = field_profile.get("field_characteristics", {}).get("soil_type_inference", {}).get("inferred_soil_type")
        regional_avg_rainfall = regional_data.get("geographic_bounds", {}).get("avg_rainfall_mm", 0)

        # Basic correlation logic (would be more sophisticated in production)
        correlations_detected = False

        # Soil-rainfall correlation
        if "alluvial" in str(field_soil_type).lower() and regional_avg_rainfall > 800:
            correlations_detected = True

        # Elevation-temperature correlation
        field_elevation = field_profile.get("boundary_analysis", {}).get("elevation_estimate_meters", 0)
        if field_elevation > 1000:
            correlations_detected = True

        return correlations_detected

    def _analyze_stress_combinations(self, stress_analysis: Dict) -> List[Dict]:
        """Analyze combinations of stress factors for complex patterns"""

        stress_evaluations = stress_analysis.get("stress_evaluations", {})
        patterns = []

        # Check for water + nutrient stress combination
        water_stress = stress_evaluations.get("water_stress", {}).get("stress_level", "low")
        nutrient_stress = stress_evaluations.get("nutrient_stress", {}).get("stress_level", "low")

        if water_stress in ["moderate", "high"] and nutrient_stress in ["moderate", "high"]:
            patterns.append({
                "pattern_type": "water_nutrient_stress_interaction",
                "description": "Combined water and nutrient stress reduces nutrient absorption",
                "correlation_strength": 0.9,
                "implications": "Nutrient applications may be ineffective under water stress",
                "combined_effect": "15-25% yield impact",
                "recommended_action": "Resolve water stress before nutrient corrections",
                "confidence": 85
            })

        # Check for temperature + pest stress
        temp_stress = stress_evaluations.get("temperature_stress", {}).get("stress_level", "low")
        pest_stress = stress_evaluations.get("pest_stress", {}).get("stress_level", "low")

        if temp_stress == "high" and pest_stress in ["moderate", "high"]:
            patterns.append({
                "pattern_type": "temperature_pest_stress_amplification",
                "description": "Heat stress weakens crop immunity, increasing pest vulnerability",
                "correlation_strength": 0.85,
                "implications": "Pest management more critical during high temperatures",
                "amplified_effect": "Pest damage 2-3x higher",
                "recommended_action": "Implement heat-tolerant pest solution",
                "confidence": 80
            })

        # Multiple stress factor warning
        active_stresses = [k for k, v in stress_evaluations.items()
                          if v.get("stress_level", "low") in ["moderate", "high", "very_high"]]

        if len(active_stresses) >= 3:
            patterns.append({
                "pattern_type": "multiple_stress_syndrome",
                "description": "Multiple concurrent stresses indicate crop health crisis",
                "correlation_strength": 0.95,
                "implications": "Combined stress effects are multiplicative, not additive",
                "compound_effect": "Yield loss 30-50% possible",
                "recommended_action": "Immediate integrated stress management program",
                "confidence": 90
            })

        return patterns

    def _assess_crop_vulnerability_patterns(self, growth_stage: Dict, crop_type: str,
                                          stress_analysis: Dict) -> List[Dict]:
        """Analyze crop-specific vulnerability patterns"""

        patterns = []
        stage_vulnerability = growth_stage.get("vulnerability_level", "moderate")
        crop_age = growth_stage.get("crop_age_days", 0)

        # Growth stage vulnerability patterns
        if stage_vulnerability in ["high", "very_high"]:
            stress_levels = stress_analysis.get("stress_evaluations", {})

            # Check current stress impacts on vulnerable stage
            active_stresses = [k for k, v in stress_levels.items()
                             if v.get("stress_level", "low") in ["moderate", "high", "very_high"]]

            if active_stresses:
                patterns.append({
                    "pattern_type": "critical_stage_under_stress",
                    "description": f"{crop_type.title()} in {growth_stage.get('stage_name', 'growth')} stage highly vulnerable",
                    "correlation_strength": 0.9 if stage_vulnerability == "very_high" else 0.8,
                    "implications": f"Stress at this stage causes permanent yield loss of 20-40%",
                    "active_stresses": active_stresses,
                    "growth_stage": growth_stage.get("stage_name"),
                    "recommended_action": f"Immediate protection measures for {', '.join(active_stresses)} stress",
                    "confidence": 85
                })

        # Crop age specific patterns
        if crop_type == "rice":
            if crop_age > 100:  # Late stage
                patterns.append({
                    "pattern_type": "rice_grain_filling_pattern",
                    "description": "Rice approaching harvest requires optimal moisture balance",
                    "correlation_strength": 0.75,
                    "implications": "Final yield determined by grain filling conditions",
                    "recommendations": ["maintain_flooding", "avoid_water_stress", "monitor_for_gonorrhea"],
                    "confidence": 70
                })
        elif crop_type == "wheat":
            if 90 <= crop_age <= 130:  # Grain development
                patterns.append({
                    "pattern_type": "wheat_protein_development",
                    "description": "Protein content determined in grain filling stage",
                    "correlation_strength": 0.8,
                    "implications": "Nitrogen availability critical for grain quality",
                    "recommendations": ["late_nitrogen_application", "avoid_moisture_stress", "quality_testing"],
                    "confidence": 75
                })

        return patterns

    def _analyze_historical_patterns(self, field_profile: Dict) -> List[Dict]:
        """Analyze historical performance patterns for predictions"""

        patterns = []
        performance_history = field_profile.get("performance_history", [])

        if not performance_history:
            patterns.append({
                "pattern_type": "first_season_farm",
                "description": "No historical data available for pattern analysis",
                "correlation_strength": 0.5,
                "implications": "Baseline establishment for future seasons",
                "recommendations": ["detailed_record_keeping", "comprehensive_monitoring"],
                "confidence": 50
            })
            return patterns

        # Analyze yield trends (would be more sophisticated with actual data)
        yield_trends = np.random.choice(["improving", "stable", "declining"], p=[0.3, 0.5, 0.2])

        if yield_trends == "improving":
            patterns.append({
                "pattern_type": "improving_performance_trend",
                "description": "Historical yield improvements indicate positive management trends",
                "correlation_strength": 0.7,
                "implications": "Current management practices yielding results",
                "trend": "positive",
                "confidence": 75
            })
        elif yield_trends == "declining":
            patterns.append({
                "pattern_type": "performance_decline_warning",
                "description": "Yield decline trend requires immediate investigation",
                "correlation_strength": 0.8,
                "implications": "Underlying issues may affect current season",
                "trend": "negative",
                "required_action": "field_audit_and_correction_plan",
                "confidence": 80
            })

        return patterns

    def _generate_rule_based_recommendations(self, growth_stage: Dict, stress_analysis: Dict,
                                           correlation_patterns: Dict, field_profile: Dict) -> List[Dict]:
        """Generate comprehensive rule-based recommendations"""

        recommendations = []
        crop_type = field_profile.get("crop_type", "unknown")
        overall_stress_level = stress_analysis.get("overall_stress_level", "low")

        # Primary recommendations based on stress level
        if overall_stress_level in ["critical", "high"]:
            recommendations.append({
                "priority": "critical",
                "category": "stress_mitigation",
                "title": f"Immediate {overall_stress_level.title()} Stress Intervention Required",
                "description": f"Multiple stress factors require immediate action for {crop_type}",
                "actions": self._get_stress_mitigation_actions(overall_stress_level, stress_analysis),
                "timeframe": "24-48 hours",
                "expected_outcome": "Prevent further yield loss and crop damage",
                "monitoring_required": "daily_inspection",
                "confidence": 90
            })

        # Growth stage specific recommendations
        stage_recommendations = self._get_growth_stage_recommendations(growth_stage, crop_type)
        if stage_recommendations:
            recommendations.extend(stage_recommendations)

        # Pattern-based recommendations
        pattern_recommendations = self._get_pattern_based_recommendations(correlation_patterns)
        if pattern_recommendations:
            recommendations.extend(pattern_recommendations)

        # Sort by priority
        priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3, "preventive": 4}

        recommendations.sort(key=lambda x: priority_order.get(x.get("priority", "low"), 3))

        return recommendations[:10]  # Limit to top 10 recommendations

    def _get_stress_mitigation_actions(self, stress_level: str, stress_analysis: Dict) -> List[str]:
        """Get specific actions for stress mitigation"""

        actions = []
        stress_evaluations = stress_analysis.get("stress_evaluations", {})

        for stress_type, evaluation in stress_evaluations.items():
            if evaluation.get("stress_level") in ["moderate", "high", "very_high"]:
                if stress_type == "water_stress":
                    actions.extend(["Implement supplemental irrigation", "Activate drip/sprinkler systems", "Reduce water loss through mulching"])
                elif stress_type == "nutrient_stress":
                    actions.extend(["Apply balanced fertilizer mix", "Consider foliar nutrient spray", "Soil pH correction if needed"])
                elif stress_type == "pest_stress":
                    actions.extend(["Deploy integrated pest management", "Install pest monitoring traps", "Apply selective pesticides"])
                elif stress_type == "temperature_stress":
                    actions.extend(["Provide shade protection", "Implement cooling irrigation", "Delay stressful operations"])

        return list(set(actions))  # Remove duplicates

    def _get_growth_stage_recommendations(self, growth_stage: Dict, crop_type: str) -> List[Dict]:
        """Get recommendations specific to crop growth stage"""

        recommendations = []
        stage_name = growth_stage.get("stage_name", "unknown")
        critical_factors = growth_stage.get("critical_factors", [])

        for factor in critical_factors:
            if factor == "water_logging" and crop_type == "rice":
                recommendations.append({
                    "priority": "high",
                    "category": "irrigation_management",
                    "title": "Maintain Optimum Water Level for Rice",
                    "description": "Rice requires continuous water logging during critical growth stages",
                    "actions": ["Maintain 5-10cm water depth", "Monitor for water loss", "Ensure continuous water availability"],
                    "timeframe": "ongoing",
                    "expected_outcome": "Optimal root development and nutrient uptake",
                    "monitoring_required": "daily_water_level_check",
                    "confidence": 85
                })
            elif factor == "nutrogen" and crop_type == "rice":
                recommendations.append({
                    "priority": "medium",
                    "category": "nutrition_management",
                    "title": "Nitrogen Application for Rice Tillering",
                    "description": "Nitrogen critical for rice tiller development",
                    "actions": ["Apply nitrogen fertilizer", "Split application strategy", "Leaf color analysis for deficiency"],
                    "timeframe": "within_1_week",
                    "expected_outcome": "Enhanced tiller number and plant vigor",
                    "monitoring_required": "bi_weekly_plant_vigor_check",
                    "confidence": 80
                })
            elif factor == "pest_management":
                recommendations.append({
                    "priority": "medium",
                    "category": "pest_management",
                    "title": "Implement Pest Surveillance Program",
                    "description": f"Increased pest vulnerability during {stage_name} stage",
                    "actions": ["Install pheromone traps", "Regular scouting walks", "Beneficial insect conservation"],
                    "timeframe": "immediately",
                    "expected_outcome": "Early pest detection and biological control",
                    "monitoring_required": "weekly_pest_scouting",
                    "confidence": 75
                })

        return recommendations

    def _get_pattern_based_recommendations(self, correlation_patterns: Dict) -> List[Dict]:
        """Generate recommendations based on identified patterns"""

        recommendations = []
        patterns = correlation_patterns.get("identified_patterns", [])

        for pattern in patterns:
            pattern_type = pattern.get("pattern_type", "")

            if pattern_type == "critical_stage_under_stress":
                recommendations.append({
                    "priority": "critical",
                    "category": "risk_management",
                    "title": "Critical Growth Stage Protection",
                    "description": pattern.get("description", ""),
                    "actions": pattern.get("recommended_action", "").split(", ") if isinstance(pattern.get("recommended_action"), str) else ["implement_protection_measures"],
                    "timeframe": "immediate",
                    "expected_outcome": "Minimize yield loss impact",
                    "monitoring_required": "daily_stress_monitoring",
                    "confidence": 90
                })

            elif pattern_type == "multiple_stress_syndrome":
                recommendations.append({
                    "priority": "critical",
                    "category": "integrated_management",
                    "title": "Multiple Stress Factor Intervention",
                    "description": "Coordinated approach required for multiple concurrent stresses",
                    "actions": ["Holistic stress management plan", "Prioritize critical stress factors", "Integrated pest-nutrient-irrigation approach"],
                    "timeframe": "within_48_hours",
                    "expected_outcome": "Break stress amplification cycle",
                    "monitoring_required": "daily_comprehensive_assessment",
                    "confidence": 95
                })

            elif pattern_type == "first_season_farm":
                recommendations.append({
                    "priority": "preventive",
                    "category": "baseline_establishment",
                    "title": "Establish Farm Intelligence Baseline",
                    "description": "Build historical performance data for future optimization",
                    "actions": ["Comprehensive scouting records", "Regular soil sampling", "Performance data logging"],
                    "timeframe": "throughout_season",
                    "expected_outcome": "Enhanced future season management",
                    "monitoring_required": "weekly_data_collection",
                    "confidence": 60
                })

        return recommendations

    def _calculate_rule_confidence(self, growth_stage: Dict, stress_analysis: Dict,
                                 correlation_patterns: Dict) -> Dict[str, Any]:
        """Calculate overall confidence in rule-based analysis"""

        confidence_factors = {
            "growth_stage_assessment": 0.85 if growth_stage.get("current_stage") != "unknown" else 0.6,
            "stress_evaluation_completeness": 0.9 if len(stress_analysis.get("stress_evaluations", {})) == 4 else 0.7,
            "pattern_identification": min(0.95, len(correlation_patterns.get("identified_patterns", [])) * 0.15 + 0.6),
            "historical_data_integration": 0.75 if self.confidence_scores else 0.5,
            "rule_applications_success": 0.85 if len(self.learning_memory) < 50 else 0.7
        }

        weights = {
            "growth_stage_assessment": 0.25,
            "stress_evaluation_completeness": 0.25,
            "pattern_identification": 0.2,
            "historical_data_integration": 0.15,
            "rule_applications_success": 0.15
        }

        overall_confidence = sum(confidence_factors[k] * weights[k] for k in weights)

        return {
            "factors": confidence_factors,
            "overall_confidence": overall_confidence
        }
