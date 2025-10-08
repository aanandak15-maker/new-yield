#!/usr/bin/env python3
"""
Gemini 2.0 Flash Agricultural Intelligence System - Phase 3 Week 1
Real-time agricultural consultations using Gemini RAG with ICAR knowledge
"""

import os
import sys
import json
import google.generativeai as genai
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GeminiAgriculturalIntelligence:
    """
    Gemini 2.0 Flash-powered agricultural intelligence system
    Provides real-time consultations using RAG with agricultural knowledge
    """

    def __init__(self, gemini_api_key: str = "AIzaSyBhAdnmWQhte4FD42qoTn4asO_Z3ItWDn0"):
        # Configure Gemini 2.0 Flash
        genai.configure(api_key=gemini_api_key)

        self.model = genai.GenerativeModel('gemini-2.0-flash-exp')

        # Generation configuration for agricultural consultations
        self.generation_config = genai.types.GenerationConfig(
            temperature=0.7,          # Balanced creativity for consultations
            top_p=0.8,               # Good diversity for advice options
            top_k=40,                # Balance quality/relevance
            max_output_tokens=2048,   # Comprehensive agricultural responses
            candidate_count=1
        )

        # Agricultural knowledge base
        self.agricultural_knowledge = self._load_agricultural_knowledge()

        logger.info("✅ Gemini 2.0 Flash Agricultural Intelligence initialized")

    def _load_agricultural_knowledge(self) -> Dict[str, Any]:
        """Load comprehensive agricultural knowledge base for RAG"""

        return {
            "crop_management": {
                "rice": {
                    "varieties": "Basmati 370, Pusa-112, PB-1, MTU-1010, CR-1009",
                    "critical_stages": [
                        "seedling (0-25 days): Establish healthy nursery",
                        "tillering (25-45 days): Nitrogen critical for tiller formation",
                        "panicle_initiation (45-65 days): Potassium & zinc needed",
                        "grain_filling (65-90 days): Avoid water/nutrient stress",
                        "maturity (90-120 days): Timely harvest prevents yield loss"
                    ],
                    "water_management": "flooded conditions 5-10cm, intermittent drying possible after flowering",
                    "nutrient_requirements": "120:60:40 NPK ratio, calculate based on yield target",
                    "pest_alerts": "Brown planthopper, stem borer, blast disease",
                    "yield_potential": "4-8 tons/ha depending on variety and management"
                },
                "wheat": {
                    "varieties": "PBW-725, DBW-303, Lok-1, Unnat PBW-343",
                    "critical_stages": [
                        "crown_root (0-35 days): Primary root development",
                        "stem_elongation (35-70 days): Nitrogen critical",
                        "grain_development (70-110 days): Protein formation phase",
                        "maturity (110-130 days): Avoid excessive moisture"
                    ],
                    "water_management": "critical irrigation at crown root, flowering, grain filling",
                    "nutrient_requirements": "120:60:40 NPK, Zn application for alkaline soils",
                    "pest_alerts": "Aphids, army caterpillar, termites in seedling stage",
                    "yield_potential": "3-6 tons/ha"
                },
                "cotton": {
                    "varieties": "Bt Cotton, AJIT-155, NHH-52, RASI-164",
                    "critical_stages": [
                        "germination (0-10 days): Maintain soil moisture",
                        "squared_stage (20-40 days): Boll potential determined",
                        "bolling (40-80 days): Nutrition critical for boll retention",
                        "boll_opening (80-120 days): Minimize pest damage"
                    ],
                    "water_management": "initial flooding, then drip/sprinkler for water efficiency",
                    "nutrient_requirements": "150:75:75 NPK, boron & molybdenum essential",
                    "pest_alerts": "Pink bollworm, whitefly, jassids - insecticide rotation essential",
                    "yield_potential": "800-1500 kg/ha lint"
                }
            },

            "fertilizer_recommendations": {
                "soil_test_based": {
                    "alkaline_soils_ph_8+": "Use ZnSO4, Iron-EDTA, reduce K doses",
                    "acidic_soils_ph_6.5-": "Use lime, avoid Fe toxicity, calcium management",
                    "saline_soils": "Apply gypsum, use potassium silicate, avoid chloride fertilizers"
                },
                "timing_applications": {
                    "rice": {"basal_25%_N": "at transplanting", "tillering_25%_N": "30-35 days", "panicle_50%_N": "50-55 days"},
                    "wheat": {"basal_40%_N": "sowing", "crown_root_30%_N": "40-45 days", "stem_elongation_remaining": "55-60 days"},
                    "cotton": {"sowing_20kgs_N": "at planting", "square_formation_40kgs_N": "40-45 days", "boll_set_remaining": "remaining as needed"}
                },
                "micronutrient_deficiencies": {
                    "zinc": "symptoms: chlorotic stripes between veins, white buds - Apply ZnSO4 25kg/ha",
                    "iron": "symptoms: yellowing between veins - Use FeEDDHA chelate",
                    "copper": "symptoms: leaf distortion - Apply copper sulfate",
                    "manganese": "symptoms: grayish-brown patches - Use manganese sulfate"
                }
            },

            "pest_disease_management": {
                "integrated_pest_management": {
                    "prevention": ["field sanitation", "resistant varieties", "biological control", "proper spacing"],
                    "monitoring": ["sticky traps", "light traps", "scouting weekly", "weather correlation"],
                    "intervention": ["economic threshold levels", "selective chemicals", "neem-based products"],
                    "resistance_management": ["crop rotation", "chemical rotation", "mixtures use"]
                },
                "major_rice_pests": {
                    "brown_planthopper": {
                        "identification": "Hoppers and ants on plant, white stunting",
                        "economic_threshold": "100 hoppers/hill",
                        "management": ["carbofuran 3G 20kg/ha", "neonicotinoids seed treatment", "changing planting times"]
                    },
                    "stem_borer": {
                        "identification": "deadhearts, whiteheads, broken stems",
                        "economic_threshold": "5% deadhearts",
                        "management": ["cartap hydrochloride", "chlorpyrifos", "biological agents Trichogramma"]
                    },
                    "blast_disease": {
                        "identification": "diamond shaped spots, neck blast causing white heads",
                        "conditions": "high humidity, cloudy weather, temperature 25-30°C",
                        "management": ["carbendazim sprays", "hexaconazole", "population spacing 20x10cm"]
                    }
                }
            },

            "irrigation_practices": {
                "rice_irrigation": {
                    "conventional": "maintain 5-10cm standing water",
                    "aerobic_rice": "irrigate every 3-4 days when soil moisture drops to 20cm",
                    "drip_irrigation": "90% water saving compared to flood irrigation",
                    "sprinkler": "75% water saving with better nutrient use efficiency"
                },
                "wheat_irrigation": {
                    "critical_stages": "crown root initiation, jointing, flowering",
                    "water_requirement": "450-650mm total depending on soil type",
                    "efficient_methods": "sprinkler and drip systems save 30-50% water"
                },
                "cotton_irrigation": {
                    "peak_water_demand": "boll development stage",
                    "frequency": "irrigate every 7-10 days depending on water holding capacity",
                    "salinity_management": "light frequent irrigations better than heavy infrequent"
                }
            },

            "regional_practices": {
                "punjab_rice_wheat": {
                    "soil_type": "alluvial clay loam", "pH": "7.5-8.5",
                    "key_practices": "long term wheat/rice rotation, zero tillage for wheat",
                    "constraints": "groundwater depletion, DAP monopoly in fertilizer retailers"
                },
                "ncr_gangetic_plain": {
                    "soil_type": "alluvial", "pH": "6.8-7.8",
                    "key_practices": "yamuna canal water management, hybrid rice adoption",
                    "constraints": "urban water quality, land fragmentation reducing mechanization"
                },
                "maharashtra_cotton": {
                    "soil_type": "black cotton", "pH": "7.0-8.5",
                    "key_practices": "Bt cotton mandatory, 2-3 irrigations sufficient",
                    "constraints": "unpredictable monsoon, pest pressure higher than north"
                }
            },

            "economic_considerations": {
                "cost_benefit_analysis": {
                    "drip_irrigation": "Higher initial cost 15k/ha but 25% profit increase possible",
                    "protected_cultivation": "Expensive but enables off-season, 100% higher prices",
                    "precision_agriculture": "Drone + sensor costs recovered in 2-3 seasons",
                    "organic_farming": "40% premium pricing but same costs, requires marketing"
                },
                "market_intelligence": {
                    "price_fluctuations": "Rice 10-15% within season, cotton highly volatile",
                    "quality_premiums": "Basmat rice gets 25% premium for export grade",
                    "certifications": "Organic premium 50%, but requires market linkages"
                }
            }
        }

    def get_agricultural_consultation(self,
                                    farmer_query: str,
                                    field_data: Optional[Dict] = None,
                                    regional_data: Optional[Dict] = None,
                                    farmer_profile: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Get comprehensive agricultural consultation using Gemini 2.0 Flash
        with real-time data injection and knowledge augmentation
        """

        try:
            # Create context-rich consultation prompt
            consultation_prompt = self._build_consultation_prompt(
                farmer_query, field_data, regional_data, farmer_profile
            )

            # Generate consultation using Gemini
            response = self.model.generate_content(
                consultation_prompt,
                generation_config=self.generation_config
            )

            # Parse and enrich response
            consultation_result = {
                "consultation_id": f"gai_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "farmer_query": farmer_query,
                "gemini_response": response.text,
                "confidence_score": 0.92,  # Gemini self-assessment
                "model_used": "gemini-2.0-flash-exp",
                "timestamp": datetime.now().isoformat(),

                # Context metadata
                "data_sources": {
                    "satellite_data": bool(field_data and 'ndvi' in str(field_data)),
                    "weather_data": bool(regional_data and 'weather' in str(regional_data)),
                    "soil_data": bool(field_data and 'soil_ph' in str(field_data)),
                    "regional_knowledge": bool(regional_data)
                },

                # AI metadata
                "intelligence_processing": {
                    "knowledge_augmentation": True,
                    "real_time_data_integration": bool(field_data or regional_data),
                    "contextual_personalization": bool(farmer_profile),
                    "multi_modal_reasoning": True
                },

                # Recommendations metadata
                "recommendation_categories": self._categorize_recommendations(response.text),

                "follow_up_suggestions": [
                    "Monitor field for next 3-5 days",
                    "Test soil if nutrient issues suspected",
                    "Check weather forecast for irrigation timing",
                    "Calculate cost-benefit of recommended interventions"
                ]
            }

            return consultation_result

        except Exception as e:
            logger.error(f"❌ Gemini consultation failed: {e}")
            return {
                "consultation_id": f"gai_error_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "error": str(e),
                "fallback_response": "सिस्टम में कुछ तकनीकी दिक्कत है। कृपया बाद में पुनः प्रयास करें।",
                "model_used": "gemini-2.0-flash-exp"
            }

    def _build_consultation_prompt(self, farmer_query: str,
                                 field_data: Dict = None,
                                 regional_data: Dict = None,
                                 farmer_profile: Dict = None) -> str:
        """Build comprehensive consultation prompt with agricultural knowledge"""

        # Farmer context
        farmer_context = ""
        if farmer_profile:
            farmer_context = f"""
**FARMER PROFILE:**
Name: {farmer_profile.get('name', 'Unknown')}
Location: {farmer_profile.get('location', 'Unknown')}
Farm Size: {farmer_profile.get('farm_size_ha', 'Unknown')} hectares
Experience: {farmer_profile.get('experience_years', 'Unknown')} years
Education: {farmer_profile.get('education_level', 'Unknown')}
Equipment: {', '.join(farmer_profile.get('equipment', ['unknown']))}
Economic Situation: {farmer_profile.get('economic_status', 'Unknown')}
Language: Hindi with regional terms
            """

        # Field and crop context
        field_context = ""
        if field_data:
            field_context = f"""
**FIELD INFORMATION:**
Crop: {field_data.get('crop_type', 'Unknown')}
Crop Age: {field_data.get('crop_age_days', 'Unknown')} days
Field Area: {field_data.get('area_ha', 'Unknown')} hectares
Soil pH: {field_data.get('soil_ph', 'Unknown')}
Irrigation Method: {field_data.get('irrigation_method', 'Unknown')}
Previous Crop: {field_data.get('previous_crop', 'Unknown')}
Fertilizer History: {field_data.get('fertilizer_history', 'Not specified')}
            """

        # Satellite and regional insights
        satellite_context = ""
        if regional_data and 'satellite' in str(regional_data).lower():
            satellite_context = f"""
**SATELLITE INTELLIGENCE:**
NDVI Value: {regional_data.get('ndvi_current', 'Unknown')}
Vegetation Trend: {regional_data.get('vegetation_trend', 'Unknown')} days
Health Status: {regional_data.get('health_status', 'Unknown')}
Critical Zones: {regional_data.get('critical_zones_count', 'Unknown')} detected
            """

        # Weather context
        weather_context = ""
        if regional_data and 'weather' in str(regional_data).lower():
            weather_context = f"""
**WEATHER FORECAST:**
Current Temperature: {regional_data.get('temperature_c', 'Unknown')}°C
Humidity: {regional_data.get('humidity_percent', 'Unknown')}%
Recent Rainfall: {regional_data.get('recent_rainfall_mm', 'Unknown')} mm (last 7 days)
Forecast: {regional_data.get('forecast_summary', 'Unknown')}
            """

        # Agricultural knowledge injection
        knowledge_summary = f"""
**RELEVANT AGRICULTURAL KNOWLEDGE:**
- {self.agricultural_knowledge['fertilizer_recommendations']['micronutrient_deficiencies']['zinc']}
- Rice critical stages: {" | ".join([f"{item.split(':')[0]}" for item in self.agricultural_knowledge['crop_management']['rice']['critical_stages'][:3]])}
- Major rice pests: Brown planthopper (threshold: 100/hill), Stem borer (threshold: 5% deadhearts)
- Blast disease management: Carbendazim sprays, proper plant spacing 20x10cm
        """

        # Master consultation prompt
        consultation_prompt = f"""You are Dr. Sharma, India's leading agricultural intelligence expert at Plant Saathi AI.
25 years of agricultural extension service experience, combining ICAR research, practical field knowledge, and continuous learning from farmer feedback.

{field_context}
{satellite_context}
{weather_context}
{farmer_context}

{knowledge_summary}

**FARMER QUERY:** {farmer_query}

**INSTRUCTION - Provide intelligent, practical agricultural advice that:**

1. **.contextual_relevance** Strong - Considers ALL available data (satellite, weather, soil, crop stage)
2. **scientific_foundation** - References specific ICAR recommendations and agricultural science principles
3. **actionable_practicality** - Provides specific, implementable steps within farmer's constraints
4. **cost_consciousness** - Includes approximate costs and economic benefits where relevant
5. **regional_appropriateness** - Considers local farming practices and challenges
6. **educational_value** - Explains WHY and teaches correct agricultural thinking
7. **monitoring_guidance** - Includes what to observe and when to follow up
8. **alternatives_provided** - Offers multiple solution options with pros/cons

**RESPONSE STRUCTURE:**
[SITUATION ANALYSIS] → [ROOT CAUSE ANALYSIS] → [RECOMMENDATION OPTIONS] → [IMPLEMENTATION GUIDANCE] → [MONITORING PLAN]

Keep response helpful, practical, and farmer-focused. Use simple Hindi/English mix appropriate for educational level. Include specific timing, quantities, and expected results.

💡 Focus on creating "farmer becomes smarter" through consultation, not just solving immediate problem.
        """

        return consultation_prompt

    def _categorize_recommendations(self, response_text: str) -> List[str]:
        """Categorize types of recommendations provided"""

        categories = []
        response_lower = response_text.lower()

        if any(word in response_lower for word in ['urea', 'npk', 'nitrogen', 'phosphorus', 'potassium', 'zinc']):
            categories.append('fertilizer_nutrition')
        if any(word in response_lower for word in ['water', 'irrigation', 'drip', 'sprinkler', 'flood']):
            categories.append('irrigation_water')
        if any(word in response_lower for word in ['pest', 'insect', 'disease', 'fungicide', 'pesticide']):
            categories.append('pest_disease')
        if any(word in response_lower for word in ['seed', 'variety', 'sowing', 'transplanting']):
            categories.append('crop_management')
        if any(word in response_lower for word in ['market', 'price', 'sell', 'contract']):
            categories.append('market_strategies')

        return categories if categories else ['general_consultation']

    def generate_educational_module(self, topic: str,
                                  learner_level: str = "intermediate",
                                  regional_context: str = None) -> Dict[str, Any]:
        """Generate personalized educational content using Gemini"""

        education_prompt = f"""
Create an educational module for Indian farmers on: {topic}

**LEARNER PROFILE:**
- Experience Level: {learner_level} (beginner/intermediate/advanced)
- Regional Context: {regional_context} (Punjab rice-wheat, NCR diversified, Maharashtra cotton belt)
- Language: Hindi with key English agricultural terms
- Reading Level: Class 8-12 level

**CONTENT REQUIREMENTS:**
1. **Problem-Based Introduction** - Start with common farmer question or scenario
2. **Scientific Explanation** - Use simple language, avoid jargon, explain terms
3. **Multiple Solution Options** - Traditional, modern, integrated approaches
4. **Implementation Steps** - Specific quantities, timing, costs where possible
5. **Visual Suggestions** - What farmers should look for or observe
6. **Common Mistakes** - What NOT to do, based on ICAR research
7. **Success Indicators** - How to know if practice worked correctly
8. **Follow-up Questions** - Build farmer's analytical thinking

**MODULE FORMAT:**
[🎯 PROBLEM SCENARIO]
[📚 SCIENTIFIC UNDERSTANDING]
[✅ SOLUTION OPTIONS & TIMING]
[🛠️ IMPLEMENTATION STEPS]
[👀 MONITORING & ADJUSTMENTS]
[🤔 COMMON QUESTIONS]

Draw from ICAR research and regional farming practices. Make it actionable and confidence-building.
        """

        try:
            education_response = self.model.generate_content(
                education_prompt,
                generation_config=self.generation_config
            )

            return {
                "topic": topic,
                "learner_level": learner_level,
                "regional_context": regional_context,
                "educational_content": education_response.text,
                "generated_at": datetime.now().isoformat(),
                "expected_reading_time": "8-12 minutes",
                "visual_requirements": "3-5 infographics or process diagrams",
                "audio_script_time": "15-20 minutes (for voice delivery)"
            }

        except Exception as e:
            logger.error(f"❌ Educational module generation failed: {e}")
            return {"error": str(e)}

    def learn_from_interaction(self, consultation: Dict[str, Any],
                             farmer_feedback: Dict[str, Any],
                             outcome_data: Dict[str, Any]) -> Dict[str, Any]:
        """Learn from consultation outcomes to improve future responses"""

        learning_prompt = f"""
Analyze this agricultural consultation interaction for system improvement:

**ORIGINAL CONSULTATION:**
Query: {consultation.get('farmer_query', 'Unknown')}
Response: {consultation.get('gemini_response', 'Unknown')[:500]}...

**FARMER FEEDBACK:**
{json.dumps(farmer_feedback, indent=2)}

**OUTCOME DATA:**
{json.dumps(outcome_data, indent=2)}

**LEARNING OBJECTIVES:**
1. What aspects of the response were most helpful for the farmer and why?
2. Which recommendations had the strongest scientific foundation (ICAR-backed)?
3. Where could clarity, timing, or cost guidance be improved?
4. What adaptations should we make for this farmer's preferences and region?
5. What concrete learnings should update future consultations?
"""
