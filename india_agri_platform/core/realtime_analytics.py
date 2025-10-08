"""
Advanced Real-Time Analytics Engine for India Agricultural Intelligence Platform
Intelligent data analysis with predictive insights, anomaly detection, and trend analysis
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import deque
import statistics
from scipy import stats
import warnings

from india_agri_platform.core.error_handling import error_handler, log_system_event
from india_agri_platform.core.cache_manager import cache_manager, set_cached_value
from india_agri_platform.core.data_processing_pipeline import data_pipeline, DataSource
from india_agri_platform.database.manager import db_manager

# Suppress warnings for clean logging
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class AnalyticsType(Enum):
    """Types of analytics computations"""
    TREND_ANALYSIS = "TREND_ANALYSIS"
    ANOMALY_DETECTION = "ANOMALY_DETECTION"
    CORRELATION_ANALYSIS = "CORRELATION_ANALYSIS"
    PREDICTIVE_INSIGHTS = "PREDICTIVE_INSIGHTS"
    SEASONAL_PATTERNS = "SEASONAL_PATTERNS"
    GROWTH_VELOCITY = "GROWTH_VELOCITY"
    RISK_ASSESSMENT = "RISK_ASSESSMENT"

class AlertSeverity(Enum):
    """Alert severity levels"""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

class InsightCategory(Enum):
    """Categories for analytical insights"""
    GROWTH_PATTERNS = "GROWTH_PATTERNS"
    YIELD_PREDICTIONS = "YIELD_PREDICTIONS"
    RISK_WARNINGS = "RISK_WARNINGS"
    OPTIMIZATION_OPPORTUNITIES = "OPTIMIZATION_OPPORTUNITIES"
    CLIMATE_IMPACTS = "CLIMATE_IMPACTS"
    DISEASE_THREATS = "DISEASE_THREATS"
    SOIL_HEALTH = "SOIL_HEALTH"
    IRRIGATION_NEEDS = "IRRIGATION_NEEDS"

@dataclass
class AnalyticsWindow:
    """Time window for analytics computations"""
    start_time: datetime
    end_time: datetime
    data_points: int = 0
    quality_score: float = 0.0
    window_id: str = ""

    def contains_timestamp(self, timestamp: datetime) -> bool:
        """Check if timestamp is within this window"""
        return self.start_time <= timestamp <= self.end_time

    def get_duration_hours(self) -> float:
        """Get window duration in hours"""
        return (self.end_time - self.start_time).total_seconds() / 3600

@dataclass
class StatisticalSummary:
    """Statistical summary of a dataset"""
    count: int
    mean: Optional[float] = None
    median: Optional[float] = None
    std_dev: Optional[float] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    skewness: Optional[float] = None
    kurtosis: Optional[float] = None
    missing_percentage: float = 0.0
    outlier_percentage: float = 0.0

@dataclass
class TrendAnalysis:
    """Trend analysis results"""
    metric_name: str
    trend_direction: str  # "increasing", "decreasing", "stable", "volatile"
    slope: Optional[float] = None
    r_squared: Optional[float] = None
    confidence_level: Optional[float] = None
    seasonality_detected: bool = False
    seasonal_period_days: Optional[int] = None
    trend_strength: str = "weak"  # "weak", "moderate", "strong"

@dataclass
class AnomalyResult:
    """Anomaly detection result"""
    metric_name: str
    value: float
    expected_range: Tuple[float, float]
    deviation_sigma: float
    severity: AlertSeverity
    timestamp: datetime
    is_anomaly: bool = True
    confidence_score: Optional[float] = None

@dataclass
class CorrelationInsight:
    """Correlation analysis result"""
    factor_1: str
    factor_2: str
    correlation_coefficient: float
    p_value: Optional[float] = None
    relationship_strength: str = "weak"  # "weak", "moderate", "strong"
    causal_direction: Optional[str] = None  # "factor1_causes_factor2", "factor2_causes_factor1", "bidirectional"

@dataclass
class PredictiveInsight:
    """Predictive analytics result"""
    insight_type: InsightCategory
    title: str
    description: str
    confidence_score: float
    impact_level: str  # "low", "medium", "high", "critical"
    recommended_action: Optional[str] = None
    timeframe_days: Optional[int] = None
    affected_areas: List[str] = field(default_factory=list)
    supporting_data: Dict[str, Any] = field(default_factory=dict)

class AdvancedRealtimeAnalytics:
    """Real-time analytics engine with predictive capabilities"""

    def __init__(self, analysis_window_hours: int = 24):
        self.analysis_window_hours = analysis_window_hours
        self.analytics_cache = {}

        # Real-time data buffers (sliding windows)
        self.data_buffers: Dict[str, deque] = {}
        self.max_buffer_size = 10000  # Keep last 10k data points

        # Statistical baselines for anomaly detection
        self.baselines: Dict[str, StatisticalSummary] = {}

        # Active alerts and predictions
        self.active_alerts: Dict[str, PredictiveInsight] = {}
        self.prediction_cache: Dict[str, List[PredictiveInsight]] = {}

        # Analytics configuration
        self.analytics_enabled = {
            AnalyticsType.TREND_ANALYSIS: True,
            AnalyticsType.ANOMALY_DETECTION: True,
            AnalyticsType.CORRELATION_ANALYSIS: True,
            AnalyticsType.PREDICTIVE_INSIGHTS: True,
            AnalyticsType.SEASONAL_PATTERNS: True,
            AnalyticsType.GROWTH_VELOCITY: True,
            AnalyticsType.RISK_ASSESSMENT: True
        }

        # Thresholds for alerts
        self.alert_thresholds = {
            'anomaly_sigma_threshold': 3.0,  # 3-sigma rule for anomalies
            'correlation_min_threshold': 0.6,  # Minimum correlation for insights
            'trend_min_confidence': 0.7,  # Minimum R² for trend detection
            'growth_velocity_threshold': 0.1  # Minimum growth rate for alerts
        }

        logger.info("Advanced Real-Time Analytics Engine initialized")

    async def initialize_analytics(self) -> bool:
        """Initialize the analytics engine"""
        try:
            # Start background analytics tasks
            asyncio.create_task(self._background_analytics_loop())

            # Load historical baselines if available
            await self._load_historical_baselines()

            log_system_event(
                "realtime_analytics_initialized",
                "Real-time analytics engine started",
                {"analysis_window_hours": self.analysis_window_hours}
            )

            return True

        except Exception as e:
            error_handler.handle_error(e, {"component": "realtime_analytics", "operation": "initialization"})
            return False

    async def process_realtime_data(self, source: DataSource, data: Any, timestamp: Optional[datetime] = None):
        """Process incoming real-time data for analytics"""

        if timestamp is None:
            timestamp = datetime.utcnow()

        # Convert data to DataFrame if not already
        if not isinstance(data, pd.DataFrame):
            if isinstance(data, dict):
                data = pd.DataFrame([data])
            elif isinstance(data, list) and data and isinstance(data[0], dict):
                data = pd.DataFrame(data)
            else:
                logger.warning(f"Unsupported data format for analytics: {type(data)}")
                return

        # Add timestamp if not present
        if 'timestamp' not in data.columns:
            data['timestamp'] = timestamp

        # Extract numeric metrics for analysis
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        numeric_columns = [col for col in numeric_columns if col != 'timestamp']

        for metric in numeric_columns:
            metric_key = f"{source.value}_{metric}"
            values = data[metric].dropna().values

            # Update sliding window buffer
            await self._update_data_buffer(metric_key, values, timestamp)

        logger.debug(f"Processed real-time data batch: {source.value}, {len(data)} rows")

    async def _update_data_buffer(self, metric_key: str, values: np.ndarray, timestamp: datetime):
        """Update sliding window buffer for a metric"""

        if metric_key not in self.data_buffers:
            self.data_buffers[metric_key] = deque(maxlen=self.max_buffer_size)

        # Add new values to buffer
        for value in values:
            if not np.isnan(value) and not np.isinf(value):
                data_point = {
                    'value': float(value),
                    'timestamp': timestamp,
                    'metric': metric_key
                }
                self.data_buffers[metric_key].append(data_point)

    async def _background_analytics_loop(self):
        """Main loop for continuous analytics computations"""
        while True:
            try:
                await asyncio.sleep(60)  # Run analytics every minute

                # Run all enabled analytics computations
                await self._compute_trend_analysis()
                await self._detect_anomalies()
                await self._compute_correlations()
                await self._generate_predictive_insights()
                await self._analyze_seasonal_patterns()
                await self._calculate_growth_velocity()
                await self._perform_risk_assessment()

                # Clean up old data
                await self._cleanup_old_data()

                # Cache analytics results
                await self._cache_analytics_results()

            except Exception as e:
                logger.error(f"Background analytics loop error: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes on error

    async def _compute_trend_analysis(self):
        """Compute trend analysis for all metrics"""

        if not self.analytics_enabled[AnalyticsType.TREND_ANALYSIS]:
            return

        for metric_key, buffer in self.data_buffers.items():
            if len(buffer) < 10:  # Need minimum data points
                continue

            try:
                # Extract time series data
                values = [point['value'] for point in buffer]
                timestamps = [point['timestamp'] for point in buffer]

                # Convert timestamps to hours since start
                start_time = min(timestamps)
                x_data = [(t - start_time).total_seconds() / 3600 for t in timestamps]

                # Perform linear regression
                if len(x_data) >= 10:
                    slope, intercept, r_value, p_value, std_err = stats.linregress(x_data, values)

                    # Determine trend characteristics
                    r_squared = r_value ** 2
                    trend_direction = self._classify_trend(slope, r_squared)

                    # Check for seasonality (simplified)
                    seasonality_detected = self._detect_seasonality(values, timestamps)

                    trend = TrendAnalysis(
                        metric_name=metric_key,
                        trend_direction=trend_direction,
                        slope=slope,
                        r_squared=r_squared,
                        confidence_level=1 - p_value,
                        seasonality_detected=seasonality_detected,
                        seasonal_period_days=7 if seasonality_detected else None,  # Simplified
                        trend_strength=self._assess_trend_strength(r_squared)
                    )

                    # Store trend analysis
                    cache_key = f"trend_{metric_key}"
                    await set_cached_value(cache_key, {
                        'metric': metric_key,
                        'analysis': trend.__dict__,
                        'timestamp': datetime.utcnow().isoformat()
                    }, ttl_seconds=1800)  # 30 minutes

            except Exception as e:
                logger.warning(f"Trend analysis failed for {metric_key}: {e}")

    async def _detect_anomalies(self):
        """Detect anomalies using statistical methods"""

        if not self.analytics_enabled[AnalyticsType.ANOMALY_DETECTION]:
            return

        for metric_key, buffer in self.data_buffers.items():
            if len(buffer) < 30:  # Need sufficient baseline
                continue

            try:
                values = [point['value'] for point in buffer]

                # Calculate baseline statistics (use recent data)
                recent_values = values[-100:] if len(values) > 100 else values

                mean_val = statistics.mean(recent_values)
                stdev_val = statistics.stdev(recent_values) if len(recent_values) > 1 else 0

                if stdev_val == 0:
                    continue  # Can't detect anomalies without variance

                # Check current values for anomalies
                current_values = values[-5:]  # Last 5 points

                for i, current_val in enumerate(current_values):
                    z_score = abs(current_val - mean_val) / stdev_val

                    if z_score > self.alert_thresholds['anomaly_sigma_threshold']:
                        severity = self._calculate_anomaly_severity(z_score)

                        anomaly = AnomalyResult(
                            metric_name=metric_key,
                            value=current_val,
                            expected_range=(mean_val - 2*stdev_val, mean_val + 2*stdev_val),
                            deviation_sigma=z_score,
                            severity=severity,
                            timestamp=buffer[-(5-i)]['timestamp'],
                            confidence_score=min(1.0, z_score / 5.0)
                        )

                        # Generate alert insight
                        alert_insight = PredictiveInsight(
                            insight_type=InsightCategory.RISK_WARNINGS,
                            title=f"Anomaly Detected in {metric_key}",
                            description=f"Unusual {metric_key} value: {current_val:.2f} (expected: {mean_val:.2f} ± {2*stdev_val:.2f})",
                            confidence_score=anomaly.confidence_score,
                            impact_level=severity.value.lower(),
                            recommended_action="Investigate data source and field conditions",
                            timeframe_days=1,
                            affected_areas=["Unknown"],  # Would be determined from context
                            supporting_data=anomaly.__dict__
                        )

                        alert_key = f"alert_{metric_key}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
                        self.active_alerts[alert_key] = alert_insight

                        logger.warning(f"ANOMALY DETECTED: {alert_insight.title}")

            except Exception as e:
                logger.warning(f"Anomaly detection failed for {metric_key}: {e}")

    async def _compute_correlations(self):
        """Compute correlations between different metrics"""

        if not self.analytics_enabled[AnalyticsType.CORRELATION_ANALYSIS]:
            return

        # Get all metric keys that have sufficient data
        metric_keys = [k for k, v in self.data_buffers.items() if len(v) >= 30]

        for i, metric1 in enumerate(metric_keys):
            for metric2 in metric_keys[i+1:]:
                try:
                    # Extract overlapping time series
                    data1 = {p['timestamp']: p['value'] for p in self.data_buffers[metric1]}
                    data2 = {p['timestamp']: p['value'] for p in self.data_buffers[metric2]}

                    # Find common timestamps
                    common_timestamps = set(data1.keys()) & set(data2.keys())
                    if len(common_timestamps) < 10:
                        continue

                    values1 = [data1[ts] for ts in sorted(common_timestamps)]
                    values2 = [data2[ts] for ts in sorted(common_timestamps)]

                    # Calculate Pearson correlation
                    if len(set(values1)) > 1 and len(set(values2)) > 1:  # Check for variance
                        correlation, p_value = stats.pearsonr(values1, values2)
                        abs_correlation = abs(correlation)

                        if abs_correlation >= self.alert_thresholds['correlation_min_threshold']:
                            relationship_strength = self._assess_correlation_strength(abs_correlation)

                            correlation_insight = CorrelationInsight(
                                factor_1=metric1,
                                factor_2=metric2,
                                correlation_coefficient=correlation,
                                p_value=p_value,
                                relationship_strength=relationship_strength,
                                causal_direction=None  # Simplified - would need domain knowledge
                            )

                            # Generate insight if correlation is strong
                            if relationship_strength in ['strong', 'moderate'] and p_value < 0.05:
                                insight = PredictiveInsight(
                                    insight_type=InsightCategory.OPTIMIZATION_OPPORTUNITIES,
                                    title=f"Factor Relationship: {metric1} ↔ {metric2}",
                                    description=f"{'Positive' if correlation > 0 else 'Negative'} correlation ({correlation:.3f}) between {metric1} and {metric2}",
                                    confidence_score=1 - p_value,
                                    impact_level="medium",
                                    recommended_action="Consider both factors together in optimization strategies",
                                    timeframe_days=None,
                                    supporting_data=correlation_insight.__dict__
                                )

                                # Cache correlation insight
                                corr_key = f"correlation_{metric1}_{metric2}"
                                await set_cached_value(corr_key, insight.__dict__, ttl_seconds=3600)

                except Exception as e:
                    logger.debug(f"Correlation analysis failed for {metric1} vs {metric2}: {e}")

    async def _generate_predictive_insights(self):
        """Generate predictive insights based on current data"""

        if not self.analytics_enabled[AnalyticsType.PREDICTIVE_INSIGHTS]:
            return

        insights = []

        try:
            # Agricultural insights based on current conditions
            weather_ndvi_corr = await self._check_weather_ndvi_correlation()
            if weather_ndvi_corr:
                insights.append(weather_ndvi_corr)

            # Growth stage prediction based on NDVI trends
            growth_insights = await self._predict_growth_stage()
            insights.extend(growth_insights)

            # Irrigation needs based on soil moisture
            irrigation_insights = await self._analyze_irrigation_needs()
            insights.extend(irrigation_insights)

            # Disease risk assessment
            disease_insights = await self._assess_disease_risk()
            insights.extend(disease_insights)

            # Market timing suggestions
            market_insights = await self._suggest_market_timing()
            insights.extend(market_insights)

            # Cache predictive insights
            for insight in insights:
                if insight.confidence_score > 0.7:  # Only cache high-confidence insights
                    insight_key = f"insight_{insight.insight_type.value}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
                    await set_cached_value(insight_key, insight.__dict__, ttl_seconds=7200)  # 2 hours

                    # Store in prediction cache
                    if insight.insight_type.value not in self.prediction_cache:
                        self.prediction_cache[insight.insight_type.value] = []
                    self.prediction_cache[insight.insight_type.value].append(insight)
                    # Keep only last 20 insights per type
                    if len(self.prediction_cache[insight.insight_type.value]) > 20:
                        self.prediction_cache[insight.insight_type.value] = self.prediction_cache[insight.insight_type.value][-20:]

        except Exception as e:
            logger.error(f"Predictive insights generation failed: {e}")

    async def _analyze_seasonal_patterns(self):
        """Analyze seasonal patterns in agricultural data"""

        if not self.analytics_enabled[AnalyticsType.SEASONAL_PATTERNS]:
            return

        current_month = datetime.utcnow().month

        # Agricultural insights by season
        if 6 <= current_month <= 9:  # Monsoon season (kharif)
            monsoon_insights = await self._generate_monsoon_insights()
            for insight in monsoon_insights:
                insight_key = f"seasonal_monsoon_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
                await set_cached_value(insight_key, insight.__dict__, ttl_seconds=86400)  # 24 hours

        elif 10 <= current_month <= 2:  # Winter season (rabi)
            rabi_insights = await self._generate_rabi_insights()
            for insight in rabi_insights:
                insight_key = f"seasonal_rabi_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
                await set_cached_value(insight_key, insight.__dict__, ttl_seconds=86400)

    async def _calculate_growth_velocity(self):
        """Calculate and analyze growth velocity"""

        if not self.analytics_enabled[AnalyticsType.GROWTH_VELOCITY]:
            return

        # Look for NDVI data to calculate growth rates
        ndvi_buffers = [k for k in self.data_buffers.keys() if 'ndvi' in k.lower()]

        for ndvi_key in ndvi_buffers:
            buffer = self.data_buffers[ndvi_key]
            if len(buffer) >= 14:  # At least 2 weeks of data

                try:
                    # Calculate growth rate over last 7 days
                    recent_data = list(buffer)[-7:]  # Last 7 data points
                    values = [p['value'] for p in recent_data]

                    if len(values) >= 2:
                        growth_rate = (values[-1] - values[0]) / len(values)  # Daily growth rate

                        if abs(growth_rate) > self.alert_thresholds['growth_velocity_threshold']:

                            if growth_rate > 0:
                                velocity_type = "accelerating"
                                impact = "positive"
                            else:
                                velocity_type = "slowing"
                                impact = "concerning"

                            velocity_insight = PredictiveInsight(
                                insight_type=InsightCategory.GROWTH_PATTERNS,
                                title=f"Growth Velocity: {velocity_type.title()}",
                                description=f"NDVI growth rate: {growth_rate:.4f} per day - crop development is {velocity_type}",
                                confidence_score=0.85,
                                impact_level="medium",
                                recommended_action="Monitor closely and adjust irrigation/nutrient management" if growth_rate < 0 else "Growth is healthy, continue current management",
                                timeframe_days=3,
                                supporting_data={
                                    'growth_rate': growth_rate,
                                    'ndvi_values': values,
                                    'time_period_days': len(values)
                                }
                            )

                            # Cache growth velocity insight
                            velocity_key = f"growth_velocity_{ndvi_key}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
                            await set_cached_value(velocity_key, velocity_insight.__dict__, ttl_seconds=3600)

                except Exception as e:
                    logger.warning(f"Growth velocity calculation failed for {ndvi_key}: {e}")

    async def _perform_risk_assessment(self):
        """Perform comprehensive risk assessment"""

        if not self.analytics_enabled[AnalyticsType.RISK_ASSESSMENT]:
            return

        risk_factors = await self._assess_current_risk_factors()

        if risk_factors:
            # Calculate overall risk score
            risk_score = self._calculate_overall_risk_score(risk_factors)

            if risk_score > 0.7:  # High risk threshold
                risk_insight = PredictiveInsight(
                    insight_type=InsightCategory.RISK_WARNINGS,
                    title="High Agricultural Risk Detected",
                    description=f"Multiple risk factors present with overall risk score: {risk_score:.2f}",
                    confidence_score=risk_score,
                    impact_level="high",
                    recommended_action="Implement immediate protective measures and monitor closely",
                    timeframe_days=7,
                    affected_areas=["All monitored fields"],  # Would be more specific
                    supporting_data=risk_factors
                )

                risk_key = f"risk_assessment_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
                await set_cached_value(risk_key, risk_insight.__dict__, ttl_seconds=1800)

    # Helper methods for classifications and calculations

    def _classify_trend(self, slope: float, r_squared: float) -> str:
        """Classify trend direction"""
        if r_squared < self.alert_thresholds['trend_min_confidence']:
            return "stable"

        if abs(slope) < 0.01:  # Very small slope
            return "stable"

        return "increasing" if slope > 0 else "decreasing"

    def _detect_seasonality(self, values: List[float], timestamps: List[datetime]) -> bool:
        """Simple seasonality detection (simplified)"""
        if len(values) < 20:
            return False

        # Check for weekly pattern (simplified autocorrelation)
        try:
            # Calculate autocorrelation at lag 7 (weekly pattern)
            autocorr = np.correlate(values - np.mean(values),
                                  values - np.mean(values), mode='full')
            lag_7_idx = len(autocorr) // 2 + 7
            if lag_7_idx < len(autocorr):
                autocorr_7 = autocorr[lag_7_idx]
                return abs(autocorr_7) > 0.3  # Significant autocorrelation
        except:
            pass

        return False

    def _assess_trend_strength(self, r_squared: float) -> str:
        """Assess trend strength based on R²"""
        if r_squared >= 0.8:
            return "strong"
        elif r_squared >= 0.5:
            return "moderate"
        else:
            return "weak"

    def _calculate_anomaly_severity(self, z_score: float) -> AlertSeverity:
        """Calculate anomaly severity based on deviation"""
        if z_score > 5.0:
            return AlertSeverity.CRITICAL
        elif z_score > 4.0:
            return AlertSeverity.HIGH
        elif z_score > 3.0:
            return AlertSeverity.MEDIUM
        else:
            return AlertSeverity.LOW

    def _assess_correlation_strength(self, abs_correlation: float) -> str:
        """Assess correlation strength"""
        if abs_correlation >= 0.8:
            return "strong"
        elif abs_correlation >= 0.6:
            return "moderate"
        else:
            return "weak"

    # Agriculture-specific insight generation methods

    async def _check_weather_ndvi_correlation(self) -> Optional[PredictiveInsight]:
        """Check correlation between weather conditions and NDVI"""

        weather_temp_buffers = [k for k in self.data_buffers.keys() if 'temperature' in k.lower()]
        ndvi_buffers = [k for k in self.data_buffers.keys() if 'ndvi' in k.lower()]

        if not weather_temp_buffers or not ndvi_buffers:
            return None

        try:
            # Simplified correlation check
            temp_key = weather_temp_buffers[0]
            ndvi_key = ndvi_buffers[0]

            # This would normally compute correlation between temperature and NDVI
            # For now, return a sample insight if data exists

            if len(self.data_buffers[temp_key]) > 20 and len(self.data_buffers[ndvi_key]) > 20:
                return PredictiveInsight(
                    insight_type=InsightCategory.CLIMATE_IMPACTS,
                    title="Weather-NDVI Relationship Analysis",
                    description="Monitoring correlation between temperature patterns and vegetation health",
                    confidence_score=0.75,
                    impact_level="medium",
                    recommended_action="Continue monitoring weather impacts on crop health",
                    timeframe_days=7
                )

        except Exception as e:
            logger.warning(f"Weather-NDVI correlation check failed: {e}")

        return None

    async def _predict_growth_stage(self) -> List[PredictiveInsight]:
        """Predict crop growth stages based on NDVI trends"""
        insights = []

        ndvi_buffers = [k for k in self.data_buffers.keys() if 'ndvi' in k.lower()]

        for ndvi_key in ndvi_buffers:
            buffer = self.data_buffers[ndvi_key]
            if len(buffer) < 14:  # Need at least 2 weeks
                continue

            # Simplified growth stage prediction based on NDVI values
            current_ndvi = buffer[-1]['value']

            if 0.1 <= current_ndvi < 0.3:
                stage = "early_vegetative"
            elif 0.3 <= current_ndvi < 0.5:
                stage = "vegetative"
            elif 0.5 <= current_ndvi < 0.7:
                stage = "flowering"
            elif current_ndvi >= 0.7:
                stage = "maturation"
            else:
                continue

            insights.append(PredictiveInsight(
                insight_type=InsightCategory.GROWTH_PATTERNS,
                title=f"Crop Growth Stage: {stage.replace('_', ' ').title()}",
                description=f"Current NDVI indicates crop is in {stage.replace('_', ' ')} stage",
                confidence_score=0.8,
                impact_level="low",
                recommended_action=self._get_stage_recommendation(stage),
                timeframe_days=7,
                supporting_data={'current_ndvi': current_ndvi, 'predicted_stage': stage}
            ))

        return insights

    def _get_stage_recommendation(self, stage: str) -> str:
        """Get management recommendation based on growth stage"""
        recommendations = {
            "early_vegetative": "Ensure proper weed control and early pest management",
            "vegetative": "Monitor soil moisture and apply nutrients as needed",
            "flowering": "Avoid stress factors that could reduce pollination success",
            "maturation": "Monitor for disease in maturing crops and prepare for harvest"
        }
        return recommendations.get(stage, "Continue standard crop management practices")

    async def _analyze_irrigation_needs(self) -> List[PredictiveInsight]:
        """Analyze irrigation needs based on soil moisture data"""
        insights = []

        moisture_buffers = [k for k in self.data_buffers.keys() if 'moisture' in k.lower()]

        for moisture_key in moisture_buffers:
            buffer = self.data_buffers[moisture_key]
            if len(buffer) < 3:  # Need recent moisture data
                continue

            current_moisture = buffer[-1]['value']

            # Soil moisture recommendations (simplified thresholds)
            if current_moisture < 30:  # Very dry
                severity = "critical"
                action = "Immediate irrigation required"
                confidence = 0.95
            elif current_moisture < 50:  # Moderately dry
                severity = "high"
                action = "Irrigation recommended soon"
                confidence = 0.85
            elif current_moisture > 80:  # Too wet
                severity = "medium"
                action = "Reduce irrigation to prevent waterlogging"
                confidence = 0.8
            else:
                continue  # Optimal moisture level

            insights.append(PredictiveInsight(
                insight_type=InsightCategory.IRRIGATION_NEEDS,
                title=f"Soil Moisture Alert: {current_moisture:.1f}%",
                description=f"Current soil moisture level requires attention",
                confidence_score=confidence,
                impact_level=severity,
                recommended_action=action,
                timeframe_days=1 if severity == "critical" else 3,
                supporting_data={'soil_moisture_percent': current_moisture}
            ))

        return insights

    async def _assess_disease_risk(self) -> List[PredictiveInsight]:
        """Assess disease risk based on environmental conditions"""
        insights = []

        # Simplified disease risk assessment based on humidity and temperature
        humidity_buffers = [k for k in self.data_buffers.keys() if 'humidity' in k.lower()]
        temp_buffers = [k for k in self.data_buffers.keys() if 'temperature' in k.lower()]

        if humidity_buffers and temp_buffers:
            try:
                # Check recent conditions
                recent_humidity = self.data_buffers[humidity_buffers[0]][-3:]  # Last 3 readings
                recent_temp = self.data_buffers[temp_buffers[0]][-3:]

                avg_humidity = sum(p['value'] for p in recent_humidity) / len(recent_humidity)
                avg_temp = sum(p['value'] for p in recent_temp) / len(recent_temp)

                # Simplified disease risk model
                disease_risk = 0

                # High humidity + moderate temperature = fungal disease risk
                if avg_humidity > 75 and 20 <= avg_temp <= 30:
                    disease_risk = 0.8
                    disease_type = "fungal"
                elif avg_humidity > 80:  # Very high humidity
                    disease_risk = 0.9
                    disease_type = "bacterial"
                elif avg_temp > 35:  # Heat stress
                    disease_risk = 0.7
                    disease_type = "heat_stress"
                else:
                    disease_risk = 0.3

                if disease_risk > 0.6:
                    insights.append(PredictiveInsight(
                        insight_type=InsightCategory.DISEASE_THREATS,
                        title=f"Disease Risk Alert: {disease_type.title()}",
                        description=f"Increased risk of {disease_type} disease due to current environmental conditions",
                        confidence_score=disease_risk,
                        impact_level="high",
                        recommended_action=self._get_disease_prevention_action(disease_type),
                        timeframe_days=3,
                        supporting_data={
                            'humidity_percent': avg_humidity,
                            'temperature_celsius': avg_temp,
                            'disease_type': disease_type,
                            'risk_score': disease_risk
                        }
                    ))

            except Exception as e:
                logger.warning(f"Disease risk assessment failed: {e}")

        return insights

    def _get_disease_prevention_action(self, disease_type: str) -> str:
        """Get disease prevention recommendations"""
        actions = {
            "fungal": "Apply appropriate fungicide and improve field drainage",
            "bacterial": "Use disease-resistant varieties and practice crop rotation",
            "heat_stress": "Increase irrigation frequency and provide shade where possible"
        }
        return actions.get(disease_type, "Consult local agricultural extension services")

    async def _suggest_market_timing(self) -> List[PredictiveInsight]:
        """Suggest market timing based on yield predictions"""
        insights = []

        # Simplified market timing based on growth stage predictions
        # This would normally use actual yield predictions and market data

        current_month = datetime.utcnow().month

        # Example market insights for different seasons
        if current_month in [3, 4, 5]:  # Pre-monsoon (summer)
            insights.append(PredictiveInsight(
                insight_type=InsightCategory.YIELD_PREDICTIONS,
                title="Summer Crop Market Timing",
                description="Consider timing sales based on current market demand for summer crops",
                confidence_score=0.7,
                impact_level="medium",
                recommended_action="Monitor local market prices and consider storage options",
                timeframe_days=30
            ))

        return insights

    async def _generate_monsoon_insights(self) -> List[PredictiveInsight]:
        """Generate insights specific to monsoon season"""
        return [
            PredictiveInsight(
                insight_type=InsightCategory.CLIMATE_IMPACTS,
                title="Monsoon Season Water Management",
                description="Heavy rainfall expected - monitor for waterlogging and nutrient leaching",
                confidence_score=0.9,
                impact_level="high",
                recommended_action="Implement proper drainage and schedule nutrient applications post-rain",
                timeframe_days=7
            )
        ]

    async def _generate_rabi_insights(self) -> List[PredictiveInsight]:
        """Generate insights specific to winter season (Rabi)"""
        return [
            PredictiveInsight(
                insight_type=InsightCategory.SOIL_HEALTH,
                title="Winter Soil Preparation",
                description="Optimal time for soil testing and amendment applications",
                confidence_score=0.8,
                impact_level="medium",
                recommended_action="Schedule soil testing and plan winter fertilization",
                timeframe_days=14
            )
        ]

    async def _assess_current_risk_factors(self) -> Dict[str, Any]:
        """Assess various risk factors affecting agriculture"""
        risk_factors = {}

        try:
            # Drought risk
            moisture_buffers = [k for k in self.data_buffers.keys() if 'moisture' in k.lower()]
            if moisture_buffers:
                recent_moisture = [p['value'] for p in self.data_buffers[moisture_buffers[0]][-7:]]
                if recent_moisture and min(recent_moisture) < 25:
                    risk_factors['drought_risk'] = 0.9

            # Disease risk (simplified)
            if await self._assess_disease_risk():
                risk_factors['disease_risk'] = 0.7

            # Weather volatility
            temp_buffers = [k for k in self.data_buffers.keys() if 'temperature' in k.lower()]
            if temp_buffers:
                temperatures = [p['value'] for p in self.data_buffers[temp_buffers[0]][-24:]]  # Last 24 readings
                if len(temperatures) > 5:
                    temp_std = statistics.stdev(temperatures)
                    if temp_std > 8:  # High temperature variability
                        risk_factors['weather_volatility'] = 0.6

        except Exception as e:
            logger.warning(f"Risk assessment failed: {e}")

        return risk_factors

    def _calculate_overall_risk_score(self, risk_factors: Dict[str, Any]) -> float:
        """Calculate overall risk score from individual factors"""
        if not risk_factors:
            return 0.0

        # Weighted average of risk factors
        weights = {
            'drought_risk': 0.4,
            'disease_risk': 0.3,
            'weather_volatility': 0.3
        }

        total_weighted_risk = 0
        total_weight = 0

        for factor, risk_score in risk_factors.items():
            weight = weights.get(factor, 0.2)
            total_weighted_risk += risk_score * weight
            total_weight += weight

        return total_weighted_risk / total_weight if total_weight > 0 else 0

    async def _load_historical_baselines(self):
        """Load historical baselines for anomaly detection"""
        # This would load stored baseline statistics from database
        # Simplified for now
        pass

    async def _cleanup_old_data(self):
        """Clean up old data from buffers"""
        cutoff_time = datetime.utcnow() - timedelta(hours=self.analysis_window_hours)

        for metric_key, buffer in self.data_buffers.items():
            # Remove old data points
            while buffer and buffer[0]['timestamp'] < cutoff_time:
                buffer.popleft()

    async def _cache_analytics_results(self):
        """Cache analytics results for dashboard access"""
        analytics_summary = {
            'active_alerts_count': len(self.active_alerts),
            'insights_generated': sum(len(insights) for insights in self.prediction_cache.values()),
            'metrics_tracked': len(self.data_buffers),
            'analysis_timestamp': datetime.utcnow().isoformat(),
            'recent_insights': []
        }

        # Add recent insights
        for category, insights in self.prediction_cache.items():
            if insights:
                latest_insight = insights[-1]  # Most recent
                analytics_summary['recent_insights'].append({
                    'category': category,
                    'title': latest_insight.title,
                    'impact_level': latest_insight.impact_level,
                    'confidence_score': latest_insight.confidence_score
                })

        await set_cached_value('analytics_summary', analytics_summary, ttl_seconds=600)  # 10 minutes

    def get_analytics_dashboard_data(self) -> Dict[str, Any]:
        """Get data for analytics dashboard"""
        return {
            'metrics_count': len(self.data_buffers),
            'active_alerts': list(self.active_alerts.values())[-10:],  # Last 10 alerts
            'insights_by_category': {
                category: len(insights) for category, insights in self.prediction_cache.items()
            },
            'risk_assessment': self._get_current_risk_level(),
            'data_quality': self._get_data_quality_summary(),
            'timestamp': datetime.utcnow().isoformat()
        }

    def _get_current_risk_level(self) -> str:
        """Get current overall risk level"""
        if len(self.active_alerts) > 5:
            return "HIGH"
        elif len(self.active_alerts) > 2:
            return "MEDIUM"
        else:
            return "LOW"

    def _get_data_quality_summary(self) -> Dict[str, Any]:
        """Get data quality metrics summary"""
        total_points = sum(len(buffer) for buffer in self.data_buffers.values())
        recent_hours = 24
        cutoff_time = datetime.utcnow() - timedelta(hours=recent_hours)

        recent_points = 0
        for buffer in self.data_buffers.values():
            for point in buffer:
                if point['timestamp'] > cutoff_time:
                    recent_points += 1

        return {
            'total_data_points': total_points,
            'recent_data_points_24h': recent_points,
            'metrics_with_data': len([k for k, v in self.data_buffers.items() if v]),
            'average_data_velocity': recent_points / recent_hours if recent_hours > 0 else 0
        }

# Global analytics engine instance
realtime_analytics = AdvancedRealtimeAnalytics()

# Convenience functions
async def initialize_realtime_analytics() -> bool:
    """Initialize the real-time analytics engine"""
    return await realtime_analytics.initialize_analytics()

async def process_analytics_data(source: str, data: Any) -> bool:
    """Process data through analytics engine"""
    source_enum = DataSource(source.upper())
    await realtime_analytics.process_realtime_data(source_enum, data)
    return True

def get_analytics_dashboard() -> Dict[str, Any]:
    """Get analytics dashboard data"""
    return realtime_analytics.get_analytics_dashboard_data()
