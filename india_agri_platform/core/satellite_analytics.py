"""
Advanced Satellite Data Analytics for Agricultural Intelligence Platform
Comprehensive satellite imagery processing with vegetation analysis and crop monitoring
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd
from pathlib import Path
import json
import cv2
from scipy import ndimage, stats
from scipy.ndimage import gaussian_filter, sobel
from skimage import morphology, measure
from sklearn.cluster import KMeans
import warnings

from india_agri_platform.core.error_handling import error_handler, log_system_event
from india_agri_platform.core.cache_manager import cache_manager, set_cached_value
from india_agri_platform.core.data_processing_pipeline import data_pipeline, DataSource
from india_agri_platform.core.advanced_weather_processor import weather_processor

# Suppress warnings for clean logging
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class VegetationIndex(Enum):
    """Types of vegetation indices"""
    NDVI = "NDVI"  # Normalized Difference Vegetation Index
    NDWI = "NDWI"  # Normalized Difference Water Index
    EVI = "EVI"   # Enhanced Vegetation Index
    SAVI = "SAVI"  # Soil Adjusted Vegetation Index
    ARVI = "ARVI"  # Atmospherically Resistant Vegetation Index
    GNDVI = "GNDVI"  # Green Normalized Difference Vegetation Index

class SatelliteBand(Enum):
    """Satellite spectral bands"""
    RED = "red"
    GREEN = "green"
    BLUE = "blue"
    NIR = "nir"  # Near Infrared
    SWIR1 = "swir1"  # Short Wave Infrared 1
    SWIR2 = "swir2"  # Short Wave Infrared 2
    THERMAL = "thermal"

class CropHealthStatus(Enum):
    """Crop health status classifications"""
    EXCELLENT = "EXCELLENT"
    GOOD = "GOOD"
    FAIR = "FAIR"
    POOR = "POOR"
    CRITICAL = "CRITICAL"
    BARE_SOIL = "BARE_SOIL"
    WATER = "WATER"

class ChangeDetectionType(Enum):
    """Types of satellite-based change detection"""
    VEGETATION_LOSS = "VEGETATION_LOSS"
    WATER_STRESS = "WATER_STRESS"
    DISEASE_PATTERNS = "DISEASE_PATTERNS"
    PEST_DAMAGE = "PEST_DAMAGE"
    FLOOD_DAMAGE = "FLOOD_DAMAGE"
    CROP_MATURITY = "CROP_MATURITY"
    HARVEST_PROGRESS = "HARVEST_PROGRESS"

@dataclass
class SatelliteImage:
    """Satellite image metadata and data"""
    image_id: str
    satellite: str  # 'landsat', 'sentinel', 'modis', etc.
    acquisition_date: datetime
    cloud_cover: float
    spatial_resolution: float  # meters per pixel
    bounding_box: Tuple[float, float, float, float]  # min_lon, min_lat, max_lon, max_lat
    bands: Dict[str, np.ndarray] = field(default_factory=dict)
    quality_score: float = 0.0
    processing_status: str = "raw"
    vegetation_indices: Dict[str, np.ndarray] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'image_id': self.image_id,
            'satellite': self.satellite,
            'acquisition_date': self.acquisition_date.isoformat(),
            'cloud_cover': self.cloud_cover,
            'spatial_resolution': self.spatial_resolution,
            'bounding_box': self.bounding_box,
            'quality_score': self.quality_score,
            'processing_status': self.processing_status,
            'bands_shape': {band: arr.shape for band, arr in self.bands.items()},
            'vegetation_indices_available': list(self.vegetation_indices.keys())
        }

@dataclass
class VegetationAnalysis:
    """Vegetation analysis results"""
    ndvi_mean: float
    ndvi_std: float
    ndvi_range: Tuple[float, float]
    healthy_pixels_percentage: float
    stressed_pixels_percentage: float
    bare_soil_percentage: float
    water_bodies_percentage: float
    vegetation_density_trend: str
    health_score_overall: float
    risk_areas: List[Dict[str, Any]]

@dataclass
class CropField:
    """Detected crop field boundary and properties"""
    field_id: str
    boundary: np.ndarray  # Polygon coordinates
    area_hectares: float
    centroid: Tuple[float, float]
    crop_type: Optional[str] = None
    health_status: CropHealthStatus = CropHealthStatus.FAIR
    ndvi_trend: str = "stable"
    irrigation_status: str = "unknown"
    last_updated: datetime = field(default_factory=datetime.utcnow)

@dataclass
class ChangeDetectionResult:
    """Satellite-based change detection analysis"""
    detection_type: ChangeDetectionType
    severity: str
    affected_area_percentage: float
    confidence_score: float
    affected_pixels: int
    temporal_window_days: int
    change_mask: np.ndarray
    descriptions: List[str]
    recommended_actions: List[str]

class AdvancedSatelliteAnalytics:
    """Advanced satellite imagery analytics with ML-enhanced vegetation analysis"""

    def __init__(self):
        # Satellite image storage and management
        self.image_library: Dict[str, SatelliteImage] = {}
        self.temporal_series: Dict[str, List[SatelliteImage]] = {}  # location-based time series

        # Analytical results caching
        self.vegetation_analyses: Dict[str, VegetationAnalysis] = {}
        self.crop_fields: Dict[str, List[CropField]] = {}
        self.change_detections: List[ChangeDetectionResult] = []

        # Processing configuration
        self.processing_config = {
            'ndvi_healthy_threshold': 0.6,
            'ndvi_stressed_threshold': 0.3,
            'cloud_detection_threshold': 0.8,
            'minimum_field_size_hectares': 0.5,
            'temporal_analysis_window_days': 30,
            'change_detection_min_confidence': 0.7
        }

        # NDVI calibration factors for different satellites
        self.ndvi_calibration = {
            'landsat8': {'red': 0.0000275, 'nir': 0.0000275, 'red_offset': -0.2, 'nir_offset': -0.2},
            'sentinel2': {'red': 0.00001, 'nir': 0.00001, 'red_offset': 0.0, 'nir_offset': 0.0},
            'modis': {'red': 0.00002008, 'nir': 0.00002008, 'red_offset': 0.0, 'nir_offset': 0.0}
        }

        logger.info("Advanced Satellite Analytics initialized")

    async def initialize_satellite_analytics(self) -> bool:
        """Initialize satellite analytics system"""

        try:
            # Start background processing tasks
            asyncio.create_task(self._background_temporal_analysis())
            asyncio.create_task(self._automated_change_detection())

            log_system_event(
                "satellite_analytics_initialized",
                "Advanced Satellite Analytics started",
                {"processing_config": len(self.processing_config)}
            )

            return True

        except Exception as e:
            error_handler.handle_error(e, {"component": "satellite_analytics", "operation": "initialization"})
            return False

    async def process_satellite_image(self, image_data: Dict[str, Any],
                                    bands_data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Process satellite imagery with comprehensive analytics"""

        image_id = image_data.get('image_id') or f"sat_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

        try:
            # Create satellite image object
            satellite_image = SatelliteImage(
                image_id=image_id,
                satellite=image_data.get('satellite', 'unknown'),
                acquisition_date=datetime.fromisoformat(image_data['acquisition_date']),
                cloud_cover=image_data.get('cloud_cover', 0.0),
                spatial_resolution=image_data.get('spatial_resolution', 10.0),
                bounding_box=tuple(image_data.get('bounding_box', [0, 0, 0, 0])),
                bands=bands_data.copy(),
                quality_score=self._assess_image_quality(bands_data, image_data.get('cloud_cover', 0.0))
            )

            # Store image
            self.image_library[image_id] = satellite_image

            # Process vegetation indices
            await self._calculate_vegetation_indices(satellite_image)

            # Generate vegetation analysis
            vegetation_analysis = await self._analyze_vegetation_health(satellite_image)

            # Extract crop fields
            crop_fields = await self._extract_crop_fields(satellite_image)

            # Store results
            location_key = self._get_location_key(satellite_image.bounding_box)
            if location_key not in self.temporal_series:
                self.temporal_series[location_key] = []
            self.temporal_series[location_key].append(satellite_image)

            # Sort temporal series by date
            self.temporal_series[location_key].sort(key=lambda x: x.acquisition_date)

            # Generate comprehensive analytics results
            analytics_result = {
                'image_id': image_id,
                'processing_timestamp': datetime.utcnow().isoformat(),
                'image_metadata': satellite_image.to_dict(),
                'vegetation_analysis': vegetation_analysis.__dict__ if vegetation_analysis else None,
                'crop_fields_detected': len(crop_fields),
                'vegetation_indices_calculated': list(satellite_image.vegetation_indices.keys()),
                'cloud_cover_assessment': self._assess_cloud_impact(satellite_image),
                'processing_quality_score': satellite_image.quality_score,
                'agricultural_insights': await self._generate_satellite_insights(satellite_image, vegetation_analysis, crop_fields)
            }

            satellite_image.processing_status = "processed"

            # Cache results
            cache_key = f"satellite_analysis_{image_id}"
            await set_cached_value(cache_key, analytics_result, ttl_seconds=7200)  # 2 hours

            # Store in data processing pipeline
            await self._store_satellite_results(analytics_result, location_key)

            logger.info(f"Processed satellite image {image_id} - {satellite_image.quality_score:.2f} quality score")

            return analytics_result

        except Exception as e:
            error_handler.handle_error(e, {
                "image_id": image_id,
                "operation": "satellite_image_processing",
                "satellite": image_data.get('satellite')
            })
            return {
                'image_id': image_id,
                'error': str(e),
                'processing_status': 'failed'
            }

    async def _calculate_vegetation_indices(self, satellite_image: SatelliteImage):
        """Calculate multiple vegetation indices from satellite bands"""

        bands = satellite_image.bands
        indices = {}

        try:
            # Extract band data
            red = bands.get('red', bands.get('B4', None))  # Sentinel-2/Landsat RED band
            nir = bands.get('nir', bands.get('B8', bands.get('B5', None)))  # NIR band
            blue = bands.get('blue', bands.get('B2', None))
            green = bands.get('green', bands.get('B3', None))
            swir1 = bands.get('swir1', bands.get('B11', bands.get('B6', None)))

            if red is None or nir is None:
                logger.warning(f"Insufficient bands for vegetation indices in image {satellite_image.image_id}")
                return

            # Calculate NDVI (Normalized Difference Vegetation Index)
            indices['ndvi'] = (nir - red) / (nir + red + 1e-10)  # Avoid division by zero

            # NDWI (Normalized Difference Water Index)
            if swir1 is not None:
                indices['ndwi'] = (green - swir1) / (green + swir1 + 1e-10)
            elif nir is not None:
                indices['ndwi'] = (green - nir) / (green + nir + 1e-10)

            # EVI (Enhanced Vegetation Index)
            if blue is not None and red is not None and nir is not None:
                indices['evi'] = 2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1 + 1e-10)

            # SAVI (Soil Adjusted Vegetation Index)
            L = 0.5  # Soil adjustment factor
            indices['savi'] = ((nir - red) / (nir + red + L)) * (1 + L)

            # GNDVI (Green Normalized Difference Vegetation Index)
            if green is not None and nir is not None:
                indices['gndvi'] = (nir - green) / (nir + green + 1e-10)

            # Apply cloud masking if available
            cloud_mask = self._detect_clouds(satellite_image)
            if cloud_mask is not None:
                for idx_name, idx_array in indices.items():
                    # Set cloudy pixels to NaN
                    indices[idx_name] = np.where(cloud_mask, np.nan, idx_array)

            satellite_image.vegetation_indices = indices

            # Apply calibration if available
            if satellite_image.satellite.lower() in self.ndvi_calibration:
                cal_params = self.ndvi_calibration[satellite_image.satellite.lower()]
                # Surface reflectance calibration (simplified)
                pass

        except Exception as e:
            logger.error(f"Failed to calculate vegetation indices for {satellite_image.image_id}: {e}")

    def _detect_clouds(self, satellite_image: SatelliteImage) -> Optional[np.ndarray]:
        """Detect cloudy pixels in satellite imagery"""

        try:
            bands = satellite_image.bands

            # Simple cloud detection for visible bands
            blue = bands.get('blue')
            green = bands.get('green')
            red = bands.get('red')
            nir = bands.get('nir')

            if blue is None or green is None or red is None:
                return None

            # Cloud detection algorithm (simplified version)
            # High reflectance in visible bands and low NDVI
            visible_reflectance = (blue + green + red) / 3

            # NIR reflectance threshold
            nir_cloud_threshold = np.percentile(nir, 90) if nir is not None else 0.3

            # Cloud probability
            cloud_probability = np.where(
                (visible_reflectance > 0.3) & (nir < nir_cloud_threshold if nir is not None else True),
                1, 0
            )

            return cloud_probability > self.processing_config['cloud_detection_threshold']

        except Exception as e:
            logger.warning(f"Cloud detection failed for {satellite_image.image_id}: {e}")
            return None

    async def _analyze_vegetation_health(self, satellite_image: SatelliteImage) -> Optional[VegetationAnalysis]:
        """Comprehensive vegetation health analysis"""

        try:
            ndvi = satellite_image.vegetation_indices.get('ndvi')
            if ndvi is None or np.all(np.isnan(ndvi)):
                return None

            # Remove NaN values for calculations
            valid_pixels = ~np.isnan(ndvi)
            ndvi_valid = ndvi[valid_pixels]

            if len(ndvi_valid) == 0:
                return None

            # Basic NDVI statistics
            ndvi_mean = float(np.mean(ndvi_valid))
            ndvi_std = float(np.std(ndvi_valid))
            ndvi_min, ndvi_max = float(np.min(ndvi_valid)), float(np.max(ndvi_valid))

            # Classify pixels by health status
            healthy_threshold = self.processing_config['ndvi_healthy_threshold']
            stressed_threshold = self.processing_config['ndvi_stressed_threshold']

            healthy_pixels = np.sum((ndvi_valid >= healthy_threshold) & valid_pixels[valid_pixels])
            stressed_pixels = np.sum((ndvi_valid >= stressed_threshold) & (ndvi_valid < healthy_threshold)
                                   & valid_pixels[valid_pixels])
            poor_pixels = np.sum((ndvi_valid < stressed_threshold) & valid_pixels[valid_pixels])
            total_valid_pixels = np.sum(valid_pixels)

            # Percentages
            healthy_pct = (healthy_pixels / total_valid_pixels) * 100 if total_valid_pixels > 0 else 0
            stressed_pct = (stressed_pixels / total_valid_pixels) * 100 if total_valid_pixels > 0 else 0
            poor_pct = (poor_pixels / total_valid_pixels) * 100 if total_valid_pixels > 0 else 0

            # Water bodies detection (using NDWI if available)
            water_pct = 0
            ndwi = satellite_image.vegetation_indices.get('ndwi')
            if ndwi is not None:
                water_pixels = np.sum((ndwi > 0.1) & valid_pixels[valid_pixels])
                water_pct = (water_pixels / total_valid_pixels) * 100 if total_valid_pixels > 0 else 0

            # Bare soil percentage (assuming everything else)
            bare_soil_pct = 100 - (healthy_pct + stressed_pct + poor_pct + water_pct)

            # Identify risk areas (stressed or poor vegetation)
            risk_areas = []
            if stressed_pct > 20 or poor_pct > 10:
                risk_areas.append({
                    'area_type': 'vegetation_stress',
                    'severity': 'high' if poor_pct > 20 else 'medium',
                    'affected_percentage': stressed_pct + poor_pct,
                    'coordinates': self._identify_risk_zones(ndvi, satellite_image.bounding_box)
                })

            # Trend analysis (compared to recent history)
            trend = await self._analyze_vegetation_trend(satellite_image, ndvi_mean)

            # Overall health score (0-100)
            health_score = self._calculate_vegetation_health_score(
                ndvi_mean, healthy_pct, stressed_pct, poor_pct
            )

            vegetation_analysis = VegetationAnalysis(
                ndvi_mean=ndvi_mean,
                ndvi_std=ndvi_std,
                ndvi_range=(ndvi_min, ndvi_max),
                healthy_pixels_percentage=healthy_pct,
                stressed_pixels_percentage=stressed_pct,
                bare_soil_percentage=bare_soil_pct,
                water_bodies_percentage=water_pct,
                vegetation_density_trend=trend,
                health_score_overall=health_score,
                risk_areas=risk_areas
            )

            # Cache analysis
            analysis_key = f"veg_analysis_{satellite_image.image_id}"
            self.vegetation_analyses[analysis_key] = vegetation_analysis

            return vegetation_analysis

        except Exception as e:
            logger.error(f"Vegetation health analysis failed for {satellite_image.image_id}: {e}")
            return None

    async def _extract_crop_fields(self, satellite_image: SatelliteImage) -> List[CropField]:
        """Extract individual crop field boundaries from satellite imagery"""

        try:
            crop_fields = []

            # Use NDVI for field boundary detection
            ndvi = satellite_image.vegetation_indices.get('ndvi')
            if ndvi is None:
                return crop_fields

            # Apply smoothing and segmentation
            smoothed_ndvi = gaussian_filter(ndvi, sigma=2)

            # Threshold to create vegetation mask
            vegetation_mask = smoothed_ndvi > self.processing_config['ndvi_stressed_threshold']

            # Morphological operations to clean up the mask
            cleaned_mask = morphology.remove_small_objects(vegetation_mask, min_size=50)
            cleaned_mask = morphology.remove_small_holes(cleaned_mask, area_threshold=20)

            # Label connected components (individual fields)
            labeled_fields, num_fields = measure.label(cleaned_mask, connectivity=2, return_num=True)

            # Extract field properties
            properties = measure.regionprops(labeled_fields)

            for i, prop in enumerate(properties):
                area_pixels = prop.area

                # Convert pixel area to hectares (approximate)
                pixel_area_m2 = area_pixels * (satellite_image.spatial_resolution ** 2)
                area_hectares = pixel_area_m2 / 10000

                # Skip very small fields
                if area_hectares < self.processing_config['minimum_field_size_hectares']:
                    continue

                # Calculate centroid
                centroid_row, centroid_col = prop.centroid
                bounding_box = satellite_image.bounding_box
                lon_range = bounding_box[2] - bounding_box[0]  # max_lon - min_lon
                lat_range = bounding_box[3] - bounding_box[1]  # max_lat - min_lat

                centroid_lon = bounding_box[0] + (centroid_col / ndvi.shape[1]) * lon_range
                centroid_lat = bounding_box[1] + ((ndvi.shape[0] - centroid_row) / ndvi.shape[0]) * lat_range

                # Get field boundary coordinates
                boundary_pixels = prop.coords
                boundary_coords = []
                for row, col in boundary_pixels[:100]:  # Limit to avoid too many points
                    boundary_lon = bounding_box[0] + (col / ndvi.shape[1]) * lon_range
                    boundary_lat = bounding_box[1] + ((ndvi.shape[0] - row) / ndvi.shape[0]) * lat_range
                    boundary_coords.append([boundary_lon, boundary_lat])

                # Calculate field health based on NDVI within field
                field_mask = labeled_fields == prop.label
                field_ndvi = ndvi[field_mask]
                field_ndvi_mean = np.mean(field_ndvi) if len(field_ndvi) > 0 else 0

                # Determine health status
                if field_ndvi_mean >= self.processing_config['ndvi_healthy_threshold']:
                    health_status = CropHealthStatus.EXCELLENT
                elif field_ndvi_mean >= (self.processing_config['ndvi_healthy_threshold'] +
                                       self.processing_config['ndvi_stressed_threshold']) / 2:
                    health_status = CropHealthStatus.GOOD
                elif field_ndvi_mean >= self.processing_config['ndvi_stressed_threshold']:
                    health_status = CropHealthStatus.FAIR
                else:
                    health_status = CropHealthStatus.POOR

                # Create crop field object
                crop_field = CropField(
                    field_id=f"field_{satellite_image.image_id}_{i}",
                    boundary=np.array(boundary_coords),
                    area_hectares=round(area_hectares, 2),
                    centroid=(centroid_lon, centroid_lat),
                    crop_type=None,  # Would be determined by ML classification
                    health_status=health_status
                )

                crop_fields.append(crop_field)

            # Store crop fields for the location
            location_key = self._get_location_key(satellite_image.bounding_box)
            self.crop_fields[location_key] = crop_fields

            logger.info(f"Extracted {len(crop_fields)} crop fields from {satellite_image.image_id}")

            return crop_fields

        except Exception as e:
            logger.error(f"Crop field extraction failed for {satellite_image.image_id}: {e}")
            return []

    async def analyze_temporal_changes(self, location: str, start_date: datetime,
                                     end_date: datetime) -> Optional[Dict[str, Any]]:
        """Analyze temporal changes in satellite data for a location"""

        try:
            location_key = self._get_location_key_from_string(location)
            if location_key not in self.temporal_series:
                return None

            temporal_images = [
                img for img in self.temporal_series[location_key]
                if start_date <= img.acquisition_date <= end_date
            ]

            if len(temporal_images) < 2:
                return {'error': 'Insufficient temporal images for analysis'}

            # Sort by date
            temporal_images.sort(key=lambda x: x.acquisition_date)

            # Extract NDVI time series
            dates = [img.acquisition_date for img in temporal_images]
            ndvi_values = []

            for img in temporal_images:
                ndvi = img.vegetation_indices.get('ndvi')
                if ndvi is not None and np.any(~np.isnan(ndvi)):
                    mean_ndvi = np.nanmean(ndvi)
                    ndvi_values.append(mean_ndvi)
                else:
                    ndvi_values.append(np.nan)

            # Perform change detection
            change_results = await self._detect_satellite_changes(
                temporal_images, location, start_date, end_date
            )

            # Calculate vegetation trends
            valid_ndvi = [(dates[i], ndvi_values[i]) for i in range(len(ndvi_values))
                         if not np.isnan(ndvi_values[i])]

            trend_analysis = {'trend': 'insufficient_data', 'slope': 0, 'confidence': 0}

            if len(valid_ndvi) >= 3:
                trend_dates, trend_values = zip(*valid_ndvi)
                # Convert dates to days since start
                start_date_trend = min(trend_dates)
                x_values = [(d - start_date_trend).days for d in trend_dates]

                slope, intercept, r_value, p_value, std_err = stats.linregress(x_values, trend_values)

                trend_analysis = {
                    'trend': 'increasing' if slope > 0.001 else 'decreasing' if slope < -0.001 else 'stable',
                    'slope': slope,
                    'r_squared': r_value ** 2,
                    'confidence': 1 - p_value
                }

            return {
                'location': location,
                'analysis_period': {
                    'start_date': start_date.isoformat(),
                    'end_date': end_date.isoformat(),
                    'images_analyzed': len(temporal_images)
                },
                'temporal_data': {
                    'dates': [d.isoformat() for d in dates],
                    'ndvi_values': ndvi_values,
                    'valid_points': len(valid_ndvi)
                },
                'trend_analysis': trend_analysis,
                'change_detection': change_results
            }

        except Exception as e:
            logger.error(f"Temporal change analysis failed for {location}: {e}")
            return None

    async def _detect_satellite_changes(self, temporal_images: List[SatelliteImage],
                                      location: str, start_date: datetime, end_date: datetime) -> List[Dict[str, Any]]:
        """Detect significant changes between satellite images"""

        changes = []

        try:
            if len(temporal_images) < 2:
                return changes

            # Compare consecutive images
            for i in range(1, len(temporal_images)):
                current_img = temporal_images[i]
                previous_img = temporal_images[i-1]

                # Calculate NDVI difference
                current_ndvi = current_img.vegetation_indices.get('ndvi')
                previous_ndvi = previous_img.vegetation_indices.get('ndvi')

                if current_ndvi is None or previous_ndvi is None:
                    continue

                # Ensure same shape (simplified - would need proper registration in production)
                if current_ndvi.shape != previous_ndvi.shape:
                    continue

                # Calculate NDVI difference
                ndvi_diff = current_ndvi - previous_ndvi

                # Detect significant changes
                significant_decrease = ndvi_diff < -0.1  # 10% NDVI decrease
                significant_increase = ndvi_diff > 0.1   # 10% NDVI increase

                # Calculate affected areas
                total_pixels = current_ndvi.size
                decreased_pixels = np.sum(significant_decrease)
                increased_pixels = np.sum(significant_increase)

                decrease_percentage = (decreased_pixels / total_pixels) * 100
                increase_percentage = (increased_pixels / total_pixels) * 100

                if decrease_percentage >= 5:  # At least 5% area affected
                    changes.append({
                        'change_type': 'vegetation_stress',
                        'severity': 'high' if decrease_percentage > 20 else 'medium',
                        'affected_area_percentage': decrease_percentage,
                        'date_of_change': current_img.acquisition_date.isoformat(),
                        'change_period_days': (current_img.acquisition_date - previous_img.acquisition_date).days,
                        'description': f"Vegetation health decreased by average {abs(np.mean(ndvi_diff[significant_decrease])):.3f} NDVI"
                    })

                elif increase_percentage >= 5:  # At least 5% area improvement
                    changes.append({
                        'change_type': 'vegetation_recovery',
                        'severity': 'low',
                        'affected_area_percentage': increase_percentage,
                        'date_of_change': current_img.acquisition_date.isoformat(),
                        'change_period_days': (current_img.acquisition_date - previous_img.acquisition_date).days,
                        'description': f"Vegetation health improved by average {abs(np.mean(ndvi_diff[significant_increase])):.3f} NDVI"
                    })

            return changes

        except Exception as e:
            logger.warning(f"Change detection failed: {e}")
            return []

    async def _generate_satellite_insights(self, satellite_image: SatelliteImage,
                                         vegetation_analysis: Optional[VegetationAnalysis],
                                         crop_fields: List[CropField]) -> List[Dict[str, Any]]:
        """Generate agricultural insights from satellite analysis"""

        insights = []

        if vegetation_analysis:
            # Overall vegetation health insight
            health_score = vegetation_analysis.health_score_overall

            if health_score < 40:
                insights.append({
                    'category': 'vegetation_health',
                    'severity': 'critical',
                    'title': 'Poor Vegetation Health Detected',
                    'description': f'Overall vegetation health score: {health_score:.1f}/100. Immediate attention required.',
                    'action_items': ['investigate_soil_moisture', 'check_for_pest_damage', 'review_irrigation_schedule']
                })

            elif health_score < 60:
                insights.append({
                    'category': 'vegetation_health',
                    'severity': 'high',
                    'title': 'Below Average Vegetation Health',
                    'description': f'Vegetation health score: {health_score:.1f}/100. Monitor closely.',
                    'action_items': ['increase_monitoring_frequency', 'optimize_irrigation']
                })

            # Water stress detection
            if vegetation_analysis.stressed_pixels_percentage > 30:
                insights.append({
                    'category': 'water_stress',
                    'severity': 'high',
                    'title': 'Water Stress Detected in Large Areas',
                    'description': f'{vegetation_analysis.stressed_pixels_percentage:.1f}% of vegetation shows stress indicators.',
                    'action_items': ['implement_immediate_irrigation', 'check_water_sources', 'monitor_soil_moisture_daily']
                })

        # Crop field insights
        if crop_fields:
            healthy_fields = sum(1 for field in crop_fields if field.health_status in [CropHealthStatus.EXCELLENT, CropHealthStatus.GOOD])
            stressed_fields = sum(1 for field in crop_fields if field.health_status in [CropHealthStatus.POOR, CropHealthStatus.CRITICAL])

            if stressed_fields > healthy_fields:
                insights.append({
                    'category': 'field_health',
                    'severity': 'high',
                    'title': 'Majority of Fields Show Health Concerns',
                    'description': f'{stressed_fields} out of {len(crop_fields)} detected fields require attention.',
                    'action_items': ['prioritize_stressed_fields', 'implement_targeted_interventions']
                })

        return insights

    def get_satellite_analytics_dashboard(self) -> Dict[str, Any]:
        """Get satellite analytics dashboard data"""

        return {
            'images_processed': len(self.image_library),
            'active_locations': len(self.temporal_series),
            'total_crop_fields': sum(len(fields) for fields in self.crop_fields.values()),
            'vegetation_analyses': len(self.vegetation_analyses),
            'change_detections': len(self.change_detections),
            'system_status': 'operational',
            'last_updated': datetime.utcnow().isoformat(),
            'recent_insights': self._get_recent_satellite_insights()
        }

    def _get_recent_satellite_insights(self) -> List[Dict[str, Any]]:
        """Get recent satellite-derived insights"""

        insights = []

        # Collect insights from recent vegetation analyses
        for analysis_key, analysis in list(self.vegetation_analyses.items())[-5:]:  # Last 5 analyses
            if analysis.risk_areas:
                for risk_area in analysis.risk_areas:
                    insights.append({
                        'type': 'vegetation_risk',
                        'location': analysis_key,
                        'severity': risk_area.get('severity', 'unknown'),
                        'area_affected': risk_area.get('affected_percentage', 0),
                        'timestamp': datetime.utcnow().isoformat()
                    })

        return insights[-5:]  # Last 5 insights

    async def _analyze_vegetation_trend(self, satellite_image: SatelliteImage, current_ndvi: float) -> str:
        """Analyze vegetation trend compared to historical data"""

        try:
            location_key = self._get_location_key(satellite_image.bounding_box)
            if location_key not in self.temporal_series:
                return "insufficient_data"

            # Get recent historical data (last 30 days)
            historical_images = [
                img for img in self.temporal_series[location_key]
                if (satellite_image.acquisition_date - img.acquisition_date).days <= 30
                and img.acquisition_date < satellite_image.acquisition_date
            ]

            if len(historical_images) < 2:
                return "insufficient_data"

            # Calculate historical average NDVI
            historical_ndvi = []
            for img in historical_images:
                ndvi = img.vegetation_indices.get('ndvi')
                if ndvi is not None:
                    historical_ndvi.append(np.nanmean(ndvi))

            if not historical_ndvi:
                return "insufficient_data"

            historical_avg = np.mean(historical_ndvi)
            ndvi_change = current_ndvi - historical_avg

            if abs(ndvi_change) < 0.05:  # Less than 5% change
                return "stable"
            elif ndvi_change > 0.05:
                return "improving"
            else:
                return "declining"

        except Exception as e:
            logger.warning(f"Vegetation trend analysis failed: {e}")
            return "unknown"

    def _calculate_vegetation_health_score(self, ndvi_mean: float, healthy_pct: float,
                                         stressed_pct: float, poor_pct: float) -> float:
        """Calculate overall vegetation health score (0-100)"""

        # Weight factors
        ndvi_weight = 0.3
        healthy_weight = 0.4
        stress_weight = 0.3

        # NDVI score (0-100 based on NDVI range)
        ndvi_score = min(100, max(0, (ndvi_mean + 1) * 50))  # Scale -1 to 1 to 0-100

        # Healthy vegetation score
        healthy_score = healthy_pct  # Direct percentage

        # Stress score (inverse of poor + stressed vegetation)
        stress_score = 100 - (stressed_pct + poor_pct)

        # Weighted average
        overall_score = (
            ndvi_score * ndvi_weight +
            healthy_score * healthy_weight +
            stress_score * stress_weight
        )

        return round(overall_score, 2)

    def _assess_image_quality(self, bands: Dict[str, np.ndarray], cloud_cover: float) -> float:
        """Assess satellite image quality (0-1 scale)"""

        quality_score = 1.0

        # Penalize for cloud cover
        quality_score *= (1 - cloud_cover / 100)

        # Check band completeness
        essential_bands = ['red', 'nir']
        available_essential = sum(1 for band in essential_bands if band in bands)
        quality_score *= (available_essential / len(essential_bands))

        # Check data quality (remove extreme outliers)
        total_pixels = 0
        valid_pixels = 0

        for band_data in bands.values():
            total_pixels += band_data.size
            # Consider pixel valid if within reasonable reflectance range (0-1 for surface reflectance)
            valid_pixels += np.sum((band_data >= 0) & (band_data <= 1))

        if total_pixels > 0:
            valid_percentage = valid_pixels / total_pixels
            quality_score *= valid_percentage

        return round(quality_score, 3)

    def _assess_cloud_impact(self, satellite_image: SatelliteImage) -> str:
        """Assess the impact of cloud cover on analysis"""

        cloud_cover = satellite_image.cloud_cover

        if cloud_cover < 10:
            return "minimal_impact"
        elif cloud_cover < 30:
            return "moderate_impact"
        elif cloud_cover < 70:
            return "significant_impact"
        else:
            return "severe_impact_analysis_unreliable"

    def _identify_risk_zones(self, ndvi_array: np.ndarray, bounding_box: Tuple[float, float, float, float]) -> List[Dict[str, Any]]:
        """Identify geographical zones with vegetation risk"""

        risk_zones = []

        try:
            # Find areas with low NDVI
            low_ndvi_mask = ndvi_array < self.processing_config['ndvi_stressed_threshold']

            # Group nearby risk areas
            if np.any(low_ndvi_mask):
                # Simplified - find centroids of risk areas
                low_ndvi_coords = np.where(low_ndvi_mask)

                if len(low_ndvi_coords[0]) > 0:
                    # Calculate approximate center of risk area
                    center_row = int(np.mean(low_ndvi_coords[0]))
                    center_col = int(np.mean(low_ndvi_coords[1]))

                    # Convert to geographical coordinates
                    lon_range = bounding_box[2] - bounding_box[0]
                    lat_range = bounding_box[3] - bounding_box[1]

                    risk_lon = bounding_box[0] + (center_col / ndvi_array.shape[1]) * lon_range
                    risk_lat = bounding_box[1] + ((ndvi_array.shape[0] - center_row) / ndvi_array.shape[0]) * lat_range

                    risk_zones.append({
                        'coordinates': [risk_lon, risk_lat],
                        'affected_pixels': np.sum(low_ndvi_mask),
                        'severity': 'high' if np.sum(low_ndvi_mask) > ndvi_array.size * 0.1 else 'medium'
                    })

        except Exception as e:
            logger.warning(f"Risk zone identification failed: {e}")

        return risk_zones

    def _get_location_key(self, bounding_box: Tuple[float, float, float, float]) -> str:
        """Generate a location key from bounding box for spatial indexing"""

        # Simplified - use center coordinates rounded to 0.01 degrees
        center_lon = (bounding_box[0] + bounding_box[2]) / 2
        center_lat = (bounding_box[1] + bounding_box[3]) / 2

        return f"{round(center_lon, 2)}_{round(center_lat, 2)}"

    def _get_location_key_from_string(self, location_string: str) -> str:
        """Convert location string to location key"""

        # This would need to be implemented based on how locations are represented
        # For now, return a simplified key
        return location_string.replace(' ', '_').replace(',', '_').lower()

    async def _store_satellite_results(self, results: Dict[str, Any], location_key: str) -> None:
        """Store satellite analysis results"""

        # Store in data processing pipeline
        batch_data = {
            'satellite_analysis': results,
            'location': location_key,
            'timestamp': results['processing_timestamp']
        }

        await data_pipeline.ingest_data_batch(
            source=DataSource.SATELLITE_IMAGERY,
            data=batch_data,
            metadata={'satellite_image_id': results['image_id']}
        )

    async def _background_temporal_analysis(self):
        """Background task for continuous temporal analysis"""

        while True:
            try:
                await asyncio.sleep(86400)  # Run daily

                # Analyze recent changes for all monitored locations
                for location_key, image_series in self.temporal_series.items():
                    if len(image_series) >= 3:
                        # Analyze changes over last 30 days
                        end_date = datetime.utcnow()
                        start_date = end_date - timedelta(days=30)

                        analysis = await self.analyze_temporal_changes(location_key, start_date, end_date)

                        if analysis and analysis.get('change_detection'):
                            # Cache temporal analysis
                            cache_key = f"temporal_analysis_{location_key}"
                            await set_cached_value(cache_key, analysis, ttl_seconds=86400)  # 1 day

                            # Log significant changes
                            significant_changes = [
                                change for change in analysis['change_detection']
                                if change.get('severity') == 'high'
                            ]

                            for change in significant_changes:
                                logger.warning(f"TEMPORAL_CHANGE: {location_key} - {change['description']}")

                logger.info("Completed background temporal analysis")

            except Exception as e:
                logger.error(f"Background temporal analysis failed: {e}")

    async def _automated_change_detection(self):
        """Automated change detection processing"""

        while True:
            try:
                await asyncio.sleep(43200)  # Run every 12 hours

                # Check for new change detections in all series
                for location_key, image_series in self.temporal_series.items():
                    if len(image_series) >= 2:
                        recent_images = image_series[-5:]  # Check last 5 images

                        for i in range(1, len(recent_images)):
                            current_img = recent_images[i]
                            previous_img = recent_images[i-1]

                            # Check for rapid changes in vegetation
                            changes = await self._detect_satellite_changes(
                                [previous_img, current_img], location_key,
                                previous_img.acquisition_date, current_img.acquisition_date
                            )

                            # Store any critical changes
                            critical_changes = [
                                change for change in changes
                                if change.get('severity') == 'high'
                            ]

                            for change in critical_changes:
                                # Cache critical change alerts
                                alert_key = f"critical_change_{location_key}_{current_img.image_id}"
                                await set_cached_value(alert_key, change, ttl_seconds=172800)  # 2 days

                logger.info("Completed automated change detection")

            except Exception as e:
                logger.error(f"Automated change detection failed: {e}")

    def get_satellite_analytics_dashboard(self) -> Dict[str, Any]:
        """Get satellite analytics dashboard data"""

        return {
            'images_processed': len(self.image_library),
            'active_locations': len(self.temporal_series),
            'total_crop_fields': sum(len(fields) for fields in self.crop_fields.values()),
            'vegetation_analyses': len(self.vegetation_analyses),
            'change_detections': len(self.change_detections),
            'system_status': 'operational',
            'last_updated': datetime.utcnow().isoformat(),
            'recent_insights': self._get_recent_satellite_insights()
        }

    def _get_recent_satellite_insights(self) -> List[Dict[str, Any]]:
        """Get recent satellite-derived insights"""

        insights = []

        # Collect insights from recent vegetation analyses
        for analysis_key, analysis in list(self.vegetation_analyses.items())[-5:]:  # Last 5 analyses
            if analysis.risk_areas:
                for risk_area in analysis.risk_areas:
                    insights.append({
                        'type': 'vegetation_risk',
                        'location': analysis_key,
                        'severity': risk_area.get('severity', 'unknown'),
                        'area_affected': risk_area.get('affected_percentage', 0),
                        'timestamp': datetime.utcnow().isoformat()
                    })

        return insights[-5:]  # Last 5 insights

    def get_satellite_image_status(self, image_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed status of a satellite image"""

        if image_id not in self.image_library:
            return None

        image = self.image_library[image_id]
        analysis = self.vegetation_analyses.get(f"veg_analysis_{image_id}")

        return {
            'image_metadata': image.to_dict(),
            'vegetation_analysis': analysis.__dict__ if analysis else None,
            'processing_status': image.processing_status,
            'quality_score': image.quality_score,
            'available_indices': list(image.vegetation_indices.keys())
        }

# Global satellite analytics instance
satellite_analytics = AdvancedSatelliteAnalytics()

# Convenience functions
async def initialize_satellite_analytics() -> bool:
    """Initialize satellite analytics system"""
    return await satellite_analytics.initialize_satellite_analytics()

async def process_satellite_image(image_data: Dict[str, Any], bands_data: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """Process satellite imagery"""
    return await satellite_analytics.process_satellite_image(image_data, bands_data)

async def analyze_satellite_temporal_changes(location: str, start_date: datetime, end_date: datetime) -> Optional[Dict[str, Any]]:
    """Analyze temporal changes in satellite data"""
    return await satellite_analytics.analyze_temporal_changes(location, start_date, end_date)

def get_satellite_analytics_dashboard() -> Dict[str, Any]:
    """Get satellite analytics dashboard data"""
    return satellite_analytics.get_satellite_analytics_dashboard()
