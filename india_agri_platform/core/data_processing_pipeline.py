"""
Advanced Data Processing Pipeline for India Agricultural Intelligence Platform
Complete ETL automation, stream processing, and intelligent data management
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, AsyncIterator, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import pandas as pd
import numpy as np
from pathlib import Path
import aiofiles
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import hashlib

from india_agri_platform.core.error_handling import error_handler, log_system_event, ExternalAPIError, DataError
from india_agri_platform.core.cache_manager import cache_manager, set_cached_value
from india_agri_platform.core.task_scheduler import task_scheduler
from india_agri_platform.database.manager import db_manager

logger = logging.getLogger(__name__)

# Additional dependencies for geospatial processing
try:
    import geopandas as gpd
    from shapely.geometry import Point, Polygon
    GEOSPATIAL_AVAILABLE = True
except ImportError:
    GEOSPATIAL_AVAILABLE = False
    logger.warning("Geospatial libraries not available - spatial features will be limited")

class DataSource(Enum):
    """Supported data sources"""
    WEATHER_API = "WEATHER_API"
    GOOGLE_EARTH_ENGINE = "GOOGLE_EARTH_ENGINE"
    GOVERNMENT_DATABASE = "GOVERNMENT_DATABASE"
    SOIL_HEALTH_CARDS = "SOIL_HEALTH_CARDS"
    IoT_SENSORS = "IoT_SENSORS"
    SATELLITE_IMAGERY = "SATELLITE_IMAGERY"
    CROP_SURVEYS = "CROP_SURVEYS"
    MARKET_DATA = "MARKET_DATA"

class ProcessingStage(Enum):
    """Data processing pipeline stages"""
    INGESTION = "INGESTION"
    VALIDATION = "VALIDATION"
    CLEANSING = "CLEANSING"
    TRANSFORMATION = "TRANSFORMATION"
    FEATURE_ENGINEERING = "FEATURE_ENGINEERING"
    QUALITY_ASSURANCE = "QUALITY_ASSURANCE"
    STORAGE = "STORAGE"
    DISTRIBUTION = "DISTRIBUTION"

class DataQuality(Enum):
    """Data quality levels"""
    RAW = "RAW"
    PROCESSED = "PROCESSED"
    VALIDATED = "VALIDATED"
    FEATURE_ENRICHED = "FEATURE_ENRICHED"
    PRODUCTION_READY = "PRODUCTION_READY"

@dataclass
class DataBatch:
    """Batch of data being processed through pipeline"""
    batch_id: str
    source: DataSource
    data: Union[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]]]
    metadata: Dict[str, Any] = field(default_factory=dict)
    quality: DataQuality = DataQuality.RAW
    created_at: datetime = field(default_factory=datetime.utcnow)
    processed_at: Optional[datetime] = None
    errors: List[str] = field(default_factory=list)
    processing_time: Optional[float] = None
    row_count: int = 0
    size_bytes: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'batch_id': self.batch_id,
            'source': self.source.value,
            'metadata': self.metadata,
            'quality': self.quality.value,
            'created_at': self.created_at.isoformat(),
            'processed_at': self.processed_at.isoformat() if self.processed_at else None,
            'errors': self.errors,
            'processing_time': self.processing_time,
            'row_count': self.row_count,
            'size_bytes': self.size_bytes
        }

@dataclass
class ProcessingRule:
    """Data processing transformation rule"""
    rule_id: str
    stage: ProcessingStage
    name: str
    description: str
    transformation_function: Callable
    input_schema: Dict[str, Any] = None
    output_schema: Dict[str, Any] = None
    validation_rules: List[Dict[str, Any]] = field(default_factory=list)
    enabled: bool = True
    priority: int = 0

class AdvancedDataProcessingPipeline:
    """Complete ETL pipeline with streaming, validation, and intelligent processing"""

    def __init__(self, storage_path: Path = None):
        self.storage_path = storage_path or Path("data/processed")
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Processing components
        self.processing_rules: Dict[str, ProcessingRule] = {}
        self.active_batches: Dict[str, DataBatch] = {}
        self.data_schemas: Dict[str, Dict[str, Any]] = {}

        # Quality assurance
        self.quality_metrics: Dict[str, Any] = {}
        self.data_lineage: Dict[str, List[str]] = {}

        # Stream processing
        self.stream_processors: Dict[str, Callable] = {}
        self.stream_queues: Dict[str, asyncio.Queue] = {}

        # Thread pools for CPU-intensive tasks
        self.executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="data_processing")

        # Monitoring
        self.processing_stats: Dict[str, Any] = {}
        self.error_counts: Dict[str, int] = {}

        # Initialize standard processing rules
        self._initialize_standard_rules()

        logger.info(f"Advanced Data Processing Pipeline initialized with storage: {self.storage_path}")

    async def initialize_pipeline(self) -> bool:
        """Initialize the data processing pipeline"""
        try:
            # Create stream queues
            stream_sources = [DataSource.WEATHER_API, DataSource.IoT_SENSORS, DataSource.SATELLITE_IMAGERY]
            for source in stream_sources:
                self.stream_queues[source.value] = asyncio.Queue(maxsize=1000)

            # Start stream processors
            for source in stream_sources:
                processor_task = asyncio.create_task(
                    self._process_stream(source.value)
                )
                setattr(self, f"{source.value.lower()}_processor", processor_task)

            # Start monitoring tasks
            asyncio.create_task(self._pipeline_monitoring())
            asyncio.create_task(self._periodic_cleanup())

            log_system_event(
                "data_pipeline_initialized",
                "Advanced Data Processing Pipeline started",
                {
                    "processing_rules": len(self.processing_rules),
                    "stream_processors": len(stream_sources)
                }
            )

            return True

        except Exception as e:
            error_handler.handle_error(e, {"component": "data_pipeline", "operation": "initialization"})
            return False

    def _initialize_standard_rules(self):
        """Initialize standard processing rules for agricultural data"""

        # Weather data processing
        self.processing_rules["weather_normalization"] = ProcessingRule(
            rule_id="weather_normalization",
            stage=ProcessingStage.TRANSFORMATION,
            name="Weather Data Normalization",
            description="Convert weather data to standard units and format",
            transformation_function=self._normalize_weather_data,
            input_schema={
                "temperature": "numeric",
                "humidity": "numeric",
                "rainfall": "numeric",
                "wind_speed": "numeric"
            },
            output_schema={
                "temperature_celsius": "numeric",
                "humidity_percent": "numeric",
                "rainfall_mm": "numeric",
                "wind_speed_kmh": "numeric"
            }
        )

        # Soil data quality check
        self.processing_rules["soil_validation"] = ProcessingRule(
            rule_id="soil_validation",
            stage=ProcessingStage.VALIDATION,
            name="Soil Data Quality Validation",
            description="Validate soil parameters and flag anomalies",
            transformation_function=self._validate_soil_data,
            validation_rules=[
                {"field": "ph", "min": 0, "max": 14, "required": True},
                {"field": "nitrogen", "min": 0, "max": 1000, "required": False},
                {"field": "phosphorus", "min": 0, "max": 500, "required": False},
                {"field": "potassium", "min": 0, "max": 1000, "required": False}
            ]
        )

        # Satellite data feature engineering
        self.processing_rules["satellite_feature_engineering"] = ProcessingRule(
            rule_id="satellite_feature_engineering",
            stage=ProcessingStage.FEATURE_ENGINEERING,
            name="Satellite Data Feature Engineering",
            description="Extract vegetation indices and temporal features",
            transformation_function=self._engineer_satellite_features,
            input_schema={
                "ndvi": "numeric",
                "ndwi": "numeric",
                "temperature": "numeric",
                "date": "datetime"
            }
        )

        # Geospatial processing (if available)
        if GEOSPATIAL_AVAILABLE:
            self.processing_rules["spatial_interpolation"] = ProcessingRule(
                rule_id="spatial_interpolation",
                stage=ProcessingStage.FEATURE_ENGINEERING,
                name="Spatial Data Interpolation",
                description="Interpolate point data to spatial grids",
                transformation_function=self._interpolate_spatial_data
            )

        # Data quality scoring
        self.processing_rules["quality_score_calculation"] = ProcessingRule(
            rule_id="quality_score_calculation",
            stage=ProcessingStage.QUALITY_ASSURANCE,
            name="Data Quality Score Calculation",
            description="Calculate confidence scores for processed data",
            transformation_function=self._calculate_quality_score
        )

    async def ingest_data_batch(
        self,
        source: DataSource,
        data: Union[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]]],
        metadata: Dict[str, Any] = None,
        validate_schema: bool = True
    ) -> str:
        """Ingest a batch of data into the processing pipeline"""

        # Generate batch ID
        batch_id = f"batch_{source.value.lower()}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{hashlib.md5(str(data).encode()).hexdigest()[:8]}"

        # Create data batch
        batch = DataBatch(
            batch_id=batch_id,
            source=source,
            data=data,
            metadata=metadata or {},
            quality=DataQuality.RAW
        )

        # Calculate size and row count
        if isinstance(data, pd.DataFrame):
            batch.row_count = len(data)
            batch.size_bytes = data.memory_usage(deep=True).sum()
        elif isinstance(data, list):
            batch.row_count = len(data)
            batch.size_bytes = len(json.dumps(data).encode('utf-8'))
        else:
            batch.size_bytes = len(str(data).encode('utf-8'))

        # Add to active batches
        self.active_batches[batch_id] = batch

        # Validate schema if requested
        if validate_schema and source.value in self.data_schemas:
            if not await self._validate_batch_schema(batch):
                batch.errors.append("Schema validation failed")

        logger.info(f"Ingested data batch: {batch_id} ({batch.row_count} rows, {batch.size_bytes} bytes)")

        # Start async processing
        asyncio.create_task(self._process_batch_async(batch_id))

        return batch_id

    async def _process_batch_async(self, batch_id: str):
        """Process a data batch through the pipeline asynchronously"""
        start_time = datetime.utcnow()

        try:
            batch = self.active_batches[batch_id]
            logger.info(f"Starting pipeline processing for batch: {batch_id}")

            # Execute processing pipeline
            await self._execute_processing_pipeline(batch)

            # Update batch metadata
            batch.processed_at = datetime.utcnow()
            batch.processing_time = (batch.processed_at - start_time).total_seconds()

            # Log completion
            log_system_event(
                "batch_processing_completed",
                f"Data batch {batch_id} processed successfully",
                {
                    "batch_id": batch_id,
                    "source": batch.source.value,
                    "rows_processed": batch.row_count,
                    "processing_time": batch.processing_time,
                    "quality": batch.quality.value
                }
            )

        except Exception as e:
            error_handler.handle_error(e, {
                "batch_id": batch_id,
                "operation": "batch_processing",
                "source": self.active_batches.get(batch_id, {}).source.value if batch_id in self.active_batches else "unknown"
            })

            # Mark batch as failed
            if batch_id in self.active_batches:
                batch = self.active_batches[batch_id]
                batch.errors.append(str(e))
                batch.quality = DataQuality.RAW  # Reset quality on failure

        finally:
            # Clean up old batches (keep last 100 in memory)
            if len(self.active_batches) > 100:
                # Remove oldest completed batches
                completed_batches = [
                    bid for bid, b in self.active_batches.items()
                    if b.processed_at is not None
                ]
                if len(completed_batches) > 50:
                    oldest_batch = min(completed_batches,
                                     key=lambda x: self.active_batches[x].processed_at)
                    del self.active_batches[oldest_batch]

    async def _execute_processing_pipeline(self, batch: DataBatch):
        """Execute the complete processing pipeline for a batch"""

        # Stage 1: Validation
        await self._apply_processing_rules(batch, ProcessingStage.VALIDATION)

        # Stage 2: Cleansing
        await self._apply_processing_rules(batch, ProcessingStage.CLEANSING)

        # Stage 3: Transformation
        await self._apply_processing_rules(batch, ProcessingStage.TRANSFORMATION)

        # Stage 4: Feature Engineering
        await self._apply_processing_rules(batch, ProcessingStage.FEATURE_ENGINEERING)

        # Stage 5: Quality Assurance
        await self._apply_processing_rules(batch, ProcessingStage.QUALITY_ASSURANCE)

        # Stage 6: Storage & Distribution
        await self._store_processed_batch(batch)
        await self._distribute_processed_data(batch)

        # Update quality status
        batch.quality = DataQuality.PRODUCTION_READY

    async def _apply_processing_rules(self, batch: DataBatch, stage: ProcessingStage):
        """Apply all processing rules for a given stage"""

        applicable_rules = [
            rule for rule in self.processing_rules.values()
            if rule.stage == stage and rule.enabled
        ]

        # Sort by priority (higher priority first)
        applicable_rules.sort(key=lambda x: x.priority, reverse=True)

        for rule in applicable_rules:
            try:
                logger.debug(f"Applying rule '{rule.name}' to batch {batch.batch_id}")

                # Apply transformation in thread pool for CPU-intensive operations
                if asyncio.iscoroutinefunction(rule.transformation_function):
                    result = await rule.transformation_function(batch.data, batch.metadata)
                else:
                    result = await asyncio.get_event_loop().run_in_executor(
                        self.executor,
                        rule.transformation_function,
                        batch.data,
                        batch.metadata
                    )

                # Update batch data with result
                if result is not None:
                    batch.data = result

                    # Update lineage tracking
                    if batch.batch_id not in self.data_lineage:
                        self.data_lineage[batch.batch_id] = []
                    self.data_lineage[batch.batch_id].append(rule.rule_id)

            except Exception as e:
                error_msg = f"Processing rule '{rule.name}' failed: {str(e)}"
                batch.errors.append(error_msg)
                logger.error(error_msg)

                # Update error counts
                rule_key = f"{stage.value}_{rule.rule_id}"
                self.error_counts[rule_key] = self.error_counts.get(rule_key, 0) + 1

    async def stream_data_ingestion(
        self,
        source: DataSource,
        data_stream: AsyncIterator[Union[Dict[str, Any], List[Dict[str, Any]]]]
    ):
        """Process streaming data from real-time sources"""

        if source.value not in self.stream_queues:
            raise ValueError(f"Streaming not configured for source: {source.value}")

        queue = self.stream_queues[source.value]
        batch_data = []
        batch_size = 100  # Process in batches of 100

        async for data_item in data_stream:
            batch_data.append(data_item)

            if len(batch_data) >= batch_size:
                # Create batch and queue for processing
                batch_id = await self.ingest_data_batch(source, batch_data)
                batch_data = []

                # Avoid queue overflow
                if queue.full():
                    logger.warning(f"Stream queue full for {source.value}, dropping batch")

        # Process remaining data
        if batch_data:
            await self.ingest_data_batch(source, batch_data)

    async def _process_stream(self, source_name: str):
        """Background task to process streaming data"""
        queue = self.stream_queues[source_name]

        while True:
            try:
                # Wait for data with timeout
                try:
                    data_item = await asyncio.wait_for(queue.get(), timeout=60)
                except asyncio.TimeoutError:
                    continue

                # Process the streaming data
                if isinstance(data_item, dict) and 'batch_id' in data_item:
                    # This is a batch reference, process existing batch
                    batch_id = data_item['batch_id']
                    if batch_id in self.active_batches:
                        await self._process_batch_async(batch_id)

            except Exception as e:
                logger.error(f"Stream processing error for {source_name}: {e}")
                await asyncio.sleep(5)  # Avoid tight error loops

    async def _store_processed_batch(self, batch: DataBatch):
        """Store processed batch data with metadata"""

        # Create storage directory structure
        source_dir = self.storage_path / batch.source.value.lower()
        date_dir = source_dir / batch.created_at.strftime("%Y/%m/%d")
        date_dir.mkdir(parents=True, exist_ok=True)

        # Save data
        if isinstance(batch.data, pd.DataFrame):
            data_file = date_dir / f"{batch.batch_id}.parquet"
            batch.data.to_parquet(data_file)
        else:
            data_file = date_dir / f"{batch.batch_id}.json"
            async with aiofiles.open(data_file, 'w') as f:
                await f.write(json.dumps({
                    'data': batch.data,
                    'batch_info': batch.to_dict()
                }, indent=2, default=str))

        # Save metadata separately for quick access
        metadata_file = date_dir / f"{batch.batch_id}_metadata.json"
        async with aiofiles.open(metadata_file, 'w') as f:
            await f.write(json.dumps(batch.to_dict(), indent=2, default=str))

        # Cache processed data for fast access
        cache_key = f"processed_data:{batch.source.value}:{batch.batch_id}"
        cache_metadata = {
            "batch_id": batch.batch_id,
            "source": batch.source.value,
            "quality": batch.quality.value,
            "row_count": batch.row_count,
            "created_at": batch.created_at.isoformat(),
            "data_file": str(data_file)
        }

        await set_cached_value(cache_key, cache_metadata, ttl_seconds=86400)  # 24 hours

        logger.debug(f"Stored processed batch: {batch.batch_id}")

    async def _distribute_processed_data(self, batch: DataBatch):
        """Distribute processed data to various consumers"""

        # Notify registered consumers (other platform components)
        distribution_event = {
            "event_type": "processed_data_available",
            "batch_id": batch.batch_id,
            "source": batch.source.value,
            "quality": batch.quality.value,
            "row_count": batch.row_count,
            "timestamp": datetime.utcnow().isoformat()
        }

        # Cache the distribution event for real-time access
        event_key = f"data_event:{batch.batch_id}"
        await set_cached_value(event_key, distribution_event, ttl_seconds=3600)

        # Update processing statistics
        source_key = batch.source.value
        if source_key not in self.processing_stats:
            self.processing_stats[source_key] = {
                "batches_processed": 0,
                "total_rows": 0,
                "total_processing_time": 0,
                "errors_count": 0
            }

        stats = self.processing_stats[source_key]
        stats["batches_processed"] += 1
        stats["total_rows"] += batch.row_count
        if batch.processing_time:
            stats["total_processing_time"] += batch.processing_time
        stats["errors_count"] += len(batch.errors)

    # Data Processing Transformation Functions
    def _normalize_weather_data(self, data: pd.DataFrame, metadata: Dict[str, Any]) -> pd.DataFrame:
        """Normalize weather data to standard units"""
        df = data.copy()

        # Temperature conversions
        if 'temperature' in df.columns:
            if metadata.get('temperature_unit') == 'fahrenheit':
                df['temperature_celsius'] = (df['temperature'] - 32) * 5/9
            else:
                df['temperature_celsius'] = df['temperature']

        # Humidity (ensure percentage)
        if 'humidity' in df.columns:
            df['humidity_percent'] = df['humidity'].clip(0, 100)

        # Rainfall conversions
        if 'rainfall' in df.columns:
            if metadata.get('rainfall_unit') == 'inches':
                df['rainfall_mm'] = df['rainfall'] * 25.4
            else:
                df['rainfall_mm'] = df['rainfall']

        # Wind speed conversions
        if 'wind_speed' in df.columns:
            if metadata.get('wind_unit') == 'mph':
                df['wind_speed_kmh'] = df['wind_speed'] * 1.60934
            elif metadata.get('wind_unit') == 'm/s':
                df['wind_speed_kmh'] = df['wind_speed'] * 3.6
            else:
                df['wind_speed_kmh'] = df['wind_speed']

        return df

    def _validate_soil_data(self, data: pd.DataFrame, metadata: Dict[str, Any]) -> pd.DataFrame:
        """Validate soil data parameters"""
        df = data.copy()

        # Apply validation rules
        validation_results = []
        for _, row in df.iterrows():
            row_valid = True
            for rule in self.processing_rules["soil_validation"].validation_rules:
                field = rule["field"]
                if field in row.index:
                    value = row[field]
                    if pd.notna(value):
                        if "min" in rule and value < rule["min"]:
                            row_valid = False
                        if "max" in rule and value > rule["max"]:
                            row_valid = False
                    elif rule.get("required", False):
                        row_valid = False

            validation_results.append(row_valid)

        df['validation_passed'] = validation_results
        return df

    def _engineer_satellite_features(self, data: pd.DataFrame, metadata: Dict[str, Any]) -> pd.DataFrame:
        """Extract vegetation indices and temporal features from satellite data"""
        df = data.copy()

        if 'ndvi' in df.columns and 'ndwi' in df.columns:
            # Enhanced Vegetation Index
            df['evi'] = 2.5 * (df['ndvi'] - df['ndwi']) / (df['ndvi'] + 6 * df['ndwi'] - 7.5 + 1)

            # Leaf Area Index estimation
            df['lai'] = -np.log(1 - df['ndvi'].clip(0, 1)) / 0.5

            # Vegetation stress indicators
            df['vsi'] = df['ndvi'] / (df['temperature'] + 1).clip(1, 50)

        if 'date' in df.columns:
            # Extract temporal features
            df['date'] = pd.to_datetime(df['date'])
            df['day_of_year'] = df['date'].dt.dayofyear
            df['month'] = df['date'].dt.month
            df['season'] = pd.cut(df['date'].dt.month,
                                bins=[0, 3, 6, 9, 12],
                                labels=['winter', 'spring', 'summer', 'autumn'])

        return df

    def _interpolate_spatial_data(self, data: pd.DataFrame, metadata: Dict[str, Any]) -> pd.DataFrame:
        """Perform spatial interpolation of point data"""
        if not GEOSPATIAL_AVAILABLE:
            logger.warning("Geospatial libraries not available for spatial interpolation")
            return data

        # Convert to GeoDataFrame if coordinates are available
        if 'latitude' in data.columns and 'longitude' in data.columns:
            gdf = gpd.GeoDataFrame(
                data,
                geometry=gpd.points_from_xy(data.longitude, data.latitude),
                crs="EPSG:4326"
            )

            # Basic interpolation (simplified)
            # In production, this would use sophisticated spatial interpolation
            return data

        return data

    def _calculate_quality_score(self, data: pd.DataFrame, metadata: Dict[str, Any]) -> pd.DataFrame:
        """Calculate data quality confidence scores"""
        df = data.copy()

        # Calculate various quality metrics
        quality_factors = {}

        # Completeness score
        completeness = df.notna().mean().mean()
        quality_factors['completeness_score'] = completeness

        # Consistency score (based on reasonable value ranges)
        consistency_checks = []
        for col in df.select_dtypes(include=[np.number]).columns:
            values = df[col].dropna()
            if len(values) > 0:
                # Check for reasonable statistical distribution
                z_scores = np.abs((values - values.mean()) / values.std())
                outliers = (z_scores > 3).sum()
                consistency_checks.append(1 - outliers / len(values))

        consistency_score = np.mean(consistency_checks) if consistency_checks else 0.5
        quality_factors['consistency_score'] = consistency_score

        # Timeliness score (if temporal data)
        timeliness_score = 0.8  # Default
        if 'date' in df.columns:
            # Check how recent the data is
            max_date = pd.to_datetime(df['date']).max()
            days_old = (datetime.utcnow() - max_date.to_pydatetime()).days
            timeliness_score = max(0, 1 - days_old / 365)  # Within a year

        quality_factors['timeliness_score'] = timeliness_score

        # Overall quality score (weighted average)
        weights = {'completeness_score': 0.4, 'consistency_score': 0.4, 'timeliness_score': 0.2}
        overall_score = sum(weights[factor] * score for factor, score in quality_factors.items())

        df['quality_score'] = overall_score
        df['quality_factors'] = [quality_factors] * len(df)

        return df

    async def _validate_batch_schema(self, batch: DataBatch) -> bool:
        """Validate batch data against expected schema"""
        expected_schema = self.data_schemas.get(batch.source.value)
        if not expected_schema:
            return True  # No schema defined, consider valid

        try:
            if isinstance(batch.data, pd.DataFrame):
                # Check required columns exist
                required_columns = expected_schema.get('required_columns', [])
                missing_columns = set(required_columns) - set(batch.data.columns)
                if missing_columns:
                    batch.errors.append(f"Missing required columns: {missing_columns}")
                    return False

                # Validate data types
                for col, expected_type in expected_schema.get('column_types', {}).items():
                    if col in batch.data.columns:
                        actual_dtype = str(batch.data[col].dtype)
                        if expected_type.lower() not in actual_dtype.lower():
                            batch.errors.append(f"Column {col}: expected {expected_type}, got {actual_dtype}")

                return len(batch.errors) == 0

        except Exception as e:
            batch.errors.append(f"Schema validation error: {str(e)}")
            return False

    async def _pipeline_monitoring(self):
        """Monitor pipeline performance and health"""
        while True:
            try:
                await asyncio.sleep(300)  # Check every 5 minutes

                # Collect pipeline metrics
                metrics = {
                    "active_batches": len(self.active_batches),
                    "processing_stats": self.processing_stats,
                    "error_counts": self.error_counts,
                    "queue_sizes": {name: q.qsize() for name, q in self.stream_queues.items()},
                    "timestamp": datetime.utcnow().isoformat()
                }

                # Log performance summary
                log_system_event(
                    "pipeline_performance",
                    f"Pipeline processed {sum(s.get('batches_processed', 0) for s in self.processing_stats.values())} total batches",
                    metrics
                )

            except Exception as e:
                logger.error(f"Pipeline monitoring failed: {e}")

    async def _periodic_cleanup(self):
        """Periodically clean up old processed data files"""
        while True:
            try:
                await asyncio.sleep(3600)  # Clean up once per hour

                cutoff_date = datetime.utcnow() - timedelta(days=30)  # Keep 30 days

                # Clean up old files (simplified - in production would be more sophisticated)
                for source_dir in self.storage_path.iterdir():
                    if source_dir.is_dir():
                        for date_dir in source_dir.iterdir():
                            if date_dir.is_dir():
                                try:
                                    dir_date = datetime.strptime(date_dir.name, "%Y/%m/%d")
                                    if dir_date < cutoff_date:
                                        import shutil
                                        shutil.rmtree(date_dir)
                                        logger.info(f"Cleaned up old data directory: {date_dir}")
                                except ValueError:
                                    continue

            except Exception as e:
                logger.error(f"Periodic cleanup failed: {e}")

    # Public API methods
    def add_processing_rule(self, rule: ProcessingRule):
        """Add a new processing rule"""
        self.processing_rules[rule.rule_id] = rule
        logger.info(f"Added processing rule: {rule.name}")

    def remove_processing_rule(self, rule_id: str) -> bool:
        """Remove a processing rule"""
        if rule_id in self.processing_rules:
            del self.processing_rules[rule_id]
            logger.info(f"Removed processing rule: {rule_id}")
            return True
        return False

    def set_data_schema(self, source_name: str, schema: Dict[str, Any]):
        """Set expected schema for a data source"""
        self.data_schemas[source_name] = schema
        logger.info(f"Set schema for data source: {source_name}")

    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get comprehensive pipeline status"""
        return {
            "pipeline_active": True,
            "processing_rules": len(self.processing_rules),
            "active_batches": len(self.active_batches),
            "stream_queues": {name: q.qsize() for name, q in self.stream_queues.items()},
            "processing_stats": self.processing_stats,
            "error_counts": self.error_counts,
            "quality_metrics": self.quality_metrics,
            "last_updated": datetime.utcnow().isoformat()
        }

    def get_batch_status(self, batch_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific batch"""
        if batch_id not in self.active_batches:
            return None

        batch = self.active_batches[batch_id]
        return {
            "batch_id": batch.batch_id,
            "status": "PROCESSING" if batch.processed_at is None else "COMPLETED",
            "quality": batch.quality.value,
            "errors": batch.errors,
            "processing_time": batch.processing_time,
            "row_count": batch.row_count,
            "size_bytes": batch.size_bytes,
            "created_at": batch.created_at.isoformat(),
            "processed_at": batch.processed_at.isoformat() if batch.processed_at else None
        }

    async def get_processed_data(self, batch_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve processed data for a batch"""
        try:
            # Check cache first
            cache_key = f"processed_data:{batch_id.split('_')[1].upper()}:{batch_id}"
            cached_metadata = await cache_manager.cache.get(cache_key)

            if cached_metadata:
                data_file = Path(cached_metadata['data_file'])

                if data_file.exists():
                    async with aiofiles.open(data_file, 'r') as f:
                        content = await f.read()
                        return json.loads(content)

            return None

        except Exception as e:
            logger.error(f"Failed to retrieve processed data for batch {batch_id}: {e}")
            return None

# Global data processing pipeline instance
data_pipeline = AdvancedDataProcessingPipeline()

# Convenience functions for global use
async def initialize_data_pipeline() -> bool:
    """Initialize the data processing pipeline"""
    return await data_pipeline.initialize_pipeline()

async def ingest_batch(source: str, data: Any, metadata: Dict[str, Any] = None) -> str:
    """Ingest a batch of data for processing"""
    source_enum = DataSource(source.upper())
    return await data_pipeline.ingest_data_batch(source_enum, data, metadata)

async def stream_ingest(source: str, data_stream):
    """Ingest streaming data"""
    source_enum = DataSource(source.upper())
    await data_pipeline.stream_data_ingestion(source_enum, data_stream)

def get_pipeline_status() -> Dict[str, Any]:
    """Get data pipeline status"""
    return data_pipeline.get_pipeline_status()
