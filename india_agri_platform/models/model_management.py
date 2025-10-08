"""
Advanced Model Management System for Production AI Operations
Complete MLOps lifecycle with versioning, A/B testing, and automated deployment
"""

import asyncio
import hashlib
import json
import pickle
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, asdict, field
from enum import Enum
import logging
from concurrent.futures import ThreadPoolExecutor
import threading

from india_agri_platform.core.error_handling import error_handler, ModelError, log_system_event
from india_agri_platform.database.manager import db_manager
from india_agri_platform.core.cache_manager import cache_manager

logger = logging.getLogger(__name__)

class ModelStatus(Enum):
    """Model lifecycle status"""
    TRAINING = "TRAINING"
    VALIDATING = "VALIDATING"
    READY = "READY"
    DEPLOYED = "DEPLOYED"
    DEPRECATED = "DEPRECATED"
    FAILED = "FAILED"
    ARCHIVED = "ARCHIVED"

class ModelType(Enum):
    """Types of ML models supported"""
    LINEAR_REGRESSION = "LINEAR_REGRESSION"
    RANDOM_FOREST = "RANDOM_FOREST"
    GRADIENT_BOOSTING = "GRADIENT_BOOSTING"
    NEURAL_NETWORK = "NEURAL_NETWORK"
    LSTM_NETWORK = "LSTM_NETWORK"
    CNN_NETWORK = "CNN_NETWORK"
    ENSEMBLE = "ENSEMBLE"
    CUSTOM = "CUSTOM"

class ExperimentType(Enum):
    """Types of experiments"""
    HYPERPARAMETER_TUNING = "HYPERPARAMETER_TUNING"
    FEATURE_ENGINEERING = "FEATURE_ENGINEERING"
    ARCHITECTURE_SEARCH = "ARCHITECTURE_SEARCH"
    CROSS_VALIDATION = "CROSS_VALIDATION"
    A_B_TESTING = "A_B_TESTING"
    PERFORMANCE_BENCHMARKING = "PERFORMANCE_BENCHMARKING"

@dataclass
class ModelVersion:
    """Model version metadata"""
    model_id: str
    version: str
    model_type: ModelType
    status: ModelStatus
    created_at: datetime
    updated_at: datetime

    # Performance metrics
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    mse: Optional[float] = None
    rmse: Optional[float] = None
    mae: Optional[float] = None
    r2_score: Optional[float] = None

    # Model specifications
    input_features: List[str] = None
    output_features: List[str] = None
    hyperparams: Dict[str, Any] = None

    # Metadata
    training_time_seconds: Optional[float] = None
    dataset_size: Optional[int] = None
    framework: str = "scikit-learn"
    framework_version: Optional[str] = None

    # Validation and deployment
    validation_score: Optional[float] = None
    validation_method: Optional[str] = None
    deployed_at: Optional[datetime] = None

    # Performance monitoring
    total_predictions: int = 0
    avg_response_time: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        # Handle datetime serialization
        data['created_at'] = self.created_at.isoformat()
        data['updated_at'] = self.updated_at.isoformat()
        if self.deployed_at:
            data['deployed_at'] = self.deployed_at.isoformat()
        return data

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance metrics summary"""
        return {
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score,
            'mse': self.mse,
            'rmse': self.rmse,
            'mae': self.mae,
            'r2_score': self.r2_score,
            'validation_score': self.validation_score,
            'training_time_seconds': self.training_time_seconds
        }

@dataclass
class Experiment:
    """ML experiment tracking"""
    experiment_id: str
    name: str
    description: str = ""
    experiment_type: ExperimentType = ExperimentType.CROSS_VALIDATION
    model_ids: List[str] = None
    tags: List[str] = None

    # Experiment configuration
    config: Dict[str, Any] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None

    # Results
    results: Dict[str, Any] = None
    best_model_id: Optional[str] = None
    status: str = "RUNNING"

@dataclass
class A_B_Test:
    """A/B testing configuration and results"""
    test_id: str
    name: str
    description: str = ""
    model_a_id: str = ""
    model_b_id: Optional[str] = None

    # Test configuration
    percentage_a: float = 50.0
    percentage_b: float = 50.0
    start_time: datetime = field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None
    test_duration_days: int = 7

    # Results tracked
    samples_a: int = 0
    samples_b: int = 0
    performance_metrics_a: Dict[str, float] = field(default_factory=dict)
    performance_metrics_b: Dict[str, float] = field(default_factory=dict)

    # Statistical significance
    p_value: Optional[float] = None
    confidence_level: Optional[float] = None
    winner: Optional[str] = None  # 'A', 'B', or 'TIE'

    status: str = "RUNNING"

class AdvancedModelManager:
    """Production-ready ML model management system"""

    def __init__(self, model_storage_path: Path = None):
        self.model_storage_path = model_storage_path or Path("models/advanced_models")
        self.model_storage_path.mkdir(parents=True, exist_ok=True)

        self.current_models: Dict[str, ModelVersion] = {}
        self.experiments: Dict[str, Experiment] = {}
        self.ab_tests: Dict[str, A_B_Test] = {}

        # Model serving state
        self.serving_models: Dict[str, Any] = {}  # Loaded model objects
        self.model_weights: Dict[str, float] = {}  # For ensemble serving

        # Performance monitoring
        self.prediction_stats: Dict[str, Dict[str, Any]] = {}
        self.response_times: Dict[str, List[float]] = {}

        # Background tasks
        self.cleanup_task = None
        self.monitoring_task = None

        # Thread pool for model I/O
        self.executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="model_io")

        logger.info(f"Advanced Model Manager initialized with storage: {self.model_storage_path}")

    async def initialize(self) -> bool:
        """Initialize the model management system"""
        try:
            # Load existing model metadata
            await self._load_model_registry()

            # Load existing experiments
            await self._load_experiments()

            # Start background tasks
            self.cleanup_task = asyncio.create_task(self._periodic_cleanup())
            self.monitoring_task = asyncio.create_task(self._model_monitoring())

            log_system_event(
                "model_manager_initialized",
                "Advanced Model Manager started successfully",
                {"loaded_models": len(self.current_models)}
            )

            return True

        except Exception as e:
            error_handler.handle_error(e, {"component": "model_manager", "operation": "initialization"})
            return False

    async def register_model(
        self,
        model: Any,
        model_type: ModelType,
        input_features: List[str],
        output_features: List[str],
        hyperparams: Dict[str, Any] = None,
        performance_metrics: Dict[str, Any] = None,
        framework: str = "scikit-learn"
    ) -> str:
        """Register a new model version"""

        model_id = f"mdl_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{hashlib.md5(str(model).encode()).hexdigest()[:8]}"
        version = "v1.0.0"  # First version

        model_version = ModelVersion(
            model_id=model_id,
            version=version,
            model_type=model_type,
            status=ModelStatus.READY,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            input_features=input_features,
            output_features=output_features,
            hyperparams=hyperparams or {},
            framework=framework
        )

        # Set performance metrics if provided
        if performance_metrics:
            for key, value in performance_metrics.items():
                if hasattr(model_version, key):
                    setattr(model_version, key, value)

        try:
            # Save model to disk
            model_path = await self._save_model_to_disk(model, model_version)

            # Save metadata
            await self._save_model_metadata(model_version)

            # Add to in-memory registry
            self.current_models[model_id] = model_version

            logger.info(f"Registered new model: {model_id} ({model_type.value})")
            return model_id

        except Exception as e:
            error_handler.handle_error(e, {
                "model_id": model_id,
                "operation": "model_registration",
                "model_type": model_type.value
            })
            raise

    async def create_new_version(
        self,
        base_model_id: str,
        new_model: Any,
        version_changes: str = "Updated hyperparameters",
        new_performance: Dict[str, Any] = None
    ) -> str:
        """Create a new version of an existing model"""

        if base_model_id not in self.current_models:
            raise ValueError(f"Base model {base_model_id} not found")

        base_model = self.current_models[base_model_id]

        # Generate new version number
        current_version_parts = base_model.version.replace('v', '').split('.')
        new_patch = int(current_version_parts[2]) + 1
        new_version = f"v{current_version_parts[0]}.{current_version_parts[1]}.{new_patch}"

        model_version = ModelVersion(
            model_id=base_model_id,
            version=new_version,
            model_type=base_model.model_type,
            status=ModelStatus.READY,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            input_features=base_model.input_features.copy(),
            output_features=base_model.output_features.copy(),
            hyperparams=base_model.hyperparams.copy(),
            framework=base_model.framework
        )

        # Update performance metrics
        if new_performance:
            for key, value in new_performance.items():
                if hasattr(model_version, key):
                    setattr(model_version, key, value)

        try:
            # Save model and metadata
            await self._save_model_to_disk(new_model, model_version)
            await self._save_model_metadata(model_version)

            # Update registry
            self.current_models[f"{base_model_id}_{new_version}"] = model_version

            logger.info(f"Created new version: {base_model_id} {new_version}")

            # Check if this new version should be deployed
            await self._evaluate_version_promotion(base_model_id, model_version)

            return f"{base_model_id}_{new_version}"

        except Exception as e:
            error_handler.handle_error(e, {
                "base_model_id": base_model_id,
                "new_version": new_version,
                "operation": "version_creation"
            })
            raise

    async def deploy_model(self, model_id: str, weights: Optional[Dict[str, float]] = None) -> bool:
        """Deploy a model for serving"""

        if model_id not in self.current_models:
            raise ValueError(f"Model {model_id} not found")

        model_version = self.current_models[model_id]

        if model_version.status != ModelStatus.READY:
            raise ValueError(f"Model {model_id} is not ready for deployment (status: {model_version.status.value})")

        try:
            # Load model into memory
            model = await self._load_model_from_disk(model_version)

            # Update deployment info
            model_version.deployed_at = datetime.utcnow()
            model_version.status = ModelStatus.DEPLOYED
            model_version.updated_at = datetime.utcnow()

            # Store loaded model
            self.serving_models[model_id] = model

            # Set weights for ensemble serving
            if weights:
                self.model_weights[model_id] = weights
            else:
                self.model_weights[model_id] = 1.0

            # Save updated metadata
            await self._save_model_metadata(model_version)

            # Clear any old deployments
            await self._cleanup_old_deployments(model_version.model_id)

            log_system_event(
                "model_deployed",
                f"Model {model_id} deployed successfully",
                {"model_id": model_id, "version": model_version.version}
            )

            logger.info(f"Deployed model: {model_id} ({model_version.version})")
            return True

        except Exception as e:
            model_version.status = ModelStatus.FAILED
            error_handler.handle_error(e, {
                "model_id": model_id,
                "operation": "model_deployment",
                "model_version": model_version.version
            })
            return False

    async def undeploy_model(self, model_id: str) -> bool:
        """Remove a model from serving"""

        if model_id not in self.serving_models:
            logger.warning(f"Model {model_id} is not currently deployed")
            return True

        try:
            # Remove from serving
            del self.serving_models[model_id]
            if model_id in self.model_weights:
                del self.model_weights[model_id]

            # Update model status
            if model_id in self.current_models:
                model_version = self.current_models[model_id]
                model_version.status = ModelStatus.READY
                model_version.updated_at = datetime.utcnow()
                await self._save_model_metadata(model_version)

            logger.info(f"Undeployed model: {model_id}")
            return True

        except Exception as e:
            error_handler.handle_error(e, {
                "model_id": model_id,
                "operation": "model_undeployment"
            })
            return False

    async def predict(self, model_id: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Make prediction with deployed model"""

        start_time = datetime.utcnow()

        if model_id not in self.serving_models:
            raise ModelError(
                message=f"Model {model_id} is not deployed",
                model_name=model_id
            )

        model = self.serving_models[model_id]
        model_version = self.current_models[model_id]

        try:
            # Validate input features
            missing_features = set(model_version.input_features) - set(input_data.keys())
            if missing_features:
                raise ModelError(
                    message=f"Missing required features: {missing_features}",
                    model_name=model_id
                )

            # Make prediction
            prediction = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                self._make_prediction,
                model,
                input_data,
                model_version
            )

            # Calculate response time
            response_time = (datetime.utcnow() - start_time).total_seconds()

            # Update model statistics
            await self._update_model_statistics(model_id, response_time)

            result = {
                "model_id": model_id,
                "version": model_version.version,
                "prediction": prediction,
                "response_time_seconds": response_time,
                "timestamp": datetime.utcnow().isoformat()
            }

            return result

        except Exception as e:
            response_time = (datetime.utcnow() - start_time).total_seconds()
            error_handler.handle_error(ModelError(
                message=f"Prediction failed for model {model_id}: {str(e)}",
                model_name=model_id
            ), {
                "response_time": response_time,
                "input_features_count": len(input_data)
            })

            raise

    def _make_prediction(self, model: Any, input_data: Dict[str, Any], model_version: ModelVersion) -> Any:
        """Execute model prediction (in thread pool)"""

        # Prepare input according to model's expected features
        model_input = []
        for feature in model_version.input_features:
            if feature in input_data:
                model_input.append(input_data[feature])
            else:
                model_input.append(0)  # Default value for missing features

        if len(model_input) == 1:
            model_input = model_input[0]

        # Handle different input formats
        if hasattr(model, 'predict'):
            if len(model_version.input_features) == 1:
                model_input = [[model_input]]
            else:
                model_input = [model_input]

            prediction = model.predict(model_input)

            # Return single value if single output
            if len(model_version.output_features) == 1 and len(prediction) == 1:
                return prediction[0].item() if hasattr(prediction[0], 'item') else prediction[0]

            return prediction.tolist() if hasattr(prediction, 'tolist') else prediction

        else:
            raise ModelError(f"Model {model_version.model_id} does not have predict method")

    async def start_experiment(
        self,
        name: str,
        description: str = "",
        experiment_type: ExperimentType = ExperimentType.CROSS_VALIDATION,
        config: Dict[str, Any] = None
    ) -> str:
        """Start a new ML experiment"""

        experiment_id = f"exp_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{hashlib.md5(name.encode()).hexdigest()[:8]}"

        experiment = Experiment(
            experiment_id=experiment_id,
            name=name,
            description=description,
            experiment_type=experiment_type,
            config=config or {}
        )

        self.experiments[experiment_id] = experiment

        await self._save_experiment_metadata(experiment)

        log_system_event(
            "experiment_started",
            f"ML experiment started: {name}",
            {"experiment_id": experiment_id, "type": experiment_type.value}
        )

        logger.info(f"Started experiment: {experiment_id} ({name})")
        return experiment_id

    async def complete_experiment(self, experiment_id: str, results: Dict[str, Any],
                                best_model_id: Optional[str] = None) -> bool:
        """Complete an experiment with results"""

        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")

        experiment = self.experiments[experiment_id]

        experiment.completed_at = datetime.utcnow()
        experiment.results = results
        experiment.best_model_id = best_model_id
        experiment.status = "COMPLETED"

        await self._save_experiment_metadata(experiment)

        log_system_event(
            "experiment_completed",
            f"Experiment {experiment.name} completed",
            {
                "experiment_id": experiment_id,
                "best_model_id": best_model_id,
                "status": "COMPLETED"
            }
        )

        logger.info(f"Completed experiment: {experiment_id}")
        return True

    async def start_ab_test(
        self,
        name: str,
        model_a_id: str,
        model_b_id: str,
        percentage_a: float = 50.0,
        test_duration_days: int = 7,
        description: str = ""
    ) -> str:
        """Start an A/B test between two models"""

        if model_a_id not in self.current_models or model_b_id not in self.current_models:
            raise ValueError("Both models must be registered before A/B testing")

        if not (0 <= percentage_a <= 100):
            raise ValueError("Percentage A must be between 0 and 100")

        test_id = f"ab_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{hashlib.md5(name.encode()).hexdigest()[:8]}"

        percentage_b = 100.0 - percentage_a

        ab_test = A_B_Test(
            test_id=test_id,
            name=name,
            description=description,
            model_a_id=model_a_id,
            model_b_id=model_b_id,
            percentage_a=percentage_a,
            percentage_b=percentage_b,
            test_duration_days=test_duration_days
        )

        # Calculate end time
        ab_test.end_time = ab_test.start_time + timedelta(days=test_duration_days)

        self.ab_tests[test_id] = ab_test

        await self._save_ab_test_metadata(ab_test)

        log_system_event(
            "ab_test_started",
            f"A/B test started: {name}",
            {
                "test_id": test_id,
                "model_a": model_a_id,
                "model_b": model_b_id,
                "percentage_a": percentage_a,
                "duration_days": test_duration_days
            }
        )

        logger.info(f"Started A/B test: {test_id} ({name})")
        return test_id

    async def predict_with_ab_test(self, test_id: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Make prediction using A/B test routing"""

        if test_id not in self.ab_tests:
            raise ValueError(f"A/B test {test_id} not found")

        ab_test = self.ab_tests[test_id]

        if ab_test.status != "RUNNING" or datetime.utcnow() > ab_test.end_time:
            raise ValueError(f"A/B test {test_id} is not running")

        # Determine which model to use based on traffic allocation
        import random
        rand_percent = random.uniform(0, 100)

        if rand_percent < ab_test.percentage_a:
            model_id = ab_test.model_a_id
            samples_field = 'samples_a'
            metrics_field = 'performance_metrics_a'
        else:
            model_id = ab_test.model_b_id
            samples_field = 'samples_b'
            metrics_field = 'performance_metrics_b'

        # Make prediction
        result = await self.predict(model_id, input_data)

        # Track A/B test metrics
        setattr(ab_test, samples_field, getattr(ab_test, samples_field) + 1)

        # Add A/B test metadata
        result.update({
            "ab_test_id": test_id,
            "model_used": "A" if model_id == ab_test.model_a_id else "B",
            "test_details": {
                "percentage_a": ab_test.percentage_a,
                "percentage_b": ab_test.percentage_b,
                "samples_a": ab_test.samples_a,
                "samples_b": ab_test.samples_b
            }
        })

        return result

    async def complete_ab_test(self, test_id: str) -> bool:
        """Complete an A/B test and determine winner"""

        if test_id not in self.ab_tests:
            raise ValueError(f"A/B test {test_id} not found")

        ab_test = self.ab_tests[test_id]

        if ab_test.status != "RUNNING":
            raise ValueError(f"A/B test {test_id} is not running")

        # Perform statistical analysis
        from scipy import stats  # Assuming scipy is available

        try:
            if ab_test.samples_a > 0 and ab_test.samples_b > 0:
                # Simple t-test on performance (assuming we have performance metrics)
                # This is simplified - in practice you'd use actual performance data

                # For now, use prediction counts as a proxy
                conversions_a = ab_test.samples_a  # Placeholder
                conversions_b = ab_test.samples_b  # Placeholder

                # Chi-square test for proportions
                contingency_table = [[conversions_a, ab_test.samples_a - conversions_a],
                                   [conversions_b, ab_test.samples_b - conversions_b]]

                chi2, p_value, _, _ = stats.chi2_contingency(contingency_table)

                ab_test.p_value = p_value
                ab_test.confidence_level = (1 - p_value) * 100

                # Determine winner (simplified logic)
                if p_value < 0.05:  # 95% confidence
                    if conversions_a / max(ab_test.samples_a, 1) > conversions_b / max(ab_test.samples_b, 1):
                        ab_test.winner = "A"
                    else:
                        ab_test.winner = "B"
                else:
                    ab_test.winner = "TIE"

            else:
                ab_test.winner = "INSUFFICIENT_DATA"

        except ImportError:
            # Fall back to simple comparison if scipy not available
            if ab_test.samples_a > ab_test.samples_b:
                ab_test.winner = "A"
            elif ab_test.samples_b > ab_test.samples_a:
                ab_test.winner = "B"
            else:
                ab_test.winner = "TIE"

        ab_test.status = "COMPLETED"
        await self._save_ab_test_metadata(ab_test)

        log_system_event(
            "ab_test_completed",
            f"A/B test {ab_test.name} completed with winner: {ab_test.winner}",
            {
                "test_id": test_id,
                "winner": ab_test.winner,
                "p_value": ab_test.p_value,
                "confidence_level": ab_test.confidence_level
            }
        )

        logger.info(f"Completed A/B test: {test_id}, winner: {ab_test.winner}")
        return True

    # Private utility methods
    async def _save_model_to_disk(self, model: Any, model_version: ModelVersion) -> Path:
        """Save model object to disk"""
        model_dir = self.model_storage_path / model_version.model_id
        model_dir.mkdir(exist_ok=True)

        model_file = model_dir / f"{model_version.version}.pkl"

        def _save():
            with open(model_file, 'wb') as f:
                pickle.dump(model, f)

        await asyncio.get_event_loop().run_in_executor(self.executor, _save)
        return model_file

    async def _load_model_from_disk(self, model_version: ModelVersion) -> Any:
        """Load model object from disk"""
        model_file = self.model_storage_path / model_version.model_id / f"{model_version.version}.pkl"

        def _load():
            with open(model_file, 'rb') as f:
                return pickle.load(f)

        return await asyncio.get_event_loop().run_in_executor(self.executor, _load)

    async def _save_model_metadata(self, model_version: ModelVersion):
        """Save model metadata to database/storage"""
        metadata_file = self.model_storage_path / model_version.model_id / "metadata.json"

        def _save_metadata():
            with open(metadata_file, 'w') as f:
                json.dump(model_version.to_dict(), f, indent=2)

        await asyncio.get_event_loop().run_in_executor(self.executor, _save_metadata)

        # Also store in database if available
        try:
            await db_manager.save_model_metadata(model_version.to_dict())
        except Exception:
            pass  # Database not critical for basic operation

    async def _load_model_registry(self):
        """Load existing model registry from storage"""
        if not self.model_storage_path.exists():
            return

        for model_dir in self.model_storage_path.iterdir():
            if model_dir.is_dir():
                metadata_file = model_dir / "metadata.json"
                if metadata_file.exists():
                    try:
                        with open(metadata_file, 'r') as f:
                            data = json.load(f)

                        # Reconstruct ModelVersion object
                        model_version = ModelVersion(**data)
                        self.current_models[data['model_id']] = model_version

                    except Exception as e:
                        logger.warning(f"Failed to load model metadata: {model_dir} - {e}")

        logger.info(f"Loaded {len(self.current_models)} existing models")

    async def _update_model_statistics(self, model_id: str, response_time: float):
        """Update model performance statistics"""
        if model_id not in self.prediction_stats:
            self.prediction_stats[model_id] = {
                'total_predictions': 0,
                'avg_response_time': 0,
                'response_times': []
            }

        stats = self.prediction_stats[model_id]
        model_version = self.current_models[model_id]

        stats['total_predictions'] += 1
        stats['response_times'].append(response_time)

        # Keep only last 1000 response times for memory efficiency
        if len(stats['response_times']) > 1000:
            stats['response_times'] = stats['response_times'][-1000:]

        # Update average response time
        stats['avg_response_time'] = sum(stats['response_times']) / len(stats['response_times'])

        # Update model version
        model_version.total_predictions += 1
        model_version.avg_response_time = stats['avg_response_time']

        # Save updated metadata periodically
        if stats['total_predictions'] % 100 == 0:  # Every 100 predictions
            await self._save_model_metadata(model_version)

    async def _evaluate_version_promotion(self, model_id: str, new_version: ModelVersion):
        """Evaluate if a new version should be automatically promoted"""

        # Get current deployed version performance
        current_deployed = None
        for mid, mv in self.current_models.items():
            if mv.model_id == model_id and mv.status == ModelStatus.DEPLOYED:
                current_deployed = mv
                break

        if not current_deployed:
            # No current deployment, consider deploying this version
            if (new_version.accuracy or 0) > 0.8:  # Minimum accuracy threshold
                logger.info(f"Auto-deploying first version of {model_id}")
                await self.deploy_model(f"{model_id}_{new_version.version}")
            return

        # Compare performance (simplified comparison)
        current_accuracy = current_deployed.accuracy or 0
        new_accuracy = new_version.accuracy or 0

        if new_accuracy > current_accuracy + 0.02:  # 2% improvement threshold
            logger.info(f"Auto-promoting better performing version: {model_id} {new_version.version}")
            await self.deploy_model(f"{model_id}_{new_version.version}")
            await self.undeploy_model(current_deployed.model_id)

    async def _cleanup_old_deployments(self, model_id: str):
        """Clean up old deployments of the same model"""
        deployed_versions = [
            mv for mv in self.current_models.values()
            if mv.model_id == model_id and mv.status == ModelStatus.DEPLOYED
        ]

        # Keep only the most recent deployment
        if len(deployed_versions) > 1:
            deployed_versions.sort(key=lambda x: x.deployed_at or datetime.min, reverse=True)

            for old_version in deployed_versions[1:]:
                await self.undeploy_model(old_version.model_id)
                logger.info(f"Cleaned up old deployment: {old_version.model_id}")

    async def _periodic_cleanup(self):
        """Periodically clean up old model files"""
        while True:
            try:
                await asyncio.sleep(86400)  # Once per day

                # Clean up old archived models (older than 90 days)
                cutoff_date = datetime.utcnow() - timedelta(days=90)

                cleaned_count = 0
                for model_id, model_version in list(self.current_models.items()):
                    if (model_version.status == ModelStatus.ARCHIVED and
                        model_version.updated_at < cutoff_date):

                        # Remove files
                        model_dir = self.model_storage_path / model_version.model_id
                        if model_dir.exists():
                            shutil.rmtree(model_dir)

                        # Remove from registry
                        del self.current_models[model_id]
                        cleaned_count += 1

                if cleaned_count > 0:
                    logger.info(f"Cleaned up {cleaned_count} old archived models")

            except Exception as e:
                logger.error(f"Periodic cleanup failed: {e}")

    async def _model_monitoring(self):
        """Monitor model performance and trigger retraining if needed"""
        while True:
            try:
                await asyncio.sleep(3600)  # Check hourly

                for model_id, model_version in self.current_models.items():
                    if model_version.status == ModelStatus.DEPLOYED:

                        # Check if performance has degraded
                        stats = self.prediction_stats.get(model_id, {})

                        # Trigger retraining if:
                        # 1. Average response time too high (> 5 seconds)
                        # 2. Model older than 30 days (placeholder for drift detection)

                        avg_response_time = stats.get('avg_response_time', 0)
                        model_age_days = (datetime.utcnow() - model_version.created_at).days

                        if avg_response_time > 5.0:
                            logger.warning(f"Model {model_id} response time too high: {avg_response_time}s")
                            # Trigger retraining alert

                        if model_age_days > 30:
                            logger.info(f"Model {model_id} is {model_age_days} days old, considering retraining")
                            # Trigger age-based retraining

            except Exception as e:
                logger.error(f"Model monitoring failed: {e}")

    # Additional utility methods for experiments and A/B tests
    async def _save_experiment_metadata(self, experiment: Experiment):
        """Save experiment metadata"""
        exp_file = self.model_storage_path / "experiments" / f"{experiment.experiment_id}.json"
        exp_file.parent.mkdir(exist_ok=True)

        with open(exp_file, 'w') as f:
            json.dump({
                'experiment_id': experiment.experiment_id,
                'name': experiment.name,
                'description': experiment.description,
                'experiment_type': experiment.experiment_type.value,
                'model_ids': experiment.model_ids or [],
                'tags': experiment.tags or [],
                'config': experiment.config,
                'created_at': experiment.created_at.isoformat(),
                'completed_at': experiment.completed_at.isoformat() if experiment.completed_at else None,
                'results': experiment.results,
                'best_model_id': experiment.best_model_id,
                'status': experiment.status
            }, f, indent=2)

    async def _save_ab_test_metadata(self, ab_test: A_B_Test):
        """Save A/B test metadata"""
        test_file = self.model_storage_path / "ab_tests" / f"{ab_test.test_id}.json"
        test_file.parent.mkdir(exist_ok=True)

        with open(test_file, 'w') as f:
            json.dump({
                'test_id': ab_test.test_id,
                'name': ab_test.name,
                'description': ab_test.description,
                'model_a_id': ab_test.model_a_id,
                'model_b_id': ab_test.model_b_id,
                'percentage_a': ab_test.percentage_a,
                'percentage_b': ab_test.percentage_b,
                'start_time': ab_test.start_time.isoformat(),
                'end_time': ab_test.end_time.isoformat() if ab_test.end_time else None,
                'test_duration_days': ab_test.test_duration_days,
                'samples_a': ab_test.samples_a,
                'samples_b': ab_test.samples_b,
                'performance_metrics_a': ab_test.performance_metrics_a,
                'performance_metrics_b': ab_test.performance_metrics_b,
                'p_value': ab_test.p_value,
                'confidence_level': ab_test.confidence_level,
                'winner': ab_test.winner,
                'status': ab_test.status
            }, f, indent=2)

    async def _load_experiments(self):
        """Load existing experiments"""
        exp_dir = self.model_storage_path / "experiments"
        if not exp_dir.exists():
            return

        for exp_file in exp_dir.glob("*.json"):
            try:
                with open(exp_file, 'r') as f:
                    data = json.load(f)

                experiment = Experiment(**data)
                self.experiments[data['experiment_id']] = experiment

            except Exception as e:
                logger.warning(f"Failed to load experiment: {exp_file} - {e}")

    def get_model_status_summary(self) -> Dict[str, Any]:
        """Get comprehensive model status summary"""
        return {
            'total_models': len(self.current_models),
            'deployed_models': len([m for m in self.current_models.values() if m.status == ModelStatus.DEPLOYED]),
            'ready_models': len([m for m in self.current_models.values() if m.status == ModelStatus.READY]),
            'training_models': len([m for m in self.current_models.values() if m.status == ModelStatus.TRAINING]),
            'failed_models': len([m for m in self.current_models.values() if m.status == ModelStatus.FAILED]),
            'archived_models': len([m for m in self.current_models.values() if m.status == ModelStatus.ARCHIVED]),
            'serving_models': len(self.serving_models),
            'active_experiments': len([e for e in self.experiments.values() if e.status == "RUNNING"]),
            'running_ab_tests': len([t for t in self.ab_tests.values() if t.status == "RUNNING"])
        }

    def get_model_performance_report(self, model_id: str) -> Dict[str, Any]:
        """Get detailed performance report for a model"""
        if model_id not in self.current_models:
            raise ValueError(f"Model {model_id} not found")

        model_version = self.current_models[model_id]
        stats = self.prediction_stats.get(model_id, {})

        return {
            'model_info': model_version.to_dict(),
            'performance_summary': model_version.get_performance_summary(),
            'runtime_stats': {
                'total_predictions': model_version.total_predictions,
                'average_response_time': model_version.avg_response_time,
                'last_updated': model_version.updated_at.isoformat()
            },
            'usage_stats': stats
        }

# Global model manager instance
model_manager = AdvancedModelManager()

# Convenience functions for global use
async def initialize_model_management() -> bool:
    """Initialize the advanced model management system"""
    return await model_manager.initialize()

async def register_new_model(model: Any, model_type: ModelType, input_features: List[str],
                           output_features: List[str], **kwargs) -> str:
    """Register a new model"""
    return await model_manager.register_model(model, model_type, input_features, output_features, **kwargs)

async def deploy_model_for_serving(model_id: str) -> bool:
    """Deploy a model for production serving"""
    return await model_manager.deploy_model(model_id)

async def make_prediction(model_id: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
    """Make prediction with deployed model"""
    return await model_manager.predict(model_id, input_data)

def get_model_status() -> Dict[str, Any]:
    """Get model system status"""
    return model_manager.get_model_status_summary()
