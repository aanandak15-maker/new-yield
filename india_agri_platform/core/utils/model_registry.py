"""
Dynamic Model Registry for Multi-Crop, Multi-State Platform
"""

import os
import pickle
import joblib
import json
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class ModelRegistry:
    """Dynamic model loading and management system"""

    def __init__(self, models_dir: str = "models"):
        # Get the root directory (parent of the india_agri_platform directory)
        current_file = Path(__file__)
        platform_root = current_file.parent.parent  # Go up from core/utils/ to india_agri_platform/
        self.models_dir = platform_root / models_dir
        self.models_dir.mkdir(exist_ok=True)

        # Cache for loaded models
        self.loaded_models = {}
        self.model_metadata = {}

        # Create subdirectories
        self.crop_models_dir = self.models_dir / "crop_models"
        self.state_models_dir = self.models_dir / "state_models"
        self.ensemble_models_dir = self.models_dir / "ensemble_models"

        for dir_path in [self.crop_models_dir, self.state_models_dir, self.ensemble_models_dir]:
            dir_path.mkdir(exist_ok=True)

        # Load model registry
        self._load_registry()

    def _load_registry(self):
        """Load model registry metadata"""
        registry_file = self.models_dir / "model_registry.json"
        if registry_file.exists():
            with open(registry_file, 'r') as f:
                self.model_metadata = json.load(f)
        else:
            self.model_metadata = {
                "crop_models": {},
                "state_models": {},
                "ensemble_models": {}
            }

    def _save_registry(self):
        """Save model registry metadata"""
        registry_file = self.models_dir / "model_registry.json"
        with open(registry_file, 'w') as f:
            json.dump(self.model_metadata, f, indent=2)

    def register_crop_model(self, crop_name: str, model_data: Dict[str, Any],
                           metadata: Dict[str, Any]) -> str:
        """Register a crop-specific model"""
        model_id = f"crop_{crop_name}_{metadata.get('version', 'v1')}"

        # Save model file
        model_file = self.crop_models_dir / f"{model_id}.pkl"
        with open(model_file, 'wb') as f:
            pickle.dump(model_data, f)

        # Update registry
        self.model_metadata["crop_models"][crop_name] = {
            "model_id": model_id,
            "file_path": str(model_file),
            "metadata": metadata,
            "registered_at": datetime.now().isoformat()
        }

        self._save_registry()
        logger.info(f"Registered crop model: {model_id}")
        return model_id

    def register_state_model(self, state_name: str, crop_name: str,
                           model_data: Dict[str, Any], metadata: Dict[str, Any]) -> str:
        """Register a state-specific model"""
        model_id = f"state_{state_name}_{crop_name}_{metadata.get('version', 'v1')}"

        # Save model file
        model_file = self.state_models_dir / f"{model_id}.pkl"
        with open(model_file, 'wb') as f:
            pickle.dump(model_data, f)

        # Update registry
        state_crop_key = f"{state_name}_{crop_name}"
        if state_crop_key not in self.model_metadata["state_models"]:
            self.model_metadata["state_models"][state_crop_key] = {}

        self.model_metadata["state_models"][state_crop_key][crop_name] = {
            "model_id": model_id,
            "file_path": str(model_file),
            "metadata": metadata,
            "registered_at": datetime.now().isoformat()
        }

        self._save_registry()
        logger.info(f"Registered state model: {model_id}")
        return model_id

    def load_crop_model(self, crop_name: str) -> Optional[Dict[str, Any]]:
        """Load a crop-specific model"""
        cache_key = f"crop_{crop_name}"

        if cache_key in self.loaded_models:
            return self.loaded_models[cache_key]

        if crop_name in self.model_metadata["crop_models"]:
            model_info = self.model_metadata["crop_models"][crop_name]
            model_file = Path(model_info["file_path"])

            if model_file.exists():
                with open(model_file, 'rb') as f:
                    model_data = pickle.load(f)

                self.loaded_models[cache_key] = model_data
                logger.info(f"Loaded crop model: {crop_name}")
                return model_data

        logger.warning(f"Crop model not found: {crop_name}")
        return None

    def load_state_model(self, state_name: str, crop_name: str) -> Optional[Dict[str, Any]]:
        """Load a state-specific model for a crop"""
        cache_key = f"state_{state_name}_{crop_name}"

        if cache_key in self.loaded_models:
            return self.loaded_models[cache_key]

        state_crop_key = f"{state_name}_{crop_name}"
        if state_crop_key in self.model_metadata["state_models"]:
            if crop_name in self.model_metadata["state_models"][state_crop_key]:
                model_info = self.model_metadata["state_models"][state_crop_key][crop_name]
                model_file = Path(model_info["file_path"])

                if model_file.exists():
                    with open(model_file, 'rb') as f:
                        model_data = pickle.load(f)

                    self.loaded_models[cache_key] = model_data
                    logger.info(f"Loaded state model: {state_name}_{crop_name}")
                    return model_data

        # Fall back to crop model
        logger.info(f"State model not found, using crop model: {crop_name}")
        return self.load_crop_model(crop_name)

    def get_model_for_crop_state(self, crop_name: str, state_name: str = None) -> Optional[Dict[str, Any]]:
        """Get the best available model for crop and state combination"""
        # Try state-specific model first
        if state_name:
            state_model = self.load_state_model(state_name, crop_name)
            if state_model:
                return state_model

        # Fall back to crop model
        crop_model = self.load_crop_model(crop_name)
        if crop_model:
            return crop_model

        logger.error(f"No model found for crop: {crop_name}, state: {state_name}")
        return None

    def list_available_models(self) -> Dict[str, Any]:
        """List all available models"""
        return {
            "crop_models": list(self.model_metadata["crop_models"].keys()),
            "state_models": list(self.model_metadata["state_models"].keys()),
            "total_crop_models": len(self.model_metadata["crop_models"]),
            "total_state_models": sum(len(models) for models in self.model_metadata["state_models"].values())
        }

    def delete_model(self, model_type: str, identifier: str) -> bool:
        """Delete a model from registry"""
        try:
            if model_type == "crop":
                if identifier in self.model_metadata["crop_models"]:
                    model_info = self.model_metadata["crop_models"][identifier]
                    model_file = Path(model_info["file_path"])
                    if model_file.exists():
                        model_file.unlink()

                    del self.model_metadata["crop_models"][identifier]
                    self._save_registry()
                    logger.info(f"Deleted crop model: {identifier}")
                    return True

            elif model_type == "state":
                # identifier should be "state_crop" format
                if identifier in self.model_metadata["state_models"]:
                    for crop_name, model_info in self.model_metadata["state_models"][identifier].items():
                        model_file = Path(model_info["file_path"])
                        if model_file.exists():
                            model_file.unlink()

                    del self.model_metadata["state_models"][identifier]
                    self._save_registry()
                    logger.info(f"Deleted state models for: {identifier}")
                    return True

            return False

        except Exception as e:
            logger.error(f"Error deleting model: {e}")
            return False

    def clear_cache(self):
        """Clear loaded models cache"""
        self.loaded_models.clear()
        logger.info("Model cache cleared")

    def get_model_performance(self, crop_name: str, state_name: str = None) -> Dict[str, Any]:
        """Get performance metrics for a model"""
        model = self.get_model_for_crop_state(crop_name, state_name)
        if model and "performance" in model:
            return model["performance"]
        return {}

# Global model registry instance
model_registry = ModelRegistry()
