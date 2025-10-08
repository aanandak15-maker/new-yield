"""
Core configuration for India Agricultural Intelligence Platform
"""

import os
import json
import yaml
from typing import Dict, Any, Optional
from pathlib import Path

class ConfigManager:
    """Central configuration management for the platform"""

    def __init__(self, config_dir: str = "india_agri_platform/core/config"):
        self.config_dir = Path(config_dir)
        self.configs = {}
        self._load_all_configs()

    def _load_all_configs(self):
        """Load all configuration files"""
        # Platform-wide configs
        self.configs['platform'] = self._load_config('platform.yaml')

        # Crop configurations
        self.configs['crops'] = {}
        crops_dir = Path("india_agri_platform/crops")
        if crops_dir.exists():
            for crop_dir in crops_dir.iterdir():
                if crop_dir.is_dir():
                    config_file = crop_dir / 'config.json'
                    if config_file.exists():
                        with open(config_file, 'r') as f:
                            self.configs['crops'][crop_dir.name] = json.load(f)

        # State configurations
        self.configs['states'] = {}
        states_dir = Path("india_agri_platform/states")
        if states_dir.exists():
            for state_dir in states_dir.iterdir():
                if state_dir.is_dir():
                    config_file = state_dir / 'config.json'
                    if config_file.exists():
                        with open(config_file, 'r') as f:
                            self.configs['states'][state_dir.name] = json.load(f)

    def _load_config(self, filename: str) -> Dict[str, Any]:
        """Load a single configuration file"""
        file_path = self.config_dir / filename
        if file_path.exists():
            with open(file_path, 'r') as f:
                if filename.endswith('.yaml') or filename.endswith('.yml'):
                    return yaml.safe_load(f)
                elif filename.endswith('.json'):
                    return json.load(f)
        return {}

    def get_crop_config(self, crop_name: str) -> Dict[str, Any]:
        """Get configuration for a specific crop"""
        return self.configs.get('crops', {}).get(crop_name, {})

    def get_state_config(self, state_name: str) -> Dict[str, Any]:
        """Get configuration for a specific state"""
        return self.configs.get('states', {}).get(state_name, {})

    def get_platform_config(self) -> Dict[str, Any]:
        """Get platform-wide configuration"""
        return self.configs.get('platform', {})

    def update_crop_config(self, crop_name: str, config: Dict[str, Any]):
        """Update configuration for a specific crop"""
        if 'crops' not in self.configs:
            self.configs['crops'] = {}
        self.configs['crops'][crop_name] = config

        # Save to file
        crop_dir = Path(f"india_agri_platform/crops/{crop_name}")
        crop_dir.mkdir(exist_ok=True)
        config_file = crop_dir / 'config.json'
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)

    def update_state_config(self, state_name: str, config: Dict[str, Any]):
        """Update configuration for a specific state"""
        if 'states' not in self.configs:
            self.configs['states'] = {}
        self.configs['states'][state_name] = config

        # Save to file
        state_dir = Path(f"india_agri_platform/states/{state_name}")
        state_dir.mkdir(exist_ok=True)
        config_file = state_dir / 'config.json'
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)

# Global configuration instance
config_manager = ConfigManager()
