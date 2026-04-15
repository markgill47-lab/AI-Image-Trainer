#!/usr/bin/env python3
"""
Kohya Config Manager for Flux Training GUI
Handles loading, saving, and validating configuration for sd-scripts training
Adapted from the Hidream diffusion-pipe config manager
"""

import os
import toml
import logging

logger = logging.getLogger('FluxTrainingGUI')


class KohyaConfigManager:
    """
    Manages configuration for Kohya sd-scripts training
    Compatible with PyQt6 signal system
    """
    
    def __init__(self, update_status_callback, show_error_callback, show_info_callback):
        """
        Initialize the config manager
        
        Args:
            update_status_callback: Function to update status in the UI
            show_error_callback: Function to display error messages (title, message)
            show_info_callback: Function to display info messages (title, message)
        """
        self.update_status = update_status_callback
        self.show_error = show_error_callback
        self.show_info = show_info_callback
        
        # Config data
        self.config_file = None
        self.config_data = None
        
        # Default configuration for Flux LoRA training
        self.default_config = {
            # Model
            'pretrained_model_name_or_path': 'black-forest-labs/FLUX.1-dev',
            'output_dir': './output',
            'output_name': 'flux_lora',
            
            # Training parameters
            'learning_rate': 1e-4,
            'lr_scheduler': 'constant',
            'max_train_steps': 1000,
            'save_every_n_steps': 500,
            
            # LoRA parameters
            'network_module': 'networks.lora',
            'network_dim': 16,
            'network_alpha': 8,
            
            # Optimization
            'optimizer_type': 'AdamW8bit',
            'mixed_precision': 'bf16',
            'save_precision': 'bf16',
            
            # Dataset
            'dataset_config': 'dataset.toml',
            
            # Memory/Performance
            'gradient_checkpointing': True,
            'max_data_loader_n_workers': 2,
            'persistent_data_loader_workers': True,
            
            # Logging
            'logging_dir': './logs',
            'log_with': 'tensorboard',
            
            # Save format
            'save_model_as': 'safetensors'
        }
        
        # Type specifications for validation
        self.type_specs = {
            'learning_rate': float,
            'max_train_steps': int,
            'save_every_n_steps': int,
            'network_dim': int,
            'network_alpha': int,
            'max_data_loader_n_workers': int,
            'gradient_checkpointing': bool,
            'persistent_data_loader_workers': bool,
            'pretrained_model_name_or_path': str,
            'output_dir': str,
            'output_name': str,
            'lr_scheduler': str,
            'optimizer_type': str,
            'mixed_precision': str,
            'save_precision': str,
            'network_module': str,
            'dataset_config': str,
            'logging_dir': str,
            'save_model_as': str
        }
    
    def create_default_config(self, config_path):
        """
        Create a default configuration file
        
        Args:
            config_path: Path where to save the config
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            with open(config_path, 'w') as f:
                toml.dump(self.default_config, f)
            
            self.config_file = config_path
            self.config_data = dict(self.default_config)
            self.update_status(f"Created default config at {config_path}")
            logger.info(f"Created default config at {config_path}")
            return True
            
        except Exception as e:
            error_msg = f"Failed to create default config: {str(e)}"
            self.show_error("Config Error", error_msg)
            logger.error(error_msg, exc_info=True)
            return False
    
    def load_config(self, config_path):
        """
        Load config from TOML file
        
        Args:
            config_path: Path to the config file
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not os.path.exists(config_path):
            self.update_status(f"Config file not found: {config_path}")
            logger.warning(f"Config file not found: {config_path}")
            
            # Offer to create default
            response = self.show_info(
                "Config Not Found", 
                f"Config file not found. Create default configuration?"
            )
            return self.create_default_config(config_path)
        
        try:
            self.config_data = toml.load(config_path)
            self.config_file = config_path
            self.update_status(f"Loaded config from {config_path}")
            logger.info(f"Successfully loaded config from {config_path}")
            return True
            
        except Exception as e:
            error_msg = f"Failed to load config: {str(e)}"
            self.show_error("Config Error", error_msg)
            self.update_status(f"Error loading config: {str(e)}")
            logger.error(f"Config loading error: {str(e)}", exc_info=True)
            return False
    
    def save_config(self, config_path=None):
        """
        Save the current config to file
        
        Args:
            config_path: Optional path to save to (uses self.config_file if None)
            
        Returns:
            bool: True if successful, False otherwise
        """
        save_path = config_path or self.config_file
        
        if not save_path:
            self.show_error("Save Error", "No config file path specified")
            return False
        
        if not self.config_data:
            self.show_error("Save Error", "No config data to save")
            return False
        
        try:
            # Create backup of original file
            if os.path.exists(save_path):
                import shutil
                backup_path = f"{save_path}.backup"
                shutil.copy2(save_path, backup_path)
                logger.info(f"Created backup at {backup_path}")
            
            # Save the config
            with open(save_path, 'w') as f:
                toml.dump(self.config_data, f)
            
            self.config_file = save_path
            self.update_status(f"Saved config to {save_path}")
            logger.info(f"Successfully saved config to {save_path}")
            return True
            
        except Exception as e:
            error_msg = f"Failed to save config: {str(e)}"
            self.show_error("Save Error", error_msg)
            logger.error(f"Config saving error: {str(e)}", exc_info=True)
            return False
    
    def get_value(self, key, default=None):
        """
        Get a value from the config
        
        Args:
            key: Configuration key
            default: Default value if not found
            
        Returns:
            The value, or default if not found
        """
        if not self.config_data:
            return default
        
        return self.config_data.get(key, default)
    
    def set_value(self, key, value):
        """
        Set a value in the config
        
        Args:
            key: Configuration key
            value: Value to set
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.config_data:
            self.config_data = {}
        
        try:
            self.config_data[key] = value
            logger.debug(f"Set config value {key} = {value}")
            return True
        except Exception as e:
            logger.error(f"Error setting config value {key}: {str(e)}")
            return False
    
    def validate_config(self):
        """
        Validate the configuration
        
        Returns:
            tuple: (is_valid, issues_list)
        """
        if not self.config_data:
            return False, ["No configuration loaded"]
        
        issues = []
        
        # Check required fields
        required_fields = [
            'pretrained_model_name_or_path',
            'output_dir',
            'learning_rate',
            'max_train_steps',
            'network_dim'
        ]
        
        for field in required_fields:
            if field not in self.config_data:
                issues.append(f"Missing required field: {field}")
        
        # Validate types
        for key, expected_type in self.type_specs.items():
            if key in self.config_data:
                value = self.config_data[key]
                if not isinstance(value, expected_type):
                    issues.append(f"{key} should be {expected_type.__name__}, got {type(value).__name__}")
        
        # Validate reasonable values
        if 'learning_rate' in self.config_data:
            lr = self.config_data['learning_rate']
            if lr <= 0 or lr > 0.1:
                issues.append(f"Learning rate {lr} seems unusual (typical: 1e-5 to 1e-3)")
        
        if 'max_train_steps' in self.config_data:
            steps = self.config_data['max_train_steps']
            if steps <= 0:
                issues.append("max_train_steps must be positive")
        
        if 'network_dim' in self.config_data:
            dim = self.config_data['network_dim']
            if dim <= 0 or dim > 128:
                issues.append(f"network_dim {dim} seems unusual (typical: 4-64)")
        
        return len(issues) == 0, issues
    
    def apply_preset(self, preset_name, preset_config):
        """
        Apply a preset configuration
        
        Args:
            preset_name: Name of the preset
            preset_config: Dictionary of preset values
            
        Returns:
            bool: True if successful
        """
        if not self.config_data:
            self.config_data = dict(self.default_config)
        
        try:
            # Update config with preset values
            self.config_data.update(preset_config)
            self.update_status(f"Applied preset: {preset_name}")
            logger.info(f"Applied preset: {preset_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error applying preset: {str(e)}")
            return False
    
    def generate_training_command(self):
        """
        Generate the command line for training
        
        Returns:
            str: Training command
        """
        if not self.config_data:
            return ""
        
        # Base command
        cmd_parts = ["python", "sd-scripts\\train_network.py"]
        
        # Add all config parameters as command line arguments
        for key, value in self.config_data.items():
            if isinstance(value, bool):
                if value:
                    cmd_parts.append(f"--{key}")
            else:
                cmd_parts.append(f"--{key}=\"{value}\"")
        
        return " ".join(cmd_parts)
