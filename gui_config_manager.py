#!/usr/bin/env python3
"""
GUI Config Manager
Handles persistent GUI settings (separate from training config)
"""

import json
import os
import logging

logger = logging.getLogger('FluxTrainingGUI')


class GUIConfigManager:
    """
    Manages GUI-specific configuration that persists between sessions
    Separate from training configuration
    """
    
    def __init__(self, config_file='gui_config.json'):
        """
        Initialize GUI config manager
        
        Args:
            config_file: Path to GUI config file
        """
        self.config_file = config_file
        self.config = self.load_or_create_default()
    
    def load_or_create_default(self):
        """
        Load existing config or create default
        
        Returns:
            dict: Configuration
        """
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                logger.info(f"Loaded GUI config from {self.config_file}")
                return config
            except Exception as e:
                logger.error(f"Error loading GUI config: {e}")
                return self.get_default_config()
        else:
            logger.info("Creating default GUI config")
            config = self.get_default_config()
            self.save()
            return config
    
    def get_default_config(self):
        """
        Get default configuration
        
        Returns:
            dict: Default config
        """
        return {
            'version': '1.0',
            
            # Model selection
            'model': {
                'selected': 'sdxl',  # sdxl, flux_dev, flux_schnell
            },
            
            # Training parameters (epochs-based)
            'training': {
                'epochs': 1,
                'samples_per_image': 10,
                'learning_rate': 0.0001,
                'lora_rank': 16,
                'lora_alpha': 8,
                'save_every_n_steps': 500,
                'lr_scheduler': 'cosine',  # safer default — prevents divergence
            },
            
            # Paths
            'paths': {
                'dataset_dir': '',
                'output_dir': './output',
                'output_name': 'flux_lora',
                'comfyui_models_path': '',       # Base path to ComfyUI/models/
                'transformer_path': '',          # Override: direct path to transformer .safetensors
                'vae_path': '',                  # Override: direct path to VAE .safetensors
                'text_encoder_repo': '',         # Override: HuggingFace repo for text encoder
            },

            # Sample generation during training
            'sampling': {
                'enabled': True,
                'prompt': '',
                'sample_every': 100,
                'width': 512,
                'height': 512,
            },

            # Window state
            'window': {
                'width': 1200,
                'height': 800,
                'tab_index': 0,
            },

            # Last used preset
            'last_preset': None,

            # Claude API settings
            'claude': {
                'api_key': '',
                'caption_prompt': 'Describe this image in detail for AI image generation training. '
                                  'Include: the main subject, art style, colors, lighting, composition, '
                                  'and any notable details. Be specific but concise (under 150 words). '
                                  'Do not start with "This image shows" or similar phrases.\n\n'
                                  'After the description, add a brief quality note on a new line starting with '
                                  '"[Quality:" - evaluate if this image is suitable for training. Note any issues '
                                  'like: watermarks, text overlays, low resolution, blur, heavy compression, '
                                  'awkward cropping, or inconsistent elements. If the image is clean and '
                                  'suitable, simply write "[Quality: Good]".',
            },
        }
    
    def save(self):
        """
        Save configuration to file
        
        Returns:
            bool: True if successful
        """
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
            logger.debug("GUI config saved")
            return True
        except Exception as e:
            logger.error(f"Error saving GUI config: {e}")
            return False
    
    def get(self, key_path, default=None):
        """
        Get a configuration value using dot notation
        
        Args:
            key_path: Path to key (e.g., 'training.epochs')
            default: Default value if not found
            
        Returns:
            Value at key path or default
        """
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def set(self, key_path, value, auto_save=True):
        """
        Set a configuration value using dot notation
        
        Args:
            key_path: Path to key (e.g., 'training.epochs')
            value: Value to set
            auto_save: Whether to save immediately
            
        Returns:
            bool: True if successful
        """
        keys = key_path.split('.')
        config = self.config
        
        # Navigate to parent
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        
        # Set value
        config[keys[-1]] = value
        
        if auto_save:
            return self.save()
        
        return True
    
    def calculate_steps(self, num_images=None):
        """
        Calculate training steps based on epochs and samples
        
        Args:
            num_images: Number of images (if known)
            
        Returns:
            int: Calculated steps
        """
        epochs = self.get('training.epochs', 1)
        samples = self.get('training.samples_per_image', 10)
        
        if num_images is None:
            # Try to count from dataset directory
            dataset_dir = self.get('paths.dataset_dir', '')
            if dataset_dir and os.path.exists(dataset_dir):
                num_images = self._count_images(dataset_dir)
            else:
                num_images = 40  # Default estimate
        
        steps = num_images * samples * epochs
        return steps
    
    def _count_images(self, directory):
        """
        Count images in directory
        
        Args:
            directory: Path to directory
            
        Returns:
            int: Number of images
        """
        image_extensions = ['.jpg', '.jpeg', '.png', '.webp', '.bmp']
        
        try:
            files = os.listdir(directory)
            images = [f for f in files 
                     if os.path.splitext(f.lower())[1] in image_extensions]
            return len(images)
        except Exception as e:
            logger.error(f"Error counting images: {e}")
            return 40  # Default
    
    def get_training_info(self, num_images=None):
        """
        Get training information summary
        
        Args:
            num_images: Number of images (optional)
            
        Returns:
            dict: Training info
        """
        if num_images is None:
            dataset_dir = self.get('paths.dataset_dir', '')
            if dataset_dir and os.path.exists(dataset_dir):
                num_images = self._count_images(dataset_dir)
            else:
                num_images = 40
        
        epochs = self.get('training.epochs', 1)
        samples = self.get('training.samples_per_image', 10)
        steps = self.calculate_steps(num_images)
        
        return {
            'num_images': num_images,
            'epochs': epochs,
            'samples_per_image': samples,
            'total_steps': steps,
            'formula': f"{num_images} × {samples} × {epochs} = {steps}",
        }
    
    def apply_to_training_config(self, training_config_manager):
        """
        Apply GUI config to training config manager

        Args:
            training_config_manager: KohyaConfigManager instance
        """
        # Calculate steps
        steps = self.calculate_steps()

        # Apply model selection (CRITICAL: this was missing and caused Klein LoRAs to train on FLUX.1-dev)
        selected_model_key = self.get('model.selected')
        if selected_model_key:
            from flux_presets import MODEL_OPTIONS
            model_info = MODEL_OPTIONS.get(selected_model_key)
            if model_info:
                training_config_manager.set_value('pretrained_model_name_or_path', model_info['model_id'])
                logger.info(f"Applied model selection: {model_info['model_id']}")

        # Apply training hyperparameters
        training_config_manager.set_value('max_train_steps', steps)
        training_config_manager.set_value('learning_rate',
                                         self.get('training.learning_rate'))
        training_config_manager.set_value('network_dim',
                                         self.get('training.lora_rank'))
        training_config_manager.set_value('network_alpha',
                                         self.get('training.lora_alpha'))
        training_config_manager.set_value('save_every_n_steps',
                                         self.get('training.save_every_n_steps'))
        training_config_manager.set_value('lr_scheduler',
                                         self.get('training.lr_scheduler', 'cosine'))
        training_config_manager.set_value('output_name',
                                         self.get('paths.output_name'))
        training_config_manager.set_value('output_dir',
                                         self.get('paths.output_dir'))

        # Apply model path settings (for local/ComfyUI models)
        training_config_manager.set_value('paths', {
            'comfyui_models_path': self.get('paths.comfyui_models_path', ''),
            'transformer_path': self.get('paths.transformer_path', ''),
            'vae_path': self.get('paths.vae_path', ''),
            'text_encoder_repo': self.get('paths.text_encoder_repo', ''),
        })

        # Apply sampling configuration
        training_config_manager.set_value('sampling', {
            'enabled': self.get('sampling.enabled', True),
            'prompt': self.get('sampling.prompt', ''),
            'sample_every': self.get('sampling.sample_every', 100),
            'width': self.get('sampling.width', 512),
            'height': self.get('sampling.height', 512),
            'seed': self.get('sampling.seed', 42),
            'random_seed': self.get('sampling.random_seed', False),
        })

        logger.info(f"Applied GUI config to training config ({steps} steps)")
