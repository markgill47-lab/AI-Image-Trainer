#!/usr/bin/env python3
"""
Enhanced Training Manager
Routes training to appropriate backend based on model type:
- sd-scripts for Stable Diffusion / SDXL
- ai-toolkit for FLUX.1 models
- klein_trainer for FLUX.2 Klein models (direct Python, no subprocess)
"""

import logging
from kohya_training_manager import KohyaTrainingManager
from flux_aitoolkit_manager import AIToolkitManager
from klein_gui_manager import KleinGUIManager

logger = logging.getLogger('FluxTrainingGUI')


class EnhancedTrainingManager:
    """
    Unified training manager that routes to correct backend
    Provides consistent interface regardless of backend
    """
    
    def __init__(self, update_status_callback, update_training_output_callback,
                 finish_training_callback, handle_training_error_callback):
        """
        Initialize both training backends
        
        Args:
            update_status_callback: Function to update status in the UI
            update_training_output_callback: Function to update training output
            finish_training_callback: Function to call when training finishes
            handle_training_error_callback: Function to handle training errors
        """
        self.update_status = update_status_callback
        
        # Initialize both backends
        self.kohya_manager = KohyaTrainingManager(
            update_status_callback,
            update_training_output_callback,
            finish_training_callback,
            handle_training_error_callback
        )
        
        self.aitoolkit_manager = AIToolkitManager(
            update_status_callback,
            update_training_output_callback,
            finish_training_callback,
            handle_training_error_callback
        )

        # Klein trainer for FLUX.2 Klein models (direct Python training)
        self.klein_manager = KleinGUIManager(
            update_status_callback,
            update_training_output_callback,
            finish_training_callback,
            handle_training_error_callback
        )

        # Track which backend is active
        self.active_backend = None
        self.current_model = None

        # Option to use direct Klein trainer for Klein models (default: True)
        self.use_direct_klein = True
    
    def detect_backend(self, model_name):
        """
        Detect which backend to use based on model name

        Args:
            model_name: Model identifier

        Returns:
            str: 'kohya', 'aitoolkit', or 'klein'
        """
        model_lower = model_name.lower()

        # FLUX.2 Klein models can use direct Klein trainer
        if 'flux.2' in model_lower or 'klein' in model_lower:
            if self.use_direct_klein:
                logger.info(f"Detected FLUX.2 Klein model: {model_name} -> using direct Klein trainer")
                return 'klein'
            else:
                logger.info(f"Detected FLUX.2 Klein model: {model_name} -> using ai-toolkit")
                return 'aitoolkit'

        # FLUX.1 models use ai-toolkit
        elif 'flux' in model_lower:
            logger.info(f"Detected FLUX.1 model: {model_name} -> using ai-toolkit")
            return 'aitoolkit'

        # Stable Diffusion and SDXL use sd-scripts (Kohya)
        elif any(x in model_lower for x in ['stable-diffusion', 'sdxl', 'stabilityai']):
            logger.info(f"Detected SD/SDXL model: {model_name} -> using sd-scripts")
            return 'kohya'

        # Default to kohya for unknown
        else:
            logger.warning(f"Unknown model type: {model_name} -> defaulting to sd-scripts")
            return 'kohya'
    
    def start_training_kohya(self, command):
        """
        Start training with sd-scripts (Kohya)
        
        Args:
            command: Training command
            
        Returns:
            bool: True if started
        """
        self.active_backend = 'kohya'
        return self.kohya_manager.start_training(command)
    
    def start_training_aitoolkit(self, config_path):
        """
        Start training with ai-toolkit
        
        Args:
            config_path: Path to ai-toolkit config
            
        Returns:
            bool: True if started
        """
        self.active_backend = 'aitoolkit'
        return self.aitoolkit_manager.start_training(config_path)
    
    def start_training(self, config_manager, dataset_manager):
        """
        Start training with appropriate backend

        Args:
            config_manager: Configuration manager instance
            dataset_manager: Dataset manager instance

        Returns:
            bool: True if started successfully
        """
        # Get model name from config
        model_name = config_manager.get_value('pretrained_model_name_or_path', '')

        if not model_name:
            self.update_status("Error: No model configured")
            return False

        self.current_model = model_name

        # Detect which backend to use
        backend = self.detect_backend(model_name)

        if backend == 'klein':
            # Use direct Klein trainer for FLUX.2 Klein models
            self.update_status("Generating Klein trainer configuration...")

            gui_config = config_manager.config_data
            dataset_dir = dataset_manager.dataset_config['datasets'][0]['subsets'][0]['image_dir']
            output_dir = config_manager.get_value('output_dir', './output')
            output_name = config_manager.get_value('output_name', 'klein_lora')

            klein_config = self.klein_manager.generate_config(
                gui_config,
                dataset_dir,
                output_dir,
                output_name
            )

            self.active_backend = 'klein'
            self.update_status(f"Using Klein trainer for FLUX.2 Klein training...")
            return self.klein_manager.start_training(klein_config)

        elif backend == 'aitoolkit':
            # Generate ai-toolkit config
            self.update_status("Generating ai-toolkit configuration...")

            gui_config = config_manager.config_data
            dataset_dir = dataset_manager.dataset_config['datasets'][0]['subsets'][0]['image_dir']
            output_dir = config_manager.get_value('output_dir', './output')
            output_name = config_manager.get_value('output_name', 'flux_lora')

            config_path = self.aitoolkit_manager.generate_config(
                gui_config,
                dataset_dir,
                output_dir,
                output_name
            )

            self.update_status(f"Using ai-toolkit for FLUX training...")
            return self.start_training_aitoolkit(config_path)

        else:  # kohya
            # Generate sd-scripts command
            self.update_status("Generating training command for sd-scripts...")
            command = config_manager.generate_training_command()

            self.update_status(f"Using sd-scripts for SD/SDXL training...")
            return self.start_training_kohya(command)
    
    def stop_training(self):
        """
        Stop currently running training

        Returns:
            bool: True if stopped
        """
        if self.active_backend == 'kohya':
            return self.kohya_manager.stop_training()
        elif self.active_backend == 'aitoolkit':
            return self.aitoolkit_manager.stop_training()
        elif self.active_backend == 'klein':
            return self.klein_manager.stop_training()
        else:
            self.update_status("No training is running")
            return False

    def is_training_running(self):
        """
        Check if any training is running

        Returns:
            bool: True if training is running
        """
        return (self.kohya_manager.is_training_running() or
                self.aitoolkit_manager.is_training_running() or
                self.klein_manager.is_training_running())

    def get_progress(self):
        """
        Get progress from active backend

        Returns:
            dict: Progress information
        """
        if self.active_backend == 'kohya':
            return self.kohya_manager.get_progress()
        elif self.active_backend == 'aitoolkit':
            return self.aitoolkit_manager.get_progress()
        elif self.active_backend == 'klein':
            return self.klein_manager.get_progress()
        else:
            return {
                'current_step': 0,
                'total_steps': 0,
                'progress_percent': 0,
                'last_loss': 0.0,
                'elapsed_time': 0,
                'eta': 0
            }

    def get_loss_history(self):
        """
        Get loss history from active backend

        Returns:
            list: Loss history
        """
        if self.active_backend == 'kohya':
            return self.kohya_manager.get_loss_history()
        elif self.active_backend == 'aitoolkit':
            return self.aitoolkit_manager.get_loss_history()
        elif self.active_backend == 'klein':
            return self.klein_manager.get_loss_history()
        else:
            return []
    
    def get_active_backend(self):
        """
        Get the currently active backend
        
        Returns:
            str: 'kohya', 'aitoolkit', or None
        """
        return self.active_backend
    
    def get_backend_info(self):
        """
        Get information about available backends

        Returns:
            dict: Backend availability and info
        """
        import os

        return {
            'kohya': {
                'available': os.path.exists('sd-scripts/train_network.py'),
                'name': 'Kohya sd-scripts',
                'supports': ['SD 1.5', 'SDXL']
            },
            'aitoolkit': {
                'available': os.path.exists('ai-toolkit/run.py'),
                'name': 'ai-toolkit',
                'supports': ['FLUX.1-dev', 'FLUX.1-schnell', 'SDXL', 'SD 1.5']
            },
            'klein': {
                'available': True,  # Klein trainer is always available (direct Python)
                'name': 'Klein Trainer (Direct)',
                'supports': ['FLUX.2-klein-4B', 'FLUX.2-klein-9B']
            }
        }
