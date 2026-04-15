#!/usr/bin/env python3
"""
AI-Toolkit Manager for FLUX Training
Handles FLUX training using ai-toolkit framework
"""

import os
import yaml
import subprocess
import threading
import time
import re
import logging

logger = logging.getLogger('FluxTrainingGUI')


class AIToolkitManager:
    """
    Manages FLUX training using ai-toolkit
    Similar interface to KohyaTrainingManager for consistency
    """
    
    def __init__(self, update_status_callback, update_training_output_callback,
                 finish_training_callback, handle_training_error_callback):
        """
        Initialize the ai-toolkit manager
        
        Args:
            update_status_callback: Function to update status in the UI
            update_training_output_callback: Function to update training output
            finish_training_callback: Function to call when training finishes
            handle_training_error_callback: Function to handle training errors
        """
        self.update_status = update_status_callback
        self.update_training_output = update_training_output_callback
        self.finish_training = finish_training_callback
        self.handle_training_error = handle_training_error_callback
        
        # Training state
        self.training_process = None
        self.stop_training_flag = False
        self.training_thread = None
        
        # Progress tracking
        self.current_step = 0
        self.total_steps = 0
        self.training_start_time = 0
        self.last_known_loss = 0.0
        
        # Loss history for graphs
        self.loss_history = []
        self.loss_history_lock = threading.Lock()
        
        # ai-toolkit paths
        self.aitoolkit_dir = "ai-toolkit"
        self.aitoolkit_script = os.path.join(self.aitoolkit_dir, "run.py")
    
    def generate_config(self, gui_config, dataset_dir, output_dir, output_name):
        """
        Generate ai-toolkit YAML config from GUI settings
        
        Args:
            gui_config: Dictionary of GUI configuration
            dataset_dir: Path to dataset
            output_dir: Output directory for LoRA
            output_name: Name for output LoRA
            
        Returns:
            str: Path to generated config file
        """
        # Extract model name
        model_name = gui_config.get('pretrained_model_name_or_path', 'black-forest-labs/FLUX.1-dev')

        # Determine if FLUX (includes FLUX.1 and FLUX.2/Klein models)
        is_flux = 'flux' in model_name.lower() or 'klein' in model_name.lower()

        # Log which FLUX version is being used
        if 'flux.2' in model_name.lower() or 'klein' in model_name.lower():
            logger.info(f"Generating config for FLUX.2 Klein model: {model_name}")
        elif 'flux' in model_name.lower():
            logger.info(f"Generating config for FLUX.1 model: {model_name}")
        else:
            logger.info(f"Generating config for model: {model_name}")
        
        # Build ai-toolkit config with correct structure
        # FLUX.2 Klein models need special handling to avoid meta tensor issues
        is_flux2_klein = 'flux.2' in model_name.lower() or 'klein' in model_name.lower()
        is_klein_4b = '4b' in model_name.lower()

        # FLUX.2 needs higher default rank due to fused transformer architecture
        default_network_dim = 32 if is_flux2_klein else gui_config.get('network_dim', 16)
        if is_flux2_klein and gui_config.get('network_dim', 16) < 32:
            logger.warning(f"FLUX.2 Klein models work best with network_dim >= 32. Current: {gui_config.get('network_dim', 16)}")

        # Klein 4B is smaller and can keep text encoder on GPU for speed
        # Klein 9B needs low_vram mode (text encoder on CPU) to fit in 16GB
        use_low_vram = is_flux2_klein and not is_klein_4b

        # FLUX.2 Klein models need to specify the arch explicitly
        model_config = {
            'name_or_path': model_name,
            'is_flux': is_flux,
            'quantize': True,  # 8-bit quantization for transformer
            'quantize_te': True,  # Also quantize text encoder to save VRAM
            'low_vram': use_low_vram,  # Klein 4B: False (GPU), Klein 9B: True (CPU)
            'layer_offloading': False,  # DISABLE layer offloading - too slow even at 30%
            # Klein 4B fits with text encoder on GPU, Klein 9B needs CPU
        }

        # Add arch specification for FLUX.2 Klein models
        if is_flux2_klein:
            model_config['arch'] = 'flux2_klein'

            # Check for local VAE file (user may have downloaded manually)
            local_vae_paths = [
                'ae.safetensors',  # Root of project
                'models/ae.safetensors',
                'models/flux2_vae/ae.safetensors',
                'vae/diffusion_pytorch_model.safetensors',
            ]
            for vae_path in local_vae_paths:
                if os.path.exists(vae_path):
                    model_config['vae_path'] = os.path.abspath(vae_path)
                    logger.info(f"Using local VAE file: {vae_path}")
                    break

        config = {
            'job': 'extension',
            'config': {
                'name': output_name,
                'process': [
                    {
                        'type': 'sd_trainer',
                        'training_folder': output_dir,
                        'model': model_config,
                        'network': {
                            'type': 'lora',
                            'linear': gui_config.get('network_dim', default_network_dim),
                            'linear_alpha': gui_config.get('network_alpha', 8)
                        },
                        'save': {
                            'dtype': gui_config.get('save_precision', 'bf16'),
                            'save_every': gui_config.get('save_every_n_steps', 500),
                            'max_step_saves_to_keep': 3
                        },
                        'datasets': [
                            {
                                'folder_path': dataset_dir,
                                'caption_ext': 'txt',
                                'num_repeats': 10
                            }
                        ],
                        'train': {
                            'batch_size': 1,
                            'steps': gui_config.get('max_train_steps', 1000),
                            'gradient_accumulation_steps': 1,  # No gradient accumulation for speed
                            'train_unet': True,
                            'train_text_encoder': False,
                            'learning_rate': gui_config.get('learning_rate', 0.0001),
                            'lr_scheduler': gui_config.get('lr_scheduler', 'constant'),
                            'optimizer': 'adamw8bit',
                            'gradient_checkpointing': True,  # Keep this - saves VRAM without speed penalty
                            'noise_scheduler': 'flowmatch',
                            'max_denoising_steps': 1000,
                            # Memory optimizations for 16GB VRAM
                            'cache_latents': True,  # Pre-cache VAE latents
                            'cache_text_encoder_outputs': True,  # Pre-cache text encoder outputs
                            'dtype': 'bf16',  # Use bfloat16 for efficiency
                            'max_grad_norm': 1.0,  # Gradient clipping to prevent spikes
                        },
                    }
                ]
            }
        }

        # Add sampling configuration if enabled
        sampling_config = gui_config.get('sampling', {})
        if sampling_config.get('enabled', False) and sampling_config.get('prompt', '').strip():
            sample_prompt = sampling_config.get('prompt', '').strip()

            # Klein models are guidance-free architecturally, but still benefit from guidance during inference
            # Use 3.5 for Klein (empirically works better than 1.0), 4.0 for FLUX.1/2
            guidance_scale = 3.5 if is_flux2_klein else 4.0

            sample_config = {
                'sampler': 'flowmatch',  # Use flowmatch for FLUX
                'sample_every': sampling_config.get('sample_every', 100),
                'width': sampling_config.get('width', 512),
                'height': sampling_config.get('height', 512),
                'samples': [
                    {
                        'prompt': sample_prompt,
                        'neg': '',
                        'seed': 42,
                        'walk_seed': True,
                        'guidance_scale': guidance_scale,
                        'sample_steps': 50,  # Klein may need more denoising steps
                    }
                ],
                'neg': '',
                'seed': 42,
                'walk_seed': True,
                'guidance_scale': guidance_scale,
                'sample_steps': 50,  # Klein may need more denoising steps
                'sample_at_first': False  # Skip baseline sample (untrained model produces NaN)
            }
            config['config']['process'][0]['sample'] = sample_config
            logger.info(f"Sample generation enabled: every {sample_config['sample_every']} steps at {sample_config['width']}x{sample_config['height']} (guidance_scale={guidance_scale})")
        else:
            logger.info("Sample generation disabled")

        # Save config
        config_dir = 'configs'
        os.makedirs(config_dir, exist_ok=True)
        config_path = os.path.join(config_dir, f'{output_name}_aitoolkit.yaml')
        
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        logger.info(f"Generated ai-toolkit config: {config_path}")
        return config_path
    
    def start_training(self, config_path):
        """
        Start training with ai-toolkit
        
        Args:
            config_path: Path to ai-toolkit YAML config
            
        Returns:
            bool: True if started successfully
        """
        if self.is_training_running():
            self.handle_training_error("Training Error", "Training is already running")
            return False
        
        if not os.path.exists(self.aitoolkit_script):
            self.handle_training_error("AI-Toolkit Error", 
                                      f"ai-toolkit not found at {self.aitoolkit_script}")
            return False
        
        if not os.path.exists(config_path):
            self.handle_training_error("Config Error", 
                                      f"Config file not found: {config_path}")
            return False
        
        try:
            # Load config to get total steps
            try:
                with open(config_path, 'r') as f:
                    config_data = yaml.safe_load(f)
                    if 'config' in config_data and 'process' in config_data['config']:
                        for process in config_data['config']['process']:
                            if 'train' in process and 'steps' in process['train']:
                                self.total_steps = process['train']['steps']
                                logger.info(f"Training will run for {self.total_steps} steps")
                                break
            except Exception as e:
                logger.warning(f"Could not read total steps from config: {e}")
            
            # Build command
            command = f'python "{self.aitoolkit_script}" "{config_path}"'
            
            logger.info(f"Starting ai-toolkit training: {command}")
            self.update_status(f"Starting FLUX training with ai-toolkit...")
            self.update_training_output(f"Command: {command}\n\n")
            
            # Reset state
            self.stop_training_flag = False
            self.current_step = 0
            self.training_start_time = time.time()
            
            with self.loss_history_lock:
                self.loss_history.clear()
                self.last_known_loss = 0.0
            
            # Start training in background thread
            self.training_thread = threading.Thread(
                target=self._run_training_thread,
                args=(command,),
                daemon=True
            )
            self.training_thread.start()
            
            return True
            
        except Exception as e:
            error_msg = f"Failed to start training: {str(e)}"
            logger.error(error_msg, exc_info=True)
            self.handle_training_error("Training Error", error_msg)
            return False
    
    def _run_training_thread(self, command):
        """
        Run training in a background thread
        
        Args:
            command: Command to execute
        """
        try:
            # Start process
            self.training_process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1,
                shell=True
            )
            
            logger.info(f"Training process started with PID: {self.training_process.pid}")
            
            # Read output line by line
            for line in iter(self.training_process.stdout.readline, ''):
                if self.stop_training_flag:
                    logger.info("Stop flag detected, terminating training")
                    self.training_process.terminate()
                    break
                
                if line:
                    # Update output
                    self.update_training_output(line)
                    
                    # Parse for progress information
                    self._parse_training_output(line)
            
            # Wait for process to complete
            return_code = self.training_process.wait()
            
            # Handle completion
            if return_code == 0:
                self.update_status("Training completed successfully!")
                logger.info("Training completed successfully")
                self.finish_training({"status": "success", "return_code": return_code})
            elif self.stop_training_flag:
                self.update_status("Training stopped by user")
                logger.info("Training stopped by user")
                self.finish_training({"status": "stopped", "return_code": return_code})
            else:
                self.update_status(f"Training failed with code {return_code}")
                logger.error(f"Training failed with return code: {return_code}")
                self.finish_training({"status": "failed", "return_code": return_code})
        
        except Exception as e:
            error_msg = f"Training error: {str(e)}"
            logger.error(error_msg, exc_info=True)
            self.handle_training_error("Training Error", error_msg)
            self.finish_training({"status": "error", "error": str(e)})
        
        finally:
            self.training_process = None
    
    def _parse_training_output(self, line):
        """
        Parse training output for progress information
        ai-toolkit format is different from sd-scripts
        
        Args:
            line: Output line to parse
        """
        # Look for step information - try multiple formats
        # ai-toolkit formats:
        # "step: 100/1000" or "Step 100/1000" or "Step: 100/1000"
        # "flux_lora: 10%| 42/400 [00:10<01:30, 3.96s/it]" (progress bar)
        # "  0%|          | 0/1000 [00:00<?, ?it/s]"
        
        # Try format: step: 100/1000
        step_match = re.search(r'[Ss]tep[:\s]+(\d+)/(\d+)', line)
        if step_match:
            self.current_step = int(step_match.group(1))
            self.total_steps = int(step_match.group(2))
            logger.debug(f"Parsed step (format 1): {self.current_step}/{self.total_steps}")
        
        # Try progress bar format: 42/400 or | 42/400 [
        if not step_match:
            progress_match = re.search(r'\|\s*(\d+)/(\d+)\s*\[', line)
            if progress_match:
                self.current_step = int(progress_match.group(1))
                self.total_steps = int(progress_match.group(2))
                logger.debug(f"Parsed step (format 2): {self.current_step}/{self.total_steps}")
        
        # Try percentage format with counts: 10%| 42/400
        if not step_match and not progress_match:
            pct_match = re.search(r'\d+%\|?\s*(\d+)/(\d+)', line)
            if pct_match:
                self.current_step = int(pct_match.group(1))
                self.total_steps = int(pct_match.group(2))
                logger.debug(f"Parsed step (format 3): {self.current_step}/{self.total_steps}")
        
        # Look for loss information
        # ai-toolkit format: "loss: 0.1234" or "Loss: 0.1234" or "loss=0.1234"
        # Also handles scientific notation: "loss: 3.698e-01"
        loss_match = re.search(r'[Ll]oss[:\s=]+([0-9.]+(?:[eE][+-]?[0-9]+)?)', line)
        if loss_match:
            try:
                loss_value = float(loss_match.group(1))
                self.last_known_loss = loss_value

                # Add to history
                with self.loss_history_lock:
                    timestamp = time.time() - self.training_start_time
                    self.loss_history.append((self.current_step, loss_value, timestamp))

                logger.debug(f"Step {self.current_step}: loss={loss_value}")
            except ValueError:
                pass
    
    def stop_training(self):
        """
        Stop the training process
        
        Returns:
            bool: True if stop signal sent
        """
        if not self.is_training_running():
            self.update_status("No training is running")
            return False
        
        try:
            logger.info("Stopping training...")
            self.stop_training_flag = True
            self.update_status("Stopping training...")
            
            if self.training_process:
                self.training_process.terminate()
                
                # Wait a bit for graceful shutdown
                time.sleep(2)
                
                # Force kill if still running
                if self.training_process.poll() is None:
                    self.training_process.kill()
                    logger.warning("Training process force killed")
            
            return True
            
        except Exception as e:
            error_msg = f"Error stopping training: {str(e)}"
            logger.error(error_msg, exc_info=True)
            self.handle_training_error("Stop Error", error_msg)
            return False
    
    def is_training_running(self):
        """
        Check if training is currently running
        
        Returns:
            bool: True if training is running
        """
        return (self.training_process is not None and 
                self.training_process.poll() is None)
    
    def get_progress(self):
        """
        Get current training progress
        
        Returns:
            dict: Progress information
        """
        progress = {
            'current_step': self.current_step,
            'total_steps': self.total_steps,
            'progress_percent': 0,
            'last_loss': self.last_known_loss,
            'elapsed_time': 0,
            'eta': 0
        }
        
        if self.total_steps > 0:
            progress['progress_percent'] = (self.current_step / self.total_steps) * 100
        
        if self.training_start_time > 0:
            elapsed = time.time() - self.training_start_time
            progress['elapsed_time'] = elapsed
            
            # Estimate time remaining
            if self.current_step > 0:
                time_per_step = elapsed / self.current_step
                remaining_steps = self.total_steps - self.current_step
                progress['eta'] = time_per_step * remaining_steps
        
        return progress
    
    def get_loss_history(self):
        """
        Get loss history for visualization
        
        Returns:
            list: List of (step, loss, timestamp) tuples
        """
        with self.loss_history_lock:
            return list(self.loss_history)
