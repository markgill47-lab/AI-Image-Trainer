#!/usr/bin/env python3
"""
Klein Trainer Manager for GUI Integration
Provides direct Klein training using the klein_trainer module instead of ai-toolkit subprocess.
"""

import os
import logging
import threading
import time
from pathlib import Path
from typing import Optional, Dict, Any, Callable

from klein_trainer import KleinConfig, KleinTrainerManager, TrainingState, TrainingProgress

logger = logging.getLogger('FluxTrainingGUI')


class KleinGUIManager:
    """
    Manages Klein LoRA training with direct GUI integration.
    Uses the klein_trainer module instead of subprocess calls.
    """

    def __init__(self, update_status_callback, update_training_output_callback,
                 finish_training_callback, handle_training_error_callback):
        """
        Initialize the Klein GUI manager.

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

        # Training manager
        self.trainer_manager: Optional[KleinTrainerManager] = None

        # Progress tracking
        self.current_step = 0
        self.total_steps = 0
        self.training_start_time = 0
        self.last_known_loss = 0.0

        # Loss history for graphs
        self.loss_history = []
        self.loss_history_lock = threading.Lock()

    def generate_config(self, gui_config: Dict[str, Any], dataset_dir: str,
                       output_dir: str, output_name: str) -> KleinConfig:
        """
        Generate KleinConfig from GUI settings.

        Args:
            gui_config: Dictionary of GUI configuration
            dataset_dir: Path to dataset
            output_dir: Output directory for LoRA
            output_name: Name for output LoRA

        Returns:
            KleinConfig: Configuration object for Klein trainer
        """
        model_name = gui_config.get('pretrained_model_name_or_path', 'black-forest-labs/FLUX.2-klein-base-4B')

        # Determine variant from model name
        is_klein_4b = '4b' in model_name.lower()
        model_variant = "4B" if is_klein_4b else "9B"

        logger.info(f"Generating Klein config for {model_variant} variant")

        # Resolve model paths from GUI config
        paths_config = gui_config.get('paths', {}) if isinstance(gui_config.get('paths'), dict) else {}
        comfyui_base = paths_config.get('comfyui_models_path', '')

        # --- VAE path resolution ---
        vae_path = paths_config.get('vae_path', '') or None
        if not vae_path and comfyui_base:
            # Search ComfyUI vae directory
            for name in ['flux2-vae.safetensors', 'ae.safetensors']:
                candidate = os.path.join(comfyui_base, 'vae', name)
                if os.path.exists(candidate):
                    vae_path = os.path.abspath(candidate)
                    logger.info(f"Found VAE in ComfyUI: {vae_path}")
                    break
        if not vae_path:
            # Fall back to local search paths (relative to CWD)
            for path in ['models/flux2_vae/ae.safetensors', 'models/ae.safetensors',
                         'ae.safetensors', 'vae/diffusion_pytorch_model.safetensors']:
                if os.path.exists(path):
                    vae_path = os.path.abspath(path)
                    logger.info(f"Found VAE at: {vae_path}")
                    break
        if vae_path is None:
            logger.warning("No local VAE found, will download from HuggingFace")

        # --- Transformer path resolution ---
        transformer_path = paths_config.get('transformer_path', '') or None
        if not transformer_path and comfyui_base:
            # Search ComfyUI unet directory for Klein files matching variant
            import glob
            unet_dir = os.path.join(comfyui_base, 'unet')
            if os.path.isdir(unet_dir):
                variant_lower = model_variant.lower()
                for pattern in [f'*klein*{variant_lower}*', f'*klein*{model_variant}*']:
                    matches = glob.glob(os.path.join(unet_dir, pattern))
                    if matches:
                        transformer_path = matches[0]
                        logger.info(f"Found transformer in ComfyUI: {transformer_path}")
                        break
        if transformer_path:
            logger.info(f"Using transformer: {transformer_path}")
        else:
            logger.info("No local transformer found, will download from HuggingFace")

        # --- Text encoder repo ---
        text_encoder_repo = paths_config.get('text_encoder_repo', '') or None

        # Get training parameters from GUI config
        learning_rate = float(gui_config.get('learning_rate', 0.0002))
        network_dim = int(gui_config.get('network_dim', 32))
        network_alpha = int(gui_config.get('network_alpha', 16))
        max_train_steps = int(gui_config.get('max_train_steps', 1000))
        save_every_n_steps = int(gui_config.get('save_every_n_steps', 500))

        # Get sampling settings from the 'sampling' dict in config
        sampling_config = gui_config.get('sampling', {})
        sample_every_n_steps = int(sampling_config.get('sample_every', 100))

        # Get sample prompt - can be a single string or list
        sample_prompt = sampling_config.get('prompt', '')
        if sample_prompt:
            sample_prompts = [sample_prompt] if isinstance(sample_prompt, str) else sample_prompt
        else:
            sample_prompts = []

        # Get seed settings
        sample_seed = int(sampling_config.get('seed', 42))
        random_seed = sampling_config.get('random_seed', False)
        # If random seed is enabled, use -1 to signal random seed generation
        if random_seed:
            sample_seed = -1

        # Create config
        config = KleinConfig(
            model_variant=model_variant,
            model_path=transformer_path,  # Local file or None for HuggingFace download
            vae_path=vae_path,
            text_encoder_repo=text_encoder_repo,
            comfyui_models_path=comfyui_base or None,
            lora_rank=network_dim,
            lora_alpha=network_alpha,
            lora_dropout=0.0,
            dataset_path=dataset_dir,
            caption_extension=".txt",
            resolution=int(gui_config.get('resolution', 512)),
            enable_bucket=True,
            min_bucket_resolution=256,
            max_bucket_resolution=1024,
            bucket_resolution_step=64,
            batch_size=int(gui_config.get('batch_size', 1)),
            gradient_accumulation_steps=int(gui_config.get('gradient_accumulation_steps', 1)),
            num_epochs=int(gui_config.get('num_epochs', 1)),
            max_train_steps=max_train_steps,
            learning_rate=learning_rate,
            optimizer=gui_config.get('optimizer', 'adamw8bit'),
            lr_scheduler=gui_config.get('lr_scheduler', 'constant'),
            warmup_steps=int(gui_config.get('warmup_steps', 0)),
            max_grad_norm=float(gui_config.get('max_grad_norm', 1.0)),
            gradient_checkpointing=True,
            mixed_precision=gui_config.get('mixed_precision', 'bf16'),
            cache_latents=True,
            cache_text_encoder_outputs=True,
            # Quantization re-enabled - without it the model is too slow/large
            quantize_transformer=True,
            quantize_text_encoder=True,
            sample_every_n_steps=sample_every_n_steps,
            sample_prompts=sample_prompts if sample_prompts else ["A high quality image"],
            sample_seed=sample_seed,
            sample_steps=8,  # Use fewer steps for fast training previews
            sample_guidance_scale=3.5,
            output_dir=output_dir,
            output_name=output_name,
            save_every_n_steps=save_every_n_steps,
            save_precision=gui_config.get('save_precision', 'bf16'),
            log_every_n_steps=1,
        )

        return config

    def _on_progress(self, progress: TrainingProgress):
        """Handle progress updates from trainer."""
        self.current_step = progress.current_step
        self.total_steps = progress.total_steps
        self.last_known_loss = progress.loss

        # Add to loss history (as tuples for GUI compatibility)
        if progress.loss > 0:
            with self.loss_history_lock:
                self.loss_history.append((progress.current_step, progress.loss))

        # Format status message
        state_name = progress.state.value
        percent = (progress.current_step / progress.total_steps * 100) if progress.total_steps > 0 else 0

        if progress.state == TrainingState.TRAINING:
            msg = f"Step {progress.current_step}/{progress.total_steps} ({percent:.1f}%) - Loss: {progress.loss:.4f}"
        elif progress.state == TrainingState.SAMPLING:
            msg = f"Generating samples at step {progress.current_step}..."
        elif progress.state == TrainingState.SAVING:
            msg = f"Saving checkpoint at step {progress.current_step}..."
        elif progress.state == TrainingState.LOADING:
            msg = "Loading model and preparing for training..."
        else:
            msg = f"State: {state_name}"

        self.update_training_output(msg)

    def _on_sample(self, path: str, step: int):
        """Handle sample generation callback."""
        self.update_training_output(f"Sample saved: {path}")
        self.update_status(f"Sample generated at step {step}")

    def _on_checkpoint(self, path: str, step: int):
        """Handle checkpoint save callback."""
        self.update_training_output(f"Checkpoint saved: {path}")
        self.update_status(f"Checkpoint saved at step {step}")

    def _on_error(self, message: str):
        """Handle training errors."""
        logger.error(f"Klein training error: {message}")
        self.handle_training_error(message)

    def _on_complete(self):
        """Handle training completion."""
        logger.info("Klein training completed")
        self.update_status("Training completed!")
        self.finish_training()

    def start_training(self, config: KleinConfig) -> bool:
        """
        Start Klein training.

        Args:
            config: KleinConfig object

        Returns:
            bool: True if started successfully
        """
        try:
            logger.info(f"Starting Klein training: {config.model_variant} variant")
            self.update_status(f"Starting Klein {config.model_variant} training...")

            # Reset state
            self.current_step = 0
            self.total_steps = config.max_train_steps or 0
            self.training_start_time = time.time()
            with self.loss_history_lock:
                self.loss_history = []

            # Create trainer manager
            self.trainer_manager = KleinTrainerManager()

            # Register callbacks
            self.trainer_manager.on_progress(self._on_progress)
            self.trainer_manager.on_sample(self._on_sample)
            self.trainer_manager.on_checkpoint(self._on_checkpoint)
            self.trainer_manager.on_error(self._on_error)
            self.trainer_manager.on_complete(self._on_complete)

            # Load config and start training
            self.trainer_manager.load_config(config)
            self.trainer_manager.start_training()

            return True

        except Exception as e:
            logger.error(f"Failed to start Klein training: {e}")
            self.handle_training_error(str(e))
            return False

    def stop_training(self) -> bool:
        """
        Stop training.

        Returns:
            bool: True if stopped
        """
        if self.trainer_manager is not None:
            logger.info("Stopping Klein training...")
            self.trainer_manager.stop()
            self.update_status("Training stopped")
            return True
        return False

    def pause_training(self) -> bool:
        """
        Pause training.

        Returns:
            bool: True if paused
        """
        if self.trainer_manager is not None and self.trainer_manager.is_training():
            self.trainer_manager.pause()
            self.update_status("Training paused")
            return True
        return False

    def resume_training(self) -> bool:
        """
        Resume training.

        Returns:
            bool: True if resumed
        """
        if self.trainer_manager is not None and self.trainer_manager.is_paused():
            self.trainer_manager.resume()
            self.update_status("Training resumed")
            return True
        return False

    def is_training_running(self) -> bool:
        """
        Check if training is running.

        Returns:
            bool: True if training
        """
        if self.trainer_manager is not None:
            return self.trainer_manager.is_training()
        return False

    def get_progress(self) -> Dict[str, Any]:
        """
        Get current progress.

        Returns:
            dict: Progress information
        """
        elapsed = time.time() - self.training_start_time if self.training_start_time > 0 else 0

        if self.total_steps > 0 and self.current_step > 0:
            steps_per_sec = self.current_step / elapsed if elapsed > 0 else 0
            remaining_steps = self.total_steps - self.current_step
            eta = remaining_steps / steps_per_sec if steps_per_sec > 0 else 0
        else:
            eta = 0

        return {
            'current_step': self.current_step,
            'total_steps': self.total_steps,
            'progress_percent': (self.current_step / self.total_steps * 100) if self.total_steps > 0 else 0,
            'last_loss': self.last_known_loss,
            'elapsed_time': elapsed,
            'eta': eta
        }

    def get_loss_history(self) -> list:
        """
        Get loss history. Prefers analytics data if available.

        Returns:
            list: Loss history as list of (step, loss) tuples
        """
        # Try to get from analytics first (more reliable, persistent)
        if self.trainer_manager is not None:
            analytics = self.trainer_manager.get_analytics()
            if analytics is not None:
                return analytics.get_loss_history()

        # Fallback to in-memory history
        with self.loss_history_lock:
            return self.loss_history.copy()

    def get_trend_info(self) -> Dict[str, Any]:
        """
        Get current trend analysis from analytics.

        Returns:
            dict: Trend info with status, slope, EMA values
        """
        if self.trainer_manager is not None:
            return self.trainer_manager.get_trend_info()
        return {"status": "unknown", "slope": None}

    def get_outlier_images(self, top_n: int = 10) -> list:
        """
        Get problematic images based on loss.

        Args:
            top_n: Number of top outliers to return

        Returns:
            list: List of dicts with path, filename, mean_loss, z_score
        """
        if self.trainer_manager is not None:
            return self.trainer_manager.get_outlier_images(top_n)
        return []

    def get_ema_series(self, window: int) -> list:
        """
        Get EMA series for a specific window.

        Args:
            window: EMA window size (10, 50, or 100)

        Returns:
            list: List of (step, ema_value) tuples
        """
        if self.trainer_manager is not None:
            analytics = self.trainer_manager.get_analytics()
            if analytics is not None:
                return analytics.get_ema_series(window)
        return []

    def export_metrics(self, format: str = "json") -> str:
        """
        Export all metrics to a format.

        Args:
            format: "json" or "csv"

        Returns:
            str: Formatted metrics data
        """
        if self.trainer_manager is not None:
            analytics = self.trainer_manager.get_analytics()
            if analytics is not None:
                return analytics.export_metrics(format)
        return ""

    def update_live_config(
        self,
        learning_rate: float = None,
        sample_every: int = None,
        sample_prompt: str = None,
        save_every_n_steps: int = None,
    ) -> Dict[str, Any]:
        """
        Update training configuration on-the-fly.

        Args:
            learning_rate: New learning rate
            sample_every: New sample generation frequency
            sample_prompt: New prompt for sample generation
            save_every_n_steps: New checkpoint save frequency

        Returns:
            dict: Summary of updates applied
        """
        if self.trainer_manager is not None:
            return self.trainer_manager.update_live_config(
                learning_rate=learning_rate,
                sample_every=sample_every,
                sample_prompt=sample_prompt,
                save_every_n_steps=save_every_n_steps,
            )
        return {"error": "No trainer manager available"}
