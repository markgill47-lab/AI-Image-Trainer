"""
Manager class for integrating Klein trainer with GUI.
Provides callbacks and state management for UI updates.
"""

import torch
from typing import Optional, Dict, Any, Callable, List
from pathlib import Path
import threading
import queue
from dataclasses import dataclass
from enum import Enum

from .config import KleinConfig
from .trainer import KleinTrainer
from .analytics import TrainingAnalytics


class TrainingState(Enum):
    """Training state enumeration."""
    IDLE = "idle"
    LOADING = "loading"
    TRAINING = "training"
    SAMPLING = "sampling"
    SAVING = "saving"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"
    COMPLETE = "complete"


@dataclass
class TrainingProgress:
    """Training progress data for UI updates."""
    state: TrainingState
    current_step: int
    total_steps: int
    current_epoch: int
    total_epochs: int
    loss: float
    learning_rate: float
    samples_generated: int
    checkpoints_saved: int
    error_message: Optional[str] = None


class KleinTrainerManager:
    """
    Manager for Klein trainer with GUI integration support.

    Provides:
    - Asynchronous training in background thread
    - Progress callbacks for UI updates
    - Pause/resume/stop controls
    - Sample image callbacks
    - Checkpoint callbacks
    """

    def __init__(self):
        self.trainer: Optional[KleinTrainer] = None
        self.config: Optional[KleinConfig] = None
        self.training_thread: Optional[threading.Thread] = None
        self.state = TrainingState.IDLE

        # Analytics module for metrics tracking
        self.analytics: Optional[TrainingAnalytics] = None

        # Control flags
        self._stop_requested = False
        self._pause_requested = False

        # Callbacks
        self._progress_callbacks: List[Callable[[TrainingProgress], None]] = []
        self._sample_callbacks: List[Callable[[str, int], None]] = []
        self._checkpoint_callbacks: List[Callable[[str, int], None]] = []
        self._error_callbacks: List[Callable[[str], None]] = []
        self._complete_callbacks: List[Callable[[], None]] = []

        # Progress tracking
        self.progress = TrainingProgress(
            state=TrainingState.IDLE,
            current_step=0,
            total_steps=0,
            current_epoch=0,
            total_epochs=0,
            loss=0.0,
            learning_rate=0.0,
            samples_generated=0,
            checkpoints_saved=0,
        )

        # Message queue for thread-safe communication
        self._message_queue = queue.Queue()

    def on_progress(self, callback: Callable[[TrainingProgress], None]):
        """Register callback for progress updates."""
        self._progress_callbacks.append(callback)

    def on_sample(self, callback: Callable[[str, int], None]):
        """Register callback for sample generation (path, step)."""
        self._sample_callbacks.append(callback)

    def on_checkpoint(self, callback: Callable[[str, int], None]):
        """Register callback for checkpoint saves (path, step)."""
        self._checkpoint_callbacks.append(callback)

    def on_error(self, callback: Callable[[str], None]):
        """Register callback for errors."""
        self._error_callbacks.append(callback)

    def on_complete(self, callback: Callable[[], None]):
        """Register callback for training completion."""
        self._complete_callbacks.append(callback)

    def _notify_progress(self):
        """Notify all progress callbacks."""
        for callback in self._progress_callbacks:
            try:
                callback(self.progress)
            except Exception as e:
                print(f"Progress callback error: {e}")

    def _notify_sample(self, path: str, step: int):
        """Notify all sample callbacks."""
        for callback in self._sample_callbacks:
            try:
                callback(path, step)
            except Exception as e:
                print(f"Sample callback error: {e}")

    def _notify_checkpoint(self, path: str, step: int):
        """Notify all checkpoint callbacks."""
        for callback in self._checkpoint_callbacks:
            try:
                callback(path, step)
            except Exception as e:
                print(f"Checkpoint callback error: {e}")

    def _notify_error(self, message: str):
        """Notify all error callbacks."""
        for callback in self._error_callbacks:
            try:
                callback(message)
            except Exception as e:
                print(f"Error callback error: {e}")

    def _notify_complete(self):
        """Notify all complete callbacks."""
        for callback in self._complete_callbacks:
            try:
                callback()
            except Exception as e:
                print(f"Complete callback error: {e}")

    def load_config(self, config: KleinConfig):
        """Load training configuration."""
        self.config = config
        self.progress.total_epochs = config.num_epochs
        self._notify_progress()

    def load_config_from_dict(self, config_dict: Dict[str, Any]):
        """Load training configuration from dictionary."""
        self.config = KleinConfig.from_dict(config_dict)
        self.progress.total_epochs = self.config.num_epochs
        self._notify_progress()

    def start_training(self):
        """Start training in background thread."""
        if self.config is None:
            raise ValueError("Config must be loaded before starting training")

        if self.state == TrainingState.TRAINING:
            raise RuntimeError("Training already in progress")

        self._stop_requested = False
        self._pause_requested = False
        self.state = TrainingState.LOADING

        self.training_thread = threading.Thread(target=self._training_loop)
        self.training_thread.daemon = True
        self.training_thread.start()

    def _training_loop(self):
        """Main training loop running in background thread."""
        try:
            # Update state
            self.state = TrainingState.LOADING
            self.progress.state = TrainingState.LOADING
            self._notify_progress()

            # Create and setup trainer
            self.trainer = KleinTrainer(self.config)

            # Initialize analytics module
            self.analytics = TrainingAnalytics(
                output_dir=self.config.output_dir,
                output_name=self.config.output_name,
                enable_per_sample=True,
            )

            # Register internal callbacks
            self.trainer.register_callback("step_end", self._on_step_end)
            self.trainer.register_callback("log", self._on_log)
            self.trainer.register_callback("save", self._on_save)

            # Setup trainer
            self.trainer.setup()

            # Calculate total steps
            total_steps = self.trainer._calculate_total_steps()
            self.progress.total_steps = total_steps

            # Update state
            self.state = TrainingState.TRAINING
            self.progress.state = TrainingState.TRAINING
            self._notify_progress()

            # Modified training loop with pause/stop support
            self._run_training_with_controls()

            # Training complete
            self.state = TrainingState.COMPLETE
            self.progress.state = TrainingState.COMPLETE
            self._notify_progress()
            self._notify_complete()

        except Exception as e:
            self.state = TrainingState.ERROR
            self.progress.state = TrainingState.ERROR
            self.progress.error_message = str(e)
            self._notify_progress()
            self._notify_error(str(e))
            raise

    def _run_training_with_controls(self):
        """Run training with pause/stop controls."""
        trainer = self.trainer
        config = self.config
        total_steps = trainer._calculate_total_steps()
        gradient_accumulation_steps = config.gradient_accumulation_steps

        trainer.lora_network.train()

        accumulated_loss = 0.0
        accumulation_counter = 0

        while trainer.global_step < total_steps:
            # Check for stop request
            if self._stop_requested:
                self.state = TrainingState.STOPPED
                self.progress.state = TrainingState.STOPPED
                self._notify_progress()
                return

            # Check for pause request
            while self._pause_requested:
                self.state = TrainingState.PAUSED
                self.progress.state = TrainingState.PAUSED
                self._notify_progress()

                if self._stop_requested:
                    self.state = TrainingState.STOPPED
                    self.progress.state = TrainingState.STOPPED
                    self._notify_progress()
                    return

                import time
                time.sleep(0.1)

            # Resume from pause
            if self.state == TrainingState.PAUSED:
                self.state = TrainingState.TRAINING
                self.progress.state = TrainingState.TRAINING
                self._notify_progress()

            trainer.current_epoch += 1

            for batch in trainer.dataloader:
                if trainer.global_step >= total_steps:
                    break

                if self._stop_requested:
                    return

                while self._pause_requested:
                    if self._stop_requested:
                        return
                    import time
                    time.sleep(0.1)

                # Training step (returns loss and per-sample data)
                loss, per_sample_data = trainer._training_step(batch)
                accumulated_loss += loss
                accumulation_counter += 1
                current_per_sample_data = per_sample_data

                if accumulation_counter >= gradient_accumulation_steps:
                    torch.nn.utils.clip_grad_norm_(
                        trainer.lora_network.parameters(),
                        config.max_grad_norm,
                    )

                    trainer.optimizer.step()
                    trainer.lr_scheduler.step()
                    trainer.optimizer.zero_grad()

                    trainer.global_step += 1
                    avg_loss = accumulated_loss / accumulation_counter
                    accumulated_loss = 0.0
                    accumulation_counter = 0

                    # Record analytics
                    if self.analytics is not None:
                        lr = trainer.optimizer.param_groups[0]["lr"]
                        self.analytics.record_step(
                            step=trainer.global_step,
                            loss=avg_loss,
                            lr=lr,
                            per_sample_losses=current_per_sample_data,
                        )

                    # Update progress
                    self.progress.current_step = trainer.global_step
                    self.progress.current_epoch = trainer.current_epoch
                    self.progress.loss = avg_loss
                    self.progress.learning_rate = trainer.optimizer.param_groups[0]["lr"]
                    self._notify_progress()

                    # Sampling
                    if trainer.global_step % config.sample_every_n_steps == 0:
                        self.state = TrainingState.SAMPLING
                        self.progress.state = TrainingState.SAMPLING
                        self._notify_progress()

                        trainer._generate_samples()
                        self.progress.samples_generated += len(config.sample_prompts)

                        self.state = TrainingState.TRAINING
                        self.progress.state = TrainingState.TRAINING
                        self._notify_progress()

                    # Saving
                    if trainer.global_step % config.save_every_n_steps == 0:
                        self.state = TrainingState.SAVING
                        self.progress.state = TrainingState.SAVING
                        self._notify_progress()

                        trainer._save_checkpoint()
                        self.progress.checkpoints_saved += 1

                        self.state = TrainingState.TRAINING
                        self.progress.state = TrainingState.TRAINING
                        self._notify_progress()

        # Final save
        trainer._save_checkpoint(final=True)
        self.progress.checkpoints_saved += 1

    def _on_step_end(self, data: Dict[str, Any]):
        """Internal callback for step end."""
        self.progress.current_step = data["step"]
        self.progress.loss = data["loss"]

    def _on_log(self, data: Dict[str, Any]):
        """Internal callback for logging."""
        self.progress.learning_rate = data["lr"]

    def _on_save(self, data: Dict[str, Any]):
        """Internal callback for saving."""
        self._notify_checkpoint(data["path"], data["step"])

    def pause(self):
        """Pause training."""
        if self.state == TrainingState.TRAINING:
            self._pause_requested = True

    def resume(self):
        """Resume training."""
        if self.state == TrainingState.PAUSED:
            self._pause_requested = False

    def stop(self):
        """Stop training."""
        self._stop_requested = True
        self._pause_requested = False

    def is_training(self) -> bool:
        """Check if training is in progress."""
        return self.state in [TrainingState.TRAINING, TrainingState.SAMPLING, TrainingState.SAVING]

    def is_paused(self) -> bool:
        """Check if training is paused."""
        return self.state == TrainingState.PAUSED

    def get_progress_percent(self) -> float:
        """Get training progress as percentage."""
        if self.progress.total_steps == 0:
            return 0.0
        return (self.progress.current_step / self.progress.total_steps) * 100

    def get_analytics(self) -> Optional[TrainingAnalytics]:
        """Get the analytics module instance."""
        return self.analytics

    def get_loss_history(self) -> List:
        """Get loss history from analytics."""
        if self.analytics is not None:
            return self.analytics.get_loss_history()
        return []

    def get_trend_info(self) -> Dict[str, Any]:
        """Get current trend analysis."""
        if self.analytics is not None:
            return self.analytics.get_current_trend()
        return {"status": "unknown", "slope": None}

    def get_outlier_images(self, top_n: int = 10) -> List[Dict[str, Any]]:
        """Get problematic images based on loss."""
        if self.analytics is not None:
            return self.analytics.get_outlier_images(top_n)
        return []

    def update_live_config(
        self,
        learning_rate: Optional[float] = None,
        sample_every: Optional[int] = None,
        sample_prompt: Optional[str] = None,
        save_every_n_steps: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Update training configuration on-the-fly during training.

        Args:
            learning_rate: New learning rate (updates optimizer directly)
            sample_every: New sample generation frequency
            sample_prompt: New prompt for sample generation
            save_every_n_steps: New checkpoint save frequency

        Returns:
            dict: Summary of what was updated
        """
        updates = {}

        if self.trainer is None or self.config is None:
            return {"error": "No active training session"}

        # Update learning rate in optimizer
        if learning_rate is not None:
            old_lr = self.trainer.optimizer.param_groups[0]['lr']
            for param_group in self.trainer.optimizer.param_groups:
                param_group['lr'] = learning_rate
            self.config.learning_rate = learning_rate
            updates['learning_rate'] = {'old': old_lr, 'new': learning_rate}

        # Update sample frequency
        if sample_every is not None:
            old_val = self.config.sample_every
            self.config.sample_every = sample_every
            updates['sample_every'] = {'old': old_val, 'new': sample_every}

        # Update sample prompt
        if sample_prompt is not None:
            old_val = self.config.sample_prompts[0] if self.config.sample_prompts else ""
            self.config.sample_prompts = [sample_prompt]
            updates['sample_prompt'] = {'old': old_val[:50] + '...' if len(old_val) > 50 else old_val,
                                        'new': sample_prompt[:50] + '...' if len(sample_prompt) > 50 else sample_prompt}

        # Update checkpoint save frequency
        if save_every_n_steps is not None:
            old_val = self.config.save_every_n_steps
            self.config.save_every_n_steps = save_every_n_steps
            updates['save_every_n_steps'] = {'old': old_val, 'new': save_every_n_steps}

        return updates


# Singleton instance for GUI integration
_manager_instance: Optional[KleinTrainerManager] = None


def get_trainer_manager() -> KleinTrainerManager:
    """Get the global trainer manager instance."""
    global _manager_instance
    if _manager_instance is None:
        _manager_instance = KleinTrainerManager()
    return _manager_instance
