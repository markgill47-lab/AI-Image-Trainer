#!/usr/bin/env python3
"""
Kohya Training Manager for Flux Training GUI
Controls training processes for sd-scripts and monitors progress
Adapted from Hidream training manager
"""

import os
import subprocess
import threading
import time
import re
import logging
from diagnostics import check_system_resources

logger = logging.getLogger('FluxTrainingGUI')


class KohyaTrainingManager:
    """
    Manages training processes for Kohya sd-scripts
    Handles starting, stopping, and monitoring training
    Compatible with PyQt6 signal system
    """
    
    def __init__(self, update_status_callback, update_training_output_callback,
                finish_training_callback, handle_training_error_callback):
        """
        Initialize the training manager
        
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
    
    def start_training(self, command, working_dir=None):
        """
        Start a training process
        
        Args:
            command: Training command to run
            working_dir: Optional working directory
            
        Returns:
            bool: True if started successfully
        """
        if self.is_training_running():
            self.handle_training_error("Training Error", "Training is already running")
            return False
        
        try:
            # Log system resources
            logger.info("Checking system resources before starting training")
            resources = check_system_resources()
            
            logger.info(f"Starting training command: {command}")
            self.update_status(f"Starting training...")
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
                args=(command, working_dir),
                daemon=True
            )
            self.training_thread.start()
            
            return True
            
        except Exception as e:
            error_msg = f"Failed to start training: {str(e)}"
            logger.error(error_msg, exc_info=True)
            self.handle_training_error("Training Error", error_msg)
            return False
    
    def _run_training_thread(self, command, working_dir):
        """
        Run training in a background thread
        
        Args:
            command: Command to execute
            working_dir: Working directory
        """
        try:
            # Start process
            self.training_process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1,
                shell=True,
                cwd=working_dir
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
        
        Args:
            line: Output line to parse
        """
        # Look for step information
        # Example: "steps:  500/1000 | loss: 0.1234"
        step_match = re.search(r'steps?:\s*(\d+)/(\d+)', line, re.IGNORECASE)
        if step_match:
            self.current_step = int(step_match.group(1))
            self.total_steps = int(step_match.group(2))
        
        # Look for loss information
        # Example: "loss: 0.1234" or "train_loss=0.1234"
        loss_match = re.search(r'loss[=:]\s*([0-9.]+)', line, re.IGNORECASE)
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
    
    def generate_command(self, config_manager):
        """
        Generate training command from config
        
        Args:
            config_manager: KohyaConfigManager instance
            
        Returns:
            str: Training command
        """
        return config_manager.generate_training_command()
