"""
Klein Trainer - Minimal FLUX.2 Klein LoRA Training
Supports Klein 4B and 9B models with clean, debuggable code.
"""

from .config import KleinConfig
from .dataset import KleinDataset, BucketManager, create_dataloader
from .vae import KleinVAE, encode_images, decode_latents
from .model import load_klein_transformer, load_text_encoder, encode_text, get_klein_config
from .lora import LoRANetwork, LoRALayer, apply_lora_to_transformer
from .trainer import KleinTrainer, train_klein_lora
from .manager import KleinTrainerManager, TrainingState, TrainingProgress, get_trainer_manager

__version__ = "0.1.0"

__all__ = [
    # Config
    "KleinConfig",
    # Dataset
    "KleinDataset",
    "BucketManager",
    "create_dataloader",
    # VAE
    "KleinVAE",
    "encode_images",
    "decode_latents",
    # Model
    "load_klein_transformer",
    "load_text_encoder",
    "encode_text",
    "get_klein_config",
    # LoRA
    "LoRANetwork",
    "LoRALayer",
    "apply_lora_to_transformer",
    # Trainer
    "KleinTrainer",
    "train_klein_lora",
    # Manager (GUI integration)
    "KleinTrainerManager",
    "TrainingState",
    "TrainingProgress",
    "get_trainer_manager",
]
