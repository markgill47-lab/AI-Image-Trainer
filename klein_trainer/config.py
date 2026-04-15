"""
Configuration classes for Klein trainer.
Simple, explicit configuration with no magic.
"""

from dataclasses import dataclass, field
from typing import Optional, List
from pathlib import Path


@dataclass
class KleinConfig:
    """Main configuration for Klein LoRA training."""

    # === Model Settings ===
    model_variant: str = "4B"  # "4B" or "9B"
    model_path: Optional[str] = None  # Local path or HuggingFace repo
    vae_path: Optional[str] = None  # Path to BFL-format VAE (ae.safetensors)
    text_encoder_repo: Optional[str] = None  # HuggingFace repo for text encoder (default: Klein repo)
    comfyui_models_path: Optional[str] = None  # Base path to ComfyUI models/ dir

    # === LoRA Settings ===
    lora_rank: int = 32
    lora_alpha: int = 32
    lora_dropout: float = 0.0

    # === Dataset Settings ===
    dataset_path: str = ""
    caption_extension: str = ".txt"
    resolution: int = 512
    enable_bucket: bool = True
    min_bucket_resolution: int = 256
    max_bucket_resolution: int = 1024
    bucket_resolution_step: int = 64

    # === Training Settings ===
    batch_size: int = 1
    gradient_accumulation_steps: int = 1
    num_epochs: int = 1
    max_train_steps: Optional[int] = None  # If set, overrides epochs
    learning_rate: float = 1e-4
    optimizer: str = "adamw8bit"  # adamw, adamw8bit, prodigy
    lr_scheduler: str = "constant"  # constant, cosine, linear
    warmup_steps: int = 0
    max_grad_norm: float = 1.0

    # === Memory Optimization ===
    gradient_checkpointing: bool = True
    mixed_precision: str = "bf16"  # fp32, fp16, bf16
    cache_latents: bool = True
    cache_text_encoder_outputs: bool = True
    quantize_transformer: bool = True
    quantize_text_encoder: bool = True

    # === Sampling Settings ===
    sample_every_n_steps: int = 100
    sample_prompts: List[str] = field(default_factory=lambda: ["a photo"])
    sample_seed: int = 42
    sample_steps: int = 8  # Use fewer steps for fast training previews (full quality: 28)
    sample_guidance_scale: float = 3.5

    # === Save Settings ===
    output_dir: str = "./output"
    output_name: str = "klein_lora"
    save_every_n_steps: int = 500
    save_precision: str = "bf16"

    # === Logging ===
    log_every_n_steps: int = 1

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.model_variant not in ["4B", "9B"]:
            raise ValueError(f"model_variant must be '4B' or '9B', got '{self.model_variant}'")

        if self.mixed_precision not in ["fp32", "fp16", "bf16"]:
            raise ValueError(f"mixed_precision must be 'fp32', 'fp16', or 'bf16'")

        if not self.dataset_path:
            raise ValueError("dataset_path must be specified")

    @property
    def model_repo(self) -> str:
        """Get HuggingFace repo for the model variant."""
        if self.model_path:
            return self.model_path
        return f"black-forest-labs/FLUX.2-klein-base-{self.model_variant}"

    @property
    def dtype(self):
        """Get torch dtype from mixed_precision setting."""
        import torch
        if self.mixed_precision == "bf16":
            return torch.bfloat16
        elif self.mixed_precision == "fp16":
            return torch.float16
        return torch.float32

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            k: str(v) if isinstance(v, Path) else v
            for k, v in self.__dict__.items()
        }

    @classmethod
    def from_dict(cls, d: dict) -> "KleinConfig":
        """Create config from dictionary."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
