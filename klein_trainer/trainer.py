"""
Main trainer class for Klein LoRA training.
Implements flow matching training with proper sample generation.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from typing import Optional, Dict, Any, Callable, Tuple, List
from pathlib import Path
from tqdm import tqdm
import math
import json
from datetime import datetime

# Import sampling utilities via the isolated flux2_src package.
# We must import model.py first (it sets up flux2_src), then register
# sampling.py into the same package so its relative imports work.
from .model import Flux2Klein  # noqa: F401 — triggers flux2_src setup

import importlib.util as _ilu
_AI_TOOLKIT_PATH = Path(__file__).parent.parent / "ai-toolkit"
_sampling_path = _AI_TOOLKIT_PATH / "extensions_built_in" / "diffusion_models" / "flux2" / "src" / "sampling.py"

if "flux2_src.sampling" not in sys.modules:
    _spec = _ilu.spec_from_file_location(
        "flux2_src.sampling",
        str(_sampling_path),
        submodule_search_locations=[]
    )
    _mod = _ilu.module_from_spec(_spec)
    sys.modules["flux2_src.sampling"] = _mod
    _spec.loader.exec_module(_mod)

from flux2_src.sampling import (
    batched_prc_img,
    batched_prc_txt,
    get_schedule,
    denoise,
    scatter_ids,
)

from .config import KleinConfig
from .dataset import KleinDataset, create_dataloader
from .vae import KleinVAE, encode_images, decode_latents
from .model import load_klein_transformer, load_text_encoder, encode_text, get_klein_config
from .lora import apply_lora_to_transformer, LoRANetwork


class KleinTrainer:
    """
    Main trainer for Klein LoRA training.

    Features:
    - Flow matching training objective
    - Latent caching for memory efficiency
    - Text embedding caching
    - Proper sample generation with LoRA applied
    - Clean, debuggable code
    """

    def __init__(self, config: KleinConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = config.dtype

        # Enable CUDA optimizations BEFORE any model loading
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        # Will be initialized in setup()
        self.transformer = None
        self.lora_network = None
        self.vae = None
        self.text_encoder = None
        self.tokenizer = None
        self.optimizer = None
        self.lr_scheduler = None
        self.dataset = None
        self.dataloader = None

        # Training state
        self.global_step = 0
        self.current_epoch = 0

        # Callbacks
        self.callbacks: Dict[str, Callable] = {}

        print(f"KleinTrainer initialized for {config.model_variant} variant")

    def setup(self):
        """Initialize all components."""
        print("=" * 60)
        print("Setting up Klein trainer...")

        # Enable CUDA optimizations
        if torch.cuda.is_available():
            # Enable cuDNN autotuner to find optimal algorithms
            torch.backends.cudnn.benchmark = True
            # Enable TF32 for faster matmul on Ampere+ GPUs
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            print("  CUDA optimizations enabled (cudnn.benchmark, TF32)")
        print("=" * 60)

        # Create output directory with run name
        # Combine output_dir and output_name to create full path
        self.run_dir = Path(self.config.output_dir) / self.config.output_name
        os.makedirs(self.run_dir, exist_ok=True)

        # Load VAE
        self._setup_vae()

        # Load transformer
        self._setup_transformer()

        # Apply LoRA
        self._setup_lora()

        # Load text encoder
        self._setup_text_encoder()

        # Setup dataset
        self._setup_dataset()

        # Setup optimizer
        self._setup_optimizer()

        # Save config
        self._save_config()

        print("=" * 60)
        print("Setup complete!")
        print("=" * 60)

    def _setup_vae(self):
        """Load and setup VAE."""
        print("\n[1/6] Loading VAE...")

        vae_path = self.config.vae_path
        if not vae_path:
            raise ValueError("vae_path must be specified in config")

        self.vae = KleinVAE.from_pretrained(vae_path, dtype=self.dtype)
        self.vae = self.vae.to(self.device)
        self.vae.eval()
        self.vae.requires_grad_(False)

        print(f"  VAE loaded from {vae_path}")

    def _setup_transformer(self):
        """Load and setup transformer."""
        print("\n[2/6] Loading transformer...")

        self.transformer = load_klein_transformer(
            model_path=self.config.model_repo,
            variant=self.config.model_variant,
            device=self.device,
            dtype=self.dtype,
            quantize=self.config.quantize_transformer,
        )

    def _setup_lora(self):
        """Apply LoRA to transformer."""
        print("\n[3/6] Applying LoRA...")

        self.lora_network = apply_lora_to_transformer(
            transformer=self.transformer,
            rank=self.config.lora_rank,
            alpha=self.config.lora_alpha,
            dropout=self.config.lora_dropout,
        )

    def _setup_text_encoder(self):
        """Load and setup text encoder."""
        print("\n[4/6] Loading text encoder...")

        self.text_encoder, self.tokenizer = load_text_encoder(
            model_path=self.config.text_encoder_repo or self.config.model_repo,
            variant=self.config.model_variant,
            device=self.device,
            dtype=self.dtype,
            quantize=self.config.quantize_text_encoder,
        )

    def _setup_dataset(self):
        """Setup dataset and dataloader."""
        print("\n[5/6] Setting up dataset...")

        self.dataset = KleinDataset(
            dataset_path=self.config.dataset_path,
            caption_extension=self.config.caption_extension,
            resolution=self.config.resolution,
            enable_bucket=self.config.enable_bucket,
            min_bucket_resolution=self.config.min_bucket_resolution,
            max_bucket_resolution=self.config.max_bucket_resolution,
            bucket_resolution_step=self.config.bucket_resolution_step,
            default_caption=self.config.default_caption if hasattr(self.config, 'default_caption') else "",
        )

        # Cache latents if enabled
        if self.config.cache_latents:
            self.dataset.cache_latents(self.vae, self.device, self.dtype)
            # Offload VAE to CPU after caching to save VRAM
            self.vae.cpu()
            torch.cuda.empty_cache()
            print("  VAE offloaded to CPU after caching latents")

        # Cache text embeddings if enabled
        if self.config.cache_text_encoder_outputs:
            self.dataset.cache_text_embeddings(
                self.text_encoder, self.tokenizer, self.device, self.dtype
            )
            # Offload text encoder to CPU after caching to save VRAM
            self.text_encoder.cpu()
            torch.cuda.empty_cache()
            print("  Text encoder offloaded to CPU after caching embeddings")

        # Report memory after setup
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            reserved = torch.cuda.memory_reserved(0) / 1024**3
            print(f"  GPU memory: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")

        # Create dataloader
        self.dataloader = create_dataloader(
            dataset=self.dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,  # Windows compatibility
        )

        print(f"  Dataset: {len(self.dataset)} images")
        print(f"  Batches per epoch: {len(self.dataloader)}")

    def _setup_optimizer(self):
        """Setup optimizer and learning rate scheduler."""
        print("\n[6/6] Setting up optimizer...")

        # Get LoRA parameters
        params = list(self.lora_network.parameters())

        # Create optimizer
        if self.config.optimizer == "adamw8bit":
            try:
                import bitsandbytes as bnb
                self.optimizer = bnb.optim.AdamW8bit(
                    params,
                    lr=self.config.learning_rate,
                    betas=(0.9, 0.999),
                    weight_decay=0.01,
                )
                print("  Using AdamW8bit optimizer")
            except ImportError:
                print("  Warning: bitsandbytes not available, using AdamW")
                self.optimizer = AdamW(
                    params,
                    lr=self.config.learning_rate,
                    betas=(0.9, 0.999),
                    weight_decay=0.01,
                )
        elif self.config.optimizer == "prodigy":
            try:
                from prodigyopt import Prodigy
                self.optimizer = Prodigy(
                    params,
                    lr=1.0,  # Prodigy auto-adjusts
                    betas=(0.9, 0.999),
                    weight_decay=0.01,
                )
                print("  Using Prodigy optimizer")
            except ImportError:
                print("  Warning: prodigyopt not available, using AdamW")
                self.optimizer = AdamW(
                    params,
                    lr=self.config.learning_rate,
                    betas=(0.9, 0.999),
                    weight_decay=0.01,
                )
        else:
            self.optimizer = AdamW(
                params,
                lr=self.config.learning_rate,
                betas=(0.9, 0.999),
                weight_decay=0.01,
            )
            print("  Using AdamW optimizer")

        # Setup learning rate scheduler
        self._setup_lr_scheduler()

    def _setup_lr_scheduler(self):
        """Setup learning rate scheduler."""
        total_steps = self._calculate_total_steps()

        if self.config.lr_scheduler == "constant":
            from torch.optim.lr_scheduler import LambdaLR
            self.lr_scheduler = LambdaLR(self.optimizer, lambda _: 1.0)
        elif self.config.lr_scheduler == "cosine":
            from torch.optim.lr_scheduler import CosineAnnealingLR
            self.lr_scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=total_steps - self.config.warmup_steps,
            )
        elif self.config.lr_scheduler == "linear":
            from torch.optim.lr_scheduler import LinearLR
            self.lr_scheduler = LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=0.0,
                total_iters=total_steps - self.config.warmup_steps,
            )

        print(f"  LR scheduler: {self.config.lr_scheduler}")
        print(f"  Total steps: {total_steps}")

    def _calculate_total_steps(self) -> int:
        """Calculate total training steps."""
        if self.config.max_train_steps:
            return self.config.max_train_steps

        steps_per_epoch = len(self.dataloader) // self.config.gradient_accumulation_steps
        return steps_per_epoch * self.config.num_epochs

    def _save_config(self):
        """Save training config to output directory."""
        config_path = self.run_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)
        print(f"\nConfig saved to {config_path}")

    def train(self):
        """Main training loop."""
        print("\n" + "=" * 60)
        print("Starting training...")
        print("=" * 60)

        total_steps = self._calculate_total_steps()
        gradient_accumulation_steps = self.config.gradient_accumulation_steps

        # Enable gradient checkpointing BEFORE setting train mode
        if self.config.gradient_checkpointing:
            self.lora_network.enable_gradient_checkpointing()

        # Set LoRA to training mode (also sets transformer.train() for checkpointing)
        self.lora_network.train()

        # Progress bar
        progress_bar = tqdm(total=total_steps, desc="Training")
        progress_bar.update(self.global_step)

        # Generate initial sample at step 0 to verify base model works
        if self.global_step == 0 and self.config.sample_prompts:
            print("\n=== GENERATING BASELINE SAMPLE (STEP 0) ===")
            print("This tests if the base Klein model produces reasonable images")
            self._generate_samples()
            print("=== BASELINE SAMPLE COMPLETE ===\n")

        accumulated_loss = 0.0
        accumulation_counter = 0

        while self.global_step < total_steps:
            self.current_epoch += 1

            for batch in self.dataloader:
                if self.global_step >= total_steps:
                    break

                # Training step (returns loss and optional per-sample data)
                loss, per_sample_data = self._training_step(batch)
                accumulated_loss += loss
                accumulation_counter += 1

                # Store per-sample data for the current accumulation batch
                current_per_sample_data = per_sample_data

                # Gradient accumulation
                if accumulation_counter >= gradient_accumulation_steps:
                    # Clip gradients
                    torch.nn.utils.clip_grad_norm_(
                        self.lora_network.parameters(),
                        self.config.max_grad_norm,
                    )

                    # Optimizer step
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()

                    # Update tracking
                    self.global_step += 1
                    avg_loss = accumulated_loss / accumulation_counter
                    accumulated_loss = 0.0
                    accumulation_counter = 0

                    # Update progress bar
                    progress_bar.update(1)
                    progress_bar.set_postfix({"loss": f"{avg_loss:.4f}"})

                    # Logging
                    if self.global_step % self.config.log_every_n_steps == 0:
                        self._log_step(avg_loss)

                    # Sampling
                    if self.global_step % self.config.sample_every_n_steps == 0:
                        self._generate_samples()

                    # Saving
                    if self.global_step % self.config.save_every_n_steps == 0:
                        self._save_checkpoint()

                    # Callbacks (include per-sample data for analytics)
                    self._run_callbacks("step_end", {
                        "step": self.global_step,
                        "loss": avg_loss,
                        "per_sample_data": current_per_sample_data,
                    })

        progress_bar.close()

        # Final save
        self._save_checkpoint(final=True)

        print("\n" + "=" * 60)
        print("Training complete!")
        print("=" * 60)

    def _training_step(self, batch: Dict[str, Any]) -> Tuple[float, Optional[List[Tuple[str, float, float]]]]:
        """
        Perform a single training step with per-sample loss tracking.

        Flow matching objective: predict velocity (noise - latents).

        Returns:
            Tuple of (mean_loss, per_sample_data) where per_sample_data is
            a list of (path, loss, timestep) tuples if paths are available.
        """
        # Get latents (either cached or encode) - use non_blocking for async transfer
        if "latents" in batch:
            latents = batch["latents"].to(self.device, dtype=self.dtype, non_blocking=True)
        else:
            images = batch["images"].to(self.device, dtype=self.dtype, non_blocking=True)
            latents = encode_images(self.vae, images, self.device, self.dtype)

        # Get text embeddings (either cached or encode) - use non_blocking for async transfer
        if "text_embeds" in batch:
            text_embeds = batch["text_embeds"].to(self.device, dtype=self.dtype, non_blocking=True)
        else:
            text_embeds = encode_text(
                self.text_encoder,
                self.tokenizer,
                batch["captions"],
                self.device,
                self.dtype,
            )

        batch_size = latents.shape[0]

        # Sample noise
        noise = torch.randn_like(latents)

        # Sample timesteps uniformly [0, 1]
        timesteps = torch.rand(batch_size, device=self.device, dtype=self.dtype)

        # Create noisy latents using flow matching interpolation
        # x_t = (1 - t) * x_0 + t * noise
        timesteps_expanded = timesteps.view(-1, 1, 1, 1)
        noisy_latents = (1 - timesteps_expanded) * latents + timesteps_expanded * noise

        # Target is the velocity: noise - latents
        target = noise - latents

        # Pack latents and text for model input
        # latents shape: [B, C, H, W] -> packed: [B, H*W, C]
        packed_latents, img_ids = batched_prc_img(noisy_latents)
        packed_text, txt_ids = batched_prc_txt(text_embeds)

        # Note: Do NOT truncate text to match image sequence length
        # Klein/Flux handle different sequence lengths correctly in the transformer

        # Forward pass through transformer with LoRA
        with autocast(dtype=self.dtype):
            model_pred = self.lora_network(
                x=packed_latents,
                x_ids=img_ids,
                timesteps=timesteps,
                ctx=packed_text,
                ctx_ids=txt_ids,
                guidance=None,  # Klein doesn't use guidance
            )

            # Unpack prediction back to image shape using scatter_ids
            # This properly reconstructs spatial layout from packed sequence
            # model_pred shape: [B, seq_len, C] -> [B, C, 1, H, W] -> [B, C, H, W]
            model_pred = torch.cat(scatter_ids(model_pred, img_ids)).squeeze(2)

            # Per-sample MSE loss (for analytics)
            per_sample_losses = F.mse_loss(model_pred, target, reduction='none')
            per_sample_losses = per_sample_losses.mean(dim=(1, 2, 3))  # [B] - reduce spatial dims

            # Mean loss for backward pass
            loss = per_sample_losses.mean()

        # Backward pass
        loss.backward()

        # Build per-sample data if paths available
        per_sample_data = None
        paths = batch.get('paths')
        if paths is not None:
            per_sample_data = [
                (path, per_sample_losses[i].item(), timesteps[i].item())
                for i, path in enumerate(paths)
            ]

        return loss.item(), per_sample_data

    def _log_step(self, loss: float):
        """Log training step."""
        lr = self.optimizer.param_groups[0]["lr"]
        print(f"Step {self.global_step}: loss={loss:.4f}, lr={lr:.2e}")

        # Run logging callbacks
        self._run_callbacks("log", {
            "step": self.global_step,
            "loss": loss,
            "lr": lr,
        })

    def _generate_samples(self):
        """Generate sample images using current LoRA weights."""
        print(f"\nGenerating samples at step {self.global_step}...")

        # Move text encoder and VAE back to GPU for sampling
        if self.config.cache_text_encoder_outputs:
            self.text_encoder.to(self.device)
        if self.config.cache_latents:
            self.vae.to(self.device)

        # Set to eval mode (but keep LoRA active!)
        self.lora_network.eval()

        # Debug: Check if LoRA weights have changed from initialization
        lora_stats = []
        for name, layer in self.lora_network.lora_layers.items():
            b_weight = layer.lora_B.weight.data
            a_weight = layer.lora_A.weight.data
            b_norm = b_weight.abs().mean().item()
            a_norm = a_weight.abs().mean().item()
            lora_stats.append((name, b_norm, a_norm))

        # Print a few samples
        print(f"LoRA weight check (first 3 layers):")
        for name, b_norm, a_norm in lora_stats[:3]:
            print(f"  {name}: B_norm={b_norm:.6f}, A_norm={a_norm:.6f}")

        # If B weights are all near zero, LoRA isn't training
        if all(b_norm < 1e-6 for _, b_norm, _ in lora_stats):
            print("  WARNING: All LoRA B weights are near zero - LoRA may not be training!")

        sample_dir = self.run_dir / "samples"
        sample_dir.mkdir(exist_ok=True)

        for idx, prompt in enumerate(self.config.sample_prompts):
            # Determine seed for this sample
            if self.config.sample_seed < 0:
                # Random seed mode: generate a new random seed each time
                import random
                sample_seed = random.randint(0, 2147483647)
            else:
                # Fixed seed with variation: base seed + prompt index + step offset
                sample_seed = self.config.sample_seed + idx + (self.global_step * 100)

            image = self._sample_single(
                prompt=prompt,
                seed=sample_seed,
                num_steps=self.config.sample_steps,
                guidance_scale=self.config.sample_guidance_scale,
            )

            # Save image
            save_path = sample_dir / f"step_{self.global_step:06d}_sample_{idx}.png"
            image.save(save_path)
            print(f"  Saved: {save_path}")

        # Offload text encoder and VAE back to CPU to save VRAM
        if self.config.cache_text_encoder_outputs:
            self.text_encoder.cpu()
        if self.config.cache_latents:
            self.vae.cpu()
        torch.cuda.empty_cache()

        # Back to training mode
        self.lora_network.train()

    def _sample_single(
        self,
        prompt: str,
        seed: int = 42,
        num_steps: int = 28,
        guidance_scale: float = 3.5,
        width: int = 512,
        height: int = 512,
    ):
        """
        Generate a single sample image.

        Uses flow matching sampling with the trained LoRA.
        """
        from PIL import Image

        # Set seed
        generator = torch.Generator(device=self.device).manual_seed(seed)

        # Encode prompt
        with torch.no_grad():
            text_embeds = encode_text(
                self.text_encoder,
                self.tokenizer,
                [prompt],
                self.device,
                self.dtype,
            )

        # Calculate latent dimensions
        # Klein uses 32-channel VAE with 2x2 packing -> 128 packed channels
        # Latent size is image_size / 16 (8x VAE + 2x packing)
        latent_h = height // 16
        latent_w = width // 16
        latent_channels = 128  # Packed channels

        # Start from noise
        latents = torch.randn(
            1, latent_channels, latent_h, latent_w,
            device=self.device,
            dtype=self.dtype,
            generator=generator,
        )

        # Pack latents and text
        packed_latents, img_ids = batched_prc_img(latents)
        packed_text, txt_ids = batched_prc_txt(text_embeds)

        # Get schedule based on image sequence length
        img_seq_len = packed_latents.shape[1]
        timesteps = get_schedule(num_steps, img_seq_len)

        # Denoise
        with torch.no_grad():
            packed_latents = denoise(
                model=self.lora_network,
                img=packed_latents,
                img_ids=img_ids,
                txt=packed_text,
                txt_ids=txt_ids,
                timesteps=timesteps,
                guidance=guidance_scale,
            )

        # Unpack latents using scatter_ids (same as ai-toolkit pipeline)
        latents = torch.cat(scatter_ids(packed_latents, img_ids)).squeeze(2)

        # Decode latents to image
        with torch.no_grad():
            image = decode_latents(self.vae, latents.to(self.dtype), self.device, self.dtype)

        # Convert to PIL
        image = (image + 1) / 2  # [-1, 1] -> [0, 1]
        image = image.clamp(0, 1)

        print(f"After normalization to [0,1]:")
        print(f"  Image stats: min={image.min().item():.4f}, max={image.max().item():.4f}, mean={image.mean().item():.4f}")

        image = image[0].permute(1, 2, 0).float().cpu().numpy()  # Convert to float32 for numpy
        image = (image * 255).astype("uint8")
        image = Image.fromarray(image)

        return image

    def _save_checkpoint(self, final: bool = False):
        """Save LoRA checkpoint."""
        if final:
            filename = f"{self.config.output_name}_final.safetensors"
        else:
            filename = f"{self.config.output_name}_step_{self.global_step:06d}.safetensors"

        save_path = self.run_dir / filename

        # Convert to save dtype
        save_dtype = torch.bfloat16 if self.config.save_precision == "bf16" else torch.float16

        self.lora_network.save_weights(
            str(save_path),
            dtype=save_dtype,
            model_variant=self.config.model_variant
        )
        print(f"Checkpoint saved: {save_path}")

        # Run save callbacks
        self._run_callbacks("save", {
            "step": self.global_step,
            "path": str(save_path),
        })

    def register_callback(self, event: str, callback: Callable):
        """Register a callback for training events."""
        if event not in self.callbacks:
            self.callbacks[event] = []
        self.callbacks[event].append(callback)

    def _run_callbacks(self, event: str, data: Dict[str, Any]):
        """Run callbacks for an event."""
        if event in self.callbacks:
            for callback in self.callbacks[event]:
                try:
                    callback(data)
                except Exception as e:
                    print(f"Warning: Callback error: {e}")


def train_klein_lora(config: KleinConfig):
    """
    Main entry point for Klein LoRA training.

    Args:
        config: Training configuration

    Returns:
        Trained KleinTrainer instance
    """
    trainer = KleinTrainer(config)
    trainer.setup()
    trainer.train()
    return trainer
