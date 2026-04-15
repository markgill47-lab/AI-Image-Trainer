#!/usr/bin/env python
"""
Klein LoRA Trainer - Command Line Interface

Usage:
    python train_klein.py --config config.yaml
    python train_klein.py --dataset ./my_images --output ./output

For full options, run:
    python train_klein.py --help
"""

import argparse
import sys
from pathlib import Path

import yaml


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train LoRA for FLUX.2 Klein models (4B and 9B)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Train with config file
    python train_klein.py --config configs/my_training.yaml

    # Quick training with defaults
    python train_klein.py --dataset ./images --vae ./models/ae.safetensors

    # Custom settings
    python train_klein.py \\
        --dataset ./images \\
        --vae ./models/ae.safetensors \\
        --variant 9B \\
        --rank 64 \\
        --steps 1000 \\
        --lr 2e-4
        """,
    )

    # Config file (overrides all other args if provided)
    parser.add_argument(
        "--config",
        type=str,
        help="Path to YAML config file",
    )

    # Model settings
    parser.add_argument(
        "--variant",
        type=str,
        default="4B",
        choices=["4B", "9B"],
        help="Klein model variant (default: 4B)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        help="Local path or HuggingFace repo for Klein model",
    )
    parser.add_argument(
        "--vae",
        type=str,
        required=False,
        help="Path to VAE weights (ae.safetensors in BFL format)",
    )

    # LoRA settings
    parser.add_argument(
        "--rank",
        type=int,
        default=32,
        help="LoRA rank (default: 32)",
    )
    parser.add_argument(
        "--alpha",
        type=int,
        default=32,
        help="LoRA alpha (default: 32)",
    )

    # Dataset settings
    parser.add_argument(
        "--dataset",
        type=str,
        required=False,
        help="Path to dataset directory containing images and captions",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="Training resolution (default: 512)",
    )
    parser.add_argument(
        "--no-bucket",
        action="store_true",
        help="Disable resolution bucketing",
    )

    # Training settings
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size (default: 1)",
    )
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=1,
        help="Gradient accumulation steps (default: 1)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of epochs (default: 1)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        help="Max training steps (overrides epochs)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate (default: 1e-4)",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adamw8bit",
        choices=["adamw", "adamw8bit", "prodigy"],
        help="Optimizer (default: adamw8bit)",
    )

    # Memory settings
    parser.add_argument(
        "--no-cache-latents",
        action="store_true",
        help="Disable latent caching (uses more compute, less VRAM)",
    )
    parser.add_argument(
        "--no-quantize",
        action="store_true",
        help="Disable model quantization (uses more VRAM)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="bf16",
        choices=["fp32", "fp16", "bf16"],
        help="Mixed precision mode (default: bf16)",
    )

    # Output settings
    parser.add_argument(
        "--output",
        type=str,
        default="./output",
        help="Output directory (default: ./output)",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="klein_lora",
        help="Output name prefix (default: klein_lora)",
    )

    # Sampling settings
    parser.add_argument(
        "--sample-every",
        type=int,
        default=100,
        help="Generate samples every N steps (default: 100)",
    )
    parser.add_argument(
        "--sample-prompt",
        type=str,
        action="append",
        help="Prompt for sample generation (can specify multiple)",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=500,
        help="Save checkpoint every N steps (default: 500)",
    )

    return parser.parse_args()


def load_config_from_yaml(path: str) -> dict:
    """Load configuration from YAML file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def args_to_config_dict(args) -> dict:
    """Convert argparse args to config dictionary."""
    config = {
        "model_variant": args.variant,
        "lora_rank": args.rank,
        "lora_alpha": args.alpha,
        "resolution": args.resolution,
        "enable_bucket": not args.no_bucket,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "num_epochs": args.epochs,
        "learning_rate": args.lr,
        "optimizer": args.optimizer,
        "mixed_precision": args.precision,
        "cache_latents": not args.no_cache_latents,
        "cache_text_encoder_outputs": not args.no_cache_latents,
        "quantize_transformer": not args.no_quantize,
        "quantize_text_encoder": not args.no_quantize,
        "output_dir": args.output,
        "output_name": args.name,
        "sample_every_n_steps": args.sample_every,
        "save_every_n_steps": args.save_every,
    }

    if args.model_path:
        config["model_path"] = args.model_path

    if args.vae:
        config["vae_path"] = args.vae

    if args.dataset:
        config["dataset_path"] = args.dataset

    if args.steps:
        config["max_train_steps"] = args.steps

    if args.sample_prompt:
        config["sample_prompts"] = args.sample_prompt

    return config


def main():
    args = parse_args()

    # Load config
    if args.config:
        print(f"Loading config from {args.config}")
        config_dict = load_config_from_yaml(args.config)
    else:
        config_dict = args_to_config_dict(args)

    # Validate required fields
    if "dataset_path" not in config_dict or not config_dict["dataset_path"]:
        print("Error: --dataset is required (or dataset_path in config)")
        sys.exit(1)

    if "vae_path" not in config_dict or not config_dict["vae_path"]:
        print("Error: --vae is required (or vae_path in config)")
        sys.exit(1)

    # Import trainer (delayed to show help faster)
    from klein_trainer import KleinConfig, train_klein_lora

    # Create config
    config = KleinConfig.from_dict(config_dict)

    print("\n" + "=" * 60)
    print("Klein LoRA Trainer")
    print("=" * 60)
    print(f"Model variant: {config.model_variant}")
    print(f"LoRA rank: {config.lora_rank}, alpha: {config.lora_alpha}")
    print(f"Dataset: {config.dataset_path}")
    print(f"Resolution: {config.resolution}")
    print(f"Batch size: {config.batch_size}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Output: {config.output_dir}")
    print("=" * 60 + "\n")

    # Train
    train_klein_lora(config)


if __name__ == "__main__":
    main()
