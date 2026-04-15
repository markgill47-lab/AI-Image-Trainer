"""
Klein model loading and architecture utilities.
Supports both 4B and 9B variants using the Flux2Klein architecture.
"""

import os
import sys
import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Tuple
from pathlib import Path
import huggingface_hub
from safetensors.torch import load_file

# Add ai-toolkit to path for Flux2Klein imports
AI_TOOLKIT_PATH = Path(__file__).parent.parent / "ai-toolkit"
if str(AI_TOOLKIT_PATH) not in sys.path:
    sys.path.insert(0, str(AI_TOOLKIT_PATH))

# Import Klein model architecture from ai-toolkit
from extensions_built_in.diffusion_models.flux2.src.model_klein import (
    Flux2Klein,
    Flux2KleinParams,
    Flux2Klein4BParams,
)

# Klein architecture specs (for reference)
KLEIN_CONFIGS = {
    "4B": {
        "hidden_size": 3072,
        "num_heads": 24,
        "num_double_blocks": 5,
        "num_single_blocks": 20,
        "context_in_dim": 7680,
        "in_channels": 128,  # 32 VAE channels * 4 packing
    },
    "9B": {
        "hidden_size": 4096,
        "num_heads": 32,
        "num_double_blocks": 8,
        "num_single_blocks": 24,
        "context_in_dim": 12288,
        "in_channels": 128,
    },
}

# Klein model filenames
KLEIN_FILENAMES = {
    "4B": "flux-2-klein-base-4b.safetensors",
    "9B": "flux-2-klein-base-9b.safetensors",
}


def get_klein_config(variant: str) -> Dict[str, Any]:
    """Get configuration for Klein variant."""
    if variant not in KLEIN_CONFIGS:
        raise ValueError(f"Unknown Klein variant: {variant}. Must be '4B' or '9B'")
    return KLEIN_CONFIGS[variant].copy()


def load_klein_transformer(
    model_path: str,
    variant: str = "4B",
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.bfloat16,
    quantize: bool = True,
) -> Flux2Klein:
    """
    Load Klein transformer model.

    Args:
        model_path: Path to model or HuggingFace repo ID
        variant: "4B" or "9B"
        device: Device to load model to
        dtype: Data type for model weights
        quantize: Whether to quantize to int8

    Returns:
        Loaded Flux2Klein model
    """
    print(f"Loading Klein {variant} transformer...")

    # Get model params for variant
    if variant == "4B":
        params = Flux2Klein4BParams()
    else:
        params = Flux2KleinParams()

    # Determine model file path
    klein_filename = KLEIN_FILENAMES[variant]

    # Check if local path or HuggingFace repo
    if model_path and os.path.isdir(model_path):
        # Local directory - look for the safetensors file
        local_file = os.path.join(model_path, klein_filename)
        if os.path.exists(local_file):
            transformer_path = local_file
        else:
            raise FileNotFoundError(f"Could not find {klein_filename} in {model_path}")
    elif model_path and os.path.isfile(model_path):
        # Direct path to safetensors file
        transformer_path = model_path
    else:
        # Download from HuggingFace (model_path is None or a repo ID)
        if not model_path or not model_path.startswith("black-forest-labs"):
            model_path = f"black-forest-labs/FLUX.2-klein-base-{variant}"

        print(f"Downloading from HuggingFace: {model_path}/{klein_filename}")
        transformer_path = huggingface_hub.hf_hub_download(
            repo_id=model_path,
            filename=klein_filename,
        )

    print(f"Loading weights from: {transformer_path}")

    # Load state dict
    state_dict = load_file(transformer_path, device="cpu")

    # Cast to target dtype
    for key in state_dict:
        state_dict[key] = state_dict[key].to(dtype)

    # Create model on meta device first (saves memory)
    with torch.device("meta"):
        transformer = Flux2Klein(params)

    # Materialize on CPU
    transformer = transformer.to_empty(device="cpu")

    # Load weights
    transformer.load_state_dict(state_dict, assign=True)

    # Apply quantization if requested
    if quantize:
        transformer = quantize_model(transformer)

    # Move to target device
    transformer = transformer.to(device)
    transformer.requires_grad_(False)
    transformer.eval()

    config = KLEIN_CONFIGS[variant]
    print(f"Klein {variant} transformer loaded successfully")
    print(f"  - Double blocks: {config['num_double_blocks']}")
    print(f"  - Single blocks: {config['num_single_blocks']}")
    print(f"  - Hidden size: {config['hidden_size']}")

    return transformer


def quantize_model(model: nn.Module) -> nn.Module:
    """
    Quantize model to int8 using optimum.quanto.

    Args:
        model: Model to quantize

    Returns:
        Quantized model
    """
    try:
        from optimum.quanto import quantize, freeze, qint8

        print("Quantizing transformer to int8...")
        quantize(model, weights=qint8)
        freeze(model)
        print("Quantization complete")
        return model

    except ImportError:
        print("Warning: optimum.quanto not available, skipping quantization")
        return model


def load_text_encoder(
    model_path: str,
    variant: str = "4B",
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.bfloat16,
    quantize: bool = True,
) -> Tuple[nn.Module, Any]:
    """
    Load Qwen3 text encoder for Klein models.

    Args:
        model_path: Path to model or HuggingFace repo ID (not used, auto-downloads from Klein repo)
        variant: "4B" or "9B" - determines which Klein repo to use
        device: Device to load model to
        dtype: Data type for model weights
        quantize: Whether to quantize to int8

    Returns:
        Tuple of (text_encoder, tokenizer)
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Klein uses its own Qwen3 text encoder from the Klein repo
    # Allow custom HuggingFace repo override (e.g., for mirrors or alternative versions)
    if model_path and "/" in model_path and not os.path.exists(model_path):
        klein_repo = model_path  # Treat as HuggingFace repo ID
    else:
        klein_repo = f"black-forest-labs/FLUX.2-klein-base-{variant}"

    print(f"Loading Qwen3 text encoder from {klein_repo}...")

    tokenizer = AutoTokenizer.from_pretrained(
        klein_repo,
        subfolder="tokenizer",
    )

    text_encoder = AutoModelForCausalLM.from_pretrained(
        klein_repo,
        subfolder="text_encoder",
        torch_dtype=dtype,
        device_map="cpu",  # Load to CPU first
    )

    if quantize:
        try:
            from optimum.quanto import quantize as quanto_quantize, freeze, qint8
            print("Quantizing text encoder to int8...")
            quanto_quantize(text_encoder, weights=qint8)
            freeze(text_encoder)
        except ImportError:
            print("Warning: optimum.quanto not available, skipping text encoder quantization")

    text_encoder = text_encoder.to(device)
    text_encoder.requires_grad_(False)
    text_encoder.eval()

    print("Text encoder loaded successfully")

    return text_encoder, tokenizer


def encode_text(
    text_encoder: nn.Module,
    tokenizer: Any,
    prompts: list,
    device: torch.device,
    dtype: torch.dtype,
    max_length: int = 512,
) -> torch.Tensor:
    """
    Encode text prompts to embeddings.

    Klein uses Qwen3 with layers [9, 18, 27] stacked together.
    Text must be formatted using apply_chat_template for correct encoding.

    Args:
        text_encoder: Qwen3 text encoder
        tokenizer: Qwen3 tokenizer
        prompts: List of text prompts
        device: Device for computation
        dtype: Data type for computation
        max_length: Maximum sequence length

    Returns:
        Text embeddings of shape [batch, seq_len, hidden_size * 3]
    """
    from einops import rearrange

    # Format prompts using chat template (required for correct Qwen3 encoding)
    all_input_ids = []
    all_attention_masks = []

    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        model_inputs = tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

        all_input_ids.append(model_inputs["input_ids"])
        all_attention_masks.append(model_inputs["attention_mask"])

    # Move to encoder device
    encoder_device = next(text_encoder.parameters()).device
    input_ids = torch.cat(all_input_ids, dim=0).to(encoder_device)
    attention_mask = torch.cat(all_attention_masks, dim=0).to(encoder_device)

    # Get embeddings
    with torch.no_grad():
        outputs = text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )

        # BFL uses layers [9, 18, 27] for Qwen3 (NOT [10, 20, 30]!)
        OUTPUT_LAYERS_QWEN3 = [9, 18, 27]
        out = torch.stack([outputs.hidden_states[k] for k in OUTPUT_LAYERS_QWEN3], dim=1)
        embeds = rearrange(out, "b c l d -> b l (c d)")

    return embeds.to(device).to(dtype)


def get_trainable_params(model: nn.Module) -> int:
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_total_params(model: nn.Module) -> int:
    """Count total parameters in model."""
    return sum(p.numel() for p in model.parameters())
