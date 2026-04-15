# Klein Trainer Module

A minimal, clean implementation for training LoRA models on FLUX.2 Klein (4B and 9B variants).

## Overview

This module provides:
- Flow matching training objective
- LoRA injection into transformer attention and MLP layers
- Latent and text embedding caching for memory efficiency
- Gradient checkpointing for reduced VRAM usage
- ComfyUI-compatible LoRA export

## Module Structure

```
klein_trainer/
├── __init__.py      # Public API exports
├── config.py        # KleinConfig dataclass
├── dataset.py       # Dataset loading, bucketing, caching
├── model.py         # Klein transformer and text encoder loading
├── vae.py           # VAE encode/decode operations
├── lora.py          # LoRA layer implementation
├── trainer.py       # Main training loop
└── manager.py       # GUI integration and process management
```

## Quick Start

### Programmatic Usage

```python
from klein_trainer import KleinConfig, KleinTrainer

config = KleinConfig(
    model_variant="4B",
    dataset_path="./datasets/my_style",
    output_dir="./output",
    output_name="my_lora",
    learning_rate=1e-4,
    max_train_steps=1000,
    lora_rank=32,
)

trainer = KleinTrainer(config)
trainer.setup()
trainer.train()
```

### With GUI

The `KleinTrainerManager` class integrates with the PyQt6 GUI:

```python
from klein_trainer import KleinTrainerManager, get_trainer_manager

manager = get_trainer_manager()
manager.start_training(config, callbacks={
    'on_step': update_progress,
    'on_sample': display_sample,
    'on_complete': training_finished,
})
```

## Configuration

### KleinConfig Parameters

```python
@dataclass
class KleinConfig:
    # Model
    model_variant: str = "4B"           # "4B" or "9B"
    model_repo: str = None              # HuggingFace repo or local path
    vae_path: str = None                # Path to Klein VAE

    # Dataset
    dataset_path: str = ""              # Path to training images
    caption_extension: str = ".txt"     # Caption file extension
    resolution: int = 512               # Training resolution
    enable_bucket: bool = True          # Aspect ratio bucketing

    # Training
    learning_rate: float = 1e-4
    max_train_steps: int = 1000
    batch_size: int = 1
    gradient_accumulation_steps: int = 1

    # LoRA
    lora_rank: int = 32
    lora_alpha: int = 32
    lora_dropout: float = 0.0

    # Optimization
    cache_latents: bool = True
    cache_text_encoder_outputs: bool = True
    gradient_checkpointing: bool = True
    quantize_transformer: bool = True
    quantize_text_encoder: bool = True

    # Output
    output_dir: str = "./output"
    output_name: str = "klein_lora"
    save_every_n_steps: int = 500
    sample_every_n_steps: int = 100
    sample_prompts: list = None
```

## Architecture Details

### Klein Model Variants

| Variant | Hidden Size | Double Blocks | Single Blocks | Context Dim | VRAM (quantized) |
|---------|-------------|---------------|---------------|-------------|------------------|
| 4B | 3072 | 5 | 20 | 7680 | ~5-6 GB |
| 9B | 4096 | 8 | 24 | 12288 | ~8-9 GB |

### LoRA Target Modules

LoRA is applied to these layers in each transformer block:

**Double Blocks:**
- `img_attn.qkv` - Image attention Q/K/V projection
- `img_attn.proj` - Image attention output projection
- `txt_attn.qkv` - Text attention Q/K/V projection
- `txt_attn.proj` - Text attention output projection
- `img_mlp.0`, `img_mlp.2` - Image MLP layers
- `txt_mlp.0`, `txt_mlp.2` - Text MLP layers

**Single Blocks:**
- `linear1` - First linear layer
- `linear2` - Second linear layer

### Text Encoder

Klein uses Qwen3 as text encoder with a specific configuration:
- **Layers used**: [9, 18, 27] (stacked together)
- **Chat template**: Required for correct tokenization
- **Hidden size**: 2560 (4B) or 4096 (9B)
- **Context dimension**: hidden_size * 3 (stacked layers)

### Flow Matching Training

The training objective is flow matching (not standard diffusion):

```python
# Interpolate between clean and noise
x_t = (1 - t) * x_0 + t * noise

# Target is velocity (noise - clean)
target = noise - x_0

# Loss is MSE between prediction and target
loss = MSE(model_pred, target)
```

## Memory Optimization

### Caching

**Latent Caching** (`cache_latents=True`):
- Pre-encodes all images through VAE at startup
- VAE offloaded to CPU after caching
- Eliminates VAE forward pass during training

**Text Embedding Caching** (`cache_text_encoder_outputs=True`):
- Pre-encodes all captions through Qwen3
- Text encoder offloaded to CPU after caching
- Eliminates text encoder forward pass during training

### Quantization

Using `optimum-quanto` for INT8 quantization:
- Transformer: ~50% VRAM reduction
- Text encoder: ~50% VRAM reduction
- Minimal quality impact for training

### Gradient Checkpointing

Recomputes activations during backward pass instead of storing them:
- Significant VRAM reduction
- ~20% training speed overhead
- Essential for fitting 9B on 24GB VRAM

## ComfyUI Compatibility

LoRA files are saved with ComfyUI-compatible key format:

```
diffusion_model.double_blocks.0.img_attn.qkv.lora_A.weight
diffusion_model.double_blocks.0.img_attn.qkv.lora_B.weight
diffusion_model.single_blocks.0.linear1.lora_A.weight
...
```

### Metadata

Saved in safetensors metadata:
- `ss_network_dim`: LoRA rank
- `ss_network_alpha`: LoRA alpha
- `ss_base_model_version`: `flux2_klein_4b` or `flux2_klein_9b`

## Dataset Format

### Folder Structure

```
dataset/
├── image_001.jpg
├── image_001.txt    # Caption for image_001
├── image_002.png
├── image_002.txt
└── ...
```

### Caption Format

Plain text files with image descriptions:

```
a portrait of a woman in impressionist style, soft brushstrokes, warm colors
```

### Aspect Ratio Bucketing

When `enable_bucket=True`:
- Images grouped by similar aspect ratios
- Each bucket resized to maintain ~same pixel count
- Prevents distortion from forced square cropping

## Sample Generation

During training, samples are generated to monitor progress:

```python
config.sample_prompts = [
    "a portrait in my_style, detailed face",
    "a landscape in my_style, mountains",
]
config.sample_every_n_steps = 100
config.sample_steps = 8  # Faster preview (use 28 for quality)
```

## API Reference

### KleinTrainer

```python
class KleinTrainer:
    def __init__(self, config: KleinConfig)
    def setup(self)           # Initialize models, dataset, optimizer
    def train(self)           # Run training loop
    def register_callback(event: str, callback: Callable)
```

**Events:**
- `"step_end"`: Called after each training step
- `"log"`: Called when logging metrics
- `"save"`: Called when saving checkpoint

### LoRANetwork

```python
class LoRANetwork:
    def __init__(self, transformer, rank, alpha, dropout, target_modules)
    def train(mode=True)      # Set training mode
    def eval()                # Set evaluation mode
    def merge_weights()       # Merge LoRA into base model
    def unmerge_weights()     # Remove LoRA from base model
    def save_weights(path, dtype, model_variant)
    def load_weights(transformer, path, device)  # classmethod
```

### KleinDataset

```python
class KleinDataset:
    def __init__(self, dataset_path, caption_extension, resolution, ...)
    def cache_latents(vae, device, dtype)
    def cache_text_embeddings(text_encoder, tokenizer, device, dtype)
```

## Troubleshooting

### "LoRA key not loaded" in ComfyUI
- Ensure using Klein model, not regular Flux
- Check ComfyUI Klein support is installed
- Verify key format matches expected pattern

### Out of Memory
- Use 4B instead of 9B
- Reduce LoRA rank to 16 or 8
- Ensure caching is enabled
- Check gradient checkpointing is active

### Slow Training
- Verify CUDA is being used
- Check caching is working (no VAE/text encoder forward passes)
- Ensure gradient checkpointing is enabled (requires `transformer.train()`)

### Poor Quality Results
- Increase training steps
- Adjust learning rate (try 5e-5 to 2e-4)
- Check caption quality
- Verify sample prompts include trigger words
