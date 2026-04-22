# Klein LoRA Trainer

A PyQt6-based GUI application for training LoRA (Low-Rank Adaptation) models on FLUX.2 Klein image generation models. Supports both Klein 4B and 9B variants with optimized training for consumer GPUs.

## Features

- **Klein Model Support**: Train LoRAs for FLUX.2 Klein 4B and 9B models
- **PyQt6 GUI**: Clean tabbed interface for configuration, dataset management, and training
- **Memory Optimized**: INT8 quantization, gradient checkpointing, and latent caching for 16GB GPUs
- **ComfyUI Compatible**: Exports LoRA files in ComfyUI-compatible format
- **Integrated Dataset Manager**: Built-in tools for image resizing, renaming, and caption management
- **Real-time Monitoring**: Loss tracking, sample generation, and progress visualization

## Requirements

### Hardware
- NVIDIA GPU with 16GB+ VRAM
- Klein 4B: 16GB VRAM sufficient (RTX 3090, 4070, etc.)
- Klein 9B: 24GB VRAM (RTX 4090, RTX PRO 4000 Blackwell, etc.)
- **Ampere / Ada Lovelace / Blackwell** architectures. Blackwell (sm_120) requires PyTorch 2.6+ with CUDA 12.6+.

### Software
- Python 3.10+
- CUDA 12.1+ (12.6+ for Blackwell GPUs)
- Linux (Ubuntu 22.04+) or Windows 10/11

### Python Dependencies
Full list in `requirements.txt`. Key packages:
- `torch`, `torchvision`, `torchaudio` (matched CUDA versions)
- `PyQt6` — GUI
- `transformers`, `safetensors`, `huggingface-hub` — model loading
- `optimum-quanto` — INT8 quantization (required for 9B to fit in 24GB)
- `einops`, `pillow`, `opencv-python` — data
- `anthropic` — optional, for Claude-assisted captioning

## Installation

### Linux (Olympus lab machines)

One-shot install:
```bash
curl -fsSL https://raw.githubusercontent.com/markgill47-lab/AI-Image-Trainer/master/install_olympus.sh | bash
```

This clones the repo, installs PyTorch + CUDA 12.6, clones ai-toolkit, applies the Klein patches, and creates a launcher at `~/AI-Image-Trainer/start_gui.sh`.

For Blackwell GPUs (RTX PRO 4000 Blackwell, etc.) you may need to upgrade torch after install:
```bash
cd ~/AI-Image-Trainer && source .venv/bin/activate
pip install --force-reinstall torch torchvision torchaudio
```

### Windows (development)

1. **Clone the repository**
   ```cmd
   git clone https://github.com/markgill47-lab/AI-Image-Trainer.git
   cd AI-Image-Trainer
   ```

2. **Create virtual environment**
   ```cmd
   python -m venv .venv
   .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```cmd
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   pip install -r requirements.txt
   ```

4. **Clone ai-toolkit and apply patches**
   ```cmd
   git clone https://github.com/ostris/ai-toolkit.git
   xcopy /E /Y ai-toolkit-patches\* ai-toolkit\
   ```

5. **Run the application**
   ```cmd
   python main.py
   ```

## Project Structure

```
AI_Image_Trainer/
├── main.py                    # Application entry point
├── ui_manager.py              # Main GUI window
├── klein_gui_manager.py       # Klein training GUI integration
├── gui_config_manager.py      # GUI configuration persistence
├── enhanced_training_manager.py  # Backend router (Klein/kohya/aitoolkit)
│
├── klein_trainer/             # Klein LoRA training module
│   ├── config.py              # KleinConfig dataclass
│   ├── dataset.py             # Dataset loading, bucketing, caching
│   ├── model.py               # Klein transformer + text encoder loading
│   ├── vae.py                 # 32-channel BFL VAE (auto-converts diffusers format)
│   ├── lora.py                # LoRA injection + ComfyUI-compatible save
│   ├── trainer.py             # Flow matching training loop
│   ├── manager.py             # Threaded training orchestration
│   └── analytics.py           # Per-sample loss tracking, outlier detection
│
├── dataset_Manager/           # GUI dataset tools (note capital M)
│   ├── embedded_dataset_manager.py
│   ├── image_processor.py     # Resize, rename, caption utilities
│   └── ...
│
├── ai-toolkit-patches/        # Local patches applied to ai-toolkit base
│   └── extensions_built_in/diffusion_models/flux2/...  # Klein model arch
│
├── setup_olympus.sh           # Linux setup (venv, deps, ai-toolkit clone)
├── install_olympus.sh         # One-shot remote installer (curl | bash)
├── start_gui.sh / start_gui.bat  # Launchers
├── ai-toolkit/                # (gitignored) cloned by setup, patches applied
├── datasets/                  # (gitignored) training data
└── output/                    # (gitignored) LoRAs, samples, analytics
```

## Usage

### Quick Start

1. **Prepare your dataset**
   - Folder with training images (`.jpg`, `.png`, `.jpeg`)
   - Matching caption files (`.txt`) with the same base filename
   - Example: `image001.jpg` + `image001.txt`
   - Use the built-in Dataset Manager tab to resize, rename, and auto-caption

2. **Launch the GUI**
   ```bash
   ./start_gui.sh    # Linux
   start_gui.bat     # Windows
   ```

3. **Configure training**
   - Select Klein 4B or 9B model
   - Set ComfyUI Models Base (auto-detects transformer + VAE from your existing ComfyUI install) OR manually browse to model files
   - Set dataset path
   - Adjust epochs, samples-per-image, LoRA rank
   - Add sample prompts for preview generation

4. **Start training**
   - Click "Start Training"
   - Monitor loss, EMA trends, and sample images in the Performance tab
   - LoRA and checkpoints saved to `output/<output_name>/`

### Using Existing ComfyUI Models

The trainer can reuse model files from a ComfyUI installation — no need to re-download:

- **Transformer** (`flux-2-klein-base-9b.safetensors`) from `ComfyUI/models/unet/` or `ComfyUI/models/diffusion_models/`
- **VAE** (`flux2-vae.safetensors`, in diffusers format) from `ComfyUI/models/vae/` — auto-converted to BFL format on load
- **Text encoder** downloads from HuggingFace on first run (~8-10GB, cached in `~/.cache/huggingface/`)

Set the ComfyUI base path in the Config tab and click "Auto-detect from ComfyUI".

**Note:** If ComfyUI is running on the same machine, stop it before training — it holds VRAM that Klein 9B training needs.

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| Model | Klein 4B | 4B fits 16GB VRAM, 9B needs 24GB+ |
| Learning Rate | 1e-4 | Lower for fine details, higher for styles |
| LoRA Rank | 32 | Higher = more capacity, more VRAM |
| Training Steps | 1000 | More steps for larger datasets |
| Batch Size | 1 | Increase if VRAM allows |
| Resolution | 512 | Training image resolution |

### Dataset Preparation

The integrated Dataset Manager provides:

- **Fix Images**: Resize and standardize dimensions
- **Mass Rename**: Batch rename with sequential numbering
- **Caption Editor**: Edit image descriptions

Recommended dataset structure:
```
datasets/my_style/
├── image_001.jpg
├── image_001.txt    # "a portrait in my_style, detailed..."
├── image_002.jpg
├── image_002.txt
└── ...
```

## Using Trained LoRAs in ComfyUI

1. Copy the `.safetensors` file from `output/` to ComfyUI's `models/loras/` folder

2. In ComfyUI:
   - Load the Klein model (FLUX.2 Klein 4B or 9B)
   - Add a LoRA Loader node
   - Select your trained LoRA
   - Set strength (0.7-1.0 typical)

3. Use your trigger words in the prompt

## Troubleshooting

### Out of Memory (OOM)
- Stop any running ComfyUI / other GPU processes before training (`nvidia-smi` to check)
- Make sure `optimum-quanto` is installed (`pip install optimum-quanto`) — without it, the 9B model won't fit in 24GB
- Use Klein 4B instead of 9B if you only have 16GB
- Reduce LoRA rank (16 or 8)
- Reduce resolution to 512

### Training Too Slow
- Ensure CUDA is available: `python -c "import torch; print(torch.cuda.is_available())"`
- Check latent caching is enabled in config
- On Blackwell GPUs, make sure you have torch 2.6+ with CUDA 12.6+ (cu121 builds don't support sm_120)

### GUI Won't Start
- Activate the venv first: `source .venv/bin/activate` (Linux) or `.venv\Scripts\activate` (Windows)
- On Linux, install PyQt6 system deps: `sudo apt install libxcb-xinerama0 libxcb-cursor0 libgl1 libegl1`
- Run `python main.py` directly to see the full traceback

### "Could not import module 'T5EncoderModel'"
- Usually means torch/torchvision versions are mismatched. Reinstall matched versions:
  `pip install --force-reinstall torch torchvision torchaudio`

### Outlier filenames don't match current dataset
- Previous runs with the same output name left stale analytics. Either change the output name, or delete `output/<name>/per_sample_losses.jsonl`. The trainer now auto-clears this when it detects a fresh run.

### LoRA Not Loading in ComfyUI
- Klein LoRAs only work with Klein models, not regular FLUX.1
- Copy the `.safetensors` from `output/<name>/` to `ComfyUI/models/loras/`

## Architecture

### Klein Models
Klein is a smaller, faster variant of FLUX:
- **Klein 4B**: 3072 hidden size, 5 double blocks, 20 single blocks
- **Klein 9B**: 4096 hidden size, 8 double blocks, 24 single blocks

### LoRA Training
Uses flow matching objective with:
- Target modules: attention (qkv, proj) and MLP layers
- Latent caching for memory efficiency
- Text embedding caching with Qwen3 encoder
- Gradient checkpointing for reduced VRAM

### Key Technologies
- **optimum-quanto**: INT8 quantization for transformer and text encoder
- **PyQt6**: Cross-platform GUI framework
- **safetensors**: Safe model weight storage

## License

MIT License

## Acknowledgments

- [Black Forest Labs](https://blackforestlabs.ai/) for FLUX.2 Klein models
- [ai-toolkit](https://github.com/ostris/ai-toolkit) for Klein architecture implementation
- [Kohya-ss](https://github.com/kohya-ss/sd-scripts) for training techniques inspiration
