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
- NVIDIA GPU with 16GB+ VRAM (RTX 4090, RTX 3090, etc.)
- For Klein 4B: 16GB VRAM sufficient
- For Klein 9B: 24GB+ VRAM recommended

### Software
- Python 3.10+
- CUDA 11.8 or 12.x
- Windows 10/11 (primary platform)

### Python Dependencies
```
torch>=2.0
PyQt6>=6.4
transformers
safetensors
pillow
einops
optimum-quanto
bitsandbytes (optional, for 8-bit optimizer)
```

## Installation

1. **Clone the repository**
   ```cmd
   git clone <repository-url>
   cd AI_Image_Trainer
   ```

2. **Create virtual environment**
   ```cmd
   python -m venv .venv
   .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```cmd
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   pip install PyQt6 transformers safetensors pillow einops tqdm matplotlib
   pip install optimum-quanto
   ```

4. **Run the application**
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
│
├── klein_trainer/             # Klein LoRA training module
│   ├── __init__.py
│   ├── config.py              # Training configuration
│   ├── dataset.py             # Dataset loading and caching
│   ├── model.py               # Klein model loading
│   ├── vae.py                 # VAE encode/decode
│   ├── lora.py                # LoRA implementation
│   ├── trainer.py             # Training loop
│   └── manager.py             # Training process management
│
├── dataset_manager/           # Dataset preparation tools
│   ├── embedded_dataset_manager.py
│   ├── image_processor.py     # Resize, rename utilities
│   └── ...
│
├── ai-toolkit/                # External: Klein model architecture
├── datasets/                  # Training datasets
└── output/                    # Training output (LoRAs, samples)
```

## Usage

### Quick Start

1. **Prepare your dataset**
   - Create a folder with training images (`.jpg`, `.png`)
   - Add caption files with the same name (`.txt`)
   - Example: `image001.jpg` + `image001.txt`

2. **Launch the GUI**
   ```cmd
   python main.py
   ```

3. **Configure training**
   - Select Klein 4B or 9B model
   - Set dataset path
   - Adjust learning rate, steps, LoRA rank
   - Add sample prompts for preview generation

4. **Start training**
   - Click "Start Training"
   - Monitor loss and sample images
   - LoRA saved to output folder

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
- Use Klein 4B instead of 9B
- Reduce LoRA rank (16 or 8)
- Enable gradient checkpointing (default)
- Reduce resolution to 512

### Training Too Slow
- Ensure CUDA is available: `python -c "import torch; print(torch.cuda.is_available())"`
- Check latent caching is enabled
- Verify gradient checkpointing is working

### LoRA Not Loading in ComfyUI
- Ensure you're using Klein model (not regular Flux)
- Check ComfyUI has Klein support/nodes installed
- Verify LoRA file is in correct folder

### GUI Won't Start
- Check Python environment is activated
- Verify PyQt6 is installed: `pip install PyQt6`
- Run from command line to see errors: `python main.py`

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
