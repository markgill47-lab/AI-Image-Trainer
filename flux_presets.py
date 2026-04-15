#!/usr/bin/env python3
"""
FLUX Training Presets
Optimized presets for FLUX.1-dev and FLUX.1-schnell models
"""

# FLUX-specific presets
FLUX_PRESETS = {
    'flux_character': {
        'name': 'FLUX Character Consistency',
        'description': 'For training consistent character appearances with FLUX',
        'model': 'black-forest-labs/FLUX.1-dev',
        'recommended_images': '30-50 images of the same character/person',
        'config': {
            'learning_rate': 8e-5,
            'max_train_steps': 1200,
            'save_every_n_steps': 400,
            'network_dim': 16,
            'network_alpha': 8,
            'optimizer_type': 'AdamW8bit',
            'mixed_precision': 'bf16',
            'save_precision': 'bf16',
            'lr_scheduler': 'constant',
        },
        'tips': [
            'FLUX works best with bf16 precision',
            'Use consistent lighting across training images',
            'Include variety of poses and angles',
            'Caption format: "a photo of [character name], [action/pose]"',
            'Training time: ~3-4 hours on RTX 4090'
        ]
    },
    
    'flux_style': {
        'name': 'FLUX Style Transfer',
        'description': 'For learning artistic styles with FLUX',
        'model': 'black-forest-labs/FLUX.1-dev',
        'recommended_images': '20-40 images in the target style',
        'config': {
            'learning_rate': 1e-4,
            'max_train_steps': 1000,
            'save_every_n_steps': 500,
            'network_dim': 32,
            'network_alpha': 16,
            'optimizer_type': 'AdamW8bit',
            'mixed_precision': 'bf16',
            'save_precision': 'bf16',
            'lr_scheduler': 'constant',
        },
        'tips': [
            'Focus on stylistic consistency over subject matter',
            'Include variety of subjects in the style',
            'Caption with style keywords: "in the style of [style name]"',
            'FLUX excels at complex artistic styles',
            'Training time: ~2-3 hours on RTX 4090'
        ]
    },
    
    'flux_environment': {
        'name': 'FLUX Environment/Location',
        'description': 'For consistent locations with FLUX',
        'model': 'black-forest-labs/FLUX.1-dev',
        'recommended_images': '25-40 images of the same location',
        'config': {
            'learning_rate': 8e-5,
            'max_train_steps': 1000,
            'save_every_n_steps': 500,
            'network_dim': 20,
            'network_alpha': 10,
            'optimizer_type': 'AdamW8bit',
            'mixed_precision': 'bf16',
            'save_precision': 'bf16',
            'lr_scheduler': 'constant',
        },
        'tips': [
            'Capture location from multiple angles',
            'Include different lighting conditions',
            'FLUX handles complex environments well',
            'Caption: "a photo of [location name], [view/angle]"',
            'Training time: ~2-3 hours on RTX 4090'
        ]
    },
    
    'flux_quick': {
        'name': 'FLUX Quick Test',
        'description': 'Fast FLUX training for testing',
        'model': 'black-forest-labs/FLUX.1-schnell',  # Use schnell for speed
        'recommended_images': '15-25 images minimum',
        'config': {
            'learning_rate': 1.5e-4,
            'max_train_steps': 600,
            'save_every_n_steps': 300,
            'network_dim': 12,
            'network_alpha': 6,
            'optimizer_type': 'AdamW8bit',
            'mixed_precision': 'bf16',
            'save_precision': 'bf16',
            'lr_scheduler': 'constant',
        },
        'tips': [
            'Uses FLUX.1-schnell for faster training',
            'Good for testing if your dataset works',
            'Results may not be production-quality',
            'Training time: ~1.5-2 hours on RTX 4090',
            'Upgrade to FLUX.1-dev for final work'
        ]
    },
    
    'flux_high_quality': {
        'name': 'FLUX High Quality',
        'description': 'Maximum quality FLUX training',
        'model': 'black-forest-labs/FLUX.1-dev',
        'recommended_images': '50-80 images recommended',
        'config': {
            'learning_rate': 6e-5,
            'max_train_steps': 2000,
            'save_every_n_steps': 500,
            'network_dim': 32,
            'network_alpha': 16,
            'optimizer_type': 'AdamW8bit',
            'mixed_precision': 'bf16',
            'save_precision': 'bf16',
            'lr_scheduler': 'cosine',
        },
        'tips': [
            'Use for final production work',
            'Requires high-quality training images',
            'FLUX.1-dev produces exceptional results',
            'Best results with 60+ well-captioned images',
            'Training time: ~5-6 hours on RTX 4090'
        ]
    },

    'flux2_character_9b': {
        'name': 'FLUX 2.0 Character (9B)',
        'description': 'Character consistency preset for FLUX.2-klein-9B (16GB VRAM optimized)',
        'model': 'black-forest-labs/FLUX.2-klein-base-9B',
        'recommended_images': '30-50 images of the same character/person',
        'config': {
            'learning_rate': 8e-5,
            'max_train_steps': 1200,
            'save_every_n_steps': 400,
            'network_dim': 16,
            'network_alpha': 8,
            'optimizer_type': 'AdamW8bit',
            'mixed_precision': 'bf16',
            'save_precision': 'bf16',
            'lr_scheduler': 'constant',
        },
        'tips': [
            'FLUX 2.0 9B offers quality matching larger models',
            'Optimized for 16GB VRAM (RTX 4060 Ti, RTX 3090, etc.)',
            'Uses layer offloading and quantization for efficiency',
            'Use consistent lighting across training images',
            'Include variety of poses and angles',
            'Caption format: "a photo of [character name], [action/pose]"',
            'Training time: ~4-5 hours on RTX 4060 Ti 16GB'
        ]
    },

    'flux2_style_9b': {
        'name': 'FLUX 2.0 Style (9B)',
        'description': 'Style transfer for FLUX.2-klein-9B (16GB VRAM optimized)',
        'model': 'black-forest-labs/FLUX.2-klein-base-9B',
        'recommended_images': '20-40 images in the target style',
        'config': {
            'learning_rate': 1e-4,
            'max_train_steps': 1000,
            'save_every_n_steps': 500,
            'network_dim': 32,
            'network_alpha': 16,
            'optimizer_type': 'AdamW8bit',
            'mixed_precision': 'bf16',
            'save_precision': 'bf16',
            'lr_scheduler': 'constant',
        },
        'tips': [
            'Focus on stylistic consistency over subject matter',
            'Include variety of subjects in the style',
            'FLUX 2.0 excels at complex artistic styles',
            'Optimized for 16GB VRAM with layer offloading',
            'Caption with style keywords: "in the style of [style name]"',
            'Training time: ~3-4 hours on RTX 4060 Ti 16GB'
        ]
    },

    'flux2_quick_4b': {
        'name': 'FLUX 2.0 Quick Test (4B)',
        'description': 'Fast testing with FLUX.2-klein-4B',
        'model': 'black-forest-labs/FLUX.2-klein-base-4B',
        'recommended_images': '15-25 images minimum',
        'config': {
            'learning_rate': 1.5e-4,
            'max_train_steps': 600,
            'save_every_n_steps': 300,
            'network_dim': 12,
            'network_alpha': 6,
            'optimizer_type': 'AdamW8bit',
            'mixed_precision': 'bf16',
            'save_precision': 'bf16',
            'lr_scheduler': 'constant',
        },
        'tips': [
            'Uses FLUX.2-klein-4B for ultra-fast training',
            'Fits RTX 3090/4070 (13GB VRAM)',
            'Good for testing if your dataset works',
            'Results may not be production-quality',
            'Training time: ~1-1.5 hours on RTX 4090',
            'Upgrade to 9B model for final work'
        ]
    },

    'flux2_high_quality_9b': {
        'name': 'FLUX 2.0 High Quality (9B)',
        'description': 'Maximum quality FLUX 2.0 training (16GB VRAM optimized)',
        'model': 'black-forest-labs/FLUX.2-klein-base-9B',
        'recommended_images': '50-80 images recommended',
        'config': {
            'learning_rate': 6e-5,
            'max_train_steps': 2000,
            'save_every_n_steps': 500,
            'network_dim': 32,
            'network_alpha': 16,
            'optimizer_type': 'AdamW8bit',
            'mixed_precision': 'bf16',
            'save_precision': 'bf16',
            'lr_scheduler': 'cosine',
        },
        'tips': [
            'Use for final production work',
            'Requires high-quality training images',
            'FLUX 2.0 9B produces exceptional results',
            'Optimized for 16GB VRAM with advanced memory management',
            'Best results with 60+ well-captioned images',
            'Training time: ~6-7 hours on RTX 4060 Ti 16GB',
            'Comparable to larger models with less VRAM'
        ]
    }
}


# Model options for dropdown
MODEL_OPTIONS = {
    'sdxl': {
        'name': 'Stable Diffusion XL 1.0',
        'model_id': 'stabilityai/stable-diffusion-xl-base-1.0',
        'backend': 'kohya',
        'description': 'Fast, stable, proven model for LoRA training',
        'precision': 'fp16'
    },
    'flux_dev': {
        'name': 'FLUX.1-dev',
        'model_id': 'black-forest-labs/FLUX.1-dev',
        'backend': 'aitoolkit',
        'description': 'Cutting-edge quality, best for production work',
        'precision': 'bf16'
    },
    'flux_schnell': {
        'name': 'FLUX.1-schnell',
        'model_id': 'black-forest-labs/FLUX.1-schnell',
        'backend': 'aitoolkit',
        'description': 'Fast FLUX variant, good for iteration',
        'precision': 'bf16'
    },
    'flux2_klein_9b': {
        'name': 'FLUX.2-klein-9B',
        'model_id': 'black-forest-labs/FLUX.2-klein-base-9B',
        'backend': 'aitoolkit',
        'description': 'FLUX 2.0 9B - Matches larger models, fits RTX 4090 (29GB VRAM)',
        'precision': 'bf16'
    },
    'flux2_klein_4b': {
        'name': 'FLUX.2-klein-4B',
        'model_id': 'black-forest-labs/FLUX.2-klein-base-4B',
        'backend': 'aitoolkit',
        'description': 'FLUX 2.0 4B - Fast and efficient, fits RTX 3090/4070 (13GB VRAM)',
        'precision': 'bf16'
    }
}


def get_flux_preset(preset_name):
    """
    Get a FLUX preset by name
    
    Args:
        preset_name: Name of the preset
        
    Returns:
        dict: Preset configuration or None
    """
    return FLUX_PRESETS.get(preset_name)


def get_flux_preset_names():
    """
    Get list of FLUX preset names
    
    Returns:
        list: List of preset names
    """
    return list(FLUX_PRESETS.keys())


def get_all_presets():
    """
    Get all presets (including SDXL from kohya_presets)
    
    Returns:
        dict: All presets combined
    """
    from kohya_presets import TRAINING_PRESETS
    
    all_presets = {}
    all_presets.update(TRAINING_PRESETS)
    all_presets.update(FLUX_PRESETS)
    
    return all_presets


def get_presets_for_model(model_id):
    """
    Get appropriate presets for a specific model

    Args:
        model_id: Model identifier

    Returns:
        dict: Presets suitable for this model
    """
    from kohya_presets import TRAINING_PRESETS

    model_lower = model_id.lower()

    if 'flux' in model_lower or 'klein' in model_lower:
        return FLUX_PRESETS
    else:
        return TRAINING_PRESETS


def get_model_options():
    """
    Get available model options for selection
    
    Returns:
        dict: Model options
    """
    return MODEL_OPTIONS


def get_recommended_precision(model_id):
    """
    Get recommended precision for a model

    Args:
        model_id: Model identifier

    Returns:
        str: 'bf16' or 'fp16'
    """
    model_lower = model_id.lower()
    if 'flux' in model_lower or 'klein' in model_lower:
        return 'bf16'
    else:
        return 'fp16'
