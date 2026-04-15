#!/usr/bin/env python3
"""
Kohya Training Presets for Flux Training GUI
Pre-configured settings for common use cases in mass communications video production
"""

# Presets for different training scenarios
TRAINING_PRESETS = {
    'character_consistency': {
        'name': 'Character Consistency',
        'description': 'For training consistent character appearances across shots',
        'recommended_images': '30-50 images of the same character/person',
        'config': {
            'learning_rate': 1e-4,
            'max_train_steps': 1200,
            'save_every_n_steps': 400,
            'network_dim': 16,
            'network_alpha': 8,
            'optimizer_type': 'AdamW8bit',
            'mixed_precision': 'bf16',
            'lr_scheduler': 'constant',
        },
        'tips': [
            'Use consistent lighting across training images',
            'Include variety of poses and angles',
            'Describe character consistently in captions',
            'Example caption: "a photo of [character name], [pose/action]"'
        ]
    },
    
    'style_transfer': {
        'name': 'Style Transfer',
        'description': 'For learning artistic styles (illustration, painting, brand aesthetic)',
        'recommended_images': '20-40 images in the target style',
        'config': {
            'learning_rate': 2e-4,
            'max_train_steps': 1000,
            'save_every_n_steps': 500,
            'network_dim': 32,
            'network_alpha': 16,
            'optimizer_type': 'AdamW8bit',
            'mixed_precision': 'bf16',
            'lr_scheduler': 'constant',
        },
        'tips': [
            'Focus on stylistic consistency over subject matter',
            'Include variety of subjects in the style',
            'Caption with style keywords: "in the style of [style name]"',
            'Examples: "oil painting style", "flat illustration", "brand X aesthetic"'
        ]
    },
    
    'environment_location': {
        'name': 'Environment/Location',
        'description': 'For consistent locations, sets, or environments',
        'recommended_images': '25-40 images of the same location',
        'config': {
            'learning_rate': 1e-4,
            'max_train_steps': 1000,
            'save_every_n_steps': 500,
            'network_dim': 20,
            'network_alpha': 10,
            'optimizer_type': 'AdamW8bit',
            'mixed_precision': 'bf16',
            'lr_scheduler': 'constant',
        },
        'tips': [
            'Capture location from multiple angles',
            'Include different lighting conditions if relevant',
            'Caption with location markers: "a photo of [location name]"',
            'Example: "interior of coffee shop X", "exterior of campus building Y"'
        ]
    },
    
    'object_product': {
        'name': 'Object/Product',
        'description': 'For consistent product shots or specific objects',
        'recommended_images': '20-30 images of the same object',
        'config': {
            'learning_rate': 1.5e-4,
            'max_train_steps': 800,
            'save_every_n_steps': 400,
            'network_dim': 16,
            'network_alpha': 8,
            'optimizer_type': 'AdamW8bit',
            'mixed_precision': 'bf16',
            'lr_scheduler': 'constant',
        },
        'tips': [
            'Clean, well-lit product photos work best',
            'Vary angles and contexts',
            'Caption with object type: "a photo of [object name]"',
            'Example: "a photo of red coffee mug", "smartphone model X"'
        ]
    },
    
    'quick_test': {
        'name': 'Quick Test (Fast)',
        'description': 'Fast training for testing - lower quality but quick results',
        'recommended_images': '10-20 images minimum',
        'config': {
            'learning_rate': 2e-4,
            'max_train_steps': 500,
            'save_every_n_steps': 250,
            'network_dim': 8,
            'network_alpha': 4,
            'optimizer_type': 'AdamW8bit',
            'mixed_precision': 'bf16',
            'lr_scheduler': 'constant',
        },
        'tips': [
            'Use this for quick experiments',
            'Results may not be production-quality',
            'Good for testing if your dataset works',
            'Training time: ~1-2 hours on RTX 4090'
        ]
    },
    
    'high_quality': {
        'name': 'High Quality (Slow)',
        'description': 'Maximum quality training - longer training time',
        'recommended_images': '40-60 images recommended',
        'config': {
            'learning_rate': 8e-5,
            'max_train_steps': 2000,
            'save_every_n_steps': 500,
            'network_dim': 32,
            'network_alpha': 16,
            'optimizer_type': 'AdamW8bit',
            'mixed_precision': 'bf16',
            'lr_scheduler': 'cosine',
        },
        'tips': [
            'Use for final production work',
            'Requires high-quality training images',
            'Training time: ~5-7 hours on RTX 4090',
            'Best results with 50+ well-captioned images'
        ]
    }
}


def get_preset(preset_name):
    """
    Get a specific preset by name
    
    Args:
        preset_name: Name of the preset
        
    Returns:
        dict: Preset configuration or None if not found
    """
    return TRAINING_PRESETS.get(preset_name)


def get_preset_names():
    """
    Get list of available preset names
    
    Returns:
        list: List of preset names
    """
    return list(TRAINING_PRESETS.keys())


def get_preset_descriptions():
    """
    Get preset names with descriptions
    
    Returns:
        dict: {name: description} mapping
    """
    return {
        name: preset['description']
        for name, preset in TRAINING_PRESETS.items()
    }


def apply_preset_to_config(config_manager, preset_name):
    """
    Apply a preset to a config manager
    Works with both SDXL and FLUX presets
    
    Args:
        config_manager: KohyaConfigManager instance
        preset_name: Name of preset to apply
        
    Returns:
        bool: True if successful
    """
    # Try to get from SDXL presets first
    preset = get_preset(preset_name)
    
    # If not found, try FLUX presets
    if not preset:
        try:
            from flux_presets import get_flux_preset
            preset = get_flux_preset(preset_name)
        except ImportError:
            pass
    
    if not preset:
        return False
    
    # Apply the config values
    success = config_manager.apply_preset(preset['name'], preset['config'])
    
    # If preset has a model specified, update that too
    if 'model' in preset:
        config_manager.set_value('pretrained_model_name_or_path', preset['model'])
    
    return success


def get_preset_tips(preset_name):
    """
    Get tips for a specific preset
    
    Args:
        preset_name: Name of the preset
        
    Returns:
        list: List of tip strings
    """
    preset = get_preset(preset_name)
    if preset:
        return preset.get('tips', [])
    return []


def get_estimated_training_time(preset_name, gpu_type='RTX 4090'):
    """
    Get estimated training time for a preset
    
    Args:
        preset_name: Name of the preset
        gpu_type: GPU type (for estimation)
        
    Returns:
        str: Estimated time string
    """
    preset = get_preset(preset_name)
    if not preset:
        return "Unknown"
    
    steps = preset['config']['max_train_steps']
    
    # Rough estimates for RTX 4090 (adjust based on actual performance)
    # Assuming ~2-3 seconds per step
    time_minutes = (steps * 2.5) / 60
    
    if time_minutes < 60:
        return f"~{int(time_minutes)} minutes"
    else:
        hours = time_minutes / 60
        return f"~{hours:.1f} hours"


# Preset categories for UI organization
PRESET_CATEGORIES = {
    'common': ['character_consistency', 'style_transfer', 'environment_location'],
    'specialized': ['object_product'],
    'utility': ['quick_test', 'high_quality']
}


def get_presets_by_category():
    """
    Get presets organized by category
    
    Returns:
        dict: {category: [preset_names]}
    """
    return PRESET_CATEGORIES
