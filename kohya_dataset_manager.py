#!/usr/bin/env python3
"""
Kohya Dataset Manager for Flux Training GUI
Handles dataset operations and validation for Kohya format
Kohya uses simple: image.jpg + image.txt (caption) format
"""

import os
import toml
import logging
from PIL import Image

logger = logging.getLogger('FluxTrainingGUI')


class KohyaDatasetManager:
    """
    Manages dataset operations for Kohya sd-scripts training
    Kohya dataset structure: folder with images and matching .txt caption files
    """
    
    def __init__(self, update_status_callback, show_error_callback, show_info_callback):
        """
        Initialize the dataset manager
        
        Args:
            update_status_callback: Function to update status in the UI
            show_error_callback: Function to display error messages (title, message)
            show_info_callback: Function to display info messages (title, message)
        """
        self.update_status = update_status_callback
        self.show_error = show_error_callback
        self.show_info = show_info_callback
        
        # Dataset configuration
        self.dataset_config_file = None
        self.dataset_config = None
        
        # Default dataset config for Kohya
        self.default_dataset_config = {
            'general': {
                'resolution': 512,
                'batch_size': 1
            },
            'datasets': [
                {
                    'resolution': 512,
                    'batch_size': 1,
                    'subsets': [
                        {
                            'image_dir': 'datasets/my_dataset',
                            'num_repeats': 10,
                            'caption_extension': '.txt'  # Default to .txt files
                        }
                    ]
                }
            ]
        }
    
    def create_default_dataset_config(self, config_path='dataset.toml'):
        """
        Create a default dataset configuration
        
        Args:
            config_path: Path where to save the config
            
        Returns:
            bool: True if successful
        """
        try:
            with open(config_path, 'w') as f:
                toml.dump(self.default_dataset_config, f)
            
            self.dataset_config_file = config_path
            self.dataset_config = dict(self.default_dataset_config)
            self.update_status(f"Created default dataset config at {config_path}")
            logger.info(f"Created default dataset config at {config_path}")
            return True
            
        except Exception as e:
            error_msg = f"Failed to create dataset config: {str(e)}"
            self.show_error("Dataset Config Error", error_msg)
            logger.error(error_msg, exc_info=True)
            return False
    
    def load_dataset_config(self, config_path='dataset.toml'):
        """
        Load dataset configuration
        
        Args:
            config_path: Path to dataset config
            
        Returns:
            bool: True if successful
        """
        if not os.path.exists(config_path):
            self.update_status(f"Dataset config not found: {config_path}")
            logger.warning(f"Dataset config not found: {config_path}")
            return self.create_default_dataset_config(config_path)
        
        try:
            self.dataset_config = toml.load(config_path)
            self.dataset_config_file = config_path
            self.update_status(f"Loaded dataset config from {config_path}")
            logger.info(f"Successfully loaded dataset config from {config_path}")
            return True
            
        except Exception as e:
            error_msg = f"Failed to load dataset config: {str(e)}"
            self.show_error("Dataset Config Error", error_msg)
            logger.error(error_msg, exc_info=True)
            return False
    
    def save_dataset_config(self, config_path=None):
        """
        Save dataset configuration
        
        Args:
            config_path: Optional path (uses self.dataset_config_file if None)
            
        Returns:
            bool: True if successful
        """
        save_path = config_path or self.dataset_config_file
        
        if not save_path:
            self.show_error("Save Error", "No dataset config path specified")
            return False
        
        if not self.dataset_config:
            self.show_error("Save Error", "No dataset config to save")
            return False
        
        try:
            # Create backup
            if os.path.exists(save_path):
                import shutil
                backup_path = f"{save_path}.backup"
                shutil.copy2(save_path, backup_path)
                logger.info(f"Created backup at {backup_path}")
            
            # Save
            with open(save_path, 'w') as f:
                toml.dump(self.dataset_config, f)
            
            self.dataset_config_file = save_path
            self.update_status(f"Saved dataset config to {save_path}")
            logger.info(f"Successfully saved dataset config to {save_path}")
            return True
            
        except Exception as e:
            error_msg = f"Failed to save dataset config: {str(e)}"
            self.show_error("Save Error", error_msg)
            logger.error(error_msg, exc_info=True)
            return False
    
    def set_image_directory(self, image_dir, num_repeats=10, resolution=512):
        """
        Set the image directory for training
        
        Args:
            image_dir: Path to directory containing images
            num_repeats: Number of times to repeat the dataset per epoch
            resolution: Training resolution
            
        Returns:
            bool: True if successful
        """
        if not self.dataset_config:
            self.dataset_config = dict(self.default_dataset_config)
        
        try:
            # Preserve existing caption_extension if it exists
            existing_caption_ext = None
            try:
                existing_caption_ext = self.dataset_config['datasets'][0]['subsets'][0].get('caption_extension')
            except (KeyError, IndexError):
                pass
            
            # Update dataset config
            self.dataset_config['datasets'][0]['subsets'][0]['image_dir'] = image_dir
            self.dataset_config['datasets'][0]['subsets'][0]['num_repeats'] = num_repeats
            
            # Preserve or set caption_extension
            if existing_caption_ext:
                self.dataset_config['datasets'][0]['subsets'][0]['caption_extension'] = existing_caption_ext
            else:
                self.dataset_config['datasets'][0]['subsets'][0]['caption_extension'] = '.txt'
            
            self.dataset_config['datasets'][0]['resolution'] = resolution
            self.dataset_config['general']['resolution'] = resolution
            
            self.update_status(f"Set dataset directory to: {image_dir}")
            logger.info(f"Set dataset directory to: {image_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Error setting image directory: {str(e)}")
            return False
    
    def validate_dataset(self, image_dir, progress_callback=None):
        """
        Validate dataset directory
        Checks for:
        - Valid images
        - Matching caption files
        - Image formats
        
        Args:
            image_dir: Directory to validate
            progress_callback: Optional callback for progress updates
            
        Returns:
            tuple: (total_images, valid_pairs, issues_list)
        """
        if not os.path.exists(image_dir):
            return 0, 0, [f"Directory does not exist: {image_dir}"]
        
        self.update_status(f"\n=== Validating Dataset: {image_dir} ===")
        
        # Supported image formats
        image_extensions = ('.jpg', '.jpeg', '.png', '.webp', '.bmp')
        
        # Find all images
        try:
            all_files = os.listdir(image_dir)
        except Exception as e:
            return 0, 0, [f"Cannot read directory: {str(e)}"]
        
        image_files = [f for f in all_files if f.lower().endswith(image_extensions)]
        total_images = len(image_files)
        
        self.update_status(f"Found {total_images} images")
        logger.info(f"Found {total_images} images in {image_dir}")
        
        valid_pairs = 0
        issues = []
        
        for idx, image_file in enumerate(image_files):
            if progress_callback:
                progress_callback(idx + 1, total_images)
            
            image_path = os.path.join(image_dir, image_file)
            
            # Check if image is valid
            try:
                with Image.open(image_path) as img:
                    img.verify()
            except Exception as e:
                issues.append(f"Corrupted image: {image_file} - {str(e)}")
                continue
            
            # Check for caption file
            base_name = os.path.splitext(image_file)[0]
            caption_file = base_name + '.txt'
            caption_path = os.path.join(image_dir, caption_file)
            
            if not os.path.exists(caption_path):
                issues.append(f"Missing caption: {caption_file} for {image_file}")
                continue
            
            # Check if caption file has content
            try:
                with open(caption_path, 'r', encoding='utf-8') as f:
                    caption_text = f.read().strip()
                    if not caption_text:
                        issues.append(f"Empty caption: {caption_file}")
                        continue
            except Exception as e:
                issues.append(f"Cannot read caption: {caption_file} - {str(e)}")
                continue
            
            # Valid pair!
            valid_pairs += 1
        
        # Summary
        self.update_status(f"\n=== Validation Summary ===")
        self.update_status(f"Total images: {total_images}")
        self.update_status(f"Valid pairs: {valid_pairs}")
        self.update_status(f"Issues: {len(issues)}")
        
        if issues:
            self.update_status("\nIssues found:")
            for issue in issues[:10]:  # Show first 10 issues
                self.update_status(f"  - {issue}")
            if len(issues) > 10:
                self.update_status(f"  ... and {len(issues) - 10} more issues")
        
        logger.info(f"Validation complete: {valid_pairs}/{total_images} valid pairs")
        
        return total_images, valid_pairs, issues
    
    def get_dataset_stats(self, image_dir=None):
        """
        Get statistics about the dataset
        
        Args:
            image_dir: Optional directory (uses config if None)
            
        Returns:
            dict: Dataset statistics
        """
        # Get directory from config if not specified
        if not image_dir and self.dataset_config:
            try:
                image_dir = self.dataset_config['datasets'][0]['subsets'][0]['image_dir']
            except (KeyError, IndexError):
                return {'error': 'No dataset directory configured'}
        
        if not image_dir or not os.path.exists(image_dir):
            return {'error': 'Dataset directory not found'}
        
        stats = {
            'total_images': 0,
            'with_captions': 0,
            'without_captions': 0,
            'corrupted_images': 0
        }
        
        image_extensions = ('.jpg', '.jpeg', '.png', '.webp', '.bmp')
        
        try:
            all_files = os.listdir(image_dir)
            image_files = [f for f in all_files if f.lower().endswith(image_extensions)]
            stats['total_images'] = len(image_files)
            
            for image_file in image_files:
                image_path = os.path.join(image_dir, image_file)
                
                # Check if image is valid
                try:
                    with Image.open(image_path) as img:
                        img.verify()
                except:
                    stats['corrupted_images'] += 1
                    continue
                
                # Check for caption
                base_name = os.path.splitext(image_file)[0]
                caption_path = os.path.join(image_dir, base_name + '.txt')
                
                if os.path.exists(caption_path):
                    stats['with_captions'] += 1
                else:
                    stats['without_captions'] += 1
        
        except Exception as e:
            stats['error'] = str(e)
        
        return stats
    
    def create_missing_captions(self, image_dir, default_caption="training image"):
        """
        Create placeholder caption files for images missing them
        
        Args:
            image_dir: Directory containing images
            default_caption: Default caption text to use
            
        Returns:
            int: Number of caption files created
        """
        if not os.path.exists(image_dir):
            self.show_error("Error", f"Directory not found: {image_dir}")
            return 0
        
        image_extensions = ('.jpg', '.jpeg', '.png', '.webp', '.bmp')
        
        try:
            all_files = os.listdir(image_dir)
            image_files = [f for f in all_files if f.lower().endswith(image_extensions)]
            
            created_count = 0
            
            for image_file in image_files:
                base_name = os.path.splitext(image_file)[0]
                caption_path = os.path.join(image_dir, base_name + '.txt')
                
                if not os.path.exists(caption_path):
                    try:
                        with open(caption_path, 'w', encoding='utf-8') as f:
                            f.write(default_caption)
                        created_count += 1
                        logger.info(f"Created caption file: {base_name}.txt")
                    except Exception as e:
                        logger.error(f"Failed to create caption for {image_file}: {str(e)}")
            
            if created_count > 0:
                self.update_status(f"Created {created_count} caption files")
                self.show_info("Captions Created", 
                              f"Created {created_count} placeholder caption files.\n"
                              f"Please edit them to describe your images.")
            
            return created_count
            
        except Exception as e:
            error_msg = f"Error creating captions: {str(e)}"
            self.show_error("Error", error_msg)
            logger.error(error_msg, exc_info=True)
            return 0
