#!/usr/bin/env python3
"""
Image processing utilities for the Image Gallery application
"""

import os
import json
import shutil
import random
import string
from PIL import Image
from PyQt6.QtWidgets import QApplication, QProgressDialog, QMessageBox
from PyQt6.QtCore import Qt


class ImageProcessor:
    """Handles image processing operations like resizing and augmentation"""
    
    def __init__(self, parent=None):
        self.parent = parent
        
    @staticmethod
    def get_image_files(folder_path):
        """Get all supported image files from a folder"""
        image_extensions = ('.jpg', '.jpeg', '.png', '.webp')
        return [f for f in os.listdir(folder_path) 
                if os.path.isfile(os.path.join(folder_path, f)) 
                and f.lower().endswith(image_extensions)]
    
    @staticmethod
    def validate_image(img_path, allow_small_images=False):
        """Validate that an image is not corrupt and optionally check size requirements"""
        try:
            with Image.open(img_path) as img:
                # Check if image can be loaded and verify
                img.verify()
                
            # Need to reopen after verify()
            with Image.open(img_path) as img:
                width, height = img.size
                
                # Check minimum size requirement (512x512) only if not allowing small images
                if not allow_small_images and (width < 512 or height < 512):
                    return False, f"Image too small: {width}x{height} (minimum 512x512)"
                
                # Check for extremely small images that can't be reasonably upscaled
                if width < 32 or height < 32:
                    return False, f"Image too small to process: {width}x{height} (minimum 32x32)"
                
            return True, "Valid"
            
        except Exception as e:
            return False, f"Invalid or corrupt image: {str(e)}"
    
    def fix_images(self, source_folder, target_size, output_folder, resize_small_images=False, images_to_process=None, status_callback=None):
        """
        Process images to fix dimensions and format (always preserves aspect ratio)

        Args:
            source_folder: Source directory with images
            target_size: Target size for longest dimension
            output_folder: Output directory
            resize_small_images: Whether to upscale small images instead of skipping them
            images_to_process: List of image filenames to process (if None, processes all)
            status_callback: Function to call with status updates

        Returns:
            dict: Results summary
        """
        if images_to_process is None:
            # Process all images in folder
            image_files = self.get_image_files(source_folder)
        else:
            # Process only specified images
            image_files = images_to_process
        
        if not image_files:
            return {'error': "No images found in the specified scope"}
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_folder):
            try:
                os.makedirs(output_folder)
            except Exception as e:
                return {'error': f"Could not create output folder: {str(e)}"}
        
        # Create progress dialog
        if self.parent:
            progress = QProgressDialog("Processing images...", "Cancel", 0, len(image_files), self.parent)
            progress.setWindowTitle("Fix Images")
            progress.setWindowModality(Qt.WindowModality.WindowModal)
            progress.setMinimumDuration(0)
            progress.setValue(0)
        else:
            progress = None
        
        # Track progress
        processed_images = 0
        skipped_images = 0
        invalid_images = []
        
        # Process each image
        for i, img_file in enumerate(image_files):
            # Update progress
            if progress:
                progress.setValue(i)
                if progress.wasCanceled():
                    break
                    
            img_path = os.path.join(source_folder, img_file)
            
            # Validate image first
            is_valid, validation_message = self.validate_image(img_path, allow_small_images=resize_small_images)
            if not is_valid:
                invalid_images.append((img_file, validation_message))
                if status_callback:
                    status_callback(f"Skipping {img_file}: {validation_message}")
                if progress:
                    QApplication.processEvents()
                continue
            
            try:
                # Open the image
                img = Image.open(img_path)
                width, height = img.size
                
                # Check if image needs upscaling first (only if resize_small_images is enabled)
                needs_upscaling = resize_small_images and (width < 512 or height < 512)
                
                if needs_upscaling:
                    # Upscale small image to meet minimum size requirements
                    img = self._upscale_small_image(img, target_size)
                    width, height = img.size
                
                # Check if already correct size (longest dimension matches target)
                is_correct_size = max(width, height) == target_size
                
                if is_correct_size and not needs_upscaling:
                    # Already good, just copy if different folders
                    if source_folder != output_folder:
                        fixed_img_path = os.path.join(output_folder, img_file)
                        shutil.copy2(img_path, fixed_img_path)
                    skipped_images += 1
                else:
                    # Need to process (either resize or was upscaled)
                    if not needs_upscaling:
                        # Normal resize
                        processed_img = self._resize_image(img, target_size)
                    else:
                        # Already upscaled, may need further processing
                        if not is_correct_size:
                            processed_img = self._resize_image(img, target_size)
                        else:
                            processed_img = img
                    
                    # Save the processed image
                    fixed_img_path = os.path.join(output_folder, img_file)
                    processed_img.save(fixed_img_path, quality=95)
                    processed_images += 1
                
                # Copy associated text file if it exists
                self._copy_text_file(source_folder, output_folder, img_file)
                
                # Update status
                if status_callback:
                    status_callback(f"Processing images: {processed_images + skipped_images}/{len(image_files)}")
                if progress:
                    QApplication.processEvents()
                
            except Exception as e:
                print(f"Error processing {img_file}: {str(e)}")
                invalid_images.append((img_file, str(e)))
        
        # Copy JSON files
        if source_folder != output_folder:
            self._copy_json_files(source_folder, output_folder)
        
        # Close progress dialog properly
        if progress:
            progress.close()
            progress.deleteLater()
        
        return {
            'processed': processed_images,
            'skipped': skipped_images,
            'invalid': invalid_images,
            'total': len(image_files)
        }
    
    def create_duplicates(self, input_folder, output_folder, transformations, images_to_process=None, status_callback=None):
        """Create duplicated images with transformations"""
        
        if images_to_process is None:
            image_files = self.get_image_files(input_folder)
        else:
            image_files = images_to_process
        
        if not image_files:
            return {'error': "No images found in the specified scope"}

        # Build transform list
        transform_ops = []
        if transformations.get('flip_horizontal'):
            transform_ops.append(('flip', 'Horizontal Flip'))
        if transformations.get('rotate_90_left'):
            transform_ops.append(('rot90l', '90° Left'))
        if transformations.get('rotate_90_right'):
            transform_ops.append(('rot90r', '90° Right'))
        if transformations.get('rotate_180'):
            transform_ops.append(('rot180', '180°'))
        
        # If no transformations, add duplicate
        if not transform_ops:
            transform_ops.append(('duplicate', 'Simple Duplicate'))
        
        # Create output directory if it doesn't exist and different from input
        if input_folder != output_folder and not os.path.exists(output_folder):
            try:
                os.makedirs(output_folder)
            except Exception as e:
                return {'error': f"Could not create output folder: {str(e)}"}

        # Create progress dialog
        if self.parent:
            progress = QProgressDialog("Creating duplicates...", "Cancel", 0, len(image_files), self.parent)
            progress.setWindowTitle("Create Duplicates")
            progress.setWindowModality(Qt.WindowModality.WindowModal)
            progress.setMinimumDuration(0)
            progress.setValue(0)
        else:
            progress = None
        
        created_files = 0
        error_files = []
        
        # Process each image
        for i, img_file in enumerate(image_files):
            if progress:
                if progress.wasCanceled():
                    break
                progress.setValue(i)
                progress.setLabelText(f"Processing {img_file}...")
                QApplication.processEvents()
            
            img_path = os.path.join(input_folder, img_file)
            base_name = os.path.splitext(img_file)[0]
            img_ext = os.path.splitext(img_file)[1]
            
            try:
                original_img = Image.open(img_path)
                
                # Only copy original if different folders
                if input_folder != output_folder:
                    original_output_path = os.path.join(output_folder, img_file)
                    original_img.save(original_output_path, quality=95)
                    created_files += 1
                    self._copy_text_file(input_folder, output_folder, img_file)
                
                # Create transformed versions
                for transform_key, transform_name in transform_ops:
                    try:
                        transformed_img, suffix = self._apply_transformation(original_img, transform_key)
                        new_img_name = f"{base_name}{suffix}{img_ext}"
                        new_img_path = os.path.join(output_folder, new_img_name)
                        
                        transformed_img.save(new_img_path, quality=95)
                        created_files += 1
                        
                        self._copy_transformed_text_file(input_folder, output_folder, img_file, suffix)
                        
                    except Exception as e:
                        error_msg = f"Error creating {transform_name} of {img_file}: {str(e)}"
                        error_files.append(error_msg)
                        print(f"Error: {error_msg}")
                    
            except Exception as e:
                error_msg = f"Error processing {img_file}: {str(e)}"
                error_files.append(error_msg)
                print(f"Error: {error_msg}")
        
        # Update JSON file with all variants
        self._create_augmented_json(input_folder, output_folder, transform_ops)
        
        # Close progress dialog properly
        if progress:
            progress.close()
            progress.deleteLater()
        
        return {
            'created_files': created_files,
            'transformations': len(transform_ops),
            'original_images': len(image_files),
            'errors': error_files,
            'same_folder': input_folder == output_folder
        }
    
    def mass_rename_images(self, folder_path, prefix, images_to_process=None, scramble_order=False, status_callback=None):
        """
        Rename images in a folder with a new prefix and sequential numbers
        
        Args:
            folder_path: Path to folder containing images
            prefix: New prefix for filenames
            images_to_process: List of image data dicts to process (if None, processes all)
            scramble_order: If True, adds random alpha characters to scramble order
            status_callback: Function to call with status updates
        
        Returns:
            dict: Results summary
        """
        if images_to_process is None:
            # Process all images in folder
            image_files = self.get_image_files(folder_path)
            if not image_files:
                return {'error': "No images found in the folder"}
            image_files.sort()
        else:
            # Process only specified images
            image_files = [img_data['filename'] for img_data in images_to_process]
            if not image_files:
                return {'error': "No images specified for processing"}
        
        # Calculate number of digits needed for zero-padding
        num_digits = len(str(len(image_files)))
        if num_digits < 3:
            num_digits = 3  # Minimum 3 digits (001, 002, etc.)
        
        # Generate scramble characters if needed
        scramble_chars = []
        if scramble_order:
            scramble_chars = self._generate_scramble_characters(len(image_files))
        
        # Check for potential naming conflicts
        conflicts = self._check_rename_conflicts(folder_path, prefix, image_files, num_digits, scramble_chars)
        if conflicts:
            return {'error': f"Naming conflicts detected. These files already exist: {', '.join(conflicts[:5])}{'...' if len(conflicts) > 5 else ''}"}
        
        # Create progress dialog
        if self.parent:
            progress = QProgressDialog("Renaming images...", "Cancel", 0, len(image_files), self.parent)
            progress.setWindowTitle("Mass Rename")
            progress.setWindowModality(Qt.WindowModality.WindowModal)
            progress.setMinimumDuration(0)
            progress.setValue(0)
        else:
            progress = None
        
        renamed_images = 0
        renamed_descriptions = 0
        errors = []
        old_to_new_mapping = {}
        
        # Rename files
        for i, old_filename in enumerate(image_files):
            if progress:
                if progress.wasCanceled():
                    break
                    
                progress.setValue(i)
                progress.setLabelText(f"Renaming {old_filename}...")
                QApplication.processEvents()
            
            try:
                # Generate new filename
                file_extension = os.path.splitext(old_filename)[1]
                
                if scramble_order:
                    # Include scramble character: prefix_a001.jpg
                    scramble_char = scramble_chars[i]
                    new_filename = f"{prefix}_{scramble_char}{str(i + 1).zfill(num_digits)}{file_extension}"
                else:
                    # Standard numbering: prefix_001.jpg
                    new_filename = f"{prefix}_{str(i + 1).zfill(num_digits)}{file_extension}"
                
                old_path = os.path.join(folder_path, old_filename)
                new_path = os.path.join(folder_path, new_filename)
                
                # Rename image file
                os.rename(old_path, new_path)
                renamed_images += 1
                old_to_new_mapping[old_filename] = new_filename
                
                # Rename corresponding description file if it exists
                old_desc_filename = os.path.splitext(old_filename)[0] + '.txt'
                new_desc_filename = os.path.splitext(new_filename)[0] + '.txt'
                old_desc_path = os.path.join(folder_path, old_desc_filename)
                new_desc_path = os.path.join(folder_path, new_desc_filename)
                
                if os.path.exists(old_desc_path):
                    os.rename(old_desc_path, new_desc_path)
                    renamed_descriptions += 1
                
                if status_callback:
                    status_callback(f"Renamed {renamed_images}/{len(image_files)} images...")
                
            except Exception as e:
                error_msg = f"Failed to rename {old_filename}: {str(e)}"
                errors.append(error_msg)
                print(error_msg)
        
        # Update JSON files with new names
        self._update_json_with_new_names(folder_path, old_to_new_mapping)
        
        # Close progress dialog properly
        if progress:
            progress.close()
            progress.deleteLater()
        
        return {
            'renamed_images': renamed_images,
            'renamed_descriptions': renamed_descriptions,
            'errors': errors,
            'total_images': len(image_files),
            'scrambled': scramble_order
        }
    
    def _generate_scramble_characters(self, count):
        """
        Generate random alphabetic characters for scrambling
        
        Args:
            count: Number of characters needed
        
        Returns:
            List of single characters with no duplicates
        """
        # Start with all lowercase letters
        available_chars = list(string.ascii_lowercase)
        random.shuffle(available_chars)
        
        scramble_chars = []
        char_index = 0
        
        for i in range(count):
            # If we've used all letters, reshuffle and start over
            if char_index >= len(available_chars):
                random.shuffle(available_chars)
                char_index = 0
            
            scramble_chars.append(available_chars[char_index])
            char_index += 1
        
        return scramble_chars
    
    def _upscale_small_image(self, img, target_size):
        """Upscale a small image to meet minimum size requirements"""
        # Convert to RGB mode if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        width, height = img.size
        
        # Calculate scale factor to make the image at least 512x512
        min_dimension = min(width, height)
        scale_factor = max(512 / min_dimension, 1.0)  # Ensure we don't downscale
        
        # Calculate new dimensions
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        
        # Upscale using high-quality resampling
        upscaled_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        return upscaled_img
    
    def _resize_image(self, img, target_size):
        """Resize image to fit target size while preserving aspect ratio"""
        # Convert to RGB mode if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')

        width, height = img.size

        # Determine scaling based on longest dimension
        if width >= height:
            new_width = target_size
            new_height = int((height / width) * target_size)
        else:
            new_height = target_size
            new_width = int((width / height) * target_size)

        # Resize image while maintaining aspect ratio
        return img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    def _apply_transformation(self, img, transform_key):
        """Apply a specific transformation to an image"""
        if transform_key == 'flip':
            transformed_img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            suffix = "_flipHor"
        elif transform_key == 'rot90l':
            transformed_img = img.transpose(Image.Transpose.ROTATE_90)
            suffix = "_rotLeft"
        elif transform_key == 'rot90r':
            transformed_img = img.transpose(Image.Transpose.ROTATE_270)
            suffix = "_rotRight"
        elif transform_key == 'rot180':
            transformed_img = img.transpose(Image.Transpose.ROTATE_180)
            suffix = "_flipVert"
        elif transform_key == 'duplicate':
            # Simple duplicate - no transformation, just copy
            transformed_img = img.copy()
            suffix = "_dup"
        else:
            raise ValueError(f"Unknown transformation: {transform_key}")
        
        return transformed_img, suffix
    
    def _copy_text_file(self, source_folder, output_folder, img_file):
        """Copy associated text file if it exists"""
        base_name = os.path.splitext(img_file)[0]
        txt_file = f"{base_name}.txt"
        txt_path = os.path.join(source_folder, txt_file)
        
        if os.path.isfile(txt_path):
            output_txt_path = os.path.join(output_folder, txt_file)
            shutil.copy2(txt_path, output_txt_path)
    
    def _copy_transformed_text_file(self, source_folder, output_folder, img_file, suffix):
        """Copy and rename text file for transformed image"""
        base_name = os.path.splitext(img_file)[0]
        txt_file = f"{base_name}.txt"
        txt_path = os.path.join(source_folder, txt_file)
        
        if os.path.isfile(txt_path):
            new_txt_name = f"{base_name}{suffix}.txt"
            new_txt_path = os.path.join(output_folder, new_txt_name)
            shutil.copy2(txt_path, new_txt_path)
    
    def _copy_json_files(self, source_folder, output_folder):
        """Copy JSON files from source to output"""
        json_files = [f for f in os.listdir(source_folder) if f.endswith('.json')]
        for json_file in json_files:
            src_path = os.path.join(source_folder, json_file)
            dst_path = os.path.join(output_folder, json_file)
            try:
                shutil.copy2(src_path, dst_path)
            except Exception as e:
                print(f"Error copying JSON file: {str(e)}")
    
    def _create_augmented_json(self, input_folder, output_folder, transform_list):
        """Create updated JSON file with all image variants"""
        json_files = [f for f in os.listdir(input_folder) if f.endswith('.json')]
        if not json_files:
            return
        
        try:
            # Load original JSON
            json_path = os.path.join(input_folder, json_files[0])
            with open(json_path, 'r', encoding='utf-8') as f:
                original_json = json.load(f)
            
            # Create new JSON with all variants
            new_json_data = []
            
            for item in original_json:
                if 'fileName' in item and 'description' in item:
                    original_filename = item['fileName']
                    base_name = os.path.splitext(original_filename)[0]
                    img_ext = os.path.splitext(original_filename)[1]
                    description = item['description']
                    
                    # Always add original (it exists in both cases)
                    new_json_data.append({
                        'fileName': original_filename,
                        'description': description
                    })
                    
                    # Add transformed versions
                    for transform_key, transform_name in transform_list:
                        if transform_key == 'flip':
                            suffix = "_flipHor"
                        elif transform_key == 'rot90l':
                            suffix = "_rotLeft"
                        elif transform_key == 'rot90r':
                            suffix = "_rotRight"
                        elif transform_key == 'rot180':
                            suffix = "_flipVert"
                        elif transform_key == 'duplicate':
                            suffix = "_dup"
                        
                        new_filename = f"{base_name}{suffix}{img_ext}"
                        new_json_data.append({
                            'fileName': new_filename,
                            'description': description
                        })
            
            # Save updated JSON
            if input_folder == output_folder:
                # Same folder - update the original JSON file
                new_json_path = json_path
            else:
                # Different folder - create new augmented JSON
                base_json_name = os.path.splitext(json_files[0])[0]
                new_json_path = os.path.join(output_folder, f"{base_json_name}_augmented.json")
                
            with open(new_json_path, 'w', encoding='utf-8') as f:
                json.dump(new_json_data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            print(f"Error processing JSON file: {str(e)}")
    
    def _check_rename_conflicts(self, folder_path, prefix, image_files, num_digits, scramble_chars=None):
        """Check if the new filenames would conflict with existing files"""
        conflicts = []
        existing_files = set(os.listdir(folder_path))
        
        for i, old_filename in enumerate(image_files):
            file_extension = os.path.splitext(old_filename)[1]
            
            if scramble_chars:
                # Include scramble character
                scramble_char = scramble_chars[i] if i < len(scramble_chars) else 'z'
                new_filename = f"{prefix}_{scramble_char}{str(i + 1).zfill(num_digits)}{file_extension}"
            else:
                # Standard numbering
                new_filename = f"{prefix}_{str(i + 1).zfill(num_digits)}{file_extension}"
            
            # Skip if this would be renaming to itself
            if new_filename == old_filename:
                continue
                
            if new_filename in existing_files:
                conflicts.append(new_filename)
        
        return conflicts
    
    def _update_json_with_new_names(self, folder_path, old_to_new_mapping):
        """Update JSON files with the new filenames"""
        json_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]
        
        for json_filename in json_files:
            try:
                json_path = os.path.join(folder_path, json_filename)
                
                # Load JSON
                with open(json_path, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
                
                # Update filenames in JSON
                updated_data = []
                for item in json_data:
                    if isinstance(item, dict) and 'fileName' in item:
                        old_name = item['fileName']
                        if old_name in old_to_new_mapping:
                            item['fileName'] = old_to_new_mapping[old_name]
                        updated_data.append(item)
                    else:
                        updated_data.append(item)
                
                # Save updated JSON
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(updated_data, f, indent=2, ensure_ascii=False)
                    
            except Exception as e:
                print(f"Error updating JSON file {json_filename}: {str(e)}")