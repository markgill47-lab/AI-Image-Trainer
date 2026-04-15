#!/usr/bin/env python3
"""
Data management utilities for the Image Gallery application
"""

import os
import json
from typing import List, Dict, Tuple, Optional


class DataManager:
    """Manages image data, descriptions, and file operations"""
    
    def __init__(self):
        self.images_data: List[Dict] = []
        self.current_folder: Optional[str] = None
        
    @staticmethod
    def get_image_files(folder_path: str) -> List[str]:
        """Get all supported image files from a folder"""
        image_extensions = ('.jpg', '.jpeg', '.png', '.webp')
        try:
            return [f for f in os.listdir(folder_path) 
                    if os.path.isfile(os.path.join(folder_path, f)) 
                    and f.lower().endswith(image_extensions)]
        except (OSError, FileNotFoundError):
            return []
    
    def load_images_from_folder(self, folder_path: str) -> Tuple[bool, str, List[Dict]]:
        """
        Load images and their descriptions from a folder
        
        Returns:
            Tuple of (success, message, images_data)
        """
        try:
            if not os.path.exists(folder_path):
                return False, "Folder does not exist", []
            
            if not os.path.isdir(folder_path):
                return False, "Path is not a directory", []
            
            # Get image files
            image_files = self.get_image_files(folder_path)
            if not image_files:
                return False, "No supported image files found in folder", []
            
            # Sort files for consistent ordering
            image_files.sort()

            # Load existing descriptions and quality ratings from JSON file if available
            json_descriptions, quality_ratings = self._load_descriptions_from_json(folder_path)

            # Create image data list
            images_data = []
            for filename in image_files:
                img_path = os.path.join(folder_path, filename)

                # Try to get description from various sources
                description = self._get_description_for_image(folder_path, filename, json_descriptions)

                img_data = {
                    'filename': filename,
                    'path': img_path,
                    'description': description,
                    'row_index': len(images_data)  # Will be updated when populating table
                }

                # Add quality rating if available
                if filename in quality_ratings:
                    img_data['quality'] = quality_ratings[filename]

                images_data.append(img_data)
            
            # Store the data
            self.images_data = images_data
            self.current_folder = folder_path
            
            message = f"Loaded {len(images_data)} images from {os.path.basename(folder_path)}"
            return True, message, images_data
            
        except Exception as e:
            return False, f"Error loading images: {str(e)}", []
    
    def _load_descriptions_from_json(self, folder_path: str) -> Tuple[Dict[str, str], Dict[str, str]]:
        """Load descriptions and quality ratings from JSON file if it exists"""
        json_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]

        for json_file in json_files:
            try:
                json_path = os.path.join(folder_path, json_file)
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # Handle different JSON formats
                descriptions = {}
                quality_ratings = {}
                if isinstance(data, list):
                    # List format: [{"fileName": "...", "description": "...", "quality": "..."}, ...]
                    for item in data:
                        if isinstance(item, dict) and 'fileName' in item and 'description' in item:
                            descriptions[item['fileName']] = item['description']
                            if 'quality' in item:
                                quality_ratings[item['fileName']] = item['quality']
                elif isinstance(data, dict):
                    # Dict format: {"filename1.jpg": "description1", ...}
                    descriptions = data

                if descriptions:
                    return descriptions, quality_ratings
            except Exception as e:
                print(f"Error loading JSON file {json_file}: {str(e)}")
                continue

        return {}, {}
    
    def _get_description_for_image(self, folder_path: str, filename: str, json_descriptions: Dict[str, str]) -> str:
        """Get description for an image from various sources"""
        # Priority 1: JSON descriptions
        if filename in json_descriptions:
            return json_descriptions[filename]
        
        # Priority 2: Individual .txt file
        txt_filename = os.path.splitext(filename)[0] + '.txt'
        txt_path = os.path.join(folder_path, txt_filename)
        
        if os.path.exists(txt_path):
            try:
                with open(txt_path, 'r', encoding='utf-8') as f:
                    return f.read().strip()
            except Exception as e:
                print(f"Error reading {txt_filename}: {str(e)}")
        
        # Default: empty description
        return ""
    
    def update_description(self, index: int, description: str) -> bool:
        """Update description for an image"""
        if 0 <= index < len(self.images_data):
            self.images_data[index]['description'] = description
            return True
        return False

    def update_quality(self, index: int, quality: str) -> bool:
        """Update quality rating for an image (good/ok/bad)"""
        if 0 <= index < len(self.images_data):
            self.images_data[index]['quality'] = quality
            return True
        return False

    def get_quality(self, index: int) -> Optional[str]:
        """Get quality rating for an image"""
        if 0 <= index < len(self.images_data):
            return self.images_data[index].get('quality')
        return None
    
    def get_image_data(self, index: int) -> Optional[Dict]:
        """Get image data by index"""
        if 0 <= index < len(self.images_data):
            return self.images_data[index]
        return None
    
    def find_image_by_row_index(self, row: int) -> Tuple[int, Optional[Dict]]:
        """Find image data by table row index"""
        for i, img_data in enumerate(self.images_data):
            if img_data.get('row_index') == row:
                return i, img_data
        return -1, None
    
    def remove_image(self, index: int) -> bool:
        """Remove image from the list"""
        if 0 <= index < len(self.images_data):
            del self.images_data[index]
            # Update row indices for remaining images
            for i, img_data in enumerate(self.images_data):
                img_data['row_index'] = i
            return True
        return False
    
    def append_keyword_to_all(self, keyword: str) -> int:
        """Append keyword to all descriptions"""
        if not keyword.strip():
            return 0
        
        updated_count = 0
        keyword = keyword.strip()
        
        for img_data in self.images_data:
            current_desc = img_data['description'].strip()
            if current_desc:
                # Add keyword with proper spacing
                if not current_desc.endswith(', ') and not current_desc.endswith(' '):
                    img_data['description'] = f"{current_desc}, {keyword}"
                else:
                    img_data['description'] = f"{current_desc}{keyword}"
            else:
                # If description is empty, just set the keyword
                img_data['description'] = keyword
            
            updated_count += 1
        
        return updated_count
    
    def save_descriptions(self) -> Tuple[bool, str]:
        """Save descriptions to both individual .txt files and consolidated JSON"""
        if not self.current_folder or not self.images_data:
            return False, "No data to save"
        
        try:
            saved_txt_files = 0
            errors = []
            
            # Save individual .txt files
            for img_data in self.images_data:
                filename = img_data['filename']
                description = img_data['description']
                
                # Create .txt filename
                txt_filename = os.path.splitext(filename)[0] + '.txt'
                txt_path = os.path.join(self.current_folder, txt_filename)
                
                try:
                    # Only save if description is not empty
                    if description.strip():
                        with open(txt_path, 'w', encoding='utf-8') as f:
                            f.write(description)
                        saved_txt_files += 1
                    else:
                        # Remove .txt file if description is empty
                        if os.path.exists(txt_path):
                            os.remove(txt_path)
                except Exception as e:
                    errors.append(f"Error saving {txt_filename}: {str(e)}")
            
            # Save consolidated JSON file
            json_filename = f"{os.path.basename(self.current_folder)}_descriptions.json"
            json_path = os.path.join(self.current_folder, json_filename)
            
            json_data = []
            for img_data in self.images_data:
                entry = {
                    'fileName': img_data['filename'],
                    'description': img_data['description']
                }
                # Include quality if present
                if img_data.get('quality'):
                    entry['quality'] = img_data['quality']
                json_data.append(entry)
            
            try:
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(json_data, f, indent=2, ensure_ascii=False)
            except Exception as e:
                errors.append(f"Error saving JSON file: {str(e)}")
            
            # Create success message
            message = f"Saved {saved_txt_files} description files and consolidated JSON"
            if errors:
                message += f" (with {len(errors)} errors)"
                # Log errors for debugging
                for error in errors:
                    print(error)
            
            return True, message
            
        except Exception as e:
            return False, f"Save failed: {str(e)}"
    
    def get_images_with_descriptions(self) -> List[Dict]:
        """Get only images that have descriptions"""
        return [img for img in self.images_data if img['description'].strip()]
    
    def get_empty_description_count(self) -> int:
        """Get count of images without descriptions"""
        return len([img for img in self.images_data if not img['description'].strip()])
    
    def clear_all_descriptions(self) -> int:
        """Clear all descriptions (useful for testing)"""
        count = 0
        for img_data in self.images_data:
            if img_data['description'].strip():
                img_data['description'] = ""
                count += 1
        return count
    
    def get_stats(self) -> Dict[str, int]:
        """Get statistics about the current dataset"""
        total_images = len(self.images_data)
        with_descriptions = len(self.get_images_with_descriptions())
        without_descriptions = self.get_empty_description_count()
        
        return {
            'total_images': total_images,
            'with_descriptions': with_descriptions,
            'without_descriptions': without_descriptions
        }