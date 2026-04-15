"""
Dataset Manager Package
-----------------------
Provides tools for managing image datasets and editing descriptions.
"""

from .embedded_dataset_manager import EmbeddedDatasetManager
from .data_manager import DataManager
from .image_processor import ImageProcessor

__all__ = ['EmbeddedDatasetManager', 'DataManager', 'ImageProcessor']
