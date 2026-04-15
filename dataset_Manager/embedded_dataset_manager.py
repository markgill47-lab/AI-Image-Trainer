#!/usr/bin/env python3
"""
Embedded Dataset Manager Widget
-------------------------------
A widget that can be embedded into the main training app's Dataset tab.
Provides all the functionality of the standalone image gallery app.
"""

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QTabWidget
from PyQt6.QtGui import QAction

# Import our custom modules (relative imports for when used as a package)
from .ui_components_simple import GalleryTab, UtilsTab
from .event_handlers_simple import EventHandlers
from .data_manager import DataManager
from .image_processor import ImageProcessor


class EmbeddedDatasetManager(QWidget):
    """
    Dataset manager widget that can be embedded in other applications.
    Provides image gallery, description editing, and image processing utilities.
    """

    def __init__(self, parent=None, status_callback=None):
        """
        Initialize the embedded dataset manager.

        Args:
            parent: Parent widget
            status_callback: Optional function to call with status messages
        """
        super().__init__(parent)

        # Store reference to parent for status updates
        self.parent_app = parent
        self.status_callback = status_callback

        # Initialize core components
        self.data_manager = DataManager()
        self.image_processor = ImageProcessor(self)
        self.current_image_index = -1

        # Clipboard for description copying
        self.description_clipboard = ""

        # Initialize event handlers (after other components)
        self.event_handlers = EventHandlers(self)

        # Set up UI
        self.setup_ui()
        self.setup_shortcuts()

    def setup_ui(self):
        """Set up the widget UI structure"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Create tab widget for Gallery and Utils
        self.tab_widget = QTabWidget()

        # Create tabs (pass self so they can access the manager)
        self.gallery_tab = GalleryTab(self)
        self.utils_tab = UtilsTab(self)

        self.tab_widget.addTab(self.gallery_tab, "Gallery")
        self.tab_widget.addTab(self.utils_tab, "Utils")

        # Initially disable Utils tab until images are loaded
        self.tab_widget.setTabEnabled(1, False)

        layout.addWidget(self.tab_widget)

        # Set initial status
        self.set_status("Select a folder with images to begin")

    def setup_shortcuts(self):
        """Set up keyboard shortcuts"""
        # Force refresh shortcut
        refresh_action = QAction(self)
        refresh_action.setShortcut("F5")
        refresh_action.triggered.connect(self.event_handlers.refresh_gallery)
        self.addAction(refresh_action)

        # Copy/Paste shortcuts
        copy_action = QAction(self)
        copy_action.setShortcut("Ctrl+Shift+C")
        copy_action.triggered.connect(self.event_handlers.copy_description)
        self.addAction(copy_action)

        paste_action = QAction(self)
        paste_action.setShortcut("Ctrl+Shift+V")
        paste_action.triggered.connect(self.event_handlers.paste_description)
        self.addAction(paste_action)

    def set_status(self, message):
        """Set status message - forwards to parent if callback provided"""
        if self.status_callback:
            self.status_callback(message)
        elif self.parent_app and hasattr(self.parent_app, 'update_status'):
            self.parent_app.update_status(message)

    def load_folder(self, folder_path):
        """
        Load images from a folder programmatically.

        Args:
            folder_path: Path to folder containing images
        """
        if folder_path and folder_path.strip():
            # Simulate selecting a folder
            success, message, images_data = self.data_manager.load_images_from_folder(folder_path)

            if success:
                self.event_handlers.populate_table(images_data)

                # Select first image if available
                if images_data:
                    self.gallery_tab.table.selectRow(0)

                # Enable Utils tab
                self.tab_widget.setTabEnabled(1, True)

                self.set_status(f"{message}")
                return True
            else:
                self.set_status(f"Error: {message}")
                return False
        return False

    def get_current_folder(self):
        """Get the currently loaded folder path"""
        return self.data_manager.current_folder

    def get_image_count(self):
        """Get the number of loaded images"""
        return len(self.data_manager.images_data)

    def get_images_with_descriptions(self):
        """Get count of images that have descriptions"""
        count = 0
        for img_data in self.data_manager.images_data:
            if img_data.get('description', '').strip():
                count += 1
        return count
