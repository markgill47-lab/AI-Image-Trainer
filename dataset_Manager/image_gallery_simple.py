#!/usr/bin/env python3
"""
Image Gallery with Descriptions - Simplified Version
-------------------------------------------------
A streamlined tool for organizing datasets for image generating AI models.
Focus on efficient description management with copy/paste workflow.

Windows Native Version
"""

import sys
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QStatusBar, QTabWidget
)
from PyQt6.QtGui import QAction, QFont

# Import our custom modules
from ui_components_simple import GalleryTab, UtilsTab
from event_handlers_simple import EventHandlers
from data_manager import DataManager
from image_processor import ImageProcessor


class ImageGalleryApp(QMainWindow):
    """Main application window - coordinates between modules"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Gallery with Descriptions - Simplified Edition")
        self.resize(1400, 900)
        
        # Initialize core components
        self.data_manager = DataManager()
        self.image_processor = ImageProcessor(self)
        self.current_image_index = -1
        
        # Clipboard for description copying
        self.description_clipboard = ""
        
        # Initialize event handlers (after other components)
        self.event_handlers = EventHandlers(self)
        
        # Font settings
        self.current_font_size = 12
        self.default_font = QFont("Segoe UI", self.current_font_size)
        
        # Set up UI
        self.setup_ui()
        self.setup_shortcuts()
        self.update_fonts()
        
    def setup_ui(self):
        """Set up the main UI structure"""
        # Set up central widget with tabs
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        # Create status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        # Main layout for central widget
        main_layout = QVBoxLayout(self.central_widget)
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        
        # Create tabs (pass self so they can access the main app)
        self.gallery_tab = GalleryTab(self)
        self.utils_tab = UtilsTab(self)
        
        self.tab_widget.addTab(self.gallery_tab, "Gallery")
        self.tab_widget.addTab(self.utils_tab, "Utils")
        
        # Initially disable Utils tab until images are loaded
        self.tab_widget.setTabEnabled(1, False)
        
        # Set initial status
        self.set_status("Select a folder with images to begin • Simplified description workflow")
        
        # Add tab widget to main layout
        main_layout.addWidget(self.tab_widget)
        
    def setup_shortcuts(self):
        """Set up keyboard shortcuts"""
        # Font size shortcuts
        increase_font_action = QAction(self)
        increase_font_action.setShortcut("Ctrl++")
        increase_font_action.triggered.connect(self.increase_font_size)
        self.addAction(increase_font_action)
        
        decrease_font_action = QAction(self)
        decrease_font_action.setShortcut("Ctrl+-")
        decrease_font_action.triggered.connect(self.decrease_font_size)
        self.addAction(decrease_font_action)
        
        reset_font_action = QAction(self)
        reset_font_action.setShortcut("Ctrl+0")
        reset_font_action.triggered.connect(self.reset_font_size)
        self.addAction(reset_font_action)
        
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
    
    def update_fonts(self):
        """Update all fonts in the application"""
        self.default_font = QFont("Segoe UI", self.current_font_size)
        QApplication.setFont(self.default_font)
        
        # Update specific widgets if they exist
        if hasattr(self.gallery_tab, 'description_text'):
            desc_font = QFont("Segoe UI", self.current_font_size)
            self.gallery_tab.description_text.setFont(desc_font)
        
        self.set_status(f"Font size: {self.current_font_size}")
        
    def increase_font_size(self):
        """Increase font size"""
        if self.current_font_size < 24:
            self.current_font_size += 1
            self.update_fonts()
        
    def decrease_font_size(self):
        """Decrease font size"""
        if self.current_font_size > 8:
            self.current_font_size -= 1
            self.update_fonts()
            
    def reset_font_size(self):
        """Reset font size to default"""
        self.current_font_size = 12
        self.update_fonts()
    
    def set_status(self, message):
        """Set status bar message"""
        self.status_bar.showMessage(message)


def main():
    """Main application entry point"""
    app = QApplication(sys.argv)
    
    # Set application-wide style for Windows
    app.setStyle('Fusion')
    
    window = ImageGalleryApp()
    window.show()
    
    # Print helpful info on startup
    print("\n" + "="*60)
    print("IMAGE GALLERY - SIMPLIFIED EDITION")
    print("="*60)
    print("Keyboard shortcuts:")
    print("  F5              - Refresh gallery")
    print("  Ctrl++          - Increase font size")
    print("  Ctrl+-          - Decrease font size")
    print("  Ctrl+0          - Reset font size")
    print("  Ctrl+Shift+C    - Copy description from current image")
    print("  Ctrl+Shift+V    - Paste description to selected image(s)")
    print()
    print("Features:")
    print("  • Simple text description editing")
    print("  • Copy/paste descriptions between images")
    print("  • Multi-selection for batch paste operations")
    print("  • Image processing utilities")
    print("  • Mass rename with order scrambling")
    print("="*60 + "\n")
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
