#!/usr/bin/env python3
"""
Flux Training GUI - Main Entry Point
Simplified version for Kohya sd-scripts training
"""

import sys
import os
import logging

# Fix for Wayland/X11 display issues on Linux/WSL2
if os.name == 'posix':
    os.environ.setdefault('QT_QPA_PLATFORM', 'xcb')
    if not os.environ.get('DISPLAY'):
        os.environ['DISPLAY'] = ':0'

# Import PyQt6
try:
    from PyQt6.QtWidgets import QApplication, QMessageBox
    from PyQt6.QtCore import Qt
    from PyQt6.QtGui import QIcon
    PYQT6_AVAILABLE = True
except ImportError as e:
    print(f"Error importing PyQt6: {e}")
    print("Please install PyQt6: pip install PyQt6")
    sys.exit(1)

# Import our UI manager
try:
    from ui_manager import UIManager
except Exception as e:
    import traceback
    print(f"Error importing UI manager: {e}")
    traceback.print_exc()
    print("\nMake sure all required files are in the same directory")
    sys.exit(1)

# Initialize logger
logging.basicConfig(
    filename='flux_training_gui.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger('FluxTrainingGUI')


def setup_application_style(app):
    """Set up the application style and theme"""
    # Use Fusion style for modern appearance
    app.setStyle('Fusion')
    
    # Set application properties
    app.setApplicationName("Flux Training GUI")
    app.setApplicationVersion("1.0")
    app.setOrganizationName("VizLab")
    app.setOrganizationDomain("scsu.edu")


def main():
    """Main entry point for the application"""
    try:
        logger.info("Starting Flux Training GUI")
        
        # Create the PyQt application
        app = QApplication(sys.argv)
        
        # Set up application style and properties
        setup_application_style(app)
        
        # Enable high DPI scaling (for modern displays)
        try:
            app.setAttribute(Qt.ApplicationAttribute.AA_EnableHighDpiScaling, True)
            app.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps, True)
        except Exception as e:
            logger.info(f"Could not enable high DPI scaling: {str(e)}")
        
        # Set application icon (if available)
        try:
            if os.path.exists('icon.png'):
                app.setWindowIcon(QIcon('icon.png'))
        except Exception as e:
            logger.info(f"Could not set application icon: {str(e)}")
        
        # Create and show the main window
        logger.info("Initializing UI Manager")
        window = UIManager()
        
        # Show the window
        window.show()
        
        # Center the window on screen
        try:
            screen = app.screens()[0]
            screen_geometry = screen.availableGeometry()
            window_geometry = window.frameGeometry()
            center_point = screen_geometry.center()
            window_geometry.moveCenter(center_point)
            window.move(window_geometry.topLeft())
        except Exception as e:
            logger.info(f"Could not center window: {str(e)}")
        
        logger.info("Entering main event loop")
        
        # Start the event loop
        exit_code = app.exec()
        
        logger.info(f"Application closed with exit code: {exit_code}")
        return exit_code
        
    except Exception as e:
        error_msg = f"Unhandled exception in main: {str(e)}"
        
        # Log the error
        if 'logger' in locals():
            logger.critical(error_msg, exc_info=True)
        else:
            print(error_msg)
        
        # Show error dialog if possible
        if PYQT6_AVAILABLE and 'app' in locals():
            try:
                msg_box = QMessageBox()
                msg_box.setIcon(QMessageBox.Icon.Critical)
                msg_box.setWindowTitle("Critical Error")
                msg_box.setText("A critical error occurred:")
                msg_box.setDetailedText(f"{str(e)}\n\nCheck flux_training_gui.log for details.")
                msg_box.exec()
            except Exception as dialog_error:
                print(f"Could not show error dialog: {dialog_error}")
        else:
            print(f"Critical error: {str(e)}")
        
        return 1


if __name__ == "__main__":
    sys.exit(main())
