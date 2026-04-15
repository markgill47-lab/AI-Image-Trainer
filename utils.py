#!/usr/bin/env python3
"""
Utility functions for Diffusion-Pipe Training GUI (PyQt6 Version)
"""

import os
import datetime
import logging
from PyQt6.QtWidgets import (
    QLabel, QLineEdit, QPushButton, QHBoxLayout, QVBoxLayout, 
    QGridLayout, QGroupBox, QTextEdit, QFileDialog, QMessageBox,
    QProgressBar, QDialog, QDialogButtonBox, QApplication,
    QCheckBox, QComboBox, QSpinBox, QDoubleSpinBox, QFrame,
    QScrollArea, QWidget, QSplitter
)
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal, QObject
from PyQt6.QtGui import QFont, QPixmap, QPalette, QIcon

logger = logging.getLogger('DiffusionGUI')


def create_labeled_entry(parent, label_text, initial_value="", browse_callback=None, browse_text="Browse"):
    """
    Create a labeled entry with optional browse button for PyQt6
    
    Args:
        parent: Parent widget
        label_text: Label text
        initial_value: Initial value for the entry
        browse_callback: Command to execute when browse button is clicked
        browse_text: Text for the browse button
        
    Returns:
        tuple: (layout, label, entry, button) widgets
    """
    layout = QHBoxLayout()
    
    label = QLabel(label_text)
    layout.addWidget(label)
    
    entry = QLineEdit(initial_value)
    layout.addWidget(entry, 1)  # Stretch factor to expand
    
    button = None
    if browse_callback:
        button = QPushButton(browse_text)
        button.clicked.connect(browse_callback)
        layout.addWidget(button)
    
    return layout, label, entry, button


def create_group_box(title, layout_type="vertical"):
    """
    Create a group box with the specified layout
    
    Args:
        title: Group box title
        layout_type: "vertical", "horizontal", or "grid"
        
    Returns:
        tuple: (group_box, layout)
    """
    group_box = QGroupBox(title)
    
    if layout_type == "vertical":
        layout = QVBoxLayout()
    elif layout_type == "horizontal":
        layout = QHBoxLayout()
    elif layout_type == "grid":
        layout = QGridLayout()
    else:
        raise ValueError(f"Invalid layout type: {layout_type}")
    
    group_box.setLayout(layout)
    return group_box, layout


def create_text_area(readonly=True, font_family="Consolas", font_size=10):
    """
    Create a text area widget
    
    Args:
        readonly: Whether the widget should be read-only
        font_family: Font family name
        font_size: Font size
        
    Returns:
        QTextEdit: The created text area
    """
    text_area = QTextEdit()
    text_area.setReadOnly(readonly)
    
    # Set font
    font = QFont(font_family, font_size)
    text_area.setFont(font)
    
    return text_area


def create_scroll_area(widget):
    """
    Create a scroll area containing the given widget
    
    Args:
        widget: Widget to put in the scroll area
        
    Returns:
        QScrollArea: The scroll area
    """
    scroll_area = QScrollArea()
    scroll_area.setWidget(widget)
    scroll_area.setWidgetResizable(True)
    scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
    scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
    
    return scroll_area


def setup_window_geometry(window, width=1200, height=900, center=True):
    """
    Setup window geometry and optionally center on screen
    
    Args:
        window: Window to configure
        width: Window width
        height: Window height
        center: Whether to center the window on screen
    """
    window.resize(width, height)
    
    if center:
        try:
            # Get the primary screen
            screen = QApplication.primaryScreen()
            if screen:
                screen_geometry = screen.availableGeometry()
                window_geometry = window.frameGeometry()
                center_point = screen_geometry.center()
                window_geometry.moveCenter(center_point)
                window.move(window_geometry.topLeft())
        except Exception as e:
            logger.debug(f"Could not center window: {e}")


def apply_dark_theme(app):
    """
    Apply a dark theme to the application
    
    Args:
        app: QApplication instance
    """
    app.setStyle('Fusion')
    
    # Create dark palette
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, Qt.GlobalColor.black)
    palette.setColor(QPalette.ColorRole.WindowText, Qt.GlobalColor.white)
    palette.setColor(QPalette.ColorRole.Base, Qt.GlobalColor.black)
    palette.setColor(QPalette.ColorRole.AlternateBase, Qt.GlobalColor.darkGray)
    palette.setColor(QPalette.ColorRole.ToolTipBase, Qt.GlobalColor.black)
    palette.setColor(QPalette.ColorRole.ToolTipText, Qt.GlobalColor.white)
    palette.setColor(QPalette.ColorRole.Text, Qt.GlobalColor.white)
    palette.setColor(QPalette.ColorRole.Button, Qt.GlobalColor.darkGray)
    palette.setColor(QPalette.ColorRole.ButtonText, Qt.GlobalColor.white)
    palette.setColor(QPalette.ColorRole.BrightText, Qt.GlobalColor.red)
    palette.setColor(QPalette.ColorRole.Link, Qt.GlobalColor.blue)
    palette.setColor(QPalette.ColorRole.Highlight, Qt.GlobalColor.blue)
    palette.setColor(QPalette.ColorRole.HighlightedText, Qt.GlobalColor.black)
    
    app.setPalette(palette)


def show_info_dialog(parent, title, message, details=None):
    """
    Show an information dialog with optional details
    
    Args:
        parent: Parent widget
        title: Dialog title
        message: Main message
        details: Optional detailed information
        
    Returns:
        int: Dialog result
    """
    msg_box = QMessageBox(parent)
    msg_box.setIcon(QMessageBox.Icon.Information)
    msg_box.setWindowTitle(title)
    msg_box.setText(message)
    
    if details:
        msg_box.setDetailedText(details)
    
    return msg_box.exec()


def show_error_dialog(parent, title, message, details=None):
    """
    Show an error dialog with optional details
    
    Args:
        parent: Parent widget
        title: Dialog title
        message: Main message
        details: Optional detailed information
        
    Returns:
        int: Dialog result
    """
    msg_box = QMessageBox(parent)
    msg_box.setIcon(QMessageBox.Icon.Critical)
    msg_box.setWindowTitle(title)
    msg_box.setText(message)
    
    if details:
        msg_box.setDetailedText(details)
    
    return msg_box.exec()


def show_question_dialog(parent, title, message, default_yes=True):
    """
    Show a yes/no question dialog
    
    Args:
        parent: Parent widget
        title: Dialog title
        message: Question message
        default_yes: Whether "Yes" should be the default button
        
    Returns:
        bool: True if Yes was clicked, False otherwise
    """
    msg_box = QMessageBox(parent)
    msg_box.setIcon(QMessageBox.Icon.Question)
    msg_box.setWindowTitle(title)
    msg_box.setText(message)
    msg_box.setStandardButtons(
        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
    )
    
    if default_yes:
        msg_box.setDefaultButton(QMessageBox.StandardButton.Yes)
    else:
        msg_box.setDefaultButton(QMessageBox.StandardButton.No)
    
    result = msg_box.exec()
    return result == QMessageBox.StandardButton.Yes


class ProgressDialog(QDialog):
    """
    A progress dialog for long-running operations
    """
    
    def __init__(self, parent=None, title="Progress", message="Processing..."):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        self.resize(400, 120)
        
        layout = QVBoxLayout(self)
        
        # Message label
        self.message_label = QLabel(message)
        layout.addWidget(self.message_label)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)
        
        # Cancel button
        self.button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)
        
        # Setup progress bar
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        
        self.cancelled = False
        
    def set_progress(self, value, message=None):
        """Update progress value and optionally the message"""
        self.progress_bar.setValue(value)
        if message:
            self.message_label.setText(message)
        QApplication.processEvents()  # Update UI
        
    def is_cancelled(self):
        """Check if the dialog was cancelled"""
        return self.cancelled
        
    def reject(self):
        """Handle cancel button"""
        self.cancelled = True
        super().reject()


class WorkerThread(QThread):
    """
    Generic worker thread for background operations
    """
    
    # Signals
    progress_updated = pyqtSignal(int, str)  # progress, message
    finished_successfully = pyqtSignal(object)  # result
    error_occurred = pyqtSignal(str)  # error message
    
    def __init__(self, work_function, *args, **kwargs):
        super().__init__()
        self.work_function = work_function
        self.args = args
        self.kwargs = kwargs
        self.result = None
        self.error = None
        
    def run(self):
        """Run the work function in the background"""
        try:
            # Add progress callback to kwargs if the function supports it
            if 'progress_callback' in self.kwargs:
                self.kwargs['progress_callback'] = self.emit_progress
            
            self.result = self.work_function(*self.args, **self.kwargs)
            self.finished_successfully.emit(self.result)
            
        except Exception as e:
            self.error = str(e)
            self.error_occurred.emit(self.error)
            logger.error(f"Worker thread error: {e}", exc_info=True)
    
    def emit_progress(self, current, total, message=""):
        """Emit progress signal"""
        if total > 0:
            percentage = int((current / total) * 100)
        else:
            percentage = 0
        self.progress_updated.emit(percentage, message)


def formatted_timestamp():
    """
    Get a formatted timestamp
    
    Returns:
        str: Formatted timestamp
    """
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def format_time(seconds):
    """
    Format time in seconds to human-readable format
    
    Args:
        seconds: Time in seconds
        
    Returns:
        str: Formatted time
    """
    if seconds < 60:
        return f"{seconds:.1f} seconds"
    elif seconds < 3600:
        return f"{seconds/60:.1f} minutes"
    else:
        return f"{seconds/3600:.1f} hours"


def format_bytes(bytes_value):
    """
    Format bytes to human-readable format
    
    Args:
        bytes_value: Number of bytes
        
    Returns:
        str: Formatted byte string
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.1f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.1f} PB"


def create_separator(orientation="horizontal"):
    """
    Create a separator line
    
    Args:
        orientation: "horizontal" or "vertical"
        
    Returns:
        QFrame: Separator frame
    """
    separator = QFrame()
    if orientation == "horizontal":
        separator.setFrameShape(QFrame.Shape.HLine)
    else:
        separator.setFrameShape(QFrame.Shape.VLine)
    separator.setFrameShadow(QFrame.Shadow.Sunken)
    return separator


def add_stretch_to_layout(layout, stretch_factor=1):
    """
    Add stretch to a layout
    
    Args:
        layout: Layout to add stretch to
        stretch_factor: Stretch factor
    """
    if isinstance(layout, (QHBoxLayout, QVBoxLayout)):
        layout.addStretch(stretch_factor)


def create_config_widget(key, value, value_changed_callback):
    """
    Create an appropriate widget for a configuration value
    
    Args:
        key: Configuration key name
        value: Configuration value
        value_changed_callback: Callback for when value changes
        
    Returns:
        QWidget: Appropriate widget for the value type
    """
    if isinstance(value, bool):
        widget = QCheckBox()
        widget.setChecked(value)
        widget.stateChanged.connect(
            lambda state: value_changed_callback(key, state == Qt.CheckState.Checked.value)
        )
        
    elif isinstance(value, int):
        widget = QSpinBox()
        widget.setRange(-999999, 999999)
        widget.setValue(value)
        widget.valueChanged.connect(lambda val: value_changed_callback(key, val))
        
    elif isinstance(value, float):
        widget = QDoubleSpinBox()
        widget.setDecimals(6)
        widget.setRange(-999999.0, 999999.0)
        widget.setValue(value)
        widget.valueChanged.connect(lambda val: value_changed_callback(key, val))
        
    elif isinstance(value, list):
        widget = QLineEdit()
        widget.setText(str(value))
        widget.textChanged.connect(lambda text: value_changed_callback(key, text))
        
    else:
        widget = QLineEdit()
        widget.setText(str(value))
        widget.textChanged.connect(lambda text: value_changed_callback(key, text))
    
    return widget


def safe_disconnect(signal):
    """
    Safely disconnect all slots from a signal
    
    Args:
        signal: PyQt signal to disconnect
    """
    try:
        signal.disconnect()
    except TypeError:
        # Signal was not connected
        pass


def set_widget_enabled_recursive(widget, enabled):
    """
    Recursively enable/disable a widget and all its children
    
    Args:
        widget: Widget to enable/disable
        enabled: True to enable, False to disable
    """
    widget.setEnabled(enabled)
    for child in widget.findChildren(QWidget):
        child.setEnabled(enabled)


def clear_layout(layout):
    """
    Clear all widgets from a layout
    
    Args:
        layout: Layout to clear
    """
    while layout.count():
        child = layout.takeAt(0)
        if child.widget():
            child.widget().deleteLater()


def find_widgets_by_type(parent, widget_type):
    """
    Find all widgets of a specific type within a parent widget
    
    Args:
        parent: Parent widget to search in
        widget_type: Type of widget to find
        
    Returns:
        list: List of widgets of the specified type
    """
    return parent.findChildren(widget_type)


def create_splitter(orientation="horizontal", widgets=None):
    """
    Create a splitter with optional widgets
    
    Args:
        orientation: "horizontal" or "vertical"
        widgets: List of widgets to add to the splitter
        
    Returns:
        QSplitter: Created splitter
    """
    if orientation == "horizontal":
        splitter = QSplitter(Qt.Orientation.Horizontal)
    else:
        splitter = QSplitter(Qt.Orientation.Vertical)
    
    if widgets:
        for widget in widgets:
            splitter.addWidget(widget)
    
    return splitter


def validate_file_path(file_path, must_exist=True, extension=None):
    """
    Validate a file path
    
    Args:
        file_path: Path to validate
        must_exist: Whether the file must exist
        extension: Required file extension (without dot)
        
    Returns:
        tuple: (is_valid, error_message)
    """
    if not file_path:
        return False, "File path is empty"
    
    if must_exist and not os.path.exists(file_path):
        return False, f"File does not exist: {file_path}"
    
    if extension:
        if not file_path.lower().endswith(f".{extension.lower()}"):
            return False, f"File must have .{extension} extension"
    
    if must_exist and not os.path.isfile(file_path):
        return False, f"Path is not a file: {file_path}"
    
    return True, ""


def validate_directory_path(dir_path, must_exist=True, must_be_writable=False):
    """
    Validate a directory path
    
    Args:
        dir_path: Directory path to validate
        must_exist: Whether the directory must exist
        must_be_writable: Whether the directory must be writable
        
    Returns:
        tuple: (is_valid, error_message)
    """
    if not dir_path:
        return False, "Directory path is empty"
    
    if must_exist and not os.path.exists(dir_path):
        return False, f"Directory does not exist: {dir_path}"
    
    if must_exist and not os.path.isdir(dir_path):
        return False, f"Path is not a directory: {dir_path}"
    
    if must_be_writable and must_exist:
        if not os.access(dir_path, os.W_OK):
            return False, f"Directory is not writable: {dir_path}"
    
    return True, ""


class DelayedCallback:
    """
    Execute a callback after a delay, cancelling previous calls
    Useful for text input validation or auto-save functionality
    """
    
    def __init__(self, callback, delay_ms=500):
        self.callback = callback
        self.timer = QTimer()
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self._execute_callback)
        self.delay_ms = delay_ms
        self.pending_args = None
        self.pending_kwargs = None
    
    def __call__(self, *args, **kwargs):
        """Call the delayed callback"""
        self.pending_args = args
        self.pending_kwargs = kwargs
        self.timer.start(self.delay_ms)
    
    def _execute_callback(self):
        """Execute the actual callback"""
        if self.callback and self.pending_args is not None:
            self.callback(*self.pending_args, **self.pending_kwargs)
    
    def cancel(self):
        """Cancel any pending callback"""
        self.timer.stop()
        self.pending_args = None
        self.pending_kwargs = None