#!/usr/bin/env python3
"""
Dialog windows for the Image Gallery application
"""

import os
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGroupBox, QRadioButton, 
    QButtonGroup, QPushButton, QLabel, QLineEdit, QFileDialog,
    QCheckBox, QMessageBox
)
from PyQt6.QtCore import Qt


class ImageFixDialog(QDialog):
    """Dialog for configuring image fixing/resizing options"""
    
    def __init__(self, parent=None, default_folder=None):
        super().__init__(parent)
        self.default_folder = default_folder
        self.result = {}
        self.setup_ui()
        
    def setup_ui(self):
        self.setWindowTitle("Fix Images")
        self.setModal(True)
        self.resize(500, 400)
        
        layout = QVBoxLayout(self)
        
        # Source folder selection
        source_group = QGroupBox("Source Folder")
        source_layout = QVBoxLayout(source_group)
        
        source_folder_layout = QHBoxLayout()
        source_folder_layout.addWidget(QLabel("Source Folder:"))
        self.source_folder_entry = QLineEdit()
        if self.default_folder:
            self.source_folder_entry.setText(self.default_folder)
        self.source_browse_btn = QPushButton("Browse...")
        self.source_browse_btn.clicked.connect(self.browse_source_folder)
        
        source_folder_layout.addWidget(self.source_folder_entry, 1)
        source_folder_layout.addWidget(self.source_browse_btn)
        source_layout.addLayout(source_folder_layout)
        
        # Output folder selection
        output_group = QGroupBox("Output Folder")
        output_layout = QVBoxLayout(output_group)
        
        # Same folder checkbox
        self.same_folder_check = QCheckBox("Save to same folder as source")
        self.same_folder_check.setChecked(True)  # Default to same folder
        self.same_folder_check.toggled.connect(self.toggle_output_folder)
        output_layout.addWidget(self.same_folder_check)
        
        # Output folder selection (initially disabled)
        output_folder_layout = QHBoxLayout()
        output_folder_layout.addWidget(QLabel("Output Folder:"))
        self.output_folder_entry = QLineEdit()
        if self.default_folder:
            self.output_folder_entry.setText(self.default_folder)
        self.output_folder_entry.setEnabled(False)
        
        self.output_browse_btn = QPushButton("Browse...")
        self.output_browse_btn.clicked.connect(self.browse_output_folder)
        self.output_browse_btn.setEnabled(False)
        
        output_folder_layout.addWidget(self.output_folder_entry, 1)
        output_folder_layout.addWidget(self.output_browse_btn)
        output_layout.addLayout(output_folder_layout)
        
        # Image size settings
        size_group = QGroupBox("Image Size")
        size_layout = QVBoxLayout(size_group)
        
        self.size_group = QButtonGroup()
        self.size_512 = QRadioButton("512x512")
        self.size_1024 = QRadioButton("1024x1024")
        self.size_2048 = QRadioButton("2048x2048")
        self.size_1024.setChecked(True)  # Default
        
        self.size_group.addButton(self.size_512)
        self.size_group.addButton(self.size_1024)
        self.size_group.addButton(self.size_2048)
        
        size_layout.addWidget(self.size_512)
        size_layout.addWidget(self.size_1024)
        size_layout.addWidget(self.size_2048)

        # Note about aspect ratio
        aspect_note = QLabel("Images will be resized to fit the target size while preserving aspect ratio.")
        aspect_note.setStyleSheet("QLabel { color: #666; font-size: 10px; font-style: italic; }")
        aspect_note.setWordWrap(True)
        size_layout.addWidget(aspect_note)

        # Buttons
        button_layout = QHBoxLayout()
        self.ok_btn = QPushButton("Process Images")
        self.cancel_btn = QPushButton("Cancel")
        
        self.ok_btn.clicked.connect(self.accept)
        self.cancel_btn.clicked.connect(self.reject)
        
        button_layout.addStretch()
        button_layout.addWidget(self.ok_btn)
        button_layout.addWidget(self.cancel_btn)
        
        # Add all groups to main layout
        layout.addWidget(source_group)
        layout.addWidget(output_group)
        layout.addWidget(size_group)
        layout.addLayout(button_layout)
    
    def toggle_output_folder(self, checked):
        """Enable/disable output folder selection based on checkbox"""
        self.output_folder_entry.setEnabled(not checked)
        self.output_browse_btn.setEnabled(not checked)
        
        if checked and self.source_folder_entry.text():
            # Set output to same as source
            self.output_folder_entry.setText(self.source_folder_entry.text())
    
    def browse_source_folder(self):
        """Browse for source folder"""
        folder = QFileDialog.getExistingDirectory(
            self, "Select Source Folder", 
            self.source_folder_entry.text() or self.default_folder or ""
        )
        if folder:
            self.source_folder_entry.setText(folder)
            # If same folder is checked, update output too
            if self.same_folder_check.isChecked():
                self.output_folder_entry.setText(folder)
    
    def browse_output_folder(self):
        """Browse for output folder"""
        folder = QFileDialog.getExistingDirectory(
            self, "Select Output Folder", 
            self.output_folder_entry.text() or self.default_folder or ""
        )
        if folder:
            self.output_folder_entry.setText(folder)
    
    def accept(self):
        """Validate and accept the dialog"""
        source_folder = self.source_folder_entry.text().strip()
        output_folder = self.output_folder_entry.text().strip()
        
        if not source_folder:
            QMessageBox.warning(self, "Missing Source", "Please select a source folder.")
            return
        
        if not os.path.exists(source_folder):
            QMessageBox.warning(self, "Invalid Source", "Source folder does not exist.")
            return
        
        if not output_folder:
            QMessageBox.warning(self, "Missing Output", "Please select an output folder.")
            return
        
        # Create output folder if it doesn't exist
        if not os.path.exists(output_folder):
            try:
                os.makedirs(output_folder)
            except Exception as e:
                QMessageBox.critical(self, "Folder Creation Error", f"Could not create output folder: {str(e)}")
                return
        
        # Get selected size
        if self.size_512.isChecked():
            size = 512
        elif self.size_1024.isChecked():
            size = 1024
        else:
            size = 2048
        
        # Store results
        self.result = {
            'source_folder': source_folder,
            'output_folder': output_folder,
            'size': size
        }

        super().accept()


class ImageDuplicateDialog(QDialog):
    """Dialog for configuring image duplication and transformation options"""
    
    def __init__(self, parent=None, default_folder=None):
        super().__init__(parent)
        self.default_folder = default_folder
        self.result = {}
        self.setup_ui()
        
    def setup_ui(self):
        self.setWindowTitle("Create Duplicates")
        self.setModal(True)
        self.resize(500, 450)
        
        layout = QVBoxLayout(self)
        
        # Input folder selection
        input_group = QGroupBox("Input Folder")
        input_layout = QVBoxLayout(input_group)
        
        input_folder_layout = QHBoxLayout()
        input_folder_layout.addWidget(QLabel("Input Folder:"))
        self.input_folder_entry = QLineEdit()
        if self.default_folder:
            self.input_folder_entry.setText(self.default_folder)
        self.input_browse_btn = QPushButton("Browse...")
        self.input_browse_btn.clicked.connect(self.browse_input_folder)
        
        input_folder_layout.addWidget(self.input_folder_entry, 1)
        input_folder_layout.addWidget(self.input_browse_btn)
        input_layout.addLayout(input_folder_layout)
        
        # Output folder selection
        output_group = QGroupBox("Output Folder")
        output_layout = QVBoxLayout(output_group)
        
        # Same folder checkbox
        self.same_folder_check = QCheckBox("Save to same folder as input")
        self.same_folder_check.setChecked(True)  # Default to same folder
        self.same_folder_check.toggled.connect(self.toggle_output_folder)
        output_layout.addWidget(self.same_folder_check)
        
        # Output folder selection (initially disabled)
        output_folder_layout = QHBoxLayout()
        output_folder_layout.addWidget(QLabel("Output Folder:"))
        self.output_folder_entry = QLineEdit()
        if self.default_folder:
            self.output_folder_entry.setText(self.default_folder)
        self.output_folder_entry.setEnabled(False)
        
        self.output_browse_btn = QPushButton("Browse...")
        self.output_browse_btn.clicked.connect(self.browse_output_folder)
        self.output_browse_btn.setEnabled(False)
        
        output_folder_layout.addWidget(self.output_folder_entry, 1)
        output_folder_layout.addWidget(self.output_browse_btn)
        output_layout.addLayout(output_folder_layout)
        
        # Transformation options
        transform_group = QGroupBox("Transformations")
        transform_layout = QVBoxLayout(transform_group)
        
        info_label = QLabel("Select transformations to apply (if none selected, simple duplicates will be created):")
        info_label.setWordWrap(True)
        info_label.setStyleSheet("QLabel { color: #666; font-size: 10px; margin-bottom: 5px; }")
        transform_layout.addWidget(info_label)
        
        self.flip_horizontal = QCheckBox("Horizontal Flip")
        self.rotate_90_left = QCheckBox("Rotate 90° Left")
        self.rotate_90_right = QCheckBox("Rotate 90° Right")
        self.rotate_180 = QCheckBox("Rotate 180°")
        
        transform_layout.addWidget(self.flip_horizontal)
        transform_layout.addWidget(self.rotate_90_left)
        transform_layout.addWidget(self.rotate_90_right)
        transform_layout.addWidget(self.rotate_180)
        
        # Buttons
        button_layout = QHBoxLayout()
        self.ok_btn = QPushButton("Create Duplicates")
        self.cancel_btn = QPushButton("Cancel")
        
        self.ok_btn.clicked.connect(self.accept)
        self.cancel_btn.clicked.connect(self.reject)
        
        button_layout.addStretch()
        button_layout.addWidget(self.ok_btn)
        button_layout.addWidget(self.cancel_btn)
        
        # Add all groups to main layout
        layout.addWidget(input_group)
        layout.addWidget(output_group)
        layout.addWidget(transform_group)
        layout.addLayout(button_layout)
    
    def toggle_output_folder(self, checked):
        """Enable/disable output folder selection based on checkbox"""
        self.output_folder_entry.setEnabled(not checked)
        self.output_browse_btn.setEnabled(not checked)
        
        if checked and self.input_folder_entry.text():
            # Set output to same as input
            self.output_folder_entry.setText(self.input_folder_entry.text())
    
    def browse_input_folder(self):
        """Browse for input folder"""
        folder = QFileDialog.getExistingDirectory(
            self, "Select Input Folder", 
            self.input_folder_entry.text() or self.default_folder or ""
        )
        if folder:
            self.input_folder_entry.setText(folder)
            # If same folder is checked, update output too
            if self.same_folder_check.isChecked():
                self.output_folder_entry.setText(folder)
    
    def browse_output_folder(self):
        """Browse for output folder"""
        folder = QFileDialog.getExistingDirectory(
            self, "Select Output Folder", 
            self.output_folder_entry.text() or self.default_folder or ""
        )
        if folder:
            self.output_folder_entry.setText(folder)
    
    def accept(self):
        """Validate and accept the dialog"""
        input_folder = self.input_folder_entry.text().strip()
        output_folder = self.output_folder_entry.text().strip()
        
        if not input_folder:
            QMessageBox.warning(self, "Missing Input", "Please select an input folder.")
            return
        
        if not os.path.exists(input_folder):
            QMessageBox.warning(self, "Invalid Input", "Input folder does not exist.")
            return
        
        if not output_folder:
            QMessageBox.warning(self, "Missing Output", "Please select an output folder.")
            return
        
        # Create output folder if it doesn't exist (and it's different from input)
        if input_folder != output_folder and not os.path.exists(output_folder):
            try:
                os.makedirs(output_folder)
            except Exception as e:
                QMessageBox.critical(self, "Folder Creation Error", f"Could not create output folder: {str(e)}")
                return
        
        # Store results
        self.result = {
            'input_folder': input_folder,
            'output_folder': output_folder,
            'flip_horizontal': self.flip_horizontal.isChecked(),
            'rotate_90_left': self.rotate_90_left.isChecked(),
            'rotate_90_right': self.rotate_90_right.isChecked(),
            'rotate_180': self.rotate_180.isChecked()
        }
        
        super().accept()