#!/usr/bin/env python3
"""
Simplified UI Components for Image Gallery
No tagging system - just straightforward description text editing with copy/paste
"""

from PyQt6.QtWidgets import (
    QWidget, QPushButton, QLabel, QLineEdit, QHBoxLayout, QVBoxLayout,
    QTableWidget, QTableWidgetItem, QSplitter, QTextEdit, QHeaderView,
    QGroupBox, QFrame, QRadioButton, QButtonGroup, QCheckBox, QSizePolicy
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QColor


class GalleryTab(QWidget):
    """Main gallery tab with simplified description editing"""
    
    def __init__(self, parent_app):
        super().__init__()
        self.app = parent_app
        self.setup_ui()
    
    def setup_ui(self):
        """Set up the gallery tab UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Top button bar
        button_bar = QHBoxLayout()
        
        self.select_folder_btn = QPushButton("📁 Select Folder")
        self.select_folder_btn.setMinimumHeight(35)
        self.select_folder_btn.clicked.connect(self.app.event_handlers.select_folder)
        
        self.save_btn = QPushButton("💾 Save Descriptions")
        self.save_btn.setMinimumHeight(35)
        self.save_btn.clicked.connect(self.app.event_handlers.save_descriptions)
        
        self.refresh_btn = QPushButton("🔄 Refresh")
        self.refresh_btn.setMinimumHeight(35)
        self.refresh_btn.clicked.connect(self.app.event_handlers.refresh_gallery)
        
        button_bar.addWidget(self.select_folder_btn)
        button_bar.addWidget(self.save_btn)
        button_bar.addWidget(self.refresh_btn)
        button_bar.addStretch()
        
        layout.addLayout(button_bar)
        
        # Main splitter - horizontal split
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left panel - Image list
        left_panel = self.create_left_panel()
        
        # Right panel - Image preview and description editing
        right_panel = self.create_right_panel()
        
        main_splitter.addWidget(left_panel)
        main_splitter.addWidget(right_panel)
        main_splitter.setStretchFactor(0, 2)  # Table gets more space
        main_splitter.setStretchFactor(1, 3)  # Preview/edit gets even more
        
        layout.addWidget(main_splitter)
    
    def create_left_panel(self):
        """Create the left panel with image table"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Table for images
        self.table = QTableWidget()
        self.table.setColumnCount(2)
        self.table.setHorizontalHeaderLabels(["Filename", "Description"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.ExtendedSelection)
        self.table.setAlternatingRowColors(True)
        self.table.selectionModel().selectionChanged.connect(self.app.event_handlers.on_table_select)
        self.table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self.app.event_handlers.show_context_menu)
        
        layout.addWidget(self.table)
        
        # Selection info label
        self.selection_label = QLabel("No images loaded")
        self.selection_label.setStyleSheet("QLabel { color: #666; font-size: 11px; padding: 5px; }")
        layout.addWidget(self.selection_label)
        
        return panel
    
    def create_right_panel(self):
        """Create the right panel with image preview and description editing"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Image preview
        image_frame = QFrame()
        image_frame.setFrameShape(QFrame.Shape.Box)
        image_frame.setStyleSheet("QFrame { background-color: #2b2b2b; }")
        image_layout = QVBoxLayout(image_frame)
        
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumHeight(200)
        self.image_label.setMaximumHeight(400)
        self.image_label.setScaledContents(False)
        self.image_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Ignored
        )
        self.image_label.setStyleSheet("QLabel { color: #999; }")
        self.image_label.setText("No image selected")
        image_layout.addWidget(self.image_label)
        
        layout.addWidget(image_frame, stretch=3)
        
        # Description editing section
        desc_group = QGroupBox("Description")
        desc_layout = QVBoxLayout(desc_group)
        
        # Description text box
        self.description_text = QTextEdit()
        self.description_text.setPlaceholderText("Enter description for this image...")
        self.description_text.setMinimumHeight(120)
        self.description_text.textChanged.connect(self.app.event_handlers.on_description_changed)
        desc_layout.addWidget(self.description_text)
        
        # Copy/Paste buttons
        button_row = QHBoxLayout()
        
        self.copy_btn = QPushButton("📋 Copy Description")
        self.copy_btn.setToolTip("Copy description from current image (Ctrl+Shift+C)")
        self.copy_btn.clicked.connect(self.app.event_handlers.copy_description)
        
        self.paste_btn = QPushButton("📄 Paste to Selected")
        self.paste_btn.setToolTip("Paste description to selected image(s) (Ctrl+Shift+V)")
        self.paste_btn.clicked.connect(self.app.event_handlers.paste_description)
        
        self.append_keyword_btn = QPushButton("➕ Append to All")
        self.append_keyword_btn.setToolTip("Append text to all image descriptions")
        self.append_keyword_btn.clicked.connect(self.app.event_handlers.show_append_dialog)
        
        button_row.addWidget(self.copy_btn)
        button_row.addWidget(self.paste_btn)
        button_row.addWidget(self.append_keyword_btn)
        
        desc_layout.addLayout(button_row)
        
        # Clipboard status
        self.clipboard_label = QLabel("Clipboard: (empty)")
        self.clipboard_label.setStyleSheet("QLabel { color: #666; font-size: 10px; padding: 3px; }")
        desc_layout.addWidget(self.clipboard_label)
        
        layout.addWidget(desc_group, stretch=2)
        
        return panel


class UtilsTab(QWidget):
    """Utils tab with image processing tools"""
    
    def __init__(self, parent_app):
        super().__init__()
        self.app = parent_app
        self.setup_ui()
    
    def setup_ui(self):
        """Set up the utils tab UI"""
        layout = QVBoxLayout(self)

        # Scope indicator at top
        self.scope_label = QLabel()
        self.scope_label.setStyleSheet("""
            QLabel {
                background-color: #e8f5e9;
                color: #2e7d32;
                padding: 8px;
                border-radius: 4px;
                font-weight: bold;
            }
        """)
        self.scope_label.setVisible(False)
        layout.addWidget(self.scope_label)

        # Image processing panel
        processing_panel = self.create_processing_panel()
        layout.addWidget(processing_panel)
    
    def create_processing_panel(self):
        """Create image processing panel"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        title = QLabel("Image Processing")
        title.setStyleSheet("QLabel { font-size: 14px; font-weight: bold; }")
        layout.addWidget(title)
        
        # Fix Images section
        fix_group = QGroupBox("Fix Images")
        fix_layout = QVBoxLayout(fix_group)
        
        scope_layout = QHBoxLayout()
        self.fix_all_radio = QRadioButton("All Images")
        self.fix_selected_radio = QRadioButton("Selected Only")
        self.fix_all_radio.setChecked(True)
        
        self.fix_button_group = QButtonGroup()
        self.fix_button_group.addButton(self.fix_all_radio)
        self.fix_button_group.addButton(self.fix_selected_radio)
        
        scope_layout.addWidget(self.fix_all_radio)
        scope_layout.addWidget(self.fix_selected_radio)
        fix_layout.addLayout(scope_layout)
        
        self.fix_btn = QPushButton("Fix Images")
        self.fix_btn.clicked.connect(self.app.event_handlers.show_fix_dialog)
        fix_layout.addWidget(self.fix_btn)
        
        layout.addWidget(fix_group)
        
        # Mass Rename section
        rename_group = QGroupBox("Mass Rename")
        rename_layout = QVBoxLayout(rename_group)
        
        scope_layout = QHBoxLayout()
        self.rename_all_radio = QRadioButton("All Images")
        self.rename_selected_radio = QRadioButton("Selected Only")
        self.rename_all_radio.setChecked(True)
        
        self.rename_button_group = QButtonGroup()
        self.rename_button_group.addButton(self.rename_all_radio)
        self.rename_button_group.addButton(self.rename_selected_radio)
        
        scope_layout.addWidget(self.rename_all_radio)
        scope_layout.addWidget(self.rename_selected_radio)
        rename_layout.addLayout(scope_layout)
        
        prefix_layout = QHBoxLayout()
        prefix_layout.addWidget(QLabel("Prefix:"))
        self.prefix_entry = QLineEdit()
        self.prefix_entry.setPlaceholderText("e.g., 'portrait_'")
        prefix_layout.addWidget(self.prefix_entry)
        rename_layout.addLayout(prefix_layout)
        
        self.scramble_order_check = QCheckBox("Scramble image order (adds random letters)")
        rename_layout.addWidget(self.scramble_order_check)
        
        self.rename_btn = QPushButton("Rename Images")
        self.rename_btn.clicked.connect(self.app.event_handlers.mass_rename)
        rename_layout.addWidget(self.rename_btn)
        
        layout.addWidget(rename_group)
        
        # Dataset Augmentation section
        aug_group = QGroupBox("Dataset Augmentation")
        aug_layout = QVBoxLayout(aug_group)
        
        scope_layout = QHBoxLayout()
        self.aug_all_radio = QRadioButton("All Images")
        self.aug_selected_radio = QRadioButton("Selected Only")
        self.aug_all_radio.setChecked(True)
        
        self.aug_button_group = QButtonGroup()
        self.aug_button_group.addButton(self.aug_all_radio)
        self.aug_button_group.addButton(self.aug_selected_radio)
        
        scope_layout.addWidget(self.aug_all_radio)
        scope_layout.addWidget(self.aug_selected_radio)
        aug_layout.addLayout(scope_layout)
        
        self.aug_btn = QPushButton("Create Duplicates")
        self.aug_btn.clicked.connect(self.app.event_handlers.show_duplicate_dialog)
        aug_layout.addWidget(self.aug_btn)
        
        layout.addWidget(aug_group)

        # Claude AI Captioning section
        caption_group = QGroupBox("AI Captioning (Claude)")
        caption_layout = QVBoxLayout(caption_group)

        # Scope selection
        scope_layout = QHBoxLayout()
        self.caption_all_radio = QRadioButton("All Images")
        self.caption_selected_radio = QRadioButton("Selected Only")
        self.caption_missing_radio = QRadioButton("Missing Captions Only")
        self.caption_all_radio.setChecked(True)

        self.caption_button_group = QButtonGroup()
        self.caption_button_group.addButton(self.caption_all_radio)
        self.caption_button_group.addButton(self.caption_selected_radio)
        self.caption_button_group.addButton(self.caption_missing_radio)

        scope_layout.addWidget(self.caption_all_radio)
        scope_layout.addWidget(self.caption_selected_radio)
        scope_layout.addWidget(self.caption_missing_radio)
        caption_layout.addLayout(scope_layout)

        # Mode selection
        mode_layout = QHBoxLayout()
        self.caption_overwrite_check = QCheckBox("Overwrite existing captions")
        self.caption_overwrite_check.setToolTip("If unchecked, will append to existing captions")
        mode_layout.addWidget(self.caption_overwrite_check)
        mode_layout.addStretch()
        caption_layout.addLayout(mode_layout)

        # Prompt template
        prompt_label = QLabel("Caption Prompt Template:")
        caption_layout.addWidget(prompt_label)

        self.caption_prompt_text = QTextEdit()
        self.caption_prompt_text.setMaximumHeight(80)
        self.caption_prompt_text.setPlaceholderText("Enter instructions for how Claude should describe images...")
        caption_layout.addWidget(self.caption_prompt_text)

        # Generate button
        self.generate_captions_btn = QPushButton("Generate Captions with Claude")
        self.generate_captions_btn.setMinimumHeight(35)
        self.generate_captions_btn.setStyleSheet("""
            QPushButton {
                background-color: #7c4dff;
                color: white;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #651fff;
            }
            QPushButton:disabled {
                background-color: #ccc;
            }
        """)
        self.generate_captions_btn.clicked.connect(self.app.event_handlers.generate_captions_claude)
        caption_layout.addWidget(self.generate_captions_btn)

        # Status label
        self.caption_status_label = QLabel("")
        self.caption_status_label.setStyleSheet("color: #666; font-size: 11px;")
        caption_layout.addWidget(self.caption_status_label)

        layout.addWidget(caption_group)

        layout.addStretch()

        return panel

    def update_scope_info(self, selected_count, total_count):
        """Update the scope indicator"""
        if selected_count > 0:
            self.scope_label.setText(f"📋 {selected_count} of {total_count} images selected")
            self.scope_label.setVisible(True)

            # Enable/disable "Selected Only" radio buttons
            self.fix_selected_radio.setEnabled(True)
            self.rename_selected_radio.setEnabled(True)
            self.aug_selected_radio.setEnabled(True)
            self.caption_selected_radio.setEnabled(True)
        else:
            self.scope_label.setVisible(False)

            # Disable "Selected Only" options
            self.fix_selected_radio.setEnabled(False)
            self.rename_selected_radio.setEnabled(False)
            self.aug_selected_radio.setEnabled(False)
            self.caption_selected_radio.setEnabled(False)

            # Switch to "All" if "Selected" was checked
            if self.fix_selected_radio.isChecked():
                self.fix_all_radio.setChecked(True)
            if self.rename_selected_radio.isChecked():
                self.rename_all_radio.setChecked(True)
            if self.aug_selected_radio.isChecked():
                self.aug_all_radio.setChecked(True)
            if self.caption_selected_radio.isChecked():
                self.caption_all_radio.setChecked(True)
