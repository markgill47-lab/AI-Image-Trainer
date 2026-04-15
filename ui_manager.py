#!/usr/bin/env python3
"""
UI Manager for Flux Training GUI (PyQt6 Version)
Simplified version adapted from Hidream for Kohya sd-scripts
Main GUI class that integrates all managers
"""

import os
import sys
import json
import logging
import threading

# Ensure current directory is in path for package imports
_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

# PyQt6 imports
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QTabWidget, QLabel, QPushButton, QLineEdit,
    QTextEdit, QComboBox, QFileDialog, QMessageBox, QDialog,
    QGroupBox, QSpinBox, QDoubleSpinBox, QProgressBar, QCheckBox
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QObject
from PyQt6.QtGui import QFont, QPixmap

# Matplotlib for performance graphs
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# Our managers
from kohya_config_manager import KohyaConfigManager
from kohya_dataset_manager import KohyaDatasetManager
from enhanced_training_manager import EnhancedTrainingManager
from kohya_presets import TRAINING_PRESETS, get_preset, apply_preset_to_config
from flux_presets import FLUX_PRESETS, MODEL_OPTIONS, get_all_presets, get_presets_for_model
from gui_config_manager import GUIConfigManager

# Import dataset_manager package (handle path issues on Windows)
import importlib.util
_dm_init = os.path.join(_script_dir, 'dataset_manager', '__init__.py')
_dm_spec = importlib.util.spec_from_file_location(
    'dataset_manager', _dm_init,
    submodule_search_locations=[os.path.join(_script_dir, 'dataset_manager')]
)
_dataset_manager = importlib.util.module_from_spec(_dm_spec)
sys.modules['dataset_manager'] = _dataset_manager
_dm_spec.loader.exec_module(_dataset_manager)
EmbeddedDatasetManager = _dataset_manager.EmbeddedDatasetManager

# Initialize logger
logger = logging.getLogger('FluxTrainingGUI')


class TrainingSignals(QObject):
    """Signals for thread-safe communication"""
    output_updated = pyqtSignal(str)
    training_finished = pyqtSignal(object)
    status_updated = pyqtSignal(str)
    claude_analysis_done = pyqtSignal(bool, str)  # success, result


class UIManager(QMainWindow):
    """
    Main GUI class for Flux Training GUI
    """
    
    def __init__(self):
        """Initialize the UI manager"""
        super().__init__()
        
        # App configuration
        self.app_config_file = 'flux_gui_config.json'
        self.app_config = self.load_app_config()
        
        # Set up window
        self.setWindowTitle("Flux Training GUI - VizLab")
        self.restore_window_geometry()
        
        # Configure fonts
        self.setup_fonts()
        
        # Initialize variables
        self.config_file = self.app_config.get('config_file', 'training_config.toml')
        self.dataset_config_file = self.app_config.get('dataset_config_file', 'dataset.toml')
        self.dataset_dir = self.app_config.get('dataset_dir', '')
        self.output_dir = self.app_config.get('output_dir', './output')
        
        # Performance tracking
        self.performance_figure = None
        self.performance_canvas = None
        self.loss_line = None
        
        # Set up signals
        self.signals = TrainingSignals()
        self.signals.output_updated.connect(self.update_training_output_safe)
        self.signals.training_finished.connect(self.handle_training_finished)
        self.signals.status_updated.connect(self.update_status_safe)
        self.signals.claude_analysis_done.connect(self._update_claude_analysis_results)
        
        # Initialize managers
        self.config_manager = KohyaConfigManager(
            self.update_status,
            self.show_error,
            self.show_info
        )
        
        self.dataset_manager = KohyaDatasetManager(
            self.update_status,
            self.show_error,
            self.show_info
        )
        
        # Initialize GUI config manager (epochs-based settings)
        self.gui_config = GUIConfigManager('gui_config.json')
        logger.info("GUI config initialized")
        
        self.training_manager = EnhancedTrainingManager(
            self.update_status,
            self.safe_update_output,
            self.handle_training_finished_internal,
            self.handle_training_error_internal
        )
        
        # Set up UI
        self.create_widgets()
        
        # Set up timers
        self.setup_timers()
        
        # Load initial config if exists
        if os.path.exists(self.config_file):
            self.config_manager.load_config(self.config_file)
            self.load_config_to_ui()

        # Always refresh dataset stats on startup if dataset directory exists
        self.refresh_dataset_on_startup()
    
    def load_app_config(self):
        """Load application configuration"""
        default_config = {
            'window_width': 1200,
            'window_height': 900,
            'window_x': 100,
            'window_y': 100,
            'config_file': 'training_config.toml',
            'dataset_config_file': 'dataset.toml',
            'dataset_dir': '',
            'output_dir': './output'
        }
        
        try:
            if os.path.exists(self.app_config_file):
                with open(self.app_config_file, 'r') as f:
                    config = json.load(f)
                    default_config.update(config)
                logger.info(f"Loaded app config from {self.app_config_file}")
        except Exception as e:
            logger.warning(f"Error loading app config: {e}")
        
        return default_config
    
    def save_app_config(self):
        """Save application configuration"""
        try:
            geometry = self.geometry()
            self.app_config.update({
                'window_width': geometry.width(),
                'window_height': geometry.height(),
                'window_x': geometry.x(),
                'window_y': geometry.y(),
                'config_file': self.config_file,
                'dataset_config_file': self.dataset_config_file,
                'dataset_dir': self.dataset_dir,
                'output_dir': self.output_dir
            })
            
            with open(self.app_config_file, 'w') as f:
                json.dump(self.app_config, f, indent=2)
            
            logger.info("Saved app configuration")
        except Exception as e:
            logger.error(f"Error saving app config: {e}")
    
    def restore_window_geometry(self):
        """Restore window size and position"""
        width = self.app_config.get('window_width', 1200)
        height = self.app_config.get('window_height', 900)
        x = self.app_config.get('window_x', 100)
        y = self.app_config.get('window_y', 100)
        self.setGeometry(x, y, width, height)
    
    def closeEvent(self, event):
        """Handle application close"""
        self.save_app_config()
        
        # Check if training is running
        if self.training_manager.is_training_running():
            reply = QMessageBox.question(
                self, 'Training Running',
                'Training is running. Stop and exit?',
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            
            if reply == QMessageBox.StandardButton.Yes:
                self.training_manager.stop_training()
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()
    
    def setup_fonts(self):
        """Set up fonts"""
        self.default_font = QFont("Arial", 10)
        self.header_font = QFont("Arial", 12, QFont.Weight.Bold)
        self.mono_font = QFont("Consolas", 9)
    
    def create_widgets(self):
        """Create main UI widgets"""
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Tab widget
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)
        
        # Create tabs
        self.setup_tab = QWidget()
        self.config_tab = QWidget()
        self.dataset_tab = QWidget()
        self.training_tab = QWidget()
        self.performance_tab = QWidget()
        
        # Add tabs
        self.tab_widget.addTab(self.setup_tab, "Setup")
        self.tab_widget.addTab(self.config_tab, "Configuration")
        self.tab_widget.addTab(self.dataset_tab, "Dataset")
        self.tab_widget.addTab(self.training_tab, "Training")
        self.tab_widget.addTab(self.performance_tab, "Performance")
        
        # Set up each tab
        self.setup_setup_tab()
        self.setup_config_tab()
        self.setup_dataset_tab()
        self.setup_training_tab()
        self.setup_performance_tab()
    
    def setup_setup_tab(self):
        """Create setup tab"""
        layout = QVBoxLayout(self.setup_tab)
        
        # Directories group
        dir_group = QGroupBox("Project Setup")
        dir_layout = QGridLayout(dir_group)
        
        # Config file
        dir_layout.addWidget(QLabel("Training Config:"), 0, 0)
        self.config_file_entry = QLineEdit(self.config_file)
        dir_layout.addWidget(self.config_file_entry, 0, 1)
        browse_config_btn = QPushButton("Browse")
        browse_config_btn.clicked.connect(self.browse_config_file)
        dir_layout.addWidget(browse_config_btn, 0, 2)
        
        # Output directory
        dir_layout.addWidget(QLabel("Output Directory:"), 1, 0)
        self.output_dir_entry = QLineEdit(self.output_dir)
        dir_layout.addWidget(self.output_dir_entry, 1, 1)
        browse_output_btn = QPushButton("Browse")
        browse_output_btn.clicked.connect(self.browse_output_dir)
        dir_layout.addWidget(browse_output_btn, 1, 2)
        
        layout.addWidget(dir_group)
        
        # Actions group
        actions_group = QGroupBox("Actions")
        actions_layout = QVBoxLayout(actions_group)
        
        # Load/Create config buttons
        btn_layout = QHBoxLayout()
        
        load_config_btn = QPushButton("Load Config")
        load_config_btn.clicked.connect(self.load_config)
        btn_layout.addWidget(load_config_btn)
        
        create_config_btn = QPushButton("Create Default Config")
        create_config_btn.clicked.connect(self.create_default_config)
        btn_layout.addWidget(create_config_btn)
        
        save_config_btn = QPushButton("Save Config")
        save_config_btn.clicked.connect(self.save_config)
        btn_layout.addWidget(save_config_btn)
        
        actions_layout.addLayout(btn_layout)
        layout.addWidget(actions_group)

        # Claude API group
        claude_group = QGroupBox("Claude API (for AI Captioning & Analysis)")
        claude_layout = QGridLayout(claude_group)

        # API key input
        claude_layout.addWidget(QLabel("API Key:"), 0, 0)
        self.claude_api_key_entry = QLineEdit()
        self.claude_api_key_entry.setEchoMode(QLineEdit.EchoMode.Password)
        self.claude_api_key_entry.setPlaceholderText("Enter your Anthropic API key (sk-ant-...)")
        saved_key = self.gui_config.get('claude.api_key', '')
        if saved_key:
            self.claude_api_key_entry.setText(saved_key)
        self.claude_api_key_entry.textChanged.connect(self.on_claude_api_key_changed)
        claude_layout.addWidget(self.claude_api_key_entry, 0, 1)

        # Show/Hide button
        self.claude_show_key_btn = QPushButton("Show")
        self.claude_show_key_btn.setFixedWidth(60)
        self.claude_show_key_btn.clicked.connect(self.toggle_claude_key_visibility)
        claude_layout.addWidget(self.claude_show_key_btn, 0, 2)

        # Test connection button
        self.claude_test_btn = QPushButton("Test Connection")
        self.claude_test_btn.clicked.connect(self.test_claude_connection)
        claude_layout.addWidget(self.claude_test_btn, 1, 0)

        # Status label
        self.claude_status_label = QLabel("Not tested")
        self.claude_status_label.setStyleSheet("color: #666;")
        claude_layout.addWidget(self.claude_status_label, 1, 1, 1, 2)

        layout.addWidget(claude_group)

        # Status display
        status_group = QGroupBox("Status")
        status_layout = QVBoxLayout(status_group)
        
        self.status_text = QTextEdit()
        self.status_text.setReadOnly(True)
        self.status_text.setMaximumHeight(200)
        self.status_text.setFont(self.mono_font)
        status_layout.addWidget(self.status_text)
        
        layout.addWidget(status_group)
        layout.addStretch()
    
    def setup_config_tab(self):
        """Create configuration tab with epochs-based training"""
        layout = QVBoxLayout(self.config_tab)
        
        # Model Selection group
        model_group = QGroupBox("Model Selection")
        model_layout = QVBoxLayout(model_group)
        
        model_select_layout = QHBoxLayout()
        model_select_layout.addWidget(QLabel("Select Model:"))
        
        self.model_combo = QComboBox()
        for model_key, model_info in MODEL_OPTIONS.items():
            display_name = f"{model_info['name']} ({model_info['backend']})"
            self.model_combo.addItem(display_name, model_key)
        
        # Set from saved config
        saved_model = self.gui_config.get('model.selected', 'sdxl')
        for i in range(self.model_combo.count()):
            if self.model_combo.itemData(i) == saved_model:
                self.model_combo.setCurrentIndex(i)
                break
        
        self.model_combo.currentIndexChanged.connect(self.on_model_selected)
        model_select_layout.addWidget(self.model_combo)
        
        model_layout.addLayout(model_select_layout)
        
        # Model description
        self.model_description = QLabel("")
        self.model_description.setWordWrap(True)
        model_layout.addWidget(self.model_description)
        
        layout.addWidget(model_group)

        # Model Paths group (for local/ComfyUI models)
        paths_group = QGroupBox("Model Paths (Optional - for local/ComfyUI models)")
        paths_layout = QGridLayout(paths_group)

        # ComfyUI Models Base
        paths_layout.addWidget(QLabel("ComfyUI Models Base:"), 0, 0)
        self.comfyui_base_entry = QLineEdit()
        self.comfyui_base_entry.setPlaceholderText("e.g. /home/user/ComfyUI/models or C:/ComfyUI/models")
        saved_comfyui = self.gui_config.get('paths.comfyui_models_path', '')
        if saved_comfyui:
            self.comfyui_base_entry.setText(saved_comfyui)
        paths_layout.addWidget(self.comfyui_base_entry, 0, 1)
        browse_comfyui_btn = QPushButton("Browse")
        browse_comfyui_btn.clicked.connect(self.browse_comfyui_base)
        paths_layout.addWidget(browse_comfyui_btn, 0, 2)

        # Auto-detect button
        auto_detect_btn = QPushButton("Auto-detect from ComfyUI")
        auto_detect_btn.clicked.connect(self.auto_detect_model_paths)
        paths_layout.addWidget(auto_detect_btn, 0, 3)

        # Transformer Override
        paths_layout.addWidget(QLabel("Transformer:"), 1, 0)
        self.transformer_path_entry = QLineEdit()
        self.transformer_path_entry.setPlaceholderText("Auto-detected or manual override (.safetensors)")
        saved_transformer = self.gui_config.get('paths.transformer_path', '')
        if saved_transformer:
            self.transformer_path_entry.setText(saved_transformer)
        paths_layout.addWidget(self.transformer_path_entry, 1, 1, 1, 2)
        browse_transformer_btn = QPushButton("Browse")
        browse_transformer_btn.clicked.connect(self.browse_transformer_path)
        paths_layout.addWidget(browse_transformer_btn, 1, 3)

        # VAE Override
        paths_layout.addWidget(QLabel("VAE:"), 2, 0)
        self.vae_path_entry = QLineEdit()
        self.vae_path_entry.setPlaceholderText("Auto-detected or manual override (.safetensors)")
        saved_vae = self.gui_config.get('paths.vae_path', '')
        if saved_vae:
            self.vae_path_entry.setText(saved_vae)
        paths_layout.addWidget(self.vae_path_entry, 2, 1, 1, 2)
        browse_vae_btn = QPushButton("Browse")
        browse_vae_btn.clicked.connect(self.browse_vae_path)
        paths_layout.addWidget(browse_vae_btn, 2, 3)

        # Text Encoder Repo
        paths_layout.addWidget(QLabel("Text Encoder Repo:"), 3, 0)
        self.text_encoder_repo_entry = QLineEdit()
        self.text_encoder_repo_entry.setPlaceholderText("Leave empty for default (downloads from Klein HuggingFace repo)")
        saved_te_repo = self.gui_config.get('paths.text_encoder_repo', '')
        if saved_te_repo:
            self.text_encoder_repo_entry.setText(saved_te_repo)
        paths_layout.addWidget(self.text_encoder_repo_entry, 3, 1, 1, 3)

        # Status label for auto-detect results
        self.model_paths_status = QLabel("")
        self.model_paths_status.setStyleSheet("color: gray; font-style: italic;")
        self.model_paths_status.setWordWrap(True)
        paths_layout.addWidget(self.model_paths_status, 4, 0, 1, 4)

        layout.addWidget(paths_group)

        # Preset selector
        preset_group = QGroupBox("Training Presets (Optional)")
        preset_layout = QVBoxLayout(preset_group)
        
        preset_info_label = QLabel("Presets apply recommended settings. You can customize below.")
        preset_info_label.setStyleSheet("color: gray; font-style: italic;")
        preset_layout.addWidget(preset_info_label)
        
        preset_select_layout = QHBoxLayout()
        preset_select_layout.addWidget(QLabel("Select Preset:"))
        
        self.preset_combo = QComboBox()
        self.preset_combo.addItem("-- Custom Settings --")
        for preset_name, preset_data in TRAINING_PRESETS.items():
            display_name = preset_data['name']
            self.preset_combo.addItem(display_name, preset_name)
        self.preset_combo.currentIndexChanged.connect(self.on_preset_selected)
        preset_select_layout.addWidget(self.preset_combo)
        
        preset_layout.addLayout(preset_select_layout)
        
        # Preset description
        self.preset_description = QLabel("")
        self.preset_description.setWordWrap(True)
        preset_layout.addWidget(self.preset_description)
        
        layout.addWidget(preset_group)
        
        # Training Duration group (epochs-based)
        duration_group = QGroupBox("Training Duration")
        duration_layout = QGridLayout(duration_group)
        
        row = 0
        
        # Explanation
        info_label = QLabel("Training steps are calculated automatically:")
        info_label.setStyleSheet("font-weight: bold;")
        duration_layout.addWidget(info_label, row, 0, 1, 3)
        row += 1
        
        # Formula display
        self.formula_label = QLabel("Steps = Images × Samples × Epochs")
        self.formula_label.setStyleSheet("color: #0066cc; font-family: monospace;")
        duration_layout.addWidget(self.formula_label, row, 0, 1, 3)
        row += 1
        
        # Calculated steps display (big and bold)
        calc_layout = QHBoxLayout()
        calc_layout.addWidget(QLabel("Total Steps:"))
        self.calculated_steps_label = QLabel("400")
        self.calculated_steps_label.setStyleSheet(
            "font-size: 18pt; font-weight: bold; color: #006600;"
        )
        calc_layout.addWidget(self.calculated_steps_label)
        self.estimated_time_label = QLabel("(~7 hours)")
        self.estimated_time_label.setStyleSheet("color: gray;")
        calc_layout.addWidget(self.estimated_time_label)
        calc_layout.addStretch()
        duration_layout.addLayout(calc_layout, row, 0, 1, 3)
        row += 1
        
        # Separator
        duration_layout.addWidget(QLabel(""), row, 0)
        row += 1
        
        # Epochs
        duration_layout.addWidget(QLabel("Epochs:"), row, 0)
        self.epochs_spinbox = QSpinBox()
        self.epochs_spinbox.setRange(1, 50)
        self.epochs_spinbox.setValue(self.gui_config.get('training.epochs', 1))
        self.epochs_spinbox.setToolTip("Number of times to train on the entire dataset")
        self.epochs_spinbox.valueChanged.connect(self.on_training_param_changed)
        duration_layout.addWidget(self.epochs_spinbox, row, 1)
        epoch_help = QLabel("(Full passes through dataset)")
        epoch_help.setStyleSheet("color: gray; font-size: 9pt;")
        duration_layout.addWidget(epoch_help, row, 2)
        row += 1
        
        # Samples per image
        duration_layout.addWidget(QLabel("Samples per Image:"), row, 0)
        self.samples_spinbox = QSpinBox()
        self.samples_spinbox.setRange(1, 100)
        self.samples_spinbox.setValue(self.gui_config.get('training.samples_per_image', 10))
        self.samples_spinbox.setToolTip("How many times each image is used per epoch")
        self.samples_spinbox.valueChanged.connect(self.on_training_param_changed)
        duration_layout.addWidget(self.samples_spinbox, row, 1)
        samples_help = QLabel("(Repeats per image)")
        samples_help.setStyleSheet("color: gray; font-size: 9pt;")
        duration_layout.addWidget(samples_help, row, 2)
        row += 1
        
        # Number of images (read-only, informational)
        duration_layout.addWidget(QLabel("Images in Dataset:"), row, 0)
        self.num_images_label = QLabel("40 (estimate)")
        self.num_images_label.setStyleSheet("font-weight: bold;")
        duration_layout.addWidget(self.num_images_label, row, 1)
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self.refresh_image_count)
        duration_layout.addWidget(refresh_btn, row, 2)
        row += 1
        
        layout.addWidget(duration_group)
        
        # Training Parameters group
        params_group = QGroupBox("Training Parameters")
        params_layout = QGridLayout(params_group)
        
        row = 0
        
        # Learning rate
        params_layout.addWidget(QLabel("Learning Rate:"), row, 0)
        self.lr_spinbox = QDoubleSpinBox()
        self.lr_spinbox.setDecimals(6)
        self.lr_spinbox.setRange(0.000001, 0.01)
        self.lr_spinbox.setValue(self.gui_config.get('training.learning_rate', 0.0001))
        self.lr_spinbox.setSingleStep(0.00001)
        self.lr_spinbox.setToolTip("Controls how fast the model learns")
        self.lr_spinbox.valueChanged.connect(self.on_training_param_changed)
        params_layout.addWidget(self.lr_spinbox, row, 1)
        row += 1
        
        # LoRA Rank
        params_layout.addWidget(QLabel("LoRA Rank (dim):"), row, 0)
        self.dim_spinbox = QSpinBox()
        self.dim_spinbox.setRange(4, 128)
        self.dim_spinbox.setValue(self.gui_config.get('training.lora_rank', 16))
        self.dim_spinbox.setToolTip("Higher = more detail, slower training")
        self.dim_spinbox.valueChanged.connect(self.on_training_param_changed)
        params_layout.addWidget(self.dim_spinbox, row, 1)
        rank_help = QLabel("(8-16: fast, 32: balanced, 64+: detailed)")
        rank_help.setStyleSheet("color: gray; font-size: 9pt;")
        params_layout.addWidget(rank_help, row, 2)
        row += 1
        
        # LoRA Alpha
        params_layout.addWidget(QLabel("LoRA Alpha:"), row, 0)
        self.alpha_spinbox = QSpinBox()
        self.alpha_spinbox.setRange(1, 128)
        self.alpha_spinbox.setValue(self.gui_config.get('training.lora_alpha', 8))
        self.alpha_spinbox.setToolTip("Usually same as or half of rank")
        self.alpha_spinbox.valueChanged.connect(self.on_training_param_changed)
        params_layout.addWidget(self.alpha_spinbox, row, 1)
        row += 1
        
        # Save frequency
        params_layout.addWidget(QLabel("Save Every N Steps:"), row, 0)
        self.save_every_spinbox = QSpinBox()
        self.save_every_spinbox.setRange(100, 10000)
        self.save_every_spinbox.setValue(self.gui_config.get('training.save_every_n_steps', 500))
        self.save_every_spinbox.setSingleStep(100)
        self.save_every_spinbox.setToolTip("How often to save checkpoints")
        self.save_every_spinbox.valueChanged.connect(self.on_training_param_changed)
        params_layout.addWidget(self.save_every_spinbox, row, 1)
        row += 1
        
        layout.addWidget(params_group)
        
        # Output group
        output_group = QGroupBox("Output Settings")
        output_layout = QGridLayout(output_group)
        
        # Output name
        output_layout.addWidget(QLabel("Output Name:"), 0, 0)
        self.output_name_entry = QLineEdit(self.gui_config.get('paths.output_name', 'flux_lora'))
        self.output_name_entry.setToolTip("Name for your trained LoRA")
        self.output_name_entry.textChanged.connect(self.on_training_param_changed)
        output_layout.addWidget(self.output_name_entry, 0, 1)
        
        layout.addWidget(output_group)

        # Sample Generation group
        sample_group = QGroupBox("Sample Generation During Training")
        sample_layout = QGridLayout(sample_group)

        row = 0

        # Enable sampling checkbox
        self.sample_enabled_checkbox = QCheckBox("Enable sample generation")
        self.sample_enabled_checkbox.setChecked(self.gui_config.get('sampling.enabled', True))
        self.sample_enabled_checkbox.setToolTip("Generate preview images during training")
        self.sample_enabled_checkbox.stateChanged.connect(self.on_sample_enabled_changed)
        sample_layout.addWidget(self.sample_enabled_checkbox, row, 0, 1, 3)
        row += 1

        # Sample prompt
        sample_layout.addWidget(QLabel("Sample Prompt:"), row, 0, Qt.AlignmentFlag.AlignTop)
        self.sample_prompt_text = QTextEdit()
        self.sample_prompt_text.setPlainText(self.gui_config.get('sampling.prompt', ''))
        self.sample_prompt_text.setPlaceholderText("Enter a prompt to generate samples during training (e.g., 'a photo of TOK person')")
        self.sample_prompt_text.setMaximumHeight(100)
        self.sample_prompt_text.setToolTip("Prompt used to generate preview images. Use your trigger word if applicable.")
        self.sample_prompt_text.textChanged.connect(self.on_training_param_changed)
        sample_layout.addWidget(self.sample_prompt_text, row, 1, 1, 2)
        row += 1

        # Sample frequency
        sample_layout.addWidget(QLabel("Generate Every N Steps:"), row, 0)
        self.sample_every_spinbox = QSpinBox()
        self.sample_every_spinbox.setRange(10, 1000)
        self.sample_every_spinbox.setValue(self.gui_config.get('sampling.sample_every', 100))
        self.sample_every_spinbox.setSingleStep(10)
        self.sample_every_spinbox.setToolTip("How often to generate sample images")
        self.sample_every_spinbox.valueChanged.connect(self.on_training_param_changed)
        sample_layout.addWidget(self.sample_every_spinbox, row, 1)

        # Calculate number of samples
        self.sample_count_label = QLabel("(~12 samples)")
        self.sample_count_label.setStyleSheet("color: gray; font-size: 9pt;")
        sample_layout.addWidget(self.sample_count_label, row, 2)
        row += 1

        # Sample resolution
        sample_layout.addWidget(QLabel("Sample Resolution:"), row, 0)
        resolution_layout = QHBoxLayout()

        self.sample_width_spinbox = QSpinBox()
        self.sample_width_spinbox.setRange(256, 2048)
        self.sample_width_spinbox.setValue(self.gui_config.get('sampling.width', 512))
        self.sample_width_spinbox.setSingleStep(64)
        self.sample_width_spinbox.setToolTip("Width of sample images")
        self.sample_width_spinbox.valueChanged.connect(self.on_training_param_changed)
        resolution_layout.addWidget(self.sample_width_spinbox)

        resolution_layout.addWidget(QLabel("×"))

        self.sample_height_spinbox = QSpinBox()
        self.sample_height_spinbox.setRange(256, 2048)
        self.sample_height_spinbox.setValue(self.gui_config.get('sampling.height', 512))
        self.sample_height_spinbox.setSingleStep(64)
        self.sample_height_spinbox.setToolTip("Height of sample images")
        self.sample_height_spinbox.valueChanged.connect(self.on_training_param_changed)
        resolution_layout.addWidget(self.sample_height_spinbox)

        resolution_layout.addStretch()
        sample_layout.addLayout(resolution_layout, row, 1, 1, 2)
        row += 1

        # Seed controls
        sample_layout.addWidget(QLabel("Sample Seed:"), row, 0)
        seed_layout = QHBoxLayout()

        self.sample_seed_spinbox = QSpinBox()
        self.sample_seed_spinbox.setRange(0, 2147483647)  # Max int32
        self.sample_seed_spinbox.setValue(self.gui_config.get('sampling.seed', 42))
        self.sample_seed_spinbox.setToolTip("Seed for reproducible sample generation")
        self.sample_seed_spinbox.valueChanged.connect(self.on_training_param_changed)
        seed_layout.addWidget(self.sample_seed_spinbox)

        self.random_seed_checkbox = QCheckBox("Random seed each time")
        self.random_seed_checkbox.setChecked(self.gui_config.get('sampling.random_seed', False))
        self.random_seed_checkbox.setToolTip("Use a different random seed for each sample generation")
        self.random_seed_checkbox.stateChanged.connect(self.on_random_seed_changed)
        seed_layout.addWidget(self.random_seed_checkbox)

        seed_layout.addStretch()
        sample_layout.addLayout(seed_layout, row, 1, 1, 2)
        row += 1

        # Apply initial random seed state
        self.sample_seed_spinbox.setEnabled(not self.random_seed_checkbox.isChecked())

        # Info label
        sample_info = QLabel("Samples saved to output folder (use 'Samples Folder' button in Performance tab)")
        sample_info.setStyleSheet("color: gray; font-style: italic; font-size: 9pt;")
        sample_layout.addWidget(sample_info, row, 0, 1, 3)

        layout.addWidget(sample_group)

        # Live Update section (for applying changes during training)
        live_update_layout = QHBoxLayout()

        self.apply_changes_btn = QPushButton("⚡ Apply Changes to Running Training")
        self.apply_changes_btn.setMinimumHeight(35)
        self.apply_changes_btn.setStyleSheet("""
            QPushButton {
                background-color: #ff9800;
                color: white;
                font-weight: bold;
                padding: 8px 16px;
            }
            QPushButton:hover {
                background-color: #f57c00;
            }
            QPushButton:disabled {
                background-color: #ccc;
                color: #666;
            }
        """)
        self.apply_changes_btn.setToolTip(
            "Apply Learning Rate, Sample Prompt, Sample Frequency, and Save Frequency\n"
            "changes to the currently running training without restarting."
        )
        self.apply_changes_btn.clicked.connect(self.apply_live_changes)
        self.apply_changes_btn.setEnabled(False)  # Disabled until training starts
        live_update_layout.addWidget(self.apply_changes_btn)

        self.live_update_status = QLabel("")
        self.live_update_status.setStyleSheet("color: #666; font-style: italic;")
        live_update_layout.addWidget(self.live_update_status)
        live_update_layout.addStretch()

        layout.addLayout(live_update_layout)

        # Auto-save indicator
        self.autosave_label = QLabel("✓ Settings auto-save on change")
        self.autosave_label.setStyleSheet("color: green; font-style: italic;")
        layout.addWidget(self.autosave_label)
        
        layout.addStretch()
        
        # Trigger initial calculation
        self.on_training_param_changed()
    
    def setup_dataset_tab(self):
        """Create dataset tab with embedded dataset manager"""
        layout = QVBoxLayout(self.dataset_tab)
        layout.setContentsMargins(5, 5, 5, 5)

        # Dataset directory selector (linked to training config)
        dir_group = QGroupBox("Dataset Location")
        dir_layout = QHBoxLayout(dir_group)

        dir_layout.addWidget(QLabel("Image Folder:"))
        self.dataset_dir_entry = QLineEdit(self.dataset_dir)
        self.dataset_dir_entry.textChanged.connect(self.on_dataset_dir_changed)
        dir_layout.addWidget(self.dataset_dir_entry)

        browse_dataset_btn = QPushButton("Browse")
        browse_dataset_btn.clicked.connect(self.browse_dataset_dir)
        dir_layout.addWidget(browse_dataset_btn)

        load_btn = QPushButton("Load")
        load_btn.clicked.connect(self.load_dataset_folder)
        dir_layout.addWidget(load_btn)

        layout.addWidget(dir_group)

        # Embedded dataset manager widget
        self.embedded_dataset_manager = EmbeddedDatasetManager(
            parent=self,
            status_callback=self.update_status
        )
        layout.addWidget(self.embedded_dataset_manager, stretch=1)

        # Load the dataset folder if one is already configured
        if self.dataset_dir and os.path.isdir(self.dataset_dir):
            self.embedded_dataset_manager.load_folder(self.dataset_dir)
            # Ensure gui_config has the dataset path for steps calculation
            self.gui_config.set('paths.dataset_dir', self.dataset_dir)

    def on_dataset_dir_changed(self, text):
        """Handle dataset directory text changes"""
        self.dataset_dir = text

    def load_dataset_folder(self):
        """Load the current dataset folder into the embedded manager"""
        folder = self.dataset_dir_entry.text().strip()
        if folder and os.path.isdir(folder):
            self.dataset_dir = folder
            if self.embedded_dataset_manager.load_folder(folder):
                # Update gui_config with new dataset path so steps calculation works
                self.gui_config.set('paths.dataset_dir', folder)

                # Refresh training calculations
                self.refresh_image_count()
        elif folder:
            self.show_error(f"Folder not found: {folder}")
    
    def setup_training_tab(self):
        """Create training tab"""
        layout = QVBoxLayout(self.training_tab)
        
        # Training control
        control_group = QGroupBox("Training Control")
        control_layout = QVBoxLayout(control_group)
        
        btn_layout = QHBoxLayout()
        
        self.start_training_btn = QPushButton("Start Training")
        self.start_training_btn.clicked.connect(self.start_training)
        btn_layout.addWidget(self.start_training_btn)
        
        self.stop_training_btn = QPushButton("Stop Training")
        self.stop_training_btn.clicked.connect(self.stop_training)
        self.stop_training_btn.setEnabled(False)
        btn_layout.addWidget(self.stop_training_btn)
        
        control_layout.addLayout(btn_layout)
        layout.addWidget(control_group)
        
        # Progress
        progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout(progress_group)
        
        self.progress_bar = QProgressBar()
        progress_layout.addWidget(self.progress_bar)
        
        self.progress_label = QLabel("Not started")
        progress_layout.addWidget(self.progress_label)
        
        layout.addWidget(progress_group)
        
        # Training output
        output_group = QGroupBox("Training Output")
        output_layout = QVBoxLayout(output_group)
        
        self.training_output = QTextEdit()
        self.training_output.setReadOnly(True)
        self.training_output.setFont(self.mono_font)
        output_layout.addWidget(self.training_output)
        
        layout.addWidget(output_group)
    
    def setup_performance_tab(self):
        """Create performance tab with streamlined monitoring"""
        layout = QVBoxLayout(self.performance_tab)

        # Compact metrics panel
        metrics_group = QGroupBox("Training Metrics")
        metrics_layout = QGridLayout(metrics_group)

        # Row 0: Step info and Loss
        metrics_layout.addWidget(QLabel("Current Step:"), 0, 0)
        self.current_step_label = QLabel("0")
        self.current_step_label.setStyleSheet("font-weight: bold; font-size: 12pt;")
        metrics_layout.addWidget(self.current_step_label, 0, 1)

        metrics_layout.addWidget(QLabel("Total Steps:"), 0, 2)
        self.total_steps_label = QLabel("0")
        self.total_steps_label.setStyleSheet("font-weight: bold; font-size: 12pt;")
        metrics_layout.addWidget(self.total_steps_label, 0, 3)

        metrics_layout.addWidget(QLabel("Current Loss:"), 0, 4)
        self.last_loss_label = QLabel("0.0000")
        self.last_loss_label.setStyleSheet("font-weight: bold; font-size: 12pt; color: #CC0000;")
        metrics_layout.addWidget(self.last_loss_label, 0, 5)

        # Claude Analysis button (upper right)
        self.analyze_with_claude_btn = QPushButton("🤖 Analyze with Claude")
        self.analyze_with_claude_btn.setStyleSheet("""
            QPushButton {
                background-color: #7c4dff;
                color: white;
                font-weight: bold;
                padding: 4px 12px;
            }
            QPushButton:hover {
                background-color: #651fff;
            }
            QPushButton:disabled {
                background-color: #ccc;
            }
        """)
        self.analyze_with_claude_btn.clicked.connect(self.analyze_training_with_claude)
        metrics_layout.addWidget(self.analyze_with_claude_btn, 0, 6)

        # Row 1: Progress, ETA, Avg Loss
        metrics_layout.addWidget(QLabel("Progress:"), 1, 0)
        self.progress_percent_label = QLabel("0%")
        self.progress_percent_label.setStyleSheet("font-weight: bold; font-size: 12pt;")
        metrics_layout.addWidget(self.progress_percent_label, 1, 1)

        metrics_layout.addWidget(QLabel("ETA:"), 1, 2)
        self.eta_label = QLabel("Calculating...")
        self.eta_label.setStyleSheet("font-weight: bold; color: #0066CC;")
        metrics_layout.addWidget(self.eta_label, 1, 3)

        metrics_layout.addWidget(QLabel("Avg Loss (100):"), 1, 4)
        self.avg_loss_label = QLabel("0.0000")
        self.avg_loss_label.setStyleSheet("font-weight: bold; font-size: 12pt;")
        metrics_layout.addWidget(self.avg_loss_label, 1, 5)

        layout.addWidget(metrics_group)

        # Placeholder labels for removed widgets (to avoid errors in update methods)
        self.speed_label = QLabel()
        self.time_per_step_label = QLabel()
        self.elapsed_time_label = QLabel()
        self.trend_status_label = QLabel()
        self.gpu_util_label = QLabel()
        self.vram_used_label = QLabel()
        self.gpu_temp_label = QLabel()
        self.gpu_power_label = QLabel()

        # Loss graph - stretch to fill available space
        graph_group = QGroupBox("Loss History")
        graph_layout = QVBoxLayout(graph_group)
        
        self.performance_figure = Figure(figsize=(10, 5))
        self.performance_canvas = FigureCanvas(self.performance_figure)
        self.performance_ax = self.performance_figure.add_subplot(111)
        
        # Style the graph
        self.performance_ax.set_xlabel('Step', fontsize=12)
        self.performance_ax.set_ylabel('Loss', fontsize=12)
        self.performance_ax.set_title('Training Loss Over Time', fontsize=14, fontweight='bold')
        self.performance_ax.grid(True, alpha=0.3)
        self.performance_figure.tight_layout()
        
        graph_layout.addWidget(self.performance_canvas)
        layout.addWidget(graph_group, stretch=1)  # Give graph priority for vertical space

        # Problematic Images Panel (collapsible)
        outlier_group = QGroupBox("High-Loss Images (Potential Issues)")
        outlier_group.setCheckable(True)
        outlier_group.setChecked(False)  # Collapsed by default
        outlier_layout = QVBoxLayout(outlier_group)

        self.outlier_info_label = QLabel("Analyzing dataset... (data will appear after ~100 steps)")
        self.outlier_info_label.setStyleSheet("color: #666; font-style: italic;")
        outlier_layout.addWidget(self.outlier_info_label)

        self.outlier_list = QTextEdit()
        self.outlier_list.setReadOnly(True)
        self.outlier_list.setMaximumHeight(120)
        self.outlier_list.setFont(self.mono_font)
        self.outlier_list.setPlaceholderText("Images with consistently high loss will be listed here...")
        outlier_layout.addWidget(self.outlier_list)

        layout.addWidget(outlier_group)

        # Samples Folder Button
        samples_btn_layout = QHBoxLayout()
        samples_btn_layout.addStretch()
        self.samples_folder_btn = QPushButton("📁 Samples Folder")
        self.samples_folder_btn.setToolTip("Open the samples folder in Explorer")
        self.samples_folder_btn.clicked.connect(self.open_samples_folder)
        samples_btn_layout.addWidget(self.samples_folder_btn)
        samples_btn_layout.addStretch()
        layout.addLayout(samples_btn_layout)

        # Initialize tracking variables
        self.last_step_time = None
        self.last_tracked_step = 0
        self.step_times = []
    
    def setup_timers(self):
        """Set up update timers"""
        # Progress update timer
        self.progress_timer = QTimer()
        self.progress_timer.timeout.connect(self.update_progress_display)
        self.progress_timer.start(1000)  # Update every second
    
    # ========================================================================
    # Callback Methods
    # ========================================================================
    
    def update_status(self, message):
        """Thread-safe status update"""
        self.signals.status_updated.emit(message)
    
    def update_status_safe(self, message):
        """Update status text (called from signal)"""
        self.status_text.append(message)
    
    def safe_update_output(self, text):
        """Thread-safe training output update"""
        self.signals.output_updated.emit(text)
    
    def update_training_output_safe(self, text):
        """Update training output (called from signal)"""
        self.training_output.append(text.rstrip())
    
    def handle_training_finished_internal(self, result):
        """Handle training completion (internal)"""
        self.signals.training_finished.emit(result)
    
    def handle_training_error_internal(self, title, message):
        """Handle training error (internal)"""
        logger.error(f"{title}: {message}")
        self.update_status(f"ERROR: {message}")
    
    def show_error(self, title, message):
        """Show error message box"""
        QMessageBox.critical(self, title, message)
    
    def show_info(self, title, message):
        """Show info message box"""
        QMessageBox.information(self, title, message)
    
    # ========================================================================
    # UI Action Methods
    # ========================================================================
    
    def browse_config_file(self):
        """Browse for config file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Config File", "", "TOML Files (*.toml)"
        )
        if file_path:
            self.config_file = file_path
            self.config_file_entry.setText(file_path)
    
    def browse_output_dir(self):
        """Browse for output directory"""
        dir_path = QFileDialog.getExistingDirectory(
            self, "Select Output Directory"
        )
        if dir_path:
            self.output_dir = dir_path
            self.output_dir_entry.setText(dir_path)
    
    def browse_dataset_dir(self):
        """Browse for dataset directory"""
        dir_path = QFileDialog.getExistingDirectory(
            self, "Select Dataset Directory"
        )
        if dir_path:
            self.dataset_dir = dir_path
            self.dataset_dir_entry.setText(dir_path)

            # Load into embedded dataset manager and update calculations
            if hasattr(self, 'embedded_dataset_manager'):
                self.embedded_dataset_manager.load_folder(dir_path)
                self.gui_config.set('paths.dataset_dir', dir_path)
                self.refresh_image_count()

    def browse_comfyui_base(self):
        """Browse for ComfyUI models base directory"""
        dir_path = QFileDialog.getExistingDirectory(
            self, "Select ComfyUI Models Directory"
        )
        if dir_path:
            self.comfyui_base_entry.setText(dir_path)
            self.gui_config.set('paths.comfyui_models_path', dir_path)
            self.auto_detect_model_paths()

    def browse_transformer_path(self):
        """Browse for transformer safetensors file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Transformer Model",
            "", "Safetensors (*.safetensors);;All Files (*)"
        )
        if file_path:
            self.transformer_path_entry.setText(file_path)
            self.gui_config.set('paths.transformer_path', file_path)

    def browse_vae_path(self):
        """Browse for VAE safetensors file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select VAE Model",
            "", "Safetensors (*.safetensors);;All Files (*)"
        )
        if file_path:
            self.vae_path_entry.setText(file_path)
            self.gui_config.set('paths.vae_path', file_path)

    def auto_detect_model_paths(self):
        """Auto-detect model files from ComfyUI base path"""
        import glob

        base = self.comfyui_base_entry.text().strip()
        if not base or not os.path.isdir(base):
            self.model_paths_status.setText("ComfyUI base path not set or invalid.")
            return

        found = []

        # Detect transformer
        model_key = self.model_combo.itemData(self.model_combo.currentIndex())
        variant = '4b' if '4b' in (model_key or '').lower() else '9b'

        unet_dir = os.path.join(base, 'unet')
        if os.path.isdir(unet_dir):
            for pattern in [f'*klein*{variant}*', f'*klein*{variant.upper()}*']:
                matches = glob.glob(os.path.join(unet_dir, pattern))
                if matches:
                    self.transformer_path_entry.setText(matches[0])
                    self.gui_config.set('paths.transformer_path', matches[0])
                    found.append(f"Transformer: {os.path.basename(matches[0])}")
                    break
            else:
                found.append("Transformer: not found in unet/")
        else:
            found.append("Transformer: unet/ dir not found")

        # Detect VAE
        vae_dir = os.path.join(base, 'vae')
        if os.path.isdir(vae_dir):
            for name in ['flux2-vae.safetensors', 'ae.safetensors']:
                candidate = os.path.join(vae_dir, name)
                if os.path.exists(candidate):
                    self.vae_path_entry.setText(candidate)
                    self.gui_config.set('paths.vae_path', candidate)
                    found.append(f"VAE: {name}")
                    break
            else:
                found.append("VAE: not found in vae/")
        else:
            found.append("VAE: vae/ dir not found")

        self.model_paths_status.setText("Auto-detect: " + " | ".join(found))

    def on_claude_api_key_changed(self):
        """Handle Claude API key changes"""
        api_key = self.claude_api_key_entry.text().strip()
        self.gui_config.set('claude.api_key', api_key)
        self.claude_status_label.setText("Not tested")
        self.claude_status_label.setStyleSheet("color: #666;")

    def toggle_claude_key_visibility(self):
        """Toggle API key visibility"""
        if self.claude_api_key_entry.echoMode() == QLineEdit.EchoMode.Password:
            self.claude_api_key_entry.setEchoMode(QLineEdit.EchoMode.Normal)
            self.claude_show_key_btn.setText("Hide")
        else:
            self.claude_api_key_entry.setEchoMode(QLineEdit.EchoMode.Password)
            self.claude_show_key_btn.setText("Show")

    def test_claude_connection(self):
        """Test Claude API connection"""
        api_key = self.claude_api_key_entry.text().strip()

        if not api_key:
            self.claude_status_label.setText("No API key entered")
            self.claude_status_label.setStyleSheet("color: #CC0000;")
            return

        self.claude_status_label.setText("Testing...")
        self.claude_status_label.setStyleSheet("color: #0066CC;")
        self.claude_test_btn.setEnabled(False)

        # Process events to show "Testing..." immediately
        from PyQt6.QtWidgets import QApplication
        QApplication.processEvents()

        try:
            from claude_api import ClaudeAPI, ANTHROPIC_AVAILABLE

            if not ANTHROPIC_AVAILABLE:
                self.claude_status_label.setText("anthropic package not installed")
                self.claude_status_label.setStyleSheet("color: #CC0000;")
                return

            api = ClaudeAPI(api_key)
            success, message = api.test_connection()

            if success:
                self.claude_status_label.setText(message)
                self.claude_status_label.setStyleSheet("color: #00AA00; font-weight: bold;")
            else:
                self.claude_status_label.setText(message)
                self.claude_status_label.setStyleSheet("color: #CC0000;")

        except Exception as e:
            self.claude_status_label.setText(f"Error: {str(e)}")
            self.claude_status_label.setStyleSheet("color: #CC0000;")

        finally:
            self.claude_test_btn.setEnabled(True)

    def load_config(self):
        """Load configuration"""
        config_path = self.config_file_entry.text()
        if self.config_manager.load_config(config_path):
            self.load_config_to_ui()
            self.update_status("Configuration loaded successfully")
    
    def create_default_config(self):
        """Create default configuration"""
        config_path = self.config_file_entry.text()
        if self.config_manager.create_default_config(config_path):
            self.load_config_to_ui()
            self.update_status("Default configuration created")
    
    def save_config(self):
        """Save configuration"""
        self.save_ui_to_config()
        config_path = self.config_file_entry.text()
        if self.config_manager.save_config(config_path):
            self.update_status("Configuration saved successfully")
    
    def load_config_to_ui(self):
        """Load config values into UI"""
        self.lr_spinbox.setValue(self.config_manager.get_value('learning_rate', 0.0001))
        # Steps are now calculated from epochs, not loaded directly
        self.dim_spinbox.setValue(self.config_manager.get_value('network_dim', 16))
        self.alpha_spinbox.setValue(self.config_manager.get_value('network_alpha', 8))
        self.output_name_entry.setText(self.config_manager.get_value('output_name', 'flux_lora'))
        self.output_dir_entry.setText(self.config_manager.get_value('output_dir', './output'))

        # Re-scan dataset folder if it exists (refresh image/caption counts)
        if hasattr(self, 'dataset_dir_entry'):
            dataset_dir = self.dataset_dir_entry.text()
            if dataset_dir and os.path.exists(dataset_dir):
                # Update dataset stats (image and caption counts)
                self.update_dataset_stats()
                # Update the training info with fresh image count
                info = self.gui_config.get_training_info()
                if hasattr(self, 'num_images_label'):
                    self.num_images_label.setText(str(info['num_images']))

        # Update calculated steps display
        if hasattr(self, 'update_steps_calculation'):
            self.update_steps_calculation()
    
    def save_ui_to_config(self):
        """Save UI values to config"""
        self.config_manager.set_value('learning_rate', self.lr_spinbox.value())
        # Steps are calculated automatically from epochs, not saved from UI
        self.config_manager.set_value('network_dim', self.dim_spinbox.value())
        self.config_manager.set_value('network_alpha', self.alpha_spinbox.value())
        self.config_manager.set_value('output_name', self.output_name_entry.text())
        self.config_manager.set_value('output_dir', self.output_dir_entry.text())

        # Save model path settings
        self.gui_config.set('paths.comfyui_models_path', self.comfyui_base_entry.text().strip(), auto_save=False)
        self.gui_config.set('paths.transformer_path', self.transformer_path_entry.text().strip(), auto_save=False)
        self.gui_config.set('paths.vae_path', self.vae_path_entry.text().strip(), auto_save=False)
        self.gui_config.set('paths.text_encoder_repo', self.text_encoder_repo_entry.text().strip(), auto_save=False)
        self.gui_config.save()
    
    def on_model_selected(self, index):
        """Handle model selection with auto-save"""
        if index < 0:
            return
        
        model_key = self.model_combo.itemData(index)
        model_info = MODEL_OPTIONS.get(model_key)
        
        if model_info:
            # Save selection to GUI config
            self.gui_config.set('model.selected', model_key)
            
            # Update description
            desc = f"{model_info['description']}\n"
            desc += f"Backend: {model_info['backend']}\n"
            desc += f"Precision: {model_info['precision']}"
            self.model_description.setText(desc)
            
            # Update config with selected model
            self.config_manager.set_value('pretrained_model_name_or_path', model_info['model_id'])
            
            # Update precision
            if model_info['precision'] == 'bf16':
                self.config_manager.set_value('mixed_precision', 'bf16')
                self.config_manager.set_value('save_precision', 'bf16')
            else:
                self.config_manager.set_value('mixed_precision', 'fp16')
                self.config_manager.set_value('save_precision', 'fp16')
            
            # Update preset list for this model
            self.update_preset_list(model_info['model_id'])
            
            self.update_status(f"Selected model: {model_info['name']}")
    
    def update_preset_list(self, model_id):
        """Update preset dropdown based on selected model"""
        # Clear current presets
        self.preset_combo.clear()
        self.preset_combo.addItem("-- Select Preset --")
        
        # Get presets for this model
        presets = get_presets_for_model(model_id)
        
        for preset_name, preset_data in presets.items():
            display_name = preset_data['name']
            self.preset_combo.addItem(display_name, preset_name)
    
    def on_preset_selected(self, index):
        """Handle preset selection"""
        if index == 0:  # "-- Select Preset --"
            return
        
        preset_key = self.preset_combo.itemData(index)
        
        # Get preset from combined list
        all_presets = get_all_presets()
        preset = all_presets.get(preset_key)
        
        if preset:
            # Update description
            desc = f"{preset['description']}\n\n"
            desc += f"Recommended: {preset['recommended_images']}\n\n"
            desc += "Tips:\n"
            for tip in preset.get('tips', []):
                desc += f"• {tip}\n"
            self.preset_description.setText(desc)
            
            # Apply preset to config
            if apply_preset_to_config(self.config_manager, preset_key):
                self.load_config_to_ui()
                self.update_status(f"Applied preset: {preset['name']}")
    
    def apply_config_changes(self):
        """Apply configuration changes"""
        self.save_ui_to_config()
        self.update_status("Configuration updated")
    
    def on_training_param_changed(self):
        """
        Called when any training parameter changes
        Updates calculated steps and saves config
        """
        # Save values to GUI config
        self.gui_config.set('training.epochs', self.epochs_spinbox.value())
        self.gui_config.set('training.samples_per_image', self.samples_spinbox.value())
        self.gui_config.set('training.learning_rate', self.lr_spinbox.value())
        self.gui_config.set('training.lora_rank', self.dim_spinbox.value())
        self.gui_config.set('training.lora_alpha', self.alpha_spinbox.value())
        self.gui_config.set('training.save_every_n_steps', self.save_every_spinbox.value())
        self.gui_config.set('paths.output_name', self.output_name_entry.text())

        # Save sampling config
        if hasattr(self, 'sample_prompt_text'):
            self.gui_config.set('sampling.prompt', self.sample_prompt_text.toPlainText())
            self.gui_config.set('sampling.sample_every', self.sample_every_spinbox.value())
            self.gui_config.set('sampling.width', self.sample_width_spinbox.value())
            self.gui_config.set('sampling.height', self.sample_height_spinbox.value())
            self.gui_config.set('sampling.seed', self.sample_seed_spinbox.value())
            self.gui_config.set('sampling.random_seed', self.random_seed_checkbox.isChecked())

            # Update sample count estimate
            total_steps = self.gui_config.get_training_info()['total_steps']
            sample_every = self.sample_every_spinbox.value()
            num_samples = total_steps // sample_every if sample_every > 0 else 0
            self.sample_count_label.setText(f"(~{num_samples} samples)")

        # Update calculated steps
        self.update_steps_calculation()
        
        # Flash autosave indicator
        self.autosave_label.setText("✓ Settings saved")
        self.autosave_label.setStyleSheet("color: green; font-weight: bold;")
        QTimer.singleShot(1000, lambda: self.autosave_label.setStyleSheet(
            "color: green; font-style: italic;"
        ))

    def apply_live_changes(self):
        """Apply configuration changes to running training without restart"""
        if not self.training_manager.is_training():
            QMessageBox.information(
                self,
                "Not Training",
                "No training is currently running. Changes will apply when you start training."
            )
            return

        # Gather current values from UI
        learning_rate = self.lr_spinbox.value()
        sample_every = self.sample_every_spinbox.value() if hasattr(self, 'sample_every_spinbox') else None
        sample_prompt = self.sample_prompt_text.toPlainText().strip() if hasattr(self, 'sample_prompt_text') else None
        save_every = self.save_every_spinbox.value()

        # Apply changes
        result = self.training_manager.update_live_config(
            learning_rate=learning_rate,
            sample_every=sample_every,
            sample_prompt=sample_prompt if sample_prompt else None,
            save_every_n_steps=save_every,
        )

        if 'error' in result:
            self.live_update_status.setText(f"Error: {result['error']}")
            self.live_update_status.setStyleSheet("color: #CC0000;")
        else:
            # Build summary of changes
            changes = []
            if 'learning_rate' in result:
                changes.append(f"LR: {result['learning_rate']['old']:.6f} → {result['learning_rate']['new']:.6f}")
            if 'sample_every' in result:
                changes.append(f"Sample freq: {result['sample_every']['old']} → {result['sample_every']['new']}")
            if 'sample_prompt' in result:
                changes.append("Sample prompt updated")
            if 'save_every_n_steps' in result:
                changes.append(f"Save freq: {result['save_every_n_steps']['old']} → {result['save_every_n_steps']['new']}")

            if changes:
                summary = ", ".join(changes)
                self.live_update_status.setText(f"✓ Applied: {summary}")
                self.live_update_status.setStyleSheet("color: #00AA00; font-weight: bold;")
                self.update_status(f"Live config updated: {summary}")

                # Clear status after a few seconds
                QTimer.singleShot(5000, lambda: self.live_update_status.setText(""))
            else:
                self.live_update_status.setText("No changes to apply")
                self.live_update_status.setStyleSheet("color: #666;")

    def update_steps_calculation(self):
        """Update the steps calculation display"""
        # Get training info
        info = self.gui_config.get_training_info()
        
        # Update displays
        self.calculated_steps_label.setText(str(info['total_steps']))
        self.formula_label.setText(info['formula'])
        
        # Estimate time (roughly 10 seconds per step on RTX 4090 with FLUX)
        estimated_seconds = info['total_steps'] * 10
        hours = estimated_seconds // 3600
        minutes = (estimated_seconds % 3600) // 60
        
        if hours > 0:
            time_str = f"(~{hours}h {minutes}m)"
        else:
            time_str = f"(~{minutes} minutes)"
        
        self.estimated_time_label.setText(time_str)
    
    def refresh_image_count(self):
        """Refresh the image count from dataset directory"""
        dataset_dir = self.dataset_dir_entry.text() if hasattr(self, 'dataset_dir_entry') else ""

        if dataset_dir and os.path.exists(dataset_dir):
            info = self.gui_config.get_training_info()
            self.num_images_label.setText(str(info['num_images']))
            self.update_steps_calculation()
            self.update_status(f"Found {info['num_images']} images in dataset")
        else:
            self.num_images_label.setText("40 (estimate)")
            self.update_status("Select dataset directory to count images")

    def refresh_dataset_on_startup(self):
        """Refresh dataset stats and training steps on startup if dataset folder exists"""
        if not hasattr(self, 'dataset_dir_entry'):
            return

        dataset_dir = self.dataset_dir_entry.text()
        if dataset_dir and os.path.exists(dataset_dir):
            # Load dataset into embedded manager (if available)
            if hasattr(self, 'embedded_dataset_manager'):
                self.embedded_dataset_manager.load_folder(dataset_dir)

            # Update training info (image count and steps in Configuration tab)
            info = self.gui_config.get_training_info()
            if hasattr(self, 'num_images_label'):
                self.num_images_label.setText(str(info['num_images']))
            if hasattr(self, 'update_steps_calculation'):
                self.update_steps_calculation()

            logger.info(f"Startup: Refreshed dataset stats - {info['num_images']} images, {info['total_steps']} steps")

    def on_sample_enabled_changed(self, state):
        """Handle sample generation enabled/disabled"""
        enabled = state == Qt.CheckState.Checked.value
        self.gui_config.set('sampling.enabled', enabled)

        # Enable/disable sampling controls
        if hasattr(self, 'sample_prompt_text'):
            self.sample_prompt_text.setEnabled(enabled)
            self.sample_every_spinbox.setEnabled(enabled)
            self.sample_width_spinbox.setEnabled(enabled)
            self.sample_height_spinbox.setEnabled(enabled)
            self.random_seed_checkbox.setEnabled(enabled)
            # Seed spinbox depends on both enabled state and random seed checkbox
            self.sample_seed_spinbox.setEnabled(enabled and not self.random_seed_checkbox.isChecked())

    def on_random_seed_changed(self, state):
        """Handle random seed checkbox state change"""
        is_random = state == Qt.CheckState.Checked.value
        self.gui_config.set('sampling.random_seed', is_random)
        # Disable seed spinbox when random seed is enabled
        self.sample_seed_spinbox.setEnabled(not is_random)
        self.on_training_param_changed()

    def open_samples_folder(self):
        """Open the samples folder in Explorer"""
        output_name = self.output_name_entry.text() if hasattr(self, 'output_name_entry') else 'flux_lora'
        output_dir = self.output_dir if hasattr(self, 'output_dir') else './output'
        samples_dir = os.path.join(output_dir, output_name, 'samples')

        # Create folder if it doesn't exist
        if not os.path.exists(samples_dir):
            os.makedirs(samples_dir, exist_ok=True)

        # Open in Explorer
        import subprocess
        subprocess.Popen(['explorer', os.path.abspath(samples_dir)])

    def update_dataset_stats(self):
        """Update dataset statistics - now handled by embedded dataset manager"""
        # This method is kept for compatibility but delegates to the embedded manager
        if hasattr(self, 'embedded_dataset_manager'):
            folder = self.dataset_dir_entry.text()
            if folder and os.path.isdir(folder):
                self.embedded_dataset_manager.load_folder(folder)
    
    def start_training(self):
        """Start training"""
        # Save current config
        self.save_ui_to_config()
        
        # Apply GUI config values (epochs-based calculation)
        self.gui_config.apply_to_training_config(self.config_manager)
        logger.info(f"Applied GUI config: {self.gui_config.calculate_steps()} steps")
        
        # Update output directory
        output_dir = self.output_dir_entry.text()
        self.config_manager.set_value('output_dir', output_dir)
        
        # Set dataset directory in dataset config
        dataset_dir = self.dataset_dir_entry.text()
        if not dataset_dir:
            self.show_error("Error", "Please select a dataset directory")
            return
        
        # Save dataset directory to GUI config
        self.gui_config.set('paths.dataset_dir', dataset_dir)
        
        self.dataset_manager.set_image_directory(dataset_dir)
        self.dataset_manager.save_dataset_config(self.dataset_config_file)
        self.config_manager.set_value('dataset_config', self.dataset_config_file)
        
        # Use enhanced training manager (auto-detects backend)
        if self.training_manager.start_training(self.config_manager, self.dataset_manager):
            self.start_training_btn.setEnabled(False)
            self.stop_training_btn.setEnabled(True)
            self.apply_changes_btn.setEnabled(True)  # Enable live config updates

            # Show which backend is being used
            backend = self.training_manager.get_active_backend()
            backend_name = "ai-toolkit (FLUX)" if backend == 'aitoolkit' else "sd-scripts (SD/SDXL)"
            steps = self.gui_config.calculate_steps()
            self.update_status(f"Training started with {backend_name} ({steps} steps)...")
            self.training_output.clear()
    
    def stop_training(self):
        """Stop training"""
        if self.training_manager.stop_training():
            self.update_status("Stopping training...")
    
    def handle_training_finished(self, result):
        """Handle training completion"""
        self.start_training_btn.setEnabled(True)
        self.stop_training_btn.setEnabled(False)
        self.apply_changes_btn.setEnabled(False)  # Disable live config updates
        self.live_update_status.setText("")

        status = result.get('status', 'unknown')
        
        if status == 'success':
            self.show_info("Training Complete", "Training completed successfully!")
        elif status == 'stopped':
            self.show_info("Training Stopped", "Training was stopped by user")
        else:
            self.show_error("Training Failed", "Training failed. Check output for details.")
    
    def update_progress_display(self):
        """Update progress display with enhanced metrics"""
        if not self.training_manager.is_training_running():
            return
        
        import time
        current_time = time.time()
        
        progress = self.training_manager.get_progress()
        
        # Update progress bar
        if progress['total_steps'] > 0:
            self.progress_bar.setValue(int(progress['progress_percent']))
            self.progress_label.setText(
                f"Step {progress['current_step']}/{progress['total_steps']} "
                f"({progress['progress_percent']:.1f}%)"
            )
        
        # Update step metrics
        self.current_step_label.setText(str(progress['current_step']))
        self.total_steps_label.setText(str(progress['total_steps']))
        
        # Update loss
        current_loss = progress['last_loss']
        self.last_loss_label.setText(f"{current_loss:.4f}")
        
        # Color code loss
        if current_loss > 1.0:
            self.last_loss_label.setStyleSheet("font-weight: bold; font-size: 12pt; color: #CC0000;")  # Red
        elif current_loss > 0.5:
            self.last_loss_label.setStyleSheet("font-weight: bold; font-size: 12pt; color: #FF6600;")  # Orange
        else:
            self.last_loss_label.setStyleSheet("font-weight: bold; font-size: 12pt; color: #00AA00;")  # Green
        
        # Calculate average loss (last 100 steps)
        loss_history = self.training_manager.get_loss_history()
        if loss_history and len(loss_history) > 0:
            recent_losses = [entry[1] for entry in loss_history[-100:]]
            avg_loss = sum(recent_losses) / len(recent_losses)
            self.avg_loss_label.setText(f"{avg_loss:.4f}")
        
        # Calculate speed metrics
        if progress['current_step'] > 0:
            # Track step times
            if self.last_step_time is not None and progress['current_step'] != self.last_tracked_step:
                step_duration = current_time - self.last_step_time
                self.step_times.append(step_duration)
                if len(self.step_times) > 10:
                    self.step_times.pop(0)
            
            if progress['current_step'] != getattr(self, 'last_tracked_step', 0):
                self.last_step_time = current_time
                self.last_tracked_step = progress['current_step']
            
            # Calculate average time per step
            if self.step_times:
                avg_time_per_step = sum(self.step_times) / len(self.step_times)
                self.time_per_step_label.setText(f"{avg_time_per_step:.1f} sec")
                
                # Calculate speed
                steps_per_sec = 1.0 / avg_time_per_step if avg_time_per_step > 0 else 0
                self.speed_label.setText(f"{steps_per_sec:.3f} steps/sec")
                
                # Calculate ETA
                remaining_steps = progress['total_steps'] - progress['current_step']
                eta_seconds = remaining_steps * avg_time_per_step
                
                hours = int(eta_seconds // 3600)
                minutes = int((eta_seconds % 3600) // 60)
                seconds = int(eta_seconds % 60)
                
                if hours > 0:
                    self.eta_label.setText(f"{hours}h {minutes}m {seconds}s")
                elif minutes > 0:
                    self.eta_label.setText(f"{minutes}m {seconds}s")
                else:
                    self.eta_label.setText(f"{seconds}s")
        
        # Calculate elapsed time
        if hasattr(self.training_manager, 'training_start_time') and self.training_manager.training_start_time:
            elapsed = current_time - self.training_manager.training_start_time
            hours = int(elapsed // 3600)
            minutes = int((elapsed % 3600) // 60)
            seconds = int(elapsed % 60)
            self.elapsed_time_label.setText(f"{hours}:{minutes:02d}:{seconds:02d}")
        
        # Update progress percentage
        self.progress_percent_label.setText(f"{progress['progress_percent']:.1f}%")

        # Update loss graph
        self.update_loss_graph()

        # Update outlier panel (every 10 steps to avoid overhead)
        if progress['current_step'] % 10 == 0:
            self.update_outlier_panel()

    def update_outlier_panel(self):
        """Update the problematic images panel with outlier data"""
        if not hasattr(self, 'outlier_list'):
            return

        # Only works with Klein backend
        if not hasattr(self.training_manager, 'active_backend') or self.training_manager.active_backend != 'klein':
            return

        klein_mgr = getattr(self.training_manager, 'klein_manager', None)
        if klein_mgr is None:
            return

        outliers = klein_mgr.get_outlier_images(top_n=5)

        if not outliers:
            self.outlier_info_label.setText("No significant outliers detected yet")
            self.outlier_list.clear()
            return

        # Check if any are actual outliers (z_score > 0)
        actual_outliers = [o for o in outliers if o.get('z_score', 0) > 0]

        if actual_outliers:
            self.outlier_info_label.setText(
                f"Found {len(actual_outliers)} image(s) with unusually high loss:"
            )
            self.outlier_info_label.setStyleSheet("color: #CC6600; font-weight: bold;")
        else:
            self.outlier_info_label.setText("Top images by loss (no significant outliers):")
            self.outlier_info_label.setStyleSheet("color: #666; font-style: italic;")

        # Format outlier list
        lines = []
        for item in outliers:
            filename = item.get('filename', 'Unknown')
            mean_loss = item.get('mean_loss', 0)
            z_score = item.get('z_score', 0)

            if z_score > 0:
                lines.append(f"[!] {filename}: {mean_loss:.4f} (z={z_score:.1f})")
            else:
                lines.append(f"    {filename}: {mean_loss:.4f}")

        self.outlier_list.setText("\n".join(lines))

    def analyze_training_with_claude(self):
        """Analyze current training run with Claude AI"""
        # Check if API key is configured
        api_key = self.gui_config.get('claude.api_key', '')
        if not api_key:
            QMessageBox.warning(
                self,
                "API Key Required",
                "Please enter your Claude API key in the Setup tab first."
            )
            return

        # Check if training is running or has data
        if not hasattr(self.training_manager, 'klein_manager'):
            QMessageBox.information(
                self,
                "No Training Data",
                "Start a training run first to generate data for analysis."
            )
            return

        klein_mgr = getattr(self.training_manager, 'klein_manager', None)
        if klein_mgr is None:
            QMessageBox.information(
                self,
                "No Training Data",
                "No Klein training manager available."
            )
            return

        # Disable button while analyzing
        self.analyze_with_claude_btn.setEnabled(False)
        self.analyze_with_claude_btn.setText("🤖 Analyzing...")

        # Run analysis in a thread to avoid blocking UI
        import threading
        thread = threading.Thread(target=self._run_claude_analysis, args=(api_key, klein_mgr))
        thread.daemon = True
        thread.start()

    def _run_claude_analysis(self, api_key, klein_mgr):
        """Run Claude analysis in background thread"""
        try:
            from claude_api import ClaudeAPI
            import io

            # Initialize Claude API
            claude = ClaudeAPI(api_key)

            # Get current training status
            progress = klein_mgr.get_progress()
            current_status = {
                'current_step': progress.get('current_step', 0),
                'total_steps': progress.get('total_steps', 0),
                'progress_percent': progress.get('percent', 0),
                'loss': progress.get('loss', 0),
                'trend_status': 'unknown',
            }

            # Get trend info
            trend_info = klein_mgr.get_trend_info()
            if trend_info:
                current_status['trend_status'] = trend_info.get('status', 'unknown')

            # Build metrics summary
            loss_history = klein_mgr.get_loss_history()
            metrics_summary = {}
            if loss_history:
                losses = [entry[1] for entry in loss_history]
                metrics_summary = {
                    'total_steps': len(loss_history),
                    'avg_loss': sum(losses) / len(losses) if losses else 0,
                    'min_loss': min(losses) if losses else 0,
                    'max_loss': max(losses) if losses else 0,
                }
                # Add trend info
                if trend_info:
                    metrics_summary['ema_10'] = trend_info.get('ema_10')
                    metrics_summary['ema_50'] = trend_info.get('ema_50')
                    metrics_summary['ema_100'] = trend_info.get('ema_100')
                    metrics_summary['trend_slope'] = trend_info.get('slope')

            # Get high-loss images
            high_loss_images = klein_mgr.get_outlier_images(top_n=10)

            # Capture the loss graph as PNG
            graph_image_bytes = None
            try:
                buf = io.BytesIO()
                self.performance_figure.savefig(buf, format='png', dpi=100, bbox_inches='tight')
                buf.seek(0)
                graph_image_bytes = buf.read()
                buf.close()
            except Exception as e:
                logger.warning(f"Failed to capture loss graph: {e}")

            # Call Claude API
            success, result = claude.analyze_training(
                metrics_summary=metrics_summary,
                graph_image_bytes=graph_image_bytes,
                high_loss_images=high_loss_images,
                current_status=current_status,
            )

            # Update UI via signal (thread-safe)
            self.signals.claude_analysis_done.emit(success, result)

        except ImportError as e:
            error_msg = f"Failed to import Claude API: {e}\n\nMake sure anthropic package is installed:\npip install anthropic"
            self.signals.claude_analysis_done.emit(False, error_msg)
        except Exception as e:
            logger.error(f"Claude analysis failed: {e}")
            self.signals.claude_analysis_done.emit(False, f"Analysis failed: {str(e)}")

    def _update_claude_analysis_results(self, success: bool, result: str):
        """Show Claude analysis results in a popup dialog and save to file"""
        self.analyze_with_claude_btn.setEnabled(True)
        self.analyze_with_claude_btn.setText("🤖 Analyze with Claude")

        saved_path = None

        if success:
            # Save analysis to output folder
            try:
                from datetime import datetime
                output_dir = self.gui_config.get('paths.output_dir', './output')
                output_name = self.gui_config.get('paths.output_name', 'flux_lora')

                # Create output directory if it doesn't exist
                os.makedirs(output_dir, exist_ok=True)

                # Generate filename with timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{output_name}_claude_analysis_{timestamp}.txt"
                filepath = os.path.join(output_dir, filename)

                # Write analysis to file
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(f"Claude Training Analysis\n")
                    f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"{'=' * 60}\n\n")
                    f.write(result)

                saved_path = filepath
                logger.info(f"Saved Claude analysis to {filepath}")
            except Exception as e:
                logger.error(f"Failed to save Claude analysis: {e}")

            # Create analysis dialog
            dialog = QDialog(self)
            dialog.setWindowTitle("Claude Training Analysis")
            dialog.setMinimumSize(600, 500)
            dialog.resize(700, 600)

            layout = QVBoxLayout(dialog)

            # Header
            header = QLabel("🤖 AI Training Analysis")
            header.setStyleSheet("font-size: 16pt; font-weight: bold; color: #7c4dff; padding: 10px;")
            layout.addWidget(header)

            # Saved path info
            if saved_path:
                saved_label = QLabel(f"📁 Saved to: {saved_path}")
                saved_label.setStyleSheet("color: #00AA00; font-size: 10pt; padding: 5px;")
                saved_label.setWordWrap(True)
                layout.addWidget(saved_label)

            # Results text
            results_text = QTextEdit()
            results_text.setReadOnly(True)
            results_text.setFont(self.mono_font)
            results_text.setText(result)
            layout.addWidget(results_text)

            # Close button
            close_btn = QPushButton("Close")
            close_btn.clicked.connect(dialog.accept)
            close_btn.setMinimumHeight(35)
            layout.addWidget(close_btn)

            dialog.exec()
        else:
            QMessageBox.warning(
                self,
                "Analysis Failed",
                f"Claude analysis failed:\n\n{result}"
            )

    def update_gpu_stats(self):
        """Update GPU statistics using nvidia-smi"""
        try:
            import subprocess
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw',
                 '--format=csv,noheader,nounits'],
                capture_output=True,
                text=True,
                timeout=1
            )
            
            if result.returncode == 0:
                values = result.stdout.strip().split(',')
                if len(values) >= 5:
                    gpu_util = values[0].strip()
                    vram_used = float(values[1].strip())
                    vram_total = float(values[2].strip())
                    temp = values[3].strip()
                    power = values[4].strip()
                    
                    self.gpu_util_label.setText(f"{gpu_util}%")
                    self.vram_used_label.setText(f"{vram_used/1024:.1f} / {vram_total/1024:.1f} GB")
                    self.gpu_temp_label.setText(f"{temp}°C")
                    self.gpu_power_label.setText(f"{power} W")
                    
                    # Color code VRAM usage
                    vram_percent = (vram_used / vram_total) * 100
                    if vram_percent > 95:
                        self.vram_used_label.setStyleSheet("color: #CC0000; font-weight: bold;")
                    elif vram_percent > 85:
                        self.vram_used_label.setStyleSheet("color: #FF6600; font-weight: bold;")
                    else:
                        self.vram_used_label.setStyleSheet("color: #00AA00; font-weight: bold;")
        except:
            pass  # GPU monitoring not critical
    
    def update_loss_graph(self):
        """Update loss history graph with EMA lines and trend analysis"""
        loss_history = self.training_manager.get_loss_history()

        if not loss_history or len(loss_history) < 2:
            return

        steps = [entry[0] for entry in loss_history]
        losses = [entry[1] for entry in loss_history]

        self.performance_ax.clear()

        # Plot raw loss line (thinner, semi-transparent)
        self.performance_ax.plot(steps, losses, 'b-', linewidth=1, alpha=0.4, label='Loss')

        # Try to get EMA data from analytics (Klein backend)
        ema_plotted = False
        trend_info = {"status": "unknown"}

        if hasattr(self.training_manager, 'active_backend') and self.training_manager.active_backend == 'klein':
            klein_mgr = getattr(self.training_manager, 'klein_manager', None)
            if klein_mgr is not None:
                # Get EMA series
                ema_10 = klein_mgr.get_ema_series(10)
                ema_50 = klein_mgr.get_ema_series(50)
                ema_100 = klein_mgr.get_ema_series(100)

                if ema_10 and len(ema_10) > 5:
                    s, v = zip(*ema_10)
                    self.performance_ax.plot(s, v, 'g-', linewidth=2, alpha=0.8, label='EMA-10')
                    ema_plotted = True

                if ema_50 and len(ema_50) > 20:
                    s, v = zip(*ema_50)
                    self.performance_ax.plot(s, v, color='orange', linewidth=2, alpha=0.8, label='EMA-50')

                if ema_100 and len(ema_100) > 50:
                    s, v = zip(*ema_100)
                    self.performance_ax.plot(s, v, 'r-', linewidth=2, alpha=0.8, label='EMA-100')

                # Get trend info
                trend_info = klein_mgr.get_trend_info()

        # Fallback to simple moving average if no EMA available
        if not ema_plotted and len(losses) > 10:
            window = min(20, len(losses) // 4)
            moving_avg = []
            for i in range(len(losses)):
                start = max(0, i - window + 1)
                avg = sum(losses[start:i + 1]) / (i - start + 1)
                moving_avg.append(avg)
            self.performance_ax.plot(steps, moving_avg, 'r--', linewidth=1.5,
                                     alpha=0.7, label=f'Moving Avg ({window})')

        # Add target loss line
        if losses:
            self.performance_ax.axhline(y=0.5, color='g', linestyle=':',
                                        linewidth=1, alpha=0.5, label='Target (<0.5)')

            if max(losses) > 2.0:
                self.performance_ax.axhline(y=2.0, color='r', linestyle=':',
                                            linewidth=1, alpha=0.5, label='High Loss Warning')

        # Styling
        self.performance_ax.set_xlabel('Step', fontsize=12)
        self.performance_ax.set_ylabel('Loss', fontsize=12)

        # Dynamic title based on trend status
        trend_status = trend_info.get("status", "unknown")
        trend_colors = {
            "improving": "green",
            "stable": "blue",
            "degrading": "red",
            "converged": "purple",
            "unknown": "gray"
        }
        trend_labels = {
            "improving": "IMPROVING",
            "stable": "STABLE",
            "degrading": "DEGRADING",
            "converged": "CONVERGED",
            "unknown": "Analyzing..."
        }

        title_color = trend_colors.get(trend_status, "black")
        title_label = trend_labels.get(trend_status, trend_status.upper())
        title = f'Training Loss - {title_label}'
        self.performance_ax.set_title(title, fontsize=14, fontweight='bold', color=title_color)

        # Update trend status label in metrics
        if hasattr(self, 'trend_status_label'):
            self.trend_status_label.setText(title_label)
            style_colors = {
                "improving": "#00AA00",
                "stable": "#0066CC",
                "degrading": "#CC0000",
                "converged": "#9900CC",
                "unknown": "#666666"
            }
            self.trend_status_label.setStyleSheet(
                f"font-weight: bold; font-size: 12pt; color: {style_colors.get(trend_status, '#666666')};"
            )

        self.performance_ax.grid(True, alpha=0.3)
        self.performance_ax.legend(loc='upper right', fontsize=9)

        # Auto-scale axes
        if losses and steps:
            y_min = min(0, min(losses) - 0.1)
            y_max = max(losses) + 0.2
            self.performance_ax.set_ylim(y_min, y_max)

            x_max = max(steps) + max(10, max(steps) * 0.05)
            self.performance_ax.set_xlim(0, x_max)

        self.performance_figure.tight_layout()
        self.performance_canvas.draw()
