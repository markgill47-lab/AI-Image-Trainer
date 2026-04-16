#!/usr/bin/env python3
"""
Simplified Event Handlers - Focus on description copy/paste workflow
"""

import os
from PyQt6.QtWidgets import (
    QFileDialog, QMessageBox, QTableWidgetItem, QMenu, QInputDialog,
    QProgressDialog, QApplication
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap, QColor

from PIL import Image

# Handle both package and standalone imports
try:
    from .dialogs import ImageFixDialog, ImageDuplicateDialog
except ImportError:
    from dialogs import ImageFixDialog, ImageDuplicateDialog


class EventHandlers:
    """Handles all user interface events and interactions"""
    
    def __init__(self, app):
        self.app = app
    
    def select_folder(self):
        """Open file dialog to select a folder with images"""
        folder_path = QFileDialog.getExistingDirectory(self.app, "Select Folder with Images")
        
        if not folder_path:
            return
            
        success, message, images_data = self.app.data_manager.load_images_from_folder(folder_path)
        
        if success:
            self.populate_table(images_data)
            
            # Select first image if available
            if images_data:
                self.app.gallery_tab.table.selectRow(0)
            
            # Enable Utils tab
            self.app.tab_widget.setTabEnabled(1, True)
            
            self.app.set_status(f"{message} • Utils tab now available")
        else:
            QMessageBox.critical(self.app, "Load Error", message)
    
    def populate_table(self, images_data):
        """Populate the table with image data"""
        table = self.app.gallery_tab.table
        table.setRowCount(0)

        for row_position, img_data in enumerate(images_data):
            table.insertRow(row_position)

            # Filename column
            filename_item = QTableWidgetItem(img_data['filename'])

            # Apply quality color coding
            quality = img_data.get('quality')
            if quality == 'good':
                filename_item.setForeground(QColor('#228B22'))  # Forest green
            elif quality == 'ok':
                filename_item.setForeground(QColor('#DAA520'))  # Goldenrod/yellow
            elif quality == 'bad':
                filename_item.setForeground(QColor('#DC143C'))  # Crimson red

            table.setItem(row_position, 0, filename_item)

            # Description column - simple text
            description = img_data.get('description', '')
            desc_item = QTableWidgetItem(description)
            desc_item.setData(Qt.ItemDataRole.UserRole, img_data['filename'])
            table.setItem(row_position, 1, desc_item)

            # Update row index
            img_data['row_index'] = row_position

        # Update selection info
        self.update_selection_info()

        # Update utils tab
        self.app.utils_tab.update_scope_info(0, len(images_data))
    
    def on_table_select(self):
        """Handle table selection change"""
        selected_rows = self.app.gallery_tab.table.selectionModel().selectedRows()
        selected_count = len(selected_rows)
        total_count = len(self.app.data_manager.images_data)
        
        # Update utils tab scope
        self.app.utils_tab.update_scope_info(selected_count, total_count)
        
        # Update selection info
        self.update_selection_info()
        
        if selected_count == 0:
            self.app.current_image_index = -1
            self._clear_image_display()
            return
        
        if selected_count == 1:
            # Single selection - show image and description
            row = selected_rows[0].row()
            img_data = self.app.data_manager.images_data[row]
            self.app.current_image_index = row
            self.show_selected_image()
        else:
            # Multiple selection - show count
            self.app.current_image_index = -1
            self._show_multiple_selection(selected_count)
    
    def show_selected_image(self):
        """Display the currently selected image"""
        if self.app.current_image_index < 0:
            return
        
        img_data = self.app.data_manager.get_image_data(self.app.current_image_index)
        if not img_data:
            return
        
        # Load and display image
        try:
            pixmap = QPixmap(img_data['path'])
            if not pixmap.isNull():
                # Scale to a fixed max size to prevent the label from growing unbounded.
                # Using a capped QSize avoids the feedback loop where setPixmap enlarges
                # the label, which then enlarges the next scaled pixmap, etc.
                from PyQt6.QtCore import QSize
                max_preview = QSize(600, 400)
                scaled_pixmap = pixmap.scaled(
                    max_preview,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )
                self.app.gallery_tab.image_label.setPixmap(scaled_pixmap)
        except Exception as e:
            self.app.gallery_tab.image_label.setText(f"Error loading image:\n{str(e)}")
        
        # Load description
        description = img_data.get('description', '')
        self.app.gallery_tab.description_text.blockSignals(True)
        self.app.gallery_tab.description_text.setPlainText(description)
        self.app.gallery_tab.description_text.blockSignals(False)
        
        self.app.set_status(f"Viewing: {img_data['filename']}")
    
    def _clear_image_display(self):
        """Clear the image display"""
        self.app.gallery_tab.image_label.clear()
        self.app.gallery_tab.image_label.setText("No image selected")
        self.app.gallery_tab.description_text.clear()
    
    def _show_multiple_selection(self, count):
        """Show info for multiple selection"""
        self.app.gallery_tab.image_label.clear()
        self.app.gallery_tab.image_label.setText(
            f"📋 {count} images selected\n\n"
            f"Use 'Paste to Selected' to apply description\n"
            f"or Utils tab for batch operations"
        )
        self.app.gallery_tab.description_text.clear()
        self.app.gallery_tab.description_text.setPlaceholderText(
            f"{count} images selected - edit one at a time or paste to all"
        )
    
    def update_selection_info(self):
        """Update the selection info label"""
        selected_rows = self.app.gallery_tab.table.selectionModel().selectedRows()
        total = len(self.app.data_manager.images_data)
        selected = len(selected_rows)
        
        if selected == 0:
            info = f"Total: {total} images"
        elif selected == 1:
            info = f"1 image selected (of {total})"
        else:
            info = f"{selected} images selected (of {total})"
        
        self.app.gallery_tab.selection_label.setText(info)
    
    def on_description_changed(self):
        """Handle description text changes"""
        if self.app.current_image_index < 0:
            return
        
        new_description = self.app.gallery_tab.description_text.toPlainText()
        self.app.data_manager.update_description(self.app.current_image_index, new_description)
        
        # Update table cell
        self.app.gallery_tab.table.item(self.app.current_image_index, 1).setText(new_description)
    
    def copy_description(self):
        """Copy description from current image to clipboard"""
        if self.app.current_image_index < 0:
            QMessageBox.information(self.app, "No Selection", 
                                   "Please select an image first to copy its description.")
            return
        
        img_data = self.app.data_manager.get_image_data(self.app.current_image_index)
        if img_data:
            self.app.description_clipboard = img_data.get('description', '')
            
            # Update clipboard label
            preview = self.app.description_clipboard[:50]
            if len(self.app.description_clipboard) > 50:
                preview += "..."
            self.app.gallery_tab.clipboard_label.setText(f"Clipboard: {preview}")
            
            self.app.set_status(f"Copied description from: {img_data['filename']}")
            QMessageBox.information(self.app, "Copied", 
                                   f"Description copied to clipboard!\n\n"
                                   f"You can now paste it to other images.")
    
    def paste_description(self):
        """Paste description to selected images"""
        if not self.app.description_clipboard:
            QMessageBox.information(self.app, "Empty Clipboard",
                                   "Clipboard is empty. Copy a description first using 'Copy Description'.")
            return
        
        selected_rows = self.app.gallery_tab.table.selectionModel().selectedRows()
        if not selected_rows:
            QMessageBox.information(self.app, "No Selection",
                                   "Please select one or more images to paste the description to.")
            return
        
        # Confirm if pasting to multiple images
        if len(selected_rows) > 1:
            reply = QMessageBox.question(
                self.app, "Confirm Paste",
                f"Paste description to {len(selected_rows)} selected images?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
        
        # Paste to all selected
        pasted_count = 0
        for row_model in selected_rows:
            row = row_model.row()
            self.app.data_manager.update_description(row, self.app.description_clipboard)
            
            # Update table
            self.app.gallery_tab.table.item(row, 1).setText(self.app.description_clipboard)
            pasted_count += 1
        
        # Refresh current image if it was one of the selected
        if self.app.current_image_index >= 0 and self.app.current_image_index in [r.row() for r in selected_rows]:
            self.app.gallery_tab.description_text.blockSignals(True)
            self.app.gallery_tab.description_text.setPlainText(self.app.description_clipboard)
            self.app.gallery_tab.description_text.blockSignals(False)
        
        self.app.set_status(f"Pasted description to {pasted_count} image(s)")
        QMessageBox.information(self.app, "Success",
                               f"Description pasted to {pasted_count} image(s)")
    
    def show_append_dialog(self):
        """Show dialog to append text to all descriptions"""
        text, ok = QInputDialog.getText(
            self.app, "Append to All Descriptions",
            "Enter text to append to all image descriptions:",
            text=""
        )
        
        if ok and text.strip():
            count = self.app.data_manager.append_keyword_to_all(text.strip())
            
            # Refresh table
            self.refresh_gallery()
            
            QMessageBox.information(self.app, "Success",
                                   f"Appended text to {count} descriptions")
            self.app.set_status(f"Appended text to {count} descriptions")
    
    def save_descriptions(self):
        """Save all descriptions"""
        if not self.app.data_manager.images_data:
            QMessageBox.information(self.app, "No Data", "No images loaded to save.")
            return
        
        success, message = self.app.data_manager.save_descriptions()
        
        if success:
            QMessageBox.information(self.app, "Saved", message)
            self.app.set_status(message)
        else:
            QMessageBox.critical(self.app, "Save Error", message)
    
    def refresh_gallery(self):
        """Refresh the gallery display"""
        if self.app.data_manager.images_data:
            self.populate_table(self.app.data_manager.images_data)
            self.app.set_status("Gallery refreshed")
    
    def show_context_menu(self, position):
        """Show right-click context menu"""
        menu = QMenu()
        
        remove_action = menu.addAction("🗑️ Remove from List")
        
        action = menu.exec(self.app.gallery_tab.table.viewport().mapToGlobal(position))
        
        if action == remove_action:
            self.remove_selected_images()
    
    def remove_selected_images(self):
        """Remove selected images from the list"""
        selected_rows = self.app.gallery_tab.table.selectionModel().selectedRows()
        if not selected_rows:
            return
        
        reply = QMessageBox.question(
            self.app, "Confirm Removal",
            f"Remove {len(selected_rows)} image(s) from the list?\n\n"
            f"(This only removes them from the current session, not from disk)",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply != QMessageBox.StandardButton.Yes:
            return
        
        # Remove in reverse order to maintain indices
        for row_model in sorted(selected_rows, reverse=True):
            row = row_model.row()
            self.app.data_manager.remove_image(row)
        
        self.refresh_gallery()
        self.app.set_status(f"Removed {len(selected_rows)} image(s) from list")
    
    def show_fix_dialog(self):
        """Show Fix Images dialog"""
        if not self.app.data_manager.images_data:
            QMessageBox.information(self.app, "No Images", 
                                   "Please load images in the Gallery tab first.")
            return
        
        dialog = ImageFixDialog(self.app, self.app.data_manager.current_folder)
        
        if dialog.exec():
            self.process_fix_images(dialog.result)
    
    def process_fix_images(self, config):
        """Process images with the fix configuration"""
        # Determine which images to process
        if self.app.utils_tab.fix_selected_radio.isChecked():
            selected_images = self.get_selected_images()
            if not selected_images:
                QMessageBox.information(self.app, "No Selection",
                                       "No images selected. Please select images first.")
                return
            images_to_process = [img['filename'] for img in selected_images]
        else:
            images_to_process = None  # Process all
        
        # Process
        result = self.app.image_processor.fix_images(
            source_folder=config['source_folder'],
            target_size=config['size'],
            output_folder=config['output_folder'],
            images_to_process=images_to_process
        )
        
        if 'error' in result:
            QMessageBox.critical(self.app, "Error", result['error'])
        else:
            message = f"Processed: {result.get('processed', 0)} images\n"
            message += f"Skipped: {result.get('skipped', 0)} images"
            if result.get('invalid'):
                message += f"\nInvalid: {len(result['invalid'])} images"
            
            QMessageBox.information(self.app, "Complete", message)
    
    def mass_rename(self):
        """Mass rename images"""
        if not self.app.data_manager.images_data:
            QMessageBox.information(self.app, "No Images",
                                   "Please load images in the Gallery tab first.")
            return
        
        prefix = self.app.utils_tab.prefix_entry.text().strip()
        if not prefix:
            QMessageBox.warning(self.app, "Missing Prefix",
                               "Please enter a prefix for the new filenames.")
            return
        
        # Determine scope - pass full image data dictionaries, not just filenames
        if self.app.utils_tab.rename_selected_radio.isChecked():
            selected_images = self.get_selected_images()
            if not selected_images:
                QMessageBox.information(self.app, "No Selection",
                                       "No images selected.")
                return
            images_to_rename = selected_images  # Pass the full image data dicts
        else:
            images_to_rename = self.app.data_manager.images_data  # Pass all image data
        
        scramble = self.app.utils_tab.scramble_order_check.isChecked()
        
        # Confirm
        scramble_text = " with order scrambling" if scramble else ""
        reply = QMessageBox.question(
            self.app, "Confirm Rename",
            f"Rename {len(images_to_rename)} images{scramble_text}?\n\n"
            f"Prefix: {prefix}\n\n"
            f"This will modify filenames and associated .txt files.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply != QMessageBox.StandardButton.Yes:
            return
        
        # Process
        result = self.app.image_processor.mass_rename_images(
            folder_path=self.app.data_manager.current_folder,
            prefix=prefix,
            images_to_process=images_to_rename,  # Correct parameter name
            scramble_order=scramble
        )
        
        if result.get('success'):
            QMessageBox.information(self.app, "Complete",
                                   f"Renamed {result.get('renamed', 0)} images")
            # Reload folder
            self.select_folder()
        else:
            QMessageBox.critical(self.app, "Error", result.get('error', 'Unknown error'))
    
    def show_duplicate_dialog(self):
        """Show Create Duplicates dialog"""
        if not self.app.data_manager.images_data:
            QMessageBox.information(self.app, "No Images",
                                   "Please load images in the Gallery tab first.")
            return
        
        dialog = ImageDuplicateDialog(self.app, self.app.data_manager.current_folder)
        
        if dialog.exec():
            self.process_duplicates(dialog.result)
    
    def process_duplicates(self, config):
        """Process image duplication"""
        # Determine scope
        if self.app.utils_tab.aug_selected_radio.isChecked():
            selected_images = self.get_selected_images()
            if not selected_images:
                QMessageBox.information(self.app, "No Selection",
                                       "No images selected.")
                return
            images_to_process = [img['filename'] for img in selected_images]
        else:
            images_to_process = None
        
        # Process
        result = self.app.image_processor.create_augmented_dataset(
            input_folder=config['input_folder'],
            output_folder=config['output_folder'],
            flip_horizontal=config['flip_horizontal'],
            rotate_90_left=config['rotate_90_left'],
            rotate_90_right=config['rotate_90_right'],
            rotate_180=config['rotate_180'],
            images_to_process=images_to_process
        )
        
        if 'error' in result:
            QMessageBox.critical(self.app, "Error", result['error'])
        else:
            message = f"Original images: {result.get('original', 0)}\n"
            message += f"Augmented images: {result.get('augmented', 0)}\n"
            message += f"Total output: {result.get('total', 0)}"
            
            QMessageBox.information(self.app, "Complete", message)
    
    def get_selected_images(self):
        """Get list of selected image data"""
        selected_rows = self.app.gallery_tab.table.selectionModel().selectedRows()
        return [self.app.data_manager.images_data[row.row()] for row in selected_rows]

    def generate_captions_claude(self):
        """Generate captions for images using Claude API"""
        if not self.app.data_manager.images_data:
            QMessageBox.information(self.app, "No Images",
                                   "Please load images in the Gallery tab first.")
            return

        # Get API key from parent app's config
        try:
            # Try to access gui_config through the status_callback's parent
            parent = self.app
            while parent is not None:
                if hasattr(parent, 'gui_config'):
                    api_key = parent.gui_config.get('claude.api_key', '')
                    default_prompt = parent.gui_config.get('claude.caption_prompt', '')
                    break
                parent = getattr(parent, 'parent_app', None) or getattr(parent, 'parent', None)
            else:
                api_key = ''
                default_prompt = ''
        except Exception:
            api_key = ''
            default_prompt = ''

        if not api_key:
            QMessageBox.warning(self.app, "No API Key",
                               "Please enter your Claude API key in the Setup tab first.")
            return

        # Load saved prompt if available, otherwise use default
        prompt = self.app.utils_tab.caption_prompt_text.toPlainText().strip()
        if not prompt:
            prompt = default_prompt or (
                "Describe this image in detail for AI image generation training. "
                "Include: the main subject, art style, colors, lighting, composition, "
                "and any notable details. Be specific but concise (under 150 words). "
                "Do not start with 'This image shows' or similar phrases.\n\n"
                "After the description, add a brief quality note on a new line starting with "
                "'[Quality:' - evaluate if this image is suitable for training. Note any issues "
                "like: watermarks, text overlays, low resolution, blur, heavy compression, "
                "awkward cropping, or inconsistent elements. If the image is clean and "
                "suitable, simply write '[Quality: Good]'."
            )
            self.app.utils_tab.caption_prompt_text.setPlainText(prompt)

        # Determine which images to process
        if self.app.utils_tab.caption_selected_radio.isChecked():
            selected_images = self.get_selected_images()
            if not selected_images:
                QMessageBox.information(self.app, "No Selection",
                                       "No images selected.")
                return
            images_to_process = selected_images
        elif self.app.utils_tab.caption_missing_radio.isChecked():
            images_to_process = [
                img for img in self.app.data_manager.images_data
                if not img.get('description', '').strip()
            ]
            if not images_to_process:
                QMessageBox.information(self.app, "All Captioned",
                                       "All images already have captions.")
                return
        else:  # All images
            images_to_process = self.app.data_manager.images_data

        overwrite = self.app.utils_tab.caption_overwrite_check.isChecked()

        # Confirm
        mode_text = "overwrite" if overwrite else "append to"
        reply = QMessageBox.question(
            self.app, "Confirm Caption Generation",
            f"Generate captions for {len(images_to_process)} image(s)?\n\n"
            f"Mode: {mode_text} existing captions\n\n"
            f"This will use the Claude API and may take some time.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply != QMessageBox.StandardButton.Yes:
            return

        # Initialize Claude API
        try:
            import sys
            sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from claude_api import ClaudeAPI, ANTHROPIC_AVAILABLE

            if not ANTHROPIC_AVAILABLE:
                QMessageBox.critical(self.app, "Missing Package",
                                    "The 'anthropic' package is not installed.\n\n"
                                    "Run: pip install anthropic")
                return

            api = ClaudeAPI(api_key)

        except ImportError as e:
            QMessageBox.critical(self.app, "Import Error",
                                f"Could not import Claude API module:\n{str(e)}")
            return

        # Create progress dialog
        progress = QProgressDialog(
            "Generating captions...", "Cancel", 0, len(images_to_process), self.app
        )
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)

        # Update status label
        self.app.utils_tab.caption_status_label.setText("Processing...")

        success_count = 0
        error_count = 0
        errors = []

        for i, img_data in enumerate(images_to_process):
            if progress.wasCanceled():
                break

            progress.setValue(i)
            progress.setLabelText(f"Processing: {img_data['filename']}")
            QApplication.processEvents()

            # Generate caption
            success, result = api.describe_image(img_data['path'], prompt)

            if success:
                # Parse out quality assessment if present
                description_text, quality_rating = self._parse_quality_from_caption(result)

                # Get existing description
                existing = img_data.get('description', '').strip()

                if overwrite or not existing:
                    new_description = description_text
                else:
                    # Append mode
                    new_description = f"{existing} {description_text}"

                # Update data manager
                row_index = img_data.get('row_index', i)
                self.app.data_manager.update_description(row_index, new_description)

                # Update quality rating
                if quality_rating:
                    self.app.data_manager.update_quality(row_index, quality_rating)

                # Update table
                self.app.gallery_tab.table.item(row_index, 1).setText(new_description)

                # Update filename color based on quality
                self._update_row_quality_color(row_index, quality_rating)

                success_count += 1
            else:
                error_count += 1
                errors.append(f"{img_data['filename']}: {result}")

        progress.setValue(len(images_to_process))

        # Update status
        status = f"Completed: {success_count} success, {error_count} errors"
        self.app.utils_tab.caption_status_label.setText(status)

        # Show results
        message = f"Caption generation complete!\n\n"
        message += f"Successful: {success_count}\n"
        message += f"Failed: {error_count}"

        if errors:
            message += f"\n\nErrors:\n" + "\n".join(errors[:5])
            if len(errors) > 5:
                message += f"\n... and {len(errors) - 5} more errors"

        QMessageBox.information(self.app, "Caption Generation Complete", message)

        # Refresh current image display if applicable
        if self.app.current_image_index >= 0:
            self.show_selected_image()

        self.app.set_status(f"Generated {success_count} captions")

    def _parse_quality_from_caption(self, caption: str) -> tuple:
        """
        Parse quality assessment from caption text.

        Returns:
            tuple: (description_without_quality, quality_rating)
            quality_rating is one of: 'good', 'ok', 'bad', or None
        """
        import re

        # Look for [Quality: ...] pattern
        quality_match = re.search(r'\[Quality:\s*([^\]]+)\]', caption, re.IGNORECASE)

        if not quality_match:
            return caption.strip(), None

        quality_text = quality_match.group(1).strip().lower()
        description = caption[:quality_match.start()].strip()

        # Categorize the quality
        if quality_text in ['good', 'excellent', 'great', 'perfect', 'clean']:
            quality_rating = 'good'
        elif any(word in quality_text for word in ['ok', 'okay', 'acceptable', 'minor', 'slight', 'small']):
            quality_rating = 'ok'
        else:
            # Any other quality note indicates issues
            quality_rating = 'bad'

        return description, quality_rating

    def _update_row_quality_color(self, row_index: int, quality: str):
        """Update the filename cell color based on quality rating"""
        if not hasattr(self.app, 'gallery_tab') or not self.app.gallery_tab.table:
            return

        filename_item = self.app.gallery_tab.table.item(row_index, 0)
        if not filename_item:
            return

        if quality == 'good':
            filename_item.setForeground(QColor('#228B22'))  # Forest green
        elif quality == 'ok':
            filename_item.setForeground(QColor('#DAA520'))  # Goldenrod/yellow
        elif quality == 'bad':
            filename_item.setForeground(QColor('#DC143C'))  # Crimson red
        else:
            filename_item.setForeground(QColor('#000000'))  # Default black
