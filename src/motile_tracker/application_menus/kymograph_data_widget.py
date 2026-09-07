from __future__ import annotations

from pathlib import Path

import napari
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from motile_tracker.data_views.views_coordinator.tracks_viewer import TracksViewer
from motile_tracker.import_export.geff_collection import (
    GeffGroup,
    LoadedGeff,
    add_geff_group_to_viewer,
    find_geff_groups,
)


class KymographDataWidget(QWidget):
    """Browse a zarr store for GEFF groups (for example one per mother machine
    trench, each with its image and segmentation next to the graph) and load one of
    them, image included, straight into the kymograph view.

    Loaded tracks go to the Tracks List, so they can be edited, saved and exported
    like tracks loaded any other way.
    """

    def __init__(self, viewer: napari.Viewer):
        super().__init__()
        self.viewer = viewer
        self.tracks_viewer = TracksViewer.get_instance(viewer)
        self.groups: list[GeffGroup] = []
        self._loaded_image_layers: list[napari.layers.Image] = []

        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)

        # --- store ---
        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("Path to a zarr store (or a single GEFF)")
        self.path_edit.editingFinished.connect(self.scan)
        browse_button = QPushButton("Browse…")
        browse_button.clicked.connect(self._browse)
        path_row = QHBoxLayout()
        path_row.addWidget(self.path_edit)
        path_row.addWidget(browse_button)

        store_box = QGroupBox("Data store")
        store_layout = QVBoxLayout(store_box)
        store_layout.addLayout(path_row)

        # --- group selection ---
        self.group_box = QComboBox()
        self.group_box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.group_box.currentIndexChanged.connect(self._update_buttons)
        self.previous_button = QPushButton("◀")
        self.previous_button.setToolTip("Load the previous group")
        self.previous_button.setFixedWidth(32)
        self.previous_button.clicked.connect(self.load_previous)
        self.next_button = QPushButton("▶")
        self.next_button.setToolTip("Load the next group")
        self.next_button.setFixedWidth(32)
        self.next_button.clicked.connect(self.load_next)
        group_row = QHBoxLayout()
        group_row.addWidget(self.previous_button)
        group_row.addWidget(self.group_box)
        group_row.addWidget(self.next_button)

        self.kymograph_checkbox = QCheckBox("Open in kymograph view")
        self.kymograph_checkbox.setChecked(True)
        self.replace_checkbox = QCheckBox("Remove previously loaded images")
        self.replace_checkbox.setChecked(True)
        self.replace_checkbox.setToolTip(
            "Remove the image layers added by earlier loads. Tracks stay in the "
            "Tracks List, so no edits are lost."
        )
        self.load_button = QPushButton("Load")
        self.load_button.clicked.connect(self.load_selected)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)

        groups_box = QGroupBox("GEFF groups")
        groups_layout = QVBoxLayout(groups_box)
        groups_layout.addLayout(group_row)
        groups_layout.addWidget(self.kymograph_checkbox)
        groups_layout.addWidget(self.replace_checkbox)
        groups_layout.addWidget(self.load_button)
        groups_layout.addWidget(self.status_label)

        layout = QVBoxLayout(self)
        layout.addWidget(store_box)
        layout.addWidget(groups_box)
        layout.addStretch(1)

        self._update_buttons()

    # ------------------------------------------------------------------
    # store
    # ------------------------------------------------------------------
    def _browse(self) -> None:
        start = self.path_edit.text().strip() or str(Path.home())
        folder = QFileDialog.getExistingDirectory(
            self, "Select a zarr store with GEFF groups", start
        )
        if folder:
            self.set_store_path(Path(folder))

    def set_store_path(self, path: Path) -> list[GeffGroup]:
        """Set the store path and list the geff groups in it."""
        self.path_edit.setText(str(path))
        return self.scan()

    def scan(self) -> list[GeffGroup]:
        """List the geff groups in the store named in the path field."""
        self.group_box.clear()
        self.groups = []
        text = self.path_edit.text().strip()
        if not text:
            self._set_status("")
            self._update_buttons()
            return []

        path = Path(text).expanduser()
        if not path.is_dir():
            self._set_status(f"Not a directory: {path}", error=True)
            self._update_buttons()
            return []

        self.groups = find_geff_groups(path)
        self.group_box.addItems([group.name for group in self.groups])
        count = len(self.groups)
        if count == 0:
            self._set_status("No GEFF groups found in this store", error=True)
        else:
            self._set_status(f"Found {count} GEFF group{'s' if count != 1 else ''}")
        self._update_buttons()
        return self.groups

    # ------------------------------------------------------------------
    # loading
    # ------------------------------------------------------------------
    def current_group(self) -> GeffGroup | None:
        index = self.group_box.currentIndex()
        if 0 <= index < len(self.groups):
            return self.groups[index]
        return None

    def select_group(self, name: str) -> bool:
        """Select the group with the given name. Returns False if there is none."""
        for index, group in enumerate(self.groups):
            if group.name == name:
                self.group_box.setCurrentIndex(index)
                return True
        return False

    def load_selected(self) -> LoadedGeff | None:
        """Load the selected group into the viewer."""
        group = self.current_group()
        if group is None:
            return None

        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            # switch to the spatial view first so that the image layers added by
            # earlier loads are back in the viewer and can be removed
            if self.tracks_viewer.view_mode == "kymograph":
                self.tracks_viewer.set_view_mode("spatial")
            if self.replace_checkbox.isChecked():
                self._remove_loaded_images()
            loaded = add_geff_group_to_viewer(
                self.viewer,
                group.path,
                group.name,
                kymograph=self.kymograph_checkbox.isChecked(),
            )
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "Error", f"Failed to load {group.name}: {e}")
            self._set_status(f"Failed to load {group.name}: {e}", error=True)
            return None
        finally:
            QApplication.restoreOverrideCursor()

        if loaded.image_layer is not None:
            self._loaded_image_layers.append(loaded.image_layer)

        graph = loaded.tracks.graph
        summary = f"{graph.num_nodes()} nodes, {graph.num_edges()} edges"
        if loaded.image is None:
            summary += ", no image"
        self._set_status(f"Loaded {group.name}: {summary}")
        return loaded

    def load_next(self) -> LoadedGeff | None:
        """Select and load the group after the current one."""
        index = self.group_box.currentIndex()
        if index + 1 >= len(self.groups):
            return None
        self.group_box.setCurrentIndex(index + 1)
        return self.load_selected()

    def load_previous(self) -> LoadedGeff | None:
        """Select and load the group before the current one."""
        index = self.group_box.currentIndex()
        if index <= 0:
            return None
        self.group_box.setCurrentIndex(index - 1)
        return self.load_selected()

    def _remove_loaded_images(self) -> None:
        for layer in self._loaded_image_layers:
            if layer in self.viewer.layers:
                self.viewer.layers.remove(layer)
        self._loaded_image_layers = []

    # ------------------------------------------------------------------
    # ui state
    # ------------------------------------------------------------------
    def _update_buttons(self) -> None:
        index = self.group_box.currentIndex()
        count = len(self.groups)
        self.load_button.setEnabled(count > 0)
        self.previous_button.setEnabled(index > 0)
        self.next_button.setEnabled(0 <= index < count - 1)

    def _set_status(self, text: str, error: bool = False) -> None:
        self.status_label.setText(text)
        self.status_label.setStyleSheet("color: orange;" if error else "")
