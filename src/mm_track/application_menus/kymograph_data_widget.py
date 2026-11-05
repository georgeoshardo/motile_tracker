from __future__ import annotations

from pathlib import Path

import napari
from qtpy.QtCore import Qt, QTimer
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

from mm_track import experiment_layout
from mm_track.data_views.views_coordinator.tracks_viewer import TracksViewer
from mm_track.import_export.geff_collection import (
    GeffGroup,
    LoadedGeff,
    add_geff_group_to_viewer,
    find_geff_groups,
)


class KymographDataWidget(QWidget):
    """Browse a zarr store for GEFF groups (for example one per mother machine
    trench, each with its image and segmentation next to the graph) and load one of
    them, image included, straight into the kymograph view. A group loads as soon
    as it is picked from the list, or stepped to with the arrows.

    Loaded tracks go to the Tracks List, so they can be edited, saved and exported
    like tracks loaded any other way.

    For a PyMMM experiment group (see :mod:`mm_track.experiment_layout`) the list
    has one entry per trench for the tracking run, marked ✓ when the trench is
    reviewed and ✎ when it is edited but not reviewed. "Reviewed" marks the loaded
    trench finished; only reviewed trenches count as curated.
    """

    MARKS = {"reviewed": "✓", "edited": "✎"}

    def __init__(self, viewer: napari.Viewer):
        super().__init__()
        self.viewer = viewer
        self.tracks_viewer = TracksViewer.get_instance(viewer)
        self.groups: list[GeffGroup] = []
        self.run: str | None = None  # an experiment group's tracking run
        self.experiment = False  # whether the store is a PyMMM experiment group
        self._loaded_image_layers: list[napari.layers.Image] = []
        self._loaded_name: str | None = None
        self._loaded: LoadedGeff | None = None

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
        # activated is only emitted when the user picks a group, not when the list
        # is filled or a group is selected from code
        self.group_box.activated.connect(self._on_group_picked)
        self.previous_button = QPushButton("◀")
        self.previous_button.setToolTip("Load the previous group")
        self.previous_button.setFixedWidth(32)
        self.previous_button.clicked.connect(self.load_previous)
        self.next_button = QPushButton("▶")
        self.next_button.setToolTip("Load the next group")
        self.next_button.setFixedWidth(32)
        self.next_button.clicked.connect(self.load_next)
        # next to the list, so it is in view whenever the trench is
        self.reviewed_checkbox = QCheckBox("Reviewed")
        self.reviewed_checkbox.setToolTip(
            "Mark the loaded tracks finished: every cell checked and corrected. "
            "Only reviewed trenches count as curated; an edit alone does not."
        )
        self.reviewed_checkbox.clicked.connect(self._on_reviewed_clicked)
        group_row = QHBoxLayout()
        group_row.addWidget(self.previous_button)
        group_row.addWidget(self.group_box)
        group_row.addWidget(self.next_button)
        group_row.addWidget(self.reviewed_checkbox)

        self.kymograph_checkbox = QCheckBox("Open in kymograph view")
        self.kymograph_checkbox.setChecked(True)
        self.replace_checkbox = QCheckBox("Remove previously loaded images")
        self.replace_checkbox.setChecked(True)
        self.replace_checkbox.setToolTip(
            "Remove the image layers added by earlier loads. Tracks stay in the "
            "Tracks List, so no edits are lost."
        )

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)

        # --- review ---
        self.review_label = QLabel("")
        self.review_label.setWordWrap(True)
        # the edits made after a review, and a first edit moving the tracks to their
        # curated copy, show up here without a signal of their own
        self._review_timer = QTimer(self)
        self._review_timer.setInterval(1000)
        self._review_timer.timeout.connect(self._update_review)
        self._review_timer.start()

        groups_box = QGroupBox("GEFF groups")
        groups_layout = QVBoxLayout(groups_box)
        groups_layout.addLayout(group_row)
        groups_layout.addWidget(self.kymograph_checkbox)
        groups_layout.addWidget(self.replace_checkbox)
        groups_layout.addWidget(self.review_label)
        groups_layout.addWidget(self.status_label)

        layout = QVBoxLayout(self)
        layout.addWidget(store_box)
        layout.addWidget(groups_box)
        layout.addStretch(1)

        self._update_buttons()
        self._update_review()

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

    def set_store_path(self, path: Path, run: str | None = None) -> list[GeffGroup]:
        """Set the store path and list the geff groups in it (for an experiment
        group: the trenches with tracks of `run`, by default its first run)."""
        self.run = run
        self.path_edit.setText(str(path))
        return self.scan()

    def scan(self) -> list[GeffGroup]:
        """List the geff groups in the store named in the path field."""
        self.group_box.clear()
        self.groups = []
        self.experiment = False
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

        experiment = self.experiment = experiment_layout.is_experiment(path)
        if experiment and self.run is None:
            self.run = next(iter(experiment_layout.runs(path)), None)
        self.groups = find_geff_groups(path, run=self.run if experiment else None)
        self.group_box.addItems([group.name for group in self.groups])
        if experiment:
            for index in range(len(self.groups)):
                self._mark_group(index)
        count = len(self.groups)
        if count == 0:
            self._set_status("No GEFF groups found in this store", error=True)
        elif experiment:
            reviewed = sum(
                experiment_layout.trench_state(g.path) == "reviewed" for g in self.groups
            )
            self._set_status(
                f"{count} trenches with tracks of {self.run}; {reviewed} reviewed"
            )
        else:
            self._set_status(f"Found {count} GEFF group{'s' if count != 1 else ''}")
        self._update_buttons()
        return self.groups

    def _mark_group(self, index: int, state: str | None = None) -> None:
        """Show a group's state (reviewed, edited; read from disk unless given)
        after its name in the list."""
        group = self.groups[index]
        if state is None:
            state = experiment_layout.trench_state(group.path)
        mark = self.MARKS.get(state)
        text = f"{group.name} {mark}" if mark else group.name
        if self.group_box.itemText(index) != text:
            self.group_box.setItemText(index, text)

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
            # One batch for the canvas: napari rebuilds its scene graph, after a full
            # garbage collection, on every layer removed outside a batch, and a switch
            # removes and adds about twenty layers (0.15 s each on a large heap).
            with self.viewer.layers.batched_update():
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

        self._loaded_image_layers.extend(loaded.image_layers)

        graph = loaded.tracks.graph
        summary = f"{graph.num_nodes()} nodes, {graph.num_edges()} edges"
        if loaded.image is None:
            summary += ", no image"
        self._loaded_name = group.name
        self._loaded = loaded
        self._set_status(f"Loaded {group.name}: {summary}")
        self._update_review()
        return loaded

    def _on_group_picked(self, _index: int) -> None:
        """Load the group picked from the list, unless it is the one already
        loaded."""
        group = self.current_group()
        if group is not None and group.name != self._loaded_name:
            self.load_selected()

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
    # review
    # ------------------------------------------------------------------
    def _loaded_session(self):
        if self._loaded is None:
            return None
        session = getattr(self._loaded.tracks, "edit_session", None)
        return None if session is None or session.closed else session

    def _on_reviewed_clicked(self, checked: bool) -> None:
        session = self._loaded_session()
        if session is None:
            self._update_review()
            return
        if not checked and self._stale(session):
            checked = True  # a click on a mark edits have outdated reviews them too
        elif not checked and (
            QMessageBox.question(
                self,
                "Remove the review mark?",
                f"{self._loaded_name} will no longer count as curated.",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            != QMessageBox.Yes
        ):
            self._update_review()
            return
        try:
            session.set_reviewed(checked)
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(
                self, "Error", f"Could not mark {self._loaded_name} reviewed: {e}"
            )
        self._update_review()

    @staticmethod
    def _stale(session) -> bool:
        """Whether the tracks were edited after they were marked reviewed."""
        review = session.review
        return review is not None and session.revision > int(review.get("revision", 0))

    def _update_review(self) -> None:
        """Show the loaded tracks' review mark, and where their edits are saved."""
        session = self._loaded_session()
        if session is not None and self.experiment:
            state = (
                "reviewed" if session.review
                else "edited" if session.path.parent.name == experiment_layout.CURATION
                else None
            )
            for index, group in enumerate(self.groups):
                if group.name == self._loaded_name:
                    self._mark_group(index, state or "")
        self.reviewed_checkbox.setEnabled(session is not None)
        review = session.review if session is not None else None
        if self.reviewed_checkbox.isChecked() != bool(review):
            self.reviewed_checkbox.blockSignals(True)
            self.reviewed_checkbox.setChecked(bool(review))
            self.reviewed_checkbox.blockSignals(False)
        if session is None:
            text = ""
        elif review:
            text = (
                f"Reviewed by {review.get('by', '?')} on "
                f"{str(review.get('at', '?')).replace('T', ' ')[:16]} "
                f"(revision {review.get('revision', '?')})"
            )
            if self._stale(session):
                text += (
                    f"; edited since (revision {session.revision}). Click Reviewed "
                    "to mark this revision reviewed."
                )
        else:
            target = experiment_layout.curation_target(session.path)
            if target is not None:
                text = (
                    "Not reviewed. The tracker's output: the first edit makes the "
                    f"curated copy, curation/{target.run}.geff."
                )
            elif session.path.parent.name == experiment_layout.CURATION:
                text = f"Not reviewed. Edits are saved to curation/{session.path.name}."
            else:
                text = "Not reviewed."
        if self.review_label.text() != text:
            self.review_label.setText(text)
            self.review_label.setToolTip(str(session.path) if session else "")
        if session is not None and self._stale(session):
            self.reviewed_checkbox.setStyleSheet("color: orange;")
        else:
            self.reviewed_checkbox.setStyleSheet("")

    # ------------------------------------------------------------------
    # ui state
    # ------------------------------------------------------------------
    def _update_buttons(self) -> None:
        index = self.group_box.currentIndex()
        count = len(self.groups)
        self.previous_button.setEnabled(index > 0)
        self.next_button.setEnabled(0 <= index < count - 1)

    def _set_status(self, text: str, error: bool = False) -> None:
        self.status_label.setText(text)
        self.status_label.setStyleSheet("color: orange;" if error else "")
