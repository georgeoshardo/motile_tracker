import contextlib

import napari
from napari_orthogonal_views.ortho_view_manager import _VIEWER_MANAGERS
from psygnal import Signal
from qtpy.QtCore import QSignalBlocker
from qtpy.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QPushButton,
    QRadioButton,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)
from superqt import QLabeledDoubleSlider

from motile_tracker.data_views.views.ortho_views import initialize_ortho_views
from motile_tracker.data_views.views_coordinator.tracks_viewer import TracksViewer


class VisualizationConfigWidget(QWidget):
    """Sliders and checkboxes for adjusting the opacity and contour display."""

    update_visualization = Signal()

    def __init__(
        self,
        label: str,
        default_opacity: float,
        default_contour: bool,
        use_contour: bool = True,
    ):
        super().__init__()

        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        self.orth_views_connection = None
        self.orth_view_manager = None

        box = QGroupBox(label)
        box.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)

        box_layout = QVBoxLayout(box)
        box_layout.setContentsMargins(8, 6, 8, 6)
        box_layout.setSpacing(6)

        self.opacity = QLabeledDoubleSlider()
        self.opacity.setValue(default_opacity)
        self.opacity.setSingleStep(0.1)
        self.opacity.setRange(0, 1)
        self.opacity.setDecimals(2)
        self.opacity.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.opacity.valueChanged.connect(self.update_visualization)
        box_layout.addWidget(self.opacity)

        self.contour = None
        if use_contour:
            self.contour = QCheckBox("Fill")
            self.contour.setChecked(default_contour)
            self.contour.stateChanged.connect(self.update_visualization)
            self.contour.setEnabled(False)
            self.contour.setVisible(False)
            self.contour.setToolTip(
                "When checked, will fill labels instead of showing contours only"
            )
            self.contour.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
            box_layout.addWidget(self.contour)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)
        layout.addWidget(box)
        layout.addStretch(0)


class ModeWidget(QWidget):
    """Compact radio buttons for choosing one of several viewer modes."""

    update_mode = Signal(str)

    def __init__(self, title: str, options: list[tuple[str, str]]):
        """
        Args:
            title (str): the group box title.
            options (list[tuple[str, str]]): (button text, mode value) pairs. The first
                option is checked initially.
        """
        super().__init__()

        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)

        box = QGroupBox(title)
        box.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)

        self.radio_group = QButtonGroup(self)
        box_layout = QHBoxLayout(box)
        box_layout.setContentsMargins(12, 8, 12, 8)
        box_layout.setSpacing(14)

        for index, (text, mode) in enumerate(options):
            btn = QRadioButton(text)
            btn.setProperty("mode", mode)
            btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

            btn.setStyleSheet("""
                QRadioButton {
                    padding: 3px 8px;
                }
            """)

            self.radio_group.addButton(btn)
            box_layout.addWidget(btn)

            if index == 0:
                btn.setChecked(True)

        self.radio_group.buttonToggled.connect(self._on_toggled)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        layout.addWidget(box)

    @property
    def current_mode(self) -> str:
        btn = self.radio_group.checkedButton()
        return btn.property("mode") if btn else None

    def button_for_mode(self, mode: str) -> QRadioButton:
        for btn in self.radio_group.buttons():
            if btn.property("mode") == mode:
                return btn
        raise KeyError(f"No radio button for mode '{mode}'")

    def _on_toggled(self, button, checked):
        if checked:
            self.update_mode.emit(button.property("mode"))


class VisualizationWidget(QWidget):
    """Widget to switch between the spatial and kymograph views, adjust the kymograph
    paging, and adjust opacity and contour display in the different TrackLabels layer
    display modes."""

    def __init__(self, viewer: napari.Viewer):
        super().__init__()

        self.viewer = viewer
        self.tracks_viewer = TracksViewer.get_instance(viewer)

        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)

        # Initialize ortho-views attributes
        self.orth_views_connection = None
        self.orth_view_manager = None

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        self.tracks_viewer.mode_updated.connect(self._update_widget_availability)
        self.tracks_viewer.view_mode_updated.connect(self._update_widget_availability)
        self.tracks_viewer.kymograph_updated.connect(self._update_widget_availability)
        self.tracks_viewer.tracks_updated.connect(self._update_widget_availability)

        self.view_mode_widget = ModeWidget(
            "View Mode",
            [("Spatial", "spatial"), ("Kymograph", "kymograph")],
        )
        self.view_mode_widget.update_mode.connect(self._update_view_mode)

        self.mode_widget = ModeWidget(
            "Display Mode",
            [("All", "all"), ("Lineage", "lineage"), ("Group", "group")],
        )
        self.mode_widget.update_mode.connect(self._update_mode)

        self.kymograph_box = self._make_kymograph_box()

        self.highlight_widget = VisualizationConfigWidget(
            "Highlight opacity", 1.0, True
        )
        self.foreground_widget = VisualizationConfigWidget(
            "Foreground opacity", 0.6, True
        )
        self.background_widget = VisualizationConfigWidget(
            "Background opacity", 0.3, True, use_contour=False
        )

        self.highlight_widget.update_visualization.connect(self._update_visualization)
        self.foreground_widget.update_visualization.connect(self._update_visualization)
        self.background_widget.update_visualization.connect(self._update_visualization)

        self.background_widget.setEnabled(False)  # initially disabled

        main_layout.addWidget(self.view_mode_widget)
        main_layout.addWidget(self.mode_widget)
        main_layout.addWidget(self.kymograph_box)
        main_layout.addWidget(self.highlight_widget)
        main_layout.addWidget(self.foreground_widget)
        main_layout.addWidget(self.background_widget)

        self.show_ortho_views = QCheckBox("Orthogonal views")
        self.show_ortho_views.stateChanged.connect(self.initialize_ortho_views)
        self.show_ortho_views.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        main_layout.addWidget(self.show_ortho_views)
        main_layout.addStretch(1)

        # No maximum height: the kymograph controls need the room, and the menu
        # manager wraps this widget in a scroll area.

        self._update_widget_availability()

    def _make_kymograph_box(self) -> QGroupBox:
        """Create the group box with the kymograph paging and link controls."""

        self.image_layer_box = QComboBox()
        self.image_layer_box.setToolTip(
            "Image layer to show behind the kymograph. Must have the same shape (and "
            "scale) as the tracked segmentation."
        )
        self.image_layer_box.currentTextChanged.connect(self._update_image_layer)

        self.page_start_box = QSpinBox()
        self.page_start_box.setMinimum(0)
        self.page_start_box.setToolTip("First frame shown on the current page")
        self.page_start_box.valueChanged.connect(self._update_page_start)

        self.page_length_box = QSpinBox()
        self.page_length_box.setMinimum(1)
        self.page_length_box.setToolTip("Number of frames shown side by side")
        self.page_length_box.valueChanged.connect(self._update_page_length)

        self.prev_page_button = QPushButton("Previous")
        self.prev_page_button.clicked.connect(
            lambda: self.tracks_viewer.step_kymograph_page(-1)
        )
        self.next_page_button = QPushButton("Next")
        self.next_page_button.clicked.connect(
            lambda: self.tracks_viewer.step_kymograph_page(1)
        )

        self.paths_checkbox = QCheckBox("Show paths")
        self.paths_checkbox.setChecked(True)
        self.paths_checkbox.setToolTip(
            "Draw lines between consecutive nodes of a track"
        )
        self.paths_checkbox.stateChanged.connect(self._update_show_paths)

        self.branches_checkbox = QCheckBox("Show branches")
        self.branches_checkbox.setChecked(True)
        self.branches_checkbox.setToolTip("Draw dashed lines for division edges")
        self.branches_checkbox.stateChanged.connect(self._update_show_branches)

        self.boundaries_checkbox = QCheckBox("Show frame boundaries")
        self.boundaries_checkbox.setChecked(True)
        self.boundaries_checkbox.stateChanged.connect(self._update_show_boundaries)

        box = QGroupBox("Kymograph")
        box.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        form = QFormLayout(box)
        form.addRow("Image Layer", self.image_layer_box)
        form.addRow("Page Start", self.page_start_box)
        form.addRow("Page Length", self.page_length_box)

        nav_row = QHBoxLayout()
        nav_row.addWidget(self.prev_page_button)
        nav_row.addWidget(self.next_page_button)
        form.addRow("Page", nav_row)
        form.addRow(self.paths_checkbox)
        form.addRow(self.branches_checkbox)
        form.addRow(self.boundaries_checkbox)
        return box

    def initialize_ortho_views(self, checked: bool):
        """Initializes the ortho views."""

        if self.show_ortho_views.isChecked() != checked:
            # sync checkbox state, since there are two ways to trigger this function (checkbox or menu action in ortho view widget)
            self.show_ortho_views.setChecked(checked)
        if self.viewer in _VIEWER_MANAGERS:
            self.orth_view_manager = _VIEWER_MANAGERS[self.viewer]
            if not checked:
                self.orth_view_manager.hide()
                self.orth_view_manager.set_splitter_sizes(
                    0.0, 0.0
                )  # minimal size for right and bottom
            else:
                self.orth_view_manager.show()
        else:
            self.orth_view_manager = initialize_ortho_views(
                self.viewer
            )  # store to be able to disconnect later
            self.orth_views_connection = (
                self.orth_view_manager.main_controls_widget.show_orth_views.connect(
                    self.initialize_ortho_views
                )
            )
            # remove connection and reset checkbox when destroyed
            self.orth_view_manager.main_controls_widget.destroyed.connect(
                self._on_ortho_cleanup
            )
            self.orth_view_manager.show()

    def _on_ortho_cleanup(self):
        """Called when ortho_view_manager cleans up and deletes its widgets."""
        self._disconnect_ortho_views()
        # Uncheck the checkbox without triggering initialize_ortho_views
        self.show_ortho_views.blockSignals(True)
        self.show_ortho_views.setChecked(False)
        self.show_ortho_views.blockSignals(False)

    def _disconnect_ortho_views(self):
        """Safely disconnect from ortho views signals."""

        if (
            self.orth_views_connection is not None
            and self.orth_view_manager is not None
        ):
            with contextlib.suppress(TypeError, RuntimeError):
                self.orth_view_manager.main_controls_widget.show_orth_views.disconnect(
                    self.orth_views_connection
                )
            self.orth_views_connection = None
        self.orth_view_manager = None

    def _update_view_mode(self, mode: str) -> None:
        """Switch between the spatial and kymograph view on the TracksViewer"""

        self.tracks_viewer.set_view_mode(mode)
        self._update_widget_availability()

    def _update_mode(self, mode: str) -> None:
        """Update the display mode on the Tracksviewer"""

        self.tracks_viewer.set_display_mode(mode)
        self._update_widget_availability()

    def _update_image_layer(self, layer_name: str) -> None:
        self.tracks_viewer.set_kymograph_image_layer(
            None if layer_name == "None" else layer_name
        )

    def _update_page_start(self, page_start: int) -> None:
        self.tracks_viewer.set_kymograph_page_start(page_start)

    def _update_page_length(self, page_length: int) -> None:
        self.tracks_viewer.set_kymograph_page_length(page_length)

    def _update_show_boundaries(self) -> None:
        self.tracks_viewer.set_show_kymograph_boundaries(
            self.boundaries_checkbox.isChecked()
        )

    def _update_show_paths(self) -> None:
        self.tracks_viewer.set_show_kymograph_paths(self.paths_checkbox.isChecked())

    def _update_show_branches(self) -> None:
        self.tracks_viewer.set_show_kymograph_branches(
            self.branches_checkbox.isChecked()
        )

    def _update_widget_availability(self):
        """Sync the widget with the TracksViewer: check the right radio buttons,
        refresh the kymograph controls, and show/hide the opacity widgets (they only
        apply when a segmentation layer is present). Also show/hide the contour
        checkboxes when changing between contour and normal mode, and disable the
        background widget when the display mode is 'All', as there are no background
        labels in that case."""

        # ensure the correct radio buttons are checked
        mode = self.tracks_viewer.mode
        with QSignalBlocker(self.mode_widget.radio_group):
            self.mode_widget.button_for_mode(mode).setChecked(True)
        with QSignalBlocker(self.view_mode_widget.radio_group):
            self.view_mode_widget.button_for_mode(
                self.tracks_viewer.view_mode
            ).setChecked(True)

        self._update_kymograph_controls()

        seg_layer = self.tracks_viewer.tracking_layers.seg_layer
        if seg_layer is not None:
            self.highlight_widget.setVisible(True)
            self.foreground_widget.setVisible(True)
            self.background_widget.setVisible(True)

            self.background_widget.setEnabled(mode != "all")

            show_contour = seg_layer.contour > 0 and mode != "all"

            for w in (self.highlight_widget, self.foreground_widget):
                w.contour.setVisible(show_contour)
                w.contour.setEnabled(show_contour)

            self._update_visualization()

        else:
            self.highlight_widget.setVisible(False)
            self.foreground_widget.setVisible(False)
            self.background_widget.setVisible(False)

    def _update_kymograph_controls(self):
        """Refresh the kymograph controls from the KymographLayerGroup state."""

        kymograph_layers = self.tracks_viewer.kymograph_layers

        image_layers = ["None", *kymograph_layers.available_image_layers()]
        with QSignalBlocker(self.image_layer_box):
            self.image_layer_box.clear()
            self.image_layer_box.addItems(image_layers)
            current_layer = kymograph_layers.image_layer_name or "None"
            self.image_layer_box.setCurrentText(
                current_layer if current_layer in image_layers else "None"
            )

        geometry = kymograph_layers.geometry
        page_length = kymograph_layers.page_length
        if geometry is not None:
            self.page_start_box.setMaximum(max(0, geometry.t_size - page_length))
            self.page_length_box.setMaximum(max(1, geometry.t_size))
        with QSignalBlocker(self.page_start_box):
            self.page_start_box.setValue(kymograph_layers.page_start)
        with QSignalBlocker(self.page_length_box):
            self.page_length_box.setValue(page_length)
        for checkbox, checked in (
            (self.boundaries_checkbox, kymograph_layers.show_boundaries),
            (self.paths_checkbox, kymograph_layers.show_paths),
            (self.branches_checkbox, kymograph_layers.show_branches),
        ):
            with QSignalBlocker(checkbox):
                checkbox.setChecked(checked)

        tracks = self.tracks_viewer.tracks
        can_show_kymograph = tracks is not None and tracks.ndim == 3
        self.view_mode_widget.button_for_mode("kymograph").setEnabled(
            can_show_kymograph
        )
        self.kymograph_box.setEnabled(can_show_kymograph)
        # the image layer can only be changed while the spatial view is shown
        self.image_layer_box.setEnabled(
            can_show_kymograph and self.tracks_viewer.view_mode != "kymograph"
        )

    def _update_visualization(self):
        """Apply the values from the widget and send an update signal."""

        layers = [
            self.tracks_viewer.tracking_layers.seg_layer,
            self.tracks_viewer.kymograph_layers.labels_layer,
        ]
        updated = False
        for layer in layers:
            if layer is None:
                continue
            layer.highlight_opacity = self.highlight_widget.opacity.value()
            layer.foreground_opacity = self.foreground_widget.opacity.value()
            layer.background_opacity = self.background_widget.opacity.value()
            layer.highlight_contour = not self.highlight_widget.contour.isChecked()
            layer.foreground_contour = not self.foreground_widget.contour.isChecked()
            updated = True

        if updated:
            self.tracks_viewer.update_selection(set_view=False)
