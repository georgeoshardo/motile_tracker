import contextlib

import napari
from psygnal import Signal
from qtpy.QtCore import QSignalBlocker, Qt
from qtpy.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QListWidget,
    QListWidgetItem,
    QRadioButton,
    QSizePolicy,
    QWidget,
)

from mm_track.application_menus.visualization_widget import compact_box
from mm_track.data_views.views_coordinator.tracks_viewer import TracksViewer

# napari-orthogonal-views needs OpenCV, so it is only imported once the views
# are opened


def initialize_ortho_views(viewer):
    from mm_track.data_views.views.ortho_views import (
        initialize_ortho_views as initialize,
    )

    return initialize(viewer)


def _viewer_managers():
    from napari_orthogonal_views.ortho_view_manager import _VIEWER_MANAGERS

    return _VIEWER_MANAGERS


class ModeWidget(QWidget):
    """A row of radio buttons for choosing one of several viewer modes."""

    update_mode = Signal(str)

    def __init__(self, title: str, options: list[tuple[str, str]]):
        """
        Args:
            title (str): what is being chosen, shown beside the buttons.
            options (list[tuple[str, str]]): (button text, mode value) pairs. The first
                option is checked initially.
        """
        super().__init__()

        self.title = title
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)

        self.radio_group = QButtonGroup(self)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        for index, (text, mode) in enumerate(options):
            btn = QRadioButton(text)
            btn.setProperty("mode", mode)
            btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
            self.radio_group.addButton(btn)
            layout.addWidget(btn)

            if index == 0:
                btn.setChecked(True)
        layout.addStretch(1)

        self.radio_group.buttonToggled.connect(self._on_toggled)

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


class ViewModeWidget(QWidget):
    """Choose what the canvas shows: the spatial or the kymograph view, which tracks
    are displayed (all, a lineage or a group), the orthogonal views and the help text,
    and the image layers the kymograph is drawn from."""

    def __init__(self, viewer: napari.Viewer):
        super().__init__()

        self.viewer = viewer
        self.tracks_viewer = TracksViewer.get_instance(viewer)

        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)

        self.orth_views_connection = None
        self.orth_view_manager = None

        self.tracks_viewer.mode_updated.connect(self._update_controls)
        self.tracks_viewer.view_mode_updated.connect(self._update_controls)
        self.tracks_viewer.kymograph_updated.connect(self._update_controls)
        self.tracks_viewer.tracks_updated.connect(self._update_controls)

        self.view_mode_widget = ModeWidget(
            "View",
            [("Spatial", "spatial"), ("Kymograph", "kymograph")],
        )
        self.view_mode_widget.update_mode.connect(self._update_view_mode)

        self.mode_widget = ModeWidget(
            "Display",
            [("All", "all"), ("Lineage", "lineage"), ("Group", "group")],
        )
        self.mode_widget.update_mode.connect(self._update_mode)

        self.show_ortho_views = QCheckBox("Orthogonal views")
        self.show_ortho_views.stateChanged.connect(self.initialize_ortho_views)
        self.show_ortho_views.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        self.show_help_text = QCheckBox("Show help text")
        self.show_help_text.setToolTip(
            "Show the click and key help in the bottom-right corner of the canvas"
        )
        self.show_help_text.setChecked(self.tracks_viewer.show_overlay_text)
        self.show_help_text.stateChanged.connect(
            lambda: self.tracks_viewer.set_show_overlay_text(
                self.show_help_text.isChecked()
            )
        )
        self.show_help_text.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        self.image_layer_box = QComboBox()
        self.image_layer_box.setToolTip(
            "Image layer to show behind the kymograph. Must have the same shape (and "
            "scale) as the tracked segmentation."
        )
        self.image_layer_box.setMinimumWidth(170)
        self.image_layer_box.currentTextChanged.connect(self._update_image_layer)

        self.overlay_list = QListWidget()
        self.overlay_list.setToolTip(
            "Image layers drawn as kymographs over the image layer, each in its own "
            "colors (for example the other channels)."
        )
        self.overlay_list.setMaximumHeight(64)
        self.overlay_list.setMinimumWidth(170)
        self.overlay_list.itemChanged.connect(self._update_overlay_layers)

        view_box, view_form = compact_box("View")
        view_form.addRow(self.view_mode_widget.title, self.view_mode_widget)
        view_form.addRow(self.mode_widget.title, self.mode_widget)
        view_form.addRow(self.show_ortho_views)
        view_form.addRow(self.show_help_text)

        self.kymograph_box, kymograph_form = compact_box("Kymograph")
        kymograph_form.addRow("Image", self.image_layer_box)
        kymograph_form.addRow("Overlays", self.overlay_list)

        # side by side, like the Visualization menu, for the strip along the bottom
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(6)
        layout.addWidget(view_box, 0, Qt.AlignTop)
        layout.addWidget(self.kymograph_box, 0, Qt.AlignTop)
        layout.addStretch(1)

        self._update_controls()

    def initialize_ortho_views(self, checked: bool):
        """Initializes the ortho views."""

        if self.show_ortho_views.isChecked() != checked:
            # sync checkbox state, since there are two ways to trigger this function (checkbox or menu action in ortho view widget)
            self.show_ortho_views.setChecked(checked)
        managers = _viewer_managers()
        if self.viewer in managers:
            self.orth_view_manager = managers[self.viewer]
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
        self._update_controls()

    def _update_mode(self, mode: str) -> None:
        """Update the display mode on the Tracksviewer"""

        self.tracks_viewer.set_display_mode(mode)
        self._update_controls()

    def _update_image_layer(self, layer_name: str) -> None:
        self.tracks_viewer.set_kymograph_image_layer(
            None if layer_name == "None" else layer_name
        )

    def _update_overlay_layers(self, _item=None) -> None:
        names = [
            self.overlay_list.item(row).text()
            for row in range(self.overlay_list.count())
            if self.overlay_list.item(row).checkState() == Qt.Checked
        ]
        self.tracks_viewer.set_kymograph_overlay_layers(names)

    def _update_controls(self) -> None:
        """Sync the widget with the TracksViewer: check the right radio buttons, list
        the image layers, and only offer the kymograph view for 2D+time tracks."""

        with QSignalBlocker(self.mode_widget.radio_group):
            self.mode_widget.button_for_mode(self.tracks_viewer.mode).setChecked(True)
        with QSignalBlocker(self.view_mode_widget.radio_group):
            self.view_mode_widget.button_for_mode(
                self.tracks_viewer.view_mode
            ).setChecked(True)

        kymograph_layers = self.tracks_viewer.kymograph_layers
        image_layers = ["None", *kymograph_layers.available_image_layers()]
        with QSignalBlocker(self.image_layer_box):
            self.image_layer_box.clear()
            self.image_layer_box.addItems(image_layers)
            current_layer = kymograph_layers.image_layer_name or "None"
            self.image_layer_box.setCurrentText(
                current_layer if current_layer in image_layers else "None"
            )

        # In kymograph mode the image layers are detached from the viewer, so the
        # list is only rebuilt in the spatial view (where it can be edited).
        if self.tracks_viewer.view_mode != "kymograph":
            overlays = set(kymograph_layers.overlay_image_layer_names)
            with QSignalBlocker(self.overlay_list):
                self.overlay_list.clear()
                for name in image_layers[1:]:
                    if name == kymograph_layers.image_layer_name:
                        continue
                    item = QListWidgetItem(name)
                    item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                    item.setCheckState(Qt.Checked if name in overlays else Qt.Unchecked)
                    self.overlay_list.addItem(item)

        tracks = self.tracks_viewer.tracks
        can_show_kymograph = tracks is not None and tracks.ndim == 3
        self.view_mode_widget.button_for_mode("kymograph").setEnabled(
            can_show_kymograph
        )
        # the image layers can only be changed while the spatial view is shown
        editable = can_show_kymograph and self.tracks_viewer.view_mode != "kymograph"
        self.image_layer_box.setEnabled(editable)
        self.overlay_list.setEnabled(editable)
