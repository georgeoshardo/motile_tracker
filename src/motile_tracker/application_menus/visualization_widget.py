import napari
from psygnal import Signal
from qtpy.QtCore import QSignalBlocker
from qtpy.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QRadioButton,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)
from superqt import QLabeledDoubleSlider

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

        box = QGroupBox(label)
        box_layout = QVBoxLayout(box)

        self.opacity = QLabeledDoubleSlider()
        self.opacity.setValue(default_opacity)
        self.opacity.setSingleStep(0.1)
        self.opacity.setRange(0, 1)
        self.opacity.setDecimals(2)
        self.opacity.valueChanged.connect(self.update_visualization)
        box_layout.addWidget(self.opacity)

        if use_contour:
            self.contour = QCheckBox("Fill")
            self.contour.setChecked(default_contour)
            self.contour.stateChanged.connect(self.update_visualization)
            self.contour.setEnabled(False)
            self.contour.setVisible(False)
            self.contour.setToolTip(
                "When checked, will fill labels instead of showing contours only"
            )
            box_layout.addWidget(self.contour)

        layout = QVBoxLayout(self)
        layout.addWidget(box)


class ModeWidget(QWidget):
    """Radio buttons for changing a viewer mode."""

    update_mode = Signal(str)

    def __init__(self, title: str, options: list[tuple[str, str]]):
        super().__init__()

        box = QGroupBox(title)

        self.radio_group = QButtonGroup(self)
        box_layout = QHBoxLayout(box)

        for index, (text, mode) in enumerate(options):
            btn = QRadioButton(text)
            btn.setProperty("mode", mode)
            self.radio_group.addButton(btn)
            box_layout.addWidget(btn)

            if index == 0:
                btn.setChecked(True)

        self.radio_group.buttonToggled.connect(self._on_toggled)

        layout = QHBoxLayout(self)
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


class LabelVisualizationWidget(QWidget):
    """Widget to adjust opacity and contour display in different TrackLabels layer display modes."""

    def __init__(self, viewer: napari.Viewer):
        super().__init__()

        self.tracks_viewer = TracksViewer.get_instance(viewer)
        layout = QVBoxLayout(self)

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

        self.image_layer_box = QComboBox()
        self.image_layer_box.currentTextChanged.connect(self._update_image_layer)
        self.page_start_box = QSpinBox()
        self.page_start_box.setMinimum(0)
        self.page_start_box.valueChanged.connect(self._update_page_start)
        self.page_length_box = QSpinBox()
        self.page_length_box.setMinimum(1)
        self.page_length_box.valueChanged.connect(self._update_page_length)
        self.prev_page_button = QPushButton("Previous")
        self.prev_page_button.clicked.connect(lambda: self.tracks_viewer.step_kymograph_page(-1))
        self.next_page_button = QPushButton("Next")
        self.next_page_button.clicked.connect(lambda: self.tracks_viewer.step_kymograph_page(1))
        self.boundaries_checkbox = QCheckBox("Show frame boundaries")
        self.boundaries_checkbox.setChecked(True)
        self.boundaries_checkbox.stateChanged.connect(self._update_show_boundaries)

        self.kymograph_box = QGroupBox("Kymograph")
        kymograph_layout = QVBoxLayout(self.kymograph_box)

        image_row = QHBoxLayout()
        image_row.addWidget(self.image_layer_box)
        kymograph_layout.addLayout(image_row)

        page_row = QHBoxLayout()
        page_row.addWidget(self.page_start_box)
        page_row.addWidget(self.page_length_box)
        kymograph_layout.addLayout(page_row)

        nav_row = QHBoxLayout()
        nav_row.addWidget(self.prev_page_button)
        nav_row.addWidget(self.next_page_button)
        kymograph_layout.addLayout(nav_row)
        kymograph_layout.addWidget(self.boundaries_checkbox)

        self.highlight_widget = VisualizationConfigWidget(
            "Highlight opacity", default_opacity=1.0, default_contour=True
        )
        self.foreground_widget = VisualizationConfigWidget(
            "Foreground opacity", default_opacity=0.6, default_contour=True
        )
        self.background_widget = VisualizationConfigWidget(
            "Background opacity",
            default_opacity=0.3,
            default_contour=True,
            use_contour=False,
        )

        self.highlight_widget.update_visualization.connect(self._update_visualization)
        self.foreground_widget.update_visualization.connect(self._update_visualization)
        self.background_widget.update_visualization.connect(self._update_visualization)

        self.background_widget.setEnabled(False)  # initially disabled

        layout.addWidget(self.view_mode_widget)
        layout.addWidget(self.mode_widget)
        layout.addWidget(self.kymograph_box)
        layout.addWidget(self.highlight_widget)
        layout.addWidget(self.foreground_widget)
        layout.addWidget(self.background_widget)

        self.setMaximumHeight(450)
        self._update_widget_availability()

    def _update_view_mode(self, mode: str) -> None:
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

    def _update_widget_availability(self):
        """Update the radio buttons, show/hide the contour checkboxes when changing
        between contour and normal mode. Disable the background widget when the display
        mode is 'All', as there are no background labels in that case."""

        with QSignalBlocker(self.mode_widget.radio_group):
            self.mode_widget.button_for_mode(self.tracks_viewer.mode).setChecked(True)
        with QSignalBlocker(self.view_mode_widget.radio_group):
            self.view_mode_widget.button_for_mode(self.tracks_viewer.view_mode).setChecked(
                True
            )

        image_layers = ["None", *self.tracks_viewer.kymograph_layers.available_image_layers()]
        with QSignalBlocker(self.image_layer_box):
            self.image_layer_box.clear()
            self.image_layer_box.addItems(image_layers)
            current_layer = self.tracks_viewer.kymograph_layers.image_layer_name or "None"
            self.image_layer_box.setCurrentText(
                current_layer if current_layer in image_layers else "None"
            )

        geometry = self.tracks_viewer.kymograph_layers.geometry
        page_length = self.tracks_viewer.kymograph_layers.page_length
        if geometry is not None:
            max_page_start = max(0, geometry.t_size - page_length)
            self.page_start_box.setMaximum(max_page_start)
            self.page_length_box.setMaximum(max(1, geometry.t_size))
        with QSignalBlocker(self.page_start_box):
            self.page_start_box.setValue(self.tracks_viewer.kymograph_layers.page_start)
        with QSignalBlocker(self.page_length_box):
            self.page_length_box.setValue(page_length)
        with QSignalBlocker(self.boundaries_checkbox):
            self.boundaries_checkbox.setChecked(
                self.tracks_viewer.kymograph_layers.show_boundaries
            )

        can_show_kymograph = (
            self.tracks_viewer.tracks is not None and self.tracks_viewer.tracks.ndim == 3
        )
        self.view_mode_widget.button_for_mode("kymograph").setEnabled(can_show_kymograph)
        self.kymograph_box.setEnabled(can_show_kymograph)
        self.image_layer_box.setEnabled(
            can_show_kymograph and self.tracks_viewer.view_mode != "kymograph"
        )

        seg_layer = self.tracks_viewer.tracking_layers.seg_layer
        has_seg = seg_layer is not None
        self.highlight_widget.setEnabled(has_seg)
        self.foreground_widget.setEnabled(has_seg)
        self.background_widget.setEnabled(has_seg and self.tracks_viewer.mode != "all")

        if has_seg:
            show_contour = seg_layer.contour > 0 and self.tracks_viewer.mode != "all"
            for w in (self.highlight_widget, self.foreground_widget):
                w.contour.setVisible(show_contour)
                w.contour.setEnabled(show_contour)
            self._update_visualization()
        else:
            for w in (self.highlight_widget, self.foreground_widget):
                w.contour.setVisible(False)
                w.contour.setEnabled(False)

    def _update_visualization(self):
        """Apply the values from the widget and send an update signal."""

        layer = self.tracks_viewer.tracking_layers.seg_layer

        if layer is not None:
            layer.highlight_opacity = self.highlight_widget.opacity.value()
            layer.foreground_opacity = self.foreground_widget.opacity.value()
            layer.background_opacity = self.background_widget.opacity.value()
            layer.highlight_contour = not self.highlight_widget.contour.isChecked()
            layer.foreground_contour = not self.foreground_widget.contour.isChecked()
            self.tracks_viewer.update_selection(set_view=False)
