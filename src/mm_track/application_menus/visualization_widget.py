import napari
import numpy as np
from psygnal import Signal
from qtpy.QtCore import QSignalBlocker, Qt
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QWidget,
)
from superqt import QLabeledDoubleSlider

from mm_track.data_views.views_coordinator.tracks_viewer import TracksViewer


def compact_box(title: str) -> tuple[QGroupBox, QFormLayout]:
    """A titled group with a tightly spaced form, sized to its contents."""
    box = QGroupBox(title)
    box.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)
    form = QFormLayout(box)
    form.setContentsMargins(6, 12, 6, 4)  # the top margin clears the title
    form.setHorizontalSpacing(6)
    form.setVerticalSpacing(2)
    return box, form


class VisualizationConfigWidget(QWidget):
    """An opacity slider, with a checkbox for filling the labels in contour mode, on
    one labelled row."""

    update_visualization = Signal()

    def __init__(
        self,
        label: str,
        default_opacity: float,
        default_contour: bool,
        use_contour: bool = True,
    ):
        super().__init__()

        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self.orth_views_connection = None
        self.orth_view_manager = None

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        # in the widget rather than in a form, so that it greys out with the slider
        name = QLabel(label)
        name.setMinimumWidth(name.fontMetrics().boundingRect("Background").width() + 20)
        layout.addWidget(name)

        self.opacity = QLabeledDoubleSlider()
        self.opacity.setValue(default_opacity)
        self.opacity.setSingleStep(0.1)
        self.opacity.setRange(0, 1)
        self.opacity.setDecimals(2)
        self.opacity.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.opacity.setMinimumWidth(150)
        self.opacity.valueChanged.connect(self.update_visualization)
        layout.addWidget(self.opacity)

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
            layout.addWidget(self.contour)


class VisualizationWidget(QWidget):
    """Widget to adjust the kymograph paging and its track lines and markers, and the
    opacity and contour display in the different TrackLabels layer display modes. The
    view and display modes are chosen in the View mode menu (ViewModeWidget)."""

    def __init__(self, viewer: napari.Viewer):
        super().__init__()

        self.viewer = viewer
        self.tracks_viewer = TracksViewer.get_instance(viewer)

        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)

        self.tracks_viewer.mode_updated.connect(self._update_widget_availability)
        self.tracks_viewer.view_mode_updated.connect(self._update_widget_availability)
        self.tracks_viewer.kymograph_updated.connect(self._update_widget_availability)
        self.tracks_viewer.tracks_updated.connect(self._update_widget_availability)

        self.kymograph_box = self._make_kymograph_box()

        self.highlight_widget = VisualizationConfigWidget("Highlight", 1.0, True)
        self.foreground_widget = VisualizationConfigWidget("Foreground", 0.6, True)
        self.background_widget = VisualizationConfigWidget(
            "Background", 0.3, True, use_contour=False
        )

        self.highlight_widget.update_visualization.connect(self._update_visualization)
        self.foreground_widget.update_visualization.connect(self._update_visualization)
        self.background_widget.update_visualization.connect(self._update_visualization)

        self.background_widget.setEnabled(False)  # initially disabled

        self.opacity_box, opacity_form = compact_box("Mask opacity")
        for widget in (
            self.highlight_widget,
            self.foreground_widget,
            self.background_widget,
        ):
            opacity_form.addRow(widget)

        # The menus share a short strip along the bottom of the window (see
        # MenuManager.arrange_default_layout), so the groups stand side by side; the
        # menu manager scrolls the strip when they are wider than it.
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(4, 2, 4, 2)
        main_layout.setSpacing(6)
        main_layout.addWidget(self.kymograph_box, 0, Qt.AlignTop)
        main_layout.addWidget(self.opacity_box, 0, Qt.AlignTop)
        self.mask_color_box = self._make_mask_color_box()
        main_layout.addWidget(self.mask_color_box, 0, Qt.AlignTop)
        main_layout.addStretch(1)

        self._update_widget_availability()

    # Diverging maps first (for properties with a sign), then sequential ones.
    MASK_COLORMAPS = ("PRGn", "PiYG", "coolwarm", "inferno", "inferno_r", "turbo", "viridis")

    def _make_mask_color_box(self) -> QGroupBox:
        """Colour the masks by track (default) or by a numeric node property."""
        self.mask_feature_box = QComboBox()
        self.mask_feature_box.setToolTip(
            "Colour each mask by this node property instead of its track. Masks without "
            "a value are grey."
        )
        self.mask_colormap_box = QComboBox()
        self.mask_colormap_box.addItems(self.MASK_COLORMAPS)
        self.mask_low_box = QDoubleSpinBox()
        self.mask_high_box = QDoubleSpinBox()
        for box in (self.mask_low_box, self.mask_high_box):
            box.setRange(-1e9, 1e9)
            box.setDecimals(2)
            box.setKeyboardTracking(False)
        self.mask_feature_box.currentTextChanged.connect(self._on_mask_feature)
        self.mask_colormap_box.currentTextChanged.connect(self._apply_mask_color)
        self.mask_low_box.valueChanged.connect(self._apply_mask_color)
        self.mask_high_box.valueChanged.connect(self._apply_mask_color)

        range_row = QHBoxLayout()
        range_row.addWidget(self.mask_low_box)
        range_row.addWidget(self.mask_high_box)
        form = QFormLayout()
        form.addRow("Property", self.mask_feature_box)
        form.addRow("Colormap", self.mask_colormap_box)
        form.addRow("Range", range_row)
        box = QGroupBox("Colour masks by")
        box.setLayout(form)
        return box

    def _refresh_mask_features(self) -> None:
        features = ["Track", *self.tracks_viewer.numeric_node_features()]
        setting = self.tracks_viewer.mask_color_by
        current = setting["feature"] if setting else "Track"
        with QSignalBlocker(self.mask_feature_box):
            self.mask_feature_box.clear()
            self.mask_feature_box.addItems(features)
            self.mask_feature_box.setCurrentText(current if current in features else "Track")
        if setting and current in features:  # show a setting made elsewhere (e.g. at launch)
            with (QSignalBlocker(self.mask_low_box), QSignalBlocker(self.mask_high_box),
                  QSignalBlocker(self.mask_colormap_box)):
                self.mask_low_box.setValue(setting["range"][0])
                self.mask_high_box.setValue(setting["range"][1])
                if self.mask_colormap_box.findText(setting["colormap"]) < 0:
                    self.mask_colormap_box.addItem(setting["colormap"])
                self.mask_colormap_box.setCurrentText(setting["colormap"])
        if setting and current not in features:  # the property went with the data
            self.tracks_viewer.set_mask_color_by(None)
        self._update_mask_controls()

    def _update_mask_controls(self) -> None:
        by_feature = self.mask_feature_box.currentText() not in ("", "Track")
        for widget in (self.mask_colormap_box, self.mask_low_box, self.mask_high_box):
            widget.setEnabled(by_feature)

    def _on_mask_feature(self, feature: str) -> None:
        """A newly chosen property starts on its 5th-95th percentile range (symmetric
        around 0 when it has values on both sides)."""
        if feature not in ("", "Track") and self.tracks_viewer.tracks is not None:
            _, values = self.tracks_viewer.node_feature_values(feature)
            values = values[np.isfinite(values)]
            if values.size:
                low, high = (float(v) for v in np.percentile(values, [5, 95]))
                if low < 0 < high:
                    low, high = -max(-low, high), max(-low, high)
                if high <= low:
                    high = low + 1
                with QSignalBlocker(self.mask_low_box), QSignalBlocker(self.mask_high_box):
                    self.mask_low_box.setValue(low)
                    self.mask_high_box.setValue(high)
        self._update_mask_controls()
        self._apply_mask_color()

    def _apply_mask_color(self, *_args) -> None:
        feature = self.mask_feature_box.currentText()
        if self.tracks_viewer.tracks is None:
            return
        setting = (
            None
            if feature in ("", "Track")
            else {
                "feature": feature,
                "range": (self.mask_low_box.value(), self.mask_high_box.value()),
                "colormap": self.mask_colormap_box.currentText(),
            }
        )
        if setting != self.tracks_viewer.mask_color_by:
            self.tracks_viewer.set_mask_color_by(setting)

    def _make_kymograph_box(self) -> QGroupBox:
        """Create the group box with the kymograph paging and link controls."""

        self.page_start_box = QSpinBox()
        self.page_start_box.setMinimum(0)
        self.page_start_box.setToolTip("First frame shown on the current page")
        self.page_start_box.valueChanged.connect(self._update_page_start)

        self.page_length_box = QSpinBox()
        self.page_length_box.setMinimum(1)
        self.page_length_box.setToolTip("Number of frames shown side by side")
        self.page_length_box.valueChanged.connect(self._update_page_length)

        self.prev_page_button = QPushButton("◀")
        self.prev_page_button.setToolTip("Previous page")
        self.prev_page_button.setFixedWidth(28)
        self.prev_page_button.clicked.connect(
            lambda: self.tracks_viewer.step_kymograph_page(-1)
        )
        self.next_page_button = QPushButton("▶")
        self.next_page_button.setToolTip("Next page")
        self.next_page_button.setFixedWidth(28)
        self.next_page_button.clicked.connect(
            lambda: self.tracks_viewer.step_kymograph_page(1)
        )

        self.paths_checkbox = QCheckBox("Paths")
        self.paths_checkbox.setChecked(True)
        self.paths_checkbox.setToolTip(
            "Draw lines between consecutive nodes of a track"
        )
        self.paths_checkbox.stateChanged.connect(self._update_show_paths)

        self.branches_checkbox = QCheckBox("Branches")
        self.branches_checkbox.setChecked(True)
        self.branches_checkbox.setToolTip("Draw dashed lines for division edges")
        self.branches_checkbox.stateChanged.connect(self._update_show_branches)

        self.boundaries_checkbox = QCheckBox("Frames")
        self.boundaries_checkbox.setToolTip("Draw the boundaries between frames")
        self.boundaries_checkbox.setChecked(True)
        self.boundaries_checkbox.stateChanged.connect(self._update_show_boundaries)

        kymograph_layers = self.tracks_viewer.kymograph_layers
        self.link_width_box = self._make_scale_box(
            "Thickness of the track lines, relative to the default",
            kymograph_layers.set_link_width_scale,
        )
        self.point_size_box = self._make_scale_box(
            "Size of the node markers, relative to the default",
            kymograph_layers.set_point_size_scale,
        )

        page_row = QHBoxLayout()
        page_row.setSpacing(2)
        page_row.addWidget(self.prev_page_button)
        page_row.addWidget(self.page_start_box)
        page_row.addWidget(self.next_page_button)
        show_row = QHBoxLayout()
        show_row.setSpacing(8)
        for checkbox in (
            self.paths_checkbox,
            self.branches_checkbox,
            self.boundaries_checkbox,
        ):
            show_row.addWidget(checkbox)
        box, form = compact_box("Page && tracks")
        form.addRow("Start", page_row)
        form.addRow("Length", self.page_length_box)
        form.addRow("Show", show_row)
        form.addRow("Lines", self.link_width_box)
        form.addRow("Markers", self.point_size_box)

        return box

    @staticmethod
    def _make_scale_box(tooltip: str, apply) -> QDoubleSpinBox:
        """A − value + box for a size factor, applied on every step so that holding a
        button (or the arrow keys) resizes the kymograph as it changes."""
        box = QDoubleSpinBox()
        box.setRange(0.1, 10.0)
        box.setSingleStep(0.1)
        box.setDecimals(1)
        box.setValue(1.0)
        box.setSuffix(" ×")
        box.setToolTip(tooltip)
        box.valueChanged.connect(apply)
        return box

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
        """Sync the widget with the TracksViewer: refresh the kymograph controls,
        and show/hide the opacity widgets (they only
        apply when a segmentation layer is present). Also show/hide the contour
        checkboxes when changing between contour and normal mode, and disable the
        background widget when the display mode is 'All', as there are no background
        labels in that case."""

        mode = self.tracks_viewer.mode

        self._update_kymograph_controls()
        self._refresh_mask_features()

        seg_layer = self.tracks_viewer.tracking_layers.seg_layer
        self.opacity_box.setVisible(seg_layer is not None)
        self.mask_color_box.setVisible(seg_layer is not None)
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
        for scale_box, scale in (
            (self.link_width_box, kymograph_layers.link_width_scale),
            (self.point_size_box, kymograph_layers.point_size_scale),
        ):
            with QSignalBlocker(scale_box):
                scale_box.setValue(scale)

        tracks = self.tracks_viewer.tracks
        can_show_kymograph = tracks is not None and tracks.ndim == 3
        self.kymograph_box.setEnabled(can_show_kymograph)

    def _update_visualization(self):
        """Apply the values from the widget and send an update signal.

        Returns without touching the layer when the widgets already agree with
        it. This runs on every `mode_updated`, which includes pressing Q - and
        there only the *enabled* state of these widgets changes, never their
        values. Applying them anyway triggers a second `update_selection`, and
        each one re-slices the whole displayed volume: 6 s in 3D on a
        1.7e9-voxel timepoint, for a result identical to the refresh
        `set_display_mode` has already done.
        """

        layers = [
            self.tracks_viewer.tracking_layers.seg_layer,
            self.tracks_viewer.kymograph_layers.labels_layer,
        ]
        wanted = (
            self.highlight_widget.opacity.value(),
            self.foreground_widget.opacity.value(),
            self.background_widget.opacity.value(),
            not self.highlight_widget.contour.isChecked(),
            not self.foreground_widget.contour.isChecked(),
        )
        updated = False
        for layer in layers:
            if layer is None:
                continue
            current = (
                layer.highlight_opacity,
                layer.foreground_opacity,
                layer.background_opacity,
                layer.highlight_contour,
                layer.foreground_contour,
            )
            if wanted == current:
                continue
            (
                layer.highlight_opacity,
                layer.foreground_opacity,
                layer.background_opacity,
                layer.highlight_contour,
                layer.foreground_contour,
            ) = wanted
            updated = True

        if updated:
            self.tracks_viewer.update_selection(set_view=False)
