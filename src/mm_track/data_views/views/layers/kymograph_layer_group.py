from __future__ import annotations

from typing import TYPE_CHECKING

import napari
import numpy as np
from napari.utils.notifications import show_info
from psygnal import Signal
from qtpy.QtCore import QEvent, QObject, QTimer

from mm_track.data_views.views.kymograph_time_slider import (
    KymographTimeSlider,
)
from mm_track.data_views.views.kymograph_utils import (
    KymographGeometry,
    KymographLinkRenderData,
    build_kymograph_link_data,
    clamp_page_start,
    concat_time_to_kymograph,
    default_page_length,
    frame_boundary_segments,
    infer_kymograph_geometry,
    point_to_kymograph_coords,
    visible_frame_range,
)
from mm_track.data_views.views.layers.kymograph_track_labels import (
    KymographTrackLabels,
)
from mm_track.data_views.views.layers.kymograph_track_links import (
    KymographTrackLinks,
)
from mm_track.data_views.views.layers.kymograph_track_points import (
    KymographTrackPoints,
)

if TYPE_CHECKING:
    from funtracks.data_model import Tracks

    from mm_track.data_views.views_coordinator.tracks_viewer import TracksViewer

# A Shift+drag shorter than this many screen pixels in both directions is a Shift+click,
# which the layer under the cursor handles (append one node).
MIN_BOX_PX = 5


class _LeaveFilter(QObject):
    """Calls ``on_leave`` when the mouse leaves the widget it is installed on."""

    def __init__(self, on_leave) -> None:
        super().__init__()
        self._on_leave = on_leave

    def eventFilter(self, obj, event) -> bool:  # noqa: N802 (Qt name)
        if event.type() == QEvent.Leave:
            self._on_leave()
        return False


def _zoom_box(viewer: napari.Viewer):
    """napari's zoom-box overlay: viewer._zoom_box up to napari 0.7, then
    viewer.canvas.overlays._zoom_box."""
    box = getattr(viewer, "_zoom_box", None)
    return box if box is not None else viewer.canvas.overlays._zoom_box


class KymographLayerGroup:
    # The frames on screen or the frame under the cursor changed (visible_frames,
    # hover_frame); the lineage tree draws both.
    view_changed = Signal()

    def __init__(
        self,
        viewer: napari.Viewer,
        tracks_viewer: TracksViewer,
    ):
        self.viewer = viewer
        self.tracks_viewer = tracks_viewer
        self.tracks: Tracks | None = None
        self.name = ""

        self.image_layer_name: str | None = None
        # further image layers (for example other channels) drawn as kymographs over
        # the background, each in its own colors with additive blending
        self.overlay_image_layer_names: list[str] = []
        self.page_start = 0
        self.page_length = 1
        self.show_boundaries = True
        self.show_paths = True
        self.show_branches = True
        # thickness of the track lines and size of the markers, relative to their
        # defaults; kept when the layers are rebuilt (a new page or trench)
        self.link_width_scale = 1.0
        self.point_size_scale = 1.0
        self.geometry: KymographGeometry | None = None
        self.active = False
        self.detached_image_layer: napari.layers.Image | None = None
        self.visible_nodes: list[int] | str = "all"
        # the visible_nodes the track lines were last built for (see update_visible)
        self._links_visible_nodes: list[int] | str | None = None

        self.background_layer: napari.layers.Image | None = None
        self._background_source: napari.layers.Image | None = None
        # overlay kymograph layers, keyed by the name of their source layer
        self.overlay_layers: dict[str, napari.layers.Image] = {}
        # slider under the canvas, spanning its width, that pans the kymograph
        # along time one frame per step; created once, shown only in kymograph mode
        self.time_slider: KymographTimeSlider | None = None
        self._time_slider_dock = None  # only used when the canvas layout is not found
        self._time_events_connected = False
        self._syncing_time = False
        # the frame under the cursor on the kymograph, None when it is elsewhere
        self.hover_frame: int | None = None
        self._leave_filter = _LeaveFilter(self._on_cursor_left)
        self.labels_layer: KymographTrackLabels | None = None
        self.points_layer: KymographTrackPoints | None = None
        self.continuation_links_layer: KymographTrackLinks | None = None
        self.branch_links_layer: KymographTrackLinks | None = None
        self.bounds_layer: napari.layers.Shapes | None = None
        # Shift+drag selects the nodes in a box, as in the lineage tree
        self.viewer.mouse_drag_callbacks.append(self._box_select)

    def set_tracks(self, tracks: Tracks | None, name: str) -> None:
        self.tracks = tracks
        self.name = name
        self.geometry = self._resolve_geometry()
        if self.geometry is not None:
            self.page_length = default_page_length(self.geometry)
            self.page_start = clamp_page_start(
                self.page_start,
                self.geometry,
                self.page_length,
            )
        if self.active:
            # rebuild the layers so that they pick up the new tracks and name
            self.remove_napari_layers()
            self._refresh()
            self.show_opening_view()

    def set_image_layer_name(self, layer_name: str | None) -> None:
        self.image_layer_name = layer_name or None
        self.geometry = self._resolve_geometry()
        if self.geometry is not None:
            self.page_length = default_page_length(self.geometry)
            self.page_start = clamp_page_start(
                self.page_start,
                self.geometry,
                self.page_length,
            )
        if self.active:
            self._refresh()

    def set_overlay_image_layer_names(self, layer_names: list[str]) -> None:
        """Select the image layers drawn as kymographs over the background."""
        self.overlay_image_layer_names = [name for name in layer_names if name]
        if self.active:
            self._refresh()

    def set_detached_image_layer(
        self,
        layer: napari.layers.Image | None,
    ) -> None:
        self.detached_image_layer = layer

    def set_page(
        self, *, page_start: int | None = None, page_length: int | None = None
    ) -> None:
        if self.geometry is None:
            return
        if page_length is not None:
            self.page_length = max(1, int(page_length))
        if page_start is not None:
            self.page_start = int(page_start)
        self.page_start = clamp_page_start(
            self.page_start, self.geometry, self.page_length
        )
        if self.active:
            self._refresh()

    def set_show_boundaries(self, show_boundaries: bool) -> None:
        self.show_boundaries = bool(show_boundaries)
        if self.bounds_layer is not None:
            self.bounds_layer.visible = self.show_boundaries

    def set_show_paths(self, show_paths: bool) -> None:
        self.show_paths = bool(show_paths)
        if self.continuation_links_layer is not None:
            self.continuation_links_layer.visible = self.show_paths

    def set_show_branches(self, show_branches: bool) -> None:
        self.show_branches = bool(show_branches)
        if self.branch_links_layer is not None:
            self.branch_links_layer.visible = self.show_branches

    def set_link_width_scale(self, scale: float) -> None:
        """Draw the track lines (paths and branches) ``scale`` times thicker."""
        self.link_width_scale = float(scale)
        for layer in (self.continuation_links_layer, self.branch_links_layer):
            if layer is not None:
                layer.set_width_scale(self.link_width_scale)

    def set_point_size_scale(self, scale: float) -> None:
        """Draw the node markers ``scale`` times larger."""
        self.point_size_scale = float(scale)
        if self.points_layer is not None:
            self.points_layer.set_size_scale(self.point_size_scale)

    def available_image_layers(self) -> list[str]:
        layers = []
        for layer in self._candidate_image_layers():
            if layer.name not in layers:
                layers.append(layer.name)
        return layers

    def validate_configuration(self) -> tuple[bool, str]:
        self.geometry = self._resolve_geometry()
        if self.tracks is None:
            return False, "Load tracks before entering kymograph mode."
        if self.tracks.ndim != 3:
            return False, "Kymograph mode currently supports 2D+time tracks only."
        if self.tracks.segmentation is None and self._resolved_image_layer() is None:
            return (
                False,
                "Point-only tracks require a matching image layer for kymograph mode.",
            )
        if self.geometry is None:
            return False, "Could not determine kymograph geometry."
        return True, ""

    def activate(self) -> tuple[bool, str]:
        ok, message = self.validate_configuration()
        if not ok:
            return ok, message
        self.active = True
        self._refresh()
        self.show_opening_view()
        return True, ""

    def deactivate(self) -> None:
        self.active = False
        self.remove_napari_layers()
        self.hover_frame = None
        self.view_changed.emit()

    def remove_napari_layer(self, layer: napari.layers.Layer | None) -> None:
        if layer is not None and layer in self.viewer.layers:
            self.viewer.layers.remove(layer)

    def remove_napari_layers(self) -> None:
        self._hide_time_slider()
        self.remove_napari_layer(self.background_layer)
        for layer in self.overlay_layers.values():
            self.remove_napari_layer(layer)
        self.remove_napari_layer(self.continuation_links_layer)
        self.remove_napari_layer(self.branch_links_layer)
        self.remove_napari_layer(self.bounds_layer)
        self.remove_napari_layer(self.points_layer)
        self.remove_napari_layer(self.labels_layer)
        self.background_layer = None
        self._background_source = None
        self.overlay_layers = {}
        self.continuation_links_layer = None
        self.branch_links_layer = None
        self.bounds_layer = None
        self.points_layer = None
        self.labels_layer = None

    def _resolve_geometry(self) -> KymographGeometry | None:
        if self.tracks is None:
            return None
        if self.tracks.ndim != 3:
            return None
        if self.tracks.segmentation is not None:
            return infer_kymograph_geometry(
                tuple(int(v) for v in self.tracks.segmentation.shape),
                self.tracks.scale,
            )

        image_layer = self._resolved_image_layer()
        if image_layer is None or image_layer.data.ndim != 3:
            return None
        return infer_kymograph_geometry(
            tuple(int(v) for v in image_layer.data.shape),
            image_layer.scale,
        )

    def _resolved_image_layer(self) -> napari.layers.Image | None:
        if not self.image_layer_name:
            return None

        for layer in self._candidate_image_layers():
            if layer.name == self.image_layer_name and self._is_usable_image_layer(
                layer
            ):
                return layer
        return None

    def _candidate_image_layers(self) -> list[napari.layers.Image]:
        layers = [
            layer
            for layer in self.viewer.layers
            if isinstance(layer, napari.layers.Image)
            and layer is not self.background_layer
            and not layer.multiscale
        ]
        if (
            self.detached_image_layer is not None
            and self.detached_image_layer not in layers
        ):
            layers.append(self.detached_image_layer)
        return layers

    def _is_usable_image_layer(self, layer: napari.layers.Image) -> bool:
        if layer.data.ndim != 3:
            return False

        if self.tracks is not None and self.tracks.segmentation is not None:
            if tuple(layer.data.shape) != tuple(self.tracks.segmentation.shape):
                return False
            track_scale_values = (
                self.tracks.scale if self.tracks.scale is not None else (1.0, 1.0, 1.0)
            )
            track_scale = tuple(float(v) for v in track_scale_values)
            layer_scale = tuple(float(v) for v in layer.scale)
            if len(layer_scale) >= 3 and tuple(layer_scale[-3:]) != tuple(
                track_scale[-3:]
            ):
                return False
        return True

    def _resolved_overlay_layers(self) -> list[napari.layers.Image]:
        """The usable overlay source layers, in the order they were selected.

        In kymograph mode the sources are detached from the viewer (with every other
        time-series layer), so the tracks viewer's detached layers are searched too.
        """
        detached = [
            layer
            for layer, _, _ in getattr(
                self.tracks_viewer, "_detached_kymograph_layers", []
            )
        ]
        candidates = [*self._candidate_image_layers(), *detached]
        background = self._resolved_image_layer()
        layers: list[napari.layers.Image] = []
        for name in self.overlay_image_layer_names:
            for layer in candidates:
                if (
                    layer.name == name
                    and layer is not background
                    and layer not in layers
                    and isinstance(layer, napari.layers.Image)
                    and not layer.multiscale
                    and self._is_usable_image_layer(layer)
                ):
                    layers.append(layer)
                    break
        return layers

    def _page_image_of(self, image_layer: napari.layers.Image) -> np.ndarray:
        # only the frames of the current page are read from the (possibly lazy) data
        return concat_time_to_kymograph(
            image_layer.data,
            page_start=self.page_start,
            page_length=self.page_length,
        )

    def _page_image(self) -> np.ndarray | None:
        image_layer = self._resolved_image_layer()
        if image_layer is None:
            return None
        return self._page_image_of(image_layer)

    def _page_nodes(self) -> list[int]:
        if self.tracks is None or self.geometry is None:
            return []
        start, stop = visible_frame_range(
            self.geometry, self.page_start, self.page_length
        )
        node_ids = self.tracks.graph.node_ids()
        if not node_ids:
            return []
        times = self.tracks.get_times(node_ids)  # bulk lookup: per-node calls are slow
        return [
            int(node)
            for node, timepoint in zip(node_ids, times, strict=True)
            if start <= timepoint < stop
        ]

    def _refresh_background(self) -> None:
        source = self._resolved_image_layer()
        if source is None:
            self.remove_napari_layer(self.background_layer)
            self.background_layer = None
            self._background_source = None
        else:
            image = self._page_image_of(source)
            scale = (self.geometry.y_scale, self.geometry.x_scale)
            if self.background_layer is None:
                self.background_layer = self.viewer.add_image(
                    image,
                    name=f"{self.name}_kymograph_image",
                    scale=scale,
                    colormap=source.colormap,
                    contrast_limits=source.contrast_limits,
                    blending="opaque",
                )
            else:
                self.background_layer.data = image
                self.background_layer.scale = scale
                if source is not self._background_source:
                    # a different image: take over its colors once, so contrast
                    # adjusted on the kymograph survives paging
                    self.background_layer.colormap = source.colormap
                    self.background_layer.contrast_limits = source.contrast_limits
            self._background_source = source
        self._refresh_overlays()

    def _refresh_overlays(self) -> None:
        """Draw each overlay image as a kymograph over the background."""
        sources = self._resolved_overlay_layers()
        wanted = {layer.name for layer in sources}
        for name in list(self.overlay_layers):
            if name not in wanted:
                self.remove_napari_layer(self.overlay_layers.pop(name))
        scale = (self.geometry.y_scale, self.geometry.x_scale)
        for source in sources:
            image = self._page_image_of(source)
            layer = self.overlay_layers.get(source.name)
            if layer is None:
                below = [
                    self.viewer.layers.index(existing)
                    for existing in (
                        self.background_layer,
                        *self.overlay_layers.values(),
                    )
                    if existing is not None and existing in self.viewer.layers
                ]
                # shown or hidden as its source is: a channel hidden in the image is
                # listed, ready to be switched on, but not drawn
                layer = self.viewer.add_image(
                    image,
                    name=f"{self.name}_kymograph_{source.name}",
                    scale=scale,
                    colormap=source.colormap,
                    contrast_limits=source.contrast_limits,
                    blending="additive",
                    visible=source.visible,
                )
                self.overlay_layers[source.name] = layer
                # keep the overlays with the background, under labels and points
                if below and max(below) + 1 < self.viewer.layers.index(layer):
                    self.viewer.layers.move(
                        self.viewer.layers.index(layer), max(below) + 1
                    )
            else:
                layer.data = image
                layer.scale = scale

    def _refresh_labels(self) -> None:
        if (
            self.tracks is None
            or self.tracks.segmentation is None
            or self.geometry is None
        ):
            self.remove_napari_layer(self.labels_layer)
            self.labels_layer = None
            return

        if self.labels_layer is None:
            self.labels_layer = KymographTrackLabels(
                viewer=self.viewer,
                name=f"{self.name}_kymograph_seg",
                opacity=0.6,
                tracks_viewer=self.tracks_viewer,
                geometry=self.geometry,
                page_start=self.page_start,
                page_length=self.page_length,
            )
            self.viewer.add_layer(self.labels_layer)
        else:
            self.labels_layer.update_page(
                geometry=self.geometry,
                page_start=self.page_start,
                page_length=self.page_length,
            )

    def _refresh_points(self) -> None:
        if (
            self.tracks is None
            or self.geometry is None
            or self.tracks.graph.num_nodes() == 0
        ):
            self.remove_napari_layer(self.points_layer)
            self.points_layer = None
            return

        if self.points_layer is None:
            self.points_layer = KymographTrackPoints(
                name=f"{self.name}_kymograph_points",
                tracks_viewer=self.tracks_viewer,
                geometry=self.geometry,
                page_start=self.page_start,
                page_length=self.page_length,
                size_scale=self.point_size_scale,
            )
            self.viewer.add_layer(self.points_layer)
        else:
            self.points_layer.update_page(
                geometry=self.geometry,
                page_start=self.page_start,
                page_length=self.page_length,
            )

    def _link_render_data(
        self,
    ) -> tuple[KymographLinkRenderData, KymographLinkRenderData]:
        if self.tracks is None or self.geometry is None:
            return (KymographLinkRenderData.empty(), KymographLinkRenderData.empty())

        return build_kymograph_link_data(
            tracks=self.tracks,
            geometry=self.geometry,
            page_start=self.page_start,
            page_length=self.page_length,
            track_color_resolver=lambda track_id: self.tracks_viewer.colormap.map(
                track_id
            ),
            visible_nodes=self.visible_nodes,
        )

    def _sync_link_layer(
        self,
        *,
        layer: KymographTrackLinks | None,
        render_data: KymographLinkRenderData,
        name: str,
        edge_width: float,
        visible: bool,
    ) -> KymographTrackLinks | None:
        if not render_data.segments:
            self.remove_napari_layer(layer)
            return None

        if layer is None:
            layer = KymographTrackLinks(
                name=name,
                tracks_viewer=self.tracks_viewer,
                render_data=render_data,
                edge_width=edge_width,
                width_scale=self.link_width_scale,
            )
            self.viewer.add_layer(layer)
        else:
            layer.width_scale = self.link_width_scale
            layer.update_render_data(render_data)
        layer.visible = visible
        return layer

    def _refresh_links(self) -> None:
        continuation_data, branch_data = self._link_render_data()
        self.continuation_links_layer = self._sync_link_layer(
            layer=self.continuation_links_layer,
            render_data=continuation_data,
            name=f"{self.name}_kymograph_paths",
            edge_width=1.2,
            visible=self.show_paths,
        )
        self.branch_links_layer = self._sync_link_layer(
            layer=self.branch_links_layer,
            render_data=branch_data,
            name=f"{self.name}_kymograph_branches",
            edge_width=1.0,
            visible=self.show_branches,
        )
        self._links_visible_nodes = (
            self.visible_nodes
            if isinstance(self.visible_nodes, str)
            else list(self.visible_nodes)
        )

    def _refresh_boundaries(self) -> None:
        if self.geometry is None:
            return

        bounds = frame_boundary_segments(
            self.geometry,
            page_start=self.page_start,
            page_length=self.page_length,
        )
        if self.bounds_layer is None:
            self.bounds_layer = self.viewer.add_shapes(
                bounds,
                shape_type="line",
                name=f"{self.name}_kymograph_bounds",
                edge_color="yellow",
                edge_width=0.5,
                face_color="transparent",
                opacity=0.7,
            )
            self.bounds_layer.editable = False
        else:
            self.bounds_layer.data = bounds
        self.bounds_layer.visible = self.show_boundaries

    def _refresh(self) -> None:
        if not self.active:
            return

        self.geometry = self._resolve_geometry()
        ok, message = self.validate_configuration()
        if not ok:
            show_info(message)
            self.remove_napari_layers()
            return

        self.page_start = clamp_page_start(
            self.page_start, self.geometry, self.page_length
        )
        self._refresh_background()
        self._refresh_labels()
        self._refresh_points()
        self._refresh_links()
        self._refresh_boundaries()
        self._refresh_time_slider()
        self._ensure_interactive_selection()
        if self.viewer.dims.ndim == 2:
            self.viewer.dims.axis_labels = ("y", "x(time)")
        self._sync_time_slider_from_camera()
        self.view_changed.emit()

    def _ensure_interactive_selection(self) -> None:
        interactive_layers = [
            layer
            for layer in (
                self.labels_layer,
                self.points_layer,
                self.continuation_links_layer,
                self.branch_links_layer,
                self.background_layer,
            )
            if layer is not None and layer in self.viewer.layers
        ]
        if not interactive_layers:
            return

        active_layer = self.viewer.layers.selection.active
        if active_layer in interactive_layers:
            return

        preferred_layer = (
            self.labels_layer or self.points_layer or self.background_layer
        )
        if preferred_layer is None or preferred_layer not in self.viewer.layers:
            return

        self.viewer.layers.selection.clear()
        self.viewer.layers.selection.add(preferred_layer)

    # ------------------------------------------------------------------
    # time slider
    # ------------------------------------------------------------------
    def _refresh_time_slider(self) -> None:
        """Show the time slider under the canvas, fitted to the movie.

        The slider is created and placed once and hidden when the kymograph is not
        shown. Moving it pans the view to that frame (see show_timepoint) and
        panning the view moves it (see _sync_time_slider_from_camera).
        """
        if self.geometry is None:
            return
        if self.time_slider is None:
            self.time_slider = KymographTimeSlider()
            self.time_slider.timepoint_changed.connect(self.show_timepoint)
            self._attach_time_slider()
        self.time_slider.set_range(self.geometry.t_size, self.page_length)
        self.time_slider.setVisible(True)
        if self._time_slider_dock is not None:
            self._time_slider_dock.setVisible(True)
        self._connect_time_events()

    def _attach_time_slider(self) -> None:
        """Place the slider directly under the canvas, spanning its full width.

        napari's central viewer widget stacks the canvas and its own dims sliders
        in a vertical layout; the time slider goes in there right below the canvas.
        Should that layout not be found (another napari layout, or a viewer without
        a window), the slider is docked at the bottom of the window instead.
        """
        window = getattr(self.viewer, "window", None)
        if window is None:
            return
        try:
            canvas = window._qt_viewer.canvas.native
            layout = canvas.parentWidget().layout()
            index = layout.indexOf(canvas)
        except AttributeError:
            layout, index = None, -1
        if layout is None or index < 0:
            self._time_slider_dock = window.add_dock_widget(
                self.time_slider, area="bottom", name="Kymograph time"
            )
            return
        # A slider left behind by an earlier layer group for this viewer (a new
        # TracksViewer is made per test, for instance) is replaced, not stacked:
        # every extra row would shrink the canvas.
        for position in reversed(range(layout.count())):
            widget = layout.itemAt(position).widget()
            if (
                isinstance(widget, KymographTimeSlider)
                and widget is not self.time_slider
            ):
                layout.removeWidget(widget)
                widget.hide()
                widget.setParent(None)
                widget.deleteLater()
        layout.insertWidget(layout.indexOf(canvas) + 1, self.time_slider)

    def _hide_time_slider(self) -> None:
        self._disconnect_time_events()
        if self.time_slider is not None:
            self.time_slider.setVisible(False)
        if self._time_slider_dock is not None:
            self._time_slider_dock.setVisible(False)

    def _connect_time_events(self) -> None:
        if self._time_events_connected:
            return
        self.viewer.camera.events.center.connect(self._on_camera_center)
        self.viewer.camera.events.zoom.connect(self._on_camera_zoom)
        self.viewer.cursor.events.position.connect(self._on_cursor_moved)
        canvas = self._canvas_widget()
        if canvas is not None:
            canvas.installEventFilter(self._leave_filter)
        self._time_events_connected = True

    def _disconnect_time_events(self) -> None:
        if not self._time_events_connected:
            return
        self.viewer.camera.events.center.disconnect(self._on_camera_center)
        self.viewer.camera.events.zoom.disconnect(self._on_camera_zoom)
        self.viewer.cursor.events.position.disconnect(self._on_cursor_moved)
        canvas = self._canvas_widget()
        if canvas is not None:
            canvas.removeEventFilter(self._leave_filter)
        self._time_events_connected = False

    def _canvas_widget(self):
        try:
            return self.viewer.window._qt_viewer.canvas.native
        except AttributeError:  # a viewer without a window
            return None

    def visible_frames(self) -> tuple[float, float] | None:
        """The stretch of time on screen, in frames, from the canvas' left edge to
        its right edge and limited to the frames on the page; frame t spans
        t - 0.5 .. t + 0.5. None outside the kymograph view."""
        geometry = self.geometry
        zoom = float(self.viewer.camera.zoom)
        if not self.active or geometry is None or zoom <= 0:
            return None
        canvas_width = float(getattr(self.viewer, "_canvas_size", (600, 800))[1])
        half_width = canvas_width / zoom / 2.0
        x_center = float(self.viewer.camera.center[-1])

        def frame_at(x: float) -> float:
            # pixel edges sit at half-steps, frames are x_size pixels wide
            return self.page_start + (x / geometry.x_scale + 0.5) / geometry.x_size

        start, stop = visible_frame_range(geometry, self.page_start, self.page_length)
        low = max(frame_at(x_center - half_width), start) - 0.5
        high = min(frame_at(x_center + half_width), stop) - 0.5
        return (low, high) if high > low else None

    def _frame_under_cursor(self) -> int | None:
        geometry = self.geometry
        if not self.active or geometry is None:
            return None
        y, x = (float(v) for v in self.viewer.cursor.position[-2:])
        row = y / geometry.y_scale + 0.5
        column = x / geometry.x_scale + 0.5
        if not 0 <= row < geometry.y_size or column < 0:
            return None
        timepoint = self.page_start + int(column // geometry.x_size)
        start, stop = visible_frame_range(geometry, self.page_start, self.page_length)
        return timepoint if start <= timepoint < stop else None

    def _set_hover_frame(self, frame: int | None) -> None:
        if frame != self.hover_frame:
            self.hover_frame = frame
            self.view_changed.emit()

    def _on_cursor_moved(self, event=None) -> None:
        self._set_hover_frame(self._frame_under_cursor())

    def _box_select(self, viewer: napari.Viewer, event):
        """Shift+drag on the kymograph: draw a box and toggle the nodes in it, like
        Shift+drag in the lineage tree (nodes already selected are deselected).

        Only in pan/zoom mode (the active layer's, or with none active), where napari
        does not pan while a modifier is held. The box is napari's own zoom-box overlay
        (Alt+drag), drawn without zooming. A node is in the box when its marker's centre
        is; nodes hidden on the kymograph are left out.
        """
        if (
            not self.active
            or self.points_layer is None
            or event.type != "mouse_press"
            or event.button != 1
            or set(event.modifiers) != {"Shift"}
        ):
            return
        active = viewer.layers.selection.active
        if active is not None and active.mode != "pan_zoom":
            return
        box = _zoom_box(viewer)
        press_pos = move_pos = event.pos[::-1]
        press = release = event.position
        before = set(self.tracks_viewer.selected_nodes.as_list)
        yield
        while event.type == "mouse_move":
            move_pos = event.pos[::-1]
            release = event.position
            box.position = (press_pos, move_pos)
            box.visible = True
            yield
        box.visible = False
        if np.abs(np.subtract(press_pos, move_pos)).max() < MIN_BOX_PX:
            return
        picked = set(self.points_layer.nodes_in_box(press, release))
        if not picked:
            return
        # after the layer under the cursor has handled the release: a quick drag counts
        # as a click there (see detect_click) and may have toggled the node it started
        # on, so the box toggles against the selection from before the press
        QTimer.singleShot(0, lambda: self._select_box(before ^ picked))

    def _select_box(self, nodes: set[int]) -> None:
        with self.tracks_viewer.selection_updates(set_view=False):
            if nodes:
                self.tracks_viewer.selected_nodes.add_list(sorted(nodes))
            else:
                self.tracks_viewer.selected_nodes.reset()

    def _on_cursor_left(self) -> None:
        self._set_hover_frame(None)

    def _on_camera_zoom(self, event=None) -> None:
        if self.active:
            self.view_changed.emit()

    @property
    def current_timepoint(self) -> int | None:
        """The timepoint the time slider points at, or None without a slider."""
        if self.time_slider is None:
            return None
        return self.time_slider.value()

    def _on_camera_center(self, event=None) -> None:
        if not self.active or self.geometry is None:
            return
        self.view_changed.emit()
        if self._syncing_time:
            return
        self._sync_time_slider_from_camera()

    def _frame_center_x(self, timepoint: int) -> float:
        """World x of the centre of a frame on the current page. Pixel i is centred
        at i * scale, so a frame of width w spans pixels t*w .. t*w + w - 1."""
        geometry = self.geometry
        column = (timepoint - self.page_start) * geometry.x_size
        return (column + (geometry.x_size - 1) / 2.0) * geometry.x_scale

    def _timepoint_at_camera_center(self) -> int:
        """The timepoint whose frame is under the centre of the view, limited to
        the frames on the current page."""
        geometry = self.geometry
        x_center = float(self.viewer.camera.center[-1])
        start, stop = visible_frame_range(
            self.geometry, self.page_start, self.page_length
        )
        column = x_center / geometry.x_scale + 0.5  # pixel edges sit at half-steps
        timepoint = start + int(np.floor(column / geometry.x_size))
        return int(np.clip(timepoint, start, max(start, stop - 1)))

    def _sync_time_slider_from_camera(self) -> None:
        if self._syncing_time or self.geometry is None or self.time_slider is None:
            return
        self.time_slider.set_value_silently(self._timepoint_at_camera_center())

    def _visible_half_width_frames(self) -> float:
        """Half the width of the canvas, in frames, at the current zoom."""
        if self.geometry is None:
            return 0.0
        zoom = float(self.viewer.camera.zoom)
        if zoom <= 0:
            return 0.0
        canvas_size = getattr(self.viewer, "_canvas_size", (600, 800))
        canvas_width = float(canvas_size[1])
        return canvas_width / zoom / self.geometry.frame_world_width / 2.0

    def show_timepoint(self, timepoint: int) -> None:
        """Pan the view so that the frame at `timepoint` is centred, turning the
        page first when that frame (or the part of the movie around it that fits on
        screen) is not on the current page.
        """
        if self.geometry is None or not self.active:
            return
        t_size = int(self.geometry.t_size)
        timepoint = int(np.clip(int(timepoint), 0, t_size - 1))

        self._syncing_time = True
        try:
            start, stop = visible_frame_range(
                self.geometry, self.page_start, self.page_length
            )
            half_width = self._visible_half_width_frames()
            off_page = timepoint < start or timepoint >= stop
            # the view would show empty space where more frames exist; only worth
            # a page turn when the view fits on a page at all
            view_fits = 2 * half_width <= self.page_length
            view_past_start = view_fits and timepoint - half_width < start and start > 0
            view_past_stop = (
                view_fits and timepoint + half_width >= stop and stop < t_size
            )
            if off_page or view_past_start or view_past_stop:
                new_start = clamp_page_start(
                    timepoint - self.page_length // 2, self.geometry, self.page_length
                )
                if new_start != self.page_start:
                    self.set_page(page_start=new_start)

            x_center = self._frame_center_x(timepoint)
            camera_center = list(self.viewer.camera.center)
            camera_center[-1] = x_center
            self.viewer.camera.center = tuple(camera_center)
            if self.time_slider is not None:
                self.time_slider.set_value_silently(timepoint)
        finally:
            self._syncing_time = False

    def show_opening_view(self, *, deferred: bool = True) -> None:
        """Show the start of the movie: the first frame centred, zoomed so that a
        frame's height fills the canvas.

        Applied when the kymograph opens or shows other tracks. The canvas may
        still change size while the window is being laid out (at start-up, or when
        docks appear), so by default it is applied once more after Qt has processed
        pending events.
        """
        if self.geometry is None or not self.active:
            return
        geometry = self.geometry
        if self.page_start != 0:
            self.set_page(page_start=0)
        canvas_height = float(getattr(self.viewer, "_canvas_size", (0, 0))[0])
        frame_height = geometry.y_size * geometry.y_scale
        if canvas_height > 0 and frame_height > 0:
            self.viewer.camera.zoom = canvas_height / frame_height
        center = list(self.viewer.camera.center)
        center[-2] = (geometry.y_size - 1) / 2.0 * geometry.y_scale
        self.viewer.camera.center = tuple(center)
        self.show_timepoint(0)
        if deferred:
            from qtpy.QtCore import QTimer

            tracks = self.tracks
            QTimer.singleShot(
                0,
                lambda: (
                    self.show_opening_view(deferred=False)
                    if self.active and self.tracks is tracks
                    else None
                ),
            )

    def update_visible(self, visible_nodes: list[int] | str):
        self.visible_nodes = visible_nodes
        if self.points_layer is not None:
            self.points_layer.update_point_outline(visible_nodes)
        if self.labels_layer is not None:
            if isinstance(visible_nodes, str):
                visible = visible_nodes
            else:
                page_nodes = set(self._page_nodes())
                visible = [node for node in visible_nodes if node in page_nodes]
            self.labels_layer.update_label_colormap(visible)
        # This runs on every selection change, but the lines only change when other
        # nodes are shown: whatever else changes them (edits, colors, the page)
        # rebuilds them through _refresh.
        if visible_nodes != self._links_visible_nodes:
            self._refresh_links()

    def center_view(self, node: int):
        if self.tracks is None or self.geometry is None:
            return

        coords = point_to_kymograph_coords(
            timepoint=self.tracks.get_time(node),
            position=self.tracks.get_position(node),
            geometry=self.geometry,
            page_start=self.page_start,
            page_length=self.page_length,
        )
        if coords is None:
            return

        y_pos, x_pos = coords
        camera_center = list(self.viewer.camera.center)
        camera_center[-2] = y_pos
        camera_center[-1] = x_pos
        self.viewer.camera.center = tuple(camera_center)

    def node_on_page(self, node: int) -> bool:
        if self.tracks is None or self.geometry is None:
            return False
        start, stop = visible_frame_range(
            self.geometry, self.page_start, self.page_length
        )
        timepoint = self.tracks.get_time(node)
        return start <= timepoint < stop
