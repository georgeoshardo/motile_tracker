from __future__ import annotations

from typing import TYPE_CHECKING

import napari
import numpy as np
from napari.utils.notifications import show_info

from motile_tracker.data_views.views.kymograph_utils import (
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
from motile_tracker.data_views.views.layers.kymograph_track_labels import (
    KymographTrackLabels,
)
from motile_tracker.data_views.views.layers.kymograph_track_links import (
    KymographTrackLinks,
)
from motile_tracker.data_views.views.layers.kymograph_track_points import (
    KymographTrackPoints,
)

if TYPE_CHECKING:
    from funtracks.data_model import Tracks

    from motile_tracker.data_views.views_coordinator.tracks_viewer import TracksViewer


class KymographLayerGroup:
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
        self.page_start = 0
        self.page_length = 1
        self.show_boundaries = True
        self.show_paths = True
        self.show_branches = True
        self.geometry: KymographGeometry | None = None
        self.active = False
        self.detached_image_layer: napari.layers.Image | None = None
        self.visible_nodes: list[int] | str = "all"

        self.background_layer: napari.layers.Image | None = None
        self.labels_layer: KymographTrackLabels | None = None
        self.points_layer: KymographTrackPoints | None = None
        self.continuation_links_layer: KymographTrackLinks | None = None
        self.branch_links_layer: KymographTrackLinks | None = None
        self.bounds_layer: napari.layers.Shapes | None = None

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
            self._refresh()

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

    def set_detached_image_layer(
        self,
        layer: napari.layers.Image | None,
    ) -> None:
        self.detached_image_layer = layer

    def set_page(self, *, page_start: int | None = None, page_length: int | None = None) -> None:
        if self.geometry is None:
            return
        if page_length is not None:
            self.page_length = max(1, int(page_length))
        if page_start is not None:
            self.page_start = int(page_start)
        self.page_start = clamp_page_start(self.page_start, self.geometry, self.page_length)
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
        return True, ""

    def deactivate(self) -> None:
        self.active = False
        self.remove_napari_layers()

    def remove_napari_layer(self, layer: napari.layers.Layer | None) -> None:
        if layer is not None and layer in self.viewer.layers:
            self.viewer.layers.remove(layer)

    def remove_napari_layers(self) -> None:
        self.remove_napari_layer(self.background_layer)
        self.remove_napari_layer(self.continuation_links_layer)
        self.remove_napari_layer(self.branch_links_layer)
        self.remove_napari_layer(self.bounds_layer)
        self.remove_napari_layer(self.points_layer)
        self.remove_napari_layer(self.labels_layer)
        self.background_layer = None
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
            if layer.name == self.image_layer_name and self._is_usable_image_layer(layer):
                return layer
        return None

    def _candidate_image_layers(self) -> list[napari.layers.Image]:
        layers = [
            layer
            for layer in self.viewer.layers
            if isinstance(layer, napari.layers.Image) and layer is not self.background_layer
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
                self.tracks.scale
                if self.tracks.scale is not None
                else (1.0, 1.0, 1.0)
            )
            track_scale = tuple(float(v) for v in track_scale_values)
            layer_scale = tuple(float(v) for v in layer.scale)
            if len(layer_scale) >= 3 and tuple(layer_scale[-3:]) != tuple(
                track_scale[-3:]
            ):
                return False
        return True

    def _page_image(self) -> np.ndarray | None:
        image_layer = self._resolved_image_layer()
        if image_layer is None:
            return None
        return concat_time_to_kymograph(
            np.asarray(image_layer.data),
            page_start=self.page_start,
            page_length=self.page_length,
        )

    def _page_nodes(self) -> list[int]:
        if self.tracks is None or self.geometry is None:
            return []
        start, stop = visible_frame_range(self.geometry, self.page_start, self.page_length)
        nodes: list[int] = []
        for node in self.tracks.graph.nodes:
            timepoint = self.tracks.get_time(node)
            if start <= timepoint < stop:
                nodes.append(node)
        return nodes

    def _refresh_background(self) -> None:
        image = self._page_image()
        if image is None:
            self.remove_napari_layer(self.background_layer)
            self.background_layer = None
            return

        if self.background_layer is None:
            self.background_layer = self.viewer.add_image(
                image,
                name=f"{self.name}_kymograph_image",
                scale=(self.geometry.y_scale, self.geometry.x_scale),
                colormap="gray",
                blending="opaque",
            )
        else:
            self.background_layer.data = image
            self.background_layer.scale = (self.geometry.y_scale, self.geometry.x_scale)

    def _refresh_labels(self) -> None:
        if self.tracks is None or self.tracks.segmentation is None or self.geometry is None:
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
        if self.tracks is None or self.geometry is None or self.tracks.graph.number_of_nodes() == 0:
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
            track_color_resolver=lambda track_id: self.tracks_viewer.colormap.map(track_id),
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
            )
            self.viewer.add_layer(layer)
        else:
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

        self.page_start = clamp_page_start(self.page_start, self.geometry, self.page_length)
        self._refresh_background()
        self._refresh_labels()
        self._refresh_points()
        self._refresh_links()
        self._refresh_boundaries()
        self._ensure_interactive_selection()
        if self.viewer.dims.ndim == 2:
            self.viewer.dims.axis_labels = ("y", "x(time)")

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

        preferred_layer = self.labels_layer or self.points_layer or self.background_layer
        if preferred_layer is None or preferred_layer not in self.viewer.layers:
            return

        self.viewer.layers.selection.clear()
        self.viewer.layers.selection.add(preferred_layer)

    def update_visible(self, visible_nodes: list[int] | str):
        self.visible_nodes = visible_nodes
        if self.points_layer is not None:
            self.points_layer.update_point_outline(visible_nodes)
        if self.labels_layer is not None:
            if isinstance(visible_nodes, str):
                visible = visible_nodes
            else:
                visible = [node for node in visible_nodes if node in self._page_nodes()]
            self.labels_layer.update_label_colormap(visible)
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
        start, stop = visible_frame_range(self.geometry, self.page_start, self.page_length)
        timepoint = self.tracks.get_time(node)
        return start <= timepoint < stop
