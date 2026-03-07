from __future__ import annotations

import math
import warnings
from typing import TYPE_CHECKING

import napari
import numpy as np
from funtracks.data_model import Tracks
from funtracks.exceptions import InvalidActionError
from funtracks.user_actions import UserAddNode, UserDeleteNodes, UserUpdateNodeAttrs
from napari.layers import Points
from napari.utils.notifications import show_info

from motile_tracker.data_views.keybindings_config import KEYMAP, bind_keymap
from motile_tracker.data_views.node_type import NodeType
from motile_tracker.data_views.views.kymograph_utils import (
    KymographGeometry,
    kymograph_coords_to_indices,
    point_to_kymograph_coords,
)
from motile_tracker.data_views.views.layers.click_utils import (
    detect_click,
    get_click_value,
)
from motile_tracker.data_views.views.layers.track_points import custom_select
from motile_tracker.data_views.views_coordinator.user_dialogs import (
    confirm_force_operation,
)

if TYPE_CHECKING:
    from napari.utils.events import Event

    from motile_tracker.data_views.views_coordinator.tracks_viewer import TracksViewer


class KymographTrackPoints(Points):
    _drag_modes = Points._drag_modes.copy()
    _drag_modes[Points._modeclass.SELECT] = custom_select

    @property
    def _type_string(self) -> str:
        return "points"

    def __init__(
        self,
        *,
        name: str,
        tracks_viewer: TracksViewer,
        geometry: KymographGeometry,
        page_start: int,
        page_length: int,
    ):
        self.tracks_viewer = tracks_viewer
        self.geometry = geometry
        self.page_start = int(page_start)
        self.page_length = int(page_length)
        self.nodes: list[int] = []
        self.node_index_dict: dict[int, int] = {}
        self.default_size = 5

        super().__init__(
            data=np.zeros((0, 2), dtype=float),
            name=name,
            symbol=[],
            face_color=[],
            size=self.default_size,
            properties={"node_id": [], "track_id": []},
            border_color=[1, 1, 1, 1],
            blending="translucent_no_depth",
        )

        bind_keymap(self, KEYMAP, self.tracks_viewer)

        @self.mouse_drag_callbacks.append
        def click(layer, event):
            if event.type == "mouse_press" and self.mode == "pan_zoom":
                was_click = yield from detect_click(event)
                if was_click:
                    value = get_click_value(self, event)
                    self.process_click(event, value)

        self.events.data.connect(self._update_data)
        self.events.current_size.connect(
            lambda: self.set_point_size(size=self.current_size)
        )
        self.selected_data.events.items_changed.connect(self._update_selection)
        self._refresh()

    def update_page(
        self,
        *,
        geometry: KymographGeometry,
        page_start: int,
        page_length: int,
    ) -> None:
        self.geometry = geometry
        self.page_start = int(page_start)
        self.page_length = int(page_length)
        self._refresh()

    def add(self, coords: list[float]):
        with self.events.current_size.blocker():
            super().add(coords)

    def process_click(
        self,
        event: Event,
        point_index: int | None,
        _layer: napari.layers.Points | None = None,
    ):
        with self.tracks_viewer.selection_updates(set_view=False):
            if point_index is None:
                self.tracks_viewer.selected_nodes.reset()
            else:
                node_id = self.nodes[point_index]
                append = "Shift" in event.modifiers
                jump = "Control" in event.modifiers
                if jump:
                    self.tracks_viewer.center_on_node(node_id)
                else:
                    self.tracks_viewer.selected_nodes.add(node_id, append)

    def set_point_size(self, size: int) -> None:
        self.default_size = size
        self._refresh()

    def _page_nodes(self) -> list[int]:
        nodes: list[int] = []
        start = self.page_start
        stop = self.page_start + self.page_length
        for node in self.tracks_viewer.tracks.graph.nodes:
            timepoint = self.tracks_viewer.tracks.get_time(node)
            if start <= timepoint < stop:
                nodes.append(node)
        return nodes

    def _node_data(self, node: int) -> tuple[float, float] | None:
        coords = point_to_kymograph_coords(
            timepoint=self.tracks_viewer.tracks.get_time(node),
            position=self.tracks_viewer.tracks.get_position(node),
            geometry=self.geometry,
            page_start=self.page_start,
            page_length=self.page_length,
        )
        if coords is None:
            return None
        return coords

    def _refresh(self):
        self.events.data.disconnect(self._update_data)

        self.nodes = self._page_nodes()
        self.node_index_dict = {node: idx for idx, node in enumerate(self.nodes)}

        track_ids = [self.tracks_viewer.tracks.get_track_id(node) for node in self.nodes]
        coords = [self._node_data(node) for node in self.nodes]
        coords = [coord for coord in coords if coord is not None]
        self.data = np.asarray(coords, dtype=float) if coords else np.zeros((0, 2), dtype=float)
        self.symbol = self.get_symbols(self.tracks_viewer.tracks, self.tracks_viewer.symbolmap)
        self.face_color = [
            self.tracks_viewer.colormap.map(track_id) for track_id in track_ids[: len(self.data)]
        ]
        self.properties = {
            "node_id": self.nodes[: len(self.data)],
            "track_id": track_ids[: len(self.data)],
        }
        self.size = self.default_size
        self.border_color = [1, 1, 1, 1]

        self.events.data.connect(self._update_data)

    def _map_kymograph_point(self, point: np.ndarray):
        coords = kymograph_coords_to_indices(
            y_world=float(point[0]),
            x_global_world=float(point[1]),
            geometry=self.geometry,
            page_start=self.page_start,
            page_length=self.page_length,
        )
        if coords is None:
            return None

        timepoint, _, x_idx = coords
        x_local_world = float(point[1]) - (
            (timepoint - self.page_start) * self.geometry.frame_world_width
        )
        return int(timepoint), [float(point[0]), x_local_world]

    def _create_node_attrs(self, new_point: np.ndarray) -> dict | None:
        mapped = self._map_kymograph_point(new_point)
        if mapped is None:
            return None

        timepoint, position = mapped
        if self.tracks_viewer.selected_track is None:
            self.tracks_viewer.set_new_track_id()

        features = self.tracks_viewer.tracks.features
        return {
            features.position_key: position,
            features.time_key: timepoint,
            features.tracklet_key: self.tracks_viewer.selected_track,
        }

    def _update_data(self, event: Event):
        if event.action == "added":
            if self.tracks_viewer.tracking_layers.seg_layer is None:
                attributes = self._create_node_attrs(event.value[-1])
                if attributes is None:
                    self._refresh()
                    return
                try:
                    new_node_id = self.tracks_viewer.tracks._get_new_node_ids(1)[0]
                    UserAddNode(
                        self.tracks_viewer.tracks,
                        node=new_node_id,
                        attributes=attributes,
                        force=self.tracks_viewer.force,
                    )
                except InvalidActionError as e:
                    if e.forceable:
                        force, always_force = confirm_force_operation(message=str(e))
                        self.tracks_viewer.force = always_force
                        self._refresh()
                        if force:
                            new_node_id = self.tracks_viewer.tracks._get_new_node_ids(1)[0]
                            UserAddNode(
                                self.tracks_viewer.tracks,
                                node=new_node_id,
                                attributes=attributes,
                                force=True,
                            )
                    else:
                        warnings.warn(str(e), stacklevel=2)
                        self._refresh()
            else:
                show_info(
                    "Mixed point and segmentation nodes not allowed: add nodes by "
                    "drawing on the labels layer"
                )
                self._refresh()

        elif event.action == "removed":
            UserDeleteNodes(
                self.tracks_viewer.tracks,
                nodes=self.tracks_viewer.selected_nodes.as_list,
            )

        elif event.action == "changed":
            if self.tracks_viewer.tracking_layers.seg_layer is None:
                time_key = self.tracks_viewer.tracks.features.time_key
                pos_key = self.tracks_viewer.tracks.features.position_key
                for ind in self.selected_data:
                    mapped = self._map_kymograph_point(self.data[ind])
                    if mapped is None:
                        self._refresh()
                        return
                    timepoint, position = mapped
                    node_id = self.properties["node_id"][ind]
                    if timepoint != self.tracks_viewer.tracks.get_time(node_id):
                        show_info("Moving nodes across frame boundaries is not supported.")
                        self._refresh()
                        return
                    UserUpdateNodeAttrs(
                        self.tracks_viewer.tracks,
                        node=node_id,
                        attrs={pos_key: position},
                    )
            else:
                self._refresh()

    def _update_selection(self):
        if self.mode == "select":
            selected_points = self.selected_data
            with self.tracks_viewer.selection_updates(set_view=False):
                self.tracks_viewer.selected_nodes.reset()
                for point in selected_points:
                    node_id = self.nodes[point]
                    self.tracks_viewer.selected_nodes.add(node_id, True)

    def get_symbols(self, tracks: Tracks, symbolmap: dict[NodeType, str]) -> list[str]:
        statemap = {
            0: NodeType.END,
            1: NodeType.CONTINUE,
            2: NodeType.SPLIT,
        }
        symbols = [symbolmap[statemap[tracks.graph.out_degree(node)]] for node in self.nodes]
        return symbols

    def update_point_outline(self, visible_nodes: list[int] | str) -> None:
        if isinstance(visible_nodes, str):
            self.shown[:] = True
        else:
            if self.tracks_viewer.mode == "group":
                visible_nodes = list(visible_nodes) + self.tracks_viewer.selected_nodes.as_list
            indices = np.where(np.isin(self.properties["node_id"], visible_nodes))[0].tolist()
            self.shown[:] = False
            self.shown[indices] = True

        self.border_color = [1, 1, 1, 1]
        self.size = self.default_size
        for node in self.tracks_viewer.selected_nodes:
            if node not in self.node_index_dict:
                continue
            index = self.node_index_dict[node]
            self.border_color[index] = (0, 1, 1, 1)
            self.size[index] = math.ceil(self.default_size + 0.3 * self.default_size)

        self.border_color = self.border_color
        self.size = self.size
        self.refresh()
