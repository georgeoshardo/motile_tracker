from __future__ import annotations

import math
import warnings
from typing import TYPE_CHECKING

import napari
import numpy as np
from funtracks.data_model import Tracks
from funtracks.exceptions import InvalidActionError
from funtracks.user_actions import UserAddNode, UserDeleteNodes, UserUpdateNodesAttrs
from napari.layers import Points
from napari.utils.notifications import show_info
from psygnal.containers import Selection

from motile_tracker.data_views.keybindings_config import KEYMAP, bind_keymap
from motile_tracker.data_views.node_type import NodeType
from motile_tracker.data_views.views.kymograph_utils import (
    KymographGeometry,
    kymograph_coords_to_indices,
    point_to_kymograph_coords,
)
from motile_tracker.data_views.views.layers.click_utils import (
    detect_click,
    detect_side_button,
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
    """Points layer showing the nodes on the current kymograph page.

    Every node is drawn at its (y, x) position, shifted along x by the offset of its
    frame within the page, so that all frames of the page appear side by side.
    Clicking a point selects the node (without re-centering the view), and adding,
    moving or deleting points edits the underlying Tracks object, like TrackPoints.
    """

    # overwrite the select function to block the current_size event signal
    _drag_modes = Points._drag_modes.copy()
    _drag_modes[Points._modeclass.SELECT] = custom_select

    @property
    def _type_string(self) -> str:
        return "points"  # to make sure that the layer is treated as points layer for saving

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

        # Key bindings (should be specified both on the viewer (in tracks_viewer)
        bind_keymap(self, KEYMAP, self.tracks_viewer)

        # Connect to click events to select nodes
        @self.mouse_drag_callbacks.append
        def click(layer, event):
            side_button = detect_side_button(event)
            if side_button is not None:
                self.process_click(event, side_button=side_button)
            elif event.type == "mouse_press" and self.mode == "pan_zoom":
                was_click = yield from detect_click(event)
                if was_click:
                    point_index = get_click_value(self, event)
                    self.process_click(event, value=point_index)

        # listen to updates of the data
        self.events.data.connect(self._update_data)
        # listen to updates in the point size
        self.events.current_size.connect(
            lambda: self.set_point_size(size=self.current_size)
        )
        # listen to updates in the selection (in select mode)
        self.selected_data.events.items_changed.connect(self._update_selection)
        self._refresh()

    @property
    def selected_data(self) -> Selection[int]:
        """Set of currently selected point indices."""

        return Points.selected_data.fget(self)

    @selected_data.setter
    def selected_data(self, selected_data) -> None:
        """Block the current_size event while changing the selection, so that the point
        size will not accumulate size increases when selecting points.
        """

        with self.events.current_size.blocker():
            Points.selected_data.fset(self, selected_data)

    def update_page(
        self,
        *,
        geometry: KymographGeometry,
        page_start: int,
        page_length: int,
    ) -> None:
        """Show a different page (or geometry) and redraw the points."""
        self.geometry = geometry
        self.page_start = int(page_start)
        self.page_length = int(page_length)
        self._refresh()

    def add(self, coords: list[float]):
        """Block the current_size event when adding points, so that the point size will
        not accumulate size increases when adding points."""
        with self.events.current_size.blocker():
            super().add(coords)

    def process_click(
        self,
        event: Event,
        value: int | None = None,
        side_button: int | None = None,
        layer: napari.layers.Points | None = None,
    ):
        """Select the clicked point(s), without re-centering the view.

        Args:
            event (Event): The mouse event
            value (int | None): The index of the clicked point, or None if no point
                was clicked
            side_button (int | None): the button index (4: back, 5: forward) if a
                mouse side button was used, or None if no side button was used.
            layer (napari.layers.Points | None): Optional, unused. Kept for signature
                compatibility with TrackPoints.
        """

        # Intercept mouse side button navigation (back/forward)
        if side_button is not None:
            self.tracks_viewer.select_node_set_from_history(previous=side_button == 4)
            return

        with self.tracks_viewer.selection_updates(set_view=False):
            if value is None:
                self.tracks_viewer.selected_nodes.reset()
            else:
                node_id = int(self.nodes[value])
                append = "Shift" in event.modifiers
                jump = "Control" in event.modifiers
                if jump:
                    self.tracks_viewer.center_on_node(node_id)
                else:
                    self.tracks_viewer.selected_nodes.add(node_id, append)

    def set_point_size(self, size: int) -> None:
        """Sets a new default point size (see TrackPoints.set_point_size)."""

        self.default_size = size
        self._refresh()

    def _page_nodes(self) -> list[int]:
        """The nodes whose time point falls on the current page."""
        tracks = self.tracks_viewer.tracks
        start = self.page_start
        stop = self.page_start + self.page_length
        node_ids = tracks.graph.node_ids()
        if not node_ids:
            return []
        times = tracks.get_times(node_ids)  # bulk lookup: per-node calls are slow
        return [
            int(node)
            for node, timepoint in zip(node_ids, times, strict=True)
            if start <= timepoint < stop
        ]

    def _refresh(self):
        """Refresh the points layer data from the tracks (nodes on the current page)."""
        self.events.data.disconnect(
            self._update_data
        )  # do not listen to new events until updates are complete

        tracks = self.tracks_viewer.tracks
        page_nodes = self._page_nodes()
        if page_nodes:
            # bulk lookups: per-node calls are slow on large graphs
            page_times = tracks.get_times(page_nodes)
            page_positions = np.asarray(tracks.get_positions(page_nodes))
            page_track_ids = tracks.get_track_ids(page_nodes)
        else:
            page_times, page_positions, page_track_ids = [], [], []

        nodes: list[int] = []
        track_ids: list[int] = []
        coords: list[tuple[float, float]] = []
        for node, timepoint, position, track_id in zip(
            page_nodes, page_times, page_positions, page_track_ids, strict=True
        ):
            coord = point_to_kymograph_coords(
                timepoint=int(timepoint),
                position=position,
                geometry=self.geometry,
                page_start=self.page_start,
                page_length=self.page_length,
            )
            if coord is None:
                continue
            nodes.append(int(node))
            track_ids.append(int(track_id))
            coords.append(coord)

        self.nodes = nodes
        self.node_index_dict = {node: idx for idx, node in enumerate(self.nodes)}

        self.data = (
            np.asarray(coords, dtype=float) if coords else np.zeros((0, 2), dtype=float)
        )
        self.symbol = self.get_symbols(tracks, self.tracks_viewer.symbolmap)
        self.face_color = self._map_track_colors(track_ids)
        self.properties = {"node_id": self.nodes, "track_id": track_ids}
        self.size = self.default_size
        self.border_color = [1, 1, 1, 1]

        self.events.data.connect(self._update_data)

    def _map_kymograph_point(self, point: np.ndarray) -> tuple[int, list[float]] | None:
        """Map a (y, x) point on the kymograph page back to (time, [y, x]) in the
        original frame, or None if the point is outside the page."""
        coords = kymograph_coords_to_indices(
            y_world=float(point[0]),
            x_global_world=float(point[1]),
            geometry=self.geometry,
            page_start=self.page_start,
            page_length=self.page_length,
        )
        if coords is None:
            return None

        timepoint, _, _ = coords
        x_local_world = float(point[1]) - (
            (timepoint - self.page_start) * self.geometry.frame_world_width
        )
        return int(timepoint), [float(point[0]), x_local_world]

    def _create_node_attrs(self, new_point: np.ndarray) -> dict | None:
        """Create attributes for a new node at the time point of the clicked frame, or
        None if the point is outside the page."""
        mapped = self._map_kymograph_point(new_point)
        if mapped is None:
            return None

        timepoint, position = mapped
        # Activate a new track_id if necessary
        if self.tracks_viewer.selected_track is None:
            self.tracks_viewer.set_new_track_id()

        features = self.tracks_viewer.tracks.features
        return {
            features.position_key: position,
            features.time_key: timepoint,
            features.tracklet_key: self.tracks_viewer.selected_track,
        }

    def _update_data(self, event: Event):
        """Calls the UserActions to update the data in the Tracks object and
        dispatch the update
        """

        if event.action == "added":
            # we only want to allow this update if there is no seg layer
            if self.tracks_viewer.tracking_layers.seg_layer is None:
                attributes = self._create_node_attrs(event.value[-1])
                if attributes is None:
                    show_info("Points can only be added inside a frame of the page.")
                    self._refresh()
                    return
                try:
                    with self.tracks_viewer.center_node.blocked():
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
                            new_node_id = self.tracks_viewer.tracks._get_new_node_ids(
                                1
                            )[0]
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
                nodes=[int(n) for n in self.tracks_viewer.selected_nodes.as_list],
            )

        elif event.action == "changed":
            # we only want to allow this update if there is no seg layer
            if self.tracks_viewer.tracking_layers.seg_layer is None:
                tracks = self.tracks_viewer.tracks
                position_key = tracks.features.position_key
                nodes: list[int] = []
                positions: list[list[float]] = []
                for ind in self.selected_data:
                    mapped = self._map_kymograph_point(self.data[ind])
                    if mapped is None:
                        show_info(
                            "Points can only be moved inside a frame of the page."
                        )
                        self._refresh()
                        return
                    timepoint, position = mapped
                    node_id = int(self.properties["node_id"][ind])
                    if timepoint != tracks.get_time(node_id):
                        show_info(
                            "Moving nodes across frame boundaries is not supported."
                        )
                        self._refresh()
                        return
                    nodes.append(node_id)
                    positions.append(position)

                if nodes:
                    UserUpdateNodesAttrs(
                        tracks,
                        nodes=nodes,
                        attrs={position_key: positions},
                    )
            else:
                self._refresh()  # refresh to move points back where they belong

    def _update_selection(self):
        """Replaces the list of selected_nodes with the selection provided by the user"""

        if self.mode != "select":
            return

        nodes = [int(self.nodes[index]) for index in sorted(self.selected_data)]
        with self.tracks_viewer.selection_updates(set_view=False):
            if nodes:
                self.tracks_viewer.selected_nodes.add_list(nodes)
            else:
                self.tracks_viewer.selected_nodes.reset()

    def _map_track_colors(self, track_ids: list[int]) -> np.ndarray:
        """Map track ids to an (N, 4) array of face colors in a single colormap call
        (see TrackPoints._map_track_colors). With no points, a single white color is
        returned, because napari raises on an empty color array."""
        if len(track_ids) == 0:
            return np.ones((1, 4))
        return self.tracks_viewer.colormap.map(np.asarray(track_ids))

    def get_symbols(self, tracks: Tracks, symbolmap: dict[NodeType, str]) -> list[str]:
        statemap = {
            0: NodeType.END,
            1: NodeType.CONTINUE,
            2: NodeType.SPLIT,
        }
        degrees = tracks.graph.out_degree(self.nodes) if self.nodes else []
        return [symbolmap[statemap[int(degree)]] for degree in degrees]

    def update_point_outline(self, visible_nodes: list[int] | str) -> None:
        """Update the outline color of the selected points and visibility according to
        display mode

        Args:
            visible_nodes (list[int] | str): A list of track ids, or "all"
        """

        # filter out the non-selected tracks if in lineage mode
        if isinstance(visible_nodes, str):
            self.shown[:] = True
        else:
            if self.tracks_viewer.mode == "group":
                visible_nodes = (
                    list(visible_nodes) + self.tracks_viewer.selected_nodes.as_list
                )
            indices = np.where(np.isin(self.properties["node_id"], visible_nodes))[
                0
            ].tolist()
            self.shown[:] = False
            self.shown[indices] = True

        n_points = len(self.data)
        if n_points == 0:
            # nothing to style, and napari warns when assigning empty color arrays
            self.refresh()
            return

        # Set border color and size for the selected items. Both are built up first and
        # then assigned once, because every assignment emits an event.
        border_colors = np.tile([1.0, 1.0, 1.0, 1.0], (n_points, 1))
        sizes = np.full(n_points, self.default_size)
        selected_size = math.ceil(self.default_size + 0.3 * self.default_size)
        for node in self.tracks_viewer.selected_nodes:
            index = self.node_index_dict.get(node, None)
            if index is not None:
                border_colors[index] = (0, 1, 1, 1)
                sizes[index] = selected_size

        self.size = sizes
        self.border_color = border_colors
        self.refresh()
