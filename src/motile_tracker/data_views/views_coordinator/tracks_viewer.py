from __future__ import annotations

from contextlib import contextmanager
from typing import Optional

import napari
import pandas as pd
from funtracks.actions import AddNode, BasicAction, DeleteNode
from funtracks.data_model import SolutionTracks
from funtracks.exceptions import InvalidActionError
from funtracks.user_actions import (
    UserAddEdge,
    UserDeleteEdge,
    UserDeleteNodes,
    UserSwapPredecessors,
)
from napari.utils.notifications import show_warning
from psygnal import Signal
from qtpy.QtWidgets import QMessageBox

from motile_tracker.data_views.keybindings_config import (
    KEYMAP,
    bind_keymap,
)
from motile_tracker.data_views.node_type import NodeType
from motile_tracker.data_views.views.kymograph_utils import clamp_page_start
from motile_tracker.data_views.views.layers.kymograph_layer_group import (
    KymographLayerGroup,
)
from motile_tracker.data_views.views.layers.track_labels import new_label
from motile_tracker.data_views.views.layers.tracks_layer_group import TracksLayerGroup
from motile_tracker.data_views.views.tree_view.tree_widget_utils import (
    extract_lineage_tree,
    extract_sorted_tracks,
)
from motile_tracker.data_views.views_coordinator.groups import (
    CollectionWidget,
)
from motile_tracker.data_views.views_coordinator.node_selection_history import (
    NodeSelectionHistory,
)
from motile_tracker.data_views.views_coordinator.tracks_list import TracksList
from motile_tracker.data_views.views_coordinator.user_dialogs import (
    confirm_force_operation,
)

BASE_TEXT = "Click: select node\nShift+Click: append to selection\nCtrl/Cmd+Click: center node\n[Q]: toggle display\nCurrent display mode: "


class TracksViewer:
    """Purposes of the TracksViewer:

    - Emit signals that all widgets should use to update selection or update
      the currently displayed Tracks object
    - Storing the currently displayed tracks
    - Store shared rendering information like colormaps (or symbol maps)
    """

    tracks_updated = Signal(Optional[bool])  # noqa: UP007 UP045
    update_track_id = Signal()
    mode_updated = Signal()
    view_mode_updated = Signal()
    kymograph_updated = Signal()
    center_node = Signal(int)  # emitted when any component wants to center on a node
    node_selection_updated = Signal(bool)

    @classmethod
    def get_instance(cls, viewer=None):
        if not hasattr(cls, "_instance") or (
            viewer is not None and cls._instance.viewer is not viewer
        ):
            if viewer is None:
                raise ValueError("Make a viewer first please!")
            # The outgoing instance is about to become unreachable, but psygnal
            # connections keep it subscribed to its tracks object. Unsubscribe it,
            # or a tracks object shown in successive viewers ends up notifying
            # every TracksViewer ever built (see _disconnect_tracks).
            if hasattr(cls, "_instance"):
                cls._instance._disconnect_tracks()
            cls._instance = TracksViewer(viewer)
        return cls._instance

    def __init__(
        self,
        viewer: napari.Viewer,
    ):
        self.viewer = viewer
        self.viewer.mouse_double_click_callbacks.clear()  # no double click to zoom
        self.menu_manager = None  # will be set by MenuManager after initialization
        self.tree_widget_present = False
        self.table_widget_present = False

        def _clear_if_current():
            self._disconnect_tracks()
            if hasattr(TracksViewer, "_instance") and TracksViewer._instance is self:
                del TracksViewer._instance

        viewer.window._qt_window.destroyed.connect(_clear_if_current)
        self.colormap = napari.utils.colormaps.label_colormap(
            49,
            seed=0.5,
            background_value=0,
        )

        self.symbolmap: dict[NodeType, str] = {
            NodeType.END: "x",
            NodeType.CONTINUE: "disc",
            NodeType.SPLIT: "triangle_up",
        }
        self.mode = "all"
        self.view_mode = "spatial"
        self.tracks: SolutionTracks | None = None
        self.visible: list | str = []
        self.tracking_layers = TracksLayerGroup(self.viewer, self.tracks, "", self)
        self.kymograph_layers = KymographLayerGroup(self.viewer, self)
        self.center_node.connect(self._center_view)
        self.selected_nodes = NodeSelectionHistory()
        self.selected_nodes.selection_updated.connect(self.update_selection)

        self.track_df = pd.DataFrame()  # initialize empty dataframe
        self.axis_order: list[int] = []

        self.tracks_list = TracksList()
        self.tracks_list.view_tracks.connect(self.update_tracks)
        self.tracks_list.request_colormap.connect(self.set_colormap_to_trackslist)
        self.selected_track = None
        self.track_id_color = [0, 0, 0, 0]
        self.force = False
        # state needed to restore the spatial view after leaving kymograph mode
        self._detached_kymograph_layers: list[
            tuple[napari.layers.Layer, int, bool]
        ] = []
        self._detached_kymograph_image_index: int | None = None
        self._detached_kymograph_image_visible = True
        self._spatial_ndisplay = int(self.viewer.dims.ndisplay)
        self._spatial_axis_labels = tuple(self.viewer.dims.axis_labels)
        self._selection_updates_set_view = True

        self.collection_widget = None

        self.set_keybinds()

        self.viewer.dims.events.ndisplay.connect(self.update_selection)

    def get_collection_widget(self) -> CollectionWidget:
        """Return a reference to the groups widget"""
        if self.collection_widget is None or getattr(
            self.collection_widget, "_is_deleted", False
        ):
            self.collection_widget = CollectionWidget(self)
            if self.tracks is not None:
                self.collection_widget.retrieve_existing_groups()

            # track destruction
            self.collection_widget.destroyed.connect(
                lambda: self._on_collection_widget_destroyed()
            )

        return self.collection_widget

    def _on_collection_widget_destroyed(self):
        self.collection_widget = None

    def set_colormap_to_trackslist(self):
        """Set the current colormap on the TracksList, so that it can be exported."""
        self.tracks_list.colormap = self.colormap

    def _update_overlay_text(self) -> None:
        """Show the current display mode and view mode in the viewer text overlay."""
        view_text = "Spatial" if self.view_mode == "spatial" else "Kymograph"
        if self.mode == "lineage":
            display_text = "Lineage"
        elif self.mode == "group":
            display_text = "Group"
        else:
            display_text = "All"

        text = BASE_TEXT + f"{display_text}\nCurrent view: {view_text}"
        if self.view_mode == "kymograph":
            text += "\nTime slider (below): pan along time, one frame per step"
        self.viewer.text_overlay.text = text
        self.viewer.text_overlay.visible = True
        self.viewer.text_overlay.font_size = 8

    def set_keybinds(self):
        bind_keymap(self.viewer, KEYMAP, self)

    def request_new_track(self) -> None:
        """Request a new track id (with new segmentation label if a seg layer is present)"""

        if (
            self.view_mode == "kymograph"
            and self.kymograph_layers.labels_layer is not None
        ):
            self.kymograph_layers.labels_layer.assign_new_label()
        elif self.tracking_layers.seg_layer is not None:
            new_label(self.tracking_layers.seg_layer)
        else:
            self.set_new_track_id()

    def set_new_track_id(self) -> None:
        """Set a new track id (if needed), update the color, and emit signal. Only updates
        the track id if the tracks.max_track_id value is used already."""

        self.selected_track = self.tracks.max_track_id  # to check if available
        if (
            self.selected_track in self.tracks.track_id_to_node
            or self.selected_track == 0
        ):
            self.selected_track = self.tracks.get_next_track_id()
        self.set_track_id_color(self.selected_track)
        self.update_track_id.emit()

    def set_track_id_color(self, track_id: int) -> None:
        """Update self.track_id color with the rgba color or given track_id, or a list of
        0 if the provided  track_id is None"""

        self.track_id_color = (
            [0, 0, 0, 0] if track_id is None else self.colormap.map(track_id)
        )

    def update_track_df(
        self, initialization: bool | None = False, refresh_view: bool | None = False
    ) -> None:
        """Create or update the pandas dataframe used by the TreeWidget and TableWidget.

        The track_df should be updated when:

        - a tree or table widget is being initialized (initialization=True) and no
          tree or table widget exists yet
        - a normal update event happens (initialization = False) AND a tree widget
          and/or table widget exists on menu_manager

        Args:
            initialization (bool | None = False): whether or not this is called by a tree
                or table widget that is initializing.
            refresh_view (bool | None = False): whether or not we should not pass on the
                previous axis_order. Should be False if we want to use the previous axis
                order (current tracks got updated). Should be True if we have a new tracks
                object and should therefore recompute the axis_order.
        """

        if self.tracks is None:
            return

        if not initialization and (
            self.tree_widget_present is False and self.table_widget_present is False
        ):
            # no need to update if there are no tracks or there is no widget that needs
            # the dataframe
            return

        if initialization and (self.tree_widget_present or self.table_widget_present):
            # no need to call for update, since we already should have it for the existing
            # table or tree widget
            return

        # in the case menu_manager was never initialized, we cannot directly check if
        # widgets exist, so we always update the track_df if self.tracks is not None.

        if refresh_view:
            self.track_df, self.axis_order = extract_sorted_tracks(
                self.tracks, self.colormap
            )
        else:
            self.track_df, self.axis_order = extract_sorted_tracks(
                self.tracks,
                self.colormap,
                self.axis_order,
            )

    def _refresh(self, node: str | None = None, refresh_view: bool = False) -> None:
        """Call refresh function on napari layers and the submit signal that tracks are
        updated. Restore the selected_nodes, if possible
        """

        if self.collection_widget is not None:
            self.collection_widget._refresh()

        if len(self.selected_nodes) > 0 and any(
            not self.tracks.graph.has_node(node) for node in self.selected_nodes
        ):
            self.selected_nodes.reset()

        self.tracking_layers._refresh()
        self.kymograph_layers._refresh()

        self.update_track_df(initialization=False, refresh_view=refresh_view)

        self.tracks_updated.emit(refresh_view)

        # if a new node was added, we would like to select this one now (call this after
        # emitting the signal, because if the node is a new node, we have to update the
        # table in the tree widget first, or it won't be present)
        if node is not None:
            self.selected_nodes.add(node)

        # restore selection and/or highlighting in all napari Views (napari Views do not
        # know about their selection ('all' vs 'lineage'), but TracksViewer does)
        self.update_selection(update_counts=True)

    def _disconnect_tracks(self) -> None:
        """Stop listening to the currently displayed tracks object.

        The connections below live on the Tracks object, not on this TracksViewer,
        so they outlive both the viewer and the singleton reference unless they are
        explicitly removed. Because one Tracks object can be handed to more than one
        viewer over a session, leaving them in place means an edit notifies every
        TracksViewer that ever displayed those tracks.
        """
        tracks = getattr(self, "tracks", None)
        if tracks is not None:
            tracks.refresh.disconnect(self._refresh)
            tracks.action_applied.disconnect(self._on_action_applied)

    def update_tracks(self, tracks: SolutionTracks, name: str) -> None:
        """Stop viewing a previous set of tracks and replace it with a new one.
        Will create new segmentation and tracks layers and add them to the viewer.

        Args:
            tracks (funtracks.data_model.Tracks): The tracks to visualize in napari.
            name (str): The name of the tracks to display in the layer names
        """
        self.selected_nodes.reset()

        self._disconnect_tracks()

        self.tracks = tracks
        self.selected_nodes.deleted_items.clear()  # Reset deleted nodes when switching tracks

        # listen to refresh signals from the tracks
        self.tracks.refresh.connect(self._refresh)
        # connect to action_applied signal to track deleted nodes
        self.tracks.action_applied.connect(self._on_action_applied)

        # deactivate the input labels layer (but never our own kymograph view layers)
        kymograph_owned_layers = {
            self.kymograph_layers.labels_layer,
            self.kymograph_layers.points_layer,
        }
        for layer in self.viewer.layers:
            if (
                isinstance(layer, (napari.layers.Labels | napari.layers.Points))
                and layer not in kymograph_owned_layers
            ):
                layer.visible = False

        # retrieve existing groups
        if self.collection_widget is not None:
            self.collection_widget.retrieve_existing_groups()

        self.set_display_mode("all")
        self.tracking_layers.set_tracks(
            tracks,
            name,
            add_to_viewer=self.view_mode == "spatial",
        )
        self.kymograph_layers.set_tracks(tracks, name)
        self.selected_nodes.reset()

        # ensure a valid track is selected from the start
        self.request_new_track()

        if self.view_mode == "kymograph":
            ok, message = self.kymograph_layers.activate()
            if not ok:
                # the new tracks cannot be shown as a kymograph: fall back to spatial
                show_warning(message)
                self._leave_kymograph_mode()
                self.view_mode_updated.emit()
                self.kymograph_updated.emit()
            else:
                self.viewer.dims.ndisplay = 2
            self._update_overlay_text()

        self.update_track_df(initialization=False, refresh_view=True)

        # emit the update signal
        self.tracks_updated.emit(True)

        # Update visualization widget
        self.mode_updated.emit()

    def toggle_display_mode(self, event=None) -> None:
        """Toggle the display mode between available options.

        Skips 'group' mode when no groups exist, alternating only between
        'all' and 'lineage' in that case.
        """

        has_groups = (
            self.collection_widget is not None
            and self.collection_widget.collection_list.count() > 0
        )

        if self.mode == "lineage":
            self.set_display_mode("group" if has_groups else "all")
        elif self.mode == "group":
            self.set_display_mode("all")
        else:
            self.set_display_mode("lineage")

    def set_display_mode(self, mode: str) -> None:
        """Update the display mode and call to update colormaps for points, labels, and tracks"""

        if mode == "lineage":
            self.mode = "lineage"
        elif mode == "group":
            self.mode = "group"
        else:
            self.mode = "all"

        self._update_overlay_text()
        self.filter_visible_nodes()
        self._active_layer_group().update_visible(self.visible)
        self.mode_updated.emit()

    def _active_layer_group(self) -> TracksLayerGroup | KymographLayerGroup:
        """Return the layer group that is currently shown in the viewer."""
        if self.view_mode == "kymograph":
            return self.kymograph_layers
        return self.tracking_layers

    # ------------------------------------------------------------------
    # Kymograph view mode
    # ------------------------------------------------------------------
    def _layers_to_detach_for_kymograph(
        self,
        selected_image: napari.layers.Image | None,
    ) -> list[tuple[napari.layers.Layer, int, bool]]:
        """List the (layer, index, visible) of the time-series layers that must be
        removed from the viewer while in kymograph mode: every layer with more than
        two dimensions that is not one of our own tracking layers or the image that
        is shown behind the kymograph."""
        spatial_layers = {
            layer
            for layer in (
                self.tracking_layers.tracks_layer,
                self.tracking_layers.points_layer,
                self.tracking_layers.seg_layer,
            )
            if layer is not None
        }
        detached_layers: list[tuple[napari.layers.Layer, int, bool]] = []
        for index, layer in enumerate(list(self.viewer.layers)):
            if layer in spatial_layers or layer is selected_image:
                continue
            if getattr(layer, "ndim", 2) > 2:
                detached_layers.append((layer, index, bool(layer.visible)))
        return detached_layers

    def _detach_layers_for_kymograph(self) -> None:
        """Remove the time-series layers (and the selected kymograph image layer) from
        the viewer, remembering where they were so they can be restored later."""
        selected_image = self.kymograph_layers._resolved_image_layer()
        selected_image_index = (
            self.viewer.layers.index(selected_image)
            if selected_image is not None and selected_image in self.viewer.layers
            else None
        )
        self._detached_kymograph_layers = self._layers_to_detach_for_kymograph(
            selected_image
        )
        for layer, _, _visible in reversed(self._detached_kymograph_layers):
            if layer in self.viewer.layers:
                self.viewer.layers.remove(layer)

        if selected_image is None:
            self.kymograph_layers.set_detached_image_layer(None)
            self._detached_kymograph_image_index = None
            self._detached_kymograph_image_visible = True
            return

        self._detached_kymograph_image_index = selected_image_index
        self._detached_kymograph_image_visible = bool(selected_image.visible)
        self.viewer.layers.remove(selected_image)
        selected_image.visible = self._detached_kymograph_image_visible
        self.kymograph_layers.set_detached_image_layer(selected_image)

    def _restore_detached_kymograph_layers(self) -> None:
        for layer, index, visible in self._detached_kymograph_layers:
            if layer not in self.viewer.layers:
                self.viewer.add_layer(layer)
                if index < len(self.viewer.layers) - 1:
                    self.viewer.layers.move(len(self.viewer.layers) - 1, index)
            layer.visible = visible
        self._detached_kymograph_layers = []

    def _restore_detached_kymograph_image(self) -> None:
        layer = self.kymograph_layers.detached_image_layer
        if layer is None:
            return

        if layer not in self.viewer.layers:
            self.viewer.add_layer(layer)
            if (
                self._detached_kymograph_image_index is not None
                and self._detached_kymograph_image_index < len(self.viewer.layers) - 1
            ):
                self.viewer.layers.move(
                    len(self.viewer.layers) - 1,
                    self._detached_kymograph_image_index,
                )
        layer.visible = self._detached_kymograph_image_visible
        self.kymograph_layers.set_detached_image_layer(None)
        self._detached_kymograph_image_index = None
        self._detached_kymograph_image_visible = True

    def _leave_kymograph_mode(self) -> None:
        """Tear down the kymograph layers and restore the spatial layers and dims."""
        self.kymograph_layers.deactivate()
        self._restore_detached_kymograph_image()
        self._restore_detached_kymograph_layers()
        self.tracking_layers.add_napari_layers()
        self.viewer.dims.ndisplay = self._spatial_ndisplay
        if len(self._spatial_axis_labels) == self.viewer.dims.ndim:
            self.viewer.dims.axis_labels = self._spatial_axis_labels
        self.view_mode = "spatial"

    def set_view_mode(self, mode: str) -> bool:
        """Switch between the 'spatial' view (the regular napari layers) and the
        'kymograph' view (all frames of the current page side by side).

        Returns:
            bool: True if the viewer is now in the requested mode, False if the
                kymograph could not be shown (a warning is displayed in that case).
        """
        if mode == self.view_mode:
            self.view_mode_updated.emit()
            return True

        if mode == "kymograph":
            self._spatial_ndisplay = int(self.viewer.dims.ndisplay)
            self._spatial_axis_labels = tuple(self.viewer.dims.axis_labels)
            self.tracking_layers.remove_napari_layers()
            self._detach_layers_for_kymograph()
            ok, message = self.kymograph_layers.activate()
            if not ok:
                self._restore_detached_kymograph_layers()
                self._restore_detached_kymograph_image()
                self.tracking_layers.add_napari_layers()
                show_warning(message)
                return False
            self.viewer.dims.ndisplay = 2
            self.view_mode = "kymograph"
        else:
            self._leave_kymograph_mode()

        self._update_overlay_text()
        self.update_selection(set_view=False)
        self.view_mode_updated.emit()
        self.kymograph_updated.emit()
        return True

    def set_kymograph_image_layer(self, layer_name: str | None) -> None:
        """Select the image layer to show behind the kymograph (None for no image)."""
        self.kymograph_layers.set_image_layer_name(layer_name)
        self.kymograph_updated.emit()

    def set_kymograph_page_length(self, page_length: int) -> None:
        """Set the number of frames shown side by side on one kymograph page."""
        self.kymograph_layers.set_page(page_length=page_length)
        self.kymograph_updated.emit()

    def set_kymograph_page_start(self, page_start: int) -> None:
        """Set the first frame of the kymograph page."""
        self.kymograph_layers.set_page(page_start=page_start)
        self.kymograph_updated.emit()

    def step_kymograph_page(self, delta: int) -> None:
        """Move `delta` pages forward (positive) or backward (negative)."""
        self.kymograph_layers.set_page(
            page_start=self.kymograph_layers.page_start
            + delta * self.kymograph_layers.page_length
        )
        self.kymograph_updated.emit()

    def set_show_kymograph_boundaries(self, show_boundaries: bool) -> None:
        self.kymograph_layers.set_show_boundaries(show_boundaries)
        self.kymograph_updated.emit()

    def set_show_kymograph_paths(self, show_paths: bool) -> None:
        self.kymograph_layers.set_show_paths(show_paths)
        self.kymograph_updated.emit()

    def set_show_kymograph_branches(self, show_branches: bool) -> None:
        self.kymograph_layers.set_show_branches(show_branches)
        self.kymograph_updated.emit()

    def _center_view(self, node: int) -> None:
        """Center the active view on the node. In kymograph mode, first jump to the
        page that contains the node's time point."""
        if self.view_mode == "kymograph":
            if (
                self.kymograph_layers.geometry is not None
                and not self.kymograph_layers.node_on_page(node)
            ):
                node_time = self.tracks.get_time(node)
                page_start = clamp_page_start(
                    node_time - self.kymograph_layers.page_length // 2,
                    self.kymograph_layers.geometry,
                    self.kymograph_layers.page_length,
                )
                self.kymograph_layers.set_page(page_start=page_start)
                self.kymograph_updated.emit()
            self.kymograph_layers.center_view(node)
        else:
            self.tracking_layers.center_view(node)

    def filter_visible_nodes(self) -> list[int] | str:
        """Construct a list of node_ids that should be displayed according to the display
        mode: 'all', 'lineage', or 'group'). Note that whether a node is truly
        displayed also depends on whether it is in the current selection (not computed
        here). Additionally, if the mode is 'lineage' and the selection is cleared we
        keep the previous list of nodes visible to not have an entirely empty viewer.
        """

        if self.tracks is None or self.tracks.graph is None:
            self.visible = []
        if self.mode == "lineage":
            # if no nodes are selected, check which nodes were previously visible and
            # filter those
            if len(self.selected_nodes) == 0 and self.visible is not None:
                prev_visible = [
                    node for node in self.visible if self.tracks.graph.has_node(node)
                ]
                self.visible = []
                for node_id in prev_visible:
                    self.visible += extract_lineage_tree(self.tracks.graph, node_id)
                    if set(prev_visible).issubset(self.visible):
                        break
            else:
                self.visible = []
                for node in self.selected_nodes:
                    self.visible += extract_lineage_tree(self.tracks.graph, node)
        elif self.mode == "group":
            if (
                self.collection_widget is not None
                and self.collection_widget.selected_collection is not None
            ):
                self.visible = list(
                    self.collection_widget.selected_collection.collection
                )
            else:
                self.visible = []
        else:
            self.visible = "all"

    def center_on_node(self, node: int) -> None:
        """Request all views to center on the given node.

        Emits the center_node signal which is listened to by the tracking layers
        and tree view to synchronize centering.

        Args:
            node: The node ID to center on.
        """
        self.center_node.emit(node)

    def _on_action_applied(self, action: BasicAction) -> None:
        """Handle action_applied signal from tracks.

        Updates the deleted_items set to track which nodes have been deleted,
        and clears nodes that were added back (via undo or re-addition).

        Args:
            action: The action that was applied (from funtracks)
        """

        if isinstance(action, DeleteNode):
            self.selected_nodes.deleted_items.add(action.node)
        elif isinstance(action, AddNode):
            self.selected_nodes.deleted_items.discard(action.node)

    @contextmanager
    def selection_updates(self, *, set_view: bool):
        """Control whether selection updates triggered inside the block re-center the
        view. The kymograph layers select nodes without moving the camera."""
        previous = self._selection_updates_set_view
        self._selection_updates_set_view = bool(set_view)
        try:
            yield
        finally:
            self._selection_updates_set_view = previous

    def update_selection(
        self, set_view: bool | None = None, update_counts: bool = False
    ) -> None:
        """Sets the view and triggers visualization updates in other components"""

        if set_view is None:
            set_view = self._selection_updates_set_view

        if set_view and len(self.selected_nodes) == 1:
            self.center_on_node(self.selected_nodes[0])

        self.filter_visible_nodes()
        self._active_layer_group().update_visible(self.visible)

        if len(self.selected_nodes) > 0:
            self.selected_track = self.tracks.get_track_id(self.selected_nodes[-1])
        else:
            self.selected_track = None

        self.set_track_id_color(self.selected_track)
        self.update_track_id.emit()
        self.node_selection_updated.emit(update_counts)

    def delete_node(self, event=None):
        """Calls the UserAction to delete currently selected nodes"""

        if self.tracks is None:
            return
        UserDeleteNodes(
            self.tracks, nodes=[int(n) for n in self.selected_nodes.as_list]
        )

    def delete_edge(self, event=None):
        """Calls the UserAction to delete an edge between the two currently
        selected nodes
        """

        if self.tracks is None:
            return
        if len(self.selected_nodes) == 2:
            node1 = self.selected_nodes[0]
            node2 = self.selected_nodes[1]

            time1 = self.tracks.get_time(node1)
            time2 = self.tracks.get_time(node2)

            if time1 > time2:
                node1, node2 = node2, node1

            node1, node2 = int(node1), int(node2)
            UserDeleteEdge(self.tracks, (node1, node2))

    def swap_nodes(self, event=None):
        """Calls the UserAction to swap the predecessors of the two currently
        selected nodes
        """

        if len(self.selected_nodes) == 2:
            node1 = self.selected_nodes[0]
            node2 = self.selected_nodes[1]

            UserSwapPredecessors(self.tracks, nodes=(int(node1), int(node2)))

    def create_edge(self, event=None):
        """Add an edge between the two currently selected nodes"""

        if self.tracks is None:
            return
        if len(self.selected_nodes) == 2:
            node1 = self.selected_nodes[0]
            node2 = self.selected_nodes[1]

            time1 = self.tracks.get_time(node1)
            time2 = self.tracks.get_time(node2)

            if time1 > time2:
                node1, node2 = node2, node1

            node1, node2 = int(node1), int(node2)

            if self.tracks.graph.out_degree(node1) >= 2:
                QMessageBox.warning(
                    None,
                    "Cannot add edge",
                    f"Node {node1} already has 2 children. Delete one of the "
                    "existing daughter edges before adding a new one.",
                )
                return

            try:
                UserAddEdge(self.tracks, (node1, node2), force=self.force)
            except InvalidActionError as e:
                if e.forceable:
                    # Ask the user if the action should be forced
                    force, always_force = confirm_force_operation(message=str(e))
                    self.force = always_force
                    if force:
                        UserAddEdge(self.tracks, (node1, node2), force=True)
                else:
                    # Re-raise the exception if it is not forceable
                    raise

    def undo(self, event=None):
        if self.tracks is None:
            return
        self.tracks.undo()

    def redo(self, event=None):
        if self.tracks is None:
            return
        self.tracks.redo()

    def hide_panels(self, event=None):
        """Show/hide menu and tree view panels without destroying"""

        if self.menu_manager is not None:
            self.menu_manager.toggle_menu_panel_visibility()

    def deselect(self, event=None):
        self.selected_nodes.reset()

    def restore_selection(self, event=None):
        self.selected_nodes.restore()

    def select_node_set_from_history(self, previous: bool):
        """Move forwards or backwards through selection history."""
        self.selected_nodes.select_node_set_from_history(previous=previous)

    def select_next(self, event=None):
        """Select next node set from history"""
        self.select_node_set_from_history(previous=False)

    def select_previous(self, event=None):
        """Select previous node set from history"""
        self.select_node_set_from_history(previous=True)
