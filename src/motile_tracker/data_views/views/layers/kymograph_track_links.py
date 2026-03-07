from __future__ import annotations

from typing import TYPE_CHECKING

import napari
import numpy as np
from napari.layers import Shapes

from motile_tracker.data_views.views.layers.click_utils import (
    detect_click,
    get_click_value,
)

if TYPE_CHECKING:
    from napari.utils.events import Event

    from motile_tracker.data_views.views.kymograph_utils import KymographLinkRenderData
    from motile_tracker.data_views.views_coordinator.tracks_viewer import TracksViewer


class KymographTrackLinks(Shapes):
    @property
    def _type_string(self) -> str:
        return "shapes"

    def __init__(
        self,
        *,
        name: str,
        tracks_viewer: TracksViewer,
        render_data: KymographLinkRenderData,
        edge_width: float,
    ):
        self.tracks_viewer = tracks_viewer
        self.default_edge_width = float(edge_width)

        super().__init__(
            data=render_data.segments,
            shape_type="line",
            name=name,
            edge_color=render_data.edge_colors or "white",
            edge_width=self.default_edge_width,
            face_color="transparent",
            opacity=0.9,
            properties=render_data.properties,
            blending="translucent_no_depth",
        )
        self.editable = False

        @self.mouse_drag_callbacks.append
        def click(layer, event):
            if (
                event.type == "mouse_press"
                and self.mode == "pan_zoom"
                and self.tracks_viewer.viewer.layers.selection.active is self
            ):
                was_click = yield from detect_click(event)
                if was_click:
                    value = get_click_value(self, event)
                    self.process_click(event, value)

        self.selected_data.events.items_changed.connect(self._update_selection)

    def update_render_data(self, render_data: KymographLinkRenderData) -> None:
        self.data = render_data.segments
        self.properties = render_data.properties
        self.edge_color = render_data.edge_colors or "white"
        self.edge_width = self.default_edge_width
        self.refresh()

    def process_click(
        self,
        event: Event,
        value: tuple[int | None, int | None] | None,
    ) -> None:
        shape_index = None if value is None else value[0]
        with self.tracks_viewer.selection_updates(set_view=False):
            if shape_index is None:
                self.tracks_viewer.selected_nodes.reset()
                return

            source_node = int(self.properties["source_node"][shape_index])
            target_node = int(self.properties["target_node"][shape_index])
            append = "Shift" in event.modifiers
            self.tracks_viewer.selected_nodes.add_list(
                [source_node, target_node],
                append=append,
            )

    def _update_selection(self) -> None:
        if self.mode != "select":
            return

        selected_nodes: list[int] = []
        selected_indices = sorted(int(index) for index in self.selected_data)
        for index in selected_indices:
            selected_nodes.extend(
                [
                    int(self.properties["source_node"][index]),
                    int(self.properties["target_node"][index]),
                ]
            )

        deduped_nodes = list(dict.fromkeys(selected_nodes))
        with self.tracks_viewer.selection_updates(set_view=False):
            if deduped_nodes:
                self.tracks_viewer.selected_nodes.add_list(deduped_nodes)
            else:
                self.tracks_viewer.selected_nodes.reset()
