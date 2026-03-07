from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
from funtracks.exceptions import InvalidActionError
from funtracks.user_actions import UserUpdateSegmentation
from napari.utils import DirectLabelColormap
from napari.utils.notifications import show_info

from motile_tracker.data_views.views.kymograph_utils import (
    KymographGeometry,
    concat_time_to_kymograph,
)
from motile_tracker.data_views.views.layers.track_labels import TrackLabels
from motile_tracker.data_views.views_coordinator.user_dialogs import (
    confirm_force_operation,
)

if TYPE_CHECKING:
    from napari.utils.events import Event

    from motile_tracker.data_views.views_coordinator.tracks_viewer import TracksViewer


class KymographTrackLabels(TrackLabels):
    def __init__(
        self,
        *,
        viewer,
        name: str,
        opacity: float,
        tracks_viewer: TracksViewer,
        geometry: KymographGeometry,
        page_start: int,
        page_length: int,
    ):
        self.geometry = geometry
        self.page_start = int(page_start)
        self.page_length = int(page_length)

        super().__init__(
            viewer=viewer,
            data=np.zeros((geometry.y_size, geometry.x_size), dtype=np.int32),
            name=name,
            opacity=opacity,
            scale=(geometry.y_scale, geometry.x_scale),
            tracks_viewer=tracks_viewer,
        )
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
        self.scale = (geometry.y_scale, geometry.x_scale)
        self._refresh()

    def _next_unused_label(self) -> int:
        segmentation = np.asarray(self.tracks_viewer.tracks.segmentation)
        return int(segmentation.max()) + 1

    def _set_new_label(self, *, new_track_id: bool) -> None:
        if new_track_id or self.tracks_viewer.selected_track is None:
            self.tracks_viewer.set_new_track_id()

        self.selected_label = self._next_unused_label()
        self.colormap.color_dict[self.selected_label] = (
            self.tracks_viewer.track_id_color
        )
        with self.events.selected_label.blocker():
            self.colormap = DirectLabelColormap(color_dict=self.colormap.color_dict)

    def assign_new_label(self, event=None):
        self._set_new_label(new_track_id=True)

    def _refresh(self):
        segmentation = np.asarray(self.tracks_viewer.tracks.segmentation)
        self.data = concat_time_to_kymograph(
            segmentation,
            page_start=self.page_start,
            page_length=self.page_length,
        )
        self.colormap = self._get_colormap()
        self.refresh()

    def _ensure_valid_label_for_time(self, current_timepoint: int) -> None:
        update_colormap = False
        tracks = self.tracks_viewer.tracks

        if tracks is not None:
            if tracks.graph.has_node(self.selected_label):
                self.tracks_viewer.selected_track = tracks.get_track_id(self.selected_label)
                existing_time = tracks.get_time(self.selected_label)
                if existing_time != current_timepoint:
                    edit = False
                    if self.tracks_viewer.selected_track in tracks.track_id_to_node:
                        for node in tracks.track_id_to_node[self.tracks_viewer.selected_track]:
                            if tracks.get_time(node) == current_timepoint:
                                self.selected_label = int(node)
                                edit = True
                                break
                    if not edit:
                        self._set_new_label(new_track_id=False)
            else:
                if self.tracks_viewer.selected_track in tracks.track_id_to_node:
                    for node in tracks.track_id_to_node[self.tracks_viewer.selected_track]:
                        if tracks.get_time(node) == current_timepoint:
                            self.selected_label = int(node)
                            break
                elif self.tracks_viewer.selected_track is None:
                    self.tracks_viewer.selected_track = tracks.get_next_track_id()
                    update_colormap = True

        self.tracks_viewer.set_track_id_color(self.tracks_viewer.selected_track)
        if update_colormap:
            self.colormap.color_dict[self.selected_label] = (
                self.tracks_viewer.track_id_color
            )
            with self.events.selected_label.blocker():
                self.colormap = DirectLabelColormap(color_dict=self.colormap.color_dict)
        self.tracks_viewer.update_track_id.emit()

    def _parse_paint_event_by_time(
        self,
        event_val,
    ) -> dict[int, list[tuple[tuple[np.ndarray, ...], int]]]:
        concatenated_indices = tuple(
            np.concatenate([ev[0][dim] for ev in event_val]) for dim in range(2)
        )
        concatenated_values = np.concatenate([ev[1] for ev in event_val])

        y_indices = concatenated_indices[0].astype(int)
        page_x_indices = concatenated_indices[1].astype(int)

        frame_offsets = page_x_indices // self.geometry.x_size
        time_points = self.page_start + frame_offsets
        local_x_indices = page_x_indices - frame_offsets * self.geometry.x_size

        by_time: dict[int, list[tuple[tuple[np.ndarray, ...], int]]] = {}
        for old_value in np.unique(concatenated_values):
            mask = concatenated_values == old_value
            for timepoint in np.unique(time_points[mask]):
                time_mask = mask & (time_points == timepoint)
                pixels = (
                    np.full(np.sum(time_mask), int(timepoint), dtype=int),
                    y_indices[time_mask],
                    local_x_indices[time_mask],
                )
                by_time.setdefault(int(timepoint), []).append((pixels, int(old_value)))

        return by_time

    def _on_paint(self, event):
        if (
            self.mode == "erase"
            or (self.mode == "fill" and self.selected_label == 0)
            or (self.mode == "paint" and self.selected_label == 0)
        ):
            target_value = 0
        else:
            target_value = self.selected_label

        pixels_by_time = self._parse_paint_event_by_time(event.value)
        if len(pixels_by_time) > 1:
            show_info("Kymograph painting cannot cross frame boundaries in one stroke.")
            super().undo()
            self._refresh()
            return

        target_time = next(iter(pixels_by_time), None)
        if target_time is not None and target_value != 0:
            self._ensure_valid_label_for_time(target_time)
            target_value = self.selected_label

        with self.events.selected_label.blocker():
            try:
                for timepoint, updated_pixels in pixels_by_time.items():
                    self._ensure_valid_label_for_time(timepoint) if target_value != 0 else None
                    UserUpdateSegmentation(
                        tracks=self.tracks_viewer.tracks,
                        new_value=(self.selected_label if target_value != 0 else 0),
                        updated_pixels=updated_pixels,
                        current_track_id=self.tracks_viewer.selected_track,
                        force=self.tracks_viewer.force,
                    )
            except InvalidActionError as e:
                if e.forceable:
                    force, always_force = confirm_force_operation(message=str(e))
                    self.tracks_viewer.force = always_force
                    super().undo()
                    if not force:
                        self._refresh()
                    else:
                        for timepoint, updated_pixels in pixels_by_time.items():
                            self._ensure_valid_label_for_time(timepoint) if target_value != 0 else None
                            UserUpdateSegmentation(
                                tracks=self.tracks_viewer.tracks,
                                new_value=(self.selected_label if target_value != 0 else 0),
                                updated_pixels=updated_pixels,
                                current_track_id=self.tracks_viewer.selected_track,
                                force=True,
                            )
                else:
                    warnings.warn(str(e), stacklevel=2)
                    super().undo()
                    self._refresh()

    def _ensure_valid_label(self, event: Event | None = None):
        # The kymograph view has no viewer time slider, so validation must happen with
        # the explicit timepoint derived from the click/paint position.
        return None
