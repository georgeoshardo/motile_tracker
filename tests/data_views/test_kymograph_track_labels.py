from unittest.mock import patch

import numpy as np
from funtracks.data_model import SolutionTracks

from motile_tracker.data_views.views_coordinator.tracks_viewer import TracksViewer


class MockEvent:
    def __init__(self, value=None, modifiers=None):
        self.value = value
        self.modifiers = modifiers or []


def create_kymograph_event_val(
    *,
    y: tuple[int, int],
    x: tuple[int, int],
    old_val: int,
    target_val: int,
):
    y_idx = np.arange(y[0], y[1])
    x_idx = np.arange(x[0], x[1])
    yy, xx = np.meshgrid(y_idx, x_idx, indexing="ij")

    indices = (
        yy.ravel(),
        xx.ravel(),
    )
    old_vals = np.full(indices[0].shape, old_val, dtype=np.uint16)
    return [(indices, old_vals, target_val)]


def test_kymograph_paint_and_erase_event(
    make_napari_viewer,
    graph_2d,
    segmentation_2d,
):
    viewer = make_napari_viewer()
    tracks = SolutionTracks(graph=graph_2d, segmentation=segmentation_2d, ndim=3)

    tracks_viewer = TracksViewer.get_instance(viewer)
    tracks_viewer.update_tracks(tracks=tracks, name="test")
    tracks_viewer.set_view_mode("kymograph")
    tracks_viewer.set_kymograph_page_length(1)
    tracks_viewer.set_kymograph_page_start(3)
    tracks_viewer.request_new_track()

    labels_layer = tracks_viewer.kymograph_layers.labels_layer
    new_label = int(labels_layer.selected_label)

    event = MockEvent(
        create_kymograph_event_val(
            y=(10, 12),
            x=(20, 22),
            old_val=0,
            target_val=99,
        )
    )
    labels_layer.mode = "paint"
    labels_layer._on_paint(event)

    assert tracks.segmentation[3, 10, 20] == new_label
    assert tracks.graph.has_node(new_label)

    tracks_viewer.set_kymograph_page_start(4)
    erase_event = MockEvent(
        create_kymograph_event_val(
            y=(0, 2),
            x=(0, 2),
            old_val=5,
            target_val=0,
        )
    )
    labels_layer.mode = "erase"
    labels_layer._on_paint(erase_event)

    assert tracks.segmentation[4, 0, 0] == 0

    tracks_viewer.undo()
    assert tracks.segmentation[4, 0, 0] == 5


def test_kymograph_paint_rejects_cross_frame_stroke(
    make_napari_viewer,
    graph_2d,
    segmentation_2d,
):
    viewer = make_napari_viewer()
    tracks = SolutionTracks(graph=graph_2d, segmentation=segmentation_2d, ndim=3)

    tracks_viewer = TracksViewer.get_instance(viewer)
    tracks_viewer.update_tracks(tracks=tracks, name="test")
    tracks_viewer.set_view_mode("kymograph")
    tracks_viewer.set_kymograph_page_length(2)
    tracks_viewer.set_kymograph_page_start(3)
    tracks_viewer.request_new_track()

    labels_layer = tracks_viewer.kymograph_layers.labels_layer
    labels_layer.mode = "paint"

    event = MockEvent(
        [
            (
                (
                    np.asarray([10, 10]),
                    np.asarray([99, 100]),
                ),
                np.asarray([0, 0], dtype=np.uint16),
                77,
            )
        ]
    )

    with patch(
        "motile_tracker.data_views.views.layers.kymograph_track_labels.show_info"
    ) as info_mock:
        labels_layer._on_paint(event)

    info_mock.assert_called_once()
    assert tracks.segmentation[3, 10, 99] == 0
    assert tracks.segmentation[4, 10, 0] == 0


def test_kymograph_label_click_selects_without_centering_view(
    make_napari_viewer,
    graph_2d,
    segmentation_2d,
):
    viewer = make_napari_viewer()
    tracks = SolutionTracks(graph=graph_2d, segmentation=segmentation_2d, ndim=3)

    tracks_viewer = TracksViewer.get_instance(viewer)
    tracks_viewer.update_tracks(tracks=tracks, name="test")
    tracks_viewer.set_view_mode("kymograph")

    labels_layer = tracks_viewer.kymograph_layers.labels_layer

    with patch.object(tracks_viewer, "center_on_node") as center_mock:
        labels_layer.process_click(MockEvent(), 1)

    center_mock.assert_not_called()
    assert list(tracks_viewer.selected_nodes) == [1]
