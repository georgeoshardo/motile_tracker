from unittest.mock import patch

import numpy as np
import pytest

from motile_tracker.data_views.views_coordinator.tracks_viewer import TracksViewer


@pytest.fixture(autouse=True)
def clear_viewer_layers(viewer):
    """Clear viewer layers between tests."""
    yield
    viewer.layers.clear()


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
    """Create a napari paint event value covering the y and x index ranges on the
    (2D) kymograph page."""
    y_idx = np.arange(y[0], y[1])
    x_idx = np.arange(x[0], x[1])
    yy, xx = np.meshgrid(y_idx, x_idx, indexing="ij")

    indices = (
        yy.ravel(),
        xx.ravel(),
    )
    old_vals = np.full(indices[0].shape, old_val, dtype=np.uint16)
    return [(indices, old_vals, target_val)]


@pytest.fixture
def kymograph_tracks_viewer(viewer, solution_tracks_2d):
    tracks_viewer = TracksViewer.get_instance(viewer)
    tracks_viewer.update_tracks(tracks=solution_tracks_2d, name="test")
    assert tracks_viewer.set_view_mode("kymograph") is True
    return tracks_viewer


def test_kymograph_labels_show_the_current_page(kymograph_tracks_viewer):
    tracks_viewer = kymograph_tracks_viewer
    tracks = tracks_viewer.tracks
    segmentation = np.asarray(tracks.segmentation)
    labels_layer = tracks_viewer.kymograph_layers.labels_layer

    # by default all 5 frames fit on one page
    assert labels_layer.data.shape == (100, 500)
    np.testing.assert_array_equal(labels_layer.data[:, 100:200], segmentation[1])

    tracks_viewer.set_kymograph_page_length(2)
    tracks_viewer.set_kymograph_page_start(3)

    assert labels_layer.data.shape == (100, 200)
    np.testing.assert_array_equal(labels_layer.data[:, 0:100], segmentation[3])
    np.testing.assert_array_equal(labels_layer.data[:, 100:200], segmentation[4])


def test_kymograph_paint_and_erase_event(kymograph_tracks_viewer):
    tracks_viewer = kymograph_tracks_viewer
    tracks = tracks_viewer.tracks
    tracks_viewer.set_kymograph_page_length(1)
    tracks_viewer.set_kymograph_page_start(3)
    tracks_viewer.request_new_track()

    labels_layer = tracks_viewer.kymograph_layers.labels_layer
    new_label = int(labels_layer.selected_label)
    assert not tracks.graph.has_node(new_label)

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

    # the stroke on page frame 0 (= time point 3) created a node at t=3
    assert np.asarray(tracks.segmentation)[3, 10, 20] == new_label
    assert tracks.graph.has_node(new_label)
    assert tracks.get_time(new_label) == 3

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

    assert np.asarray(tracks.segmentation)[4, 0, 0] == 0

    tracks_viewer.undo()
    assert np.asarray(tracks.segmentation)[4, 0, 0] == 5


def test_kymograph_paint_rejects_cross_frame_stroke(kymograph_tracks_viewer):
    tracks_viewer = kymograph_tracks_viewer
    tracks = tracks_viewer.tracks
    tracks_viewer.set_kymograph_page_length(2)
    tracks_viewer.set_kymograph_page_start(3)
    tracks_viewer.request_new_track()

    labels_layer = tracks_viewer.kymograph_layers.labels_layer
    labels_layer.mode = "paint"

    # x index 99 is the last column of frame 3, x index 100 is the first of frame 4
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
    segmentation = np.asarray(tracks.segmentation)
    assert segmentation[3, 10, 99] == 0
    assert segmentation[4, 10, 0] == 0


def test_kymograph_label_click_selects_without_centering_view(
    kymograph_tracks_viewer,
):
    tracks_viewer = kymograph_tracks_viewer
    labels_layer = tracks_viewer.kymograph_layers.labels_layer

    with patch.object(tracks_viewer, "center_on_node") as center_mock:
        labels_layer.process_click(MockEvent(), np.int64(1))

    center_mock.assert_not_called()
    assert list(tracks_viewer.selected_nodes) == [1]

    labels_layer.process_click(MockEvent(modifiers=["Shift"]), 3)
    assert set(tracks_viewer.selected_nodes) == {1, 3}
