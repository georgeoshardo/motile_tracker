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
    def __init__(self, action=None, value=None, modifiers=None):
        self.action = action
        self.value = value
        self.modifiers = modifiers or []


@pytest.fixture
def kymograph_tracks_viewer(viewer, solution_tracks_2d_without_segmentation):
    """A TracksViewer showing point-only tracks as a kymograph over a raw image."""
    viewer.add_image(np.zeros((5, 100, 100)), name="raw")

    tracks_viewer = TracksViewer.get_instance(viewer)
    tracks_viewer.update_tracks(
        tracks=solution_tracks_2d_without_segmentation, name="test"
    )
    tracks_viewer.set_kymograph_image_layer("raw")
    assert tracks_viewer.set_view_mode("kymograph") is True
    return tracks_viewer


def test_kymograph_points_are_offset_by_frame(kymograph_tracks_viewer):
    tracks_viewer = kymograph_tracks_viewer
    tracks = tracks_viewer.tracks
    points_layer = tracks_viewer.kymograph_layers.points_layer

    assert set(points_layer.nodes) == set(tracks.graph.node_ids())
    # node 3 is at t=1, position (60, 45): drawn one frame width (100) to the right
    index = points_layer.node_index_dict[3]
    np.testing.assert_allclose(points_layer.data[index], [60.0, 145.0])
    assert points_layer.properties["track_id"][index] == tracks.get_track_id(3)


def test_kymograph_point_add_move_delete_and_undo(kymograph_tracks_viewer):
    tracks_viewer = kymograph_tracks_viewer
    tracks = tracks_viewer.tracks
    points_layer = tracks_viewer.kymograph_layers.points_layer
    initial_node_count = tracks.graph.num_nodes()

    # a point in the second frame of the page (x in [100, 200)) is a node at t=1
    points_layer._update_data(
        MockEvent(action="added", value=np.asarray([[40.0, 120.0]]))
    )

    assert tracks.graph.num_nodes() == initial_node_count + 1

    new_node = int(max(tracks.graph.node_ids()))
    assert tracks.get_time(new_node) == 1
    np.testing.assert_allclose(tracks.get_position(new_node), [40.0, 20.0])

    new_index = points_layer.node_index_dict[new_node]
    points_layer.selected_data.add(new_index)
    points_layer.data[new_index] = np.asarray([42.0, 125.0])
    points_layer._update_data(MockEvent(action="changed"))

    np.testing.assert_allclose(tracks.get_position(new_node), [42.0, 25.0])

    tracks_viewer.selected_nodes.reset()
    tracks_viewer.selected_nodes.add(new_node)
    points_layer._update_data(MockEvent(action="removed"))

    assert not tracks.graph.has_node(new_node)

    tracks_viewer.undo()
    assert tracks.graph.has_node(new_node)


def test_kymograph_point_move_cannot_cross_frame_boundary(kymograph_tracks_viewer):
    tracks_viewer = kymograph_tracks_viewer
    tracks = tracks_viewer.tracks
    points_layer = tracks_viewer.kymograph_layers.points_layer
    node = 3
    node_index = points_layer.node_index_dict[node]
    original_position = np.array(tracks.get_position(node), copy=True)

    points_layer.selected_data.add(node_index)
    points_layer.data[node_index] = np.asarray([60.0, 220.0])  # frame 2

    with patch(
        "motile_tracker.data_views.views.layers.kymograph_track_points.show_info"
    ) as info_mock:
        points_layer._update_data(MockEvent(action="changed"))

    info_mock.assert_called_once()
    assert tracks.get_time(node) == 1
    np.testing.assert_array_equal(tracks.get_position(node), original_position)


def test_kymograph_point_click_selects_without_centering_view(
    kymograph_tracks_viewer,
):
    tracks_viewer = kymograph_tracks_viewer
    points_layer = tracks_viewer.kymograph_layers.points_layer

    with patch.object(tracks_viewer, "center_on_node") as center_mock:
        points_layer.process_click(MockEvent(), 0)

    center_mock.assert_not_called()
    assert list(tracks_viewer.selected_nodes) == [points_layer.nodes[0]]

    points_layer.process_click(MockEvent(modifiers=["Shift"]), 1)
    assert set(tracks_viewer.selected_nodes) == set(points_layer.nodes[:2])

    points_layer.process_click(MockEvent(), None)
    assert len(tracks_viewer.selected_nodes) == 0


def test_kymograph_points_outline_selected_nodes(kymograph_tracks_viewer):
    tracks_viewer = kymograph_tracks_viewer
    points_layer = tracks_viewer.kymograph_layers.points_layer

    tracks_viewer.selected_nodes.add(3)

    index = points_layer.node_index_dict[3]
    np.testing.assert_allclose(points_layer.border_color[index], [0, 1, 1, 1])
    assert points_layer.size[index] > points_layer.default_size
    other = points_layer.node_index_dict[1]
    np.testing.assert_allclose(points_layer.border_color[other], [1, 1, 1, 1])
