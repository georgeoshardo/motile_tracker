from unittest.mock import patch

import numpy as np
from funtracks.data_model import SolutionTracks

from motile_tracker.data_views.views_coordinator.tracks_viewer import TracksViewer


class MockEvent:
    def __init__(self, action=None, value=None):
        self.action = action
        self.value = value


def test_kymograph_point_add_move_delete_and_undo(
    make_napari_viewer,
    graph_2d,
):
    viewer = make_napari_viewer()
    viewer.add_image(np.zeros((5, 100, 100)), name="raw")
    tracks = SolutionTracks(graph=graph_2d, ndim=3)

    tracks_viewer = TracksViewer.get_instance(viewer)
    tracks_viewer.update_tracks(tracks=tracks, name="test")
    tracks_viewer.set_kymograph_image_layer("raw")
    tracks_viewer.set_view_mode("kymograph")

    points_layer = tracks_viewer.kymograph_layers.points_layer
    initial_node_count = tracks.graph.number_of_nodes()

    points_layer._update_data(
        MockEvent(action="added", value=np.asarray([[40.0, 120.0]]))
    )

    assert tracks.graph.number_of_nodes() == initial_node_count + 1

    new_node = max(tracks.graph.nodes)
    assert tracks.get_time(new_node) == 1
    assert tracks.get_position(new_node) == [40.0, 20.0]

    new_index = points_layer.node_index_dict[new_node]
    points_layer.selected_data.add(new_index)
    points_layer.data[new_index] = np.asarray([42.0, 125.0])
    points_layer._update_data(MockEvent(action="changed"))

    assert tracks.get_position(new_node) == [42.0, 25.0]

    tracks_viewer.selected_nodes.reset()
    tracks_viewer.selected_nodes.add(new_node)
    points_layer._update_data(MockEvent(action="removed"))

    assert not tracks.graph.has_node(new_node)

    tracks_viewer.undo()
    assert tracks.graph.has_node(new_node)


def test_kymograph_point_move_cannot_cross_frame_boundary(
    make_napari_viewer,
    graph_2d,
):
    viewer = make_napari_viewer()
    viewer.add_image(np.zeros((5, 100, 100)), name="raw")
    tracks = SolutionTracks(graph=graph_2d, ndim=3)

    tracks_viewer = TracksViewer.get_instance(viewer)
    tracks_viewer.update_tracks(tracks=tracks, name="test")
    tracks_viewer.set_kymograph_image_layer("raw")
    tracks_viewer.set_view_mode("kymograph")

    points_layer = tracks_viewer.kymograph_layers.points_layer
    node = 3
    node_index = points_layer.node_index_dict[node]
    original_position = tracks.get_position(node).copy()

    points_layer.selected_data.add(node_index)
    points_layer.data[node_index] = np.asarray([60.0, 220.0])

    with patch(
        "motile_tracker.data_views.views.layers.kymograph_track_points.show_info"
    ) as info_mock:
        points_layer._update_data(MockEvent(action="changed"))

    info_mock.assert_called_once()
    assert tracks.get_time(node) == 1
    assert tracks.get_position(node) == original_position
