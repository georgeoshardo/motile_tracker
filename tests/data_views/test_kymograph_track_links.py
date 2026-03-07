from unittest.mock import patch

import numpy as np
from funtracks.data_model import SolutionTracks

from motile_tracker.data_views.views_coordinator.tracks_viewer import TracksViewer


class MockEvent:
    def __init__(self, modifiers=None):
        self.modifiers = modifiers or []


def test_kymograph_link_click_selects_endpoints_without_centering_view(
    make_napari_viewer,
    graph_2d,
    segmentation_2d,
):
    viewer = make_napari_viewer()
    viewer.add_image(np.asarray(segmentation_2d, dtype=float), name="raw")
    tracks = SolutionTracks(graph=graph_2d, segmentation=segmentation_2d, ndim=3)

    tracks_viewer = TracksViewer.get_instance(viewer)
    tracks_viewer.update_tracks(tracks=tracks, name="test")
    tracks_viewer.set_kymograph_image_layer("raw")
    tracks_viewer.set_view_mode("kymograph")

    links_layer = tracks_viewer.kymograph_layers.continuation_links_layer

    with patch.object(tracks_viewer, "center_on_node") as center_mock:
        links_layer.process_click(MockEvent(), (0, None))

    center_mock.assert_not_called()
    assert set(tracks_viewer.selected_nodes) == {3, 4}


def test_kymograph_branch_link_click_selects_branch_endpoints(
    make_napari_viewer,
    graph_2d,
    segmentation_2d,
):
    viewer = make_napari_viewer()
    viewer.add_image(np.asarray(segmentation_2d, dtype=float), name="raw")
    tracks = SolutionTracks(graph=graph_2d, segmentation=segmentation_2d, ndim=3)

    tracks_viewer = TracksViewer.get_instance(viewer)
    tracks_viewer.update_tracks(tracks=tracks, name="test")
    tracks_viewer.set_kymograph_image_layer("raw")
    tracks_viewer.set_view_mode("kymograph")

    links_layer = tracks_viewer.kymograph_layers.branch_links_layer
    logical_id = int(links_layer.properties["logical_link_id"][0])
    first_index = int(
        np.flatnonzero(links_layer.properties["logical_link_id"] == logical_id)[0]
    )

    links_layer.process_click(MockEvent(), (first_index, None))

    assert set(tracks_viewer.selected_nodes) == {1, 2}
