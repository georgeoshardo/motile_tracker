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
    def __init__(self, modifiers=None):
        self.modifiers = modifiers or []


@pytest.fixture
def kymograph_tracks_viewer(viewer, solution_tracks_2d, segmentation_2d):
    """A TracksViewer showing solution_tracks_2d in kymograph mode with a raw image."""
    viewer.add_image(np.asarray(segmentation_2d, dtype=float), name="raw")

    tracks_viewer = TracksViewer.get_instance(viewer)
    tracks_viewer.update_tracks(tracks=solution_tracks_2d, name="test")
    tracks_viewer.set_kymograph_image_layer("raw")
    assert tracks_viewer.set_view_mode("kymograph") is True
    return tracks_viewer


def test_kymograph_link_click_selects_endpoints_without_centering_view(
    kymograph_tracks_viewer,
):
    tracks_viewer = kymograph_tracks_viewer
    links_layer = tracks_viewer.kymograph_layers.continuation_links_layer

    with patch.object(tracks_viewer, "center_on_node") as center_mock:
        links_layer.process_click(MockEvent(), (0, None))

    center_mock.assert_not_called()
    assert set(tracks_viewer.selected_nodes) == {3, 4}


def test_kymograph_branch_link_click_selects_branch_endpoints(
    kymograph_tracks_viewer,
):
    tracks_viewer = kymograph_tracks_viewer
    links_layer = tracks_viewer.kymograph_layers.branch_links_layer

    logical_id = int(links_layer.properties["logical_link_id"][0])
    first_index = int(
        np.flatnonzero(links_layer.properties["logical_link_id"] == logical_id)[0]
    )
    source = int(links_layer.properties["source_node"][first_index])
    target = int(links_layer.properties["target_node"][first_index])

    links_layer.process_click(MockEvent(), (first_index, None))

    assert set(tracks_viewer.selected_nodes) == {source, target}
    assert source == 1


def test_clicking_empty_space_on_links_layer_clears_selection(
    kymograph_tracks_viewer,
):
    tracks_viewer = kymograph_tracks_viewer
    links_layer = tracks_viewer.kymograph_layers.continuation_links_layer

    links_layer.process_click(MockEvent(), (0, None))
    assert len(tracks_viewer.selected_nodes) == 2

    links_layer.process_click(MockEvent(), None)
    assert len(tracks_viewer.selected_nodes) == 0
