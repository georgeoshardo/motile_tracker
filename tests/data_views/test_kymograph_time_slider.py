"""The kymograph view's time slider (a Qt slider docked under the canvas) pans the
kymograph one frame per step and follows the view."""

import numpy as np
import pytest

from motile_tracker.data_views.views.kymograph_time_slider import KymographTimeSlider
from motile_tracker.data_views.views_coordinator.tracks_viewer import TracksViewer


@pytest.fixture(autouse=True)
def clear_viewer_layers(viewer):
    yield
    tracks_viewer = getattr(TracksViewer, "_instance", None)
    if tracks_viewer is not None and tracks_viewer.view_mode == "kymograph":
        tracks_viewer.set_view_mode("spatial")
    viewer.layers.clear()


@pytest.fixture
def kymograph(viewer, solution_tracks_2d):
    """A TracksViewer showing solution_tracks_2d (5 frames of 100x100) as a
    kymograph, all frames on one page."""
    tracks_viewer = TracksViewer.get_instance(viewer)
    tracks_viewer.update_tracks(tracks=solution_tracks_2d, name="test")
    assert tracks_viewer.set_view_mode("kymograph")
    return tracks_viewer


def test_kymograph_mode_shows_a_time_slider(viewer, kymograph):
    layers = kymograph.kymograph_layers
    slider = layers.time_slider

    assert isinstance(slider, KymographTimeSlider)
    assert not slider.isHidden()
    # it sits in napari's central viewer widget, right below the canvas, so it
    # spans the canvas width rather than living in a dock area
    canvas = viewer.window._qt_viewer.canvas.native
    layout = canvas.parentWidget().layout()
    assert layout.indexOf(slider) == layout.indexOf(canvas) + 1
    assert layers._time_slider_dock is None
    assert slider.slider.minimum() == 0
    assert slider.slider.maximum() == 4  # one step per frame
    assert slider.label.text() == "t = 2 / 4"  # the frame under the view centre
    # the viewer itself stays a plain 2D view
    assert viewer.dims.ndim == 2
    assert viewer.layers.selection.active is layers.labels_layer
    assert "Time slider" in viewer.text_overlay.text


def test_a_new_tracks_viewer_replaces_the_old_slider(viewer, kymograph):
    """Only one time slider ever sits under the canvas, even when a new TracksViewer
    (and with it a new layer group) is created for the same viewer."""
    old_slider = kymograph.kymograph_layers.time_slider
    canvas = viewer.window._qt_viewer.canvas.native
    layout = canvas.parentWidget().layout()
    # what the reset_tracks_viewer fixture does between tests
    viewer.keymap.clear()
    del TracksViewer._instance

    new_tracks_viewer = TracksViewer.get_instance(viewer)
    new_tracks_viewer.update_tracks(tracks=kymograph.tracks, name="new")
    assert new_tracks_viewer.set_view_mode("kymograph")

    sliders = [
        layout.itemAt(i).widget()
        for i in range(layout.count())
        if isinstance(layout.itemAt(i).widget(), KymographTimeSlider)
    ]
    assert sliders == [new_tracks_viewer.kymograph_layers.time_slider]
    assert old_slider not in sliders
    new_tracks_viewer.set_view_mode("spatial")


def test_slider_pans_the_view_to_the_frame(viewer, kymograph):
    layers = kymograph.kymograph_layers
    frame_width = layers.geometry.frame_world_width
    y_before = viewer.camera.center[-2]

    layers.time_slider.slider.setValue(3)

    assert layers.page_start == 0  # all frames fit on the page: no page turn
    assert viewer.camera.center[-1] == pytest.approx(3.5 * frame_width, abs=1.0)
    assert viewer.camera.center[-2] == pytest.approx(y_before)
    assert layers.current_timepoint == 3
    assert layers.time_slider.label.text() == "t = 3 / 4"


def test_arrow_buttons_step_one_frame(viewer, kymograph):
    layers = kymograph.kymograph_layers
    frame_width = layers.geometry.frame_world_width
    layers.time_slider.slider.setValue(1)

    layers.time_slider.next_button.click()
    assert layers.current_timepoint == 2
    assert viewer.camera.center[-1] == pytest.approx(2.5 * frame_width, abs=1.0)

    layers.time_slider.previous_button.click()
    layers.time_slider.previous_button.click()
    assert layers.current_timepoint == 0
    layers.time_slider.previous_button.click()  # stays at the first frame
    assert layers.current_timepoint == 0


def test_slider_turns_the_page_when_the_frame_is_not_on_it(viewer, kymograph):
    layers = kymograph.kymograph_layers
    kymograph.set_kymograph_page_length(2)
    assert layers.page_start == 0
    assert layers.time_slider.slider.pageStep() == 2
    frame_width = layers.geometry.frame_world_width

    layers.time_slider.slider.setValue(4)

    # the page is moved to hold frame 4 (start clamped to t_size - page_length)
    assert layers.page_start == 3
    assert viewer.camera.center[-1] == pytest.approx(
        (4 - 3 + 0.5) * frame_width, abs=1.0
    )
    assert layers.current_timepoint == 4
    assert layers.labels_layer.data.shape == (100, 2 * 100)


def test_panning_the_view_moves_the_slider(viewer, kymograph):
    layers = kymograph.kymograph_layers
    frame_width = layers.geometry.frame_world_width
    center = list(viewer.camera.center)

    center[-1] = 3.4 * frame_width
    viewer.camera.center = tuple(center)

    assert layers.current_timepoint == 3
    # the slider follows the view without snapping the camera back to the frame centre
    assert viewer.camera.center[-1] == pytest.approx(3.4 * frame_width)


def test_centering_on_a_node_moves_the_slider(viewer, kymograph):
    kymograph.center_on_node(5)  # node 5 is at t=4

    assert kymograph.kymograph_layers.current_timepoint == 4


def test_show_timepoint_clamps_to_the_movie(viewer, kymograph):
    layers = kymograph.kymograph_layers

    layers.show_timepoint(99)
    assert layers.current_timepoint == 4
    layers.show_timepoint(-3)
    assert layers.current_timepoint == 0


def test_page_start_change_keeps_the_slider_in_step(viewer, kymograph):
    layers = kymograph.kymograph_layers
    kymograph.set_kymograph_page_length(2)
    layers.time_slider.slider.setValue(0)
    assert layers.current_timepoint == 0

    # turning the page from the visualization controls keeps the camera where it
    # is, so the frame under it (and the slider) advance with the page
    kymograph.step_kymograph_page(1)

    assert layers.page_start == 2
    assert layers.current_timepoint == 2


def test_leaving_kymograph_mode_hides_the_time_slider(viewer, kymograph):
    layers = kymograph.kymograph_layers
    slider = layers.time_slider

    kymograph.set_view_mode("spatial")

    assert slider.isHidden()
    # panning the spatial view no longer touches the slider
    value = layers.current_timepoint
    center = list(viewer.camera.center)
    center[-1] += 100
    viewer.camera.center = tuple(center)
    assert layers.current_timepoint == value

    assert kymograph.set_view_mode("kymograph")
    assert layers.time_slider is slider  # created once, shown again
    assert not slider.isHidden()


def test_reloading_tracks_in_kymograph_mode_keeps_the_slider(
    viewer, kymograph, solution_tracks_2d
):
    layers = kymograph.kymograph_layers
    slider = layers.time_slider

    kymograph.update_tracks(tracks=solution_tracks_2d, name="again")

    assert kymograph.view_mode == "kymograph"
    assert layers.time_slider is slider
    assert slider.slider.maximum() == 4
    slider.slider.setValue(2)
    assert viewer.camera.center[-1] == pytest.approx(
        2.5 * layers.geometry.frame_world_width, abs=1.0
    )


def test_edge_across_a_frame_gap_is_drawn(viewer, kymograph, solution_tracks_2d):
    """Linking a node at t to one at t+3 (skipping frames) draws a dashed line
    across the skipped frames, as for any same-track link."""
    layers = kymograph.kymograph_layers
    tracks = kymograph.tracks
    assert tracks.graph.out_degree(2) == 0  # node 2 (t=1) ends its track
    assert tracks.graph.in_degree(6) == 0  # node 6 (t=4) starts one
    before = len(layers.continuation_links_layer.data)

    kymograph.selected_nodes.add(2)
    kymograph.selected_nodes.add(6, append=True)
    kymograph.force = True  # no confirmation dialog for the unusual link
    kymograph.create_edge()

    assert tracks.graph.has_edge(2, 6)
    assert tracks.get_track_id(6) == tracks.get_track_id(2)
    paths = layers.continuation_links_layer
    assert len(paths.data) > before
    is_new = (paths.properties["source_node"] == 2) & (
        paths.properties["target_node"] == 6
    )
    assert is_new.sum() > 1  # dashed: several fragments
    assert set(paths.properties["link_kind"][is_new]) == {"gap"}
    fragments = np.concatenate(
        [seg for seg, new in zip(paths.data, is_new, strict=True) if new]
    )
    # spans from frame 1 to frame 4 along the time axis
    frame_width = layers.geometry.frame_world_width
    assert fragments[:, 1].min() >= 1 * frame_width
    assert fragments[:, 1].max() <= 5 * frame_width
    assert fragments[:, 1].max() - fragments[:, 1].min() > 2 * frame_width

    kymograph.undo()
    assert not tracks.graph.has_edge(2, 6)
    paths = layers.continuation_links_layer
    assert not (
        (paths.properties["source_node"] == 2) & (paths.properties["target_node"] == 6)
    ).any()
