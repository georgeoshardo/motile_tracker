import numpy as np
import pytest

from mm_track.application_menus.visualization_widget import (
    VisualizationWidget,
)
from mm_track.data_views.views_coordinator.tracks_viewer import TracksViewer


@pytest.fixture(autouse=True)
def clear_viewer_layers(viewer):
    """Clear viewer layers between tests."""
    yield
    viewer.layers.clear()


@pytest.fixture
def visualization_widget(viewer, solution_tracks_3d, qtbot):
    tracks_viewer = TracksViewer.get_instance(viewer)
    tracks_viewer.update_tracks(tracks=solution_tracks_3d, name="test")

    widget = VisualizationWidget(viewer)
    qtbot.addWidget(widget)

    assert tracks_viewer.tracking_layers.seg_layer is not None

    return widget, tracks_viewer


@pytest.mark.parametrize("mode", ["lineage", "group", "all"])
def test_background_opacity_only_with_background_labels(visualization_widget, mode):
    """There are no background labels in the 'all' display mode, so the background
    slider is disabled there."""

    widget, tracks_viewer = visualization_widget

    tracks_viewer.set_display_mode(mode)

    assert widget.background_widget.isEnabled() is (mode != "all")


def test_opacity_updates_seg_layer(visualization_widget):
    """Test that changing the opacity in the widget updates the highlighted/foreground/
    background opacity of the labels."""

    widget, tracks_viewer = visualization_widget
    layer = tracks_viewer.tracking_layers.seg_layer

    widget.highlight_widget.opacity.setValue(0.25)
    widget.foreground_widget.opacity.setValue(0.5)
    widget.background_widget.opacity.setValue(0.75)

    assert layer.highlight_opacity == pytest.approx(0.25)
    assert layer.foreground_opacity == pytest.approx(0.5)
    assert layer.background_opacity == pytest.approx(0.75)


def test_contour_checkbox_updates_layer(visualization_widget):
    """Test that contour (fill) checkboxes are hidden, unless in contour mode, and that
    toggling them changes the contour state on the seg_layer."""

    widget, tracks_viewer = visualization_widget
    layer = tracks_viewer.tracking_layers.seg_layer

    # mode where contour is not available
    tracks_viewer.set_display_mode("all")
    layer.contour = 0

    assert widget.highlight_widget.contour.isHidden()
    assert widget.foreground_widget.contour.isHidden()

    # still hidden, because contour is still 0
    tracks_viewer.set_display_mode("lineage")

    assert widget.highlight_widget.contour.isHidden()
    assert widget.foreground_widget.contour.isHidden()

    # Enable contours, ensure widgets are visible
    layer.contour = 1

    assert not widget.highlight_widget.contour.isHidden()
    assert not widget.foreground_widget.contour.isHidden()

    # Check = fill = contour OFF
    widget.highlight_widget.contour.setChecked(True)
    widget.foreground_widget.contour.setChecked(False)

    assert layer.highlight_contour is False
    assert layer.foreground_contour is True


@pytest.mark.parametrize(
    "mode", ["all", "visible_no_contours", "visible_with_contours"]
)
def test_update_label_colormap_when_selecting(
    viewer,
    solution_tracks_3d,
    mode,
):
    """Test the actual values on the label colormap"""
    tracks_viewer = TracksViewer.get_instance(viewer)
    tracks_viewer.update_tracks(tracks=solution_tracks_3d, name="test")

    seg_layer = tracks_viewer.tracking_layers.seg_layer
    assert hasattr(seg_layer, "update_label_colormap")

    cmap = seg_layer.colormap

    # Select specific labels for deterministic testing
    keys = list(cmap.color_dict.keys())
    numeric_keys = [k for k in keys if isinstance(k, int) and k != 0][:3]
    k0, k1, k2 = numeric_keys[:3]  # two labels for testing

    # Set a random starting value, to ensure it got updated
    for k in [k1, k2]:
        cmap.color_dict[k][-1] = 0.5

    assert seg_layer.background_opacity == 0.3
    assert seg_layer.foreground_opacity == 0.6
    assert seg_layer.highlight_opacity == 1.0

    # Make the viewer highlight one label
    tracks_viewer.selected_nodes.add_list([k2], append=False)

    # Call update_label_colormap in each test mode
    if mode == "all":
        seg_layer.update_label_colormap("all")
        # visible == "all" → all non-0, non-None get alpha 0.6 (foreground opacity)
        assert seg_layer.colormap.color_dict[k0][-1] == pytest.approx(
            seg_layer.foreground_opacity
        )
        assert seg_layer.colormap.color_dict[k1][-1] == pytest.approx(
            seg_layer.foreground_opacity
        )
        assert seg_layer.colormap.color_dict[k2][-1] == seg_layer.highlight_opacity

        assert seg_layer.filled_labels == []

    elif mode == "visible_no_contours":
        visible = [k1]  # simulate lineage/group mode
        seg_layer.update_label_colormap(visible)

        # normal mode: background labels get 0.3, foreground labels get 0.6, highlighted gets 1
        assert seg_layer.colormap.color_dict[k0][-1] == pytest.approx(
            seg_layer.background_opacity
        )
        assert seg_layer.colormap.color_dict[k1][-1] == pytest.approx(
            seg_layer.foreground_opacity
        )
        assert seg_layer.colormap.color_dict[k2][-1] == seg_layer.highlight_opacity

        assert seg_layer.filled_labels == []

    elif mode == "visible_with_contours":
        seg_layer.contour = 1
        visible = [k1]
        seg_layer.update_label_colormap(visible)

        # contour mode: background labels have 0.3,
        assert seg_layer.colormap.color_dict[k0][-1] == pytest.approx(
            seg_layer.background_opacity
        )  # background
        assert seg_layer.colormap.color_dict[k1][-1] == pytest.approx(
            seg_layer.foreground_opacity
        )  # foreground
        assert seg_layer.colormap.color_dict[k2][-1] == pytest.approx(
            seg_layer.highlight_opacity
        )  # highlighted

        assert set(seg_layer.filled_labels) == {k1, k2}


def test_selecting_node_does_not_highlight_same_track_nodes(
    viewer,
    solution_tracks_3d_with_division,
):
    """Highlighting one node must not change the opacity of other nodes that share
    its track id.

    Regression test for per-track color-array aliasing: _get_colormap must give
    each node its own color array, because set_opacity mutates the alpha in place.
    In solution_tracks_3d_with_division, nodes 1 and 2 share track id 1 (the
    tracklet before the division). Selecting node 2 must leave node 1 at the
    foreground opacity, not the highlight opacity.
    """
    tracks_viewer = TracksViewer.get_instance(viewer)
    tracks_viewer.update_tracks(tracks=solution_tracks_3d_with_division, name="test")
    seg_layer = tracks_viewer.tracking_layers.seg_layer

    same_track_unselected, selected = 1, 2
    assert tracks_viewer.tracks.get_track_id(
        same_track_unselected
    ) == tracks_viewer.tracks.get_track_id(selected)

    tracks_viewer.selected_nodes.add(selected, append=False)
    seg_layer.update_label_colormap("all")

    color_dict = seg_layer.colormap.color_dict
    assert color_dict[selected][-1] == pytest.approx(seg_layer.highlight_opacity)
    assert color_dict[same_track_unselected][-1] == pytest.approx(
        seg_layer.foreground_opacity
    )


@pytest.fixture
def visualization_widget_2d(viewer, solution_tracks_2d, segmentation_2d, qtbot):
    """A VisualizationWidget on 2D+time tracks with a matching raw image layer, so
    that the kymograph view is available."""
    viewer.add_image(np.asarray(segmentation_2d, dtype=float), name="raw")

    tracks_viewer = TracksViewer.get_instance(viewer)
    tracks_viewer.update_tracks(tracks=solution_tracks_2d, name="test")
    tracks_viewer.set_kymograph_image_layer("raw")

    widget = VisualizationWidget(viewer)
    qtbot.addWidget(widget)

    return widget, tracks_viewer


class TestKymographControls:
    """Tests for the view mode radio buttons and the kymograph controls."""

    def test_kymograph_controls_disabled_for_3d_tracks(self, visualization_widget):
        """The kymograph view only supports 2D+time tracks."""
        widget, _ = visualization_widget

        assert widget.kymograph_box.isEnabled() is False

    def test_visualization_widget_is_not_height_capped(self, visualization_widget_2d):
        """The kymograph controls need room: the widget must not clip its contents."""
        widget, _ = visualization_widget_2d

        assert widget.maximumHeight() >= widget.sizeHint().height()

    def test_page_controls_follow_the_kymograph_layers(self, visualization_widget_2d):
        widget, tracks_viewer = visualization_widget_2d
        tracks_viewer.set_view_mode("kymograph")

        widget.page_length_box.setValue(2)
        assert tracks_viewer.kymograph_layers.page_length == 2
        assert widget.page_start_box.maximum() == 3  # 5 frames, 2 per page

        widget.next_page_button.click()
        assert tracks_viewer.kymograph_layers.page_start == 2
        assert widget.page_start_box.value() == 2

        widget.prev_page_button.click()
        assert tracks_viewer.kymograph_layers.page_start == 0
        assert widget.page_start_box.value() == 0

    def test_kymograph_link_visibility_checkboxes_toggle_layers(
        self, visualization_widget_2d
    ):
        widget, tracks_viewer = visualization_widget_2d

        tracks_viewer.set_view_mode("kymograph")

        kymograph_layers = tracks_viewer.kymograph_layers
        assert kymograph_layers.continuation_links_layer is not None
        assert kymograph_layers.branch_links_layer is not None
        assert widget.paths_checkbox.isChecked() is True
        assert widget.branches_checkbox.isChecked() is True

        widget.paths_checkbox.setChecked(False)
        widget.branches_checkbox.setChecked(False)

        assert kymograph_layers.show_paths is False
        assert kymograph_layers.show_branches is False
        assert kymograph_layers.continuation_links_layer.visible is False
        assert kymograph_layers.branch_links_layer.visible is False

        widget.paths_checkbox.setChecked(True)
        widget.branches_checkbox.setChecked(True)

        assert kymograph_layers.continuation_links_layer.visible is True
        assert kymograph_layers.branch_links_layer.visible is True

    def test_line_width_and_marker_size_scale_the_kymograph(
        self, visualization_widget_2d
    ):
        """The two size boxes resize the lines and markers as they change, and the
        sizes are kept when the layers are rebuilt for another page."""
        widget, tracks_viewer = visualization_widget_2d
        tracks_viewer.set_view_mode("kymograph")
        kymograph_layers = tracks_viewer.kymograph_layers
        paths = kymograph_layers.continuation_links_layer
        branches = kymograph_layers.branch_links_layer
        points = kymograph_layers.points_layer
        path_width = paths.edge_width[0]
        branch_width = branches.edge_width[0]
        point_size = points.size[0]

        try:
            widget.link_width_box.setValue(2.5)
            widget.point_size_box.setValue(0.5)

            assert np.allclose(paths.edge_width, 2.5 * path_width)
            assert np.allclose(branches.edge_width, 2.5 * branch_width)
            assert np.allclose(points.size, 0.5 * point_size)

            widget.page_length_box.setValue(4)  # rebuilds the page
            paths = kymograph_layers.continuation_links_layer
            assert np.allclose(paths.edge_width, 2.5 * path_width)
            assert np.allclose(kymograph_layers.points_layer.size, 0.5 * point_size)
        finally:
            widget.link_width_box.setValue(1.0)
            widget.point_size_box.setValue(1.0)
