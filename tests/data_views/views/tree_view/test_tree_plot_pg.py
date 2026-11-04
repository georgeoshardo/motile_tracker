"""Tests for the pyqtgraph tree canvas (tree_plot_pg.TreePlot), the one used where
fastplotlib can't run."""

import numpy as np
import pandas as pd
import pytest
from qtpy.QtCore import QPoint, QPointF, Qt
from qtpy.QtGui import QWheelEvent
from qtpy.QtWidgets import QApplication

from mm_track.data_views.views.tree_view import tree_plot_shared as shared
from mm_track.data_views.views.tree_view.tree_plot_pg import TreePlot

RED, GREEN, BLUE = (255, 0, 0, 255), (0, 255, 0, 255), (0, 0, 255, 255)


def lineage_df() -> pd.DataFrame:
    """Track 1 (lane 1, t 0-2) divides into tracks 2 (lane 0) and 3 (lane 2)."""
    rows = [
        # node_id, parent_id, t, lane, track, parent_track, symbol, color, area
        (1, 0, 0, 1, 1, 0, "o", RED, 10.0),
        (2, 1, 1, 1, 1, 0, "o", RED, 11.0),
        (3, 2, 2, 1, 1, 0, "t1", RED, 12.0),
        (4, 3, 3, 0, 2, 1, "o", GREEN, 6.0),
        (5, 4, 4, 0, 2, 1, "x", GREEN, 7.0),
        (6, 3, 3, 2, 3, 1, "o", BLUE, 5.0),
        (7, 6, 4, 2, 3, 1, "x", BLUE, 6.0),
    ]
    columns = [
        "node_id",
        "parent_id",
        "t",
        "x_axis_pos",
        "track_id",
        "parent_track_id",
        "symbol",
        "color",
        "area",
    ]
    df = pd.DataFrame(rows, columns=columns)
    df["color"] = [np.array(c, dtype=np.uint8) for c in df["color"]]
    return df


@pytest.fixture
def plot(qtbot):
    tree_plot = TreePlot()
    qtbot.addWidget(tree_plot)
    tree_plot.resize(600, 400)
    tree_plot.show()
    qtbot.waitExposed(tree_plot)
    tree_plot.update(lineage_df(), "vertical", "tree", None, [], reset_view=True)
    QApplication.processEvents()
    return tree_plot


def widget_pos(plot: TreePlot, x: float, y: float) -> QPoint:
    """Where the data point (x, y) is on the graphics view's viewport."""
    scene = plot._view.mapViewToScene(QPointF(x, y))
    return plot._widget.mapFromScene(scene)


def node_pos(plot: TreePlot, node_id: int) -> QPoint:
    row = plot._id_to_row[node_id]
    return widget_pos(plot, *plot._positions[row, :2])


def in_view(plot: TreePlot, node_id: int) -> bool:
    row = plot._id_to_row[node_id]
    x, y = plot._positions[row, :2]
    (x0, x1), (y0, y1) = plot._view.viewRange()
    return x0 <= x <= x1 and y0 <= y <= y1


def test_draws_nodes_and_edges(plot):
    assert len(plot._scatter.points()) == 7
    assert sum(len(lines) for _, lines in plot._edges._groups) == 6
    # an edge takes the colour of its child: three tracks, three colours
    assert len(plot._edges._groups) == 3
    symbols = [point.symbol() for point in plot._scatter.points()]
    assert symbols == ["o", "o", "t1", "o", "x", "o", "x"]

    plot.update(pd.DataFrame(), "vertical", "tree", None, [])
    assert len(plot._scatter.points()) == 0
    assert len(plot._node_ids) == 0


def test_vertical_layout_puts_time_down_and_lanes_across(plot):
    row = plot._id_to_row[5]
    assert tuple(plot._positions[row, :2]) == (0.0, -4.0)
    # the time axis reads t, not the negated coordinate it is drawn at
    assert plot._left_axis.tickStrings([-4.0], 1, 1) == ["4"]


def test_clicks_emit_node_clicked_and_jump_to_node(plot, qtbot):
    clicked, jumped = [], []
    plot.node_clicked.connect(lambda node, append: clicked.append((node, append)))
    plot.jump_to_node.connect(jumped.append)
    viewport = plot._widget.viewport()

    qtbot.mouseClick(viewport, Qt.LeftButton, Qt.NoModifier, node_pos(plot, 2))
    qtbot.mouseClick(viewport, Qt.LeftButton, Qt.ShiftModifier, node_pos(plot, 6))
    qtbot.mouseClick(viewport, Qt.LeftButton, Qt.ControlModifier, node_pos(plot, 7))

    assert clicked == [(2, False), (6, True)]
    assert jumped == [7]


def test_selection_outlines_and_enlarges_nodes(plot):
    points = plot._scatter.points()
    row = plot._id_to_row[4]
    before = points[row].size()

    plot.set_selection([4], "tree")
    assert points[row].pen().color().getRgb()[:3] == (0, 255, 255)
    assert points[row].size() > before

    plot.set_selection([], "tree")
    assert points[row].pen().style() == Qt.NoPen
    assert points[row].size() == pytest.approx(before)


def test_box_select_emits_the_nodes_inside(plot):
    received = []
    plot.nodes_selected.connect(lambda nodes, append: received.append((nodes, append)))
    # lanes 0-1, t 2-4 (y is -t)
    plot.select_points_in_rect(-0.5, 1.5, -4.5, -1.5)
    assert received == [([3, 4, 5], True)]


def test_centering_pans_without_zooming_in(plot):
    plot._view.setRange(xRange=(0.5, 1.5), yRange=(-1.5, 0.5), padding=0)
    QApplication.processEvents()
    width = np.diff(plot._view.viewRange()[0])[0]
    assert not in_view(plot, 7)

    plot.center_on_node(7)
    assert in_view(plot, 7)
    assert np.diff(plot._view.viewRange()[0])[0] == pytest.approx(width)


def test_selection_brings_off_screen_nodes_into_view(plot):
    plot._view.setRange(xRange=(0.5, 1.5), yRange=(-1.5, 0.5), padding=0)
    QApplication.processEvents()
    plot.set_selection([1], "tree")
    view = plot._view.viewRange()

    # both on screen: the zoom the user picked them out in is kept
    plot.set_selection([1, 2], "tree")
    assert plot._view.viewRange() == view

    # one just added off screen: the view widens to show both
    plot.set_selection([1, 2, 7], "tree")
    assert in_view(plot, 1) and in_view(plot, 7)


def test_track_id_axis(plot):
    axis = plot._bottom_axis
    plot.set_show_track_ids(True)
    values = axis.tickValues(-0.5, 2.5, 600)[0][1]
    assert values == [0, 1, 2]
    assert axis.tickStrings(values, 1, 1) == ["2", "1", "3"]

    # flipped, lanes run down the left axis at y = -lane
    plot.update(lineage_df(), "horizontal", "tree", None, [], reset_view=True)
    axis = plot._left_axis
    values = axis.tickValues(-2.5, 0.5, 400)[0][1]
    assert sorted(values) == [-2, -1, 0]
    assert axis.tickStrings([-2], 1, 1) == ["3"]


def test_feature_plot_reads_values_upward(plot):
    plot.update(lineage_df(), "horizontal", "feature", "area", [], reset_view=True)
    row = plot._id_to_row[3]
    assert tuple(plot._positions[row, :2]) == (2.0, 12.0)
    assert plot._left_axis.tickStrings([12.0], 1, 1) == ["12"]


def test_markers_and_labels_follow_the_zoom(plot):
    plot._view.setRange(xRange=(0.5, 1.5), yRange=(-1.5, 0.5), padding=0)
    QApplication.processEvents()
    assert plot._size_scale > 1
    shown = {item.toPlainText() for item in plot._node_labels if item.isVisible()}
    assert shown == {"1", "2"}

    plot._view.setRange(xRange=(-100, 100), yRange=(-100, 100), padding=0)
    QApplication.processEvents()
    assert plot._size_scale == pytest.approx(shared.MIN_SIZE / shared.BASE_SIZE)
    assert not any(item.isVisible() for item in plot._node_labels)


def test_right_click_resets_the_view(plot, qtbot):
    full = plot._view.viewRange()
    plot._view.setRange(xRange=(0.5, 1.5), yRange=(-1.5, 0.5), padding=0)
    QApplication.processEvents()
    qtbot.mouseClick(
        plot._widget.viewport(), Qt.RightButton, Qt.NoModifier, QPoint(300, 200)
    )
    assert np.allclose(plot._view.viewRange(), full)


def test_scrolling_is_recorded(plot):
    plot.reset_scrolled()
    viewport = plot._widget.viewport()
    pos = QPointF(300, 200)
    event = QWheelEvent(
        pos,
        QPointF(viewport.mapToGlobal(pos.toPoint())),
        QPoint(0, 0),
        QPoint(0, 120),
        Qt.NoButton,
        Qt.NoModifier,
        Qt.NoScrollPhase,
        False,
    )
    QApplication.sendEvent(viewport, event)
    assert plot.scrolled
    plot.reset_scrolled()
    assert not plot.scrolled


def test_close_figure_drops_the_data(plot):
    plot.close_figure()
    assert len(plot._scatter.points()) == 0
    plot.update(lineage_df(), "vertical", "tree", None, [])
    assert len(plot._scatter.points()) == 0


def test_hover_describes_the_node():
    df = lineage_df()
    text = shared.describe_node(4, df["node_id"].to_numpy(), shared.track_lookups(df))
    assert text == "node 5\ntrack 2\nlineage 1\nt 4"


def test_time_marks_geometry():
    positions = np.array([[0, 0, 0], [5, -3, 0]], dtype=np.float32)  # (t, -lane)
    band, line = shared.time_marks(positions, "horizontal", (1.5, 3.5), 2)
    assert band == (1.5, -4.0, 3.5, 1.0)
    assert line.tolist() == [[2.0, -4.0], [2.0, 1.0]]
    vertical = np.array([[0, 0, 0], [3, -5, 0]], dtype=np.float32)  # (lane, -t)
    band, line = shared.time_marks(vertical, "vertical", (1.5, 3.5), None)
    assert band == (-1.0, -3.5, 4.0, -1.5)
    assert line is None
    assert shared.time_marks(np.empty((0, 3)), "horizontal", (1, 2), 1) == (None, None)


def test_time_marks_are_drawn_and_kept_through_a_rebuild(plot):
    plot.update(lineage_df(), "horizontal", "tree", None, [])
    plot.set_time_marks((0.5, 2.5), 1)
    assert plot._view_band.isVisible()
    assert plot._hover_line.isVisible()
    assert plot._view_band.rect().left() == pytest.approx(0.5)
    plot.update(lineage_df(), "horizontal", "tree", None, [])
    assert plot._view_band.isVisible()
    plot.set_time_marks(None, None)
    assert not plot._view_band.isVisible()
    assert not plot._hover_line.isVisible()


def test_view_follows_the_band_along_time_only():
    view = (50.0, -10.0, 100.0, 20.0)  # frames 0..100
    assert shared.view_following_band(view, (40, -20, 60, 0), "horizontal") is None
    cx, cy, w, h = shared.view_following_band(view, (120, -20, 130, 0), "horizontal")
    assert (cy, w, h) == (-10.0, 100.0, 20.0)  # lanes and zoom kept
    assert cx + w / 2 - shared.FOLLOW_MARGIN * w == pytest.approx(130)  # just in view
    cx, *_ = shared.view_following_band(view, (-30, -20, -20, 0), "horizontal")
    assert cx - w / 2 + shared.FOLLOW_MARGIN * w == pytest.approx(-30)
    # vertical: time runs down y as -t
    vertical = (5.0, -50.0, 10.0, 100.0)  # frames 0..100
    _, cy, _, _ = shared.view_following_band(vertical, (0, -130, 10, -120), "vertical")
    assert cy - 50 + shared.FOLLOW_MARGIN * 100 == pytest.approx(-130)


def test_tree_opens_at_a_readable_time_scale():
    rect = (-5.0, 255.0, -40.0, 1.0)  # 260 frames
    assert shared.limit_time_span(rect, "horizontal", 70, None) == (
        -5.0,
        65.0,
        -40.0,
        1.0,
    )
    assert shared.limit_time_span(rect, "horizontal", 70, 150) == (
        115.0,
        185.0,
        -40.0,
        1.0,
    )
    assert shared.limit_time_span(rect, "horizontal", 70, 250) == (
        185.0,
        255.0,
        -40.0,
        1.0,
    )
    assert shared.limit_time_span(rect, "horizontal", 300, 150) == rect
    vertical = (-1.0, 40.0, -255.0, 5.0)
    assert shared.limit_time_span(vertical, "vertical", 70, None) == (
        -1.0,
        40.0,
        -65.0,
        5.0,
    )
