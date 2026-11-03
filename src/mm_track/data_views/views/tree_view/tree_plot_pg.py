"""Lineage tree canvas drawn with pyqtgraph (Qt's painter), for where wgpu can't run.

The same interface and behaviour as tree_plot_fpl.TreePlot, whose layout, sizing,
labelling and selection logic it shares (tree_plot_shared); only the drawing is
its own.
"""

import contextlib
from typing import Any

import numpy as np
import pandas as pd
import pyqtgraph as pg
from psygnal import Signal
from qtpy.QtCore import QLineF, QRectF, Qt
from qtpy.QtGui import QColor, QCursor, QFont
from qtpy.QtWidgets import QGraphicsRectItem, QToolTip, QVBoxLayout, QWidget

from mm_track.data_views.views.tree_view import tree_plot_shared as shared
from mm_track.data_views.views.tree_view.tree_plot_shared import (
    BASE_SIZE,
    NODE_LABEL_MAX,
    NODE_LABEL_MIN_SIZE,
    SELECT_COLOR,
    SELECT_EDGE_WIDTH,
    SIZE_SCALE_EPS,
    TOOLTIP_OFFSET,
)

_AXIS_COLOR = (153, 153, 153)  # grey axis lines/ticks/labels
_EDGE_WIDTH = 1.5  # px
_NODE_LABEL_COLOR = (0, 0, 0)  # black on the track-coloured fill
_RUBBER_COLOR = (255, 255, 0)  # yellow box-select rectangle
_SELECT_PEN = pg.mkPen(QColor.fromRgbF(*SELECT_COLOR), width=SELECT_EDGE_WIDTH)
_NO_PEN = pg.mkPen(None)


class _EdgesItem(pg.GraphicsObject):
    """Every parent -> child edge, in its track's colour: one painter call per colour."""

    def __init__(self) -> None:
        super().__init__()
        self._groups: list[tuple[Any, list[QLineF]]] = []
        self._rect = QRectF()

    def set_edges(
        self, starts: np.ndarray, ends: np.ndarray, colors: np.ndarray
    ) -> None:
        """``starts``/``ends``: (E, 2) positions; ``colors``: (E, 4) RGBA 0-255."""
        self.prepareGeometryChange()
        groups: dict[tuple, list[QLineF]] = {}
        for color, a, b in zip(map(tuple, colors), starts, ends, strict=True):
            groups.setdefault(color, []).append(
                QLineF(float(a[0]), float(a[1]), float(b[0]), float(b[1]))
            )
        self._groups = [
            (pg.mkPen(QColor(*map(int, color)), width=_EDGE_WIDTH), lines)
            for color, lines in groups.items()
        ]
        if len(starts):
            points = np.concatenate([starts, ends])
            low, high = points.min(axis=0), points.max(axis=0)
            self._rect = QRectF(
                float(low[0]),
                float(low[1]),
                float(high[0] - low[0]),
                float(high[1] - low[1]),
            )
        else:
            self._rect = QRectF()
        self.update()

    def boundingRect(self) -> QRectF:  # noqa: N802 (Qt-style name)
        return self._rect

    def paint(self, painter, *args) -> None:
        for pen, lines in self._groups:
            painter.setPen(pen)
            painter.drawLines(lines)


class _Axis(pg.AxisItem):
    """An axis showing plain values, values of a negated coordinate (time and tree
    lanes are drawn at y = -value), tick marks only, or a track id per lane."""

    def __init__(self, orientation: str) -> None:
        super().__init__(orientation)
        self._negate = False
        self._lane_labels: dict[int, int] | None = None
        self.setPen(_AXIS_COLOR)
        self.setTextPen(_AXIS_COLOR)

    def configure(
        self,
        *,
        values: bool,
        negate: bool = False,
        lane_labels: dict[int, int] | None = None,
        title: str | None = None,
    ) -> None:
        self._negate = negate
        self._lane_labels = lane_labels
        self.setStyle(showValues=values or lane_labels is not None)
        if title:
            self.setLabel(title)
        else:
            self.showLabel(False)
        self.picture = None
        self.update()

    def tickValues(self, minVal, maxVal, size):  # noqa: N802, N803 (pyqtgraph names)
        if self._lane_labels is None:
            return super().tickValues(minVal, maxVal, size)
        sign = -1 if self._negate else 1
        low, high = sorted((sign * minVal, sign * maxVal))
        lanes = shared.labelled_lanes(low, high, size, self._lane_labels)
        return [(1.0, [sign * lane for lane in lanes])]

    def tickStrings(self, values, scale, spacing):  # noqa: N802 (pyqtgraph name)
        if self._lane_labels is not None:
            sign = -1 if self._negate else 1
            return [
                str(self._lane_labels.get(round(sign * value), "")) for value in values
            ]
        if self._negate:
            return super().tickStrings([-value for value in values], scale, spacing)
        return super().tickStrings(values, scale, spacing)


class _TreeViewBox(pg.ViewBox):
    """Pan/zoom as pyqtgraph does it (left drag pans, right drag squeezes an axis,
    wheel zooms), plus the tree's own mouse bindings: right-click resets the view,
    shift-drag box-selects, and the mouse's back/forward buttons step through the
    selection history."""

    def __init__(self, plot: "TreePlot") -> None:
        super().__init__(enableMenu=False)
        self._plot = plot
        self._rubber = QGraphicsRectItem()
        self._rubber.setPen(pg.mkPen(_RUBBER_COLOR, width=1))
        self._rubber.hide()
        self.addItem(self._rubber, ignoreBounds=True)

    def mouseClickEvent(self, ev) -> None:  # noqa: N802 (Qt-style name)
        self._plot.setFocus()
        button = ev.button()
        if button == Qt.MouseButton.RightButton:
            self._plot._reset_view()
        elif button == Qt.MouseButton.BackButton:
            self._plot.update_selection.emit(True)
        elif button == Qt.MouseButton.ForwardButton:
            self._plot.update_selection.emit(False)
        else:
            return
        ev.accept()

    def mouseDragEvent(self, ev, axis=None) -> None:  # noqa: N802 (Qt-style name)
        self._plot.setFocus()
        if not (
            ev.button() == Qt.MouseButton.LeftButton
            and ev.modifiers() & Qt.KeyboardModifier.ShiftModifier
        ):
            super().mouseDragEvent(ev, axis)
            return
        ev.accept()
        start = self.mapSceneToView(ev.buttonDownScenePos())
        end = self.mapSceneToView(ev.scenePos())
        if ev.isFinish():
            self._rubber.hide()
            self._plot.select_points_in_rect(start.x(), end.x(), start.y(), end.y())
            return
        self._rubber.setRect(QRectF(start, end).normalized())
        self._rubber.show()

    def wheelEvent(self, ev, axis=None) -> None:  # noqa: N802 (Qt-style name)
        self._plot._scrolled = True
        super().wheelEvent(ev, axis)


class TreePlot(QWidget):
    """pyqtgraph canvas for the lineage tree, interchangeable with
    tree_plot_fpl.TreePlot: the same signals (``node_clicked``, ``jump_to_node``,
    ``nodes_selected``, ``update_selection``) and public methods."""

    node_clicked = Signal(Any, bool)  # node_id, append
    jump_to_node = Signal(int)
    nodes_selected = Signal(list, bool)
    update_selection = Signal(bool)  # forward/backward in selection history

    def __init__(self) -> None:
        super().__init__()
        self.setFocusPolicy(Qt.StrongFocus)

        self._closed = False  # set by close_figure()
        self._scrolled = False  # set by the wheel handler, read via `scrolled`
        self.view_direction = "vertical"
        self.plot_type = "tree"
        self.feature = None
        self.track_df = pd.DataFrame()

        # per-render caches (stable node_id <-> row index mapping)
        self._node_ids = np.empty(0, dtype=np.int64)
        self._id_to_row: dict[int, int] = {}
        self._positions = np.empty((0, 3), dtype=np.float32)
        self._base_sizes = np.empty(0, dtype=np.float32)
        self._selected_rows: list[int] = []
        self._size_scale = 1.0  # zoom-dependent multiplier on _base_sizes
        self._lookups = shared.empty_lookups()
        self.show_track_ids = False  # toggled from the Visualization tab
        self.show_hover_info = True  # ...as is the hover tooltip

        self._view = _TreeViewBox(self)
        self._left_axis = _Axis("left")
        self._bottom_axis = _Axis("bottom")
        self._widget = pg.PlotWidget(
            viewBox=self._view,
            axisItems={"left": self._left_axis, "bottom": self._bottom_axis},
        )
        # keys go to TreePlot (and on to TreeWidget), not to the graphics view
        self._widget.setFocusPolicy(Qt.NoFocus)
        self._widget.setMinimumHeight(120)
        self._view.disableAutoRange()

        self._edges = _EdgesItem()
        self._scatter = pg.ScatterPlotItem(pxMode=True, hoverable=True, tip=None)
        self._scatter.sigClicked.connect(self._on_click)
        self._scatter.sigHovered.connect(self._on_hover)
        self._view.addItem(self._edges)
        self._view.addItem(self._scatter)
        self._node_labels: list[pg.TextItem] = []

        self._view.sigRangeChanged.connect(self._on_view_changed)
        self._view.sigResized.connect(self._on_view_changed)
        self._configure_axes()

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._widget)
        self.setLayout(layout)

    # ------------------------------------------------------------------ #
    # pan/zoom + X/Y axis lock
    # ------------------------------------------------------------------ #
    @property
    def scrolled(self) -> bool:
        """Whether the user scrolled since the last `reset_scrolled` call.

        X and Y double as zoom modifiers (held down while scrolling) and as plain
        keyboard shortcuts, so TreeWidget resets this when such a key goes down and
        checks it when the key comes back up.
        """
        return self._scrolled

    def reset_scrolled(self) -> None:
        """Forget any scrolling seen so far."""
        self._scrolled = False

    def setMouseEnabled(self, x: bool, y: bool) -> None:  # noqa: N802 (Qt-style name)
        """Restrict zoom/pan to the given axes (X or Y key held)."""
        self._view.setMouseEnabled(x=x, y=y)

    # ------------------------------------------------------------------ #
    # teardown
    # ------------------------------------------------------------------ #
    def close_figure(self) -> None:
        """Drop the plotted data; nothing is drawn after this."""
        if self._closed:
            return
        self._closed = True
        with contextlib.suppress(Exception):
            self._view.sigRangeChanged.disconnect(self._on_view_changed)
            self._view.sigResized.disconnect(self._on_view_changed)
        self._scatter.clear()
        self._edges.set_edges(np.empty((0, 2)), np.empty((0, 2)), np.empty((0, 4)))
        for label in self._node_labels:
            label.hide()
        self._positions = np.empty((0, 3), dtype=np.float32)
        self._base_sizes = np.empty(0, dtype=np.float32)
        self._node_ids = np.empty(0, dtype=np.int64)
        self._id_to_row = {}
        self._selected_rows = []
        self._lookups = shared.empty_lookups()
        QToolTip.hideText()

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt-style name)
        self.close_figure()
        super().closeEvent(event)

    # ------------------------------------------------------------------ #
    # public API expected by TreeWidget
    # ------------------------------------------------------------------ #
    def update(
        self,
        track_df: pd.DataFrame,
        view_direction: str,
        plot_type: str,
        feature: str,
        selected_nodes: list[Any],
        reset_view: bool | None = False,
        allow_flip: bool | None = True,
    ) -> None:
        if self._closed:
            return
        if plot_type == "feature" and (feature is None or feature == ""):
            plot_type = "tree"
        self.view_direction = view_direction
        self.plot_type = plot_type
        self.feature = feature
        self.track_df = track_df if track_df is not None else pd.DataFrame()

        self._rebuild()
        self._configure_axes()
        self.set_selection(selected_nodes, plot_type)
        if reset_view:
            self._reset_view()
        self._update_marker_scale()

    def set_view(
        self,
        view_direction: str,
        plot_type: str,
        reset_view: bool | None = False,
        allow_flip: bool | None = True,
    ) -> None:
        """Store the view direction / plot type (``allow_flip`` is accepted for
        interface parity; orientation follows from ``view_direction``)."""
        self.view_direction = view_direction
        self.plot_type = plot_type
        self._configure_axes()
        if reset_view:
            self._reset_view()

    def _update_viewed_data(self, view_direction: str) -> None:
        """Re-apply positions for the given view direction (used by flip_axes),
        keeping the selection: a flip changes the layout, not what is selected."""
        self.view_direction = view_direction
        selected = [int(self._node_ids[row]) for row in self._selected_rows]
        self._rebuild()
        self._configure_axes()
        self.set_selection(selected, self.plot_type)

    def set_show_hover_info(self, show: bool) -> None:
        """Enable/disable the hover tooltip (Visualization tab toggle)."""
        self.show_hover_info = bool(show)
        if not self.show_hover_info:
            QToolTip.hideText()

    def set_show_track_ids(self, show: bool) -> None:
        """Show/hide track ids along the lane axis (Visualization tab toggle)."""
        show = bool(show)
        if show == self.show_track_ids or self._closed:
            return
        self.show_track_ids = show
        self._configure_axes()

    # ------------------------------------------------------------------ #
    # axes
    # ------------------------------------------------------------------ #
    def _track_id_axis_shown(self) -> bool:
        """The track-id axis is a tree-mode thing: a feature plot's perpendicular axis
        is the feature value, which needs its own numbers."""
        return self.show_track_ids and self.plot_type != "feature"

    def _configure_axes(self) -> None:
        """Time axis on the left (vertical) or bottom (horizontal), numbered; the
        perpendicular axis shows feature values in feature mode, and in tree mode tick
        marks only, or track ids when they are switched on."""
        feature = self.plot_type == "feature"
        lanes = self._lookups.lane_track_ids if self._track_id_axis_shown() else None
        if self.view_direction == "vertical":
            time_axis, lane_axis = self._left_axis, self._bottom_axis
            time_negated, lane_negated = True, False
        else:
            time_axis, lane_axis = self._bottom_axis, self._left_axis
            time_negated, lane_negated = False, not feature
        time_axis.configure(values=True, negate=time_negated, title="Time Point")
        if feature:
            lane_axis.configure(values=True)
        else:
            lane_axis.configure(values=False, negate=lane_negated, lane_labels=lanes)

    # ------------------------------------------------------------------ #
    # rendering
    # ------------------------------------------------------------------ #
    def _rebuild(self) -> None:
        """Full rebuild from ``self.track_df``. Only called when the data or view
        direction changes — not on selection."""
        self._selected_rows = []
        df = self.track_df
        if df is None or df.empty:
            self._node_ids = np.empty(0, dtype=np.int64)
            self._id_to_row = {}
            self._positions = np.empty((0, 3), dtype=np.float32)
            self._base_sizes = np.empty(0, dtype=np.float32)
            self._lookups = shared.empty_lookups()
            self._scatter.clear()
            self._edges.set_edges(np.empty((0, 2)), np.empty((0, 2)), np.empty((0, 4)))
            self._update_node_labels()
            return

        self._node_ids = df["node_id"].to_numpy()
        self._id_to_row = {int(nid): i for i, nid in enumerate(self._node_ids)}
        self._positions = shared.node_positions(
            df, self.view_direction, self.plot_type, self.feature
        )
        self._lookups = shared.track_lookups(df)

        # track_df["color"] is per-row RGBA in 0-255; edges take their child's
        colors = np.stack(df["color"].to_numpy()).astype(np.float32)
        parent_rows, child_rows = shared.edge_rows(df, self._node_ids)
        self._edges.set_edges(
            self._positions[parent_rows, :2],
            self._positions[child_rows, :2],
            colors[child_rows],
        )

        symbols = df["symbol"].to_numpy()
        self._base_sizes = shared.base_sizes(symbols)
        brushes = {}
        for color in map(tuple, colors):
            if color not in brushes:
                brushes[color] = pg.mkBrush(QColor(*map(int, color)))
        self._scatter.setData(
            x=self._positions[:, 0],
            y=self._positions[:, 1],
            symbol=list(symbols),
            brush=[brushes[color] for color in map(tuple, colors)],
            pen=_NO_PEN,
            size=self._base_sizes * self._size_scale,
            data=np.arange(len(self._node_ids)),
        )
        self._update_node_labels()

    # ------------------------------------------------------------------ #
    # zoom-dependent marker size and node-id labels
    # ------------------------------------------------------------------ #
    def _on_view_changed(self, *args) -> None:
        self._update_marker_scale()
        self._update_node_labels()

    def _compute_size_scale(self) -> float:
        """Multiplier on ``_base_sizes`` at the current zoom (``shared.marker_scale``);
        a degenerate view keeps the current scale."""
        if self.plot_type == "feature":
            return 1.0
        x_units, y_units = self._view.viewPixelSize()  # data units per pixel
        if not x_units or not y_units:
            return self._size_scale
        return shared.marker_scale(
            self.plot_type,
            self.view_direction,
            1 / x_units,
            1 / y_units,
            self._base_sizes,
        )

    def _update_marker_scale(self) -> None:
        """Resize the markers if the zoom changed the lane width enough."""
        if self._closed or not len(self._node_ids):
            return
        scale = self._compute_size_scale()
        largest = max(scale, self._size_scale)
        if abs(scale - self._size_scale) < SIZE_SCALE_EPS * largest:
            return
        self._size_scale = scale
        sizes = self._base_sizes * scale
        if self._selected_rows:
            sizes[self._selected_rows] += self._select_bump()
        self._scatter.setSize(sizes)

    def _select_bump(self) -> float:
        return shared.select_bump(self._size_scale)

    def _update_node_labels(self) -> None:
        """Write each on-screen node's id inside its marker, once the markers are drawn
        big enough to hold the text and few enough of them are visible."""
        visible = np.empty(0, dtype=np.int64)
        if (
            not self._closed
            and len(self._node_ids)
            and BASE_SIZE * self._size_scale >= NODE_LABEL_MIN_SIZE
        ):
            cx, cy, w, h = self._current_view()
            visible = np.nonzero(
                shared.rows_in_rect(
                    self._positions, cx - w / 2, cx + w / 2, cy - h / 2, cy + h / 2
                )
            )[0]
            if len(visible) > NODE_LABEL_MAX:
                visible = visible[:0]
        labels = [str(int(self._node_ids[row])) for row in visible]
        font = QFont()
        if labels:
            size = shared.node_label_font_size(BASE_SIZE * self._size_scale, labels)
            font.setPixelSize(max(1, round(size)))
        while len(self._node_labels) < len(labels):
            item = pg.TextItem(color=_NODE_LABEL_COLOR, anchor=(0.5, 0.5))
            self._view.addItem(item, ignoreBounds=True)
            self._node_labels.append(item)
        for i, item in enumerate(self._node_labels):
            if i >= len(labels):
                item.hide()
                continue
            row = int(visible[i])
            item.setText(labels[i])
            item.setFont(font)
            item.setPos(float(self._positions[row, 0]), float(self._positions[row, 1]))
            item.show()

    # ------------------------------------------------------------------ #
    # selection
    # ------------------------------------------------------------------ #
    def set_selection(self, selected_nodes: list[Any], plot_type: str) -> None:
        """Outline the selected nodes in cyan and enlarge them; the fill keeps its
        track colour. Only the rows whose state changes are touched."""
        if self._closed or not len(self._node_ids):
            return
        points = self._scatter.points()
        previous = set(self._selected_rows)
        for row in self._selected_rows:
            points[row].setPen(_NO_PEN)
            points[row].setSize(self._base_sizes[row] * self._size_scale)

        bump = self._select_bump()
        new_rows = []
        for node_id in selected_nodes:
            row = self._id_to_row.get(int(node_id))
            if row is not None:
                points[row].setPen(_SELECT_PEN)
                points[row].setSize(self._base_sizes[row] * self._size_scale + bump)
                new_rows.append(row)
        self._selected_rows = new_rows

        # Fit the view to a multi-node selection only when a node that was *just* added
        # to it is off screen (see tree_plot_fpl.TreePlot.set_selection).
        added = [row for row in new_rows if row not in previous]
        if len(new_rows) > 1 and added and not self._rows_fit(added):
            self._center_on_rows(new_rows)

    # ------------------------------------------------------------------ #
    # picking
    # ------------------------------------------------------------------ #
    def _on_click(self, scatter, points, ev) -> None:
        if not len(points):
            return
        node_id = int(self._node_ids[int(points[0].data())])
        mods = ev.modifiers()
        if mods & (
            Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.MetaModifier
        ):
            self.jump_to_node.emit(node_id)
        else:
            append = bool(mods & Qt.KeyboardModifier.ShiftModifier)
            self.node_clicked.emit(node_id, append)
            self.setFocus()
        ev.accept()

    def _on_hover(self, scatter, points, ev) -> None:
        """Tooltip with the identity of the node under the pointer."""
        if self._closed or not self.show_hover_info:
            return
        if not len(points):
            QToolTip.hideText()
            return
        row = int(points[0].data())
        QToolTip.showText(
            QCursor.pos() + TOOLTIP_OFFSET,
            shared.describe_node(row, self._node_ids, self._lookups),
            self,
        )

    def select_points_in_rect(self, x0, x1, y0, y1) -> None:
        """Box-select: emit all node ids whose positions fall in the rect."""
        if len(self._positions) == 0:
            return
        inside = shared.rows_in_rect(self._positions, x0, x1, y0, y1)
        picked = [int(nid) for nid in self._node_ids[inside]]
        if picked:
            self.nodes_selected.emit(picked, True)

    # ------------------------------------------------------------------ #
    # view
    # ------------------------------------------------------------------ #
    def _current_view(self):
        """The view's (cx, cy, width, height)."""
        (x0, x1), (y0, y1) = self._view.viewRange()
        return (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0

    def _set_view(self, cx, cy, w, h) -> None:
        self._view.setRange(
            xRange=(cx - w / 2, cx + w / 2), yRange=(cy - h / 2, cy + h / 2), padding=0
        )

    def _reset_view(self) -> None:
        """Fit the view to the node positions, filling the canvas."""
        rect = shared.reset_rect(self._positions)
        if rect is not None:
            self._view.setRange(
                xRange=rect[:2], yRange=rect[2:], padding=0, update=True
            )

    def center_on_node(self, node_id: int) -> None:
        if self._closed:
            return
        row = self._id_to_row.get(int(node_id))
        if row is not None:
            self._center_on_rows([row])

    def _rows_fit(self, rows: list[int]) -> bool:
        return shared.rows_fit(self._positions, rows, self._current_view())

    def _center_on_rows(self, rows: list[int]) -> None:
        view = shared.view_fitting_rows(self._positions, rows, self._current_view())
        if view is not None:
            self._set_view(*view)
