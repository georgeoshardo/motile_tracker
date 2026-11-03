# do not put the from __future__ import annotations as it breaks the injection

import contextlib
import weakref
from typing import Any

import fastplotlib as fpl
import numpy as np
import pandas as pd
import pygfx as gfx
from psygnal import Signal
from pygfx.utils.enums import MarkerInt
from qtpy.QtCore import Qt
from qtpy.QtGui import QCursor
from qtpy.QtWidgets import QToolTip, QVBoxLayout, QWidget

from mm_track.data_views.views.tree_view import tree_plot_shared as shared
from mm_track.data_views.views.tree_view.tree_plot_shared import (
    BASE_SIZE,
    NODE_LABEL_MAX,
    NODE_LABEL_MIN_SIZE,
    RIGHT_CLICK_DRAG_PX,
    SELECT_COLOR,
    SELECT_EDGE_WIDTH,
    SIZE_SCALE_EPS,
    TOOLTIP_OFFSET,
)

_EDGE_NONE = np.array(
    [0.0, 0.0, 0.0, 0.0], dtype=np.float32
)  # transparent (unselected)
_NODE_LABEL_COLOR = (0.0, 0.0, 0.0, 1.0)  # black on the track-coloured fill
# Track ids along the lane axis live in the same dock as the perpendicular ruler,
# which they replace when shown.
_DOCK_TRACK_ID_PX = 22.0  # dock margin needed for one row of track-id labels
# node-type glyphs: track_df["symbol"] uses pyqtgraph names; map to pygfx per-vertex
# marker codes. continue="o" (circle), split="t1" (triangle), end="x" (cross).
_MARKER_CODES = {
    "o": int(MarkerInt.circle),
    "t1": int(MarkerInt.triangle_up),
    "x": int(MarkerInt.cross),
}
_DEFAULT_MARKER = int(MarkerInt.circle)
_AXIS_COLOR = (0.6, 0.6, 0.6, 1.0)  # grey axis lines/ticks/labels (like pyqtgraph)
# Axes live in the subplot's docks (separate fixed-width viewports, like pyqtgraph's
# AxisItem cells around a central ViewBox). The main plot area then clips its contents
# at the dock boundary automatically, so panning the tree past the axis makes cells
# disappear under it — material clipping planes have no effect on the marker material.
_DOCK_LEFT_PX = 54.0  # left margin for a vertical axis (line + tick labels)
_DOCK_BOTTOM_PX = 34.0  # bottom margin for a horizontal axis (line + tick labels)
# thin margin for the minimal (tree-mode) perpendicular axis: just tick marks, no
# numbers — keep it small so the tracks reach almost to the marks (no big empty bar)
_DOCK_MINIMAL_PX = 2.0
# world width assigned to a dock's own camera (arbitrary; the dock viewport is narrow)
_DOCK_WORLD = 100.0
# the axis line sits just inside the dock edge adjacent to the plot. It must NOT be
# exactly at the edge (NDC ±1): pygfx's ruler treats a line lying on the clip boundary
# as off-screen and then generates zero tick labels. Insetting also leaves room for the
# inward-pointing tick marks to stay within the dock viewport.
_AXIS_LINE = _DOCK_WORLD * 0.88
# the perpendicular axis in tree mode is drawn "minimal": no axis line, no numbers,
# just tick marks near the *outer* edge of the canvas (matching the old pyqtgraph look).
_AXIS_EDGE = _DOCK_WORLD * 0.10
# Tick labels hang in the margin between the axis line and the dock's outer edge, and a
# dock clips at its viewport, so a label wider than that margin loses its leading
# digits. The default left dock leaves ~37 px, which a feature value big enough to be
# formatted in scientific notation ("1.5e+04", ~44 px) overflows — so grow the dock to
# fit whatever the ruler is currently labelling.
_DOCK_LEFT_MAX_PX = 120.0  # never eat more of the canvas than this, however big
_TICK_LABEL_CHAR_PX = 0.55  # character advance as a fraction of the font size
_TICK_LABEL_MARGIN_PX = 14.0  # ruler's own tick-to-label gap (10) + room to the edge
_DOCK_RESIZE_EPS_PX = 2.0  # a resize relayouts the whole figure; ignore smaller changes


def _blend_alpha(alpha: float) -> float:
    """The alpha that makes white over black look as bright as ``alpha`` does in
    Qt: pygfx blends in linear light, so the result is shown brighter."""
    return ((alpha + 0.055) / 1.055) ** 2.4


def _flag_canvas_closed(canvas_ref: weakref.ref) -> None:
    """Mark a rendercanvas canvas as closed once its C++ widget is gone."""
    canvas = canvas_ref()
    if canvas is not None:
        canvas._is_closed = True


class TreePlot(QWidget):
    """fastplotlib (pygfx/wgpu) canvas for the lineage tree.

    Drop-in replacement for the pyqtgraph ``TreePlot``: exposes the same signals
    (``node_clicked``, ``jump_to_node``, ``nodes_selected``, ``update_selection``)
    and the same public methods (``update``, ``set_selection``, ``set_view``,
    ``_update_viewed_data``, ``center_on_node``, ``setMouseEnabled``) so
    ``TreeWidget`` needs no changes beyond which class it instantiates.
    """

    node_clicked = Signal(Any, bool)  # node_id, append
    jump_to_node = Signal(int)
    nodes_selected = Signal(list, bool)
    update_selection = Signal(bool)  # forward/backward in selection history

    def __init__(self) -> None:
        super().__init__()
        self.setFocusPolicy(Qt.StrongFocus)

        self._closed = False  # set by close_figure(); the canvas is gone after that
        self._scrolled = False  # set by the wheel handler, read via `scrolled`
        self.view_direction = "horizontal"
        self.plot_type = "tree"
        self.feature = None
        self.track_df = pd.DataFrame()

        # per-render caches (stable node_id <-> row index mapping)
        self._node_ids = np.empty(0, dtype=np.int64)
        self._id_to_row: dict[int, int] = {}
        self._positions = np.empty((0, 3), dtype=np.float32)
        self._base_colors = np.empty((0, 4), dtype=np.float32)
        self._base_sizes = np.empty(0, dtype=np.float32)
        self._edge_colors = (
            None  # pygfx per-vertex edge-color buffer (selection outline)
        )
        self._selected_rows: list[int] = []
        self._size_scale = 1.0  # zoom-dependent multiplier on _base_sizes
        # per-row lookups for the hover tooltip and the lane -> track id map behind
        # the optional track-id axis, filled in _rebuild
        self._lookups = shared.empty_lookups()
        self.show_track_ids = False  # toggled from the Visualization tab
        self.show_hover_info = True  # ...as is the hover tooltip

        # shift-drag rectangle (box-select) state
        self._drag_start: tuple[float, float] | None = None
        self._rubber = None
        # the kymograph's place: frames on screen, and the frame under the cursor
        self._time_marks: tuple = (None, None)
        self._band_fill = None  # made on first use, and again after each rebuild
        self._band_edge = None
        self._hover_line = None
        # right-button press position (screen px), to tell a click (reset) from a
        # drag (axis squeeze/zoom, handled natively by the pan/zoom controller)
        self._rpress_xy: tuple | None = None

        # build the figure (Qt auto-detected; bitmap present method is dock-safe)
        # names=[[""]] suppresses the default subplot title (which otherwise shows
        # the grid position "(0, 0)" as a title bar above the plot)
        self._figure = fpl.Figure(
            size=(900, 300),
            names=[[""]],
            canvas_kwargs={"present_method": "bitmap"},
        )
        self._subplot = self._figure[0, 0]
        # font_size 0 collapses the title bar's reserved height to the minimum
        # (fastplotlib always reserves 4 + font_size + 4 px above the plot)
        self._subplot.title.font_size = 0
        self._scatter = None
        self._edges = None

        # Axes: fastplotlib's default Axes overlay draws rulers at the *data* bbox
        # edges (so the y-axis floats inset from the canvas and moves with the data)
        # plus xy grids. The old pyqtgraph tree had no grid, no x-axis (in tree mode),
        # and a grey time axis in a fixed left margin that clipped the tree.
        # Reproduce that with pygfx Rulers living in the subplot's docks (separate
        # fixed-width viewports); the main plot area then clips at the dock boundary.
        self._subplot.axes.visible = False
        self._dock_left = self._subplot.docks["left"]
        self._dock_bottom = self._subplot.docks["bottom"]
        self._dock_left.camera.maintain_aspect = False
        self._dock_bottom.camera.maintain_aspect = False
        # the tree fills a wide canvas non-uniformly — never preserve aspect on the
        # MAIN camera (its default is True), else show_rect/reset letterboxes and the
        # tree won't fill. Set persistently so pan/zoom/resize can't revert it.
        self._subplot.camera.maintain_aspect = False
        self._ruler_time = gfx.Ruler(tick_side="left")
        self._ruler_feature = gfx.Ruler(tick_side="left")
        for ruler in (self._ruler_time, self._ruler_feature):
            ruler.color = _AXIS_COLOR
            ruler.tick_size = 6
        # rotated "Time Point" axis title, lives in the time-axis dock
        self._time_label = gfx.Text(
            text="Time Point",
            font_size=13,
            screen_space=True,
            anchor="middle-center",
            material=gfx.TextMaterial(color=_AXIS_COLOR),
        )
        # track ids along the lane axis: one MultiText re-used across frames, living in
        # whichever dock currently holds that axis (see _configure_docks)
        self._track_labels = gfx.MultiText(
            font_size=11,
            screen_space=True,
            anchor="middle-center",
            material=gfx.TextMaterial(color=_AXIS_COLOR),
        )
        # node ids drawn inside the markers once they are big enough (main scene)
        self._node_labels = gfx.MultiText(
            font_size=10,
            screen_space=True,
            anchor="middle-center",
            material=gfx.TextMaterial(color=_NODE_LABEL_COLOR),
        )
        self._node_labels.visible = False
        self._configure_docks()

        canvas_widget = self._figure.show()
        # The render canvas (and its inner QRenderWidget child) grab keyboard focus
        # and swallow key events. Force NoFocus on the whole subtree so Q / X / Y
        # bubble up to TreeWidget.keyPressEvent; TreePlot itself keeps StrongFocus
        # and grabs focus on click (see _on_canvas_pointer_down).
        canvas_widget.setFocusPolicy(Qt.NoFocus)
        for child in canvas_widget.findChildren(QWidget):
            child.setFocusPolicy(Qt.NoFocus)
        # keep the canvas tall enough that the subplot viewport height never goes
        # negative during dock layout (avoids wgpu "Viewport size < zero" draw errors)
        canvas_widget.setMinimumHeight(120)
        # last-resort unregistration for every teardown path we don't see coming
        # (see _flag_canvas_closed); close_figure() is the orderly route.
        render_widget = getattr(canvas_widget, "_subwidget", canvas_widget)
        render_widget_ref = weakref.ref(render_widget)
        render_widget.destroyed.connect(lambda: _flag_canvas_closed(render_widget_ref))

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(canvas_widget)
        self.setLayout(layout)

        # keep the rulers pinned to the viewport edges every frame (they follow the
        # camera as the user pans/zooms), and rescale the markers with the zoom level
        self._animations = (
            self._update_rulers,
            self._update_marker_scale,
            self._update_node_labels,
        )
        self._subplot.add_animations(*self._animations)

        # canvas mouse handling: right-click reset + shift-drag box-select, plus
        # scroll bookkeeping so a zoom modifier key can be told apart from a plain
        # keyboard shortcut (see `scrolled`)
        self._canvas_handlers = (
            (self._on_canvas_pointer_down, "pointer_down"),
            (self._on_canvas_pointer_move, "pointer_move"),
            (self._on_canvas_pointer_up, "pointer_up"),
            (self._on_canvas_wheel, "wheel"),
        )
        for handler, event_type in self._canvas_handlers:
            self._figure.renderer.add_event_handler(handler, event_type)

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

    def _on_canvas_wheel(self, ev) -> None:
        """Record that the user scrolled over the canvas."""
        self._scrolled = True

    def setMouseEnabled(self, x: bool, y: bool) -> None:  # noqa: N802 (Qt-style name)
        """Restrict zoom/pan to the given axes (X or Y key held). Reconfigures the
        subplot's own PanZoomController so it stays correctly event-registered."""
        cam = self._subplot.camera
        ctrl = self._subplot.controller
        with contextlib.suppress(Exception):
            ctrl.remove_camera(cam)
            if x and y:
                ctrl.add_camera(cam)
            elif x:
                ctrl.add_camera(cam, include_state={"x", "width"})
            elif y:
                ctrl.add_camera(cam, include_state={"y", "height"})

    # ------------------------------------------------------------------ #
    # teardown
    # ------------------------------------------------------------------ #
    def close_figure(self) -> None:
        """Close the fastplotlib figure while its Qt widgets are still alive. Closing
        the figure ourselves, before the widgets are deleted, unregisters the canvas and
        releases the wgpu/pygfx resources, so it does not crash with "wrapped C/C++
        object of type QRenderWidget has been deleted".
        """
        if self._closed:
            return
        self._closed = True
        for animation in self._animations:
            with contextlib.suppress(Exception):
                self._subplot.remove_animation(animation)
        for handler, event_type in self._canvas_handlers:
            with contextlib.suppress(Exception):
                self._figure.renderer.remove_event_handler(handler, event_type)
        with contextlib.suppress(Exception):
            self._subplot.clear()
        self._scatter = None
        self._edges = None
        self._rubber = None
        self._edge_colors = None
        self._positions = np.empty((0, 3), dtype=np.float32)
        self._base_colors = np.empty((0, 4), dtype=np.float32)
        self._base_sizes = np.empty(0, dtype=np.float32)
        self._node_ids = np.empty(0, dtype=np.int64)
        self._id_to_row = {}
        self._selected_rows = []
        self._lookups = shared.empty_lookups()
        QToolTip.hideText()
        with contextlib.suppress(Exception):
            self._figure.close()

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

        self._configure_docks()
        self._rebuild()
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
        """Store the view direction / plot type. Axis rulers update themselves each
        frame in ``_update_rulers``; ``allow_flip`` is accepted for interface parity
        with the old pyqtgraph TreePlot but is not needed here (orientation is derived
        from ``view_direction`` in ``_compute_positions``)."""
        self.view_direction = view_direction
        self.plot_type = plot_type
        self._configure_docks()
        if reset_view:
            self._reset_view()

    def _update_viewed_data(self, view_direction: str) -> None:
        """Re-apply positions for the given view direction (used by flip_axes).

        The rebuild drops the scatter and its selection buffers, so carry the selected
        node ids across it — a flip changes the layout, not what is selected.
        """
        self.view_direction = view_direction
        self._configure_docks()
        selected = [int(self._node_ids[row]) for row in self._selected_rows]
        self._rebuild()
        self.set_selection(selected, self.plot_type)

    # ------------------------------------------------------------------ #
    # axis rulers (hosted in dock viewports; synced to the main camera)
    # ------------------------------------------------------------------ #
    def _configure_docks(self) -> None:
        """Assign the two rulers to their docks and size both docks. Both axes are
        always shown (the perpendicular axis shows tick marks even in tree mode; its
        number labels are toggled per-mode in ``_update_rulers``). Time axis: left
        dock (vertical) / bottom dock (horizontal); feature axis: the other dock."""
        for obj in (
            self._ruler_time,
            self._ruler_feature,
            self._time_label,
            self._track_labels,
        ):
            if obj.parent is not None:
                obj.parent.remove(obj)

        # the perpendicular axis in tree mode is either a thin strip of tick marks or,
        # with the track-id axis switched on, a row of track ids that replaces them
        minimal = _DOCK_TRACK_ID_PX if self._track_id_axis_shown() else _DOCK_MINIMAL_PX
        # a numbered left dock may have been grown to fit wide tick labels: keep that
        # width rather than snapping back to the base and re-growing a frame later
        # (_fit_left_dock shrinks it again as soon as the labels get shorter)
        numbered_left = max(_DOCK_LEFT_PX, self._dock_left.size)
        if self.view_direction == "vertical":
            self._dock_left.scene.add(self._ruler_time, self._time_label)
            self._dock_bottom.scene.add(self._ruler_feature, self._track_labels)
            self._dock_left.size = numbered_left
            self._dock_bottom.size = (
                _DOCK_BOTTOM_PX if self.plot_type == "feature" else minimal
            )
        else:  # horizontal: time along the bottom
            self._dock_bottom.scene.add(self._ruler_time, self._time_label)
            self._dock_left.scene.add(self._ruler_feature, self._track_labels)
            self._dock_bottom.size = _DOCK_BOTTOM_PX
            self._dock_left.size = (
                numbered_left if self.plot_type == "feature" else minimal
            )

    def _track_id_axis_shown(self) -> bool:
        """The track-id axis is a tree-mode thing: a feature plot's perpendicular axis
        is the feature value, which needs its own numbers."""
        return self.show_track_ids and self.plot_type != "feature"

    def set_show_hover_info(self, show: bool) -> None:
        """Enable/disable the hover tooltip (Visualization tab toggle)."""
        self.show_hover_info = bool(show)
        if not self.show_hover_info:
            QToolTip.hideText()  # a tooltip already on screen would otherwise linger

    def set_show_track_ids(self, show: bool) -> None:
        """Show/hide track ids along the lane axis (Visualization tab toggle)."""
        show = bool(show)
        if show == self.show_track_ids or self._closed:
            return
        self.show_track_ids = show
        self._configure_docks()  # the dock has to grow/shrink to fit the labels
        with contextlib.suppress(Exception):
            self._figure.canvas.request_draw()

    def _update_rulers(self) -> None:
        """Each frame, sync both rulers' dock cameras to the main camera's matching
        range and lay each ruler along its dock edge next to the plot. The dock is a
        separate viewport, so the tree is clipped at the axis for free. Number labels
        show on the time axis always, and on the feature axis only in feature mode
        (tree mode shows the perpendicular axis as tick marks only, like pyqtgraph)."""
        if self._scatter is None:
            self._ruler_time.visible = False
            self._ruler_feature.visible = False
            self._time_label.visible = False
            self._track_labels.visible = False
            return

        state = self._subplot.camera.get_state()
        px, py = state["position"][0], state["position"][1]
        w, h = state["width"], state["height"]
        left, right = px - w / 2, px + w / 2
        bottom, top = py - h / 2, py + h / 2
        # tree mode: the perpendicular axis is "minimal" (edge tick marks only)
        minimal = self.plot_type != "feature"

        if self.view_direction == "vertical":
            self._sync_left(self._ruler_time, bottom, top, False, self._time_label)
            self._sync_bottom(self._ruler_feature, left, right, minimal)
        else:
            self._sync_bottom(self._ruler_time, left, right, False, self._time_label)
            # a numeric feature axis increases upward (not negated like tree lanes)
            self._sync_left(
                self._ruler_feature,
                bottom,
                top,
                minimal,
                negated=self.plot_type != "feature",
            )
        # runs last: with track ids on, they take the place of the perpendicular
        # ruler's tick marks, which _sync_* has just switched back on
        self._update_track_labels(left, right, bottom, top)

    def _sync_left(self, ruler, bottom, top, minimal, label=None, negated=True) -> None:
        """Lay a vertical ruler in the left dock. Normal: axis line just inside the
        right edge (next to the plot), tick marks toward the plot, numbers in the
        margin to the left. Minimal (tree-mode perpendicular axis): no line, no
        numbers, just tick marks near the far-left edge of the canvas.

        The ruler's value always increases from ``start_pos`` toward ``end_pos``.
        ``negated=True`` (time axis / tree lanes, whose world y = -value) lays it
        top->bottom so values increase downward (parent-at-top). ``negated=False``
        (a numeric feature axis, whose world y = +value) lays it bottom->top so
        feature values increase upward."""
        dock = self._dock_left
        prepared = self._prepare_ruler(
            dock, ruler, minimal, (0, _DOCK_WORLD, bottom, top)
        )
        if prepared is None:
            return
        dv, axis = prepared
        # tick_side is relative to the ruler's start->end direction. For the downward
        # ruler "right" puts marks toward the plot and numbers in the left margin; for
        # the upward ruler that same margin side is "left" (else numbers render into the
        # clipped plot area and disappear).
        if negated:  # world y = -value: lay top->bottom, value increases downward
            ruler.tick_side = "right"
            ruler.start_value = -top
            ruler.start_pos = (axis, top, 0)
            ruler.end_pos = (axis, bottom, 0)
        else:  # world y = +value: lay bottom->top, value increases upward
            ruler.tick_side = "left"
            ruler.start_value = bottom
            ruler.start_pos = (axis, bottom, 0)
            ruler.end_pos = (axis, top, 0)
        stats = self._finish_ruler(dock, ruler, dv, minimal)
        if not minimal:
            self._fit_left_dock(ruler, stats["tick_values"])
        if label is not None:
            label.visible = not minimal
            label.local.rotation = (0.0, 0.0, 0.70710677, 0.70710677)  # +90°
            label.local.position = (_DOCK_WORLD * 0.14, (top + bottom) / 2, 0)

    def _prepare_ruler(self, dock, ruler, minimal, rect):
        """Set up a dock to draw one of the two rulers: frame its camera on ``rect``
        (the visible range along the ruler, across the dock's own world width) and pick
        where the axis sits inside it — next to the plot for a numbered axis, at the
        canvas edge for a minimal one.

        Returns ``(viewport_size, axis_position)``, or None if the dock's viewport is
        transiently degenerate, in which case the caller must leave the ruler alone
        (keep the last-good geometry rather than hide it).
        """
        dv = dock.viewport.logical_size
        if dv[0] < 1 or dv[1] < 1:
            return None
        dock.camera.show_rect(*rect, depth=1)
        ruler.visible = True
        return dv, (_AXIS_EDGE if minimal else _AXIS_LINE)

    def _finish_ruler(self, dock, ruler, dv, minimal):
        """Lay out the ruler's ticks for the range its dock camera now shows, and drop
        the parts a minimal axis does without (its line and its numbers). Returns the
        stats dict from ``Ruler.update`` (``tick_step`` and ``tick_values``)."""
        stats = ruler.update(dock.camera, dv)
        ruler._line.visible = not minimal
        ruler._text.visible = not minimal
        return stats

    def _fit_left_dock(self, ruler, tick_values) -> None:
        """Widen the left dock when its tick labels no longer fit in it (and let it
        shrink back when they do). Only the vertical rulers need this: their labels
        grow sideways, into the dock's narrow dimension.

        Resizing here cannot feed back on itself — the dock's width does not affect the
        vertical span the ruler labels, so the labels stay put and the size settles
        after one frame.
        """
        fmt = ruler.tick_format
        if not isinstance(fmt, str) or fmt == "km":
            return  # a custom formatter: no cheap way to know what it will produce
        widest = max((len(format(v, fmt)) for v in tick_values), default=0)
        if not widest:
            return
        label_px = widest * _TICK_LABEL_CHAR_PX * ruler.text.font_size
        # the labels sit left of the axis line, which is _AXIS_LINE/_DOCK_WORLD of the
        # way across the dock — so that fraction of the dock has to hold them
        needed = (label_px + _TICK_LABEL_MARGIN_PX) / (_AXIS_LINE / _DOCK_WORLD)
        needed = float(np.clip(needed, _DOCK_LEFT_PX, _DOCK_LEFT_MAX_PX))
        if abs(needed - self._dock_left.size) > _DOCK_RESIZE_EPS_PX:
            self._dock_left.size = needed
            self._figure.canvas.request_draw()

    def _sync_bottom(self, ruler, left, right, minimal, label=None) -> None:
        """Lay a horizontal ruler in the bottom dock. Normal: axis line just inside
        the top edge (next to the plot), tick marks toward the plot, numbers in the
        margin below. Minimal (tree-mode perpendicular axis): no line, no numbers,
        just tick marks near the bottom edge of the canvas."""
        dock = self._dock_bottom
        prepared = self._prepare_ruler(
            dock, ruler, minimal, (left, right, 0, _DOCK_WORLD)
        )
        if prepared is None:
            return
        dv, axis = prepared
        ruler.tick_side = "right"  # marks toward plot, numbers extend down into margin
        ruler.start_value = left
        ruler.start_pos = (left, axis, 0)
        ruler.end_pos = (right, axis, 0)
        self._finish_ruler(dock, ruler, dv, minimal)
        if label is not None:
            label.visible = not minimal
            label.local.rotation = (0.0, 0.0, 0.0, 1.0)  # horizontal
            label.local.position = ((left + right) / 2, _DOCK_WORLD * 0.14, 0)

    # ------------------------------------------------------------------ #
    # track-id axis (optional row of labels along the lane axis)
    # ------------------------------------------------------------------ #
    def _update_track_labels(self, left, right, bottom, top) -> None:
        """Lay one track id per visible lane in the perpendicular axis' dock, thinned
        out so neighbouring labels don't overlap (see ``shared.labelled_lanes``)."""
        lane_track_ids = self._lookups.lane_track_ids
        if not self._track_id_axis_shown() or not lane_track_ids:
            self._track_labels.visible = False
            return

        # the dock holding the lane axis, and the lane range it currently spans. In
        # vertical view lanes run along x; flipped, they run along y as -x_axis_pos.
        if self.view_direction == "vertical":
            dock, low, high = self._dock_bottom, left, right
            span_px = dock.viewport.logical_size[0]
        else:
            dock, low, high = self._dock_left, -top, -bottom
            span_px = dock.viewport.logical_size[1]
        lanes = shared.labelled_lanes(low, high, span_px, lane_track_ids)
        if not lanes:
            self._track_labels.visible = False
            return

        # the ruler in this dock is the tick strip these labels replace
        self._ruler_feature.visible = False
        self._track_labels.visible = True
        self._ensure_text_blocks(self._track_labels, len(lanes))
        for i in range(self._track_labels.get_text_block_count()):
            block = self._track_labels.get_text_block(i)
            if i >= len(lanes):
                block.set_text("")
                continue
            lane = lanes[i]
            block.set_text(str(lane_track_ids[lane]))
            if self.view_direction == "vertical":
                block.set_position(lane, _DOCK_WORLD * 0.5)
            else:
                block.set_position(_DOCK_WORLD * 0.5, -lane)

    @staticmethod
    def _ensure_text_blocks(multi_text, count: int) -> None:
        """Grow a MultiText to at least ``count`` blocks. Blocks are never removed —
        they are reused across frames, and surplus ones are blanked by the caller."""
        missing = count - multi_text.get_text_block_count()
        if missing > 0:
            multi_text.create_text_blocks(missing)

    # ------------------------------------------------------------------ #
    # node-id labels (drawn inside the markers once they are big enough)
    # ------------------------------------------------------------------ #
    def _update_node_labels(self) -> None:
        """Write each on-screen node's id inside its marker, once the markers are drawn
        big enough to hold the text and few enough of them are visible."""
        if self._closed or self._scatter is None:
            return
        if BASE_SIZE * self._size_scale < NODE_LABEL_MIN_SIZE:
            self._node_labels.visible = False
            return

        state = self._subplot.camera.get_state()
        px, py = state["position"][0], state["position"][1]
        w, h = state["width"], state["height"]
        xs, ys = self._positions[:, 0], self._positions[:, 1]
        visible = np.nonzero(
            (xs >= px - w / 2)
            & (xs <= px + w / 2)
            & (ys >= py - h / 2)
            & (ys <= py + h / 2)
        )[0]
        if len(visible) == 0 or len(visible) > NODE_LABEL_MAX:
            self._node_labels.visible = False
            return

        labels = [str(int(self._node_ids[row])) for row in visible]
        self._node_labels.visible = True
        self._node_labels.font_size = shared.node_label_font_size(
            BASE_SIZE * self._size_scale, labels
        )
        self._ensure_text_blocks(self._node_labels, len(visible))
        for i in range(self._node_labels.get_text_block_count()):
            block = self._node_labels.get_text_block(i)
            if i >= len(visible):
                block.set_text("")
                continue
            row = int(visible[i])
            block.set_text(labels[i])
            block.set_position(float(xs[row]), float(ys[row]))

    # ------------------------------------------------------------------ #
    # rendering
    # ------------------------------------------------------------------ #
    def _compute_positions(self, df: pd.DataFrame) -> np.ndarray:
        return shared.node_positions(
            df, self.view_direction, self.plot_type, self.feature
        )

    def _rebuild(self) -> None:
        """Full rebuild of the scene from ``self.track_df``. Only called when the
        data or view direction changes — not on selection."""
        self._subplot.clear()
        self._scatter = None
        self._edges = None
        self._edge_colors = None
        self._rubber = None  # cleared with the scene; recreated lazily on next drag
        self._band_fill = self._band_edge = self._hover_line = None  # ...as are these
        self._selected_rows = []

        df = self.track_df
        if df is None or df.empty:
            self._node_ids = np.empty(0, dtype=np.int64)
            self._id_to_row = {}
            self._base_sizes = np.empty(0, dtype=np.float32)
            # the per-row arrays describe the data that just went away; leaving
            # them in place makes _reset_view frame the camera on coordinates
            # that are no longer drawn
            self._positions = np.empty((0, 3), dtype=np.float32)
            self._base_colors = np.empty((0, 4), dtype=np.float32)
            self._lookups = shared.empty_lookups()
            self._draw_time_marks()
            return

        self._node_ids = df["node_id"].to_numpy()
        self._id_to_row = {int(nid): i for i, nid in enumerate(self._node_ids)}
        self._positions = self._compute_positions(df)
        self._lookups = shared.track_lookups(df)

        # colors: track_df["color"] is per-row RGBA in 0-255 -> float 0-1
        colors = np.stack(df["color"].to_numpy()).astype(np.float32) / 255.0
        self._base_colors = colors.copy()

        # edges under nodes: ALL parent->child edges in ONE NaN-separated line
        # (single draw call / buffer upload — far cheaper than a per-edge collection)
        edge_pos, edge_cols = self._build_edges(df, colors)
        if edge_pos is not None:
            self._edges = self._subplot.add_line(
                data=edge_pos,
                colors=edge_cols,
                thickness=1.5,
                name="edges",
            )

        # nodes: single scatter with per-vertex marker shapes (node-type glyphs).
        # A single scatter keeps selection surgical; per-vertex markers are enabled by
        # setting the material's marker_mode to "vertex" + a geometry "markers" int
        # buffer (pygfx's per-vertex marker path), so we avoid finn's 3-scatter split.
        symbols = df["symbol"].to_numpy()
        self._base_sizes = shared.base_sizes(symbols)
        # keep the scale from the previous frame here: the camera has not been framed
        # for this data yet, so it would be measured against a stale view. It is
        # corrected right after (_reset_view, or the per-frame _update_marker_scale).
        self._scatter = self._subplot.add_scatter(
            data=self._positions,
            colors=colors,
            sizes=self._base_sizes * self._size_scale,
            markers="circle",
            name="nodes",
        )
        codes = np.array(
            [_MARKER_CODES.get(s, _DEFAULT_MARKER) for s in symbols], dtype=np.int32
        )
        wo = self._scatter.world_object
        wo.material.marker_mode = "vertex"
        wo.geometry.markers = gfx.Buffer(codes)
        # Selection is shown as an OUTLINE (like the old pyqtgraph tree): a uniform
        # edge that is transparent for unselected nodes and cyan for selected ones, so
        # a selected node keeps its track-color fill. Per-vertex edge colors need
        # edge_color_mode="vertex" + a geometry "edge_colors" buffer; edge_width is
        # uniform (pygfx has no per-vertex width), transparent nodes just show no edge.
        wo.material.edge_color_mode = "vertex"
        wo.material.edge_width = SELECT_EDGE_WIDTH
        edge_colors = np.zeros((len(self._node_ids), 4), dtype=np.float32)
        wo.geometry.edge_colors = gfx.Buffer(edge_colors)
        self._edge_colors = wo.geometry.edge_colors
        self._scatter.add_event_handler(self._on_click, "pointer_down")
        self._scatter.add_event_handler(self._on_hover, "pointer_move")
        self._scatter.add_event_handler(self._on_hover_leave, "pointer_leave")
        # node-id labels are raw pygfx, so _subplot.clear() above dropped them
        self._subplot.scene.add(self._node_labels)
        self._draw_time_marks()

    def _build_edges(self, df: pd.DataFrame, colors: np.ndarray):
        """Vectorized edge build for a single NaN-separated ``gfx.Line``.

        Returns ``(positions, colors)`` where positions is ``(3E, 3)`` laid out as
        ``[parent, child, NaN]`` per edge (the NaN row breaks the polyline into
        independent segments), and colors matches. Returns ``(None, None)`` if
        there are no edges.
        """
        par_rows, child_rows = shared.edge_rows(df, self._node_ids)
        if not len(child_rows):
            return None, None
        e = len(child_rows)

        edge_pos = np.full((3 * e, 3), np.nan, dtype=np.float32)
        edge_pos[0::3] = self._positions[par_rows]
        edge_pos[1::3] = self._positions[child_rows]

        edge_cols = np.ones((3 * e, 4), dtype=np.float32)
        c = colors[child_rows]
        edge_cols[0::3] = c
        edge_cols[1::3] = c
        edge_cols[2::3] = c
        return edge_pos, edge_cols

    # ------------------------------------------------------------------ #
    # zoom-dependent marker size
    # ------------------------------------------------------------------ #
    def _compute_size_scale(self) -> float:
        """Multiplier on ``_base_sizes`` at the current zoom (``shared.marker_scale``);
        a transiently degenerate viewport/camera keeps the current scale."""
        if self.plot_type == "feature":
            return 1.0
        with contextlib.suppress(Exception):
            viewport_size = self._subplot.viewport.logical_size
            state = self._subplot.camera.get_state()
            width, height = state["width"], state["height"]
            if viewport_size[0] < 1 or viewport_size[1] < 1 or not width or not height:
                return self._size_scale
            # one unit on each axis, in pixels
            return shared.marker_scale(
                self.plot_type,
                self.view_direction,
                viewport_size[0] / width,
                viewport_size[1] / height,
                self._base_sizes,
            )
        return self._size_scale

    def _update_marker_scale(self) -> None:
        """Per frame: resize the markers if the zoom changed the lane width enough."""
        if self._scatter is None or self._closed:
            return
        scale = self._compute_size_scale()
        largest = max(scale, self._size_scale)
        if abs(scale - self._size_scale) < SIZE_SCALE_EPS * largest:
            return
        self._size_scale = scale
        sizes = self._base_sizes * scale
        if self._selected_rows:
            sizes[self._selected_rows] += self._select_bump()
        self._scatter.sizes = sizes

    def _select_bump(self) -> float:
        return shared.select_bump(self._size_scale)

    # ------------------------------------------------------------------ #
    # selection (surgical: only touch changed rows)
    # ------------------------------------------------------------------ #
    def set_selection(self, selected_nodes: list[Any], plot_type: str) -> None:
        if self._scatter is None or self._closed:
            return
        # Selection = a cyan outline + a size bump; the fill keeps its track color so
        # you can still tell which track a selected node belongs to (like the old tree).
        # Surgical: only touch changed rows' edge-color buffer + size.
        # restore previously selected rows: clear the outline and the size bump
        previous = set(self._selected_rows)
        for row in self._selected_rows:
            self._set_edge_color(row, _EDGE_NONE)
            self._scatter.sizes[row] = self._base_sizes[row] * self._size_scale

        bump = self._select_bump()
        new_rows = []
        for node_id in selected_nodes:
            row = self._id_to_row.get(int(node_id))
            if row is not None:
                self._set_edge_color(row, SELECT_COLOR)
                self._scatter.sizes[row] = (
                    self._base_sizes[row] * self._size_scale + bump
                )
                new_rows.append(row)
        self._selected_rows = new_rows

        # Fit the view to a multi-node selection only when a node that was *just* added
        # to it is off screen. Shift-clicking a node in the tree can only hit a node
        # that is already in view, and zooming out to also fit the rest of the selection
        # would throw away the zoom the user picked it out in — which makes picking a
        # run of nodes by hand painful. A selection made elsewhere (the two endpoints of
        # a broken edge, the table, the napari layers) can land outside the view though,
        # and then it still has to be brought in.
        added = [row for row in new_rows if row not in previous]
        if len(new_rows) > 1 and added and not self._rows_fit(added):
            self._center_on_rows(new_rows)

    def _set_edge_color(self, row: int, rgba: np.ndarray) -> None:
        """Set one node's per-vertex outline color and upload just that element."""
        if self._edge_colors is None:
            return
        self._edge_colors.data[row] = rgba
        self._edge_colors.update_range(row, 1)

    # ------------------------------------------------------------------ #
    # picking
    # ------------------------------------------------------------------ #
    def _on_canvas_pointer_down(self, ev) -> None:
        # any click on the canvas gives TreePlot keyboard focus so Q/X/Y work
        self.setFocus()
        button = getattr(ev, "button", None)
        # right button (2): a plain click resets the view; a drag squeezes/zooms the
        # x or y axis (the controller's native mouse2 "zoom" drag). Only record the
        # press here — resetting now would override the drag-zoom. pointer_up decides
        # click vs drag. Do NOT disable the controller, so the drag-zoom still runs.
        if button == 2:
            self._rpress_xy = (getattr(ev, "x", None), getattr(ev, "y", None))
            return
        # mouse side buttons -> step through selection history (like browser back/fwd).
        # back (4) = previous selection, forward (5) = next. TreeWidget connects
        # update_selection -> tracks_viewer.select_node_set_from_history(previous=...).
        if button == 4:
            self.update_selection.emit(True)
            return
        if button == 5:
            self.update_selection.emit(False)
            return
        # left button + Shift -> start a rubber-band box-select (like the old tree).
        # Disable the pan controller for the duration so the drag draws a box
        # instead of panning.
        if button == 1 and "Shift" in set(getattr(ev, "modifiers", ()) or ()):
            world = self._subplot.map_screen_to_world(ev, allow_outside=True)
            if world is not None:
                self._drag_start = (float(world[0]), float(world[1]))
                self._subplot.controller.enabled = False

    def _on_canvas_pointer_move(self, ev) -> None:
        if self._drag_start is None:
            return
        world = self._subplot.map_screen_to_world(ev, allow_outside=True)
        if world is not None:
            self._update_rubber(self._drag_start, (float(world[0]), float(world[1])))

    def _on_canvas_pointer_up(self, ev) -> None:
        # right-button release: reset only if it was a click (moved <= threshold). A
        # larger movement was a drag-zoom already applied by the controller.
        if self._rpress_xy is not None:
            x0, y0 = self._rpress_xy
            self._rpress_xy = None
            x1, y1 = getattr(ev, "x", None), getattr(ev, "y", None)
            moved = abs(x1 - x0) + abs(y1 - y0) if None not in (x0, y0, x1, y1) else 0.0
            if moved <= RIGHT_CLICK_DRAG_PX:
                self._reset_view()
            return

        if self._drag_start is None:
            return
        world = self._subplot.map_screen_to_world(ev, allow_outside=True)
        x0, y0 = self._drag_start
        self._drag_start = None
        self._clear_rubber()
        self._subplot.controller.enabled = True
        if world is None:
            return
        x1, y1 = float(world[0]), float(world[1])
        # ignore a plain shift-click (no drag) — that's handled as node append
        if abs(x1 - x0) > 1e-9 or abs(y1 - y0) > 1e-9:
            self.select_points_in_rect(x0, x1, y0, y1)

    def set_time_marks(
        self, frames: tuple[float, float] | None, hover: int | None
    ) -> None:
        """Show the kymograph's place: a band over the frames it shows and a line at
        the frame under its cursor (see ``tree_plot_shared.time_marks``). When the
        frames shown change, pan along time to keep the band in view."""
        moved = frames != self._time_marks[0]
        self._time_marks = (frames, hover)
        if self._closed:
            return
        self._draw_time_marks()
        if moved:
            band, _ = shared.time_marks(
                self._positions, self.view_direction, frames, None
            )
            with contextlib.suppress(Exception):  # the camera can't be read yet
                view = shared.view_following_band(
                    self._camera_view(), band, self.view_direction
                )
                if view is not None:
                    self._set_camera_view(view)

    def _ensure_time_marks(self) -> None:
        if self._band_fill is not None:
            return
        # under the edges and nodes, and never picked: clicks go to the nodes
        self._band_fill = gfx.Mesh(
            gfx.plane_geometry(1, 1),
            gfx.MeshBasicMaterial(
                color=(
                    *shared.VIEW_BAND_FILL[:3],
                    _blend_alpha(shared.VIEW_BAND_FILL[3]),
                ),
                alpha_mode="blend",
                pick_write=False,
            ),
        )
        self._band_fill.render_order = -3
        self._subplot.scene.add(self._band_fill)
        self._band_edge = self._subplot.add_line(
            np.zeros((5, 3), dtype=np.float32),
            colors=shared.VIEW_BAND_EDGE,
            thickness=1.0,
            name="_view_band",
        )
        self._hover_line = self._subplot.add_line(
            np.zeros((2, 3), dtype=np.float32),
            colors=shared.HOVER_LINE_COLOR,
            thickness=1.0,
            name="_hover_line",
        )
        for graphic, order in ((self._band_edge, -2), (self._hover_line, -1)):
            graphic.world_object.render_order = order
            graphic.world_object.material.pick_write = False

    def _draw_time_marks(self) -> None:
        band, line = shared.time_marks(
            self._positions, self.view_direction, *self._time_marks
        )
        if band is None and line is None and self._band_fill is None:
            return
        self._ensure_time_marks()
        if band is not None:
            x0, y0, x1, y1 = band
            self._band_fill.local.position = ((x0 + x1) / 2, (y0 + y1) / 2, 0)
            self._band_fill.local.scale = (x1 - x0, y1 - y0, 1)
            self._band_edge.data[:] = np.array(
                [[x0, y0, 0], [x1, y0, 0], [x1, y1, 0], [x0, y1, 0], [x0, y0, 0]],
                dtype=np.float32,
            )
        self._band_fill.visible = self._band_edge.visible = band is not None
        if line is not None:
            self._hover_line.data[:] = np.column_stack(
                [line, np.zeros(2, dtype=np.float32)]
            )
        self._hover_line.visible = line is not None
        self._figure.canvas.request_draw()

    def _ensure_rubber(self):
        if self._rubber is None:
            self._rubber = self._subplot.add_line(
                np.zeros((5, 3), dtype=np.float32),
                colors="yellow",
                thickness=1.0,
                name="_rubber",
            )
            self._rubber.visible = False
        return self._rubber

    def _update_rubber(self, start: tuple, cur: tuple) -> None:
        x0, y0 = start
        x1, y1 = cur
        r = self._ensure_rubber()
        r.data[:] = np.array(
            [[x0, y0, 0], [x1, y0, 0], [x1, y1, 0], [x0, y1, 0], [x0, y0, 0]],
            dtype=np.float32,
        )
        r.visible = True
        self._figure.canvas.request_draw()

    def _clear_rubber(self) -> None:
        if self._rubber is not None:
            self._rubber.visible = False
            self._figure.canvas.request_draw()

    def _reset_view(self) -> None:
        """Fit the view to the node positions, filling the canvas; in tree mode, at
        most as many frames as fit at TREE_PX_PER_FRAME, around the kymograph's.

        Framed directly from the scatter data via ``camera.show_rect`` rather than
        ``auto_scale``: auto_scale unions the bounds of *all* scene objects (the
        rulers, the NaN-separated edge line, the idle rubber rectangle), so it can't
        reliably tighten to the tree — e.g. when zoomed out, the rulers sit at the
        wide viewport edges. Framing from ``self._positions`` ignores everything else.
        maintain_aspect=False fills the wide canvas; request_draw is required because
        a camera change doesn't repaint on its own between interactions."""
        rect = shared.reset_rect(self._positions)
        if rect is None:
            return
        # in tree mode, at most as many frames as fit at TREE_PX_PER_FRAME, around
        # the kymograph's
        with contextlib.suppress(Exception):
            width, height = self._subplot.viewport.logical_size
            length = height if self.view_direction == "vertical" else width
            if self.plot_type == "tree" and length > 50:
                frames = self._time_marks[0]
                focus = None if frames is None else (frames[0] + frames[1]) / 2
                rect = shared.limit_time_span(
                    rect, self.view_direction, length / shared.TREE_PX_PER_FRAME, focus
                )
        # the axis lives in a dock outside the plot area, so no gutter compensation
        # is needed here — fit the data to the full (already-inset) plot viewport.
        with contextlib.suppress(Exception):
            cam = self._subplot.camera
            cam.maintain_aspect = False
            cam.show_rect(*rect, depth=1)
            # drop any in-flight pan/zoom momentum, else the PanZoomController's
            # per-frame tick() keeps driving the camera from its own cached target
            # and overrides show_rect (reset would appear stuck at the drag position).
            actions = getattr(self._subplot.controller, "_actions", None)
            if actions:
                actions.clear()
            self._figure.canvas.request_draw()

    def _on_click(self, ev) -> None:
        # ignore middle/right buttons (0 = primary/left, 1 = left on some backends)
        if getattr(ev, "button", 1) not in (1, 0):
            return
        idx = ev.pick_info.get("vertex_index")
        if idx is None:
            return
        node_id = int(self._node_ids[int(idx)])
        mods = set(getattr(ev, "modifiers", ()) or ())
        if "Control" in mods or "Meta" in mods:
            self.jump_to_node.emit(node_id)
        else:
            append = "Shift" in mods
            self.node_clicked.emit(node_id, append)
            self.setFocus()

    def _on_hover(self, ev) -> None:
        """Tooltip with the identity of the node under the pointer."""
        if self._closed or not self.show_hover_info or self._drag_start is not None:
            return  # mid box-select: a tooltip would just follow the rubber band
        idx = ev.pick_info.get("vertex_index")
        if idx is None:
            QToolTip.hideText()
            return
        QToolTip.showText(
            QCursor.pos() + TOOLTIP_OFFSET, self._describe_node(int(idx)), self
        )

    def _on_hover_leave(self, ev) -> None:  # noqa: ARG002 (pygfx passes the event)
        QToolTip.hideText()

    def _describe_node(self, row: int) -> str:
        return shared.describe_node(row, self._node_ids, self._lookups)

    def select_points_in_rect(self, x0, x1, y0, y1) -> None:
        """Box-select: emit all node ids whose positions fall in the rect."""
        if len(self._positions) == 0:
            return
        inside = shared.rows_in_rect(self._positions, x0, x1, y0, y1)
        picked = [int(nid) for nid in self._node_ids[inside]]
        if picked:
            self.nodes_selected.emit(picked, True)

    # ------------------------------------------------------------------ #
    # centering
    # ------------------------------------------------------------------ #
    def center_on_node(self, node_id: int) -> None:
        if self._closed:
            return
        row = self._id_to_row.get(int(node_id))
        if row is None:
            return
        self._center_on_rows([row])

    def _camera_view(self):
        """The camera's (cx, cy, width, height)."""
        state = self._subplot.camera.get_state()
        return (
            state["position"][0],
            state["position"][1],
            state["width"],
            state["height"],
        )

    def _rows_fit(self, rows: list[int]) -> bool:
        """Whether all the given rows are inside the current viewport. True (nothing to
        do) for an empty list, and if the camera can't be read."""
        with contextlib.suppress(Exception):
            return shared.rows_fit(self._positions, rows, self._camera_view())
        return True

    def _center_on_rows(self, rows: list[int]) -> None:
        """Bring the given rows into view (``shared.view_fitting_rows``)."""
        with contextlib.suppress(Exception):
            view = shared.view_fitting_rows(self._positions, rows, self._camera_view())
            if view is None:
                return
            self._set_camera_view(view)

    def _set_camera_view(self, view) -> None:
        """Move the camera to ``view`` = (cx, cy, width, height)."""
        cam = self._subplot.camera
        state = cam.get_state()
        new = dict(state)
        new["position"] = (view[0], view[1], state["position"][2])
        new["width"], new["height"] = view[2], view[3]
        cam.set_state(new)
        # clear in-flight momentum so the controller doesn't override the change
        actions = getattr(self._subplot.controller, "_actions", None)
        if actions:
            actions.clear()
