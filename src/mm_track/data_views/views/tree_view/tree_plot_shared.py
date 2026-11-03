"""What the lineage tree view draws, independent of how it is drawn.

The tree view has two renderers with the same interface: fastplotlib
(tree_plot_fpl, on the GPU through wgpu) and pyqtgraph (tree_plot_pg, Qt's
painter, for where wgpu can't run, such as a web browser). Both lay out, size,
label, describe and select nodes with what is here, so that a change to any of
it reaches both.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from qtpy.QtCore import QPoint

BASE_SIZE = 10.0  # bigger than the old 10 so nodes are easier to click and glyphs read
SELECT_BUMP = 6.0  # size increase for a selected node (scaled along with the markers)
SELECT_BUMP_MIN = 2.0  # ...but at least this, so a small node still reads as selected
# Markers are drawn in screen pixels, so a fixed size looks wrong at both ends of the
# zoom range: zoomed out, the lanes pack closer until parallel lineages merge into one
# blob; zoomed in, a handful of nodes float in a mostly empty canvas. In tree mode both
# grids are one unit apart (whole-numbered lanes, integer t), so scale the markers with
# whichever of the two is tighter on screen: a glyph spans MARKER_FILL of that gap,
# clamped to [MIN_SIZE, MAX_SIZE]. Feature mode keeps a fixed size — its value axis is
# continuous, so it has no grid to measure.
MIN_SIZE = 3.0  # never shrink past this: nodes must stay visible and clickable
MAX_SIZE = 100.0  # ...and never grow past this, however far you zoom in
MARKER_FILL = 0.9  # glyph diameter as a fraction of the grid spacing (<1 leaves a gap)
SIZE_SCALE_EPS = 0.02  # skip <2% changes so a slow zoom won't re-upload sizes a frame
# Once a node is drawn big enough to hold text, write its node_id inside it. Only the
# nodes currently on screen get a label, and only while there are few enough of them
# that building the text blocks stays cheap.
NODE_LABEL_MIN_SIZE = 20.0  # node diameter (px) from which the id fits inside
NODE_LABEL_MAX = 400  # give up past this many on-screen nodes
# The font has to fit the *longest* id on screen inside a circle, so it is capped two
# ways: by the dot's height, and by its width once the digits are counted (a five-digit
# id in a dot sized for two would spill its first and last digit outside the circle).
NODE_LABEL_FONT_FRACTION = 0.5  # of the dot diameter, for a short label
# digit ink width as a fraction of the font size, measured by rendering a string
# offscreen and counting pixels (0.550 at both font 20 and font 40)
NODE_LABEL_DIGIT_WIDTH = 0.55
# Track ids along the lane axis (optional, toggled from the Visualization tab).
TRACK_LABEL_MIN_PX = 30.0  # min spacing between labels; lanes in between are skipped
TRACK_LABEL_MAX = 200  # hard cap on label count, whatever the spacing works out to
# Qt drops a tooltip a little below-right of the position it is given, which leaves it
# floating away from the node. Pull it back up so it sits right next to the cursor,
# still clear of the pointer itself.
TOOLTIP_OFFSET = QPoint(10, -22)
# triangle (split) and cross (end) glyphs have less visual weight than a filled circle
# at the same bounding-box size, so enlarge them to match and stay clickable/legible.
SYMBOL_SIZE_FACTOR = {"t1": 1.9, "x": 1.4}
# a right-button press that moves <= this many pixels counts as a click (-> reset view);
# a larger movement is a drag, which squeezes/zooms the x or y axis.
RIGHT_CLICK_DRAG_PX = 4.0
SELECT_COLOR = np.array([0.0, 1.0, 1.0, 1.0], dtype=np.float32)  # cyan outline
SELECT_EDGE_WIDTH = 2.0  # px outline around a selected node (fill keeps its color)


# Where the kymograph is, drawn over the tree as in the track viewer: a band over the
# frames on screen in the kymograph, grey-outlined, and a line at the frame under
# the cursor there. RGBA, 0-1.
VIEW_BAND_FILL = (1.0, 1.0, 1.0, 0.12)  # as it should look over black
VIEW_BAND_EDGE = (0.63, 0.63, 0.63, 0.8)
HOVER_LINE_COLOR = (0.0, 0.75, 0.86, 0.9)


def time_marks(
    positions: np.ndarray,
    view_direction: str,
    frames: tuple[float, float] | None,
    hover: int | None,
) -> tuple[tuple[float, float, float, float] | None, np.ndarray | None]:
    """Where to draw the kymograph's place on the tree.

    Args:
        positions: the plotted node positions (see ``node_positions``); the band and
            line span the lanes these cover, plus one lane either side.
        view_direction: "horizontal" (time along x) or "vertical" (time down y).
        frames: the stretch of time shown in the kymograph, in frames (frame t
            spans t - 0.5 .. t + 0.5), or None.
        hover: the frame under the cursor in the kymograph, or None.

    Returns:
        The band as (x0, y0, x1, y1) and the line as a (2, 2) array of end points,
        each None when there is nothing to draw.
    """
    if len(positions) == 0:
        return None, None
    vertical = view_direction == "vertical"
    lanes = positions[:, 0] if vertical else positions[:, 1]
    low, high = float(lanes.min()) - 1.0, float(lanes.max()) + 1.0

    band = None
    if frames is not None:
        start, stop = frames
        band = (low, -stop, high, -start) if vertical else (start, low, stop, high)
    line = None
    if hover is not None:
        t = float(hover)
        ends = [[low, -t], [high, -t]] if vertical else [[t, low], [t, high]]
        line = np.array(ends, dtype=np.float32)
    return band, line


FOLLOW_MARGIN = 0.05  # of the view's time span, kept between the band and the edge


def view_following_band(
    view: tuple[float, float, float, float],
    band: tuple[float, float, float, float] | None,
    view_direction: str,
) -> tuple[float, float, float, float] | None:
    """The view (cx, cy, width, height) panned along time just far enough to show
    ``band`` (from ``time_marks``) with a margin, or None if it is in view already.
    The zoom and the lanes shown stay as they are; a band wider than the view is
    centred.
    """
    if band is None:
        return None
    cx, cy, width, height = view
    vertical = view_direction == "vertical"
    low, high = (band[1], band[3]) if vertical else (band[0], band[2])
    center, span = (cy, height) if vertical else (cx, width)
    margin = FOLLOW_MARGIN * span
    first, last = center - span / 2 + margin, center + span / 2 - margin
    if high - low > last - first:
        new_center = (low + high) / 2
    elif low < first:
        new_center = center - (first - low)
    elif high > last:
        new_center = center + (high - last)
    else:
        return None
    return (
        (cx, new_center, width, height) if vertical else (new_center, cy, width, height)
    )


def node_positions(
    df: pd.DataFrame, view_direction: str, plot_type: str, feature: str | None
) -> np.ndarray:
    """(N, 3) float32 positions. Vertical: (axis_value, -t). Horizontal swaps.

    In horizontal view the perpendicular (y) axis is negated so tree lanes read
    top-to-bottom like the old pyqtgraph tree. But for a numeric *feature* the
    y-axis should increase upward (larger values higher), so feature positions are
    NOT negated. Feature plots are always horizontal (see TreeWidget), so this only
    affects the horizontal branch.
    """
    axis_col = feature if plot_type == "feature" else "x_axis_pos"
    a = df[axis_col].to_numpy(dtype=np.float32)
    t = df["t"].to_numpy(dtype=np.float32)
    pos = np.zeros((len(df), 3), dtype=np.float32)
    if view_direction == "vertical":
        pos[:, 0] = a
        pos[:, 1] = -t
    else:  # horizontal: time along x, perpendicular axis along y
        pos[:, 0] = t
        pos[:, 1] = a if plot_type == "feature" else -a
    return pos


def base_sizes(symbols: np.ndarray) -> np.ndarray:
    """Per-node marker sizes before zoom: triangle/cross glyphs enlarged to read as
    big as the dots."""
    factors = np.array([SYMBOL_SIZE_FACTOR.get(s, 1.0) for s in symbols], np.float32)
    return (BASE_SIZE * factors).astype(np.float32)


@dataclass
class TrackLookups:
    """Per-row track ids, times and lineage ids for the hover tooltip, plus the
    lane -> track id map behind the track-id axis."""

    row_track_ids: np.ndarray
    row_times: np.ndarray
    lane_track_ids: dict[int, int]
    row_lineage_ids: np.ndarray | None  # None when the df has no parent_track_id


def empty_lookups() -> TrackLookups:
    return TrackLookups(
        np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32), {}, None
    )


def track_lookups(df: pd.DataFrame) -> TrackLookups:
    """Track, time and lineage lookups for the rows of ``df``.

    A lineage id is the track id at the root of the lineage: follow parent_track_id
    (0 means "no parent") up from each track. Deriving it here keeps the tree view
    independent of funtracks' optional lineage feature.
    """
    row_track_ids = df["track_id"].to_numpy()
    # Only whole-numbered lanes are labelled: a parent sits between its daughters'
    # lanes (see tree_widget_utils.trench_lanes), and its line leads into theirs.
    lanes = df["x_axis_pos"].to_numpy()
    whole = lanes == np.round(lanes)
    lookups = TrackLookups(
        row_track_ids=row_track_ids,
        row_times=df["t"].to_numpy(),
        lane_track_ids=dict(
            zip(
                lanes[whole].astype(int),
                row_track_ids[whole].astype(int),
                strict=True,
            )
        ),
        row_lineage_ids=None,
    )
    if "parent_track_id" not in df:
        return lookups
    parent_of = dict(
        zip(
            row_track_ids.astype(int),
            df["parent_track_id"].to_numpy().astype(int),
            strict=True,
        )
    )
    roots: dict[int, int] = {}
    for track_id in parent_of:
        chain, current = set(), track_id
        # `current not in chain` only guards against a cycle in malformed data,
        # which would otherwise spin here forever
        while (
            current not in roots
            and current not in chain
            and parent_of.get(current, 0) != 0
        ):
            chain.add(current)
            current = parent_of[current]
        root = roots.get(current, current)
        roots[current] = root
        for seen in chain:
            roots[seen] = root
    lookups.row_lineage_ids = np.array(
        [roots[int(tid)] for tid in row_track_ids], dtype=np.int64
    )
    return lookups


def edge_rows(df: pd.DataFrame, node_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Row indices of every parent -> child edge: ``(parent_rows, child_rows)``,
    both empty if there are none. Edges to a parent outside ``df`` are dropped."""
    parent_ids = df["parent_id"].to_numpy()
    # vectorized parent_id -> row index (NaN where the parent isn't present)
    row_of = pd.Series(np.arange(len(df)), index=node_ids)
    parent_rows = row_of.reindex(parent_ids).to_numpy()
    mask = (parent_ids != 0) & ~np.isnan(parent_rows)
    return parent_rows[mask].astype(np.int64), np.nonzero(mask)[0]


def marker_scale(
    plot_type: str,
    view_direction: str,
    x_px: float,
    y_px: float,
    sizes: np.ndarray,
) -> float:
    """Multiplier on the base sizes at a zoom where one unit is ``x_px`` by ``y_px``
    screen pixels, clamped to ``[MIN_SIZE, MAX_SIZE]``. Tree mode only; feature mode
    returns 1.0.

    The two axes are not symmetric. Lanes are what must not collide, so the lane
    spacing alone decides how far the markers shrink — a track densely sampled in
    time is *meant* to read as a continuous run of overlapping dots, and letting the
    time spacing shrink things too would leave the markers permanently at the floor.
    Growing is the other way round: a marker only grows past its base size once both
    spacings can take it, so zooming x alone stretches the tree without inflating the
    dots into their vertical neighbours.
    """
    if plot_type == "feature":
        return 1.0
    lane_px = x_px if view_direction == "vertical" else y_px
    # measure against the widest glyph in the plot, not the plain dot: the
    # triangle/cross factors make those nearly twice as wide, and they are the
    # ones that touch their neighbours first.
    widest = float(sizes.max()) if len(sizes) else BASE_SIZE
    shrink = lane_px * MARKER_FILL / widest  # lane spacing rules the way down
    grow = min(x_px, y_px) * MARKER_FILL / widest  # both rule the way up
    scale = min(shrink, max(grow, 1.0))
    return float(np.clip(scale, MIN_SIZE / BASE_SIZE, MAX_SIZE / widest))


def select_bump(size_scale: float) -> float:
    """Extra size for a selected node: shrinks with the markers, but never below
    ``SELECT_BUMP_MIN``, so the selection stays visible when zoomed out."""
    return max(SELECT_BUMP * size_scale, SELECT_BUMP_MIN)


def node_label_font_size(diameter: float, labels: list[str]) -> float:
    """Font size that fits the longest of ``labels`` inside a marker."""
    widest_label = max(len(label) for label in labels)
    return min(
        diameter * NODE_LABEL_FONT_FRACTION,
        diameter * MARKER_FILL / (widest_label * NODE_LABEL_DIGIT_WIDTH),
    )


def labelled_lanes(
    low: float, high: float, span_px: float, lane_track_ids: dict[int, int]
) -> list[int]:
    """The lanes between ``low`` and ``high`` (shown across ``span_px`` pixels) that
    get a track-id label, thinned out so neighbouring labels stay at least
    ``TRACK_LABEL_MIN_PX`` apart; empty if there would be too many.

    Lanes are one unit apart, so the step is a lane count, and lanes that fall
    between two labels are simply skipped rather than drawn on top of each other.
    """
    span = high - low
    if span <= 0 or span_px < 1:
        return []
    step = max(1, int(np.ceil(TRACK_LABEL_MIN_PX / (span_px / span))))
    lanes = [
        lane
        for lane in range(int(np.ceil(low)), int(np.floor(high)) + 1, step)
        if lane in lane_track_ids
    ]
    return lanes if len(lanes) <= TRACK_LABEL_MAX else []


def describe_node(row: int, node_ids: np.ndarray, lookups: TrackLookups) -> str:
    """Tooltip text for one row: node, track, lineage (when derivable) and time."""
    parts = [f"node {int(node_ids[row])}"]
    if len(lookups.row_track_ids) > row:
        parts.append(f"track {int(lookups.row_track_ids[row])}")
    if lookups.row_lineage_ids is not None and len(lookups.row_lineage_ids) > row:
        parts.append(f"lineage {int(lookups.row_lineage_ids[row])}")
    if len(lookups.row_times) > row:
        parts.append(f"t {int(lookups.row_times[row])}")
    return "\n".join(parts)


def rows_in_rect(positions: np.ndarray, x0, x1, y0, y1) -> np.ndarray:
    """Boolean mask of the rows whose positions fall in the rect."""
    xs, ys = positions[:, 0], positions[:, 1]
    return (
        (xs >= min(x0, x1))
        & (xs <= max(x0, x1))
        & (ys >= min(y0, y1))
        & (ys <= max(y0, y1))
    )


def reset_rect(positions: np.ndarray) -> tuple[float, float, float, float] | None:
    """``(xmin, xmax, ymin, ymax)`` framing every node with a 5% margin, or None if
    there are none."""
    if len(positions) == 0:
        return None
    xs, ys = positions[:, 0], positions[:, 1]
    xmin, xmax = float(np.min(xs)), float(np.max(xs))
    ymin, ymax = float(np.min(ys)), float(np.max(ys))
    px = (xmax - xmin) * 0.05 or 1.0
    py = (ymax - ymin) * 0.05 or 1.0
    return xmin - px, xmax + px, ymin - py, ymax + py


# In tree mode the view opens at a readable scale along time, this many screen
# pixels per frame, rather than squeezing the whole movie into the plot; it follows
# the kymograph from there (view_following_band).
TREE_PX_PER_FRAME = 5.0


def limit_time_span(
    rect: tuple[float, float, float, float],
    view_direction: str,
    span: float,
    focus: float | None,
) -> tuple[float, float, float, float]:
    """``rect`` (xmin, xmax, ymin, ymax) with its time range cut to at most ``span``
    frames: centred on frame ``focus`` (kept inside the data), or from the start."""
    xmin, xmax, ymin, ymax = rect
    vertical = view_direction == "vertical"
    low, high = (-ymax, -ymin) if vertical else (xmin, xmax)
    if high - low <= span:
        return rect
    if focus is None:
        low, high = low, low + span
    else:
        center = min(max(focus, low + span / 2), high - span / 2)
        low, high = center - span / 2, center + span / 2
    return (xmin, xmax, -high, -low) if vertical else (low, high, ymin, ymax)


def rows_fit(positions: np.ndarray, rows: list[int], view) -> bool:
    """Whether all the given rows are inside ``view`` = (cx, cy, width, height).
    True (nothing to do) for an empty list."""
    if not rows:
        return True
    pts = positions[rows]
    x, y, w, h = view
    return bool(
        (x - w / 2) <= pts[:, 0].min()
        and pts[:, 0].max() <= (x + w / 2)
        and (y - h / 2) <= pts[:, 1].min()
        and pts[:, 1].max() <= (y + h / 2)
    )


def view_fitting_rows(positions: np.ndarray, rows: list[int], view):
    """The view (cx, cy, width, height) that brings ``rows`` into ``view``, or None
    if they already fit.

    Keeps the current zoom if the rows fit at it; otherwise zooms out *just enough*
    (per axis) to fit them, and never zooms in. Used for single-node centering and for
    multi-node selection — e.g. the two endpoints of a just-broken edge, which may now
    sit far apart: rather than pan to their midpoint at the current zoom (leaving both
    off-screen), the view widens so both are visible.
    """
    if not rows or rows_fit(positions, rows, view):
        return None
    pts = positions[rows]
    xmin, xmax = float(pts[:, 0].min()), float(pts[:, 0].max())
    ymin, ymax = float(pts[:, 1].min()), float(pts[:, 1].max())
    _, _, w, h = view
    # dividing the span by 0.8 leaves a ~10% margin so nodes aren't flush to the
    # edge. For a single row the span is 0, so the zoom is preserved and the view
    # simply pans to it.
    return (
        (xmin + xmax) / 2,
        (ymin + ymax) / 2,
        max(w, (xmax - xmin) / 0.8),
        max(h, (ymax - ymin) / 0.8),
    )
