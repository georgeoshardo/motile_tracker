"""Composite user actions that fix segmentation-driven tracking errors.

Two mistakes of a segmentation network dominate mother machine lineages: one mask
where there should be two (a division the network did not see, or two cells that
"rejoined" after dividing), and two masks where there should be one. Each is
fixed here as ONE undoable action that changes both the segmentation and the
graph around it:

* :class:`SplitNode`: part of a node's mask becomes a new node. The new node is
  linked to the neighbouring cell of the previous frame whose track ended there,
  or else to the node's own parent as a second daughter; the node's children are
  handed to whichever part they overlap.
* :class:`MergeNodes`: the pixels of the other nodes at the same time point are
  painted with the kept node's label, which deletes them; the kept node inherits
  their parent (when it had none) and their children (when they had no other
  parent).
* :class:`CutNode`: a node's mask is cut into two or more given pieces (the cut an
  alternative segmentation offers, see :mod:`hypothesis_cut`). Every piece is
  matched to a cell of the previous frame it continues (the node's parent, or a
  neighbour whose track ended there), a piece left over becomes the parent's second
  daughter, the node's children go to the piece they overlap, and a piece without a
  child is continued by a neighbour of the next frame whose track starts there.

Both follow funtracks' own convention for composite actions: the sub-actions are
created with ``_top_level=False`` and the group is recorded once in the action
history, so a single undo restores segmentation, edges and track ids together.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from funtracks.actions import ActionGroup
from funtracks.exceptions import InvalidActionError
from funtracks.user_actions import (
    UserAddEdge,
    UserDeleteEdge,
    UserDeleteNode,
    UserUpdateSegmentation,
)
from scipy import ndimage

from mm_track.data_views.views_coordinator.mask_split import (
    bright_cells_in_frame,
    split_mask,
)

if TYPE_CHECKING:
    from funtracks.data_model import Tracks


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def next_node_id(tracks: Tracks) -> int:
    """A label no node has ever had. Deleted nodes stay in graph_full (deletion is
    soft), so their ids must not be reused either."""
    return int(max(tracks.graph_full.node_ids(), default=0)) + 1


def node_mask(tracks: Tracks, node: int) -> np.ndarray:
    """The node's boolean mask in its frame (2-D for 2D+time data)."""
    frame = np.asarray(tracks.segmentation[int(tracks.get_time(node))])
    return frame == int(node)


def mask_pixels(mask: np.ndarray, time: int) -> tuple[np.ndarray, ...]:
    """The multi-index of a mask's pixels, time axis first, as funtracks' segmentation
    actions expect it."""
    indices = np.nonzero(mask)
    return (np.full(len(indices[0]), int(time), dtype=int), *indices)


def _overlap(mask_a: np.ndarray, mask_b: np.ndarray) -> int:
    return int(np.count_nonzero(mask_a & mask_b))


def _centroid(mask: np.ndarray) -> np.ndarray:
    return np.array([axis.mean() for axis in np.nonzero(mask)])


def _closer_part(
    position: Sequence[float], part_a: np.ndarray, part_b: np.ndarray
) -> int:
    """0 if `position` is closer to part_a's centroid than to part_b's, else 1."""
    position = np.asarray(position, dtype=float)[-part_a.ndim :]
    distance_a = np.linalg.norm(_centroid(part_a) - position)
    distance_b = np.linalg.norm(_centroid(part_b) - position)
    return 0 if distance_a <= distance_b else 1


# ----------------------------------------------------------------------
# split
# ----------------------------------------------------------------------
@dataclass
class SplitPlan:
    """What a split will do, decided before anything is changed."""

    node: int
    keep_mask: np.ndarray
    new_mask: np.ndarray
    method: str = ""
    new_parent: int | None = None
    parent_division: bool = False
    moved_children: list[int] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def find_second_parent(
    tracks: Tracks, node: int, mask: np.ndarray | None = None, margin: int = 3
) -> int | None:
    """The cell of the previous frame whose track ends there next to `node`: the
    cell the merged mask swallowed. Returns None when there is no such cell.

    Args:
        tracks: The tracks.
        node: The (merged) node.
        mask: The part of the node's mask to look next to; the whole mask by default.
        margin: How far (in pixels) around the mask to look.
    """
    time = int(tracks.get_time(node))
    if time == 0:
        return None
    if mask is None:
        mask = node_mask(tracks, node)
    predecessors = {int(p) for p in tracks.predecessors(node)}
    previous = np.asarray(tracks.segmentation[time - 1])
    region = ndimage.binary_dilation(mask, iterations=margin)
    best, best_overlap = None, 0
    for label in np.unique(previous[region]):
        label = int(label)
        if label == 0 or label in predecessors:
            continue
        if len(tracks.successors(label)) != 0:
            continue
        overlap = int(np.count_nonzero(previous[region] == label))
        if overlap > best_overlap:
            best, best_overlap = label, overlap
    return best


def plan_split(
    tracks: Tracks,
    node: int,
    parts: tuple[np.ndarray, np.ndarray],
    owners: tuple[int | None, int | None] = (None, None),
    method: str = "",
) -> SplitPlan:
    """Decide which part keeps the node's label and how the two cells are linked.

    Args:
        tracks: The tracks.
        node: The node being split.
        parts: The two parts of its mask.
        owners: For each part, the node of the previous frame it was matched to (when
            the parts came from guides), else None.
        method: Name of the strategy that produced the parts, for the record.
    """
    node = int(node)
    time = int(tracks.get_time(node))
    predecessors = [int(p) for p in tracks.predecessors(node)]
    parent = predecessors[0] if predecessors else None
    notes: list[str] = []

    # the part that continues the parent keeps the label
    if parent is not None:
        if parent in owners:
            keep_index = owners.index(parent)
        else:
            parent_mask = node_mask(tracks, parent)
            overlaps = [_overlap(parent_mask, part) for part in parts]
            if max(overlaps) > 0:
                keep_index = int(np.argmax(overlaps))
            else:
                keep_index = _closer_part(tracks.get_position(parent), *parts)
    else:
        keep_index = 0 if parts[0].sum() >= parts[1].sum() else 1
    keep_mask, new_mask = parts[keep_index], parts[1 - keep_index]
    new_owner = owners[1 - keep_index]

    # a parent for the new cell: the neighbour whose track ended in the previous
    # frame, else the node's parent as a division
    new_parent = None
    parent_division = False
    candidate = new_owner
    if candidate is None or candidate == parent:
        candidate = find_second_parent(tracks, node, mask=new_mask)
    if (
        candidate is not None
        and candidate != parent
        and int(tracks.get_time(candidate)) < time
        and len(tracks.successors(candidate)) == 0
    ):
        new_parent = int(candidate)
    elif parent is not None:
        if len(tracks.successors(parent)) == 1:
            parent_division = True
        else:
            notes.append(
                f"Node {parent} already has two children, so the new cell starts a "
                "new track."
            )

    # the children go to the part they overlap (or lie closest to)
    moved_children: list[int] = []
    for child in tracks.successors(node):
        child = int(child)
        child_mask = node_mask(tracks, child)
        overlap_keep = _overlap(child_mask, keep_mask)
        overlap_new = _overlap(child_mask, new_mask)
        if overlap_keep == overlap_new:
            goes_to_new = (
                _closer_part(tracks.get_position(child), keep_mask, new_mask) == 1
            )
        else:
            goes_to_new = overlap_new > overlap_keep
        if goes_to_new:
            moved_children.append(child)

    return SplitPlan(
        node=node,
        keep_mask=keep_mask,
        new_mask=new_mask,
        method=method,
        new_parent=new_parent,
        parent_division=parent_division,
        moved_children=moved_children,
        notes=notes,
    )


def propose_split(
    tracks: Tracks, node: int, image: np.ndarray | None = None
) -> SplitPlan | None:
    """Work out how to split `node` into two cells, using the previous frame's cells
    as guides when the track structure offers them, else the image, else geometry.

    Returns None when the node cannot be split (too few pixels).
    """
    node = int(node)
    if tracks.segmentation is None or tracks.ndim != 3:
        return None
    time = int(tracks.get_time(node))
    frame = np.asarray(tracks.segmentation[time])
    mask = frame == node
    if mask.sum() < 2:
        return None
    bright_cells = bright_cells_in_frame(image, frame) if image is not None else None

    guides = None
    owners: tuple[int | None, int | None] = (None, None)
    predecessors = [int(p) for p in tracks.predecessors(node)]
    if predecessors:
        parent = predecessors[0]
        neighbour = find_second_parent(tracks, node)
        if neighbour is not None:
            guides = (node_mask(tracks, parent), node_mask(tracks, neighbour))
            owners = (parent, neighbour)

    result = split_mask(mask, image=image, guides=guides, bright_cells=bright_cells)
    if result is None:
        return None
    if not result.guide_order:
        owners = (None, None)
    return plan_split(tracks, node, result.parts, owners=owners, method=result.method)


class SplitNode(ActionGroup):
    """Split a node into two cells according to a :class:`SplitPlan`, as one undo step.

    Attributes:
        node (int): The node that was split (it keeps its label and its parent).
        new_node (int): The label of the new cell.
        notes (list[str]): Anything the user should know about the links made.
    """

    def __init__(self, tracks: Tracks, plan: SplitPlan, _top_level: bool = True):
        super().__init__(tracks, actions=[])
        self.tracks: Tracks
        if not plan.keep_mask.any() or not plan.new_mask.any():
            raise InvalidActionError("Both parts of a split must contain pixels.")
        if tracks.segmentation is None:
            raise InvalidActionError("Cannot split a node without a segmentation.")

        self.node = int(plan.node)
        self.notes = list(plan.notes)
        time = int(tracks.get_time(self.node))
        new_label = next_node_id(tracks)
        try:
            # paint the new part with a fresh label: funtracks creates the node
            # (position, bbox, mask) and shrinks the old one
            self.actions.append(
                UserUpdateSegmentation(
                    tracks,
                    new_value=new_label,
                    updated_pixels=[(mask_pixels(plan.new_mask, time), self.node)],
                    current_track_id=int(tracks.get_next_track_id()),
                    _top_level=False,
                )
            )
            for child in plan.moved_children:
                self.actions.append(
                    UserDeleteEdge(tracks, (self.node, int(child)), _top_level=False)
                )
            if plan.new_parent is not None:
                self.actions.append(
                    UserAddEdge(
                        tracks, (int(plan.new_parent), new_label), _top_level=False
                    )
                )
            elif plan.parent_division:
                parent = int(tracks.predecessors(self.node)[0])
                self.actions.append(
                    UserAddEdge(tracks, (parent, new_label), _top_level=False)
                )
            for child in plan.moved_children:
                self.actions.append(
                    UserAddEdge(tracks, (new_label, int(child)), _top_level=False)
                )
        except Exception:
            for action in reversed(self.actions):
                action.inverse()
            raise

        self.new_node = new_label
        if _top_level:
            tracks.action_history.add_new_action(self)
            tracks.refresh.emit(new_label)


# ----------------------------------------------------------------------
# cut into several pieces
# ----------------------------------------------------------------------
@dataclass
class CutPlan:
    """What a cut of a node into pieces will do, decided before anything is changed.

    Attributes:
        node: The node being cut.
        pieces: The pieces, which together are exactly the node's mask.
        keep_index: The piece that keeps the node's label (and its parent).
        parents: Per piece, the node of the previous frame it is linked to, or None.
            For the kept piece this is the node's parent, or the cell it is newly
            linked to when the node had none.
        children: Per piece, the node's children it takes.
        adopted: Per piece, the cells of the next frame without a parent (tracks
            starting there) that it is newly linked to.
        method: Where the cut came from, for the record.
        notes: Anything the user should know about the links made.
    """

    node: int
    pieces: list[np.ndarray]
    keep_index: int
    parents: list[int | None]
    children: list[list[int]]
    adopted: list[list[int]] = field(default_factory=list)
    method: str = ""
    notes: list[str] = field(default_factory=list)


def _continuing_cells(tracks: Tracks, node: int, mask: np.ndarray, margin: int) -> list[int]:
    """The cells of the previous frame a piece of `node` may continue: its parent, and
    neighbours next to the mask whose tracks end in that frame."""
    time = int(tracks.get_time(node))
    if time == 0:
        return []
    parents = [int(p) for p in tracks.predecessors(node)]
    previous = np.asarray(tracks.segmentation[time - 1])
    region = ndimage.binary_dilation(mask, iterations=margin)
    cells = list(parents)
    for label in np.unique(previous[region]):
        label = int(label)
        if label and label not in cells and len(tracks.successors(label)) == 0:
            cells.append(label)
    return cells


def _starting_cells(tracks: Tracks, node: int, mask: np.ndarray, margin: int) -> list[int]:
    """The cells of the next frame next to the mask whose tracks start there (no
    parent): where pieces of `node` may continue."""
    time = int(tracks.get_time(node))
    if time + 1 >= int(tracks.segmentation.shape[0]):
        return []
    following = np.asarray(tracks.segmentation[time + 1])
    region = ndimage.binary_dilation(mask, iterations=margin)
    return [
        int(label)
        for label in np.unique(following[region])
        if label and len(tracks.predecessors(int(label))) == 0
    ]


def _match(overlap: np.ndarray, rows: list[int], cells: list[int]) -> dict[int, int]:
    """One-to-one matching of rows (pieces) to cells by overlap, largest first."""
    matched: dict[int, int] = {}
    used: set[int] = set()
    for flat in np.argsort(-overlap, axis=None):
        k, j = divmod(int(flat), overlap.shape[1])
        if overlap[k, j] == 0:
            break
        if rows[k] not in matched and j not in used:
            matched[rows[k]] = cells[j]
            used.add(j)
    return matched


def plan_cut(
    tracks: Tracks,
    node: int,
    pieces: Sequence[np.ndarray],
    method: str = "",
    margin: int = 3,
) -> CutPlan:
    """Decide how a node cut into `pieces` is linked.

    Each piece is matched, one to one and by overlap (pieces grown by `margin`
    pixels), to a cell of the previous frame it may continue: the node's parent, or a
    neighbour whose track ends there. The piece matched to the parent keeps the
    node's label; without a parent, the largest piece keeps it. The first piece left
    unmatched becomes the parent's second daughter when the parent has no other
    child; any others start new tracks. The node's children go to the piece they
    overlap most (or lie closest to). Pieces left without a child are matched the
    same way to the cells of the next frame next to them whose tracks start there,
    which then continue them.
    """
    node = int(node)
    pieces = [np.asarray(p, dtype=bool) for p in pieces]
    mask = node_mask(tracks, node)
    time = int(tracks.get_time(node))
    predecessors = [int(p) for p in tracks.predecessors(node)]
    parent = predecessors[0] if predecessors else None
    notes: list[str] = []

    cells = _continuing_cells(tracks, node, mask, margin)
    previous = np.asarray(tracks.segmentation[time - 1]) if cells else None
    grown = [ndimage.binary_dilation(p, iterations=margin) for p in pieces]
    overlap = np.array(
        [[np.count_nonzero(previous[g] == c) for c in cells] for g in grown]
    ).reshape(len(pieces), len(cells))
    matched = _match(overlap, list(range(len(pieces))), cells)  # piece -> cell

    if parent is not None and parent in matched.values():
        keep_index = next(k for k, c in matched.items() if c == parent)
    elif parent is not None:
        parent_mask = node_mask(tracks, parent)
        reach = [_overlap(parent_mask, g) for g in grown]
        if max(reach) > 0:
            keep_index = int(np.argmax(reach))
        else:
            position = np.asarray(tracks.get_position(parent), dtype=float)[-mask.ndim :]
            keep_index = int(
                np.argmin([np.linalg.norm(_centroid(p) - position) for p in pieces])
            )
        matched.pop(keep_index, None)  # the kept piece keeps the parent
    else:
        keep_index = int(np.argmax([np.count_nonzero(p) for p in pieces]))

    parents: list[int | None] = [matched.get(k) for k in range(len(pieces))]
    if parent is not None:
        parents[keep_index] = parent
    daughters = len(tracks.successors(parent)) if parent is not None else 2
    for k in range(len(pieces)):
        if parents[k] is not None or k == keep_index:
            continue
        if daughters < 2:
            parents[k] = parent
            daughters += 1
        else:
            notes.append(
                "A piece continues no cell of the previous frame, so it starts a new "
                "track."
            )

    children: list[list[int]] = [[] for _ in pieces]
    for child in tracks.successors(node):
        child = int(child)
        child_mask = node_mask(tracks, child)
        reach = [_overlap(child_mask, p) for p in pieces]
        if max(reach) > 0:
            k = int(np.argmax(reach))
        else:
            position = np.asarray(tracks.get_position(child), dtype=float)[-mask.ndim :]
            k = int(np.argmin([np.linalg.norm(_centroid(p) - position) for p in pieces]))
        children[k].append(child)

    adopted: list[list[int]] = [[] for _ in pieces]
    childless = [k for k in range(len(pieces)) if not children[k]]
    starting = _starting_cells(tracks, node, mask, margin) if childless else []
    if starting:
        following = np.asarray(tracks.segmentation[time + 1])
        overlap = np.array(
            [[np.count_nonzero(following[grown[k]] == c) for c in starting] for k in childless]
        ).reshape(len(childless), len(starting))
        for k, cell in _match(overlap, childless, starting).items():
            adopted[k].append(cell)

    return CutPlan(
        node=node,
        pieces=pieces,
        keep_index=keep_index,
        parents=parents,
        children=children,
        adopted=adopted,
        method=method,
        notes=notes,
    )


class CutNode(ActionGroup):
    """Cut a node into pieces according to a :class:`CutPlan`, as one undo step.

    Attributes:
        node (int): The node that was cut (its label stays on the kept piece).
        new_nodes (list[int]): The labels of the other pieces, in piece order.
        notes (list[str]): Anything the user should know about the links made.
    """

    def __init__(self, tracks: Tracks, plan: CutPlan, _top_level: bool = True):
        super().__init__(tracks, actions=[])
        self.tracks: Tracks
        if tracks.segmentation is None:
            raise InvalidActionError("Cannot cut a node without a segmentation.")
        if len(plan.pieces) < 2 or any(not p.any() for p in plan.pieces):
            raise InvalidActionError("A cut needs two or more pieces with pixels.")
        mask = node_mask(tracks, plan.node)
        union = np.zeros_like(mask)
        for piece in plan.pieces:
            if (union & piece).any():
                raise InvalidActionError("The pieces of a cut must not overlap.")
            union |= piece
        if not np.array_equal(union, mask):
            raise InvalidActionError("The pieces of a cut must be exactly the node's mask.")

        self.node = int(plan.node)
        self.notes = list(plan.notes)
        time = int(tracks.get_time(self.node))
        parent = next((int(p) for p in tracks.predecessors(self.node)), None)
        labels: dict[int, int] = {plan.keep_index: self.node}
        try:
            # paint each other piece with a fresh label: funtracks creates its node
            # and shrinks the cut one
            for k, piece in enumerate(plan.pieces):
                if k == plan.keep_index:
                    continue
                labels[k] = next_node_id(tracks)
                self.actions.append(
                    UserUpdateSegmentation(
                        tracks,
                        new_value=labels[k],
                        updated_pixels=[(mask_pixels(piece, time), self.node)],
                        current_track_id=int(tracks.get_next_track_id()),
                        _top_level=False,
                    )
                )
            moved = [
                (k, child)
                for k, taken in enumerate(plan.children)
                if k != plan.keep_index
                for child in taken
            ]
            for _, child in moved:
                self.actions.append(
                    UserDeleteEdge(tracks, (self.node, int(child)), _top_level=False)
                )
            for k, cell in enumerate(plan.parents):
                if cell is None or (k == plan.keep_index and cell == parent):
                    continue
                self.actions.append(
                    UserAddEdge(tracks, (int(cell), labels[k]), _top_level=False)
                )
            for k, child in moved:
                self.actions.append(
                    UserAddEdge(tracks, (labels[k], int(child)), _top_level=False)
                )
            for k, taken in enumerate(plan.adopted):
                for cell in taken:
                    self.actions.append(
                        UserAddEdge(tracks, (labels[k], int(cell)), _top_level=False)
                    )
        except Exception:
            for action in reversed(self.actions):
                action.inverse()
            raise

        self.new_nodes = [labels[k] for k in range(len(plan.pieces)) if k != plan.keep_index]
        if _top_level:
            tracks.action_history.add_new_action(self)
            tracks.refresh.emit(self.new_nodes[0])


class CutNodes(ActionGroup):
    """Cut several nodes, each into its given pieces, as one undo step.

    The cuts are made earliest frame first and each is planned (:func:`plan_cut`)
    after the one before is made, so a stretch of frames cut together links up: the
    pieces of one frame continue the pieces of the frame before. If any cut fails,
    none is kept.

    Attributes:
        cuts (list[CutNode]): One per node, in time order.
        new_nodes (list[int]): The labels of all new pieces.
        notes (list[str]): Anything the user should know about the links made.
    """

    def __init__(
        self,
        tracks: Tracks,
        cuts: Sequence[tuple[int, Sequence[np.ndarray], str]],
        _top_level: bool = True,
    ):
        """
        Args:
            tracks: The tracks.
            cuts: (node, pieces, method) per node to cut.
        """
        super().__init__(tracks, actions=[])
        self.tracks: Tracks
        if not cuts:
            raise InvalidActionError("Nothing to cut.")
        order = sorted(cuts, key=lambda c: int(tracks.get_time(c[0])))
        self.cuts: list[CutNode] = []
        try:
            for node, pieces, method in order:
                cut = CutNode(tracks, plan_cut(tracks, node, pieces, method=method), _top_level=False)
                self.cuts.append(cut)
                self.actions.append(cut)
        except Exception:
            for action in reversed(self.actions):
                action.inverse()
            raise
        self.new_nodes = [n for cut in self.cuts for n in cut.new_nodes]
        self.notes = list(dict.fromkeys(note for cut in self.cuts for note in cut.notes))
        if _top_level:
            tracks.action_history.add_new_action(self)
            tracks.refresh.emit(self.new_nodes[0] if len(self.cuts) == 1 else None)


# ----------------------------------------------------------------------
# merge
# ----------------------------------------------------------------------
def choose_kept_node(tracks: Tracks, nodes: Sequence[int]) -> int:
    """Of the nodes to merge, keep the one that continues a track (has a parent),
    then the largest, then the lowest id."""
    frame = np.asarray(tracks.segmentation[int(tracks.get_time(nodes[0]))])

    def rank(node: int) -> tuple[int, int, int]:
        has_parent = len(tracks.predecessors(node)) > 0
        area = int(np.count_nonzero(frame == node))
        return (0 if has_parent else 1, -area, node)

    return min((int(n) for n in nodes), key=rank)


class MergeNodes(ActionGroup):
    """Merge nodes of one time point into one, as one undo step.

    The other nodes' pixels are painted with the kept node's label, which grows
    it and deletes them. Their links are detached first, so that funtracks does
    not bridge their tracks across this frame, and then inherited by the kept
    node where it has room: their parent if the kept node had none (and only one
    is on offer), their children as long as the kept node ends up with at most
    two.

    Attributes:
        kept_node (int): The node that remains.
        removed_nodes (list[int]): The nodes merged into it.
        notes (list[str]): Links that could not be carried over.
    """

    def __init__(
        self,
        tracks: Tracks,
        nodes: Sequence[int],
        keep: int | None = None,
        _top_level: bool = True,
    ):
        super().__init__(tracks, actions=[])
        self.tracks: Tracks
        nodes = [int(n) for n in dict.fromkeys(int(n) for n in nodes)]
        if len(nodes) < 2:
            raise InvalidActionError("Select at least two nodes to merge.")
        if tracks.segmentation is None:
            raise InvalidActionError("Cannot merge nodes without a segmentation.")
        times = {int(tracks.get_time(n)) for n in nodes}
        if len(times) != 1:
            raise InvalidActionError("Only nodes at the same time point can be merged.")
        time = times.pop()

        kept = int(keep) if keep is not None else choose_kept_node(tracks, nodes)
        if kept not in nodes:
            raise InvalidActionError(f"Node {kept} is not among the nodes to merge.")
        others = [n for n in nodes if n != kept]
        self.kept_node = kept
        self.removed_nodes = others
        self.notes: list[str] = []

        kept_parents = [int(p) for p in tracks.predecessors(kept)]
        other_parents = {n: [int(p) for p in tracks.predecessors(n)] for n in others}
        other_children = {n: [int(c) for c in tracks.successors(n)] for n in others}
        frame = np.asarray(tracks.segmentation[time])

        try:
            # detach the pieces before deleting them (see class docstring)
            for other in others:
                for parent in other_parents[other]:
                    self.actions.append(
                        UserDeleteEdge(tracks, (parent, other), _top_level=False)
                    )
                for child in other_children[other]:
                    self.actions.append(
                        UserDeleteEdge(tracks, (other, child), _top_level=False)
                    )
            # give their pixels to the kept node; a node without pixels is deleted
            for other in others:
                pixels = mask_pixels(frame == other, time)
                if len(pixels[0]) == 0:
                    # nothing to paint over: just drop the (already detached) node
                    self.actions.append(UserDeleteNode(tracks, other, _top_level=False))
                    continue
                self.actions.append(
                    UserUpdateSegmentation(
                        tracks,
                        new_value=kept,
                        updated_pixels=[(pixels, other)],
                        current_track_id=int(tracks.get_track_id(kept)),
                        _top_level=False,
                    )
                )
            # inherit links the kept node has room for
            if not kept_parents:
                candidates = list(
                    dict.fromkeys(p for other in others for p in other_parents[other])
                )
                if len(candidates) == 1 and len(tracks.successors(candidates[0])) < 2:
                    self.actions.append(
                        UserAddEdge(tracks, (candidates[0], kept), _top_level=False)
                    )
                elif len(candidates) > 1:
                    self.notes.append(
                        f"Nodes {candidates} were parents of the merged pieces; the "
                        f"merged cell {kept} is left without a parent."
                    )
            else:
                dropped = [
                    p
                    for other in others
                    for p in other_parents[other]
                    if p not in kept_parents
                ]
                if dropped:
                    self.notes.append(
                        f"The tracks of {sorted(set(dropped))} end before this frame; "
                        f"the merged cell {kept} continues from {kept_parents[0]}."
                    )
            for child in dict.fromkeys(
                c for other in others for c in other_children[other]
            ):
                if tracks.graph_solution.has_edge(kept, child):
                    continue
                if len(tracks.predecessors(child)) > 0:
                    continue
                if len(tracks.successors(kept)) >= 2:
                    self.notes.append(
                        f"Node {kept} already has two children; {child} starts a new "
                        "track."
                    )
                    continue
                self.actions.append(
                    UserAddEdge(tracks, (kept, child), _top_level=False)
                )
        except Exception:
            for action in reversed(self.actions):
                action.inverse()
            raise

        if _top_level:
            tracks.action_history.add_new_action(self)
            tracks.refresh.emit(kept)


def group_by_time(tracks: Tracks, nodes: Sequence[int]) -> dict[int, list[int]]:
    """The nodes per time point, in time order, without duplicates."""
    groups: dict[int, list[int]] = {}
    for node in dict.fromkeys(int(n) for n in nodes):
        groups.setdefault(int(tracks.get_time(node)), []).append(node)
    return dict(sorted(groups.items()))


def can_merge(tracks: Tracks | None, nodes: Sequence[int]) -> bool:
    """Whether the nodes include two or more at one time point (with a segmentation)."""
    if tracks is None or tracks.segmentation is None or len(nodes) < 2:
        return False
    return any(len(group) >= 2 for group in group_by_time(tracks, nodes).values())


class MergeNodesOverTime(ActionGroup):
    """Merge selected nodes frame by frame, as one undo step.

    For a cell that the segmentation cut into two pieces over several frames: the
    pieces selected at each time point are merged into one cell with
    :class:`MergeNodes`, earliest frame first, so each merge sees the links left by
    the one before and the result is a single track through the stretch. Time
    points with only one selected node are left alone and reported in
    ``skipped_nodes``.

    Attributes:
        merges (list[MergeNodes]): One merge per time point, in time order.
        kept_nodes (list[int]): The node remaining at each merged time point.
        removed_nodes (list[int]): All nodes merged away.
        skipped_nodes (list[int]): Selected nodes alone at their time point.
        notes (list[str]): Links that could not be carried over.
    """

    def __init__(self, tracks: Tracks, nodes: Sequence[int], _top_level: bool = True):
        super().__init__(tracks, actions=[])
        self.tracks: Tracks
        if tracks.segmentation is None:
            raise InvalidActionError("Cannot merge nodes without a segmentation.")
        groups = group_by_time(tracks, nodes)
        frames = {t: group for t, group in groups.items() if len(group) >= 2}
        if not frames:
            raise InvalidActionError(
                "Select two or more nodes in at least one time point to merge."
            )
        self.skipped_nodes = [
            group[0] for t, group in groups.items() if len(group) == 1
        ]
        self.merges: list[MergeNodes] = []
        try:
            for group in frames.values():
                merge = MergeNodes(tracks, group, _top_level=False)
                self.merges.append(merge)
                self.actions.append(merge)
        except Exception:
            for action in reversed(self.actions):
                action.inverse()
            raise
        self.kept_nodes = [merge.kept_node for merge in self.merges]
        self.removed_nodes = [n for merge in self.merges for n in merge.removed_nodes]
        # a note about a node merged away in a later frame is no longer true
        removed = {str(n) for n in self.removed_nodes}
        self.notes = list(
            dict.fromkeys(
                note
                for merge in self.merges
                for note in merge.notes
                if not removed & set(_numbers(note))
            )
        )
        if _top_level:
            tracks.action_history.add_new_action(self)
            tracks.refresh.emit(
                self.kept_nodes[0] if len(self.kept_nodes) == 1 else None
            )


def _numbers(text: str) -> list[str]:
    return re.findall(r"\d+", text)
