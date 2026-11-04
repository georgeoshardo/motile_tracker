"""The cuts of one mask that alternative segmentations of the same frame offer, ranked.

A mask the user wants to cut may already be cut in another segmentation of the same
image (another network checkpoint, another model). Each such segmentation
("hypothesis") offers one cut: its masks that lie mostly inside the user's mask. The
cut lines come from the hypothesis, the outline stays the user's: the pieces are
clipped to the mask and its left-over pixels go to the nearest piece, so the pieces
tile the mask exactly.

Hypotheses that cut the mask the same way are one candidate. Candidates are ranked
by how well their pieces continue the cells of the frame before or after (whichever
fits better; a cut that reproduces a neighbouring frame's cells is likely right),
then by how many hypotheses give them. :func:`continuity` of the uncut mask tells
whether cutting fits the neighbouring frames better than leaving the mask whole.

Pure array functions (numpy, scipy), independent of any tracks object, like
:mod:`mask_split`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

# a hypothesis mask belongs to the cut when at least this fraction of it lies inside
# the mask
MIN_INSIDE = 0.3
# pieces smaller than this (pixels) are dropped: slivers where a hypothesis's outline
# pokes into the mask
MIN_PIECE = 30
# two cuts are the same when their pieces match in order with at least this IoU
SAME_IOU = 0.8
# how far (pixels) around the mask the cells of a neighbouring frame are looked for
NEIGHBOURHOOD = 6


@dataclass(frozen=True)
class Cut:
    """One way to cut a mask, and how well it fits.

    Attributes:
        pieces: Boolean masks that together are exactly the mask, ordered from the
            start of the first axis (the closed end of a trench).
        hypotheses: Names of the hypotheses that cut the mask this way.
        continuity: Mean IoU of the pieces with the cells of the frame before or
            after (see :func:`continuity`), whichever is higher.
        frame: "before" or "after", the frame that gave ``continuity``; "" when
            neither frame was given.
    """

    pieces: tuple[np.ndarray, ...]
    hypotheses: tuple[str, ...]
    continuity: float = 0.0
    frame: str = ""


def _iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    union = np.count_nonzero(mask_a | mask_b)
    return np.count_nonzero(mask_a & mask_b) / union if union else 0.0


def _first_row(mask: np.ndarray) -> float:
    return float(np.nonzero(mask)[0].mean())


def fill_pieces(mask: np.ndarray, pieces: list[np.ndarray]) -> list[np.ndarray]:
    """Clip the pieces to the mask and give every pixel of the mask that no piece
    covers to the nearest piece, so the pieces tile the mask exactly."""
    labels = np.zeros(mask.shape, np.int32)
    for k, piece in enumerate(pieces, 1):
        labels[piece & mask & (labels == 0)] = k
    if (mask & (labels == 0)).any():
        _, nearest = ndimage.distance_transform_edt(labels == 0, return_indices=True)
        labels = np.where(mask, labels[tuple(nearest)], 0)
    return [labels == k for k in range(1, len(pieces) + 1)]


def hypothesis_pieces(mask: np.ndarray, labels: np.ndarray) -> list[np.ndarray]:
    """The pieces one hypothesis cuts the mask into (filled, ordered), or [] when it
    does not cut it: its masks with at least MIN_INSIDE of their area inside the mask
    and at least MIN_PIECE pixels there."""
    inside = labels[mask]
    pieces = []
    for value, count in zip(*np.unique(inside[inside > 0], return_counts=True)):
        own = labels == value
        if count >= MIN_INSIDE * np.count_nonzero(own) and count >= MIN_PIECE:
            pieces.append(own & mask)
    if len(pieces) < 2:
        return []
    return fill_pieces(mask, sorted(pieces, key=_first_row))


def continuity(
    pieces: list[np.ndarray] | tuple[np.ndarray, ...],
    cells: np.ndarray | None,
    around: np.ndarray,
) -> float:
    """Mean over the pieces of their IoU with the cells of a neighbouring frame
    (a label image), each cell matched to one piece (best matches first), looking only
    at cells that reach into `around`. 0 when there is no such frame or cell."""
    if cells is None:
        return 0.0
    ids = [int(v) for v in np.unique(cells[around]) if v]
    if not ids:
        return 0.0
    iou = np.array([[_iou(piece, cells == v) for v in ids] for piece in pieces])
    used: set[int] = set()
    total = 0.0
    for k in np.argsort(-iou.max(axis=1)):
        free = [j for j in np.argsort(-iou[k]) if j not in used]
        if free:
            used.add(int(free[0]))
            total += float(iou[k, free[0]])
    return total / len(pieces)


def neighbourhood(mask: np.ndarray) -> np.ndarray:
    """Where the cells of a neighbouring frame are looked for: the mask, grown by
    NEIGHBOURHOOD pixels (cells move a little between frames)."""
    return ndimage.binary_dilation(mask, iterations=NEIGHBOURHOOD)


def best_continuity(
    pieces, before: np.ndarray | None, after: np.ndarray | None, around: np.ndarray
) -> tuple[float, str]:
    """The better of the pieces' continuity with the frame before and after, and
    which frame gave it."""
    if before is None and after is None:
        return 0.0, ""
    fit_before = continuity(pieces, before, around)
    fit_after = continuity(pieces, after, around)
    if after is None or (before is not None and fit_before >= fit_after):
        return fit_before, "before"
    return fit_after, "after"


def with_pieces(
    frame: np.ndarray, mask: np.ndarray, pieces: list[np.ndarray] | tuple[np.ndarray, ...]
) -> np.ndarray:
    """A copy of a label image with the mask's pixels relabelled as the pieces (new
    labels above the frame's largest): the frame as it would be after the cut, for
    ranking the cuts of the next frame of a stretch cut together."""
    out = np.array(frame, dtype=np.int64)
    top = int(out.max()) if out.size else 0
    out[mask] = 0
    for k, piece in enumerate(pieces, 1):
        out[piece] = top + k
    return out


def cut_candidates(
    mask: np.ndarray,
    hypotheses: Mapping[str, np.ndarray],
    before: np.ndarray | None = None,
    after: np.ndarray | None = None,
) -> list[Cut]:
    """The distinct cuts of `mask` (into two or more pieces) that the hypotheses
    offer, best first.

    Args:
        mask: 2-D boolean mask to cut.
        hypotheses: Label image of the same frame per hypothesis name.
        before: Label image of the cells of the frame before, if any.
        after: Label image of the cells of the frame after, if any.

    Returns:
        The cuts, ranked by continuity with the frame before or after (to two
        decimals), then by how many hypotheses give them. Empty when no hypothesis
        cuts the mask.
    """
    mask = np.asarray(mask, dtype=bool)
    found: list[tuple[list[np.ndarray], list[str]]] = []
    for name, labels in hypotheses.items():
        pieces = hypothesis_pieces(mask, np.asarray(labels))
        if not pieces:
            continue
        for known, names in found:
            if len(known) == len(pieces) and all(
                _iou(a, b) >= SAME_IOU for a, b in zip(known, pieces)
            ):
                names.append(str(name))
                break
        else:
            found.append((pieces, [str(name)]))
    around = neighbourhood(mask)
    cuts = []
    for pieces, names in found:
        fit, frame = best_continuity(pieces, before, after, around)
        cuts.append(Cut(tuple(pieces), tuple(names), fit, frame))
    return sorted(cuts, key=lambda c: (-round(c.continuity, 2), -len(c.hypotheses)))
