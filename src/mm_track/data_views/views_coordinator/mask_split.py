"""Split one 2-D mask that covers two cells into two masks.

Pure array functions (numpy, scipy, scikit-image), independent of any tracks
object, so they can be unit-tested and benchmarked on their own. The strategies
were chosen and tuned on ~10,000 pairs of neighbouring mother machine cells
(merged artificially, so the true split is known); mean IoU with the true cells:

* :func:`split_by_guides` (0.96): the masks of the two cells in the previous
  frame are realigned to the merged mask (anchored at the closed end of the
  trench, stretched to its length) and each pixel goes to the nearer guide.
* :func:`split_by_watershed` with an image (0.96, the most precise cut): two
  markers from the constriction of the mask's width profile, moved to the
  darkest point of their half, then a watershed on the smoothed image, so the
  cut follows the septum (a bright ridge between dark cells, or the reverse).
  Without an image the watershed runs on the distance transform (0.95).
* :func:`split_by_intensity` (0.94): the septum as the strongest ridge (or
  valley) of the intensity profile along the cell's long axis, with a mild
  preference for cuts near the middle.
* :func:`split_by_waist` (0.92) and :func:`split_at_midpoint` (0.80): geometry
  only, as last resorts.

:func:`split_mask` combines them: the image cut is used when it is confident or
agrees with the guides; the guides decide when the image cut is weak, or when the
cells sit side by side across the trench, where a cut across the axis fails.
Parts are returned in guide order when guides were used (``guide_order``), else
with the part nearer the start of the axis (the closed end of the trench) first.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from skimage.segmentation import watershed

Parts = tuple[np.ndarray, np.ndarray]

_CROSS = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)

# a septum this much brighter (dark cells) or darker (bright cells) than the cell
# interior is trusted over the lineage guides
CONFIDENT_CONTRAST = 1.3
# image and guide cuts this similar are taken to agree
AGREEMENT_IOU = 0.8
# a merged mask this much wider than the cells in the previous frame holds cells
# side by side, where a cut across the long axis is wrong
SIDE_BY_SIDE_RATIO = 1.4


@dataclass(frozen=True)
class SplitResult:
    """Two parts of a mask and how they were found.

    Attributes:
        parts: The two boolean masks.
        method: "guides", "watershed", "intensity", "waist" or "midpoint".
        guide_order: True when parts[i] is the part of guides[i].
        contrast: The septum contrast along the cut (see :func:`cut_contrast`),
            when an image was available.
    """

    parts: Parts
    method: str
    guide_order: bool = False
    contrast: float | None = None


# ----------------------------------------------------------------------
# geometry helpers
# ----------------------------------------------------------------------
def principal_axis(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Centroid and unit direction of the mask's long axis.

    The direction points towards increasing y (down the trench, away from its
    closed end). Degenerate masks (a line or a point) fall back to the y axis.
    """
    ys, xs = np.nonzero(mask)
    centroid = np.array([ys.mean(), xs.mean()])
    direction = np.array([1.0, 0.0])
    if len(ys) >= 3:
        cov = np.cov(np.vstack([ys, xs]))
        if np.all(np.isfinite(cov)):
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            if eigenvalues[-1] > 1e-9:
                direction = eigenvectors[:, -1]
    if direction[0] < 0 or (direction[0] == 0 and direction[1] < 0):
        direction = -direction
    return centroid, direction / np.linalg.norm(direction)


def axis_coordinates(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Coordinate of every mask pixel along the long axis (relative to the first
    pixel), with the pixel indices."""
    ys, xs = np.nonzero(mask)
    centroid, direction = principal_axis(mask)
    coords = (ys - centroid[0]) * direction[0] + (xs - centroid[1]) * direction[1]
    coords = coords - coords.min()
    return coords, ys, xs


def _cut(mask: np.ndarray, coords: np.ndarray, ys, xs, threshold: float) -> Parts:
    part_a = np.zeros_like(mask, dtype=bool)
    part_b = np.zeros_like(mask, dtype=bool)
    below = coords < threshold
    part_a[ys[below], xs[below]] = True
    part_b[ys[~below], xs[~below]] = True
    return part_a, part_b


def _window(length: int, window: tuple[float, float]) -> tuple[int, int]:
    """Index range of the central part of a profile where a cut may go."""
    start = int(np.floor(length * window[0]))
    stop = int(np.ceil(length * window[1]))
    start = min(max(start, 1), length - 1)
    stop = max(min(stop, length - 1), start + 1)
    return start, stop


def _clean(mask: np.ndarray, part_a: np.ndarray, part_b: np.ndarray) -> Parts | None:
    """Make both parts non-empty and tidy: a fragment of one part that is
    disconnected from the rest of that part and touches the other part joins the
    other part. Returns None when a part ends up empty."""
    parts = [part_a & mask, part_b & mask]
    for index in (0, 1):
        own, other = parts[index], parts[1 - index]
        labels, count = ndimage.label(own, structure=_CROSS)
        if count <= 1:
            continue
        sizes = ndimage.sum(own, labels, index=range(1, count + 1))
        largest = int(np.argmax(sizes)) + 1
        for component in range(1, count + 1):
            if component == largest:
                continue
            fragment = labels == component
            if (ndimage.binary_dilation(fragment, structure=_CROSS) & other).any():
                own[fragment] = False
                other[fragment] = True
    part_a, part_b = parts
    if not part_a.any() or not part_b.any():
        return None
    return part_a, part_b


def _shift_mask(mask: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Translate a boolean mask by whole pixels, dropping what leaves the frame."""
    shifted = np.zeros_like(mask, dtype=bool)
    ys, xs = np.nonzero(mask)
    ys, xs = ys + dy, xs + dx
    inside = (ys >= 0) & (ys < mask.shape[0]) & (xs >= 0) & (xs < mask.shape[1])
    shifted[ys[inside], xs[inside]] = True
    return shifted


def _row_extent(mask: np.ndarray) -> tuple[int, int] | None:
    rows = np.nonzero(mask.any(axis=1))[0]
    return (int(rows[0]), int(rows[-1])) if rows.size else None


def _centroid(mask: np.ndarray) -> tuple[float, float]:
    ys, xs = np.nonzero(mask)
    return float(ys.mean()), float(xs.mean())


def _order_along_axis(mask: np.ndarray, parts: Parts) -> Parts:
    """Put the part nearer the start of the long axis first."""
    _, direction = principal_axis(mask)
    positions = []
    for part in parts:
        cy, cx = _centroid(part)
        positions.append(cy * direction[0] + cx * direction[1])
    return parts if positions[0] <= positions[1] else (parts[1], parts[0])


def _as_image(image: np.ndarray) -> np.ndarray:
    """The image as floats with NaN/inf replaced by the median of the rest."""
    image = np.asarray(image, dtype=float)
    finite = np.isfinite(image)
    if finite.all():
        return image
    fill = float(np.median(image[finite])) if finite.any() else 0.0
    return np.where(finite, image, fill)


# ----------------------------------------------------------------------
# polarity
# ----------------------------------------------------------------------
def cells_are_bright(mask: np.ndarray, image: np.ndarray) -> bool:
    """Whether the cell is brighter than its surroundings (a 2-px ring)."""
    image = _as_image(image)
    ring = ndimage.binary_dilation(mask, iterations=2) & ~mask
    inside = float(image[mask].mean())
    outside = float(image[ring].mean()) if ring.any() else inside
    return inside > outside


def bright_cells_in_frame(image: np.ndarray, labels: np.ndarray) -> bool:
    """Whether cells are brighter than the background of a labelled frame.

    The background is taken inside the bounding box of all labels (the lumen of a
    trench), so that dark walls outside it do not count.
    """
    image = _as_image(image)
    labels = np.asarray(labels)
    cells = labels > 0
    if not cells.any():
        return True
    ys, xs = np.nonzero(cells)
    box = np.zeros_like(cells)
    margin = 2  # the lumen reaches a little beyond the cells
    box[
        max(0, ys.min() - margin) : ys.max() + margin + 1,
        max(0, xs.min() - margin) : xs.max() + margin + 1,
    ] = True
    background = box & ~cells
    if not background.any():
        background = ~cells
    if not background.any():
        return True
    return float(np.median(image[cells])) > float(np.median(image[background]))


# ----------------------------------------------------------------------
# guides: the two cells as seen in the previous frame
# ----------------------------------------------------------------------
def _usable_guides(
    mask: np.ndarray, guides: tuple[np.ndarray, np.ndarray] | None, far_margin: int = 4
) -> Parts | None:
    """The guides as boolean masks, or None when they carry no information:
    missing, empty, identical (one parent for both cells) or far from the mask
    (a wrong relative)."""
    if guides is None:
        return None
    try:
        guide_a, guide_b = guides
    except (TypeError, ValueError):
        return None
    if guide_a is None or guide_b is None:
        return None
    guide_a = np.asarray(guide_a, dtype=bool)
    guide_b = np.asarray(guide_b, dtype=bool)
    if guide_a.shape != mask.shape or guide_b.shape != mask.shape:
        return None
    if not guide_a.any() or not guide_b.any() or np.array_equal(guide_a, guide_b):
        return None
    near = ndimage.binary_dilation(mask, iterations=far_margin)
    if not (guide_a & near).any() or not (guide_b & near).any():
        return None
    return guide_a, guide_b


def _remap_rows(
    guide: np.ndarray, source_rows: np.ndarray, valid: np.ndarray
) -> np.ndarray:
    out = np.zeros_like(guide)
    out[valid] = guide[source_rows[valid]]
    return out


def _align_guides(
    guide_a: np.ndarray,
    guide_b: np.ndarray,
    mask: np.ndarray,
    *,
    clamp: tuple[float, float] = (0.7, 1.4),
    sibling_ratio: float = 1.2,
) -> Parts:
    """Move the guides onto the mask, along the trench.

    The alignment is anchored at the closed end of the trench (the smallest row
    of the guides' union), the end that does not move, and stretched by the
    ratio of the row extents (cells grow) unless the mask is clipped by the
    frame's far edge. When the guides cover far more than the mask, one of them
    is a parent that also covers a sibling outside the mask (that cell divided);
    then the guides are only shifted, anchored at the end opposite the excess.
    Across the trench the guides are shifted to the mask's centroid.
    """
    union = guide_a | guide_b
    guide_extent, mask_extent = _row_extent(union), _row_extent(mask)
    if guide_extent is None or mask_extent is None:
        return guide_a, guide_b
    height = mask.shape[0]
    rows = np.arange(height)
    clipped = mask_extent[1] >= height - 1
    (_, guide_cx), (_, mask_cx) = _centroid(union), _centroid(mask)
    anchor_guide, anchor_mask = guide_extent[0], mask_extent[0]

    if not clipped and union.sum() > sibling_ratio * mask.sum():
        excess_top = mask_extent[0] - guide_extent[0]
        excess_bottom = guide_extent[1] - mask_extent[1]
        if excess_top > excess_bottom:
            anchor_guide, anchor_mask = guide_extent[1], mask_extent[1]
        band = np.zeros(height, dtype=bool)
        band[max(0, mask_extent[0] - 3) : min(height, mask_extent[1] + 4)] = True
        in_band = union & band[:, None]
        if in_band.any():
            guide_cx = _centroid(in_band)[1]
        dx = int(np.rint(mask_cx - guide_cx))
        source = rows - anchor_mask + anchor_guide
        valid = (source >= 0) & (source < height)
        out_a = _shift_mask(_remap_rows(guide_a, source, valid), 0, dx)
        out_b = _shift_mask(_remap_rows(guide_b, source, valid), 0, dx)
        if out_a.any() and out_b.any():
            return out_a, out_b

    scale = 1.0
    if not clipped and guide_extent[1] > guide_extent[0]:
        ratio = (mask_extent[1] - mask_extent[0]) / (guide_extent[1] - guide_extent[0])
        scale = float(np.clip(ratio, *clamp))
    source = np.rint(anchor_guide + (rows - anchor_mask) / scale).astype(int)
    valid = (source >= 0) & (source < height)
    dx = int(np.rint(mask_cx - guide_cx))
    out_a = _shift_mask(_remap_rows(guide_a, source, valid), 0, dx)
    out_b = _shift_mask(_remap_rows(guide_b, source, valid), 0, dx)
    if out_a.any() and out_b.any():
        return out_a, out_b

    # fall back to a plain shift of the centroids
    (guide_cy, guide_cx), (mask_cy, mask_cx) = _centroid(union), _centroid(mask)
    dy, dx = int(np.rint(mask_cy - guide_cy)), int(np.rint(mask_cx - guide_cx))
    out_a, out_b = _shift_mask(guide_a, dy, dx), _shift_mask(guide_b, dy, dx)
    if out_a.any() and out_b.any():
        return out_a, out_b
    return guide_a, guide_b


def _reattach_strays(part_a, part_b, guide_a, guide_b, max_passes: int = 3) -> Parts:
    """Make each part one connected component: the component overlapping its
    guide most is kept, the others are handed to the other part."""
    for _ in range(max_passes):
        changed = False
        for own, other, guide in ((part_a, part_b, guide_a), (part_b, part_a, guide_b)):
            labels, count = ndimage.label(own, structure=_CROSS)
            if count <= 1:
                continue
            index = np.arange(1, count + 1)
            overlaps = ndimage.sum(guide, labels, index=index)
            sizes = ndimage.sum(own, labels, index=index)
            main = int(np.lexsort((sizes, overlaps))[-1]) + 1
            stray = own & (labels != main)
            own[stray] = False
            other[stray] = True
            changed = True
        if not changed:
            break
    return part_a, part_b


def _rowcut_between_guides(mask, guide_a, guide_b) -> Parts | None:
    """Last resort with guides: a straight cut between their row extents."""
    extent_a, extent_b = _row_extent(guide_a), _row_extent(guide_b)
    if extent_a is None or extent_b is None:
        return None
    upper, lower = (
        (extent_a, extent_b)
        if _centroid(guide_a)[0] <= _centroid(guide_b)[0]
        else (extent_b, extent_a)
    )
    cut = (upper[1] + lower[0] + 1) / 2.0
    rows = np.arange(mask.shape[0])[:, None]
    part_a = mask & (rows < cut)
    part_b = mask & (rows >= cut)
    if _centroid(guide_a)[0] > _centroid(guide_b)[0]:
        part_a, part_b = part_b, part_a
    if not part_a.any() or not part_b.any():
        return None
    return part_a, part_b


def split_by_guides(
    mask: np.ndarray, guides: tuple[np.ndarray, np.ndarray]
) -> Parts | None:
    """Assign every pixel to the nearer of the two guide masks (the two cells as
    seen in the previous frame), after aligning the guides to the mask. Parts
    come back in guide order. None when the guides are unusable."""
    prepared = _usable_guides(mask, guides)
    if prepared is None:
        return None
    guide_a, guide_b = _align_guides(*prepared, mask)
    distance_a = ndimage.distance_transform_edt(~guide_a)
    distance_b = ndimage.distance_transform_edt(~guide_b)
    part_a = mask & (distance_a <= distance_b)
    part_b = mask & ~part_a
    part_a, part_b = _reattach_strays(part_a, part_b, guide_a, guide_b)
    if not part_a.any() or not part_b.any():
        return _rowcut_between_guides(mask, guide_a, guide_b)
    return part_a, part_b


# ----------------------------------------------------------------------
# watershed
# ----------------------------------------------------------------------
def _crop(mask: np.ndarray, pad: int = 1) -> tuple[slice, slice]:
    ys, xs = np.nonzero(mask)
    height, width = mask.shape
    return (
        slice(max(0, ys.min() - pad), min(height, ys.max() + pad + 1)),
        slice(max(0, xs.min() - pad), min(width, xs.max() + pad + 1)),
    )


def _smooth_in_mask(values: np.ndarray, mask: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian smoothing that only mixes pixels inside the mask, so that the
    background does not bleed into the cell values."""
    if sigma <= 0:
        return values.astype(float)
    weights = mask.astype(float)
    numerator = ndimage.gaussian_filter(values.astype(float) * weights, sigma)
    denominator = ndimage.gaussian_filter(weights, sigma)
    return np.where(denominator > 1e-6, numerator / np.maximum(denominator, 1e-6), 0.0)


def _profile_markers(
    distance: np.ndarray, mask: np.ndarray, trim: int = 3
) -> np.ndarray | None:
    """Two markers from the constriction of the half-width profile along the
    rows: the row with the deepest neck relative to the lower of its two flanking
    maxima, then the widest point of the mask above it and below it."""
    profile = np.where(mask, distance, 0.0).max(axis=1)
    rows = np.nonzero(profile > 0)[0]
    if rows.size < 2 * trim + 2:
        return None
    segment = profile[rows[0] : rows[-1] + 1]
    left = np.maximum.accumulate(segment)
    right = np.maximum.accumulate(segment[::-1])[::-1]
    depth = (np.minimum(left, right) - segment)[trim : segment.size - trim]
    candidates = np.nonzero(depth >= depth.max() - 1e-9)[0]
    # ties are broken towards the middle
    cut = (
        rows[0]
        + trim
        + int(candidates[np.argmin(np.abs(candidates - (depth.size - 1) / 2.0))])
    )

    markers = np.zeros(mask.shape, dtype=np.int32)
    upper = np.where(mask, distance, -1.0)
    upper[cut:, :] = -1.0
    lower = np.where(mask, distance, -1.0)
    lower[: cut + 1, :] = -1.0
    if upper.max() < 0 or lower.max() < 0:
        return None
    markers[np.unravel_index(int(np.argmax(upper)), mask.shape)] = 1
    markers[np.unravel_index(int(np.argmax(lower)), mask.shape)] = 2
    return markers


def split_by_watershed(
    mask: np.ndarray,
    image: np.ndarray | None = None,
    *,
    bright_cells: bool | None = None,
    sigma: float = 0.7,
) -> Parts | None:
    """Watershed from two markers found at the mask's constriction.

    With an image the flooding follows the smoothed image (inverted for bright
    cells), so the cut settles on the septum; each marker is first moved to the
    darkest point of its half so that neither flood engulfs the other. Without an
    image the distance transform is the landscape. None for masks of fewer than
    eight rows."""
    mask = np.asarray(mask, dtype=bool)
    if mask.sum() < 4:
        return None
    window = _crop(mask, pad=1)
    cropped = mask[window]
    if np.count_nonzero(cropped.any(axis=1)) < 4:
        return None
    distance = ndimage.distance_transform_edt(np.pad(cropped, 1))[1:-1, 1:-1]
    markers = _profile_markers(distance, cropped)
    if markers is None:
        return None

    if image is None:
        elevation = -np.where(cropped, distance, 0.0)
    else:
        values = _as_image(image)[window]
        if bright_cells is None:
            bright_cells = cells_are_bright(mask, image)
        smooth = _smooth_in_mask(values, cropped, sigma)
        elevation = -smooth if bright_cells else smooth
        # move each marker to the lowest point of its own distance basin
        basins = watershed(-distance, markers=markers, mask=cropped)
        relocated = np.zeros_like(markers)
        for label in (1, 2):
            region = basins == label
            if region.any():
                field = np.where(region, elevation, np.inf)
                relocated[np.unravel_index(int(np.argmin(field)), field.shape)] = label
        if (relocated == 1).any() and (relocated == 2).any():
            markers = relocated
    elevation = np.where(cropped, elevation, elevation[cropped].max() + 1.0)
    labels = watershed(elevation, markers=markers, mask=cropped)
    if not (labels == 1).any() or not (labels == 2).any():
        return None
    part_a = np.zeros_like(mask)
    part_b = np.zeros_like(mask)
    part_a[window] = labels == 1
    part_b[window] = labels == 2
    parts = _clean(mask, part_a, part_b)
    return None if parts is None else _order_along_axis(mask, parts)


# ----------------------------------------------------------------------
# septum in the intensity profile
# ----------------------------------------------------------------------
def _axis_profile(mask: np.ndarray, image: np.ndarray, half_band: float) -> dict:
    """Mean intensity per unit step along the long axis, over a band of
    ``2 * half_band + 1`` pixels around the mask's centre line."""
    ys, xs = np.nonzero(mask)
    centroid, direction = principal_axis(mask)
    dy, dx = ys - centroid[0], xs - centroid[1]
    along = dy * direction[0] + dx * direction[1]
    across = dx * direction[0] - dy * direction[1]
    along = along - float(np.floor(along.min() + 0.5))
    bins = np.rint(along).astype(int)
    length = int(bins.max()) + 1
    counts = np.bincount(bins, minlength=length).astype(float)
    values = _as_image(image)[ys, xs]
    with np.errstate(invalid="ignore", divide="ignore"):
        centre = np.bincount(bins, weights=across, minlength=length) / counts
        in_band = np.abs(across - centre[bins]) <= half_band
        band_counts = np.bincount(bins[in_band], minlength=length).astype(float)
        profile = (
            np.bincount(bins[in_band], weights=values[in_band], minlength=length)
            / band_counts
        )
        whole = np.bincount(bins, weights=values, minlength=length) / counts
    bad = ~np.isfinite(profile)
    profile[bad] = whole[bad]
    bad = ~np.isfinite(profile)
    if bad.any():
        good = np.nonzero(~bad)[0]
        profile[bad] = np.interp(np.nonzero(bad)[0], good, profile[good])
    return {"along": along, "ys": ys, "xs": xs, "profile": profile, "counts": counts}


def septum_position(
    mask: np.ndarray,
    image: np.ndarray,
    *,
    bright_cells: bool | None = None,
    window: tuple[float, float] = (0.15, 0.85),
    sigma: float = 0.5,
    balance: float = 0.5,
    half_band: float = 2.0,
    pole_trim: int = 2,
) -> dict | None:
    """Where the septum is along the long axis: the strongest ridge (dark cells)
    or valley (bright cells) of the centre-line intensity profile within the
    central ``window``, with a penalty of ``balance`` per half length for cuts far
    from the area median. Returns the position, the profile data and the ridge
    ratio (extremum over the profile median), or None for very short masks."""
    if bright_cells is None:
        bright_cells = cells_are_bright(mask, image)
    data = _axis_profile(mask, image, half_band)
    profile = data["profile"]
    length = len(profile)
    if length < max(2 * pole_trim + 2, 4):
        return None
    smooth = (
        ndimage.gaussian_filter1d(profile, sigma, mode="nearest")
        if sigma > 0
        else profile
    )
    if smooth.min() <= 0:  # ratios need positive values (background-subtracted images)
        smooth = smooth - smooth.min() + 1.0
    median = float(np.median(smooth))
    score = (1.0 - smooth / median) if bright_cells else (smooth / median - 1.0)

    low = max(int(np.ceil(window[0] * (length - 1))), pole_trim)
    high = min(int(np.floor(window[1] * (length - 1))), length - 1 - pole_trim)
    if high < low:
        low, high = pole_trim, length - 1 - pole_trim
        if high < low:
            return None
    index = np.arange(low, high + 1)
    total = score[index].copy()
    if balance > 0:
        cumulative = np.cumsum(data["counts"])
        reference = float(np.searchsorted(cumulative, 0.5 * cumulative[-1]))
        total -= balance * np.abs(index - reference) / max(0.5 * (length - 1), 1.0)
    best = int(index[int(np.argmax(total))])

    # sub-pixel refinement of the extremum
    extremum = -smooth if bright_cells else smooth
    position = float(best)
    if 0 < best < length - 1:
        y0, y1, y2 = (float(v) for v in extremum[best - 1 : best + 2])
        denominator = y0 - 2.0 * y1 + y2
        if denominator != 0.0:
            position = best + float(np.clip(0.5 * (y0 - y2) / denominator, -0.5, 0.5))
    position = float(np.clip(position, low - 0.5, high + 0.5))
    ratio = float(smooth[best] / median)
    return {
        "position": position,
        "ridge_ratio": ratio
        if not bright_cells
        else (1.0 / ratio if ratio > 0 else 1.0),
        **data,
    }


def split_by_intensity(
    mask: np.ndarray, image: np.ndarray, *, bright_cells: bool | None = None, **kwargs
) -> Parts | None:
    """Cut perpendicular to the long axis at the septum found by
    :func:`septum_position`. None when the mask is too short."""
    info = septum_position(mask, image, bright_cells=bright_cells, **kwargs)
    if info is None:
        return None
    below = info["along"] < info["position"]
    part_a = np.zeros_like(mask, dtype=bool)
    part_b = np.zeros_like(mask, dtype=bool)
    part_a[info["ys"][below], info["xs"][below]] = True
    part_b[info["ys"][~below], info["xs"][~below]] = True
    return _clean(mask, part_a, part_b)


# ----------------------------------------------------------------------
# geometry only
# ----------------------------------------------------------------------
def split_by_waist(
    mask: np.ndarray, *, window: tuple[float, float] = (0.2, 0.8)
) -> Parts | None:
    """Cut at the narrowest cross-section along the long axis (the constriction
    of a dividing cell), within the central `window` of the length."""
    coords, ys, xs = axis_coordinates(mask)
    widths = np.bincount(np.floor(coords).astype(int))
    length = len(widths)
    if length < 4:
        return None
    start, stop = _window(length, window)
    candidates = widths[start:stop].astype(float)
    middle = (length - 1) / 2.0
    positions = np.arange(start, stop)
    order = np.lexsort((np.abs(positions - middle), candidates))
    index = int(positions[order[0]])
    return _clean(mask, *_cut(mask, coords, ys, xs, index + 0.5))


def split_at_midpoint(mask: np.ndarray) -> Parts | None:
    """Cut half way along the long axis."""
    coords, ys, xs = axis_coordinates(mask)
    if len(coords) < 2:
        return None
    return _clean(mask, *_cut(mask, coords, ys, xs, coords.max() / 2.0 + 0.5))


# ----------------------------------------------------------------------
# combining the cues
# ----------------------------------------------------------------------
def cut_contrast(
    mask: np.ndarray, image: np.ndarray, parts: Parts, bright_cells: bool | None = None
) -> float:
    """How septum-like the boundary between two parts is: the mean intensity of
    the pixels on either side of the cut over the median of the whole mask, for
    dark cells (inverted for bright cells), so that 1 means no contrast."""
    image = _as_image(image)
    if bright_cells is None:
        bright_cells = cells_are_bright(mask, image)
    part_a, part_b = parts
    boundary = (ndimage.binary_dilation(part_a, structure=_CROSS) & part_b) | (
        ndimage.binary_dilation(part_b, structure=_CROSS) & part_a
    )
    if not boundary.any():
        return 1.0
    median = float(np.median(image[mask]))
    if median <= 0:
        return 1.0
    ratio = float(image[boundary].mean()) / median
    return (1.0 / ratio if ratio > 0 else 1.0) if bright_cells else ratio


def partition_agreement(parts: Parts, reference: Parts) -> float:
    """How similar two splits of one mask are: the mean IoU of matched parts."""

    def iou(a, b):
        union = np.count_nonzero(a | b)
        return np.count_nonzero(a & b) / union if union else 0.0

    straight = (iou(parts[0], reference[0]) + iou(parts[1], reference[1])) / 2.0
    swapped = (iou(parts[0], reference[1]) + iou(parts[1], reference[0])) / 2.0
    return max(straight, swapped)


def _order_like(parts: Parts, reference: Parts) -> Parts:
    """Order `parts` so that parts[i] matches reference[i]."""
    straight = np.count_nonzero(parts[0] & reference[0]) + np.count_nonzero(
        parts[1] & reference[1]
    )
    swapped = np.count_nonzero(parts[0] & reference[1]) + np.count_nonzero(
        parts[1] & reference[0]
    )
    return parts if straight >= swapped else (parts[1], parts[0])


def _side_by_side(mask: np.ndarray, guides: Parts) -> bool:
    """Whether the mask is much wider across the long axis than either guide,
    i.e. holds the two cells next to each other rather than in a row."""
    _, direction = principal_axis(mask)

    def across_extent(m):
        ys, xs = np.nonzero(m)
        across = xs * direction[0] - ys * direction[1]
        return float(across.max() - across.min()) + 1.0

    return across_extent(mask) > SIDE_BY_SIDE_RATIO * max(
        across_extent(guides[0]), across_extent(guides[1])
    )


def split_mask(
    mask: np.ndarray,
    image: np.ndarray | None = None,
    guides: tuple[np.ndarray, np.ndarray] | None = None,
    *,
    bright_cells: bool | None = None,
) -> SplitResult | None:
    """Split a mask into two cells with the best information available.

    Args:
        mask: 2-D boolean mask of the merged cell.
        image: The raw 2-D image of the same frame, if available.
        guides: The masks of the two cells in the previous frame, if known.
        bright_cells: Whether cells are brighter than the background. Estimated
            from the mask's surroundings when None; prefer passing the value
            measured on the whole frame (:func:`bright_cells_in_frame`).

    Returns:
        SplitResult with the two parts and the name of the strategy that produced
        them, or None if the mask cannot be split (fewer than two pixels).
    """
    mask = np.asarray(mask, dtype=bool)
    if mask.sum() < 2:
        return None
    if image is not None and bright_cells is None:
        bright_cells = cells_are_bright(mask, image)

    usable = _usable_guides(mask, guides)
    guided = split_by_guides(mask, usable) if usable is not None else None

    if image is not None:
        parts = split_by_watershed(mask, image, bright_cells=bright_cells)
        method = "watershed"
        if parts is None:
            parts = split_by_intensity(mask, image, bright_cells=bright_cells)
            method = "intensity"
        if parts is not None:
            contrast = cut_contrast(mask, image, parts, bright_cells)
            if guided is None:
                return SplitResult(parts, method, False, contrast)
            agrees = partition_agreement(parts, guided) >= AGREEMENT_IOU
            confident = contrast >= CONFIDENT_CONTRAST and not _side_by_side(
                mask, usable
            )
            if agrees or confident:
                return SplitResult(_order_like(parts, guided), method, True, contrast)
            return SplitResult(
                guided, "guides", True, cut_contrast(mask, image, guided, bright_cells)
            )

    if guided is not None:
        return SplitResult(guided, "guides", True, None)
    parts = split_by_watershed(mask)
    if parts is not None:
        return SplitResult(parts, "watershed", False, None)
    parts = split_by_waist(mask)
    if parts is not None:
        return SplitResult(parts, "waist", False, None)
    parts = split_at_midpoint(mask)
    if parts is not None:
        return SplitResult(parts, "midpoint", False, None)
    return None
