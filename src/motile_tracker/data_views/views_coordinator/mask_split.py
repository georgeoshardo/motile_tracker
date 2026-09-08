"""Split one 2-D mask that covers two cells into two masks.

These are pure array functions (numpy, scipy, scikit-image), independent of any
tracks object, so they can be unit-tested and benchmarked on their own.
:func:`split_mask` runs them as a cascade, from the most to the least informed:

1. **guides**: the masks of the two cells in the previous frame. Each pixel goes
   to the nearer guide (after shifting the guides onto the merged mask, since
   cells drift along the trench). This is the case of a division that the
   segmentation lost again.
2. **intensity**: the division septum shows in the intensity profile along the
   cell's long axis (a dip for bright cells, a peak for dark ones).
3. **waist**: the constriction, i.e. the narrowest cross-section along the axis.
4. **midpoint**: half way along the axis, as a last resort.

Parts are returned as two boolean masks. With guides they are in guide order;
otherwise the part with the smaller axis coordinate (top of the trench when the
axis runs along y) comes first.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

Parts = tuple[np.ndarray, np.ndarray]


@dataclass(frozen=True)
class SplitResult:
    parts: Parts
    method: str


# ----------------------------------------------------------------------
# geometry helpers
# ----------------------------------------------------------------------
def principal_axis(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Centroid and unit direction of the mask's long axis.

    The direction is oriented so that it points towards increasing y (down the
    trench). Degenerate masks (a line or a point) fall back to the y axis.
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


def _bin_profile(values: np.ndarray, coords: np.ndarray) -> np.ndarray:
    """Mean of `values` per unit step along the axis; empty bins are interpolated."""
    bins = np.floor(coords).astype(int)
    length = int(bins.max()) + 1
    sums = np.bincount(bins, weights=values, minlength=length)
    counts = np.bincount(bins, minlength=length)
    profile = np.full(length, np.nan)
    filled = counts > 0
    profile[filled] = sums[filled] / counts[filled]
    if not filled.all():
        positions = np.arange(length)
        profile = np.interp(positions, positions[filled], profile[filled])
    return profile


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
        labels, count = ndimage.label(own)
        if count <= 1:
            continue
        sizes = ndimage.sum(own, labels, index=range(1, count + 1))
        largest = int(np.argmax(sizes)) + 1
        for component in range(1, count + 1):
            if component == largest:
                continue
            fragment = labels == component
            touches_other = (ndimage.binary_dilation(fragment) & other).any()
            if touches_other:
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


# ----------------------------------------------------------------------
# strategies
# ----------------------------------------------------------------------
def split_by_guides(
    mask: np.ndarray, guides: tuple[np.ndarray, np.ndarray]
) -> Parts | None:
    """Assign every pixel to the nearer of two guide masks (the two cells as seen
    in the previous frame), after shifting the guides so that their joint centroid
    sits on the merged mask's centroid."""
    guide_a, guide_b = (np.asarray(g, dtype=bool) for g in guides)
    if not guide_a.any() or not guide_b.any():
        return None
    union = guide_a | guide_b
    ys, xs = np.nonzero(mask)
    uys, uxs = np.nonzero(union)
    dy = int(round(ys.mean() - uys.mean()))
    dx = int(round(xs.mean() - uxs.mean()))
    guide_a, guide_b = _shift_mask(guide_a, dy, dx), _shift_mask(guide_b, dy, dx)
    if not guide_a.any() or not guide_b.any():
        return None
    distance_a = ndimage.distance_transform_edt(~guide_a)
    distance_b = ndimage.distance_transform_edt(~guide_b)
    part_a = mask & (distance_a <= distance_b)
    part_b = mask & ~part_a
    return _clean(mask, part_a, part_b)


def split_by_intensity(
    mask: np.ndarray,
    image: np.ndarray,
    *,
    window: tuple[float, float] = (0.2, 0.8),
    sigma: float = 1.0,
    min_depth: float = 0.03,
) -> Parts | None:
    """Cut perpendicular to the long axis where the septum shows in the intensity
    profile: the deepest local dip for bright cells, the highest local peak for
    dark ones, within the central `window` of the length. Returns None when the
    profile has no feature deeper than `min_depth` of the profile's range."""
    coords, ys, xs = axis_coordinates(mask)
    values = np.asarray(image, dtype=float)[ys, xs]
    profile = _bin_profile(values, coords)
    length = len(profile)
    if length < 4:
        return None

    ring = ndimage.binary_dilation(mask, iterations=2) & ~mask
    inside = values.mean()
    outside = np.asarray(image, dtype=float)[ring].mean() if ring.any() else inside
    bright_cells = inside >= outside

    smooth = ndimage.gaussian_filter1d(profile, sigma=sigma)
    baseline = ndimage.gaussian_filter1d(profile, sigma=max(2.0, length / 4.0))
    # the septum is a dip in bright cells and a peak in dark cells: flip so that
    # it is always the maximum of `signal`
    signal = (baseline - smooth) if bright_cells else (smooth - baseline)

    start, stop = _window(length, window)
    index = start + int(np.argmax(signal[start:stop]))
    profile_range = float(profile.max() - profile.min())
    if profile_range <= 0 or signal[index] < min_depth * profile_range:
        return None
    return _clean(mask, *_cut(mask, coords, ys, xs, index + 0.5))


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
    # among equally narrow positions prefer the one closest to the middle
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


def split_mask(
    mask: np.ndarray,
    image: np.ndarray | None = None,
    guides: tuple[np.ndarray, np.ndarray] | None = None,
) -> SplitResult | None:
    """Split a mask into two cells with the best information available.

    Args:
        mask: 2-D boolean mask of the merged cell.
        image: The raw 2-D image of the same frame, if available.
        guides: The masks of the two cells in the neighbouring frame, if known.

    Returns:
        SplitResult with the two parts and the name of the strategy that produced
        them, or None if the mask cannot be split (fewer than two pixels).
    """
    mask = np.asarray(mask, dtype=bool)
    if mask.sum() < 2:
        return None
    if guides is not None:
        parts = split_by_guides(mask, guides)
        if parts is not None:
            return SplitResult(parts, "guides")
    if image is not None:
        parts = split_by_intensity(mask, image)
        if parts is not None:
            return SplitResult(parts, "intensity")
    parts = split_by_waist(mask)
    if parts is not None:
        return SplitResult(parts, "waist")
    parts = split_at_midpoint(mask)
    if parts is not None:
        return SplitResult(parts, "midpoint")
    return None
