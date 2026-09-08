"""Tests for splitting one mask that covers two cells (pure array functions)."""

import numpy as np
import pytest

from motile_tracker.data_views.views_coordinator.mask_split import (
    bright_cells_in_frame,
    cells_are_bright,
    cut_contrast,
    partition_agreement,
    principal_axis,
    split_at_midpoint,
    split_by_guides,
    split_by_intensity,
    split_by_waist,
    split_by_watershed,
    split_mask,
)


def _two_cells(shape=(60, 20), top=(5, 25), bottom=(25, 50), columns=(6, 14)):
    """Two rectangles stacked along y (the trench axis) that touch at a row."""
    cell_a = np.zeros(shape, dtype=bool)
    cell_b = np.zeros(shape, dtype=bool)
    cell_a[top[0] : top[1], columns[0] : columns[1]] = True
    cell_b[bottom[0] : bottom[1], columns[0] : columns[1]] = True
    return cell_a, cell_b


def _image(merged, septum_rows, bright_cells, noise=2.0):
    """A frame with cells and background of opposite brightness and a septum row
    between the cells that looks like background."""
    cell, background = (200.0, 10.0) if bright_cells else (10.0, 200.0)
    image = np.full(merged.shape, background)
    image[merged] = cell
    image[septum_rows[0] : septum_rows[1], :] = background
    image += np.random.default_rng(0).normal(0, noise, size=image.shape)
    return image


def _matches(parts, cell_a, cell_b, min_iou=0.95):
    """Whether the parts are the two cells (in either order), by IoU."""

    def iou(a, b):
        return np.count_nonzero(a & b) / np.count_nonzero(a | b)

    straight = min(iou(parts[0], cell_a), iou(parts[1], cell_b))
    swapped = min(iou(parts[0], cell_b), iou(parts[1], cell_a))
    return max(straight, swapped) >= min_iou


def test_principal_axis_follows_the_long_side():
    mask = np.zeros((40, 40), dtype=bool)
    mask[5:35, 18:22] = True  # tall
    _, direction = principal_axis(mask)
    assert abs(direction[0]) > 0.99  # along y, pointing down
    assert direction[0] > 0

    mask = np.zeros((40, 40), dtype=bool)
    mask[18:22, 5:35] = True  # wide
    _, direction = principal_axis(mask)
    assert abs(direction[1]) > 0.99


@pytest.mark.parametrize("bright_cells", [True, False])
def test_polarity_helpers(bright_cells):
    cell_a, cell_b = _two_cells()
    merged = cell_a | cell_b
    image = _image(merged, (24, 26), bright_cells)
    labels = np.where(cell_a, 1, np.where(cell_b, 2, 0))

    assert cells_are_bright(merged, image) is bright_cells
    assert bright_cells_in_frame(image, labels) is bright_cells


@pytest.mark.parametrize("bright_cells", [True, False])
def test_intensity_cut_finds_the_septum(bright_cells):
    cell_a, cell_b = _two_cells()
    merged = cell_a | cell_b
    image = _image(merged, (24, 26), bright_cells)

    parts = split_by_intensity(merged, image, bright_cells=bright_cells)

    assert parts is not None
    assert _matches(parts, cell_a, cell_b)


@pytest.mark.parametrize("bright_cells", [True, False])
def test_watershed_follows_the_septum(bright_cells):
    cell_a, cell_b = _two_cells(top=(5, 20), bottom=(20, 50))
    merged = cell_a | cell_b
    image = _image(merged, (19, 21), bright_cells)

    parts = split_by_watershed(merged, image, bright_cells=bright_cells)

    assert parts is not None
    assert _matches(parts, cell_a, cell_b)
    # the part nearer the top of the axis comes first
    assert np.nonzero(parts[0])[0].mean() < np.nonzero(parts[1])[0].mean()
    assert cut_contrast(merged, image, parts, bright_cells) > 1.3


def test_watershed_without_image_uses_the_constriction():
    mask = np.zeros((60, 20), dtype=bool)
    mask[5:50, 5:15] = True
    mask[27:29, 5:8] = False  # pinch in from the left around rows 27-28
    mask[27:29, 12:15] = False  # and from the right

    parts = split_by_watershed(mask)

    assert parts is not None
    upper, lower = parts
    assert np.nonzero(upper)[0].max() <= 29
    assert np.nonzero(lower)[0].min() >= 27
    assert np.array_equal(upper | lower, mask)


def test_watershed_needs_a_few_rows():
    mask = np.zeros((10, 10), dtype=bool)
    mask[2:5, 3:7] = True
    assert split_by_watershed(mask) is None


def test_guides_assign_pixels_to_the_nearer_previous_cell():
    cell_a, cell_b = _two_cells()
    merged = cell_a | cell_b
    # the cells were 3 px higher one frame earlier and a bit shorter (they grew)
    guide_a = np.roll(cell_a, -3, axis=0)
    guide_b = np.roll(cell_b, -3, axis=0)
    guide_b[45:50] = False

    parts = split_by_guides(merged, (guide_a, guide_b))

    assert parts is not None
    assert _matches(parts, cell_a, cell_b)
    # parts come back in guide order
    assert np.count_nonzero(parts[0] & cell_a) > np.count_nonzero(parts[0] & cell_b)


def test_guides_are_stretched_to_the_grown_cells():
    """Both cells grew by a third since the previous frame: the alignment is
    anchored at the closed end (top) and scaled, so the cut still lands right."""
    cell_a, cell_b = _two_cells(top=(10, 30), bottom=(30, 50))
    merged = cell_a | cell_b
    guide_a, guide_b = _two_cells(top=(10, 25), bottom=(25, 40))

    parts = split_by_guides(merged, (guide_a, guide_b))

    assert parts is not None
    assert _matches(parts, cell_a, cell_b, min_iou=0.9)


def test_guides_need_two_distinct_nearby_masks():
    cell_a, cell_b = _two_cells()
    merged = cell_a | cell_b
    empty = np.zeros_like(cell_a)
    far = np.zeros_like(cell_a)
    far[55:59, 6:14] = True

    assert split_by_guides(merged, (cell_a, empty)) is None
    assert split_by_guides(merged, (cell_a, cell_a)) is None  # one parent for both
    assert split_by_guides(merged, (cell_a, far)) is None  # a wrong relative


def test_waist_cut_finds_the_constriction():
    mask = np.zeros((60, 20), dtype=bool)
    mask[5:50, 5:15] = True
    mask[27:29, 5:8] = False
    mask[27:29, 12:15] = False

    parts = split_by_waist(mask)

    assert parts is not None
    upper, lower = parts
    assert np.nonzero(upper)[0].max() <= 28
    assert np.nonzero(lower)[0].min() >= 27


def test_midpoint_cut_halves_the_mask():
    cell_a, cell_b = _two_cells(top=(10, 30), bottom=(30, 50))

    parts = split_at_midpoint(cell_a | cell_b)

    assert parts is not None
    assert _matches(parts, cell_a, cell_b)


def test_partition_agreement():
    cell_a, cell_b = _two_cells()
    assert partition_agreement((cell_a, cell_b), (cell_b, cell_a)) == 1.0
    shifted_a, shifted_b = _two_cells(top=(5, 35), bottom=(35, 50))
    assert 0.5 < partition_agreement((cell_a, cell_b), (shifted_a, shifted_b)) < 0.9


def test_split_mask_cascade():
    cell_a, cell_b = _two_cells(top=(5, 20), bottom=(20, 50))
    merged = cell_a | cell_b
    image = _image(merged, (19, 21), bright_cells=False)

    with_guides = split_mask(
        merged, image=image, guides=(cell_b, cell_a), bright_cells=False
    )
    assert with_guides.method == "watershed"  # image cut confirmed by the guides
    assert with_guides.guide_order is True
    assert _matches((with_guides.parts[1], with_guides.parts[0]), cell_a, cell_b)

    with_image = split_mask(merged, image=image)
    assert with_image.method == "watershed"
    assert with_image.guide_order is False
    assert with_image.contrast > 1.3

    guides_only = split_mask(merged, guides=(cell_a, cell_b))
    assert guides_only.method == "guides"
    assert guides_only.guide_order is True

    geometry_only = split_mask(merged)
    assert geometry_only.method in {"watershed", "waist", "midpoint"}
    for result in (with_guides, with_image, guides_only, geometry_only):
        part_a, part_b = result.parts
        assert not (part_a & part_b).any()
        assert np.array_equal(part_a | part_b, merged)


def test_split_mask_prefers_guides_when_the_image_cut_is_weak():
    """No septum in the image and no constriction in the mask: the image cut is
    arbitrary (contrast ~1) and disagrees with the guides, which win."""
    cell_a, cell_b = _two_cells(top=(5, 20), bottom=(20, 50))
    merged = cell_a | cell_b
    image = np.where(merged, 10.0, 200.0)  # flat inside the mask

    result = split_mask(
        merged, image=image, guides=(cell_a, cell_b), bright_cells=False
    )

    assert result.method == "guides"
    assert result.guide_order is True
    assert _matches(result.parts, cell_a, cell_b)


def test_split_mask_trusts_a_clear_septum_over_stale_guides():
    """A clear septum but guides that are off by several rows: the image cut is
    kept (confident), ordered like the guides."""
    cell_a, cell_b = _two_cells(top=(5, 20), bottom=(20, 50))
    merged = cell_a | cell_b
    image = _image(merged, (19, 21), bright_cells=False)
    stale_a, stale_b = _two_cells(top=(5, 30), bottom=(30, 50))

    result = split_mask(
        merged, image=image, guides=(stale_b, stale_a), bright_cells=False
    )

    assert result.method == "watershed"
    assert result.guide_order is True
    assert _matches((result.parts[1], result.parts[0]), cell_a, cell_b)


def test_split_mask_rejects_tiny_masks():
    mask = np.zeros((10, 10), dtype=bool)
    mask[3, 3] = True
    assert split_mask(mask) is None


def test_split_mask_small_mask_falls_through_to_geometry():
    mask = np.zeros((10, 10), dtype=bool)
    mask[2:5, 3:7] = True  # 3 rows: too short for the watershed and the profile
    result = split_mask(mask)
    assert result is not None
    assert result.method in {"waist", "midpoint"}
