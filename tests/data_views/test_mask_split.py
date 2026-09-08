"""Tests for splitting one mask that covers two cells (pure array functions)."""

import numpy as np
import pytest

from motile_tracker.data_views.views_coordinator.mask_split import (
    principal_axis,
    split_at_midpoint,
    split_by_guides,
    split_by_intensity,
    split_by_waist,
    split_mask,
)


def _two_cells(shape=(60, 20), top=(5, 25), bottom=(25, 50), columns=(6, 14)):
    """Two rectangles stacked along y (the trench axis) that touch at a row."""
    cell_a = np.zeros(shape, dtype=bool)
    cell_b = np.zeros(shape, dtype=bool)
    cell_a[top[0] : top[1], columns[0] : columns[1]] = True
    cell_b[bottom[0] : bottom[1], columns[0] : columns[1]] = True
    return cell_a, cell_b


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
def test_intensity_cut_finds_the_septum(bright_cells):
    cell_a, cell_b = _two_cells()
    merged = cell_a | cell_b
    image = np.full(merged.shape, 10.0 if bright_cells else 200.0)
    image[merged] = 200.0 if bright_cells else 10.0
    # the septum: one row between the cells with intensity like the background
    image[24:26, :] = 10.0 if bright_cells else 200.0
    image += np.random.default_rng(0).normal(0, 2.0, size=image.shape)

    parts = split_by_intensity(merged, image)

    assert parts is not None
    assert _matches(parts, cell_a, cell_b)


def test_intensity_cut_gives_up_on_a_flat_profile():
    cell_a, cell_b = _two_cells()
    merged = cell_a | cell_b
    image = np.where(merged, 200.0, 10.0)  # no septum at all

    assert split_by_intensity(merged, image) is None


def test_guides_assign_pixels_to_the_nearer_previous_cell():
    cell_a, cell_b = _two_cells()
    merged = cell_a | cell_b
    # the cells were 3 px higher one frame earlier and a bit shorter
    guide_a = np.roll(cell_a, -3, axis=0)
    guide_b = np.roll(cell_b, -3, axis=0)
    guide_b[45:50] = False

    parts = split_by_guides(merged, (guide_a, guide_b))

    assert parts is not None
    assert _matches(parts, cell_a, cell_b)
    # parts come back in guide order
    assert np.count_nonzero(parts[0] & cell_a) > np.count_nonzero(parts[0] & cell_b)


def test_guides_need_two_non_empty_masks():
    cell_a, cell_b = _two_cells()
    empty = np.zeros_like(cell_a)

    assert split_by_guides(cell_a | cell_b, (cell_a, empty)) is None


def test_waist_cut_finds_the_constriction():
    mask = np.zeros((60, 20), dtype=bool)
    mask[5:50, 5:15] = True
    mask[27:29, 5:8] = False  # pinch in from the left around row 27-28
    mask[27:29, 12:15] = False  # and from the right

    parts = split_by_waist(mask)

    assert parts is not None
    upper, lower = parts
    assert upper.any() and lower.any()
    assert np.nonzero(upper)[0].max() <= 28
    assert np.nonzero(lower)[0].min() >= 27


def test_midpoint_cut_halves_the_mask():
    cell_a, cell_b = _two_cells(top=(10, 30), bottom=(30, 50))

    parts = split_at_midpoint(cell_a | cell_b)

    assert parts is not None
    assert _matches(parts, cell_a, cell_b)


def test_split_mask_cascade_and_method_names():
    cell_a, cell_b = _two_cells()
    merged = cell_a | cell_b
    image = np.where(merged, 200.0, 10.0)
    image[24:26, :] = 10.0

    with_guides = split_mask(merged, image=image, guides=(cell_a, cell_b))
    assert with_guides.method == "guides"
    with_image = split_mask(merged, image=image)
    assert with_image.method == "intensity"
    geometry_only = split_mask(merged)
    assert geometry_only.method in {"waist", "midpoint"}
    for result in (with_guides, with_image, geometry_only):
        part_a, part_b = result.parts
        assert not (part_a & part_b).any()
        assert np.array_equal(part_a | part_b, merged)


def test_split_mask_rejects_tiny_masks():
    mask = np.zeros((10, 10), dtype=bool)
    mask[3, 3] = True
    assert split_mask(mask) is None


def test_stray_fragments_join_the_part_they_touch():
    """A pixel of one cell that ends up on the wrong side of the cut is moved to
    the part it is connected to."""
    cell_a, cell_b = _two_cells()
    merged = cell_a | cell_b
    guide_a = cell_a.copy()
    guide_b = cell_b.copy()
    # a guide with a stray pixel far down in cell_b's territory
    guide_a[45, 6] = True
    guide_b[45, 6] = False

    parts = split_by_guides(merged, (guide_a, guide_b))

    assert parts is not None
    assert _matches(parts, cell_a, cell_b, min_iou=0.98)
