"""Tests for the cuts that alternative segmentations offer for one mask (pure array
functions)."""

import numpy as np

from mm_track.data_views.views_coordinator.hypothesis_cut import (
    MIN_PIECE,
    continuity,
    cut_candidates,
    fill_pieces,
    hypothesis_pieces,
    neighbourhood,
)

SHAPE = (80, 20)


def _box(rows, columns=(6, 14)):
    mask = np.zeros(SHAPE, dtype=bool)
    mask[rows[0] : rows[1], columns[0] : columns[1]] = True
    return mask


def _labels(*masks):
    labels = np.zeros(SHAPE, dtype=np.int32)
    for value, mask in enumerate(masks, 1):
        labels[mask] = value
    return labels


MASK = _box((10, 70))  # one mask over cells at rows 10-40 and 40-70
CUT_AT_40 = _labels(_box((10, 40)), _box((40, 70)))
CUT_AT_30 = _labels(_box((10, 30)), _box((30, 70)))


def _tiles(pieces, mask):
    union = np.zeros_like(mask)
    for piece in pieces:
        assert not (union & piece).any()
        union |= piece
    return np.array_equal(union, mask)


def test_a_hypothesis_with_two_masks_inside_cuts_the_mask_in_two():
    pieces = hypothesis_pieces(MASK, CUT_AT_40)

    assert len(pieces) == 2
    assert _tiles(pieces, MASK)
    # closed end first
    assert np.nonzero(pieces[0])[0].max() < np.nonzero(pieces[1])[0].min()


def test_a_hypothesis_that_agrees_with_the_mask_offers_no_cut():
    assert hypothesis_pieces(MASK, _labels(MASK)) == []
    assert cut_candidates(MASK, {"same": _labels(MASK)}) == []


def test_masks_mostly_outside_and_slivers_are_not_pieces():
    # a neighbour that pokes 3 rows into the mask, and a 2-row sliver inside it
    neighbour = _box((67, 80))
    sliver = _box((10, 12))
    assert np.count_nonzero(sliver & MASK) < MIN_PIECE
    labels = _labels(neighbour, sliver, _box((12, 67)))

    assert hypothesis_pieces(MASK, labels) == []


def test_pieces_keep_the_users_outline():
    # the hypothesis draws its cells narrower and leaves a gap between them
    labels = _labels(_box((12, 38), (7, 13)), _box((42, 68), (7, 13)))
    pieces = hypothesis_pieces(MASK, labels)

    assert len(pieces) == 2
    assert _tiles(pieces, MASK)
    assert abs(np.count_nonzero(pieces[0]) - np.count_nonzero(pieces[1])) <= 2 * 8


def test_fill_gives_every_uncovered_pixel_to_the_nearest_piece():
    pieces = fill_pieces(MASK, [_box((10, 20)), _box((60, 70))])

    assert _tiles(pieces, MASK)
    rows_0 = np.nonzero(pieces[0])[0]
    assert rows_0.max() == 39  # the gap is shared at its middle


def test_hypotheses_that_cut_the_same_way_are_one_candidate():
    cuts = cut_candidates(MASK, {"e1": CUT_AT_40, "e2": CUT_AT_40.copy(), "e3": CUT_AT_30})

    assert [c.hypotheses for c in cuts] == [("e1", "e2"), ("e3",)]
    assert all(_tiles(c.pieces, MASK) for c in cuts)


def test_a_cut_that_continues_the_frame_before_ranks_first():
    # the frame before has the two cells split at row 30: the single hypothesis that
    # cuts there beats the two that agree on row 40
    before = CUT_AT_30
    cuts = cut_candidates(MASK, {"e1": CUT_AT_40, "e2": CUT_AT_40, "e3": CUT_AT_30}, before=before)

    assert cuts[0].hypotheses == ("e3",)
    assert cuts[0].frame == "before"
    assert cuts[0].continuity > 0.99
    assert cuts[1].continuity < cuts[0].continuity


def test_the_better_of_the_two_neighbouring_frames_counts():
    cuts = cut_candidates(MASK, {"e1": CUT_AT_40}, before=_labels(MASK), after=CUT_AT_40)

    assert cuts[0].frame == "after"
    assert cuts[0].continuity > 0.99


def test_continuity_of_the_uncut_mask():
    around = neighbourhood(MASK)
    assert continuity([MASK], _labels(MASK), around) > 0.99
    assert continuity([MASK], CUT_AT_40, around) < 0.6
    assert continuity([MASK], None, around) == 0.0
    # cells far from the mask do not count
    assert continuity([MASK], _labels(_box((0, 2), (0, 2))), around) == 0.0


def test_a_cut_into_three_and_side_by_side_pieces():
    three = _labels(_box((10, 30)), _box((30, 50)), _box((50, 70)))
    side_by_side = _labels(_box((10, 70), (6, 10)), _box((10, 70), (10, 14)))

    cuts = cut_candidates(MASK, {"three": three, "sides": side_by_side})

    assert sorted(len(c.pieces) for c in cuts) == [2, 3]
    assert all(_tiles(c.pieces, MASK) for c in cuts)


def test_with_pieces_relabels_the_mask_as_its_pieces():
    from mm_track.data_views.views_coordinator.hypothesis_cut import with_pieces

    frame = _labels(_box((0, 8)), MASK)  # label 1 above, label 2 the mask
    pieces = hypothesis_pieces(MASK, CUT_AT_40)
    out = with_pieces(frame, MASK, pieces)

    assert set(np.unique(out[MASK])) == {3, 4}
    assert np.array_equal(out == 3, pieces[0]) and np.array_equal(out == 4, pieces[1])
    assert np.array_equal(out == 1, frame == 1)
    assert (frame == 2).any()  # the input is left as it was
