"""Splitting one node into two and merging nodes into one, as single undo steps."""

from unittest.mock import patch

import numpy as np
import pytest

from motile_tracker.data_views.keybindings_config import KEYMAP
from motile_tracker.data_views.views_coordinator.node_edits import (
    MergeNodes,
    SplitNode,
    choose_kept_node,
    find_second_parent,
    next_node_id,
    node_mask,
    plan_split,
    propose_split,
)
from motile_tracker.data_views.views_coordinator.tracks_viewer import TracksViewer


@pytest.fixture(autouse=True)
def clear_viewer_layers(viewer):
    yield
    viewer.layers.clear()


@pytest.fixture
def tracks_viewer_setup(viewer, solution_tracks_2d):
    """solution_tracks_2d: nodes 1 (t0) -> 2, 3 (t1); 3 -> 4 (t2) -> 5 (t4); 6 (t4)
    alone. Masks are filled boxes, labels equal node ids."""
    tracks_viewer = TracksViewer.get_instance(viewer)
    tracks_viewer.update_tracks(tracks=solution_tracks_2d, name="test")
    return viewer, tracks_viewer, tracks_viewer.tracks


def _snapshot(tracks):
    """Graph, segmentation and track ids in a directly comparable form."""
    nodes = sorted(int(n) for n in tracks.graph.node_ids())
    edges = sorted((int(s), int(t)) for s, t in tracks.graph.edge_list())
    frames = np.stack([np.asarray(tracks.segmentation[t]) for t in range(5)])
    track_ids = {n: int(tracks.get_track_id(n)) for n in nodes}
    return nodes, edges, frames.tobytes(), track_ids


def _halves(mask):
    """Split a box mask into its upper and lower rows."""
    rows = np.unique(np.nonzero(mask)[0])
    upper = mask.copy()
    upper[rows[len(rows) // 2] :] = False
    lower = mask & ~upper
    return upper, lower


def test_keybindings_for_split_and_merge():
    assert KEYMAP["split_node"] == ["c"]
    assert KEYMAP["merge_nodes"] == ["j"]


def test_next_node_id_skips_soft_deleted_ids(tracks_viewer_setup):
    _, tracks_viewer, tracks = tracks_viewer_setup
    assert next_node_id(tracks) == 7

    tracks_viewer.selected_nodes.add(6)
    tracks_viewer.delete_node()

    # 6 is deleted from the solution but its id stays taken
    assert not tracks.graph.has_node(6)
    assert next_node_id(tracks) == 7


class TestSplit:
    def test_plan_makes_a_division_when_the_parent_has_one_child(
        self, tracks_viewer_setup
    ):
        _, _, tracks = tracks_viewer_setup
        upper, lower = _halves(node_mask(tracks, 4))

        plan = plan_split(tracks, 4, (upper, lower))

        assert plan.node == 4
        # no neighbouring track ends next to node 4, so the parent divides
        assert plan.new_parent is None
        assert plan.parent_division is True
        assert np.array_equal(plan.keep_mask | plan.new_mask, node_mask(tracks, 4))
        # child 5 sits on the same box as node 4: it follows the part it overlaps more
        child = node_mask(tracks, 5)
        overlap_keep = np.count_nonzero(child & plan.keep_mask)
        overlap_new = np.count_nonzero(child & plan.new_mask)
        assert (5 in plan.moved_children) == (overlap_new > overlap_keep)

    def test_plan_keeps_the_label_on_the_parents_side(self, tracks_viewer_setup):
        _, _, tracks = tracks_viewer_setup
        # node 3 (t1) is a child of node 1 (t0); the masks of 1 and 3 overlap
        assert np.count_nonzero(node_mask(tracks, 1) & node_mask(tracks, 3)) > 0
        upper, lower = _halves(node_mask(tracks, 3))
        parent_mask = node_mask(tracks, 1)

        plan = plan_split(tracks, 3, (upper, lower))

        overlap_keep = np.count_nonzero(parent_mask & plan.keep_mask)
        overlap_new = np.count_nonzero(parent_mask & plan.new_mask)
        assert overlap_keep >= overlap_new
        # parent 1 already has two children (2 and 3): the new cell gets no parent
        assert plan.new_parent is None
        assert plan.parent_division is False
        assert any("two children" in note for note in plan.notes)

    def test_find_second_parent_looks_in_the_previous_frame(self, tracks_viewer_setup):
        """The neighbour a merged mask swallowed is a cell of the previous frame
        whose track ends there, next to the mask."""
        _, _, tracks = tracks_viewer_setup
        # nothing ends next to node 4 (t2) in frame 1: node 3 is its parent
        assert find_second_parent(tracks, 4) is None
        # node 2 (t1) has no successor: a mask over its territory finds it
        assert find_second_parent(tracks, 4, mask=node_mask(tracks, 2)) == 2
        # ... but not its parent, and nothing in the first frame
        assert find_second_parent(tracks, 4, mask=node_mask(tracks, 3)) is None
        assert find_second_parent(tracks, 1) is None

    def test_plan_with_owners_keeps_the_parents_part_and_links_the_other(
        self, tracks_viewer_setup
    ):
        _, _, tracks = tracks_viewer_setup
        upper, lower = _halves(node_mask(tracks, 4))

        # the parts came from guides: upper from parent 3, lower from node 2
        plan = plan_split(tracks, 4, (upper, lower), owners=(3, 2), method="guides")

        assert np.array_equal(plan.keep_mask, upper)
        assert np.array_equal(plan.new_mask, lower)
        assert plan.new_parent == 2  # node 2's track ended at t1: it continues here
        assert plan.parent_division is False
        assert plan.method == "guides"

        action = SplitNode(tracks, plan)
        assert [int(p) for p in tracks.predecessors(action.new_node)] == [2]
        assert tracks.get_track_id(action.new_node) == tracks.get_track_id(2)
        assert [int(c) for c in tracks.successors(3)] == [4]
        tracks.undo()
        assert not tracks.graph.has_node(action.new_node)
        assert [int(c) for c in tracks.successors(2)] == []

    def test_split_node_creates_a_second_daughter_and_undoes_in_one_step(
        self, tracks_viewer_setup, click_node
    ):
        viewer, tracks_viewer, tracks = tracks_viewer_setup
        before = _snapshot(tracks)
        original_mask = node_mask(tracks, 4)

        click_node(tracks_viewer, 4)
        tracks_viewer.split_node()

        nodes = sorted(int(n) for n in tracks.graph.node_ids())
        new = [n for n in nodes if n not in before[0]]
        assert len(new) == 1
        new = new[0]
        assert tracks.graph.has_node(4)
        assert tracks.get_time(new) == 2
        # the two masks partition the original one
        frame = np.asarray(tracks.segmentation[2])
        assert np.array_equal((frame == 4) | (frame == new), original_mask)
        assert not ((frame == 4) & (frame == new)).any()
        # parent 3 now divides into 4 and the new cell
        assert {int(c) for c in tracks.successors(3)} == {4, new}
        assert tracks.get_track_id(4) != tracks.get_track_id(new)
        assert tracks.get_lineage_id(4) == tracks.get_lineage_id(new)
        # child 5 has exactly one parent, one of the two halves
        parents_of_5 = [int(p) for p in tracks.predecessors(5)]
        assert len(parents_of_5) == 1 and parents_of_5[0] in {4, new}
        # both halves are selected
        assert set(tracks_viewer.selected_nodes.as_list) == {4, new}

        tracks_viewer.undo()
        assert _snapshot(tracks) == before

        tracks_viewer.redo()
        assert tracks.graph.has_node(new)
        assert {int(c) for c in tracks.successors(3)} == {4, new}

    def test_split_requires_exactly_one_selected_node(
        self, tracks_viewer_setup, click_node
    ):
        _, tracks_viewer, tracks = tracks_viewer_setup
        before = _snapshot(tracks)
        click_node(tracks_viewer, 2)
        click_node(tracks_viewer, 3, append=True)

        with patch(
            "motile_tracker.data_views.views_coordinator.tracks_viewer.show_warning"
        ) as warning:
            tracks_viewer.split_node()

        warning.assert_called_once()
        assert _snapshot(tracks) == before

    def test_propose_split_uses_previous_cells_as_guides(self, tracks_viewer_setup):
        _, _, tracks = tracks_viewer_setup
        # split node 4 (t2) after merging... there is no neighbour: geometry is used
        plan = propose_split(tracks, 4)
        assert plan is not None
        assert plan.method in {"waist", "midpoint"}
        assert plan.keep_mask.any() and plan.new_mask.any()

    def test_split_node_action_rejects_empty_parts(self, tracks_viewer_setup):
        _, _, tracks = tracks_viewer_setup
        from funtracks.exceptions import InvalidActionError

        from motile_tracker.data_views.views_coordinator.node_edits import SplitPlan

        mask = node_mask(tracks, 4)
        plan = SplitPlan(node=4, keep_mask=mask, new_mask=np.zeros_like(mask))
        with pytest.raises(InvalidActionError):
            SplitNode(tracks, plan)


class TestMerge:
    def test_choose_kept_node_prefers_a_node_with_a_parent(self, tracks_viewer_setup):
        _, _, tracks = tracks_viewer_setup
        # 5 (t4) has a parent, 6 (t4) has none
        assert choose_kept_node(tracks, [6, 5]) == 5

    def test_merge_sisters_and_undo_in_one_step(self, tracks_viewer_setup, click_node):
        viewer, tracks_viewer, tracks = tracks_viewer_setup
        before = _snapshot(tracks)
        union = node_mask(tracks, 2) | node_mask(tracks, 3)

        click_node(tracks_viewer, 2)
        click_node(tracks_viewer, 3, append=True)
        tracks_viewer.merge_nodes()

        nodes = {int(n) for n in tracks.graph.node_ids()}
        kept = 2 if 2 in nodes else 3
        removed = 3 if kept == 2 else 2
        assert removed not in nodes
        assert kept in nodes
        frame = np.asarray(tracks.segmentation[1])
        assert np.array_equal(frame == kept, union)
        # the kept cell continues from the parent and takes over the child of 3
        assert [int(p) for p in tracks.predecessors(kept)] == [1]
        assert [int(c) for c in tracks.successors(kept)] == [4]
        assert [int(c) for c in tracks.successors(1)] == [kept]
        # no division any more: the kept cell continues the parent's track
        assert tracks.get_track_id(kept) == tracks.get_track_id(1)
        assert tracks.get_track_id(4) == tracks.get_track_id(kept)
        assert tracks_viewer.selected_nodes.as_list == [kept]

        tracks_viewer.undo()
        assert _snapshot(tracks) == before

        tracks_viewer.redo()
        assert removed not in {int(n) for n in tracks.graph.node_ids()}
        assert [int(c) for c in tracks.successors(kept)] == [4]

    def test_merge_inherits_the_parent_when_the_kept_node_had_none(
        self, tracks_viewer_setup
    ):
        _, _, tracks = tracks_viewer_setup
        # 5 (t4, child of 4) and 6 (t4, alone): keep 6 explicitly
        action = MergeNodes(tracks, [5, 6], keep=6)

        assert action.kept_node == 6
        assert action.removed_nodes == [5]
        assert not tracks.graph.has_node(5)
        assert [int(p) for p in tracks.predecessors(6)] == [4]
        assert tracks.get_track_id(6) == tracks.get_track_id(4)
        frame = np.asarray(tracks.segmentation[4])
        assert np.count_nonzero(frame == 6) == np.count_nonzero(frame != 0)
        assert action.notes == []

        tracks.undo()
        assert tracks.graph.has_node(5)
        assert [int(p) for p in tracks.predecessors(5)] == [4]
        assert [int(p) for p in tracks.predecessors(6)] == []

    def test_merge_reports_a_dropped_parent(self, tracks_viewer_setup):
        _, _, tracks = tracks_viewer_setup
        from funtracks.user_actions import UserAddEdge

        # give node 6 (t4) its own parent 2 (t1) so both 5 and 6 have parents
        UserAddEdge(tracks, (2, 6))
        action = MergeNodes(tracks, [5, 6], keep=5)

        assert [int(p) for p in tracks.predecessors(5)] == [4]
        assert [int(c) for c in tracks.successors(2)] == []
        assert any("end before this frame" in note for note in action.notes)

    def test_merge_requires_one_time_point(self, tracks_viewer_setup, click_node):
        _, tracks_viewer, tracks = tracks_viewer_setup
        before = _snapshot(tracks)
        click_node(tracks_viewer, 1)
        click_node(tracks_viewer, 2, append=True)

        with patch(
            "motile_tracker.data_views.views_coordinator.tracks_viewer.show_warning"
        ) as warning:
            tracks_viewer.merge_nodes()

        warning.assert_called_once()
        assert _snapshot(tracks) == before

    def test_merge_then_split_restores_two_cells(self, tracks_viewer_setup):
        """The round trip behind fixing a 'rejoined' cell: merging sisters and
        splitting them again gives two cells with the original links."""
        _, _, tracks = tracks_viewer_setup
        mask_2, mask_3 = node_mask(tracks, 2), node_mask(tracks, 3)
        MergeNodes(tracks, [2, 3], keep=3)
        assert not tracks.graph.has_node(2)

        plan = propose_split(tracks, 3)
        assert plan is not None
        action = SplitNode(tracks, plan)

        new = action.new_node
        frame = np.asarray(tracks.segmentation[1])
        assert np.array_equal((frame == 3) | (frame == new), mask_2 | mask_3)
        # parent 1 divides again, into 3 and the new cell; 4 still has one parent
        assert {int(c) for c in tracks.successors(1)} == {3, new}
        assert len(tracks.predecessors(4)) == 1
