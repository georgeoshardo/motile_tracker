"""Splitting one node into two and merging nodes into one, as single undo steps."""

from unittest.mock import patch

import numpy as np
import pytest

from mm_track.data_views.keybindings_config import KEYMAP
from mm_track.data_views.views_coordinator.node_edits import (
    MergeNodes,
    SplitNode,
    choose_kept_node,
    find_second_parent,
    next_node_id,
    node_mask,
    plan_split,
    propose_split,
)
from mm_track.data_views.views_coordinator.tracks_viewer import TracksViewer


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
            "mm_track.data_views.views_coordinator.tracks_viewer.show_warning"
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

        from mm_track.data_views.views_coordinator.node_edits import SplitPlan

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
            "mm_track.data_views.views_coordinator.tracks_viewer.show_warning"
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


def test_new_cells_survive_saving_from_the_tracks_list(
    viewer, solution_tracks_2d, tmp_path
):
    """Cells created after loading (by a split or by painting) must be in the file
    the Tracks List saves: the row has to hold the object the viewer edits."""
    from funtracks.data_model import Tracks
    from funtracks.import_export import import_from_geff

    tracks_viewer = TracksViewer.get_instance(viewer)
    # a plain Tracks, as loaded from a GEFF
    plain = Tracks(
        solution_tracks_2d.graph_full,
        ndim=3,
        scale=solution_tracks_2d.scale,
        features=solution_tracks_2d.features,
    )
    tracks_list = tracks_viewer.tracks_list
    tracks_list.add_tracks(plain, "plain", select=True)
    row = tracks_list.tracks_list.itemWidget(tracks_list.tracks_list.item(0))
    assert row.tracks is tracks_viewer.tracks

    tracks_viewer.selected_nodes.add(4)
    tracks_viewer.split_node()
    new = max(int(n) for n in tracks_viewer.tracks.graph.node_ids())
    assert new > 6

    tracks_list.save_dir_line.setText(str(tmp_path))
    tracks_list.save_tracks(tracks_list.tracks_list.item(0))
    reloaded = import_from_geff(tmp_path / "plain.geff")

    assert reloaded.graph.has_node(new)
    assert reloaded.graph.num_nodes() == tracks_viewer.tracks.graph.num_nodes()
    assert np.array_equal(
        np.asarray(reloaded.segmentation[2]),
        np.asarray(tracks_viewer.tracks.segmentation[2]),
    )


class TestNewLabelAfterDeletion:
    """A new label must never be the id of a deleted node: deletion is soft, and
    painting with a deleted node's id is refused (napari-track-edit #478)."""

    def test_spatial_new_label_skips_an_undone_split(self, tracks_viewer_setup):
        from mm_track.data_views.views.layers.track_labels import new_label

        _, tracks_viewer, tracks = tracks_viewer_setup
        action = SplitNode(tracks, propose_split(tracks, 4))
        tracks_viewer.undo()
        assert not tracks.graph.has_node(action.new_node)

        seg_layer = tracks_viewer.tracking_layers.seg_layer
        new_label(seg_layer)

        assert seg_layer.selected_label == action.new_node + 1

    def test_kymograph_paint_after_deleting_the_newest_node(self, tracks_viewer_setup):
        _, tracks_viewer, tracks = tracks_viewer_setup
        assert tracks_viewer.set_view_mode("kymograph")
        tracks_viewer.selected_nodes.add(6)  # the highest id
        tracks_viewer.delete_node()
        assert not tracks.graph.has_node(6)

        tracks_viewer.request_new_track()
        labels = tracks_viewer.kymograph_layers.labels_layer
        new = int(labels.selected_label)
        assert new == 7

        class _Event:
            def __init__(self, value):
                self.value = value

        # paint a small square into empty space of frame 3 (kymograph x 300-399)
        yy, xx = np.meshgrid(np.arange(40, 44), np.arange(340, 344), indexing="ij")
        old = labels.data[yy.ravel(), xx.ravel()]
        assert not old.any()
        labels.mode = "paint"
        labels._on_paint(_Event([((yy.ravel(), xx.ravel()), old, new)]))

        assert tracks.graph.has_node(new)
        assert tracks.get_time(new) == 3
        tracks_viewer.set_view_mode("spatial")


class TestMergeOverTime:
    """A cell cut in two over several frames is merged frame by frame in one step."""

    def _over_segment(self, tracks):
        """Split node 3 (t1) and node 4 (t2) of the track 3 -> 4 -> 5."""
        masks = {n: node_mask(tracks, n) for n in (3, 4)}
        pairs = []
        for node in (3, 4):
            action = SplitNode(tracks, propose_split(tracks, node))
            pairs.append((node, action.new_node))
        return masks, pairs

    def test_can_merge_needs_two_nodes_in_some_frame(self, tracks_viewer_setup):
        from mm_track.data_views.views_coordinator.node_edits import can_merge

        _, _, tracks = tracks_viewer_setup
        assert can_merge(tracks, [2, 3])  # both t1
        assert can_merge(tracks, [2, 3, 4])  # t1 pair plus a lone t2 node
        assert not can_merge(tracks, [1, 2, 4])  # one node per frame
        assert not can_merge(tracks, [2])
        assert not can_merge(None, [2, 3])

    def test_merge_pairs_over_frames_in_one_step(self, tracks_viewer_setup):
        _, tracks_viewer, tracks = tracks_viewer_setup
        masks, pairs = self._over_segment(tracks)
        split_state = _snapshot(tracks)
        history = len(tracks.action_history.undo_stack)

        tracks_viewer.selected_nodes.add_list([n for pair in pairs for n in pair])
        tracks_viewer.merge_nodes()

        assert len(tracks.action_history.undo_stack) == history + 1
        frame1 = np.asarray(tracks.segmentation[1])
        frame2 = np.asarray(tracks.segmentation[2])
        kept = [n for n in tracks.graph.node_ids() if tracks.get_time(n) in (1, 2)]
        kept = sorted(int(n) for n in kept if n != 2)  # node 2 is the sister of 3
        assert len(kept) == 2
        # one cell per frame again, with the original masks
        assert np.array_equal(frame1 == kept[0], masks[3])
        assert np.array_equal(frame2 == kept[1], masks[4])
        # and one track through the stretch: kept(t1) -> kept(t2) -> 5
        assert [int(c) for c in tracks.successors(kept[0])] == [kept[1]]
        assert [int(c) for c in tracks.successors(kept[1])] == [5]
        assert tracks.get_track_id(kept[0]) == tracks.get_track_id(5)
        # both merged cells are selected afterwards
        assert set(tracks_viewer.selected_nodes.as_list) == set(kept)

        tracks_viewer.undo()
        assert _snapshot(tracks) == split_state

    def test_lone_nodes_are_left_alone_and_reported(self, tracks_viewer_setup):
        from mm_track.data_views.views_coordinator.node_edits import (
            MergeNodesOverTime,
        )

        _, _, tracks = tracks_viewer_setup
        action = MergeNodesOverTime(tracks, [2, 3, 1, 6])
        assert action.skipped_nodes == [1, 6]  # alone at t0 and t4
        assert len(action.kept_nodes) == 1
        assert tracks.graph.has_node(1) and tracks.graph.has_node(6)

    def test_nothing_to_merge_is_refused(self, tracks_viewer_setup):
        from funtracks.exceptions import InvalidActionError

        from mm_track.data_views.views_coordinator.node_edits import (
            MergeNodesOverTime,
        )

        _, tracks_viewer, tracks = tracks_viewer_setup
        with pytest.raises(InvalidActionError):
            MergeNodesOverTime(tracks, [1, 2, 4])
        before = _snapshot(tracks)
        tracks_viewer.selected_nodes.add_list([1, 2, 4])
        with patch(
            "mm_track.data_views.views_coordinator.tracks_viewer.show_warning"
        ) as warning:
            tracks_viewer.merge_nodes()
        warning.assert_called_once()
        assert _snapshot(tracks) == before


class TestEditsKeepTheView:
    """Split, merge, undo and redo select the edited cells without moving the
    camera, in the spatial and the kymograph view."""

    @pytest.mark.parametrize("mode", ["spatial", "kymograph"])
    def test_split_merge_undo_redo_do_not_move_the_camera(
        self, tracks_viewer_setup, mode
    ):
        viewer, tracks_viewer, tracks = tracks_viewer_setup
        assert tracks_viewer.set_view_mode(mode)
        if mode == "kymograph":
            tracks_viewer.kymograph_layers.show_opening_view(deferred=False)
        center = list(viewer.camera.center)
        center[-2] += 13.0
        center[-1] += 7.0
        viewer.camera.center = tuple(center)
        zoom = viewer.camera.zoom

        def assert_view_unchanged():
            assert viewer.camera.center == pytest.approx(tuple(center))
            assert viewer.camera.zoom == pytest.approx(zoom)

        tracks_viewer.selected_nodes.add(4)
        viewer.camera.center = tuple(center)  # selecting alone may centre; reset

        tracks_viewer.split_node()
        assert len(tracks_viewer.selected_nodes) == 2
        assert_view_unchanged()

        tracks_viewer.undo()
        assert_view_unchanged()
        tracks_viewer.redo()
        assert_view_unchanged()

        tracks_viewer.selected_nodes.add_list([2, 3])
        viewer.camera.center = tuple(center)
        tracks_viewer.merge_nodes()
        assert len(tracks_viewer.selected_nodes) == 1  # the kept cell
        assert_view_unchanged()
        tracks_viewer.undo()
        assert_view_unchanged()
        tracks_viewer.set_view_mode("spatial")


class TestCut:
    """Cutting a node into the pieces an alternative segmentation offers."""

    def test_merge_then_cut_restores_the_division(self, tracks_viewer_setup):
        _, _, tracks = tracks_viewer_setup
        from mm_track.data_views.views_coordinator.node_edits import CutNode, plan_cut

        mask_2, mask_3 = node_mask(tracks, 2), node_mask(tracks, 3)
        MergeNodes(tracks, [2, 3], keep=3)
        assert [int(c) for c in tracks.successors(1)] == [3]

        plan = plan_cut(tracks, 3, [mask_2, mask_3])
        # piece 2 lies on parent 1: it keeps the label; the other is the second daughter
        assert plan.keep_index == 1
        assert plan.parents == [1, 1]
        assert plan.children == [[], [4]]
        action = CutNode(tracks, plan)

        new = action.new_nodes[0]
        frame = np.asarray(tracks.segmentation[1])
        assert np.array_equal(frame == new, mask_2)
        assert np.array_equal(frame == 3, mask_3)
        assert {int(c) for c in tracks.successors(1)} == {3, new}
        assert [int(p) for p in tracks.predecessors(4)] == [3]

    def test_a_piece_continues_a_neighbour_whose_track_ended(self, tracks_viewer_setup):
        _, _, tracks = tracks_viewer_setup
        from funtracks.user_actions import UserUpdateSegmentation

        from mm_track.data_views.views_coordinator.node_edits import (
            CutNode,
            mask_pixels,
            plan_cut,
        )

        # grow node 4 (t2, child of 3) over where nodes 2 and 3 were at t1: node 2's
        # track ends there, so the upper piece continues it
        big = np.zeros((100, 100), dtype=bool)
        big[0:76, 0:91] = True
        UserUpdateSegmentation(
            tracks, new_value=4, updated_pixels=[(mask_pixels(big & ~node_mask(tracks, 4), 2), 0)],
            current_track_id=int(tracks.get_track_id(4)),
        )
        before = _snapshot(tracks)
        mask = node_mask(tracks, 4)
        upper, lower = mask.copy(), mask.copy()
        upper[38:] = False
        lower[:38] = False

        plan = plan_cut(tracks, 4, [upper, lower])
        assert plan.keep_index == 1  # the lower piece lies on parent 3
        assert plan.parents == [2, 3]
        action = CutNode(tracks, plan)

        new = action.new_nodes[0]
        assert [int(p) for p in tracks.predecessors(new)] == [2]
        assert tracks.get_track_id(new) == tracks.get_track_id(2)
        assert [int(p) for p in tracks.predecessors(4)] == [3]
        tracks.undo()
        assert _snapshot(tracks) == before

    def test_three_pieces_and_a_note_for_the_one_without_a_cell(self, tracks_viewer_setup):
        _, _, tracks = tracks_viewer_setup
        from mm_track.data_views.views_coordinator.node_edits import CutNode, plan_cut

        # node 3's parent 1 already has two children: nothing is left for the others
        mask = node_mask(tracks, 3)
        rows = np.nonzero(mask)[0]
        pieces = [mask & (np.arange(100)[:, None] < rows.min() + 10),
                  mask & (np.arange(100)[:, None] >= rows.min() + 10) & (np.arange(100)[:, None] < rows.min() + 20),
                  mask & (np.arange(100)[:, None] >= rows.min() + 20)]
        plan = plan_cut(tracks, 3, pieces)

        assert plan.parents[plan.keep_index] == 1
        assert sum(p is None for p in plan.parents) == 2
        assert len(plan.notes) == 2
        action = CutNode(tracks, plan)
        assert len(action.new_nodes) == 2
        frame = np.asarray(tracks.segmentation[1])
        assert np.array_equal(np.isin(frame, [3, *action.new_nodes]), mask)

    def test_a_piece_without_a_child_is_continued_by_a_track_starting_next(
        self, tracks_viewer_setup
    ):
        _, _, tracks = tracks_viewer_setup
        from funtracks.user_actions import UserDeleteEdge, UserUpdateSegmentation

        from mm_track.data_views.views_coordinator.node_edits import (
            CutNode,
            mask_pixels,
            plan_cut,
        )

        # node 4 (t2) starts a track of its own under the lower half of node 3 (t1)
        UserDeleteEdge(tracks, (3, 4))
        upper, lower = _halves(node_mask(tracks, 3))
        UserUpdateSegmentation(
            tracks, new_value=4, updated_pixels=[(mask_pixels(lower & ~node_mask(tracks, 4), 2), 0)],
            current_track_id=int(tracks.get_track_id(4)),
        )
        assert len(tracks.predecessors(4)) == 0

        plan = plan_cut(tracks, 3, [upper, lower])
        assert plan.adopted[1] == [4] and plan.adopted[0] == []
        action = CutNode(tracks, plan)

        lower_label = 3 if plan.keep_index == 1 else action.new_nodes[0]
        assert [int(p) for p in tracks.predecessors(4)] == [lower_label]
        tracks.undo()
        assert len(tracks.predecessors(4)) == 0

    def test_pieces_must_be_exactly_the_mask(self, tracks_viewer_setup):
        _, _, tracks = tracks_viewer_setup
        from funtracks.exceptions import InvalidActionError

        from mm_track.data_views.views_coordinator.node_edits import CutNode, plan_cut

        upper, lower = _halves(node_mask(tracks, 3))
        lower[np.nonzero(lower)[0].max()] = False  # a row of the mask is left out
        plan = plan_cut(tracks, 3, [upper, lower])
        with pytest.raises(InvalidActionError):
            CutNode(tracks, plan)


class TestCutFromHypotheses:
    """Shift+C: the cut an alternative segmentation offers, previewed, then applied."""

    @staticmethod
    def _hypotheses(tracks):
        upper, lower = _halves(node_mask(tracks, 3))
        labels = np.zeros((100, 100), dtype=np.int32)
        labels[upper], labels[lower] = 1, 2
        return lambda time: {"h1": labels, "h2": labels} if time == 1 else {}

    def test_keybindings(self):
        assert KEYMAP["cut_from_hypotheses"] == ["Shift-C"]
        assert KEYMAP["apply_cut_preview"] == ["Enter"]

    def test_preview_then_apply_in_one_undo_step(self, tracks_viewer_setup, click_node):
        viewer, tracks_viewer, tracks = tracks_viewer_setup
        tracks_viewer.set_cut_hypotheses(tracks, self._hypotheses(tracks))
        before = _snapshot(tracks)
        mask = node_mask(tracks, 3)
        click_node(tracks_viewer, 3)
        active = viewer.layers.selection.active

        tracks_viewer.cut_from_hypotheses()

        preview = tracks_viewer.cut_preview
        assert preview.active and preview.node == 3
        assert preview.current.hypotheses == ("h1", "h2")
        assert preview.layer in viewer.layers
        assert viewer.layers.selection.active is active  # the user's layer stays active
        assert _snapshot(tracks) == before  # nothing changed yet
        assert "Enter: apply" in preview.message()

        tracks_viewer.cut_from_hypotheses()  # one candidate: it stays shown
        assert preview.index == 0

        tracks_viewer.apply_cut_preview()
        assert not preview.active and preview.layer is None
        nodes = sorted(int(n) for n in tracks.graph.node_ids())
        new = [n for n in nodes if n not in before[0]]
        assert len(new) == 1
        frame = np.asarray(tracks.segmentation[1])
        assert np.array_equal((frame == 3) | (frame == new[0]), mask)
        assert set(tracks_viewer.selected_nodes.as_list) == {3, new[0]}

        tracks_viewer.undo()
        assert _snapshot(tracks) == before

    def test_escape_another_node_or_an_edit_drops_the_preview(
        self, tracks_viewer_setup, click_node
    ):
        viewer, tracks_viewer, tracks = tracks_viewer_setup
        tracks_viewer.set_cut_hypotheses(tracks, self._hypotheses(tracks))
        preview = tracks_viewer.cut_preview

        click_node(tracks_viewer, 3)
        tracks_viewer.cut_from_hypotheses()
        tracks_viewer.deselect()  # Esc: the preview goes, the selection stays
        assert not preview.active and preview.layer is None
        assert tracks_viewer.selected_nodes.as_list == [3]

        tracks_viewer.cut_from_hypotheses()
        click_node(tracks_viewer, 2)
        assert not preview.active

        click_node(tracks_viewer, 3)
        tracks_viewer.cut_from_hypotheses()
        tracks_viewer.delete_node()  # an edit
        assert not preview.active
        assert all(layer.name != "cut preview" for layer in viewer.layers)

    def test_no_hypotheses_or_hypotheses_of_other_tracks(
        self, tracks_viewer_setup, click_node
    ):
        _, tracks_viewer, tracks = tracks_viewer_setup
        click_node(tracks_viewer, 3)
        warn = "mm_track.data_views.views_coordinator.tracks_viewer.show_warning"
        with patch(warn) as warning:
            tracks_viewer.cut_from_hypotheses()
        warning.assert_called_once()

        tracks_viewer.set_cut_hypotheses(object(), self._hypotheses(tracks))
        with patch(warn) as warning:
            tracks_viewer.cut_from_hypotheses()
        warning.assert_called_once()
        assert not tracks_viewer.cut_preview.active

    def test_a_mask_no_hypothesis_cuts(self, tracks_viewer_setup, click_node):
        _, tracks_viewer, tracks = tracks_viewer_setup
        whole = np.zeros((100, 100), dtype=np.int32)
        whole[node_mask(tracks, 3)] = 1
        tracks_viewer.set_cut_hypotheses(tracks, lambda time: {"same": whole})
        click_node(tracks_viewer, 3)
        with patch("mm_track.data_views.views_coordinator.tracks_viewer.show_info") as info:
            tracks_viewer.cut_from_hypotheses()
        assert "c cuts a mask from the image" in info.call_args[0][0]
        assert not tracks_viewer.cut_preview.active


class TestCutManyFromHypotheses:
    """Shift+C on several selected nodes: in one frame, or a stretch of frames."""

    @staticmethod
    def _grow_node_4(tracks):
        """Node 4 (t2, child of 3) grown over where nodes 2 and 3 are at t1."""
        from funtracks.user_actions import UserUpdateSegmentation

        from mm_track.data_views.views_coordinator.node_edits import mask_pixels

        big = np.zeros((100, 100), dtype=bool)
        big[0:76, 0:91] = True
        UserUpdateSegmentation(
            tracks, new_value=4,
            updated_pixels=[(mask_pixels(big & ~node_mask(tracks, 4), 2), 0)],
            current_track_id=int(tracks.get_track_id(4)),
        )

    @staticmethod
    def _halves_labels(*masks):
        labels = np.zeros((100, 100), dtype=np.int32)
        for k, mask in enumerate(masks):
            upper, lower = _halves(mask)
            labels[upper], labels[lower] = 2 * k + 1, 2 * k + 2
        return labels

    def test_nodes_of_one_frame_are_cut_together_in_one_undo_step(
        self, tracks_viewer_setup, click_node
    ):
        _, tracks_viewer, tracks = tracks_viewer_setup
        labels = self._halves_labels(node_mask(tracks, 2), node_mask(tracks, 3))
        tracks_viewer.set_cut_hypotheses(tracks, lambda t: {"h": labels} if t == 1 else {})
        before = _snapshot(tracks)
        click_node(tracks_viewer, 2)
        click_node(tracks_viewer, 3, append=True)

        tracks_viewer.cut_from_hypotheses()
        preview = tracks_viewer.cut_preview
        assert [p.node for p in preview.proposals] == [2, 3]
        assert "Cuts for 2 of 2 selected nodes (t1)" in preview.message()
        tracks_viewer.apply_cut_preview()

        nodes = sorted(int(n) for n in tracks.graph.node_ids())
        assert len(nodes) == len(before[0]) + 2
        assert len(tracks_viewer.selected_nodes) == 4
        tracks_viewer.undo()
        assert _snapshot(tracks) == before

    def test_a_stretch_of_frames_links_up(self, tracks_viewer_setup, click_node):
        _, tracks_viewer, tracks = tracks_viewer_setup
        self._grow_node_4(tracks)
        mask_3, mask_4 = node_mask(tracks, 3), node_mask(tracks, 4)
        # t1: node 3 in halves; t2: node 4 cut where node 2 ends and node 3 begins
        upper_4 = mask_4 & (np.arange(100)[:, None] < 38)
        labels_2 = np.zeros((100, 100), dtype=np.int32)
        labels_2[upper_4], labels_2[mask_4 & ~upper_4] = 1, 2
        labels = {1: self._halves_labels(mask_3), 2: labels_2}
        tracks_viewer.set_cut_hypotheses(tracks, lambda t: {"h": labels[t]} if t in labels else {})
        before = _snapshot(tracks)
        click_node(tracks_viewer, 4)  # selection order does not matter
        click_node(tracks_viewer, 3, append=True)

        tracks_viewer.cut_from_hypotheses()
        assert [p.time for p in tracks_viewer.cut_preview.proposals] == [1, 2]
        tracks_viewer.apply_cut_preview()

        frame = np.asarray(tracks.segmentation[2])
        pieces_t2 = [int(v) for v in np.unique(frame[mask_4]) if v]
        assert len(pieces_t2) == 2
        # every piece at t2 continues a cell at t1: node 2 (whose track ended) or a
        # piece of node 3
        parents = {int(p) for n in pieces_t2 for p in tracks.predecessors(n)}
        assert 2 in parents and len(parents) == 2
        assert all(int(tracks.get_time(p)) == 1 for p in parents)
        tracks_viewer.undo()
        assert _snapshot(tracks) == before

    def test_nodes_no_hypothesis_cuts_are_skipped_and_named(
        self, tracks_viewer_setup, click_node
    ):
        _, tracks_viewer, tracks = tracks_viewer_setup
        labels = self._halves_labels(node_mask(tracks, 3))
        tracks_viewer.set_cut_hypotheses(tracks, lambda t: {"h": labels} if t == 1 else {"h": labels * 0})
        click_node(tracks_viewer, 3)
        click_node(tracks_viewer, 6, append=True)

        tracks_viewer.cut_from_hypotheses()
        preview = tracks_viewer.cut_preview
        assert [p.node for p in preview.proposals] == [3]
        assert preview.skipped == [6]
        assert "none for 6" in preview.message()
        # selecting only some of the nodes drops the preview
        click_node(tracks_viewer, 3)
        assert not preview.active

    def test_shift_c_again_steps_the_nodes_with_several_cuts(
        self, tracks_viewer_setup, click_node
    ):
        _, tracks_viewer, tracks = tracks_viewer_setup
        mask_2, mask_3 = node_mask(tracks, 2), node_mask(tracks, 3)
        halves = self._halves_labels(mask_2, mask_3)
        thirds = np.zeros((100, 100), dtype=np.int32)  # another cut of node 3 only
        rows = np.arange(100)[:, None]
        top = np.nonzero(mask_3)[0].min()
        thirds[mask_3 & (rows < top + 10)], thirds[mask_3 & (rows >= top + 10)] = 1, 2
        tracks_viewer.set_cut_hypotheses(
            tracks, lambda t: {"a": halves, "b": thirds} if t == 1 else {}
        )
        click_node(tracks_viewer, 2)
        click_node(tracks_viewer, 3, append=True)

        tracks_viewer.cut_from_hypotheses()
        preview = tracks_viewer.cut_preview
        assert [len(p.cuts) for p in preview.proposals] == [1, 2]
        assert "(1 have others)" in preview.message()
        tracks_viewer.cut_from_hypotheses()
        assert [p.index for p in preview.proposals] == [0, 1]
        tracks_viewer.cut_from_hypotheses()
        assert [p.index for p in preview.proposals] == [0, 0]


def test_a_stretch_is_not_ranked_against_its_own_merged_masks(viewer, solution_tracks_2d):
    """In a selected stretch, the frame after still holds the wrong merged mask: it
    must not make the cut look worse than leaving the mask whole."""
    from funtracks.user_actions import UserUpdateSegmentation

    from mm_track.data_views.views_coordinator.node_edits import mask_pixels

    tracks_viewer = TracksViewer.get_instance(viewer)
    tracks_viewer.update_tracks(tracks=solution_tracks_2d, name="test")
    tracks = tracks_viewer.tracks
    # node 4 (t2) becomes the same mask as node 3 (t1): one merged cell over two frames
    same = node_mask(tracks, 3)
    UserUpdateSegmentation(
        tracks, new_value=4, updated_pixels=[(mask_pixels(same & ~node_mask(tracks, 4), 2), 0)],
        current_track_id=int(tracks.get_track_id(4)),
    )
    UserUpdateSegmentation(
        tracks, new_value=0, updated_pixels=[(mask_pixels(node_mask(tracks, 4) & ~same, 2), 4)],
        current_track_id=int(tracks.get_track_id(4)),
    )
    upper, lower = _halves(same)
    labels = np.zeros((100, 100), dtype=np.int32)
    labels[upper], labels[lower] = 1, 2
    tracks_viewer.set_cut_hypotheses(tracks, lambda t: {"h": labels})

    tracks_viewer.selected_nodes.add(3)
    tracks_viewer.cut_from_hypotheses()
    alone = tracks_viewer.cut_preview.proposals[0]
    # alone, the identical merged mask at t2 makes the whole mask fit perfectly
    assert alone.uncut > 0.99
    tracks_viewer.cut_preview.clear()

    tracks_viewer.selected_nodes.add_list([3, 4])
    tracks_viewer.cut_from_hypotheses()
    first = tracks_viewer.cut_preview.proposals[0]
    assert first.node == 3
    # selected with it, node 4 is not a neighbour to fit: only the frame before counts
    assert first.uncut < 0.5
    assert first.current.frame == "before"
    viewer.layers.clear()
