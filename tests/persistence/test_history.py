"""Autosave tests use real graph actions and real on-disk databases."""

import sqlite3

import numpy as np
import pytest
from funtracks.user_actions import UserDeleteNode, UserUpdateNodeAttrs

from motile_tracker.persistence.session import EditSession


def snapshot(tracks):
    nodes = sorted(tracks.graph.node_ids())
    return (
        nodes,
        sorted(tracks.graph.edge_list()),
        {n: tracks.get_track_id(n) for n in nodes},
        np.stack([np.asarray(tracks.segmentation[t]) for t in range(5)]).tobytes(),
    )


def test_delete_reopen_undo_redo(solution_tracks_2d, tmp_path):
    path = tmp_path / "tracks.geff"
    before = snapshot(solution_tracks_2d)
    with EditSession.create(solution_tracks_2d, path) as session:
        UserDeleteNode(session.tracks, 4)
        assert session.revision == 1
        after = snapshot(session.tracks)
    with EditSession.open(path) as restored:
        assert snapshot(restored.tracks) == after
        assert restored.tracks.graph_full.has_node(4)
        assert restored.tracks.undo()
        assert snapshot(restored.tracks) == before
    with EditSession.open(path) as restored:
        assert snapshot(restored.tracks) == before
        assert restored.tracks.redo()
        assert snapshot(restored.tracks) == after
        assert restored.revision == 3


def test_split_reopen_preserves_new_cell_and_masks(solution_tracks_2d, tmp_path):
    from motile_tracker.data_views.views_coordinator.node_edits import (
        SplitNode,
        node_mask,
        plan_split,
    )

    path = tmp_path / "tracks.geff"
    before = snapshot(solution_tracks_2d)
    with EditSession.create(solution_tracks_2d, path) as session:
        mask = node_mask(session.tracks, 4)
        upper = mask.copy()
        upper[2:] = False
        SplitNode(session.tracks, plan_split(session.tracks, 4, (upper, mask & ~upper)))
        after = snapshot(session.tracks)
        assert session.tracks.graph.has_node(7)
        assert session.revision == 1
    with EditSession.open(path) as restored:
        assert snapshot(restored.tracks) == after
        restored.tracks.undo()
        assert snapshot(restored.tracks) == before
        restored.tracks.redo()
        assert snapshot(restored.tracks) == after


def test_edit_after_undo_keeps_all_operations(solution_tracks_2d, tmp_path):
    path = tmp_path / "tracks.geff"
    with EditSession.create(solution_tracks_2d, path) as session:
        UserDeleteNode(session.tracks, 6)
        session.tracks.undo()
        UserDeleteNode(session.tracks, 5)
    with EditSession.open(path) as session:
        assert session.tracks.graph.has_node(6)
        assert not session.tracks.graph.has_node(5)
        session.tracks.undo()
        assert session.tracks.graph.has_node(5)
        session.tracks.undo()
        assert not session.tracks.graph.has_node(6)
        session.tracks.undo()
        assert session.tracks.graph.has_node(6)


def test_failed_transaction_restores_memory_and_disk(
    solution_tracks_2d, tmp_path, monkeypatch
):
    path = tmp_path / "tracks.geff"
    before = snapshot(solution_tracks_2d)
    with EditSession.create(solution_tracks_2d, path) as session:

        def fail(*args, **kwargs):
            raise sqlite3.OperationalError("disk is full")

        monkeypatch.setattr(session.store, "commit", fail)
        with pytest.raises(Exception, match="disk is full"):
            UserDeleteNode(session.tracks, 4)
        assert snapshot(session.tracks) == before
        assert session.revision == 0
        assert session.error
    with EditSession.open(path) as session:
        assert snapshot(session.tracks) == before


def test_second_writer_cannot_open(solution_tracks_2d, tmp_path):
    path = tmp_path / "tracks.geff"
    with (
        EditSession.create(solution_tracks_2d, path),
        pytest.raises(RuntimeError, match="already open"),
    ):
        EditSession.open(path)


def test_history_survives_missing_graph(solution_tracks_2d, tmp_path):
    import shutil

    path = tmp_path / "tracks.geff"
    with EditSession.create(solution_tracks_2d, path) as session:
        UserDeleteNode(session.tracks, 4)
        expected = snapshot(session.tracks)
    shutil.rmtree(path / "nodes")
    with EditSession.open(path) as session:
        assert snapshot(session.tracks) == expected
        session.tracks.undo()
        assert session.tracks.graph.has_node(4)
    assert (path / "nodes").is_dir()


def test_save_copy_retains_history(solution_tracks_2d, tmp_path):
    path = tmp_path / "tracks.geff"
    copy = tmp_path / "copy.geff"
    with EditSession.create(solution_tracks_2d, path) as session:
        UserDeleteNode(session.tracks, 6)
        session.save_copy(copy)
        assert session.path == path
    with EditSession.open(copy) as session:
        assert not session.tracks.graph.has_node(6)
        session.tracks.undo()
        assert session.tracks.graph.has_node(6)


def test_custom_attributes_survive(solution_tracks_2d, tmp_path):
    tracks = solution_tracks_2d
    tracks.add_feature(
        "reviewed",
        {
            "feature_type": "node",
            "value_type": "bool",
            "num_values": 1,
            "display_name": "Reviewed",
            "default_value": False,
        },
    )
    with EditSession.create(tracks, tmp_path / "tracks.geff"):
        UserUpdateNodeAttrs(tracks, 4, {"reviewed": True})
    with EditSession.open(tmp_path / "tracks.geff") as session:
        assert session.tracks.get_node_attr(4, "reviewed")
        session.tracks.undo()
        assert not session.tracks.get_node_attr(4, "reviewed")


def test_group_delete_and_undo_across_restart(solution_tracks_2d, tmp_path):
    from motile_tracker.persistence.features import change_feature

    path = tmp_path / "tracks.geff"
    with EditSession.create(solution_tracks_2d, path) as session:
        change_feature(
            session.tracks,
            "checked",
            {
                "feature_type": "node",
                "value_type": "bool",
                "num_values": 1,
                "display_name": "Checked",
                "default_value": False,
            },
        )
        UserUpdateNodeAttrs(session.tracks, 4, {"checked": True})
        change_feature(session.tracks, "checked", None)
    with EditSession.open(path) as session:
        assert "checked" not in session.tracks.features
        session.tracks.undo()
        assert session.tracks.get_node_attr(4, "checked")
        session.tracks.undo()
        assert not session.tracks.get_node_attr(4, "checked")
        session.tracks.undo()
        assert "checked" not in session.tracks.features


def test_process_exit_without_save_or_close(solution_tracks_2d, tmp_path):
    import os
    import subprocess
    import sys

    path = tmp_path / "tracks.geff"
    with EditSession.create(solution_tracks_2d, path):
        pass
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import os, sys
from motile_tracker.persistence.session import EditSession
from funtracks.user_actions import UserDeleteNode
session = EditSession.open(sys.argv[1])
UserDeleteNode(session.tracks, 4)
os._exit(0)
""",
            str(path),
        ],
        check=True,
        env=os.environ.copy(),
    )
    with EditSession.open(path) as session:
        assert session.revision == 1
        assert not session.tracks.graph.has_node(4)
        session.tracks.undo()
        assert session.tracks.graph.has_node(4)


def test_transaction_abort_does_not_leave_partial_revision(
    solution_tracks_2d, tmp_path
):
    path = tmp_path / "tracks.geff"
    before = snapshot(solution_tracks_2d)
    with EditSession.create(solution_tracks_2d, path) as session:
        session.store.connection.execute("""CREATE TEMP TRIGGER reject_revision
            BEFORE INSERT ON revisions BEGIN SELECT RAISE(ABORT, 'injected write failure'); END""")
        with pytest.raises(RuntimeError, match="injected write failure"):
            UserDeleteNode(session.tracks, 4)
        assert snapshot(session.tracks) == before
    with EditSession.open(path) as session:
        assert snapshot(session.tracks) == before
        assert session.revision == 0


@pytest.mark.parametrize("zarr_format", [2, 3])
def test_projection_preserves_format_and_metadata(
    solution_tracks_2d, tmp_path, zarr_format
):
    import zarr
    from funtracks.import_export import write_to_geff

    path = tmp_path / "tracks.geff"
    write_to_geff(solution_tracks_2d, path, zarr_format=zarr_format)
    root = zarr.open_group(path, mode="a")
    root.attrs["experiment"] = "keep me"
    geff = dict(root.attrs["geff"])
    geff["related_objects"] = [{"path": "../images", "type": "image"}]
    root.attrs["geff"] = geff
    with EditSession.create(solution_tracks_2d, path) as session:
        UserDeleteNode(session.tracks, 6)
        session.flush()
    root = zarr.open_group(path, mode="r")
    assert root.metadata.zarr_format == zarr_format
    assert root.attrs["experiment"] == "keep me"
    assert root.attrs["geff"]["related_objects"] == geff["related_objects"]
    assert "edit_history" in root


def test_external_modification_is_not_overwritten(solution_tracks_2d, tmp_path):
    import zarr

    path = tmp_path / "tracks.geff"
    with EditSession.create(solution_tracks_2d, path):
        pass
    root = zarr.open_group(path, mode="a")
    root.attrs["changed_elsewhere"] = True
    with pytest.raises(ValueError, match="outside"):
        EditSession.open(path)
    assert zarr.open_group(path, mode="r").attrs["changed_elsewhere"]


@pytest.mark.parametrize("internal", [False, True])
def test_save_copy_rebases_image_references(solution_tracks_2d, tmp_path, internal):
    import zarr
    from funtracks.import_export import write_to_geff

    source = tmp_path / "source" / "tracks.geff"
    source.parent.mkdir()
    write_to_geff(solution_tracks_2d, source)
    image = (source if internal else source.parent) / "images"
    image.mkdir()
    root = zarr.open_group(source, mode="a")
    metadata = dict(root.attrs["geff"])
    metadata["related_objects"] = [
        {"path": "images" if internal else "../images", "type": "image"}
    ]
    root.attrs["geff"] = metadata
    destination = tmp_path / "elsewhere" / "copy.geff"
    with EditSession.create(solution_tracks_2d, source) as session:
        session.save_copy(destination)
    with EditSession.open(destination) as session:
        session.flush()
    relative = zarr.open_group(destination, mode="r").attrs["geff"]["related_objects"][
        0
    ]["path"]
    assert (destination / relative).resolve() == (
        destination / "images" if internal else image
    ).resolve()


def test_external_write_while_open_stops_projection(solution_tracks_2d, tmp_path):
    import zarr

    path = tmp_path / "tracks.geff"
    with EditSession.create(solution_tracks_2d, path) as session:
        session.flush()
        root = zarr.open_group(path, mode="a")
        root.attrs["outsider"] = "preserve"
        UserDeleteNode(session.tracks, 6)
        with pytest.raises(RuntimeError, match="outside"):
            session.flush()
    assert zarr.open_group(path, mode="r").attrs["outsider"] == "preserve"


def test_delete_every_cell_then_reopen_and_undo(solution_tracks_2d, tmp_path):
    path = tmp_path / "tracks.geff"
    before = snapshot(solution_tracks_2d)
    with EditSession.create(solution_tracks_2d, path) as session:
        for node in list(session.tracks.graph.node_ids()):
            UserDeleteNode(session.tracks, node)
        session.flush()
        assert session.tracks.graph.num_nodes() == 0
    with EditSession.open(path) as session:
        for _ in range(6):
            assert session.tracks.undo()
        assert snapshot(session.tracks) == before


def test_failed_initialization_does_not_poison_existing_geff(
    solution_tracks_2d, tmp_path
):
    from funtracks.import_export import write_to_geff

    from motile_tracker.persistence.session import has_history

    path = tmp_path / "tracks.geff"
    write_to_geff(solution_tracks_2d, path)
    # Unsupported metadata is rejected before an editing session can start.
    solution_tracks_2d.graph_full._update_metadata(unsupported=object())
    with pytest.raises(TypeError, match="Unsupported"):
        EditSession.create(solution_tracks_2d, path)
    assert not has_history(path)
    assert (path / "nodes").is_dir()


def test_failed_database_setup_removes_only_its_new_database(
    solution_tracks_2d, tmp_path
):
    import zarr
    from funtracks.import_export import write_to_geff

    from motile_tracker.persistence.session import has_history

    path = tmp_path / "tracks.geff"
    write_to_geff(solution_tracks_2d, path)
    root = zarr.open_group(path, mode="a")
    root.create_array("edit_history", data=np.array([17]))
    with pytest.raises(TypeError):
        EditSession.create(solution_tracks_2d, path)
    assert not has_history(path)
    np.testing.assert_array_equal(
        zarr.open_group(path, mode="r")["edit_history"][:], [17]
    )
    assert (path / "nodes").is_dir()


def test_measurement_toggle_is_saved_and_undoable(solution_tracks_2d, tmp_path):
    from motile_tracker.persistence.features import toggle_feature

    tracks = solution_tracks_2d
    if "area" in tracks.features:
        tracks.disable_features(["area"])
    path = tmp_path / "tracks.geff"
    with EditSession.create(tracks, path) as session:
        toggle_feature(tracks, "area", True)
        assert session.revision == 1
    with EditSession.open(path) as session:
        assert "area" in session.tracks.features
        assert session.tracks.get_node_attr(4, "area") == 16
        session.tracks.undo()
        assert "area" not in session.tracks.features
        session.tracks.redo()
        assert session.tracks.get_node_attr(4, "area") == 16


def test_earlier_revision_can_be_reconstructed(solution_tracks_2d, tmp_path):
    from motile_tracker.persistence.state import restore

    before = snapshot(solution_tracks_2d)
    with EditSession.create(solution_tracks_2d, tmp_path / "tracks.geff") as session:
        UserDeleteNode(session.tracks, 4)
        deleted = snapshot(session.tracks)
        session.tracks.undo()
        UserDeleteNode(session.tracks, 6)
        original = restore(session.store.state_at(0), session.store.get_blob)
        first_edit = restore(session.store.state_at(1), session.store.get_blob)
        assert snapshot(original) == before
        assert snapshot(first_edit) == deleted


def test_empty_projection_is_readable_by_standard_geff(solution_tracks_2d, tmp_path):
    from funtracks.import_export import import_from_geff

    path = tmp_path / "empty.geff"
    with EditSession.create(solution_tracks_2d, path) as session:
        for node in list(session.tracks.graph_solution.node_ids()):
            UserDeleteNode(session.tracks, node)
        session.flush()
    assert import_from_geff(path).graph_solution.num_nodes() == 0


def test_changed_timestamps_do_not_prevent_reopening(solution_tracks_2d, tmp_path):
    import os

    path = tmp_path / "tracks.geff"
    with EditSession.create(solution_tracks_2d, path):
        UserDeleteNode(solution_tracks_2d, 4)
    for file in (path / "nodes").rglob("*"):
        if file.is_file():
            os.utime(file, (1000000000, 1000000000))
    with EditSession.open(path) as session:
        assert not session.tracks.graph.has_node(4)


def test_failed_undo_restores_last_committed_state(
    solution_tracks_2d, tmp_path, monkeypatch
):
    path = tmp_path / "tracks.geff"
    with EditSession.create(solution_tracks_2d, path) as session:
        UserDeleteNode(session.tracks, 4)
        before = snapshot(session.tracks)
        action = session.tracks.action_history.undo_stack[-1]
        inverse = action.inverse

        def interrupted():
            inverse()
            raise RuntimeError("interrupted undo")

        monkeypatch.setattr(action, "inverse", interrupted)
        with pytest.raises(RuntimeError, match="interrupted undo"):
            session.tracks.undo()
        assert snapshot(session.tracks) == before
    with EditSession.open(path) as session:
        assert snapshot(session.tracks) == before
        assert session.tracks.undo()


def test_initially_empty_tracks_can_add_cells_and_reopen(tmp_path):
    from funtracks.data_model import SolutionTracks
    from funtracks.user_actions import UserAddNode
    from funtracks.utils.tracksdata_utils import create_empty_graph

    graph = create_empty_graph(node_attributes=["pos"], ndim=3)
    tracks = SolutionTracks(graph, ndim=3, time_attr="t")
    path = tmp_path / "manual.geff"
    with EditSession.create(tracks, path) as session:
        session.flush()
        UserAddNode(
            tracks, 1, {"t": 0, "pos": [1.0, 2.0], tracks.features.tracklet_key: 1}
        )
    with EditSession.open(path) as session:
        assert session.tracks.graph_solution.node_ids() == [1]
        session.tracks.undo()
        assert session.tracks.graph_solution.num_nodes() == 0


def test_three_dimensional_masks_survive_reopening(graph_3d, tmp_path):
    from funtracks.data_model import SolutionTracks

    tracks = SolutionTracks(graph_3d, ndim=4, time_attr="t")
    before = np.asarray(tracks.segmentation[1]).copy()
    path = tmp_path / "volume.geff"
    with EditSession.create(tracks, path):
        UserDeleteNode(tracks, 2)
    with EditSession.open(path) as session:
        assert not session.tracks.graph_solution.has_node(2)
        session.tracks.undo()
        np.testing.assert_array_equal(
            np.asarray(session.tracks.segmentation[1]), before
        )


def test_history_records_do_not_repeat_the_entire_stack(solution_tracks_2d, tmp_path):
    tracks = solution_tracks_2d
    tracks.add_feature(
        "review_number",
        {
            "feature_type": "node",
            "value_type": "int",
            "num_values": 1,
            "display_name": "Review number",
            "default_value": 0,
        },
    )
    path = tmp_path / "tracks.geff"
    with EditSession.create(tracks, path) as session:
        for value in range(1, 41):
            UserUpdateNodeAttrs(tracks, 4, {"review_number": value})
        lengths = [
            row[0]
            for row in session.store.connection.execute(
                "SELECT length(history) FROM revisions WHERE id>0 ORDER BY id"
            )
        ]
        assert sum(lengths[-10:]) < 2 * sum(lengths[:10])
    with EditSession.open(path) as session:
        for _ in range(40):
            assert session.tracks.undo()
        assert session.tracks.get_node_attr(4, "review_number") == 0


@pytest.mark.parametrize("operation", ["paint", "merge"])
def test_mask_edits_reopen_and_undo(solution_tracks_2d, tmp_path, operation):
    from funtracks.user_actions import UserUpdateSegmentation

    from motile_tracker.data_views.views_coordinator.node_edits import MergeNodes

    path = tmp_path / "tracks.geff"
    before = snapshot(solution_tracks_2d)
    with EditSession.create(solution_tracks_2d, path) as session:
        if operation == "paint":
            pixels = (
                np.array([3, 3, 3, 3]),
                np.array([10, 10, 11, 11]),
                np.array([10, 11, 10, 11]),
            )
            UserUpdateSegmentation(
                session.tracks, 7, [(pixels, 0)], session.tracks.get_next_track_id()
            )
            assert session.tracks.graph_solution.has_node(7)
        else:
            MergeNodes(session.tracks, [2, 3])
            assert session.tracks.graph_solution.num_nodes() == 5
        after = snapshot(session.tracks)
        assert session.revision == 1
    with EditSession.open(path) as session:
        assert snapshot(session.tracks) == after
        session.tracks.undo()
        assert snapshot(session.tracks) == before
