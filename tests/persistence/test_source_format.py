"""Datasets keep their own format while they are edited (persistence.source_format).

The fixture is laid out like the Trackastra trench bundle on the BioImage
Archive: a zarr v3 store with one group per trench holding ``images``,
``segmentation`` (per-frame labels, zstd) and a zarr v2 GEFF with zero-based ids
stored out of order, an id gap, properties time/y/x/coords/label/seg_id/
tracklet/lineage (``label`` repeats ``seg_id``, ``coords`` repeats ``[y, x]``),
track metadata, related objects and an ``extra`` entry.

Cells (external ids), 5 frames of 30 x 12 px:

    lineage 1: 0 (t0) -> 1 (t1) -> {2, 3} (t2); 2 -> 4 (t3); 3 -> 5 (t4, skips t3)
    lineage 2: 6 (t0) -> 8 (t1) -> 11 (t3, skips t2) -> 12 (t4)
    lineage 3: 9 (t4)
    (ids 7 and 10 do not exist)

With ``break_tracklet`` the archived tracklet number changes across both
frame-skipping links, as in the archive; the GEFF tracklet definition does not
allow that for 6 -> 8 -> 11 -> 12.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import zarr
from funtracks.user_actions import UserAddEdge, UserDeleteEdge, UserDeleteNode

from motile_tracker.import_export.geff_collection import load_geff_group
from motile_tracker.persistence import release
from motile_tracker.persistence import source_format as sf
from motile_tracker.persistence.paths import create_session
from motile_tracker.persistence.session import EditSession, PendingSession, has_history

FRAMES, HEIGHT, WIDTH = 5, 30, 12
# external id: (t, first row, last row, label, tracklet, lineage)
CELLS = {
    0: (0, 2, 9, 1, 1, 1),
    1: (1, 2, 10, 1, 1, 1),
    2: (2, 2, 6, 1, 2, 1),
    3: (2, 7, 11, 2, 3, 1),
    4: (3, 2, 6, 1, 2, 1),
    5: (4, 8, 12, 2, 4, 1),
    6: (0, 20, 25, 2, 5, 2),
    8: (1, 20, 26, 2, 5, 2),
    11: (3, 14, 18, 3, 7, 2),
    12: (4, 14, 19, 3, 7, 2),
    9: (4, 22, 26, 1, 6, 3),
}
LINKS = [(0, 1), (1, 2), (1, 3), (2, 4), (3, 5), (6, 8), (8, 11), (11, 12)]
STORED_ORDER = [6, 2, 0, 9, 4, 1, 12, 8, 5, 11, 3]  # not in id order, like the archive
COLUMNS = (3, 9)


def _array(path, data, chunks=None):
    import numcodecs

    array = zarr.create_array(
        store=str(path),
        shape=data.shape,
        chunks=chunks or tuple(max(1, s) for s in data.shape),
        dtype=data.dtype,
        zarr_format=2,
        compressors=numcodecs.Zstd(level=0),
        filters=None,
        fill_value=0,
        chunk_key_encoding={"name": "v2", "separator": "."},
        overwrite=True,
    )
    if data.size:
        array[...] = data


def _group(path):
    zarr.open_group(str(path), mode="w", zarr_format=2)


ARCHIVE_AXES = {
    "axis_names": ["time", "y", "x"],
    "axis_types": ["time", "space", "space"],
}
RAW_AXES = {"axis_names": ["time", "y", "x"]}


def write_trench(group: Path, *, break_tracklet: bool, raw: bool = False):
    """A trench in the archive's layout, or with ``raw`` in the layout of the raw
    Trackastra export: 32-bit time, label and positions, node and link
    ``weight`` properties, another property order, its own ``extra``, no
    ``axis_types`` on the arrays, and a ``_COMPLETE.json`` export marker."""
    group.mkdir(parents=True)
    (group / "zarr.json").write_text(
        json.dumps({"attributes": {}, "zarr_format": 3, "node_type": "group"}, indent=2)
    )
    labels = np.zeros((FRAMES, HEIGHT, WIDTH), dtype=np.uint16)
    for t, top, bottom, label, *_ in CELLS.values():
        labels[t, top : bottom + 1, COLUMNS[0] : COLUMNS[1]] = label
    seg = zarr.create_array(
        store=str(group / "segmentation"),
        shape=labels.shape,
        chunks=labels.shape,
        dtype="uint16",
        zarr_format=3,
        compressors=zarr.codecs.ZstdCodec(level=0, checksum=False),
        fill_value=0,
        attributes=RAW_AXES if raw else ARCHIVE_AXES,
    )
    seg[...] = labels
    images = zarr.create_array(
        store=str(group / "images"),
        shape=labels.shape,
        chunks=labels.shape,
        dtype="float32",
        zarr_format=3,
        attributes=RAW_AXES if raw else ARCHIVE_AXES,
    )
    images[...] = 100.0 - labels.astype(np.float32)

    geff = group / "tracking_graph.geff"
    _group(geff)
    for sub in ("nodes", "nodes/props", "edges", "edges/props"):
        _group(geff / sub)
    ids = np.asarray(STORED_ORDER, dtype=np.uint64)
    t = np.asarray([CELLS[n][0] for n in STORED_ORDER], dtype=np.int64)
    y = np.asarray(
        [(CELLS[n][1] + CELLS[n][2]) / 2 for n in STORED_ORDER], dtype=np.float64
    )
    x = np.full(len(ids), (COLUMNS[0] + COLUMNS[1] - 1) / 2, dtype=np.float64)
    seg_id = np.asarray([CELLS[n][3] for n in STORED_ORDER], dtype=np.int64)
    tracklet = np.asarray([CELLS[n][4] for n in STORED_ORDER], dtype=np.int64)
    if not break_tracklet:
        tracklet[STORED_ORDER.index(5)] = CELLS[3][4]
        for n in (11, 12):
            tracklet[STORED_ORDER.index(n)] = CELLS[8][4]
    lineage = np.asarray([CELLS[n][5] for n in STORED_ORDER], dtype=np.int64)
    if raw:
        t, y, x = t.astype(np.int32), y.astype(np.float32), x.astype(np.float32)
        props = {
            "tracklet": tracklet,
            "x": x,
            "coords": np.stack([y, x], axis=1),
            "lineage": lineage,
            "weight": np.arange(1, len(ids) + 1, dtype=np.int64),
            "label": seg_id.astype(np.int32),
            "y": y,
            "seg_id": seg_id,
            "time": t,
        }
        edge_props = {"weight": np.linspace(0.5, 1.0, len(LINKS)).astype(np.float32)}
        extra = {"trackastra_export": {"trench_id": 2, "model_name": "general_2d"}}
        (group / "_COMPLETE.json").write_text(json.dumps({"trench_id": 2}, indent=2))
    else:
        props = {
            "tracklet": tracklet,
            "seg_id": seg_id,
            "coords": np.stack([y, x], axis=1),
            "x": x,
            "y": y,
            "lineage": lineage,
            "label": seg_id.copy(),
            "time": t,
        }
        edge_props = {}
        extra = {
            "selected_cleanliness_export": {"trench_id": 1, "after_continuity_count": 3}
        }
    _array(geff / "nodes/ids", ids)
    for name, values in props.items():
        _group(geff / "nodes/props" / name)
        _array(geff / "nodes/props" / name / "values", values)
    _array(geff / "edges/ids", np.asarray(LINKS, dtype=np.uint64))
    for name, values in edge_props.items():
        _group(geff / "edges/props" / name)
        _array(geff / "edges/props" / name / "values", values)
    meta = {
        "geff": {
            "geff_version": "1.1",
            "directed": True,
            "axes": [
                {
                    "name": "time",
                    "type": "time",
                    "unit": None,
                    "min": 0.0,
                    "max": 4.0,
                    "scale": None,
                    "scaled_unit": None,
                    "offset": None,
                },
                {
                    "name": "y",
                    "type": "space",
                    "unit": None,
                    "min": float(y.min()),
                    "max": float(y.max()),
                    "scale": None,
                    "scaled_unit": None,
                    "offset": None,
                },
                {
                    "name": "x",
                    "type": "space",
                    "unit": None,
                    "min": float(x.min()),
                    "max": float(x.max()),
                    "scale": None,
                    "scaled_unit": None,
                    "offset": None,
                },
            ],
            "node_props_metadata": {
                name: {
                    "identifier": name,
                    "dtype": str(values.dtype),
                    "varlength": False,
                    "unit": None,
                    "name": None,
                    "description": None,
                }
                for name, values in props.items()
            },
            "edge_props_metadata": {
                name: {
                    "identifier": name,
                    "dtype": str(values.dtype),
                    "varlength": False,
                    "unit": None,
                    "name": None,
                    "description": None,
                }
                for name, values in edge_props.items()
            },
            "sphere": None,
            "ellipsoid": None,
            "track_node_props": {"lineage": "lineage", "tracklet": "tracklet"},
            "related_objects": [
                {"type": "labels", "path": "../segmentation", "label_prop": "seg_id"},
                {"type": "image", "path": "../images", "label_prop": None},
            ],
            "display_hints": None,
            "extra": extra,
        }
    }
    (geff / ".zattrs").write_text(json.dumps(meta, indent=2))
    return geff


@pytest.fixture
def archive(tmp_path):
    """A one-trench store in the archive's format, with an archived tracklet break."""
    store = tmp_path / "trenches.zarr"
    store.mkdir()
    (store / "zarr.json").write_text(
        json.dumps({"attributes": {}, "zarr_format": 3, "node_type": "group"}, indent=2)
    )
    geff = write_trench(store / "trench_a", break_tracklet=True)
    pristine = tmp_path / "pristine.zarr"
    shutil.copytree(store, pristine)
    return geff, pristine / "trench_a" / "tracking_graph.geff"


@pytest.fixture
def valid_archive(tmp_path):
    store = tmp_path / "valid.zarr"
    return write_trench(store / "trench_a", break_tracklet=False)


def files(root: Path) -> dict:
    return {
        str(p.relative_to(root)): p.read_bytes()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def internal(n):
    return n + 1  # funtracks offsets zero-based ids by one


def published(geff):
    """The published dataset as plain dictionaries keyed by external id."""
    root = zarr.open_group(str(geff), mode="r")
    ids = root["nodes/ids"][:].astype(int).tolist()
    props = {
        k: root[f"nodes/props/{k}/values"][:] for k in root["nodes/props"].group_keys()
    }
    table = {n: {k: v[i].tolist() for k, v in props.items()} for i, n in enumerate(ids)}
    edges = [tuple(e) for e in root["edges/ids"][:].astype(int).tolist()]
    labels = zarr.open_array(str(geff.parent / "segmentation"), mode="r")[:]
    return ids, table, edges, labels


def validate(geff):
    from geff import validate_structure
    from geff.core_io import read_to_memory
    from geff.validate.data import ValidationConfig, validate_data

    validate_structure(str(geff))
    validate_data(
        read_to_memory(str(geff)),
        ValidationConfig(graph=True, tracklet=True, lineage=True),
    )


# ----------------------------------------------------------------------
# capture and rebuild
# ----------------------------------------------------------------------
def test_capture_records_the_format(archive):
    geff, _ = archive
    fmt = sf.capture(geff, load_geff_group(geff).tracks)

    assert fmt["id_offset"] == 1
    assert fmt["node_order"] == STORED_ORDER
    assert fmt["roles"]["time"] == "time"
    assert fmt["roles"]["space"] == ["y", "x"]
    assert fmt["roles"]["label"] == "seg_id"
    assert fmt["roles"]["other"] == {"coords": "position_alias", "label": "label_alias"}
    assert fmt["labels"]["path"] == "../segmentation"


def test_rebuild_without_edits_is_byte_identical(valid_archive, tmp_path):
    geff = valid_archive
    tracks = load_geff_group(geff).tracks
    fmt = sf.capture(geff, tracks)
    out = tmp_path / "out"
    content = sf.write(tracks, fmt, out / "tracking_graph.geff", out / "segmentation")

    assert (
        sf.check(out / "tracking_graph.geff", out / "segmentation", fmt, content) == []
    )
    assert files(out / "tracking_graph.geff") == files(geff)
    assert files(out / "segmentation") == files(geff.parent / "segmentation")


def test_rebuild_fixes_only_archived_tracklet_breaks(archive, tmp_path):
    geff, _ = archive
    with pytest.raises(ValueError, match="Not maximal"):
        validate(geff)
    tracks = load_geff_group(geff).tracks
    fmt = sf.capture(geff, tracks)
    out = tmp_path / "out" / "tracking_graph.geff"
    sf.write(tracks, fmt, out, tmp_path / "out" / "segmentation")

    before, after = files(geff), files(out)
    assert [k for k in before if before[k] != after.get(k)] == [
        "nodes/props/tracklet/values/0"
    ]
    validate(out)
    _, table, _, _ = published(out)
    # the cell after the frame skip continues the earlier tracklet's number
    assert table[5]["tracklet"] == table[3]["tracklet"] == 3
    assert table[11]["tracklet"] == table[12]["tracklet"] == table[8]["tracklet"] == 5
    expected = {0: 1, 1: 1, 2: 2, 4: 2, 6: 5, 9: 6}
    assert {n: table[n]["tracklet"] for n in expected} == expected


# ----------------------------------------------------------------------
# editing in the app
# ----------------------------------------------------------------------
def open_for_editing(geff, tmp_path):
    loaded = load_geff_group(geff)
    session = create_session(loaded.tracks, geff, tmp_path / "working")
    return session, session.tracks if hasattr(session, "tracks") else loaded.tracks


def test_opening_writes_nothing(archive, tmp_path):
    geff, _ = archive
    trench = geff.parent
    before = files(trench)

    session, _ = open_for_editing(geff, tmp_path)
    assert isinstance(session, PendingSession)
    assert session.poll() == "No edits yet"
    session.close()

    assert files(trench) == before
    assert not has_history(geff)


def test_first_edit_publishes_in_the_dataset_format(archive, tmp_path):
    geff, _ = archive
    original_meta = json.loads((geff / ".zattrs").read_text())
    session, tracks = open_for_editing(geff, tmp_path)

    UserDeleteNode(tracks, internal(4))
    session.flush()

    assert has_history(geff)
    ids, table, edges, labels = published(geff)
    assert ids == [6, 2, 0, 9, 1, 12, 8, 5, 11, 3]  # original ids and order
    assert set(table[0]) == {
        "time",
        "y",
        "x",
        "coords",
        "label",
        "seg_id",
        "tracklet",
        "lineage",
    }
    assert (2, 4) not in edges
    assert not np.any(labels[3] == 1)  # the deleted cell's label is gone
    assert np.count_nonzero(labels[3] == 3) == 5 * 6  # the other cell is untouched
    meta = json.loads((geff / ".zattrs").read_text())
    assert meta["geff"]["related_objects"] == original_meta["geff"]["related_objects"]
    assert meta["geff"]["track_node_props"] == original_meta["geff"]["track_node_props"]
    assert meta["geff"]["extra"] == original_meta["geff"]["extra"]
    validate(geff)

    tracks.undo()
    session.flush()
    session.close()
    _, table, edges, labels = published(geff)
    assert (2, 4) in edges
    assert np.count_nonzero(labels[3] == 1) == 5 * 6


def test_split_gives_a_new_id_and_the_lowest_free_label(archive, tmp_path):
    from motile_tracker.data_views.views_coordinator.node_edits import (
        SplitNode,
        propose_split,
    )

    geff, _ = archive
    session, tracks = open_for_editing(geff, tmp_path)
    plan = propose_split(tracks, internal(1))
    action = SplitNode(tracks, plan)
    session.flush()
    session.close()

    new = action.new_node - 1
    ids, table, _, labels = published(geff)
    assert new == 13  # above every id ever used (12), not in an id gap
    assert ids[-1] == new
    assert table[new]["time"] == 1
    assert table[new]["seg_id"] == table[new]["label"] == 3  # 1 and 2 are taken in t1
    assert table[new]["coords"] == [table[new]["y"], table[new]["x"]]
    assert np.count_nonzero(labels[1] == 3) > 0
    validate(geff)


def test_links_renumber_tracklets_keeping_the_earlier_number(archive, tmp_path):
    geff, _ = archive
    session, tracks = open_for_editing(geff, tmp_path)

    # join 4 (t3, tracklet 2, lineage 1) to the lone 9 (t4, tracklet 6, lineage 3)
    UserAddEdge(tracks, (internal(4), internal(9)))
    session.flush()
    _, table, _, _ = published(geff)
    assert table[9]["tracklet"] == 2
    assert table[9]["lineage"] == 1

    # breaking 2 -> 4 leaves 4 -> 9 on its own: new tracklet and lineage numbers
    UserDeleteEdge(tracks, (internal(2), internal(4)))
    session.flush()
    session.close()
    _, table, _, _ = published(geff)
    assert table[4]["tracklet"] == table[9]["tracklet"] == 8  # above the archived 7
    assert table[4]["lineage"] == table[9]["lineage"] == 4  # above the archived 3
    assert table[2]["tracklet"] == 2
    validate(geff)


def test_reopening_does_not_rewrite_the_files(archive, tmp_path):
    geff, _ = archive
    session, tracks = open_for_editing(geff, tmp_path)
    UserDeleteNode(tracks, internal(9))
    session.close()
    before = files(geff.parent)

    with EditSession.open(geff):
        pass

    after = files(geff.parent)
    changed = [k for k in before if before[k] != after.get(k)]
    assert all(k.startswith("tracking_graph.geff/edit_history/") for k in changed)
    assert not any("nodes/" in k or "segmentation" in k for k in changed)


def test_a_rejected_publication_leaves_the_files_and_can_be_undone(archive, tmp_path):
    geff, _ = archive
    before = {k: v for k, v in files(geff.parent).items() if "edit_history" not in k}
    session, tracks = open_for_editing(geff, tmp_path)

    # 5 and 9 are both in frame 4. Funtracks makes the link; the gate refuses it.
    UserAddEdge(tracks, (internal(5), internal(9)))
    session.flush()
    assert "forward in time" in session.rejected
    assert session.error is None
    assert session.poll() == "Not written to the GEFF"
    session.close()
    after = {k: v for k, v in files(geff.parent).items() if "edit_history" not in k}
    assert after == before

    # Reopening (as the app does) is refused the same way, and the edit can be undone.
    tracks = load_geff_group(geff).tracks
    session = tracks.edit_session
    assert "forward in time" in session.rejected
    tracks.action_history.undo()
    session.flush()
    assert session.rejected is None
    assert session.poll() == "Saved"
    session.close()
    _, _, edges, _ = published(geff)
    assert (5, 9) not in edges
    validate(geff)


def test_edits_after_a_rejected_publication_are_published(archive, tmp_path):
    geff, _ = archive
    session, tracks = open_for_editing(geff, tmp_path)
    UserAddEdge(tracks, (internal(5), internal(9)))
    session.flush()
    assert session.rejected

    # Removing the bad link by hand, rather than undoing it, also clears the refusal.
    UserDeleteEdge(tracks, (internal(5), internal(9)))
    UserDeleteNode(tracks, internal(12))
    session.flush()
    session.close()
    assert session.rejected is None
    ids, _, edges, _ = published(geff)
    assert 12 not in ids
    assert (5, 9) not in edges
    validate(geff)


def test_read_only_dataset_is_edited_in_a_working_copy(archive, tmp_path):
    import os

    geff, _ = archive
    trench = geff.parent
    before = files(trench)
    paths = [*trench.rglob("*"), trench]
    try:
        for p in paths:
            p.chmod(0o555 if p.is_dir() else 0o444)
        session, tracks = open_for_editing(geff, tmp_path)
        assert session.path != geff
        UserDeleteNode(tracks, internal(9))
        session.flush()
        copy = session.path
        session.close()
    finally:
        for p in paths:
            os.chmod(p, 0o755 if p.is_dir() else 0o644)

    assert files(trench) == before
    root = zarr.open_group(str(copy), mode="r")
    labels_path = (copy / sf._labels_object(dict(root.attrs)["geff"])["path"]).resolve()
    assert labels_path.parent == copy.parent
    assert not np.any(zarr.open_array(str(labels_path), mode="r")[4] == 1)


def test_unsupported_format_is_edited_in_a_working_copy(archive, tmp_path, monkeypatch):
    geff, _ = archive
    before = files(geff.parent)

    def unsupported(*args, **kwargs):
        raise sf.UnsupportedFormat("variable-length property")

    monkeypatch.setattr(sf, "capture", unsupported)
    with pytest.warns(UserWarning, match="cannot be kept"):
        session, tracks = open_for_editing(geff, tmp_path)
    UserDeleteNode(tracks, internal(9))
    session.flush()
    session.close()

    assert session.path != geff
    assert files(geff.parent) == before


# ----------------------------------------------------------------------
# older histories and releases
# ----------------------------------------------------------------------
def test_attach_format_to_an_older_history_and_republish(archive):
    geff, pristine = archive
    tracks = load_geff_group(geff).tracks
    with EditSession.create(tracks, geff, source_format=None) as session:
        UserDeleteNode(session.tracks, internal(9))
        session.flush()
    # the old behaviour: funtracks' own format, label array no longer referenced
    assert "mask" in zarr.open_group(str(geff), mode="r")["nodes/props"]

    assert release.attach_format(geff, pristine) == "recorded"
    release.republish(geff)

    ids, table, _, labels = published(geff)
    assert 9 not in ids and set(table[0]) == {
        "time",
        "y",
        "x",
        "coords",
        "label",
        "seg_id",
        "tracklet",
        "lineage",
    }
    assert not np.any(labels[4] == 1)
    validate(geff)
    with EditSession.open(geff) as session:  # still editable afterwards
        assert not session.tracks.graph.has_node(internal(9))


def test_attach_format_refuses_a_different_reference(archive, tmp_path):
    geff, pristine = archive
    other = tmp_path / "other.zarr" / "trench_a"
    write_trench(other, break_tracklet=False)
    labels = zarr.open_array(str(other / "segmentation"), mode="r+")
    data = labels[:]
    data[0, 2, 3] = 0  # one pixel differs
    labels[...] = data
    tracks = load_geff_group(geff).tracks
    EditSession.create(tracks, geff, source_format=None).close()

    with pytest.raises(sf.UnsupportedFormat, match="differs"):
        release.attach_format(geff, other / "tracking_graph.geff")


def test_export_writes_a_release_without_histories_or_extra(archive, tmp_path):
    geff, _ = archive
    store = geff.parent.parent
    write_trench(store / "trench_b", break_tracklet=False)
    untouched = files(store / "trench_b")
    for trench, edit in (("trench_a", True), ("trench_b", False)):
        path = store / trench / "tracking_graph.geff"
        session, tracks = open_for_editing(path, tmp_path)
        if edit:
            UserDeleteNode(tracks, internal(9))
        else:
            # a history without edits, as for a trench that was only reviewed
            session.close()
            session = EditSession.create(tracks, path)
        session.flush()
        session.close()
    report = tmp_path / "report.csv"

    rows = release.export(
        store, tmp_path / "release.zarr", drop_extra=True, report=report
    )

    out = tmp_path / "release.zarr"
    assert (out / "zarr.json").is_file()
    assert not list(out.rglob("edit_history")) and not list(out.rglob("writer.lock"))
    for trench in ("trench_a", "trench_b"):
        meta = json.loads(
            (out / trench / "tracking_graph.geff" / ".zattrs").read_text()
        )
        assert "extra" not in meta["geff"]
        validate(out / trench / "tracking_graph.geff")
    released_b = files(out / "trench_b")
    assert [k for k in untouched if untouched[k] != released_b.get(k)] == [
        "tracking_graph.geff/.zattrs"
    ]
    by_name = {row["dataset"]: row for row in rows}
    assert by_name["trench_a"]["cells_removed"] == 1
    assert by_name["trench_b"]["cells_removed"] == 0
    assert by_name["trench_b"]["cells_with_new_tracklet_number"] == 0
    assert report.read_text().splitlines()[0].startswith("dataset,")


def test_release_commands_with_relative_paths(archive, tmp_path, monkeypatch):
    geff, pristine = archive
    tracks = load_geff_group(geff).tracks
    with EditSession.create(tracks, geff, source_format=None) as session:
        UserDeleteNode(session.tracks, internal(9))
        session.flush()
    monkeypatch.chdir(tmp_path)
    store = Path(geff.parent.parent.name)
    reference = Path(pristine.parent.parent.name)

    with pytest.raises(ValueError, match="no recorded format"):
        release.main(["republish", str(store)])
    assert (
        release.main(["attach-format", str(store), "--reference", str(reference)]) == 0
    )
    assert release.main(["republish", str(store)]) == 0
    validate(geff)
    assert release.main(["export", str(store), "out.zarr", "--drop-extra"]) == 0
    assert (tmp_path / "out.zarr" / "trench_a" / "segmentation").is_dir()


def test_attach_format_records_nothing_unless_every_dataset_matches(archive, tmp_path):
    from motile_tracker.persistence.store import Store

    geff, pristine = archive
    store = geff.parent.parent
    write_trench(store / "trench_b", break_tracklet=False)
    for trench in ("trench_a", "trench_b"):
        path = store / trench / "tracking_graph.geff"
        EditSession.create(
            load_geff_group(path).tracks, path, source_format=None
        ).close()
    # the reference holds trench_a only
    with pytest.raises(sf.UnsupportedFormat):
        release.main(
            ["attach-format", str(store), "--reference", str(pristine.parent.parent)]
        )
    for trench in ("trench_a", "trench_b"):
        with Store(store / trench / "tracking_graph.geff", readonly=True) as s:
            assert s.metadata("source_format") is None


def test_release_commands_leave_datasets_without_history_alone(archive, tmp_path):
    geff, pristine = archive
    store = geff.parent.parent
    write_trench(store / "trench_b", break_tracklet=False)  # never edited
    untouched = files(store / "trench_b")
    tracks = load_geff_group(geff).tracks
    with EditSession.create(tracks, geff, source_format=None) as session:
        UserDeleteNode(session.tracks, internal(9))
        session.flush()

    assert (
        release.main(
            ["attach-format", str(store), "--reference", str(pristine.parent.parent)]
        )
        == 0
    )
    assert release.main(["republish", str(store)]) == 0

    validate(geff)
    assert files(store / "trench_b") == untouched
    assert not has_history(store / "trench_b" / "tracking_graph.geff")


# ----------------------------------------------------------------------
# changing a dataset's format to another's
# ----------------------------------------------------------------------
def layout(group: Path) -> dict:
    """Everything about a trench's files except its data and dataset values."""
    geff = group / "tracking_graph.geff"

    def is_metadata(p):
        if not p.is_file() or "edit_history" in p.parts or "c" in p.parts:
            return False
        return p.name.startswith(".") or p.suffix == ".json"

    out = {
        "files": sorted(
            str(p.relative_to(group)) for p in group.rglob("*") if is_metadata(p)
        )
    }
    for p in sorted(geff.rglob(".zarray")):
        if "edit_history" in p.parts:
            continue
        meta = json.loads(p.read_text())
        out[str(p.relative_to(group))] = {
            k: v for k, v in meta.items() if k not in ("shape", "chunks")
        }
    root = json.loads((geff / ".zattrs").read_text())
    for axis in root["geff"]["axes"]:
        axis.pop("min"), axis.pop("max")
    out["root"] = json.dumps(root)  # key order matters
    for name in ("images", "segmentation"):
        meta = json.loads((group / name / "zarr.json").read_text())
        out[name] = {k: v for k, v in meta.items() if k != "shape"}
    return out


def values(group: Path) -> dict:
    root = zarr.open_group(str(group / "tracking_graph.geff"), mode="r")
    ids = root["nodes/ids"][:].astype(int).tolist()
    props = {
        k: root[f"nodes/props/{k}/values"][:].astype(float).tolist()
        for k in ("time", "y", "x", "coords", "label", "seg_id", "tracklet", "lineage")
    }
    return {
        "ids": ids,
        "props": props,
        "edges": root["edges/ids"][:].astype(int).tolist(),
        "labels": zarr.open_array(str(group / "segmentation"), mode="r")[:].tobytes(),
        "images": zarr.open_array(str(group / "images"), mode="r")[:].tobytes(),
    }


@pytest.fixture
def mixed_store(tmp_path):
    """An archive-layout trench (with a history) and two raw trenches, one with a
    history and an edit, one never edited; plus an archive-layout template."""
    store = tmp_path / "mixed.zarr"
    write_trench(store / "trench_a", break_tracklet=False)
    write_trench(store / "trench_b", break_tracklet=False, raw=True)
    write_trench(store / "trench_c", break_tracklet=False, raw=True)
    template = tmp_path / "template.zarr" / "trench_t"
    write_trench(template, break_tracklet=False)
    for trench in ("trench_a", "trench_b"):
        path = store / trench / "tracking_graph.geff"
        session, tracks = open_for_editing(path, tmp_path)
        if trench == "trench_b":
            UserDeleteNode(tracks, internal(9))
        else:
            session.close()
            session = EditSession.create(tracks, path)
        session.flush()
        session.close()
    return store, template / "tracking_graph.geff"


def test_conform_rewrites_raw_trenches_in_the_archive_layout(mixed_store):
    store, template = mixed_store
    before = {t: values(store / t) for t in ("trench_a", "trench_b", "trench_c")}
    assert layout(store / "trench_b") != layout(template.parent)

    assert (
        release.main(
            [
                "conform",
                str(store),
                "--like",
                str(template),
                "--drop-extra",
                "--remove",
                "_COMPLETE.json",
            ]
        )
        == 0
    )

    expected = layout(template.parent)
    root = json.loads(expected["root"])
    root["geff"].pop("extra")
    expected["root"] = json.dumps(root)
    for trench in ("trench_a", "trench_b", "trench_c"):
        group = store / trench
        assert layout(group) == expected, trench
        validate(group / "tracking_graph.geff")
        after = values(group)
        # the data is unchanged, only its encoding
        for key in ("ids", "edges", "props", "labels", "images"):
            assert after[key] == before[trench][key], (trench, key)
    assert not (store / "trench_b" / "_COMPLETE.json").exists()
    assert not has_history(store / "trench_c" / "tracking_graph.geff")
    assert (
        sf.read_layout(store / "trench_b" / "tracking_graph.geff")["node_props"]
        == (sf.read_layout(template)["node_props"])
    )


def test_editing_after_conform_keeps_the_new_layout(mixed_store, tmp_path):
    store, template = mixed_store
    release.main(
        [
            "conform",
            str(store),
            "--like",
            str(template),
            "--drop-extra",
            "--remove",
            "_COMPLETE.json",
        ]
    )
    expected = layout(store / "trench_a")

    for trench in ("trench_b", "trench_c"):  # with and without a history
        path = store / trench / "tracking_graph.geff"
        if has_history(path):  # as the app does for a dataset with a history
            session = EditSession.open(path)
            tracks = session.tracks
        else:
            session, tracks = open_for_editing(path, tmp_path)
        UserDeleteNode(tracks, internal(6))
        session.flush()
        session.close()
        assert layout(store / trench) == expected
        validate(path)


def test_conform_changes_nothing_unless_every_dataset_passes(mixed_store, monkeypatch):
    store, template = mixed_store
    before = files(store)
    real_check = sf.check
    calls = []

    def fail_on_third(*args, **kwargs):
        calls.append(1)
        return ["boom"] if len(calls) == 3 else real_check(*args, **kwargs)

    monkeypatch.setattr(sf, "check", fail_on_third)
    with pytest.raises(sf.PublicationRejected, match="boom"):
        release.main(["conform", str(store), "--like", str(template), "--drop-extra"])
    assert files(store) == before


def test_conform_refuses_a_template_with_properties_the_dataset_lacks(mixed_store):
    store, _ = mixed_store
    raw_template = store / "trench_c" / "tracking_graph.geff"
    with pytest.raises(sf.UnsupportedFormat, match="weight"):
        release.conform_store(store, raw_template)


def test_export_includes_datasets_without_history(mixed_store, tmp_path):
    store, template = mixed_store
    release.main(["conform", str(store), "--like", str(template), "--drop-extra"])
    rows = release.export(store, tmp_path / "out.zarr", drop_extra=True)
    assert {row["dataset"] for row in rows} == {"trench_a", "trench_b", "trench_c"}
    for trench in ("trench_a", "trench_b", "trench_c"):
        validate(tmp_path / "out.zarr" / trench / "tracking_graph.geff")
