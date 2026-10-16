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


def write_trench(group: Path, *, break_tracklet: bool):
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
    )
    seg[...] = labels
    images = zarr.create_array(
        store=str(group / "images"),
        shape=labels.shape,
        chunks=labels.shape,
        dtype="float32",
        zarr_format=3,
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
    _array(geff / "nodes/ids", ids)
    for name, values in props.items():
        _group(geff / "nodes/props" / name)
        _array(geff / "nodes/props" / name / "values", values)
    _array(geff / "edges/ids", np.asarray(LINKS, dtype=np.uint64))
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
            "edge_props_metadata": {},
            "sphere": None,
            "ellipsoid": None,
            "track_node_props": {"lineage": "lineage", "tracklet": "tracklet"},
            "related_objects": [
                {"type": "labels", "path": "../segmentation", "label_prop": "seg_id"},
                {"type": "image", "path": "../images", "label_prop": None},
            ],
            "display_hints": None,
            "extra": {
                "selected_cleanliness_export": {
                    "trench_id": 1,
                    "after_continuity_count": 3,
                }
            },
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


def test_a_rejected_publication_leaves_the_files_and_stops_autosave(
    archive, tmp_path, monkeypatch
):
    geff, _ = archive
    before = {k: v for k, v in files(geff.parent).items() if "edit_history" not in k}
    session, tracks = open_for_editing(geff, tmp_path)
    monkeypatch.setattr(sf, "check", lambda *args, **kwargs: ["boom"])

    UserDeleteNode(tracks, internal(9))
    with pytest.raises(RuntimeError, match="boom"):
        session.flush()
    assert session.poll() == "Autosave stopped"
    with pytest.raises(RuntimeError):
        session.close()

    after = {k: v for k, v in files(geff.parent).items() if "edit_history" not in k}
    assert after == before


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
