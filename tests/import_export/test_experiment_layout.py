"""PyMMM's experiment layout: one OME-Zarr image per trench, a tracker's GEFF that
is never edited, and the curated copy made at the first edit."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import zarr
from funtracks.user_actions import UserDeleteNode

from mm_track import experiment_layout
from mm_track.data_views.views_coordinator.tracks_list import TracksList
from mm_track.import_export.geff_collection import (
    find_geff_groups,
    load_geff_group,
    read_related_arrays,
)
from mm_track.ome_zarr import image_channels, multiscale_array
from mm_track.persistence.session import EditSession, PendingSession

RUN = "chain_S1"
CHUNK_KEYS = {"name": "default", "separator": "."}
EDGES = [(1, 2), (1, 3), (3, 4), (4, 5)]
TRACKLETS = {1: 1, 2: 2, 3: 3, 4: 3, 5: 3, 6: 5}
LINEAGES = {1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 2}


def _multiscale(name, axes):
    return {
        "name": name,
        "axes": axes,
        "datasets": [{"path": "0", "coordinateTransformations": [
            {"type": "scale", "scale": [1.0] * len(axes)}]}],
    }


def _write_trench(root, name, segmentation):
    """A trench as PyMMM writes it: a (t, c, y, x) OME image, the tracker's
    (t, y, x) label image and its zarr v3 GEFF (node ids from 0, seg_id = label)."""
    from geff.core_io import write_arrays
    from geff_spec import Axis, GeffMetadata, RelatedObject

    t_axes = [{"name": "t", "type": "time"}]
    yx = [{"name": "y", "type": "space"}, {"name": "x", "type": "space"}]
    image = root.create_group(name)
    image.attrs["ome"] = {
        "version": "0.5",
        "multiscales": [_multiscale(name, [*t_axes, {"name": "c", "type": "channel"}, *yx])],
        "omero": {"channels": [
            {"label": "PC", "color": "FFFFFF", "active": True,
             "window": {"min": 0, "max": 100, "start": 0, "end": 60}},
            {"label": "mCherry", "color": "FF0000", "active": False,
             "window": {"min": 0, "max": 100, "start": 0, "end": 10}},
        ]},
    }
    pixels = np.stack([segmentation * 10, segmentation], axis=1).astype(np.uint16)
    image.create_array("0", data=pixels, chunk_key_encoding=CHUNK_KEYS)
    labels = image.create_group("labels")
    label = labels.create_group(f"tracks.{RUN}")
    label.attrs.put({
        "ome": {"version": "0.5", "multiscales": [_multiscale(f"tracks.{RUN}", [*t_axes, *yx])],
                "image-label": {"source": {"image": "../../"}}},
        "pymmm": {"tracks": f"../../tracks/{RUN}.geff"},
    })
    label.create_array("0", data=segmentation.astype(np.uint16), chunks=segmentation.shape,
                       chunk_key_encoding=CHUNK_KEYS)
    labels.attrs["ome"] = {"version": "0.5", "labels": [f"tracks.{RUN}"]}

    rows = []
    for t, frame in enumerate(segmentation):
        for value in np.unique(frame[frame > 0]):
            ys, xs = np.nonzero(frame == value)
            rows.append((t, ys.mean(), xs.mean(), int(value)))
    index = {row[3]: i for i, row in enumerate(rows)}
    props = {
        "t": np.array([r[0] for r in rows], dtype=np.int32),
        "y": np.array([r[1] for r in rows], dtype=np.float32),
        "x": np.array([r[2] for r in rows], dtype=np.float32),
        "seg_id": np.array([r[3] for r in rows], dtype=np.int32),
        "track_id": np.array([TRACKLETS[r[3]] for r in rows], dtype=np.int64),
        "lineage_id": np.array([LINEAGES[r[3]] for r in rows], dtype=np.int64),
    }
    edges = np.array([(index[a], index[b]) for a, b in EDGES], dtype=np.int64)
    image.create_group("tracks")
    metadata = GeffMetadata(
        directed=True,
        axes=[Axis(name="t", type="time", unit="frame"), Axis(name="y", type="space"),
              Axis(name="x", type="space")],
        node_props_metadata={},
        edge_props_metadata={},
        track_node_props={"tracklet": "track_id", "lineage": "lineage_id"},
        related_objects=[
            RelatedObject(type="labels", path=f"../../labels/tracks.{RUN}", node_prop="seg_id"),
            RelatedObject(type="image", path="../.."),
        ],
        extra={"tracker": "chain", "trench": int(name.split("_")[-1])},
    )
    geff = Path(str(root.store.root)) / name / "tracks" / f"{RUN}.geff"
    write_arrays(
        str(geff), np.arange(len(rows), dtype=np.int64),
        {key: {"values": value, "missing": None} for key, value in props.items()},
        edges, {"score": {"values": np.ones(len(edges), dtype=np.float32), "missing": None}},
        metadata, zarr_format=3, overwrite=True,
    )
    return geff


@pytest.fixture
def experiment_store(tmp_path, segmentation_2d):
    """An experiment group with two trenches, each with tracks of RUN."""
    store = tmp_path / "experiment.zarr"
    root = zarr.open_group(store, mode="w", zarr_format=3)
    root.attrs["pymmm"] = {"layout": "experiment", "tracks": {RUN: {"tracker": "chain"}}}
    for name in ("trench_0000", "trench_0001"):
        _write_trench(root, name, np.asarray(segmentation_2d))
    return store


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    for file in sorted(path.rglob("*")):
        if file.is_file():
            digest.update(str(file.relative_to(path)).encode())
            digest.update(file.read_bytes())
    return digest.hexdigest()


def _tracker_output(trench: Path) -> dict:
    return {part: _digest(trench / part) for part in ("tracks", f"labels/tracks.{RUN}")}


def _node_of(geff_path, seg_id):
    """The loaded node id of the GEFF node with this seg_id (ids are offset by one
    on import, as the GEFF numbers its nodes from 0)."""
    seg_ids = zarr.open_array(Path(geff_path) / "nodes/props/seg_id/values", mode="r")[:]
    return int(np.flatnonzero(seg_ids == seg_id)[0]) + 1


def _open(geff_path, tmp_path, qtbot):
    """Open a trench's tracks the way the viewer does: load, then add to a list."""
    widget = TracksList()
    qtbot.addWidget(widget)
    loaded = load_geff_group(geff_path)
    tracks = widget.add_tracks(loaded.tracks, "t", source_path=loaded.geff_path)
    return widget, tracks


def test_find_geff_groups_lists_one_tracker_geff_per_trench(experiment_store):
    groups = find_geff_groups(experiment_store)

    assert [g.name for g in groups] == ["trench_0000", "trench_0001"]
    assert groups[0].path == experiment_store / "trench_0000" / "tracks" / f"{RUN}.geff"
    assert find_geff_groups(experiment_store, run="other") == []


def test_ome_groups_stand_for_their_arrays(experiment_store):
    trench = experiment_store / "trench_0000"
    related = read_related_arrays(trench / "tracks" / f"{RUN}.geff")

    assert related.labels_path == trench / "labels" / f"tracks.{RUN}" / "0"
    assert related.image_path == trench / "0"
    assert related.channel_axis == 1
    assert [c.name for c in related.channels] == ["PC", "mCherry"]
    assert related.channels[1].contrast_limits == (0.0, 10.0)
    assert not related.channels[1].visible
    # a bare array is its own array, and has no channels
    assert multiscale_array(trench / "0") == trench / "0"
    assert image_channels(trench / "0") == (None, [])


def test_tracker_geff_loads_with_its_channels(experiment_store, segmentation_2d):
    loaded = load_geff_group(experiment_store / "trench_0000" / "tracks" / f"{RUN}.geff")

    assert loaded.tracks.graph.num_nodes() == 6
    assert [c.name for c, _ in loaded.channels] == ["PC", "mCherry"]
    assert all(data.shape == segmentation_2d.shape for _, data in loaded.channels)
    np.testing.assert_array_equal(loaded.image, np.asarray(segmentation_2d) * 10)


def test_every_channel_is_listed_and_only_active_ones_shown(
    make_napari_viewer, experiment_store
):
    """The kymograph view lists no spatial image layers, so each channel other than
    the background is an overlay: there to be switched on, hidden unless its
    omero says active."""
    from mm_track.data_views.views_coordinator.tracks_viewer import TracksViewer
    from mm_track.import_export.geff_collection import add_geff_group_to_viewer

    viewer = make_napari_viewer()
    tracker = experiment_store / "trench_0000" / "tracks" / f"{RUN}.geff"

    loaded = add_geff_group_to_viewer(viewer, tracker, "trench_0000", kymograph=False)

    assert [layer.name for layer in loaded.channel_layers] == [
        "trench_0000_PC", "trench_0000_mCherry"]
    assert all(layer in viewer.layers for layer in loaded.channel_layers)
    assert viewer.layers["trench_0000_PC"].visible
    assert not viewer.layers["trench_0000_mCherry"].visible

    tracks_viewer = TracksViewer.get_instance(viewer)
    tracks_viewer.set_view_mode("kymograph")
    kymograph = tracks_viewer.kymograph_layers
    overlay = kymograph.overlay_layers["trench_0000_mCherry"]
    assert kymograph.image_layer_name == "trench_0000_PC"
    assert kymograph.background_layer in viewer.layers
    assert kymograph.background_layer.visible
    assert overlay in viewer.layers
    assert not overlay.visible
    tracks_viewer.tracks_list.close_sessions()


def test_opening_writes_nothing(experiment_store, tmp_path, qtbot):
    trench = experiment_store / "trench_0000"
    before = _digest(trench)

    widget, tracks = _open(trench / "tracks" / f"{RUN}.geff", tmp_path, qtbot)
    widget.close_sessions()

    assert isinstance(tracks.edit_session, PendingSession)
    assert _digest(trench) == before


def test_first_edit_makes_the_curated_copy(experiment_store, tmp_path, qtbot):
    trench = experiment_store / "trench_0000"
    tracker = trench / "tracks" / f"{RUN}.geff"
    before = _tracker_output(trench)
    widget, tracks = _open(tracker, tmp_path, qtbot)
    deleted = _node_of(tracker, 6)  # the cell with no links, in frame 4
    assert tracks.get_time(deleted) == 4

    UserDeleteNode(tracks, deleted)
    session = tracks.edit_session
    session.flush()

    curated = trench / "curation" / f"{RUN}.geff"
    assert session.path == curated
    assert widget.session_at(curated) is session
    assert _tracker_output(trench) == before  # the tracker's output is only read
    root = json.loads((curated / "zarr.json").read_text())["attributes"]["geff"]
    assert root["related_objects"][0]["path"] == f"../../labels/curation.{RUN}"
    assert root["related_objects"][1]["path"] == "../.."
    label = json.loads((trench / "labels" / f"curation.{RUN}" / "zarr.json").read_text())
    assert label["attributes"]["ome"]["multiscales"][0]["name"] == f"curation.{RUN}"
    assert json.loads((trench / "labels" / "zarr.json").read_text())["attributes"]["ome"][
        "labels"] == [f"tracks.{RUN}", f"curation.{RUN}"]
    masks = zarr.open_array(trench / "labels" / f"curation.{RUN}" / "0", mode="r")[:]
    original = zarr.open_array(trench / "labels" / f"tracks.{RUN}" / "0", mode="r")[:]
    assert not (masks[4] == 6).any() and (original[4] == 6).any()
    np.testing.assert_array_equal(masks[:4], original[:4])
    assert experiment_layout.trench_state(tracker) == "edited"
    widget.close_sessions()


def test_reopening_resumes_the_curated_copy(experiment_store, tmp_path, qtbot):
    tracker = experiment_store / "trench_0000" / "tracks" / f"{RUN}.geff"
    deleted = _node_of(tracker, 6)
    widget, tracks = _open(tracker, tmp_path, qtbot)
    UserDeleteNode(tracks, deleted)
    widget.close_sessions()

    loaded = load_geff_group(tracker)
    session = loaded.tracks.edit_session

    assert loaded.geff_path == experiment_store / "trench_0000" / "curation" / f"{RUN}.geff"
    assert isinstance(session, EditSession)
    assert not loaded.tracks.graph.has_node(deleted)
    loaded.tracks.undo()
    assert loaded.tracks.graph.has_node(deleted)
    session.close()


def test_review_mark_survives_edits_and_reopening(experiment_store, tmp_path, qtbot):
    tracker = experiment_store / "trench_0000" / "tracks" / f"{RUN}.geff"
    curated = tracker.parents[1] / "curation" / f"{RUN}.geff"
    widget, tracks = _open(tracker, tmp_path, qtbot)
    UserDeleteNode(tracks, _node_of(tracker, 6))
    session = tracks.edit_session

    stamp = session.set_reviewed(True)
    assert stamp["revision"] == session.revision and stamp["by"] and stamp["at"]
    UserDeleteNode(tracks, _node_of(tracker, 5))  # an edit after the review keeps it
    session.flush()
    on_disk = json.loads((curated / "zarr.json").read_text())["attributes"]["geff"]
    assert on_disk["extra"]["reviewed"] == stamp
    assert on_disk["extra"]["tracker"] == "chain"
    assert experiment_layout.trench_state(tracker) == "reviewed"
    widget.close_sessions()

    with EditSession.open(curated) as reopened:
        assert reopened.review == stamp
        assert reopened.revision > stamp["revision"]
        assert reopened.set_reviewed(False) is None
    with EditSession.open(curated) as reopened:
        assert reopened.review is None
    assert experiment_layout.trench_state(tracker) == "edited"


def test_reviewing_an_unedited_trench_makes_the_curated_copy(
    experiment_store, tmp_path, qtbot
):
    trench = experiment_store / "trench_0001"
    tracker = trench / "tracks" / f"{RUN}.geff"
    before = _tracker_output(trench)
    widget, tracks = _open(tracker, tmp_path, qtbot)
    session = tracks.edit_session
    assert session.review is None
    assert session.set_reviewed(False) is None  # nothing to remove: nothing written
    assert not (trench / "curation").exists()

    stamp = session.set_reviewed(True)

    assert session.path == trench / "curation" / f"{RUN}.geff"
    assert session.review == stamp
    assert _tracker_output(trench) == before
    np.testing.assert_array_equal(
        zarr.open_array(trench / "labels" / f"curation.{RUN}" / "0", mode="r")[:],
        zarr.open_array(trench / "labels" / f"tracks.{RUN}" / "0", mode="r")[:],
    )
    widget.close_sessions()


def test_kymograph_panel_marks_reviewed_trenches(
    make_napari_viewer, experiment_store, qtbot, monkeypatch
):
    from qtpy.QtWidgets import QMessageBox

    from mm_track.application_menus.kymograph_data_widget import KymographDataWidget

    viewer = make_napari_viewer()
    widget = KymographDataWidget(viewer)
    qtbot.addWidget(widget)
    widget.set_store_path(experiment_store)
    assert widget.run == RUN
    assert "2 trenches" in widget.status_label.text()
    assert not widget.reviewed_checkbox.isEnabled()

    widget.load_selected()
    assert widget.reviewed_checkbox.isEnabled()
    assert "first edit makes the curated copy" in widget.review_label.text()
    widget.reviewed_checkbox.click()

    curated = experiment_store / "trench_0000" / "curation" / f"{RUN}.geff"
    assert widget.reviewed_checkbox.isChecked()
    assert widget.group_box.itemText(0) == "trench_0000 ✓"
    assert "Reviewed by" in widget.review_label.text()
    assert experiment_layout.review_of(
        json.loads((curated / "zarr.json").read_text())["attributes"]) is not None

    # an edit after the review outdates the mark; a click reviews the new revision
    session = widget._loaded.tracks.edit_session
    first = session.review
    UserDeleteNode(widget._loaded.tracks, _node_of(curated, 6))
    session.flush()
    widget._update_review()
    assert "edited since" in widget.review_label.text()
    assert widget.reviewed_checkbox.isChecked()
    widget.reviewed_checkbox.click()
    assert widget.reviewed_checkbox.isChecked()
    assert session.review["revision"] == session.revision > first["revision"]
    assert "edited since" not in widget.review_label.text()

    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.Yes)
    widget.reviewed_checkbox.click()
    assert not widget.reviewed_checkbox.isChecked()
    assert widget.group_box.itemText(0) == "trench_0000 ✎"
    _close_sessions(viewer)


def _close_sessions(viewer):
    from mm_track.data_views.views_coordinator.tracks_viewer import TracksViewer

    TracksViewer.get_instance(viewer).tracks_list.close_sessions()
