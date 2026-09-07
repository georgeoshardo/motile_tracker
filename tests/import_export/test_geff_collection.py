"""Tests for loading GEFF groups together with their related image/label arrays."""

import logging

import networkx as nx
import numpy as np
from funtracks.import_export import import_from_geff
from geff import GeffMetadata
from geff import write as write_geff

from motile_tracker.data_views.views_coordinator.tracks_viewer import TracksViewer
from motile_tracker.import_export.geff_collection import (
    GeffGroup,
    add_geff_group_to_viewer,
    find_geff_groups,
    is_geff_dir,
    load_geff_group,
    node_name_map_from_metadata,
    read_related_arrays,
)
from motile_tracker.import_export.geff_io import write_geff_over


def test_find_geff_groups_names_groups_after_their_parent(geff_collection_store):
    groups = find_geff_groups(geff_collection_store)

    assert [group.name for group in groups] == ["trench_a", "trench_b"]
    assert groups[0].path == geff_collection_store / "trench_a" / "tracking_graph.geff"
    assert all(is_geff_dir(group.path) for group in groups)


def test_find_geff_groups_does_not_descend_into_arrays_or_geffs(
    geff_collection_store,
):
    """Only the geff groups are reported: not their nodes/edges subgroups, and not
    the image/segmentation arrays."""
    groups = find_geff_groups(geff_collection_store)

    assert len(groups) == 2
    assert not any("nodes" in group.path.parts for group in groups)


def test_find_geff_groups_on_a_store_that_is_a_geff(tmp_path):
    graph = nx.DiGraph()
    graph.add_node(1, time=0, y=1.0, x=2.0)
    geff_path = tmp_path / "my_tracks.geff"
    write_geff(
        graph,
        geff_path,
        axis_names=["time", "y", "x"],
        axis_types=["time", "space", "space"],
    )

    assert find_geff_groups(geff_path) == [GeffGroup("my_tracks", geff_path)]


def test_find_geff_groups_on_a_store_without_geffs(tmp_path):
    (tmp_path / "empty").mkdir()

    assert find_geff_groups(tmp_path) == []


def test_read_related_arrays_resolves_paths_relative_to_the_geff(
    geff_collection_store,
):
    geff_path = geff_collection_store / "trench_a" / "tracking_graph.geff"

    related = read_related_arrays(geff_path)

    assert related.labels_path == geff_collection_store / "trench_a" / "segmentation"
    assert related.label_prop == "seg_id"
    assert related.image_path == geff_collection_store / "trench_a" / "images"


def test_read_related_arrays_ignores_missing_objects(geff_collection_store, caplog):
    geff_path = geff_collection_store / "trench_a" / "tracking_graph.geff"
    metadata = GeffMetadata.read(geff_path)
    metadata.related_objects[1].path = "../gone_images"

    with caplog.at_level(logging.WARNING):
        related = read_related_arrays(geff_path, metadata)

    assert related.image_path is None
    assert related.labels_path is not None
    assert "gone_images" in caplog.text


def test_node_name_map_from_metadata(geff_collection_store):
    metadata = GeffMetadata.read(
        geff_collection_store / "trench_a" / "tracking_graph.geff"
    )

    name_map = node_name_map_from_metadata(metadata, label_prop="seg_id")

    assert name_map == {
        "time": "time",
        "pos": ["y", "x"],
        "track_id": "tracklet",
        "lineage_id": "lineage",
        "seg_id": "seg_id",
    }


def test_node_name_map_skips_properties_the_geff_does_not_have(
    geff_collection_store,
):
    metadata = GeffMetadata.read(
        geff_collection_store / "trench_a" / "tracking_graph.geff"
    )
    metadata.track_node_props = {"tracklet": "not_a_prop"}

    name_map = node_name_map_from_metadata(metadata, label_prop="also_missing")

    assert "track_id" not in name_map
    assert "seg_id" not in name_map
    assert name_map["time"] == "time"


def test_load_geff_group(geff_collection_store, segmentation_2d):
    geff_path = geff_collection_store / "trench_a" / "tracking_graph.geff"

    loaded = load_geff_group(geff_path, name="trench_a")

    tracks = loaded.tracks
    assert loaded.name == "trench_a"
    assert tracks.ndim == 3
    assert tracks.graph.num_nodes() == 6
    assert tracks.graph.num_edges() == 4
    assert tracks.segmentation.shape == segmentation_2d.shape
    # labels equal node ids, so the label under a node's position is the node
    assert int(np.asarray(tracks.segmentation[1])[60, 45]) == 3
    # the tracklet and lineage properties were taken over from the geff
    assert tracks.get_track_id(4) == tracks.get_track_id(5)
    assert tracks.get_lineage_id(6) != tracks.get_lineage_id(1)
    # the image was read into memory
    assert isinstance(loaded.image, np.ndarray)
    assert loaded.image.shape == segmentation_2d.shape
    assert loaded.image_path == geff_collection_store / "trench_a" / "images"


def test_add_geff_group_to_viewer_opens_the_kymograph(
    make_napari_viewer, geff_collection_store
):
    viewer = make_napari_viewer()
    geff_path = geff_collection_store / "trench_a" / "tracking_graph.geff"

    loaded = add_geff_group_to_viewer(viewer, geff_path, "trench_a")

    tracks_viewer = TracksViewer.get_instance(viewer)
    assert tracks_viewer.tracks is not None
    assert tracks_viewer.tracks.graph.num_nodes() == 6
    assert tracks_viewer.tracks_list.tracks_list.count() == 1
    assert tracks_viewer.view_mode == "kymograph"
    kymograph_layers = tracks_viewer.kymograph_layers
    assert kymograph_layers.labels_layer is not None
    assert kymograph_layers.points_layer is not None
    # the trench image is shown behind the kymograph
    assert loaded.image_layer is not None
    assert kymograph_layers.detached_image_layer is loaded.image_layer
    assert kymograph_layers.background_layer is not None
    assert kymograph_layers.background_layer.data.shape == (100, 500)


def test_add_geff_group_to_viewer_spatial(make_napari_viewer, geff_collection_store):
    viewer = make_napari_viewer()
    geff_path = geff_collection_store / "trench_a" / "tracking_graph.geff"

    loaded = add_geff_group_to_viewer(viewer, geff_path, "trench_a", kymograph=False)

    tracks_viewer = TracksViewer.get_instance(viewer)
    assert tracks_viewer.view_mode == "spatial"
    assert loaded.image_layer in viewer.layers
    assert tracks_viewer.tracking_layers.seg_layer in viewer.layers
    # the image is already selected for when the user switches to the kymograph
    assert tracks_viewer.kymograph_layers.image_layer_name == "trench_a_images"


def test_loading_a_second_group_in_kymograph_mode_uses_its_own_image(
    make_napari_viewer, geff_collection_store
):
    viewer = make_napari_viewer()
    add_geff_group_to_viewer(
        viewer, geff_collection_store / "trench_a" / "tracking_graph.geff", "trench_a"
    )
    tracks_viewer = TracksViewer.get_instance(viewer)
    assert tracks_viewer.view_mode == "kymograph"

    loaded_b = add_geff_group_to_viewer(
        viewer, geff_collection_store / "trench_b" / "tracking_graph.geff", "trench_b"
    )

    assert tracks_viewer.view_mode == "kymograph"
    assert tracks_viewer.kymograph_layers.detached_image_layer is loaded_b.image_layer
    assert tracks_viewer.kymograph_layers.labels_layer.name == "trench_b_kymograph_seg"
    assert tracks_viewer.tracks_list.tracks_list.count() == 2


def test_edits_in_kymograph_mode_are_saved_from_the_tracks_list(
    make_napari_viewer, geff_collection_store, tmp_path
):
    """The viewer works on a SolutionTracks view of the listed tracks, but both share
    the underlying graph, so saving the listed tracks (what the Tracks List's save
    button does) writes the edits made in the kymograph view."""
    viewer = make_napari_viewer()
    add_geff_group_to_viewer(
        viewer, geff_collection_store / "trench_a" / "tracking_graph.geff", "trench_a"
    )
    tracks_viewer = TracksViewer.get_instance(viewer)
    tracks_list = tracks_viewer.tracks_list.tracks_list
    listed = tracks_list.itemWidget(tracks_list.item(0)).tracks

    tracks_viewer.selected_nodes.add(3)
    tracks_viewer.selected_nodes.add(4, append=True)
    tracks_viewer.delete_edge()
    assert not tracks_viewer.tracks.graph.has_edge(3, 4)

    saved = tmp_path / "trench_a_edited.geff"
    write_geff_over(listed, saved)
    reloaded = import_from_geff(saved)

    assert reloaded.graph.num_edges() == 3
    assert not reloaded.graph.has_edge(3, 4)
    assert reloaded.graph.has_edge(1, 3)
