"""Smoke test on the real mother machine dataset, if it is available locally.

The dataset (``trackastra_selected_graph_trenches.zarr``, 44 trenches of 721
frames each) is not part of the repository; point MOTILE_TRACKER_TRENCH_ZARR at
it or keep it in the repository root to run these tests.
"""

import os
from pathlib import Path

import numpy as np
import pytest

from motile_tracker.data_views.views_coordinator.tracks_viewer import TracksViewer
from motile_tracker.import_export.geff_collection import (
    add_geff_group_to_viewer,
    find_geff_groups,
)

STORE = Path(
    os.environ.get(
        "MOTILE_TRACKER_TRENCH_ZARR",
        Path(__file__).resolve().parents[2] / "trackastra_selected_graph_trenches.zarr",
    )
)

pytestmark = pytest.mark.skipif(
    not STORE.is_dir(), reason=f"trench dataset not available at {STORE}"
)


class _PaintEvent:
    def __init__(self, value):
        self.value = value


def test_find_all_trenches():
    groups = find_geff_groups(STORE)

    trench_dirs = sorted(
        p.name for p in STORE.iterdir() if p.name.startswith("trench_")
    )
    assert len(trench_dirs) > 0
    assert [group.name for group in groups] == trench_dirs
    assert groups[0].name == "trench_0165"
    assert groups[0].path == STORE / "trench_0165" / "tracking_graph.geff"


def test_trench_loads_into_kymograph_and_is_editable(make_napari_viewer):
    viewer = make_napari_viewer()
    group = find_geff_groups(STORE)[0]

    loaded = add_geff_group_to_viewer(viewer, group.path, group.name)

    tracks_viewer = TracksViewer.get_instance(viewer)
    tracks = tracks_viewer.tracks
    assert tracks.graph.num_nodes() == 3309
    assert tracks.graph.num_edges() == 3304
    assert tracks.segmentation.shape == (721, 164, 34)
    assert loaded.image.shape == (721, 164, 34)
    assert tracks_viewer.view_mode == "kymograph"
    kymograph_layers = tracks_viewer.kymograph_layers
    assert kymograph_layers.geometry.t_size == 721
    assert kymograph_layers.background_layer is not None

    # page through the movie
    tracks_viewer.set_kymograph_page_length(120)
    tracks_viewer.set_kymograph_page_start(300)
    assert kymograph_layers.page_start == 300
    assert kymograph_layers.labels_layer.data.shape == (164, 120 * 34)

    # break and restore a link
    on_page = [
        (int(s), int(t))
        for s, t in tracks.graph.edge_list()
        if kymograph_layers.node_on_page(s) and kymograph_layers.node_on_page(t)
    ]
    source, target = on_page[len(on_page) // 2]
    tracks_viewer.selected_nodes.add(source)
    tracks_viewer.selected_nodes.add(target, append=True)
    tracks_viewer.delete_edge()
    assert not tracks.graph.has_edge(source, target)
    tracks_viewer.undo()
    assert tracks.graph.has_edge(source, target)
    tracks_viewer.selected_nodes.reset()

    # paint a new cell into an empty corner of the third frame on the page
    labels_layer = kymograph_layers.labels_layer
    tracks_viewer.request_new_track()
    new_label = int(labels_layer.selected_label)
    timepoint = kymograph_layers.page_start + 2
    x0 = 2 * kymograph_layers.geometry.x_size
    yy, xx = np.meshgrid(np.arange(2, 6), np.arange(x0 + 1, x0 + 5), indexing="ij")
    old_values = labels_layer.data[yy.ravel(), xx.ravel()]
    assert not old_values.any()
    labels_layer.mode = "paint"
    labels_layer._on_paint(_PaintEvent([((yy.ravel(), xx.ravel()), old_values, 99)]))

    assert tracks.graph.has_node(new_label)
    assert tracks.get_time(new_label) == timepoint
    assert int(np.asarray(tracks.segmentation[timepoint])[3, 2]) == new_label
    assert int(labels_layer.data[3, x0 + 2]) == new_label

    tracks_viewer.undo()
    assert not tracks.graph.has_node(new_label)
    assert int(labels_layer.data[3, x0 + 2]) == 0
