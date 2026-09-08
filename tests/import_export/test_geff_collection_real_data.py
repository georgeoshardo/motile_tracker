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

    # the time slider has one step per frame and turns the page when needed
    time_slider = kymograph_layers.time_slider
    assert time_slider is not None
    assert time_slider.slider.maximum() == 720
    time_slider.slider.setValue(600)
    assert kymograph_layers.current_timepoint == 600
    assert kymograph_layers.node_on_page(
        next(n for n in tracks.graph.node_ids() if tracks.get_time(n) == 600)
    )

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


def test_merge_then_split_two_real_cells():
    """Merging two neighbouring cells of a trench and splitting the merged mask
    again recovers both cells and their links, using the previous frame as guide."""
    from motile_tracker.data_views.views_coordinator.node_edits import (
        MergeNodes,
        SplitNode,
        node_mask,
        propose_split,
    )
    from motile_tracker.import_export.geff_collection import load_geff_group

    loaded = load_geff_group(STORE / "trench_0165" / "tracking_graph.geff")
    tracks, image = loaded.tracks, loaded.image
    t = 300
    frame = np.asarray(tracks.segmentation[t])
    labels = [int(v) for v in np.unique(frame) if v != 0]
    # two vertically adjacent cells that both continue a track and have a successor
    tops = {n: np.nonzero(frame == n)[0].min() for n in labels}
    ordered = sorted(labels, key=tops.get)
    pair = next(
        (a, b)
        for a, b in zip(ordered, ordered[1:], strict=False)
        if len(tracks.predecessors(a)) == 1
        and len(tracks.predecessors(b)) == 1
        and len(tracks.successors(a)) == 1
        and len(tracks.successors(b)) == 1
    )
    a, b = pair
    parent_a, parent_b = int(tracks.predecessors(a)[0]), int(tracks.predecessors(b)[0])
    child_a, child_b = int(tracks.successors(a)[0]), int(tracks.successors(b)[0])
    mask_a, mask_b = node_mask(tracks, a), node_mask(tracks, b)

    merged = MergeNodes(tracks, [a, b])
    kept = merged.kept_node
    other = a if kept == b else b
    assert not tracks.graph.has_node(other)
    assert np.array_equal(node_mask(tracks, kept), mask_a | mask_b)

    plan = propose_split(tracks, kept, image=np.asarray(image[t], dtype=float))
    assert plan is not None
    # the other cell's parent had lost its child, so the previous frame guided the
    # split (directly, or by confirming the image cut)
    assert plan.method in {"guides", "watershed"}
    split = SplitNode(tracks, plan)
    new = split.new_node

    def iou(x, y):
        return np.count_nonzero(x & y) / np.count_nonzero(x | y)

    part_kept, part_new = node_mask(tracks, kept), node_mask(tracks, new)
    kept_is_a = kept == a
    original_kept, original_new = (mask_a, mask_b) if kept_is_a else (mask_b, mask_a)
    assert iou(part_kept, original_kept) > 0.9
    assert iou(part_new, original_new) > 0.9
    # links: each cell continues its own parent and keeps its own child
    parent_kept, parent_new = (
        (parent_a, parent_b) if kept_is_a else (parent_b, parent_a)
    )
    child_kept, child_new = (child_a, child_b) if kept_is_a else (child_b, child_a)
    assert [int(p) for p in tracks.predecessors(kept)] == [parent_kept]
    assert [int(p) for p in tracks.predecessors(new)] == [parent_new]
    assert [int(c) for c in tracks.successors(kept)] == [child_kept]
    assert [int(c) for c in tracks.successors(new)] == [child_new]

    tracks.undo()  # the split
    tracks.undo()  # the merge
    assert tracks.graph.has_node(a) and tracks.graph.has_node(b)
    assert np.array_equal(node_mask(tracks, a), mask_a)
    assert np.array_equal(node_mask(tracks, b), mask_b)
    assert [int(c) for c in tracks.successors(parent_b)] == [b]


def test_split_quality_on_merged_neighbours():
    """The split cascade recovers artificially merged neighbouring cells: over a
    sample of touching pairs from two trenches, the mean IoU with the true cells
    stays high, with the previous frame as guide and the raw image as evidence."""
    from motile_tracker.data_views.views_coordinator.mask_split import (
        bright_cells_in_frame,
        split_mask,
    )
    from motile_tracker.import_export.geff_collection import load_geff_group

    def iou(x, y):
        return np.count_nonzero(x & y) / np.count_nonzero(x | y)

    scores = []
    for trench in ("trench_0165", "trench_0279"):
        loaded = load_geff_group(STORE / trench / "tracking_graph.geff")
        tracks, image = loaded.tracks, loaded.image
        for t in range(40, 700, 30):
            frame = np.asarray(tracks.segmentation[t])
            previous = np.asarray(tracks.segmentation[t - 1])
            labels = [int(v) for v in np.unique(frame) if v != 0]
            extents = {n: np.nonzero(frame == n)[0] for n in labels}
            ordered = sorted(labels, key=lambda n: extents[n].min())
            bright = bright_cells_in_frame(image[t], frame)
            for a, b in zip(ordered, ordered[1:], strict=False):
                if extents[b].min() - extents[a].max() > 1:  # not touching
                    continue
                if len(tracks.predecessors(a)) != 1 or len(tracks.predecessors(b)) != 1:
                    continue
                mask_a, mask_b = frame == a, frame == b
                guides = (
                    previous == int(tracks.predecessors(a)[0]),
                    previous == int(tracks.predecessors(b)[0]),
                )
                result = split_mask(
                    mask_a | mask_b, image=image[t], guides=guides, bright_cells=bright
                )
                assert result is not None
                part_a, part_b = result.parts
                straight = (iou(part_a, mask_a) + iou(part_b, mask_b)) / 2
                swapped = (iou(part_a, mask_b) + iou(part_b, mask_a)) / 2
                scores.append(max(straight, swapped))

    scores = np.asarray(scores)
    assert len(scores) >= 50
    assert scores.mean() > 0.95
    assert np.mean(scores >= 0.8) > 0.95
