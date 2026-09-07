import numpy as np

from motile_tracker.data_views.views.kymograph_utils import (
    KymographGeometry,
    KymographLinkRenderData,
    build_kymograph_link_data,
    clamp_page_start,
    concat_time_to_kymograph,
    dashed_line_segments,
    default_page_length,
    frame_boundary_segments,
    kymograph_coords_to_indices,
    point_to_kymograph_coords,
    visible_frame_range,
)


def _red_by_track(track_id: int) -> np.ndarray:
    return np.asarray([track_id, 0, 0, 1], dtype=float)


def test_default_page_length_and_visible_frame_range():
    geometry = KymographGeometry(t_size=10, y_size=20, x_size=100)

    assert default_page_length(geometry, max_width=250) == 2
    assert clamp_page_start(9, geometry, page_length=4) == 6
    assert visible_frame_range(geometry, page_start=9, page_length=4) == (6, 10)


def test_concat_time_to_kymograph_respects_page_window():
    stack = np.arange(4 * 2 * 3).reshape(4, 2, 3)

    kymograph = concat_time_to_kymograph(stack, page_start=1, page_length=2)

    expected = np.concatenate([stack[1], stack[2]], axis=1)
    np.testing.assert_array_equal(kymograph, expected)
    assert kymograph.dtype == stack.dtype


def test_concat_time_to_kymograph_reads_only_the_page_of_lazy_arrays():
    """Array-likes that only support slicing along time (like the lazy segmentation of
    a Tracks object) must work, and only the requested frames may be read."""

    class LazyStack:
        def __init__(self, data):
            self._data = data
            self.shape = data.shape
            self.dtype = data.dtype
            self.requested = []

        def __getitem__(self, item):
            self.requested.append(item)
            return self._data[item]

    data = np.arange(4 * 2 * 3).reshape(4, 2, 3)
    stack = LazyStack(data)

    kymograph = concat_time_to_kymograph(stack, page_start=2, page_length=2)

    np.testing.assert_array_equal(kymograph, np.concatenate([data[2], data[3]], axis=1))
    assert stack.requested == [slice(2, 4)]


def test_concat_time_to_kymograph_from_tracks_segmentation(solution_tracks_2d):
    segmentation = np.asarray(solution_tracks_2d.segmentation)

    kymograph = concat_time_to_kymograph(
        solution_tracks_2d.segmentation, page_start=1, page_length=2
    )

    np.testing.assert_array_equal(
        kymograph, np.concatenate([segmentation[1], segmentation[2]], axis=1)
    )


def test_kymograph_coordinate_round_trip_with_scale():
    geometry = KymographGeometry(
        t_size=5,
        y_size=20,
        x_size=10,
        y_scale=2.0,
        x_scale=0.5,
    )

    coords = point_to_kymograph_coords(
        timepoint=3,
        position=(8.0, 4.5),
        geometry=geometry,
        page_start=2,
        page_length=2,
    )

    assert coords == (8.0, 9.5)
    assert kymograph_coords_to_indices(
        y_world=coords[0],
        x_global_world=coords[1],
        geometry=geometry,
        page_start=2,
        page_length=2,
    ) == (3, 4, 9)


def test_frame_boundary_segments_use_world_coordinates():
    geometry = KymographGeometry(
        t_size=5,
        y_size=20,
        x_size=10,
        y_scale=2.0,
        x_scale=0.5,
    )

    boundaries = frame_boundary_segments(
        geometry,
        page_start=1,
        page_length=3,
    )

    assert len(boundaries) == 2
    np.testing.assert_allclose(boundaries[0], np.asarray([[0.0, 5.0], [40.0, 5.0]]))
    np.testing.assert_allclose(boundaries[1], np.asarray([[0.0, 10.0], [40.0, 10.0]]))


def test_dashed_line_segments_keep_line_endpoints_ordered():
    segments = dashed_line_segments(
        (0.0, 0.0), (0.0, 20.0), dash_length=5.0, gap_length=5.0
    )

    assert len(segments) == 2
    np.testing.assert_allclose(
        segments[0], np.asarray([[0.0, 0.0], [0.0, 5.0]], dtype=np.float32)
    )
    np.testing.assert_allclose(
        segments[1], np.asarray([[0.0, 10.0], [0.0, 15.0]], dtype=np.float32)
    )


def test_build_kymograph_link_data_classifies_continuations_and_branches(
    solution_tracks_2d,
):
    geometry = KymographGeometry(t_size=5, y_size=100, x_size=100)

    continuation, branch = build_kymograph_link_data(
        tracks=solution_tracks_2d,
        geometry=geometry,
        page_start=0,
        page_length=5,
        track_color_resolver=_red_by_track,
    )

    assert isinstance(continuation, KymographLinkRenderData)
    assert isinstance(branch, KymographLinkRenderData)
    # 3->4 is the only edge between consecutive frames within one track; 4->5 skips
    # a frame and is therefore not drawn as a path
    assert len(continuation.segments) == 1
    assert list(continuation.properties["source_node"]) == [3]
    assert list(continuation.properties["target_node"]) == [4]
    assert list(continuation.properties["track_id"]) == [
        solution_tracks_2d.get_track_id(3)
    ]
    # the division 1->(2, 3) gives two dashed branch links
    assert set(branch.properties["source_node"]) == {1}
    assert set(branch.properties["target_node"]) == {2, 3}
    assert set(branch.properties["link_kind"]) == {"branch"}
    assert len(np.unique(branch.properties["logical_link_id"])) == 2


def test_build_kymograph_link_data_omits_off_page_edges(solution_tracks_2d):
    geometry = KymographGeometry(t_size=5, y_size=100, x_size=100)

    continuation, branch = build_kymograph_link_data(
        tracks=solution_tracks_2d,
        geometry=geometry,
        page_start=1,
        page_length=2,
        track_color_resolver=_red_by_track,
    )

    assert len(continuation.segments) == 1
    assert list(continuation.properties["source_node"]) == [3]
    assert list(continuation.properties["target_node"]) == [4]
    assert branch.segments == []


def test_build_kymograph_link_data_filters_by_visible_nodes(solution_tracks_2d):
    geometry = KymographGeometry(t_size=5, y_size=100, x_size=100)

    continuation, branch = build_kymograph_link_data(
        tracks=solution_tracks_2d,
        geometry=geometry,
        page_start=0,
        page_length=5,
        track_color_resolver=_red_by_track,
        visible_nodes=[1, 2],
    )

    assert continuation.segments == []
    assert set(branch.properties["target_node"]) == {2}
