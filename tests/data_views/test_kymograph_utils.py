import numpy as np

from motile_tracker.data_views.views.kymograph_utils import (
    KymographLinkRenderData,
    KymographGeometry,
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
    segments = dashed_line_segments((0.0, 0.0), (0.0, 20.0), dash_length=5.0, gap_length=5.0)

    assert len(segments) == 2
    np.testing.assert_allclose(segments[0], np.asarray([[0.0, 0.0], [0.0, 5.0]], dtype=np.float32))
    np.testing.assert_allclose(segments[1], np.asarray([[0.0, 10.0], [0.0, 15.0]], dtype=np.float32))


def test_build_kymograph_link_data_classifies_continuations_and_branches(graph_2d):
    geometry = KymographGeometry(t_size=5, y_size=100, x_size=100)

    continuation, branch = build_kymograph_link_data(
        tracks=type("Tracks", (), {
            "graph": graph_2d,
            "get_time": lambda self, node: graph_2d.nodes[node]["time"],
            "get_position": lambda self, node: graph_2d.nodes[node]["pos"],
            "get_track_id": lambda self, node: graph_2d.nodes[node]["track_id"],
        })(),
        geometry=geometry,
        page_start=0,
        page_length=5,
        track_color_resolver=lambda track_id: np.asarray([track_id, 0, 0, 1], dtype=float),
    )

    assert isinstance(continuation, KymographLinkRenderData)
    assert isinstance(branch, KymographLinkRenderData)
    assert len(continuation.segments) == 1
    assert list(continuation.properties["source_node"]) == [3]
    assert list(continuation.properties["target_node"]) == [4]
    assert list(continuation.properties["track_id"]) == [3]
    assert set(branch.properties["source_node"]) == {1}
    assert set(branch.properties["target_node"]) == {2, 3}
    assert set(branch.properties["link_kind"]) == {"branch"}
    assert len(np.unique(branch.properties["logical_link_id"])) == 2


def test_build_kymograph_link_data_omits_off_page_edges(graph_2d):
    geometry = KymographGeometry(t_size=5, y_size=100, x_size=100)
    tracks = type("Tracks", (), {
        "graph": graph_2d,
        "get_time": lambda self, node: graph_2d.nodes[node]["time"],
        "get_position": lambda self, node: graph_2d.nodes[node]["pos"],
        "get_track_id": lambda self, node: graph_2d.nodes[node]["track_id"],
    })()

    continuation, branch = build_kymograph_link_data(
        tracks=tracks,
        geometry=geometry,
        page_start=1,
        page_length=2,
        track_color_resolver=lambda track_id: np.asarray([track_id, 0, 0, 1], dtype=float),
    )

    assert len(continuation.segments) == 1
    assert list(continuation.properties["source_node"]) == [3]
    assert list(continuation.properties["target_node"]) == [4]
    assert branch.segments == []


def test_build_kymograph_link_data_filters_by_visible_nodes(graph_2d):
    geometry = KymographGeometry(t_size=5, y_size=100, x_size=100)
    tracks = type("Tracks", (), {
        "graph": graph_2d,
        "get_time": lambda self, node: graph_2d.nodes[node]["time"],
        "get_position": lambda self, node: graph_2d.nodes[node]["pos"],
        "get_track_id": lambda self, node: graph_2d.nodes[node]["track_id"],
    })()

    continuation, branch = build_kymograph_link_data(
        tracks=tracks,
        geometry=geometry,
        page_start=0,
        page_length=5,
        track_color_resolver=lambda track_id: np.asarray([track_id, 0, 0, 1], dtype=float),
        visible_nodes=[1, 2],
    )

    assert continuation.segments == []
    assert set(branch.properties["target_node"]) == {2}
