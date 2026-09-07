from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

DEFAULT_MAX_PAGE_WIDTH = 16_384
BRANCH_DASH_LENGTH = 6.0
BRANCH_GAP_LENGTH = 4.0


@dataclass(frozen=True)
class KymographGeometry:
    t_size: int
    y_size: int
    x_size: int
    y_scale: float = 1.0
    x_scale: float = 1.0

    @property
    def frame_world_width(self) -> float:
        return float(self.x_size) * float(self.x_scale)

    @property
    def kymograph_width(self) -> int:
        return int(self.t_size) * int(self.x_size)


@dataclass(frozen=True)
class KymographLinkRenderData:
    segments: list[np.ndarray]
    properties: dict[str, np.ndarray]
    edge_colors: list[np.ndarray]

    @classmethod
    def empty(cls) -> KymographLinkRenderData:
        return cls(
            segments=[],
            properties={
                "source_node": np.asarray([], dtype=int),
                "target_node": np.asarray([], dtype=int),
                "track_id": np.asarray([], dtype=int),
                "link_kind": np.asarray([], dtype=object),
                "logical_link_id": np.asarray([], dtype=int),
            },
            edge_colors=[],
        )


def infer_kymograph_geometry(
    shape: tuple[int, int, int],
    scale: list[float] | tuple[float, ...] | None = None,
) -> KymographGeometry:
    if len(shape) != 3:
        raise ValueError(f"Expected (T, Y, X) data, got shape {shape}.")

    t_size, y_size, x_size = shape
    if scale is None:
        scale = (1.0, 1.0, 1.0)
    if len(scale) < 3:
        raise ValueError(f"Expected scale with at least 3 values, got {scale}.")

    return KymographGeometry(
        t_size=int(t_size),
        y_size=int(y_size),
        x_size=int(x_size),
        y_scale=float(scale[-2]),
        x_scale=float(scale[-1]),
    )


def default_page_length(
    geometry: KymographGeometry,
    max_width: int = DEFAULT_MAX_PAGE_WIDTH,
) -> int:
    return max(1, min(geometry.t_size, max_width // max(int(geometry.x_size), 1)))


def clamp_page_start(
    page_start: int,
    geometry: KymographGeometry,
    page_length: int,
) -> int:
    max_start = max(0, int(geometry.t_size) - int(page_length))
    return int(np.clip(int(page_start), 0, max_start))


def visible_frame_range(
    geometry: KymographGeometry,
    page_start: int,
    page_length: int,
) -> tuple[int, int]:
    start = clamp_page_start(page_start, geometry, page_length)
    stop = min(int(geometry.t_size), start + int(page_length))
    return start, stop


def concat_time_to_kymograph(
    stack,
    *,
    page_start: int = 0,
    page_length: int | None = None,
) -> np.ndarray:
    """Place the frames of one page of a (T, Y, X) stack side by side along X.

    Only the frames on the requested page are read, so `stack` may be any array-like
    that supports `shape` and slicing along the first axis (numpy, dask, zarr, or the
    lazy segmentation array of a funtracks Tracks object).
    """
    shape = tuple(int(v) for v in stack.shape)
    if len(shape) != 3:
        raise ValueError(f"Expected (T, Y, X) stack, got {shape}.")

    geometry = infer_kymograph_geometry(shape)
    if page_length is None:
        page_length = geometry.t_size
    start, stop = visible_frame_range(geometry, page_start, page_length)
    if stop <= start:
        return np.zeros((geometry.y_size, 0), dtype=getattr(stack, "dtype", float))

    try:
        page = np.asarray(stack[start:stop])
    except (TypeError, NotImplementedError):
        # array-likes without slicing support: materialize, then slice
        page = np.asarray(stack)[start:stop]
    # (n_frames, Y, X) -> (Y, n_frames * X)
    return np.ascontiguousarray(page.transpose(1, 0, 2).reshape(geometry.y_size, -1))


def point_to_kymograph_coords(
    *,
    timepoint: int,
    position: list[float] | tuple[float, float] | np.ndarray,
    geometry: KymographGeometry,
    page_start: int,
    page_length: int,
) -> tuple[float, float] | None:
    start, stop = visible_frame_range(geometry, page_start, page_length)
    if timepoint < start or timepoint >= stop:
        return None

    y_world = float(position[-2])
    x_world = float(position[-1])
    x_global = (timepoint - start) * geometry.frame_world_width + x_world
    return y_world, x_global


def kymograph_coords_to_indices(
    *,
    y_world: float,
    x_global_world: float,
    geometry: KymographGeometry,
    page_start: int,
    page_length: int,
) -> tuple[int, int, int] | None:
    start, stop = visible_frame_range(geometry, page_start, page_length)
    if y_world < 0:
        return None

    y_idx = int(np.floor(y_world / max(geometry.y_scale, 1e-12)))
    if y_idx < 0 or y_idx >= geometry.y_size:
        return None

    if x_global_world < 0:
        return None

    frame_offset = int(
        np.floor(x_global_world / max(geometry.frame_world_width, 1e-12))
    )
    timepoint = start + frame_offset
    if timepoint < start or timepoint >= stop:
        return None

    x_local_world = x_global_world - frame_offset * geometry.frame_world_width
    x_idx = int(np.floor(x_local_world / max(geometry.x_scale, 1e-12)))
    if x_idx < 0 or x_idx >= geometry.x_size:
        return None

    return int(timepoint), int(y_idx), int(x_idx)


def frame_boundary_segments(
    geometry: KymographGeometry,
    *,
    page_start: int,
    page_length: int,
) -> list[np.ndarray]:
    start, stop = visible_frame_range(geometry, page_start, page_length)
    page_frames = max(0, stop - start)
    y_max = float(geometry.y_size) * float(geometry.y_scale)

    segments: list[np.ndarray] = []
    for frame_offset in range(1, page_frames):
        x = frame_offset * geometry.frame_world_width
        segments.append(
            np.asarray(
                [[0.0, x], [y_max, x]],
                dtype=np.float32,
            )
        )
    return segments


def dashed_line_segments(
    source: tuple[float, float],
    target: tuple[float, float],
    *,
    dash_length: float = BRANCH_DASH_LENGTH,
    gap_length: float = BRANCH_GAP_LENGTH,
) -> list[np.ndarray]:
    source_arr = np.asarray(source, dtype=np.float32)
    target_arr = np.asarray(target, dtype=np.float32)
    vector = target_arr - source_arr
    length = float(np.linalg.norm(vector))
    if length <= 1e-6:
        return [np.asarray([source_arr, target_arr], dtype=np.float32)]

    direction = vector / length
    segments: list[np.ndarray] = []
    start = 0.0
    stride = max(float(dash_length) + float(gap_length), 1e-6)
    while start < length:
        end = min(start + float(dash_length), length)
        segments.append(
            np.asarray(
                [
                    source_arr + direction * start,
                    source_arr + direction * end,
                ],
                dtype=np.float32,
            )
        )
        start += stride

    return segments


def build_kymograph_link_data(
    *,
    tracks,
    geometry: KymographGeometry,
    page_start: int,
    page_length: int,
    track_color_resolver: Callable[[int], np.ndarray],
    visible_nodes: list[int] | str = "all",
    branch_dash_length: float = BRANCH_DASH_LENGTH,
    branch_gap_length: float = BRANCH_GAP_LENGTH,
) -> tuple[KymographLinkRenderData, KymographLinkRenderData]:
    visible_nodes_set = None
    if not isinstance(visible_nodes, str):
        visible_nodes_set = {int(node) for node in visible_nodes}

    continuation_segments: list[np.ndarray] = []
    continuation_edge_colors: list[np.ndarray] = []
    continuation_properties = {
        "source_node": [],
        "target_node": [],
        "track_id": [],
        "link_kind": [],
        "logical_link_id": [],
    }

    branch_segments: list[np.ndarray] = []
    branch_edge_colors: list[np.ndarray] = []
    branch_properties = {
        "source_node": [],
        "target_node": [],
        "track_id": [],
        "link_kind": [],
        "logical_link_id": [],
    }

    edges = [(int(source), int(target)) for source, target in tracks.graph.edge_list()]
    if visible_nodes_set is not None:
        edges = [
            (source, target)
            for source, target in edges
            if source in visible_nodes_set and target in visible_nodes_set
        ]
    if not edges:
        return KymographLinkRenderData.empty(), KymographLinkRenderData.empty()

    # fetch the node data in bulk: per-node lookups are slow on large graphs
    nodes = sorted({node for edge in edges for node in edge})
    times = dict(zip(nodes, (int(t) for t in tracks.get_times(nodes)), strict=True))
    track_ids = dict(
        zip(nodes, (int(t) for t in tracks.get_track_ids(nodes)), strict=True)
    )
    positions = dict(zip(nodes, np.asarray(tracks.get_positions(nodes)), strict=True))

    logical_link_id = 0
    for source_node, target_node in edges:
        source_coords = point_to_kymograph_coords(
            timepoint=times[source_node],
            position=positions[source_node],
            geometry=geometry,
            page_start=page_start,
            page_length=page_length,
        )
        target_coords = point_to_kymograph_coords(
            timepoint=times[target_node],
            position=positions[target_node],
            geometry=geometry,
            page_start=page_start,
            page_length=page_length,
        )
        if source_coords is None or target_coords is None:
            continue

        source_time = times[source_node]
        target_time = times[target_node]
        source_track_id = track_ids[source_node]
        target_track_id = track_ids[target_node]
        logical_link_id += 1

        is_branch = source_track_id != target_track_id

        if not is_branch:
            # Same track: a solid line between consecutive frames. An edge that
            # skips frames (a missed detection, or a link the user drew across a
            # gap) is drawn dashed in the track colour so the skip is visible,
            # spanning the frames in between.
            track_color = np.asarray(
                track_color_resolver(source_track_id), dtype=np.float32
            )
            if target_time == source_time + 1:
                link_kind = "continuation"
                fragments = [
                    np.asarray([source_coords, target_coords], dtype=np.float32)
                ]
            else:
                link_kind = "gap"
                fragments = dashed_line_segments(
                    source_coords,
                    target_coords,
                    dash_length=branch_dash_length,
                    gap_length=branch_gap_length,
                )
            for fragment in fragments:
                continuation_segments.append(fragment)
                continuation_edge_colors.append(track_color)
                continuation_properties["source_node"].append(int(source_node))
                continuation_properties["target_node"].append(int(target_node))
                continuation_properties["track_id"].append(source_track_id)
                continuation_properties["link_kind"].append(link_kind)
                continuation_properties["logical_link_id"].append(logical_link_id)
        else:
            fragments = dashed_line_segments(
                source_coords,
                target_coords,
                dash_length=branch_dash_length,
                gap_length=branch_gap_length,
            )
            for fragment in fragments:
                branch_segments.append(fragment)
                branch_edge_colors.append(
                    np.asarray([0.0, 1.0, 0.0, 1.0], dtype=np.float32)
                )
                branch_properties["source_node"].append(int(source_node))
                branch_properties["target_node"].append(int(target_node))
                branch_properties["track_id"].append(target_track_id)
                branch_properties["link_kind"].append("branch")
                branch_properties["logical_link_id"].append(logical_link_id)

    def _finalize(
        segments: list[np.ndarray],
        properties: dict[str, list],
        edge_colors: list[np.ndarray],
    ) -> KymographLinkRenderData:
        if not segments:
            return KymographLinkRenderData.empty()

        return KymographLinkRenderData(
            segments=segments,
            properties={
                "source_node": np.asarray(properties["source_node"], dtype=int),
                "target_node": np.asarray(properties["target_node"], dtype=int),
                "track_id": np.asarray(properties["track_id"], dtype=int),
                "link_kind": np.asarray(properties["link_kind"], dtype=object),
                "logical_link_id": np.asarray(properties["logical_link_id"], dtype=int),
            },
            edge_colors=edge_colors,
        )

    return (
        _finalize(
            continuation_segments,
            continuation_properties,
            continuation_edge_colors,
        ),
        _finalize(branch_segments, branch_properties, branch_edge_colors),
    )
