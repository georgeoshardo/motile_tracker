from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEFAULT_MAX_PAGE_WIDTH = 16_384


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
    stack: np.ndarray,
    *,
    page_start: int = 0,
    page_length: int | None = None,
) -> np.ndarray:
    if stack.ndim != 3:
        raise ValueError(f"Expected (T, Y, X) stack, got {stack.shape}.")

    geometry = infer_kymograph_geometry(tuple(int(v) for v in stack.shape))
    if page_length is None:
        page_length = geometry.t_size
    start, stop = visible_frame_range(geometry, page_start, page_length)
    frames = [stack[t] for t in range(start, stop)]
    if not frames:
        return np.zeros((geometry.y_size, 0), dtype=stack.dtype)
    return np.concatenate(frames, axis=1)


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

    frame_offset = int(np.floor(x_global_world / max(geometry.frame_world_width, 1e-12)))
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
