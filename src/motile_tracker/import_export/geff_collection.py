"""Load GEFF tracks together with the image and label arrays they reference.

Mother-machine style datasets (and any GEFF written with ``related_objects``)
keep one group per field of view: a GEFF graph next to the ``images`` and
``segmentation`` arrays it describes, for example::

    trenches.zarr/
        trench_0165/
            images            (T, Y, X) raw image stack
            segmentation      (T, Y, X) labels, label value == GEFF ``seg_id``
            tracking_graph.geff
        trench_0187/
            ...

The import dialog asks the user to point at each of these pieces by hand. The
helpers here read the GEFF metadata instead (``related_objects`` for the arrays,
``track_node_props`` and the typed ``axes`` for the property names), so a whole
store of such groups can be listed and any group loaded into the viewer in one
step, straight into the kymograph view.
"""

from __future__ import annotations

import json
import logging
import os
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import zarr
from funtracks.import_export import import_from_geff
from geff import GeffMetadata

if TYPE_CHECKING:
    from collections.abc import Iterator

    import napari
    import numpy as np
    from funtracks.data_model import Tracks

logger = logging.getLogger(__name__)

# Arrays up to this size are read into memory, which makes paging through the
# kymograph instant. Larger arrays are left as (lazy) zarr arrays.
MAX_IN_MEMORY_BYTES = 512 * 1024**2


@dataclass(frozen=True)
class GeffGroup:
    """A GEFF graph found inside a zarr store.

    Attributes:
        name (str): A short name for the group, derived from its location in the
            store (the parent group for a geff such as ``trench_0165/tracking_graph.geff``).
        path (Path): The filesystem path of the geff group itself.
    """

    name: str
    path: Path


@dataclass(frozen=True)
class RelatedArrays:
    """The arrays a GEFF refers to via its ``related_objects`` metadata."""

    labels_path: Path | None = None
    label_prop: str | None = None
    image_path: Path | None = None


@dataclass
class LoadedGeff:
    """A GEFF group loaded with its related arrays."""

    name: str
    geff_path: Path
    tracks: Tracks
    image: np.ndarray | zarr.Array | None = None
    image_path: Path | None = None
    image_layer: napari.layers.Image | None = None


@contextmanager
def silence_geff_zarr_warnings() -> Iterator[None]:
    """Silence the warnings zarr and geff emit for stores that hold non-geff members.

    A store with images next to the graph, or a zarr v2 geff inside a zarr v3
    store, is exactly what this module is for, so the warnings say nothing the
    user can act on (see also ``geff_io.write_geff_over``).
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Object at .* is not recognized as a component of a Zarr hierarchy",
            category=UserWarning,
        )
        warnings.filterwarnings(
            "ignore", message="Found non-geff members in zarr.*", category=UserWarning
        )
        yield


def _read_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def is_geff_dir(path: Path) -> bool:
    """Whether the directory is a geff group: its zarr attributes carry a 'geff' key
    (zarr v2 keeps them in ``.zattrs``, zarr v3 in ``zarr.json``)."""
    zattrs = path / ".zattrs"
    if zattrs.is_file():
        return "geff" in _read_json(zattrs)
    zjson = path / "zarr.json"
    if zjson.is_file():
        return "geff" in _read_json(zjson).get("attributes", {})
    return False


def _is_zarr_array_dir(path: Path) -> bool:
    if (path / ".zarray").is_file():
        return True
    zjson = path / "zarr.json"
    return zjson.is_file() and _read_json(zjson).get("node_type") == "array"


def _group_name(store: Path, geff_dir: Path) -> str:
    """Name a geff by the group that holds it: ``trench_0165/tracking_graph.geff``
    is called ``trench_0165``; a geff at the top of the store, or a store that is
    itself a geff, is called after the geff directory."""
    if geff_dir == store:
        return store.stem
    relative = geff_dir.relative_to(store)
    if relative.parent == Path("."):
        return relative.stem
    return relative.parent.as_posix()


def find_geff_groups(store: Path) -> list[GeffGroup]:
    """List the geff groups in a zarr store, in sorted order.

    Walks the directory tree without descending into arrays (whose chunk files
    may be numerous) or into the geffs themselves. Works for zarr v2 and v3
    stores, and for stores mixing the two, which zarr's own hierarchy walk
    refuses to traverse.

    Args:
        store (Path): The store to search. May itself be a geff group.

    Returns:
        list[GeffGroup]: The geffs found, named by their location in the store.
    """
    store = Path(store)
    if is_geff_dir(store):
        return [GeffGroup(_group_name(store, store), store)]

    found: list[Path] = []
    for dirpath, dirnames, _ in os.walk(store):
        current = Path(dirpath)
        keep: list[str] = []
        for dirname in sorted(dirnames):
            if dirname.startswith("."):
                continue
            child = current / dirname
            if is_geff_dir(child):
                found.append(child)
            elif not _is_zarr_array_dir(child):
                keep.append(dirname)
        dirnames[:] = keep

    groups: list[GeffGroup] = []
    names = [_group_name(store, path) for path in found]
    for name, path in zip(names, found, strict=True):
        if names.count(name) > 1:
            # several geffs in one group: keep them apart by their own name
            name = f"{name}/{path.stem}"
        groups.append(GeffGroup(name, path))
    return sorted(groups, key=lambda group: group.name)


def read_related_arrays(
    geff_path: Path, metadata: GeffMetadata | None = None
) -> RelatedArrays:
    """Resolve the labels and image arrays named in the geff ``related_objects``.

    Paths in ``related_objects`` are relative to the geff group. Objects that do
    not exist on disk are ignored with a warning in the log, so a geff whose
    images were moved still loads, just without them.
    """
    geff_path = Path(geff_path)
    if metadata is None:
        metadata = GeffMetadata.read(geff_path)

    labels_path = label_prop = image_path = None
    for related in metadata.related_objects or []:
        target = (geff_path / related.path).resolve()
        if not target.exists():
            logger.warning(
                "Related %s object %s of %s does not exist; ignoring it",
                related.type,
                target,
                geff_path,
            )
            continue
        if related.type == "labels" and labels_path is None:
            labels_path, label_prop = target, related.label_prop
        elif related.type == "image" and image_path is None:
            image_path = target
    return RelatedArrays(
        labels_path=labels_path, label_prop=label_prop, image_path=image_path
    )


def node_name_map_from_metadata(
    metadata: GeffMetadata, label_prop: str | None = None
) -> dict[str, str | list[str]]:
    """Map funtracks' standard node keys to this geff's property names.

    Uses the typed ``axes`` (time / space) for ``time`` and ``pos``, the
    ``track_node_props`` for ``track_id`` and ``lineage_id``, and the labels'
    ``label_prop`` for ``seg_id``. Only properties that exist in the geff are
    mapped, so funtracks can fall back to its own inference for the rest.
    """
    props = set(metadata.node_props_metadata or {})
    axes = metadata.axes or []
    name_map: dict[str, str | list[str]] = {}

    time_axes = [axis.name for axis in axes if axis.type == "time"]
    space_axes = [axis.name for axis in axes if axis.type == "space"]
    if time_axes and time_axes[0] in props:
        name_map["time"] = time_axes[0]
    if space_axes and all(axis in props for axis in space_axes):
        name_map["pos"] = space_axes  # geff axes are listed in array order

    track_props = metadata.track_node_props or {}
    if track_props.get("tracklet") in props:
        name_map["track_id"] = track_props["tracklet"]
    if track_props.get("lineage") in props:
        name_map["lineage_id"] = track_props["lineage"]

    if label_prop in props:
        name_map["seg_id"] = label_prop
    return name_map


def open_array(path: Path) -> np.ndarray | zarr.Array:
    """Open a zarr array, reading it into memory when it is small enough."""
    array = zarr.open_array(Path(path), mode="r")
    if array.nbytes <= MAX_IN_MEMORY_BYTES:
        return array[:]
    return array


def load_geff_group(
    geff_path: Path, name: str | None = None, scale: list[float] | None = None
) -> LoadedGeff:
    """Load a geff and the image it refers to, using its own metadata.

    Args:
        geff_path (Path): The geff group to load.
        name (str | None): Name for the loaded tracks. Defaults to the geff's stem.
        scale (list[float] | None): Optional spatial scale for the tracks. Defaults
            to no scaling (1 in every dimension).

    Returns:
        LoadedGeff: The tracks (with segmentation, if the geff refers to labels) and
            the related image array, if any.
    """
    geff_path = Path(geff_path)
    with silence_geff_zarr_warnings():
        metadata = GeffMetadata.read(geff_path)
        related = read_related_arrays(geff_path, metadata)
        name_map = node_name_map_from_metadata(metadata, related.label_prop)
        tracks = import_from_geff(
            geff_path,
            name_map or None,
            segmentation_path=related.labels_path,
            scale=scale,
        )
    image = open_array(related.image_path) if related.image_path is not None else None
    return LoadedGeff(
        name=name or geff_path.stem,
        geff_path=geff_path,
        tracks=tracks,
        image=image,
        image_path=related.image_path,
    )


def add_geff_group_to_viewer(
    viewer: napari.Viewer,
    geff_path: Path,
    name: str | None = None,
    *,
    kymograph: bool = True,
    scale: list[float] | None = None,
) -> LoadedGeff:
    """Load a geff group and show it in the viewer, optionally as a kymograph.

    The image is added as a napari layer, the tracks are added to the Tracks List
    (so they can be edited, saved and exported like any other tracks), and the
    image is selected as the kymograph background. If the viewer is currently in
    kymograph mode, it is switched to the spatial view first so that the new
    layers are set up correctly, and back afterwards.

    Args:
        viewer (napari.Viewer): The viewer holding the motile tracker widgets.
        geff_path (Path): The geff group to load.
        name (str | None): Name for the tracks and image layers.
        kymograph (bool): Whether to switch to the kymograph view after loading.
        scale (list[float] | None): Optional spatial scale for the tracks and image.

    Returns:
        LoadedGeff: The loaded data, with ``image_layer`` set if an image was added.
    """
    # imported here: the import/export package must stay importable without the views
    from motile_tracker.data_views.views_coordinator.tracks_viewer import TracksViewer

    loaded = load_geff_group(geff_path, name=name, scale=scale)
    tracks_viewer = TracksViewer.get_instance(viewer)

    if tracks_viewer.view_mode == "kymograph":
        tracks_viewer.set_view_mode("spatial")

    if loaded.image is not None:
        loaded.image_layer = viewer.add_image(
            loaded.image,
            name=f"{loaded.name}_images",
            colormap="gray",
            scale=loaded.tracks.scale,
        )

    tracks_viewer.tracks_list.add_tracks(loaded.tracks, loaded.name, select=True)

    if loaded.image_layer is not None:
        tracks_viewer.set_kymograph_image_layer(loaded.image_layer.name)
    if kymograph:
        tracks_viewer.set_view_mode("kymograph")
    return loaded
