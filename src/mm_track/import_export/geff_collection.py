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

A PyMMM experiment group (one OME-Zarr image per trench, see
:mod:`mm_track.experiment_layout`) lists one GEFF per trench for a tracking run,
and a tracker's GEFF loads as its curated copy once there is one. Related objects
may be OME-Zarr groups (:mod:`mm_track.ome_zarr`): an image's channels load as one
layer each.
"""

from __future__ import annotations

import json
import logging
import os
import warnings
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import zarr
from funtracks.import_export import import_from_geff
from geff import GeffMetadata

from mm_track import experiment_layout
from mm_track.ome_zarr import Channel, image_channels, multiscale_array
from mm_track.persistence.paths import open_session
from mm_track.persistence.session import EditSession, has_history

if TYPE_CHECKING:
    from collections.abc import Iterator

    import napari
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
    # an OME-Zarr image with a channel axis: the axis and its channels (omero)
    channel_axis: int | None = None
    channels: tuple[Channel, ...] = ()


@dataclass
class LoadedGeff:
    """A GEFF group loaded with its related arrays."""

    name: str
    geff_path: Path
    tracks: Tracks
    image: np.ndarray | zarr.Array | None = None
    image_path: Path | None = None
    image_layer: napari.layers.Image | None = None
    # one (channel, (T, Y, X) array) per channel of an OME-Zarr image; `image` is
    # then the first visible channel
    channels: list[tuple[Channel, np.ndarray | zarr.Array]] | None = None
    channel_layers: list[napari.layers.Image] | None = None

    @property
    def image_layers(self) -> list[napari.layers.Image]:
        """Every image layer added for this group."""
        if self.channel_layers:
            return list(self.channel_layers)
        return [self.image_layer] if self.image_layer is not None else []


@contextmanager
def silence_geff_zarr_warnings() -> Iterator[None]:
    """Silence the warnings zarr and geff emit for stores that hold non-geff members,
    and funtracks' notice that it numbers cells from 1.

    A store with images next to the graph, or a zarr v2 geff inside a zarr v3
    store, is exactly what this module is for, so the warnings say nothing the
    user can act on (see also ``geff_io.write_geff_over``). Nor does the notice:
    a geff that numbers its cells from 0 is valid, and funtracks shifts the ids
    on every load.
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
        warnings.filterwarnings(
            "ignore", message="Found node_id 0 in the loaded graph", category=UserWarning
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
    if has_history(path):
        return True
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


def find_geff_groups(store: Path, run: str | None = None) -> list[GeffGroup]:
    """List the geff groups in a zarr store, in sorted order.

    Walks the directory tree without descending into arrays (whose chunk files
    may be numerous) or into the geffs themselves. Works for zarr v2 and v3
    stores, and for stores mixing the two, which zarr's own hierarchy walk
    refuses to traverse.

    A PyMMM experiment group lists one tracking run's GEFF per trench instead,
    named after the trench (its tracker GEFF, which loads as its curated copy once
    there is one).

    Args:
        store (Path): The store to search. May itself be a geff group.
        run (str | None): For an experiment group, the tracking run to list.
            Defaults to the first one recorded.

    Returns:
        list[GeffGroup]: The geffs found, named by their location in the store.
    """
    store = Path(store)
    if is_geff_dir(store):
        return [GeffGroup(_group_name(store, store), store)]
    if experiment_layout.is_experiment(store):
        run = run or next(iter(experiment_layout.runs(store)), None)
        if run is None:
            return []
        return [
            GeffGroup(name, path)
            for name, path in experiment_layout.trench_geffs(store, run)
        ]

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
    images were moved still loads, just without them. An OME-Zarr multiscale
    group (image or image-label) stands for its full-resolution array; an image's
    channel axis and ``omero`` channels are returned with it.
    """
    geff_path = Path(geff_path)
    if metadata is None:
        metadata = GeffMetadata.read(geff_path)

    labels_path = label_prop = image_path = channel_axis = None
    channels: tuple[Channel, ...] = ()
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
            # node_prop replaced label_prop in geff-spec 1.2.1; read either.
            labels_path = multiscale_array(target)
            label_prop = getattr(related, "node_prop", None) or related.label_prop
        elif related.type == "image" and image_path is None:
            image_path = multiscale_array(target)
            channel_axis, found = image_channels(target)
            channels = tuple(found)
    return RelatedArrays(
        labels_path=labels_path,
        label_prop=label_prop,
        image_path=image_path,
        channel_axis=channel_axis,
        channels=channels,
    )


# funtracks' own node keys: a geff property with one of these names is only taken
# through the mapping above (or not at all, like tracklet ids), never as an extra.
_FUNTRACKS_KEYS = frozenset({
    "t", "time", "pos", "node_id", "seg_id", "track_id", "tracklet_id", "lineage_id",
    "mask", "bbox", "solution", "area",
})


def node_name_map_from_metadata(
    metadata: GeffMetadata,
    label_prop: str | None = None,
    geff_path: Path | None = None,
) -> dict[str, str | list[str]]:
    """Map funtracks' standard node keys to this geff's property names.

    Uses the typed ``axes`` (time / space) for ``time`` and ``pos``, the
    ``track_node_props`` lineage for ``lineage_id``, and the labels'
    ``label_prop`` for ``seg_id``. Tracklets are left to funtracks. Only properties that exist in the geff are
    mapped, so funtracks can fall back to its own inference for the rest. Given the
    geff's path, any other numeric node property with one value per node is kept
    under its own name.
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
    # Tracklets are not imported: funtracks derives them from the links (and keeps
    # them up to date while editing), and the dataset's own tracklet numbers are
    # kept by the autosave format (see persistence.source_format). Mapping them to
    # a key funtracks does not maintain would leave a stale copy in the graph.
    if track_props.get("lineage") in props:
        name_map["lineage_id"] = track_props["lineage"]

    if label_prop in props:
        name_map["seg_id"] = label_prop

    # Keep the geff's other numeric scalar node properties (scores and the like) as
    # node features under their own names, so they can colour the masks; funtracks
    # drops properties that are not in the map.
    used = {v for value in name_map.values()
            for v in (value if isinstance(value, list) else [value])}
    for prop, meta in (metadata.node_props_metadata or {}).items():
        if (prop in used or prop in name_map or prop in _FUNTRACKS_KEYS
                or prop == track_props.get("tracklet")):
            continue
        if getattr(meta, "varlength", False):
            continue
        try:
            numeric = np.issubdtype(np.dtype(meta.dtype), np.number)
        except TypeError:
            numeric = False
        if numeric and geff_path is not None and _is_scalar_prop(geff_path, prop):
            name_map[prop] = prop
    return name_map


def _is_scalar_prop(geff_path: Path, prop: str) -> bool:
    """Whether a geff node property holds one value per node.

    The metadata does not say: a property such as ``coords`` (one row of numbers per
    node) has a numeric dtype too. Kept as a node feature, a vector property breaks
    saving as soon as an edit adds a node, which has no value for it.
    """
    try:
        values = zarr.open_array(Path(geff_path) / "nodes" / "props" / prop / "values", mode="r")
    except (FileNotFoundError, KeyError, ValueError, zarr.errors.BaseZarrError):
        return False
    return values.ndim == 1


def open_array(path: Path) -> np.ndarray | zarr.Array:
    """Open a zarr array, reading it into memory when it is small enough."""
    array = zarr.open_array(Path(path), mode="r")
    if array.nbytes <= MAX_IN_MEMORY_BYTES:
        return array[:]
    return array


class _ChannelView:
    """One channel of a large zarr image, read lazily (napari slices it)."""

    def __init__(self, array: zarr.Array, axis: int, index: int):
        self.array, self.axis, self.index = array, axis, index
        shape = list(array.shape)
        del shape[axis]
        self.shape, self.dtype, self.ndim = tuple(shape), array.dtype, len(shape)

    def __getitem__(self, key):
        key = key if isinstance(key, tuple) else (key,)
        key = list(key) + [slice(None)] * (self.ndim - len(key))
        key.insert(self.axis, self.index)
        return self.array[tuple(key)]

    def __array__(self, dtype=None, copy=None):
        return np.asarray(self[()], dtype=dtype)


def _read_channels(
    related: RelatedArrays,
) -> list[tuple[Channel, np.ndarray | _ChannelView]]:
    """The channels of an OME-Zarr image, each (T, Y, X): in memory when the image
    is small enough, read lazily otherwise."""
    array = zarr.open_array(related.image_path, mode="r")
    if array.nbytes <= MAX_IN_MEMORY_BYTES:
        data = np.asarray(array[...])
        return [
            (channel, np.take(data, channel.index, axis=related.channel_axis))
            for channel in related.channels
        ]
    return [
        (channel, _ChannelView(array, related.channel_axis, channel.index))
        for channel in related.channels
    ]


def load_geff_group(
    geff_path: Path,
    name: str | None = None,
    scale: list[float] | None = None,
    session: EditSession | None = None,
) -> LoadedGeff:
    """Load a geff and the image it refers to, using its own metadata.

    Args:
        geff_path (Path): The geff group to load.
        name (str | None): Name for the loaded tracks. Defaults to the geff's stem.
        scale (list[float] | None): Optional spatial scale for the tracks. Defaults
            to no scaling (1 in every dimension).

    A tracker's GEFF with a curated copy (see :mod:`mm_track.experiment_layout`)
    loads as that copy.

    Returns:
        LoadedGeff: The tracks (with segmentation, if the geff refers to labels) and
            the related image array, if any.
    """
    geff_path = experiment_layout.editable_geff(Path(geff_path))
    supplied_session = session
    try:
        with silence_geff_zarr_warnings():
            if session is None and has_history(geff_path):
                from appdirs import AppDirs

                session = open_session(geff_path, AppDirs("mm-track").user_data_dir)
            if session is not None:
                geff_path = session.path
                # Recover root metadata before resolving its related image paths.
                session.flush()
            metadata = GeffMetadata.read(geff_path)
            related = read_related_arrays(geff_path, metadata)
            name_map = node_name_map_from_metadata(
                metadata, related.label_prop, geff_path
            )
            tracks = (
                session.tracks
                if session is not None
                else import_from_geff(
                    geff_path,
                    name_map or None,
                    segmentation_path=related.labels_path,
                    scale=scale,
                )
            )
        image = channels = None
        if related.image_path is not None and related.channel_axis is not None:
            channels = _read_channels(related)
            shown = [data for channel, data in channels if channel.visible]
            image = (shown or [data for _, data in channels] or [None])[0]
        elif related.image_path is not None:
            image = open_array(related.image_path)
        return LoadedGeff(
            name=name or geff_path.stem,
            geff_path=geff_path,
            tracks=tracks,
            image=image,
            image_path=related.image_path,
            channels=channels,
        )
    except BaseException:
        if session is not None and supplied_session is None:
            with suppress(RuntimeError):  # Preserve the original loading error.
                session.close()
        raise


def _channel_colormap(channel: Channel):
    if not channel.color:
        return "gray"
    from napari.utils.colormaps import Colormap

    color = "#" + channel.color.lstrip("#")
    if color.upper() == "#FFFFFF":
        return "gray"
    return Colormap(["#000000", color], name=f"{channel.name}_{color}")


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
        viewer (napari.Viewer): The viewer holding mm_track widgets.
        geff_path (Path): The geff group to load.
        name (str | None): Name for the tracks and image layers.
        kymograph (bool): Whether to switch to the kymograph view after loading.
        scale (list[float] | None): Optional spatial scale for the tracks and image.

    Returns:
        LoadedGeff: The loaded data, with ``image_layer`` set if an image was added.
    """
    # imported here: the import/export package must stay importable without the views
    from mm_track.data_views.views_coordinator.tracks_viewer import TracksViewer

    tracks_viewer = TracksViewer.get_instance(viewer)
    geff_path = experiment_layout.editable_geff(Path(geff_path))
    session = tracks_viewer.tracks_list.session_at(geff_path)
    loaded = load_geff_group(geff_path, name=name, scale=scale, session=session)

    if tracks_viewer.view_mode == "kymograph":
        tracks_viewer.set_view_mode("spatial")

    if loaded.channels:
        # an OME-Zarr image: one layer per channel, displayed as its omero says
        loaded.channel_layers = []
        for channel, data in loaded.channels:
            layer = viewer.add_image(
                data,
                name=f"{loaded.name}_{channel.name}",
                colormap=_channel_colormap(channel),
                contrast_limits=channel.contrast_limits,
                visible=channel.visible,
                blending="additive",
                scale=loaded.tracks.scale,
            )
            loaded.channel_layers.append(layer)
            if data is loaded.image and loaded.image_layer is None:
                loaded.image_layer = layer
    elif loaded.image is not None:
        loaded.image_layer = viewer.add_image(
            loaded.image,
            name=f"{loaded.name}_images",
            colormap="gray",
            scale=loaded.tracks.scale,
        )

    loaded.tracks = tracks_viewer.tracks_list.add_tracks(
        loaded.tracks, loaded.name, select=True, source_path=loaded.geff_path
    )

    if loaded.image_layer is not None:
        tracks_viewer.set_kymograph_image_layer(loaded.image_layer.name)
    if kymograph:
        tracks_viewer.set_view_mode("kymograph")
    return loaded
