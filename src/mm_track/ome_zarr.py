"""OME-Zarr images and label images as GEFF related objects.

A GEFF's ``related_objects`` may name an OME-Zarr multiscale group (an image, or
an image-label in an image's ``labels/`` group) rather than a bare zarr array, as
when each trench is its own OME-Zarr image::

    trench_0396/                     OME-Zarr image, (t, c, y, x)
        0
        labels/tracks.chain_S1       image-label, (t, y, x); its array is ``0``
        tracks/chain_S1.geff         related_objects -> ../../labels/tracks.chain_S1, ../..

mm-track reads and writes arrays, so such a reference stands for the group's first
(full resolution) dataset. Edited masks are then published by replacing that array
alone, never the group's OME metadata.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


def group_attributes(path: Path) -> dict | None:
    """The attributes of a zarr v3 group (or v2 ``.zattrs``), None for anything else."""
    path = Path(path)
    v3 = path / "zarr.json"
    if v3.is_file():
        doc = json.loads(v3.read_text())
        return (doc.get("attributes") or {}) if doc.get("node_type") == "group" else None
    if (path / ".zgroup").is_file():
        v2 = path / ".zattrs"
        return json.loads(v2.read_text()) if v2.is_file() else {}
    return None


def _ome(path: Path) -> dict | None:
    attrs = group_attributes(path)
    if attrs is None:
        return None
    return attrs.get("ome", attrs)  # 0.5 nests under "ome"; 0.4 does not


def is_multiscale(path: Path) -> bool:
    """Whether `path` is an OME-Zarr multiscale group (an image or an image-label)."""
    return bool((_ome(path) or {}).get("multiscales"))


def multiscale_array(path: Path) -> Path:
    """The full-resolution array of an OME-Zarr multiscale group; `path` otherwise."""
    multiscales = (_ome(path) or {}).get("multiscales")
    if not multiscales:
        return Path(path)
    return Path(path) / multiscales[0]["datasets"][0]["path"]


@dataclass(frozen=True)
class Channel:
    """One channel of an OME-Zarr image, with its ``omero`` display settings."""

    index: int
    name: str
    color: str | None = None  # RRGGBB
    contrast_limits: tuple[float, float] | None = None
    visible: bool = True


def image_channels(path: Path) -> tuple[int | None, list[Channel]]:
    """The channel axis of an OME-Zarr image group and its channels.

    Returns (None, []) for a bare array, or an image without a channel axis.
    """
    ome = _ome(path)
    multiscales = (ome or {}).get("multiscales")
    if not multiscales:
        return None, []
    axes = multiscales[0].get("axes") or []
    names = [a["name"] if isinstance(a, dict) else a for a in axes]
    types = [a.get("type") if isinstance(a, dict) else None for a in axes]
    if "channel" in types:
        axis = types.index("channel")
    elif "c" in names:
        axis = names.index("c")
    else:
        return None, []
    array = json.loads((multiscale_array(path) / "zarr.json").read_text())
    n = array["shape"][axis]
    omero = ((ome or {}).get("omero") or {}).get("channels") or []
    channels = []
    for i in range(n):
        info = omero[i] if i < len(omero) else {}
        window = info.get("window") or {}
        limits = (
            (float(window["start"]), float(window["end"]))
            if "start" in window and "end" in window
            else None
        )
        channels.append(
            Channel(
                index=i,
                name=str(info.get("label") or f"c{i}"),
                color=info.get("color"),
                contrast_limits=limits,
                visible=bool(info.get("active", True)),
            )
        )
    return axis, channels
