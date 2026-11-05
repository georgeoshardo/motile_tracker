"""The curated copy of a tracker's GEFF, made at its first edit.

A tracker's output is never edited (see :mod:`mm_track.experiment_layout`). The
first edit of a tracker's GEFF copies it, and its masks' label image, to their
curated places, and the edit history starts there::

    tracks/<run>.geff     ->  curation/<run>.geff     related labels: ../../labels/curation.<run>
    labels/tracks.<run>   ->  labels/curation.<run>   listed in labels/'s ome.labels

The tracker's files are only read. The copy is byte-for-byte except for the
references to the masks (the GEFF's ``related_objects``, the label image's own
name and the GEFF it belongs to), so the copy's history begins with the tracker's
cells exactly as they were opened.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from mm_track.experiment_layout import CURATION, TRACKS, CurationTarget
from mm_track.ome_zarr import is_multiscale

from . import source_format as sf
from .projection import dataset_signature


def _read(path: Path) -> dict:
    return json.loads((path / "zarr.json").read_text())


def _write(path: Path, doc: dict) -> None:
    (path / "zarr.json").write_text(json.dumps(doc, indent=2))


def _label_image(target: CurationTarget) -> None:
    """Copy the tracker's label image to the curated one, named as such."""
    shutil.copytree(target.source_labels, target.labels, copy_function=shutil.copyfile)
    doc = _read(target.labels)
    attrs = doc.setdefault("attributes", {})
    for multiscale in (attrs.get("ome") or {}).get("multiscales") or []:
        multiscale["name"] = target.labels.name
    attrs["pymmm"] = {
        **(attrs.get("pymmm") or {}),
        "tracks": f"../../{CURATION}/{target.run}.geff",
        "description": "the curated masks; a mask's value is its GEFF node's seg_id",
        "curated_from": f"{TRACKS}.{target.run}",
    }
    _write(target.labels, doc)


def _list_label(target: CurationTarget) -> None:
    """List the curated label image in the image's ``labels/`` (OME ``ome.labels``)."""
    labels = target.labels.parent
    doc = _read(labels)
    ome = doc.setdefault("attributes", {}).setdefault("ome", {})
    listed = list(ome.get("labels") or [])
    if target.labels.name not in listed:
        ome["labels"] = [*listed, target.labels.name]
        _write(labels, doc)


def _curation_group(target: CurationTarget) -> None:
    group = target.geff.parent
    if (group / "zarr.json").is_file():
        return
    group.mkdir(exist_ok=True)
    _write(group, {
        "zarr_format": 3,
        "node_type": "group",
        "attributes": {"pymmm": {
            "description": "curated copies of tracks/ GEFFs, each made by mm-track at "
            "the trench's first edit"
        }},
    })


def make_curated_copy(target: CurationTarget, initial: dict) -> dict:
    """Copy a tracker's GEFF and its label image to their curated places.

    Args:
        target: Where the copy goes (:func:`~mm_track.experiment_layout.curation_target`).
        initial: The :func:`~.session.snapshot` taken when the tracker's GEFF was
            opened.

    Returns:
        The snapshot for the copy, to start its history with.

    Raises:
        ValueError: if the tracker's GEFF changed since it was opened.
        FileExistsError: if a curated copy is already there.
        UnsupportedFormat: if the GEFF's format was not recorded, or its masks are
            not an OME-Zarr label image.
    """
    fmt = initial["format"]
    source = target.source_geff
    if fmt is None:
        raise sf.UnsupportedFormat(f"{source}: its format was not recorded")
    if not is_multiscale(target.source_labels):
        raise sf.UnsupportedFormat(f"{target.source_labels} is not an OME-Zarr label image")
    if (
        initial["signature"] is not None
        and dataset_signature(source, fmt) != initial["signature"]
    ):
        raise ValueError(
            "Dataset changed on disk since it was opened; reopen it before editing"
        )
    for path in (target.geff, target.labels):
        if path.exists():
            raise FileExistsError(f"{path} exists already; reopen the trench to edit it")

    made = []
    try:
        made.append(target.labels)
        _label_image(target)
        _curation_group(target)
        made.append(target.geff)
        shutil.copytree(source, target.geff, copy_function=shutil.copyfile)
        fmt = sf.rebase(fmt, source, target.geff, labels_target=target.labels)
        (target.geff / sf.root_file(fmt)).write_text(fmt["root_text"])
        _list_label(target)
    except BaseException:
        for path in made:
            shutil.rmtree(path, ignore_errors=True)
        raise
    return {
        **initial,
        "format": fmt,
        "attrs": sf.root_attrs(fmt),
        "signature": dataset_signature(target.geff, fmt),
    }


def remove_curated_copy(target: CurationTarget) -> None:
    """Remove a curated copy made by :func:`make_curated_copy` whose history could
    not be started, so the next edit makes it again."""
    labels = target.labels.parent
    if (labels / "zarr.json").is_file():
        doc = _read(labels)
        ome = (doc.get("attributes") or {}).get("ome") or {}
        if target.labels.name in (ome.get("labels") or []):
            ome["labels"] = [n for n in ome["labels"] if n != target.labels.name]
            _write(labels, doc)
    for path in (target.geff, target.labels):
        shutil.rmtree(path, ignore_errors=True)
