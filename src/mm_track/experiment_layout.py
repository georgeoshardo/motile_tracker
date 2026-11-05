"""PyMMM's experiment layout: one OME-Zarr image per trench, with its tracking runs
and their curated copies beside the pixels::

    experiment.zarr/                attrs.pymmm: layout "experiment"; tracks: the runs
        trench_0396/                OME-Zarr image, (t, c, y, x)
            labels/tracks.<run>     the tracker's masks; a mask's value is its node's seg_id
            labels/curation.<run>   the curated masks
            tracks/<run>.geff       the tracker's GEFF; never edited
            curation/<run>.geff     the curated copy, made at the trench's first edit

A tracker's GEFF opens as its curated copy when there is one. Until then it opens
as it is, and its first edit makes the copy and carries on there
(:mod:`mm_track.persistence.curation`), so the tracker's output stays as the tracker
wrote it.

An edit does not make a trench curated: the curator marks it reviewed when it is
finished. The mark is a stamp in the curated GEFF's ``geff.extra``::

    "reviewed": {"at": "2026-11-05T14:02:11+04:00", "by": "georgeos", "revision": 12}

``revision`` is the edit-history revision that was reviewed; an edit after the mark
leaves it in place, and the stamp then names an older revision than the GEFF's.
"""

from __future__ import annotations

import getpass
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from mm_track.ome_zarr import group_attributes, is_multiscale

TRACKS = "tracks"
CURATION = "curation"
REVIEW_KEY = "reviewed"


def experiment_attrs(store: Path) -> dict | None:
    """The ``pymmm`` attributes of an experiment group; None for anything else."""
    attrs = group_attributes(Path(store)) or {}
    pymmm = attrs.get("pymmm") or {}
    return pymmm if pymmm.get("layout") == "experiment" else None


def is_experiment(store: Path) -> bool:
    return experiment_attrs(store) is not None


def runs(store: Path) -> list[str]:
    """The tracking runs recorded in an experiment group, in record order."""
    return list((experiment_attrs(store) or {}).get("tracks") or {})


@dataclass(frozen=True)
class CurationTarget:
    """Where the curated copy of a tracker's GEFF goes, and what it is copied from."""

    run: str
    image: Path  # the trench image
    source_geff: Path  # tracks/<run>.geff
    source_labels: Path  # labels/tracks.<run>
    geff: Path  # curation/<run>.geff
    labels: Path  # labels/curation.<run>


def curation_target(geff_path: Path) -> CurationTarget | None:
    """The curated copy's place for a tracker's GEFF (``<image>/tracks/<run>.geff``);
    None for any other GEFF."""
    geff_path = Path(geff_path).resolve()
    image = geff_path.parent.parent
    if (
        geff_path.suffix != ".geff"
        or geff_path.parent.name != TRACKS
        or not is_multiscale(image)
    ):
        return None
    run = geff_path.stem
    return CurationTarget(
        run=run,
        image=image,
        source_geff=geff_path,
        source_labels=image / "labels" / f"{TRACKS}.{run}",
        geff=image / CURATION / f"{run}.geff",
        labels=image / "labels" / f"{CURATION}.{run}",
    )


def editable_geff(geff_path: Path) -> Path:
    """The GEFF to open for `geff_path`: a tracker's GEFF's curated copy once there
    is one, `geff_path` itself otherwise."""
    target = curation_target(geff_path)
    if target is not None and (target.geff / "zarr.json").is_file():
        return target.geff
    return Path(geff_path)


def trench_geffs(store: Path, run: str) -> list[tuple[str, Path]]:
    """(trench image name, tracker GEFF) of every trench with the run, in name order.

    A trench whose tracker GEFF is gone but whose curated copy is there is listed
    with the curated copy.
    """
    found = []
    for image in sorted(Path(store).iterdir()):
        if not image.is_dir() or image.name.startswith("."):
            continue
        for kind in (TRACKS, CURATION):
            geff = image / kind / f"{run}.geff"
            if (geff / "zarr.json").is_file():
                found.append((image.name, geff))
                break
    return found


def review_of(attrs: dict) -> dict | None:
    """The review stamp in a GEFF's root attributes, None if not reviewed."""
    return ((attrs.get("geff") or {}).get("extra") or {}).get(REVIEW_KEY)


def with_review(attrs: dict, review: dict | None) -> dict:
    """A GEFF's root attributes with the review stamp set (or removed, for None)."""
    attrs = dict(attrs)
    geff = dict(attrs.get("geff") or {})
    extra = dict(geff.get("extra") or {})
    if review is None:
        extra.pop(REVIEW_KEY, None)
    else:
        extra[REVIEW_KEY] = dict(review)
    geff["extra"] = extra
    attrs["geff"] = geff
    return attrs


def review_stamp(revision: int) -> dict:
    """A review stamp for the given edit-history revision, by the current user, now."""
    return {
        "at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "by": getpass.getuser(),
        "revision": int(revision),
    }


def trench_state(geff_path: Path) -> str | None:
    """"reviewed" or "edited" for a GEFF with a curated copy (or that is one), None
    for a tracker's GEFF nobody has edited yet."""
    path = editable_geff(geff_path)
    if path.parent.name != CURATION:
        return None
    return "reviewed" if review_of(group_attributes(path) or {}) else "edited"
