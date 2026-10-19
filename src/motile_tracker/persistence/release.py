"""Fix up and release datasets edited with this application.

Commands (``python -m motile_tracker.persistence.release <command> ...``):

``attach-format STORE --reference REFERENCE``
    For datasets whose history was started before formats were kept: record the
    format of the original dataset (REFERENCE, laid out like STORE) in each
    history, after checking that the history starts from exactly that dataset.

``republish STORE``
    Rebuild each dataset's published files from its history, in its recorded
    format (graph and label array), through the publication gate.

``conform STORE --like TEMPLATE [--drop-extra] [--remove NAME ...]``
    Rewrite every dataset in the format of TEMPLATE (a GEFF group): properties
    the template lacks are dropped with their values, data types, encodings,
    metadata layout and the label and image arrays' metadata become the
    template's. Datasets with a history get the new format recorded, so further
    editing keeps it. ``--drop-extra`` also drops the ``extra`` metadata, and
    ``--remove`` deletes the named files from each dataset's group (e.g. export
    markers). Every dataset is first rewritten in a scratch directory and must
    pass the publication gate before any is changed.

``export STORE OUT [--drop-extra] [--report CSV]``
    Write a release copy: every dataset in its recorded format, without edit
    histories or lock files, every one checked by the publication gate. Groups
    and arrays next to the GEFF (e.g. raw images) are copied as they are.

STORE is a single GEFF group or a zarr store holding one GEFF per group (such as
one group per trench).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path

from . import source_format as sf
from .projection import publish
from .state import restore
from .store import Store, history_path

SKIP = {".DS_Store", "writer.lock"}


def find_geffs(store):
    store = Path(store)
    if (store / "nodes").is_dir():
        return [store]
    return sorted(p.parent for p in store.glob("*/*/nodes") if p.is_dir())


def _initial_tracks(store):
    return restore(store.state_at(0), store.get_blob)


def load_dataset(geff):
    """A dataset's current tracks, format and revision.

    With an edit history: the latest committed revision and the recorded format.
    Without one (never edited): the files as they are, and their format.
    """
    geff = Path(geff)
    if history_path(geff).is_file():
        with Store(geff, readonly=True) as source:
            revision, state, _ = source.read(include_history=False)
            fmt = source.metadata("source_format")
            if fmt is None:
                raise ValueError(
                    f"{geff}: no recorded format (run attach-format first)"
                )
            return restore(state, source.get_blob), fmt, revision
    from motile_tracker.import_export.geff_collection import load_geff_group

    tracks = load_geff_group(geff).tracks
    return tracks, sf.capture(geff, tracks), None


def _trial(tracks, fmt):
    """Gate problems of writing the tracks in a format, in a scratch directory."""
    with tempfile.TemporaryDirectory(prefix="motile-conform-") as scratch:
        geff, labels = Path(scratch) / "tracks.geff", Path(scratch) / "labels"
        content = sf.write(tracks, fmt, geff, labels if fmt.get("labels") else None)
        return sf.check(geff, labels if fmt.get("labels") else None, fmt, content)


def _rewrite_without_history(geff, tracks, fmt):
    """Rewrite a never-edited dataset in a format: staged, gated, then swapped in."""
    geff = Path(geff)
    labels = sf.labels_path(geff, fmt)
    with tempfile.TemporaryDirectory(
        prefix=".motile-conform-", dir=geff.parent
    ) as scratch:
        staged = Path(scratch) / geff.name
        staged_labels = Path(scratch) / "labels" if labels is not None else None
        content = sf.write(tracks, fmt, staged, staged_labels)
        problems = sf.check(staged, staged_labels, fmt, content)
        if problems:
            raise sf.PublicationRejected(problems)
        for member in staged.iterdir():
            target = geff / member.name
            if member.is_dir() and target.exists():
                shutil.rmtree(target)
            os.replace(member, target)
        if labels is not None:
            os.replace(labels, Path(scratch) / "previous_labels")
            os.replace(staged_labels, labels)


def conform_store(store, template_geff, *, drop_extra=False, remove=()):
    """Rewrite every dataset in `store` in the format of `template_geff`.

    All datasets are planned and trial-written first; nothing changes unless
    every one passes the gate. Returns one row per dataset.
    """
    store = Path(store).resolve()
    template = sf.read_layout(template_geff)
    image_template = template["related"].get("image", {}).get("zarr.json")
    plans = []
    for geff in find_geffs(store):
        tracks, fmt, revision = load_dataset(geff)
        new = sf.conform(fmt, template, drop_extra=drop_extra)
        problems = _trial(tracks, new)
        if problems:
            raise sf.PublicationRejected([f"{geff.parent.name}: {p}" for p in problems])
        image_json = None
        attrs = json.loads(new["root_text"])["geff"]
        for obj in attrs.get("related_objects") or []:
            target = (geff / obj.get("path", "")).resolve()
            if (
                obj.get("type") == "image"
                and image_template
                and (target / "zarr.json").is_file()
            ):
                text = (target / "zarr.json").read_text()
                image_json = (
                    target / "zarr.json",
                    sf.conform_array_json(text, image_template),
                )
        dropped = sorted(set(fmt["node_props"]) - set(new["node_props"]))
        dropped += [
            f"edges/{p}"
            for p in sorted(set(fmt["edge_props"]) - set(new["edge_props"]))
        ]
        plans.append((geff, tracks, new, revision, image_json, dropped))

    rows = []
    for geff, tracks, new, revision, image_json, dropped in plans:
        if revision is not None:
            record_format(geff, new)
            republish(geff)
        else:
            _rewrite_without_history(geff, tracks, new)
        if image_json is not None and image_json[0].read_text() != image_json[1]:
            image_json[0].write_text(image_json[1])
        removed = []
        for name in remove:
            target = geff.parent / name
            if target.is_file():
                target.unlink()
                removed.append(name)
        rows.append(
            {
                "dataset": geff.parent.name,
                "history": revision is not None,
                "dropped": dropped,
                "removed": removed,
            }
        )
    return rows


def verify_format(geff, reference):
    """The format of `reference`, checked against the start of `geff`'s history.

    Returns None if `geff` already has a format recorded. Raises if the history
    does not start from exactly the reference dataset (ids, times and masks).
    """
    geff, reference = Path(geff), Path(reference)
    with Store(geff, readonly=True) as store:
        if store.metadata("source_format") is not None:
            return None
        tracks = _initial_tracks(store)
    # The dataset is laid out like the reference, so the reference's relative
    # paths (e.g. ../segmentation) name the dataset's own arrays.
    fmt = sf.capture(reference, tracks)
    labels = sf.labels_path(geff, fmt)
    if fmt.get("labels") and (labels is None or not labels.exists()):
        raise FileNotFoundError(
            f"{geff}: label array {fmt['labels']['path']} not found"
        )
    return fmt


def record_format(geff, fmt):
    with Store(geff) as store:
        store.set_metadata("source_format", fmt)
        # The published files are about to be rebuilt in this format.
        store.set_metadata("projected_revision", -1)


def attach_format(geff, reference):
    """Record `reference`'s format in `geff`'s history (see module docs)."""
    fmt = verify_format(geff, reference)
    if fmt is None:
        return "already recorded"
    record_format(geff, fmt)
    return "recorded"


def _require_format(geff):
    with Store(geff, readonly=True) as store:
        if store.metadata("source_format") is None:
            raise ValueError(f"{geff}: no recorded format; run attach-format first")


def republish(geff):
    geff = Path(geff)
    _require_format(geff)
    revision, signature = publish(geff)
    with Store(geff) as store:
        store.set_metadata("graph_signature", signature)
        store.set_metadata("projected_revision", revision)
    (geff / "edit_history" / "publishing").unlink(missing_ok=True)
    return revision


def _diff(fmt, content):
    """How a dataset differs from the one its history started from."""
    archived = fmt["archived"]
    order = set(content["order"])
    original = {int(k) for k in archived}
    edges = set(content["edge_order"])
    original_edges = {tuple(e) for e in fmt["edge_order"]}
    renumbered = sum(
        1
        for n in order & original
        if archived[str(n)].get("tracklet") is not None
        and content["tracklet"][n] != archived[str(n)]["tracklet"]
    )
    lineage_changed = sum(
        1
        for n in order & original
        if archived[str(n)].get("lineage") is not None
        and content["lineage"][n] != archived[str(n)]["lineage"]
    )
    return {
        "cells": len(order),
        "cells_added": len(order - original),
        "cells_removed": len(original - order),
        "masks_changed": len(content["changed"] & original),
        "links": len(edges),
        "links_added": len(edges - original_edges),
        "links_removed": len(original_edges - edges),
        "cells_with_new_tracklet_number": renumbered,
        "cells_with_new_lineage_number": lineage_changed,
        "frame_skipping_links": sum(
            1 for s, t in edges if content["times"][t] - content["times"][s] > 1
        ),
    }


def export(store, out, *, drop_extra=False, report=None):
    store, out = Path(store).resolve(), Path(out).resolve()
    if out.exists():
        raise FileExistsError(out)
    geffs = find_geffs(store)
    if not geffs:
        raise FileNotFoundError(f"No GEFF found in {store}")
    out.mkdir(parents=True)
    single = geffs == [store]
    rows, failures = [], []
    for geff in geffs:
        group = geff.parent
        target_group = out if single else out / group.relative_to(store)
        tracks, fmt, revision = load_dataset(geff)
        labels = sf.labels_path(geff, fmt)
        target_geff = (out / geff.name) if single else target_group / geff.name
        target_labels = (
            (target_geff / fmt["labels"]["path"]).resolve() if labels else None
        )
        if not single:
            # everything else in the group (raw images, group metadata) as it is
            target_group.mkdir(parents=True, exist_ok=True)
            for member in group.iterdir():
                if member.name in SKIP or member in (geff, labels):
                    continue
                destination = target_group / member.name
                if member.is_dir():
                    shutil.copytree(
                        member, destination, ignore=shutil.ignore_patterns(*SKIP)
                    )
                else:
                    shutil.copy2(member, destination)
        content = sf.write(
            tracks, fmt, target_geff, target_labels, drop_extra=drop_extra
        )
        problems = sf.check(
            target_geff, target_labels, fmt, content, drop_extra=drop_extra
        )
        row = {
            "dataset": str(group.relative_to(store)) if not single else geff.name,
            "revision": revision,
            **_diff(fmt, content),
            "gate": "ok" if not problems else "; ".join(problems[:3]),
        }
        rows.append(row)
        if problems:
            failures.append(row["dataset"])
    if not single and (store / "zarr.json").is_file():
        shutil.copy2(store / "zarr.json", out / "zarr.json")
    if report is not None:
        with open(report, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    if failures:
        raise sf.PublicationRejected([f"{d} failed the gate" for d in failures])
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m motile_tracker.persistence.release"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    attach = commands.add_parser("attach-format")
    attach.add_argument("store", type=Path)
    attach.add_argument("--reference", type=Path, required=True)
    again = commands.add_parser("republish")
    again.add_argument("store", type=Path)
    like = commands.add_parser("conform")
    like.add_argument("store", type=Path)
    like.add_argument("--like", type=Path, required=True)
    like.add_argument("--drop-extra", action="store_true")
    like.add_argument("--remove", action="append", default=[])
    out = commands.add_parser("export")
    out.add_argument("store", type=Path)
    out.add_argument("out", type=Path)
    out.add_argument("--drop-extra", action="store_true")
    out.add_argument("--report", type=Path)
    args = parser.parse_args(argv)

    store = args.store.resolve()
    if args.command == "attach-format":
        reference_root = args.reference.resolve()
        geffs = [g for g in find_geffs(store) if history_path(g).is_file()]
        skipped = len(find_geffs(store)) - len(geffs)
        # check every dataset before recording anything
        formats = {}
        for geff in geffs:
            relative = geff.relative_to(store)
            reference = reference_root / relative if relative.parts else reference_root
            formats[geff] = verify_format(geff, reference)
        results = Counter()
        for geff, fmt in formats.items():
            if fmt is None:
                results["already recorded"] += 1
                continue
            record_format(geff, fmt)
            results["recorded"] += 1
        if skipped:
            # never edited: still in their original format, which a session keeps
            results["no history (left as is)"] = skipped
        print(dict(results))
    elif args.command == "republish":
        geffs = [g for g in find_geffs(store) if history_path(g).is_file()]
        for geff in geffs:  # all or nothing
            _require_format(geff)
        for geff in geffs:
            revision = republish(geff)
            print(f"{geff.parent.name}: republished revision {revision}")
    elif args.command == "conform":
        rows = conform_store(
            store, args.like.resolve(), drop_extra=args.drop_extra, remove=args.remove
        )
        dropped = Counter(p for row in rows for p in row["dropped"])
        removed = Counter(n for row in rows for n in row["removed"])
        print(
            f"conformed {len(rows)} datasets "
            f"({sum(r['history'] for r in rows)} with history); "
            f"dropped properties {dict(dropped)}; removed files {dict(removed)}"
        )
    else:
        rows = export(
            args.store, args.out, drop_extra=args.drop_extra, report=args.report
        )
        print(f"exported {len(rows)} datasets to {args.out}; all passed the gate")
    return 0


if __name__ == "__main__":
    sys.exit(main())
