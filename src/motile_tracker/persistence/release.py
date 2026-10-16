"""Fix up and release datasets edited with this application.

Commands (``python -m motile_tracker.persistence.release <command> ...``):

``attach-format STORE --reference REFERENCE``
    For datasets whose history was started before formats were kept: record the
    format of the original dataset (REFERENCE, laid out like STORE) in each
    history, after checking that the history starts from exactly that dataset.

``republish STORE``
    Rebuild each dataset's published files from its history, in its recorded
    format (graph and label array), through the publication gate.

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
import shutil
import sys
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


def attach_format(geff, reference):
    """Record `reference`'s format in `geff`'s history (see module docs)."""
    geff, reference = Path(geff), Path(reference)
    with Store(geff) as store:
        if store.metadata("source_format") is not None:
            return "already recorded"
        tracks = _initial_tracks(store)
        # The dataset is laid out like the reference, so the reference's relative
        # paths (e.g. ../segmentation) name the dataset's own arrays.
        fmt = sf.capture(reference, tracks)
        labels = sf.labels_path(geff, fmt)
        if fmt.get("labels") and (labels is None or not labels.exists()):
            raise FileNotFoundError(
                f"{geff}: label array {fmt['labels']['path']} not found"
            )
        store.set_metadata("source_format", fmt)
        # The published files are about to be rebuilt in this format.
        store.set_metadata("projected_revision", -1)
    return "recorded"


def republish(geff):
    geff = Path(geff)
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
        if not history_path(geff).is_file():
            raise FileNotFoundError(f"{geff} has no edit history")
        with Store(geff, readonly=True) as source:
            revision, state, _ = source.read(include_history=False)
            fmt = source.metadata("source_format")
            if fmt is None:
                raise ValueError(
                    f"{geff}: no recorded format (run attach-format first)"
                )
            tracks = restore(state, source.get_blob)
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
    out = commands.add_parser("export")
    out.add_argument("store", type=Path)
    out.add_argument("out", type=Path)
    out.add_argument("--drop-extra", action="store_true")
    out.add_argument("--report", type=Path)
    args = parser.parse_args(argv)

    if args.command == "attach-format":
        geffs = find_geffs(args.store)
        reference_root = args.reference.resolve()
        results = Counter()
        for geff in geffs:
            relative = (
                geff.relative_to(args.store.resolve())
                if geff != args.store.resolve()
                else Path()
            )
            reference = reference_root / relative if relative.parts else reference_root
            results[attach_format(geff, reference)] += 1
            print(
                f"{geff.parent.name if relative.parts else geff.name}: format recorded"
            )
        print(dict(results))
    elif args.command == "republish":
        for geff in find_geffs(args.store):
            revision = republish(geff)
            print(f"{geff.parent.name}: republished revision {revision}")
    else:
        rows = export(
            args.store, args.out, drop_extra=args.drop_extra, report=args.report
        )
        print(f"exported {len(rows)} datasets to {args.out}; all passed the gate")
    return 0


if __name__ == "__main__":
    sys.exit(main())
