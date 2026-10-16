"""Writable destinations and portable references for local editing copies."""

from __future__ import annotations

import os
import shutil
import sqlite3
import warnings
from contextlib import closing
from pathlib import Path
from uuid import uuid4

import zarr

from . import source_format as sf
from .projection import check_external_change, rebase_references
from .session import EditSession, PendingSession, has_history
from .store import Store, history_path


def _make_writable(root):
    # copytree retains modes, including a read-only source's modes.
    for parent, _, files in os.walk(root):
        Path(parent).chmod(Path(parent).stat().st_mode | 0o700)
        for name in files:
            file = Path(parent) / name
            file.chmod(file.stat().st_mode | 0o600)


def _copy_labels(labels, destination):
    """Copy a dataset's label array next to its working copy."""
    target = destination.parent / f"{destination.stem}_{labels.name}"
    shutil.copytree(labels, target, copy_function=shutil.copyfile)
    _make_writable(target)
    return target


def working_copy(source, directory):
    """A writable copy of a dataset, for editing without touching the original.

    The label array a GEFF refers to is copied too, so that publishing the edited
    segmentation never writes into the original dataset.
    """
    source = Path(source).resolve()
    destination = (
        Path(directory) / f"{source.stem}_working_copy_{uuid4().hex[:12]}.geff"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    if has_history(source):
        with Store(source, readonly=True) as original:
            check_external_change(original)
            fmt = original.metadata("source_format")
    else:
        fmt = None
    shutil.copytree(source, destination, copy_function=shutil.copyfile)
    _make_writable(destination)
    if has_history(source):
        labels = sf.labels_path(source, fmt)
        copied_labels = _copy_labels(labels, destination) if labels else None
        # Obtain one database snapshot even if another reader had the source open.
        with (
            Store(source, readonly=True) as original,
            closing(sqlite3.connect(history_path(destination))) as target,
        ):
            original.connection.backup(target)
        with Store(destination) as copy:
            attrs = rebase_references(
                copy.metadata("original_attrs", {}), source, destination
            )
            copy.set_metadata("original_attrs", attrs)
            if fmt is not None:
                copy.set_metadata(
                    "source_format",
                    sf.rebase(fmt, source, destination, labels_target=copied_labels),
                )
            # The copied root is disposable; rebuild it from the database snapshot.
            (destination / "edit_history" / "publishing").write_text("working copy")
    else:
        root = zarr.open_group(destination, mode="a")
        attrs = rebase_references(dict(root.attrs), source, destination)
        original_labels = sf._labels_object(dict(root.attrs).get("geff") or {})
        labels_obj = sf._labels_object(attrs.get("geff") or {})
        if original_labels is not None and original_labels.get("path"):
            labels = (source / original_labels["path"]).resolve()
            if labels.exists():
                copied = _copy_labels(labels, destination)
                labels_obj["path"] = os.path.relpath(copied, destination)
        if (destination / ".zattrs").is_file():
            style = sf._json_style((source / ".zattrs").read_text())
            (destination / ".zattrs").write_text(sf._dumps(attrs, style))
        else:
            root.attrs.update(attrs)
    return destination


def open_session(source, working_directory):
    try:
        return EditSession.open(source)
    except PermissionError:
        return EditSession.open(working_copy(source, working_directory))


def create_session(tracks, source, working_directory):
    """Start editing tracks saved at `source`.

    An existing dataset is not changed until the first edit, which creates its
    history; a read-only dataset, or one whose format cannot be kept, is edited in
    a working copy so that the original is never rewritten. New results (nothing
    at `source` yet) are saved straight away.
    """
    source = Path(source).expanduser().resolve()
    if not (source / "nodes").is_dir():
        # A new result has no file yet: save it now, as it is.
        try:
            return EditSession.create(tracks, source)
        except PermissionError:
            return EditSession.create(tracks, working_copy(source, working_directory))
    try:
        return PendingSession(tracks, source)
    except sf.UnsupportedFormat as exc:
        warnings.warn(
            f"{source.name}: its file format cannot be kept ({exc}); edits are saved "
            "to a working copy and the original is left unchanged.",
            stacklevel=2,
        )
        copy = working_copy(source, working_directory)
        return PendingSession(tracks, copy, source_format=None)
    except PermissionError:
        return PendingSession(tracks, working_copy(source, working_directory))
