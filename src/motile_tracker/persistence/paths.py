"""Writable destinations and portable references for local editing copies."""

from __future__ import annotations

import os
import shutil
import sqlite3
from contextlib import closing
from pathlib import Path
from uuid import uuid4

import zarr

from .projection import check_external_change, rebase_references
from .session import EditSession, has_history
from .store import Store, history_path


def working_copy(source, directory):
    source = Path(source).resolve()
    destination = (
        Path(directory) / f"{source.stem}_working_copy_{uuid4().hex[:12]}.geff"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    if has_history(source):
        with Store(source, readonly=True) as original:
            check_external_change(original)
    shutil.copytree(source, destination, copy_function=shutil.copyfile)
    # copytree retains directory modes, including a read-only source's modes.
    for parent, _, _ in os.walk(destination):
        Path(parent).chmod(Path(parent).stat().st_mode | 0o700)
    if has_history(source):
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
            # The copied root is disposable; rebuild it from the database snapshot.
            (destination / "edit_history" / "publishing").write_text("working copy")
    else:
        root = zarr.open_group(destination, mode="a")
        root.attrs.update(rebase_references(dict(root.attrs), source, destination))
    return destination


def open_session(source, working_directory):
    try:
        return EditSession.open(source)
    except PermissionError:
        return EditSession.open(working_copy(source, working_directory))


def create_session(tracks, source, working_directory):
    try:
        return EditSession.create(tracks, source)
    except PermissionError:
        return EditSession.create(tracks, working_copy(source, working_directory))
