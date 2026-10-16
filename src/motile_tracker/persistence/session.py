"""Own a dataset, commit completed actions, and rebuild its GEFF projection."""

from __future__ import annotations

import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from pathlib import Path

import zarr
from funtracks.actions.action_history import ActionHistory

from . import source_format as sf
from .codec import Encoder
from .projection import (
    check_external_change,
    dataset_signature,
    graph_signature,
    publish,
    rebase_references,
)
from .state import capture, capture_history, restore, restore_history
from .store import Store, history_path

AUTO = "auto"


def _group_attrs(path):
    path = Path(path)
    if (path / ".zgroup").exists() or (path / "zarr.json").exists():
        return dict(zarr.open_group(path, mode="r").attrs)
    return {}


def _detect_format(tracks, path, attrs):
    """The format to keep for a dataset: recorded for a GEFF written by another
    tool, None for a new dataset or one written by this application."""
    path = Path(path)
    if not (path / "nodes").is_dir() or sf.is_native(attrs):
        return None
    return sf.capture(path, tracks)


def snapshot(tracks, path, source_format=AUTO):
    """Everything needed to start a dataset's history later, captured now."""
    path = Path(path).expanduser().resolve()
    attrs = _group_attrs(path)
    fmt = (
        _detect_format(tracks, path, attrs) if source_format == AUTO else source_format
    )
    encoder = Encoder()
    state = capture(tracks, encoder)
    history = capture_history(tracks.action_history, encoder)
    if (path / "nodes").is_dir():
        signature = dataset_signature(path, fmt) if fmt else graph_signature(path)
    else:
        signature = None
    return {
        "state": state,
        "history": history,
        "blobs": encoder.blobs,
        "attrs": attrs,
        "signature": signature,
        "format": fmt,
    }


class PersistentHistory(ActionHistory):
    def __init__(self, session, original):
        super().__init__()
        self.session = session
        self.undo_stack = original.undo_stack
        self.redo_stack = original.redo_stack

    def add_new_action(self, action):
        super().add_new_action(action)
        self.session.commit(type(action).__name__)

    def undo(self):
        try:
            changed = super().undo()
        except Exception as exc:
            self.session.rollback(exc)
            raise
        if changed:
            self.session.commit("undo")
            return True
        return False

    def redo(self):
        try:
            changed = super().redo()
        except Exception as exc:
            self.session.rollback(exc)
            raise
        if changed:
            self.session.commit("redo")
            return True
        return False


class EditSession:
    def __init__(self, tracks, store):
        self.tracks = tracks
        self.store = store
        self.path = store.path
        self.error = None
        self.closed = False
        self._dirty_nodes = set()
        self._action_cache = {}
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="geff-save"
        )
        self._future = None
        self.revision, self._state, self._history = store.read()
        self._attach()

    def _attach(self):
        self.tracks.action_history = PersistentHistory(self, self.tracks.action_history)
        self.tracks.edit_session = getattr(self, "facade", None) or self
        for signal in (
            self.tracks.graph_full.node_added,
            self.tracks.graph_full.node_removed,
            self.tracks.graph_full.node_updated,
        ):
            signal.connect(self._mark_nodes)

    def _mark_nodes(self, ids, *args):
        self._dirty_nodes.update(int(n) for n in ids)

    @classmethod
    def create(cls, tracks, path, *, source_format=AUTO, initial=None):
        """Start the history of a dataset.

        A GEFF written by another tool keeps its format (see :mod:`.source_format`):
        it is recorded here, and the files are left as they are until an edit is
        published. Raises :class:`~.source_format.UnsupportedFormat` if that format
        cannot be kept.

        Args:
            tracks: The tracks being edited.
            path: The dataset's GEFF group.
            source_format: The format to keep; detected by default.
            initial: A :func:`snapshot` taken earlier (when the dataset was opened)
                to use as the initial revision instead of the tracks as they are now.
        """
        path = Path(path).expanduser().resolve()
        if initial is None:
            initial = snapshot(tracks, path, source_format)
        elif initial["signature"] is not None:
            fmt = initial["format"]
            current = dataset_signature(path, fmt) if fmt else graph_signature(path)
            if current != initial["signature"]:
                raise ValueError(
                    "Dataset changed on disk since it was opened; reopen it before editing"
                )
        fmt = initial["format"]
        store = Store(path, create=True)
        try:
            store.set_metadata("original_attrs", initial["attrs"])
            if initial["signature"] is not None:
                store.set_metadata("graph_signature", initial["signature"])
            if fmt is not None:
                store.set_metadata("source_format", fmt)
                # The files on disk are the initial revision, in the dataset's format.
                store.set_metadata("projected_revision", 0)
            store.commit(
                initial["state"], initial["history"], initial["blobs"], "initial"
            )
            session = cls(tracks, store)
            if fmt is None:
                session._request_projection()
            return session
        except BaseException:
            store.close()
            # This call created the journal and owns it exclusively. Never let
            # an uncommitted initial state hide an otherwise readable GEFF.
            history_path(path).unlink(missing_ok=True)
            raise

    @classmethod
    def open(cls, path):
        store = Store(path)
        try:
            check_external_change(store)
            _, state, history = store.read()
            tracks = restore(state, store.get_blob)
            restore_history(tracks.action_history, history, tracks, store.get_blob)
            session = cls(tracks, store)
            # Reopening publishes only what is not yet on disk: an interrupted
            # publication, or revisions committed after the last one.
            marker = session.path / "edit_history" / "publishing"
            if (
                marker.exists()
                or store.metadata("projected_revision") != session.revision
            ):
                session._request_projection()
            return session
        except BaseException:
            store.close()
            raise

    def commit(self, operation, *, full=False):
        try:
            if self.closed:
                raise RuntimeError("Editing session is closed")
            if self.error:
                raise RuntimeError(f"Autosave stopped: {self.error}")
            encoder = Encoder()
            # A schema change can add/remove a value on every node; a first edit
            # made before the session existed was not tracked node by node.
            state = capture(
                self.tracks,
                encoder,
                self._state,
                None if (full or operation == "FeatureChange") else self._dirty_nodes,
            )
            cache = dict(self._action_cache)
            history = capture_history(self.tracks.action_history, encoder, cache)
            revision = self.store.commit(state, history, encoder.blobs, operation)
        except Exception as exc:
            self.rollback(exc)
            raise RuntimeError(f"Autosave failed; edit rolled back: {exc}") from exc
        self.revision, self._state, self._history = revision, state, history
        self._action_cache = cache
        self._dirty_nodes.clear()
        self._request_projection()

    def rollback(self, error):
        """Restore model and stacks together after an interrupted edit or undo."""
        self.error = str(error)
        restore(self._state, self.store.get_blob, into=self.tracks)
        restore_history(
            self.tracks.action_history, self._history, self.tracks, self.store.get_blob
        )
        self._attach()
        self._dirty_nodes.clear()
        self.tracks.refresh.emit(None)

    def _request_projection(self):
        # Coalesce rapid edits. The next status poll/flush schedules the newest
        # revision after this worker finishes; no live graph crosses threads.
        if self._future is None:
            self._future = self._executor.submit(publish, self.path)

    def poll(self):
        if self._future is not None and self._future.done():
            try:
                revision, signature = self._future.result()
                self.store.set_metadata("graph_signature", signature)
                self.store.set_metadata("projected_revision", revision)
                (self.path / "edit_history" / "publishing").unlink(missing_ok=True)
                self._future = None
                if revision < self.revision:
                    self._request_projection()
            except Exception as exc:  # noqa: BLE001 -- worker errors become a visible stopped state
                self.error = f"History saved, but GEFF update failed: {exc}"
                self._future = None
        return (
            "Autosave stopped"
            if self.error
            else ("Saving…" if self._future else "Saved")
        )

    def flush(self):
        while self._future is not None:
            with suppress(Exception):  # poll below reports the worker failure
                self._future.result()
            self.poll()
        if self.error:
            raise RuntimeError(self.error)

    def save_copy(self, path):
        self.flush()
        path = Path(path).expanduser().resolve()
        if path == self.path:
            return
        if path.exists():
            raise FileExistsError(f"Choose a new filename for the copy: {path}")
        fmt = self.store.metadata("source_format")
        labels = sf.labels_path(self.path, fmt)
        copied_labels = path.parent / f"{path.stem}_{labels.name}" if labels else None
        if copied_labels is not None and copied_labels.exists():
            raise FileExistsError(
                f"Choose a new filename for the copy: {copied_labels}"
            )
        shutil.copytree(self.path, path)
        if copied_labels is not None:
            shutil.copytree(labels, copied_labels)
        # File mtimes can differ after copying. Bind the copy to its own graph.
        with Store(path) as copy:
            attrs = rebase_references(
                copy.metadata("original_attrs", {}), self.path, path
            )
            copy.set_metadata("original_attrs", attrs)
            if fmt is not None:
                fmt = sf.rebase(fmt, self.path, path, labels_target=copied_labels)
                copy.set_metadata("source_format", fmt)
                (path / ".zattrs").write_text(fmt["root_text"])
            else:
                root = zarr.open_group(path, mode="a")
                root.attrs.update(rebase_references(dict(root.attrs), self.path, path))
            copy.set_metadata(
                "graph_signature",
                dataset_signature(path, fmt) if fmt else graph_signature(path),
            )

    def close(self):
        if self.closed:
            return
        try:
            self.flush()
        finally:
            self._executor.shutdown(wait=True)
            self.store.close()
            self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        already_failed = self.error is not None
        try:
            self.close()
        except RuntimeError:
            if not already_failed:
                raise


def has_history(path):
    return history_path(path).is_file()


class _PendingHistory(ActionHistory):
    """Starts the dataset's history at the first change."""

    def __init__(self, pending, original):
        super().__init__()
        self.pending = pending
        self.undo_stack = original.undo_stack
        self.redo_stack = original.redo_stack

    def add_new_action(self, action):
        super().add_new_action(action)
        self.pending._start(type(action).__name__)

    def undo(self):
        changed = super().undo()
        if changed:
            self.pending._start("undo")
        return changed

    def redo(self):
        changed = super().redo()
        if changed:
            self.pending._start("redo")
        return changed


class PendingSession:
    """An editing session whose history is created at the first edit.

    Opening a dataset must not change it, so nothing is written until an edit is
    made: the dataset's state (and, for a GEFF from another tool, its format) is
    captured in memory when it is opened, and becomes the history's initial
    revision when the first edit commits. Until then the dataset's files and
    folders are untouched.
    """

    def __init__(self, tracks, path, *, source_format=AUTO):
        self.tracks = tracks
        self.path = Path(path).expanduser().resolve()
        self.closed = False
        self._error = None
        self._session = None
        self._initial = snapshot(tracks, self.path, source_format)
        labels = sf.labels_path(self.path, self._initial["format"])
        for target in (self.path, labels):
            if (
                target is not None
                and target.exists()
                and not os.access(target, os.W_OK)
            ):
                raise PermissionError(f"Not writable: {target}")
        tracks.action_history = _PendingHistory(self, tracks.action_history)
        tracks.edit_session = self

    def _start(self, operation):
        if self._session is not None:
            return
        try:
            session = EditSession.create(
                self.tracks,
                self.path,
                source_format=self._initial["format"],
                initial=self._initial,
            )
        except Exception as exc:
            self._error = f"Could not start saving edits: {exc}"
            # The edit cannot be saved: return to the state as opened.
            restore(
                self._initial["state"], self._initial["blobs"].get, into=self.tracks
            )
            restore_history(
                self.tracks.action_history,
                self._initial["history"],
                self.tracks,
                self._initial["blobs"].get,
            )
            self.tracks.refresh.emit(None)
            raise RuntimeError(self._error) from exc
        session.facade = self
        self.tracks.edit_session = self
        self._session = session
        session.commit(operation, full=True)

    @property
    def started(self):
        return self._session is not None

    @property
    def revision(self):
        return self._session.revision if self._session else 0

    @property
    def error(self):
        return self._session.error if self._session else self._error

    def poll(self):
        if self._session is not None:
            return self._session.poll()
        return "Autosave stopped" if self._error else "No edits yet"

    def flush(self):
        if self._session is not None:
            self._session.flush()
        elif self._error:
            raise RuntimeError(self._error)

    def save_copy(self, path):
        if self._session is None:
            # A copy carries the history, so the history must exist first.
            self._session = EditSession.create(
                self.tracks,
                self.path,
                source_format=self._initial["format"],
                initial=self._initial,
            )
            self._session.facade = self
            self.tracks.edit_session = self
        self._session.save_copy(path)

    def close(self):
        if self.closed:
            return
        try:
            if self._session is not None:
                self._session.close()
        finally:
            self.closed = True

    def __getattr__(self, name):
        session = self.__dict__.get("_session")
        if session is None:
            raise AttributeError(name)
        return getattr(session, name)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
