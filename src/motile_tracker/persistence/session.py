"""Own a dataset, commit completed actions, and rebuild its GEFF projection."""

from __future__ import annotations

import shutil
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from pathlib import Path

import zarr
from funtracks.actions.action_history import ActionHistory

from .codec import Encoder
from .projection import (
    check_external_change,
    graph_signature,
    publish,
    rebase_references,
)
from .state import capture, capture_history, restore, restore_history
from .store import Store, history_path


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
        self.tracks.edit_session = self
        for signal in (
            self.tracks.graph_full.node_added,
            self.tracks.graph_full.node_removed,
            self.tracks.graph_full.node_updated,
        ):
            signal.connect(self._mark_nodes)

    def _mark_nodes(self, ids, *args):
        self._dirty_nodes.update(int(n) for n in ids)

    @classmethod
    def create(cls, tracks, path):
        path = Path(path).expanduser().resolve()
        attrs = {}
        if (path / ".zgroup").exists() or (path / "zarr.json").exists():
            attrs = dict(zarr.open_group(path, mode="r").attrs)
        store = Store(path, create=True)
        try:
            encoder = Encoder()
            state = capture(tracks, encoder)
            history = capture_history(tracks.action_history, encoder)
            store.set_metadata("original_attrs", attrs)
            store.set_metadata("graph_signature", graph_signature(path))
            store.commit(state, history, encoder.blobs, "initial")
            session = cls(tracks, store)
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
            session._request_projection()
            return session
        except BaseException:
            store.close()
            raise

    def commit(self, operation):
        try:
            if self.closed:
                raise RuntimeError("Editing session is closed")
            if self.error:
                raise RuntimeError(f"Autosave stopped: {self.error}")
            encoder = Encoder()
            # A schema change can add/remove a value on every node.
            state = capture(
                self.tracks,
                encoder,
                self._state,
                None if operation == "FeatureChange" else self._dirty_nodes,
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
        shutil.copytree(self.path, path)
        # File mtimes can differ after copying. Bind the copy to its own graph.
        with Store(path) as copy:
            attrs = rebase_references(
                copy.metadata("original_attrs", {}), self.path, path
            )
            copy.set_metadata("original_attrs", attrs)
            root = zarr.open_group(path, mode="a")
            root.attrs.update(rebase_references(dict(root.attrs), self.path, path))
            copy.set_metadata("graph_signature", graph_signature(path))

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
