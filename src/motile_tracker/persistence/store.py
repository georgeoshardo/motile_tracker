"""Transactional journal. All payloads needed for recovery live in this database."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import zlib
from contextlib import suppress
from datetime import UTC, datetime
from importlib.metadata import version as package_version
from pathlib import Path
from uuid import uuid4

import zarr

from .codec import dumps

HISTORY_GROUP = "edit_history"
DB_NAME = "history.sqlite"
SCHEMA_VERSION = 1


def history_path(path):
    return Path(path) / HISTORY_GROUP / DB_NAME


class Store:
    def __init__(self, path, *, create=False, readonly=False):
        self.path = Path(path).resolve()
        self.lock = None
        self.connection = None
        self._action_records = {}
        db = history_path(self.path)
        owns_new_database = False
        try:
            if not readonly:
                db.parent.mkdir(parents=True, exist_ok=True)
                self.lock = open(db.parent / "writer.lock", "a+b")  # noqa: SIM115 -- held until close()
                try:
                    if os.name == "nt":
                        import msvcrt

                        self.lock.seek(0)
                        self.lock.write(b"0")
                        self.lock.flush()
                        self.lock.seek(0)
                        msvcrt.locking(self.lock.fileno(), msvcrt.LK_NBLCK, 1)
                    else:
                        import fcntl

                        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except OSError as exc:
                    raise RuntimeError(
                        f"Tracks already open for editing: {self.path}"
                    ) from exc
            if create and db.exists():
                raise FileExistsError(f"History already exists: {db}")
            if not create and not db.exists():
                raise FileNotFoundError(db)
            owns_new_database = create
            self.connection = sqlite3.connect(
                db.as_uri() + ("?mode=ro" if readonly else "?mode=rwc"),
                uri=True,
                timeout=10,
                isolation_level=None,
            )
            if not readonly:
                self.connection.execute("PRAGMA journal_mode=DELETE")
                self.connection.execute("PRAGMA synchronous=FULL")
                self.connection.execute("PRAGMA fullfsync=ON")
            if create:
                # A real Zarr group survives GEFF's removal of its graph members.
                root = zarr.open_group(
                    self.path,
                    mode="a",
                    zarr_format=(None if (self.path / "zarr.json").exists() else 2),
                )
                root.require_group(HISTORY_GROUP)
                self.connection.executescript("""
                    BEGIN IMMEDIATE;
                    CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
                    CREATE TABLE blobs(key TEXT PRIMARY KEY, data BLOB NOT NULL);
                    CREATE TABLE actions(key TEXT PRIMARY KEY, data TEXT NOT NULL);
                    CREATE TABLE current(kind TEXT, key TEXT, data TEXT NOT NULL,
                                         PRIMARY KEY(kind, key));
                    CREATE TABLE sessions(id TEXT PRIMARY KEY, versions TEXT NOT NULL);
                    CREATE TABLE revisions(id INTEGER PRIMARY KEY, time TEXT NOT NULL,
                                           session TEXT NOT NULL,
                                           operation TEXT NOT NULL, changes BLOB NOT NULL,
                                           history TEXT NOT NULL);
                    PRAGMA user_version=1;
                    COMMIT;
                """)
            version = self.connection.execute("PRAGMA user_version").fetchone()[0]
            if version != SCHEMA_VERSION:
                raise ValueError(f"Unsupported edit-history version {version}")
            if not readonly:
                self.session_id = uuid4().hex
                self.connection.execute(
                    "INSERT INTO sessions VALUES (?,?)",
                    (
                        self.session_id,
                        dumps(
                            {
                                name: package_version(name)
                                for name in (
                                    "motile-tracker",
                                    "funtracks",
                                    "tracksdata",
                                    "geff",
                                    "zarr",
                                )
                            }
                        ),
                    ),
                )
        except BaseException:
            self.close()
            if owns_new_database:
                with suppress(OSError):
                    db.unlink(missing_ok=True)
            raise

    def get_blob(self, key):
        row = self.connection.execute(
            "SELECT data FROM blobs WHERE key=?", (key,)
        ).fetchone()
        if row is None:
            raise ValueError(f"Missing history payload {key}")
        return row[0]

    def metadata(self, key, default=None):
        row = self.connection.execute(
            "SELECT value FROM metadata WHERE key=?", (key,)
        ).fetchone()
        return json.loads(row[0]) if row else default

    def set_metadata(self, key, value):
        self.connection.execute(
            "INSERT OR REPLACE INTO metadata VALUES (?,?)", (key, dumps(value))
        )

    def commit(self, state, history, blobs, operation):
        conn = self.connection
        conn.execute("BEGIN IMMEDIATE")
        try:
            revision = conn.execute(
                "SELECT COALESCE(MAX(id),-1)+1 FROM revisions"
            ).fetchone()[0]
            changes = {}
            for kind, values in (
                ("header", {"header": state["header"]}),
                ("nodes", state["nodes"]),
                ("edges", state["edges"]),
            ):
                old = dict(
                    conn.execute("SELECT key,data FROM current WHERE kind=?", (kind,))
                )
                new = {key: dumps(value) for key, value in values.items()}
                changed = {
                    key: [old.get(key), new.get(key)]
                    for key in old.keys() | new.keys()
                    if old.get(key) != new.get(key)
                }
                changes[kind] = changed
                for key, (_, value) in changed.items():
                    if value is None:
                        conn.execute(
                            "DELETE FROM current WHERE kind=? AND key=?", (kind, key)
                        )
                    else:
                        conn.execute(
                            "INSERT OR REPLACE INTO current VALUES (?,?,?)",
                            (kind, key, value),
                        )
            conn.executemany("INSERT OR IGNORE INTO blobs VALUES (?,?)", blobs.items())
            references = {}
            new_action_records = {}
            for stack, actions in history.items():
                references[stack] = []
                for action in actions:
                    cached = self._action_records.get(id(action))
                    if cached is not None and cached[0] is action:
                        key = cached[1]
                    else:
                        data = dumps(action)
                        key = hashlib.sha256(data.encode()).hexdigest()
                        conn.execute(
                            "INSERT OR IGNORE INTO actions VALUES (?,?)", (key, data)
                        )
                        new_action_records[id(action)] = (action, key)
                    references[stack].append(key)
            previous = self.metadata("current_history", {})
            history_delta = {}
            for stack, entries in references.items():
                common = 0
                for old, new in zip(previous.get(stack, []), entries, strict=False):
                    if old != new:
                        break
                    common += 1
                history_delta[stack] = {"keep": common, "append": entries[common:]}
            self.set_metadata("current_history", references)
            conn.execute(
                "INSERT INTO revisions VALUES (?,?,?,?,?,?)",
                (
                    revision,
                    datetime.now(UTC).isoformat(),
                    self.session_id,
                    operation,
                    zlib.compress(dumps(changes).encode()),
                    dumps(history_delta),
                ),
            )
            conn.execute("COMMIT")
            self._action_records.update(new_action_records)
            return revision
        except BaseException:
            conn.execute("ROLLBACK")
            raise

    def read(self, *, include_history=True):
        # One read transaction gives the background projector a coherent revision.
        conn = self.connection
        conn.execute("BEGIN")
        try:
            row = conn.execute(
                "SELECT id,history FROM revisions ORDER BY id DESC LIMIT 1"
            ).fetchone()
            if row is None:
                raise ValueError("Editing history has no committed initial state")
            state = {"nodes": {}, "edges": {}}
            for kind, key, value in conn.execute("SELECT kind,key,data FROM current"):
                if kind == "header":
                    state["header"] = json.loads(value)
                else:
                    state[kind][key] = json.loads(value)
            # Blobs are immutable and retained forever; they remain valid after COMMIT.
            if not include_history:
                return row[0], state, None
            references = self.metadata("current_history")
            actions = {
                key: json.loads(data)
                for key, data in conn.execute("SELECT key,data FROM actions")
            }
            return (
                row[0],
                state,
                {
                    stack: [actions[key] for key in keys]
                    for stack, keys in references.items()
                },
            )
        finally:
            conn.execute("COMMIT")

    def state_at(self, revision):
        """Read a historical full state without mutating the current revision."""
        state = {"header": None, "nodes": {}, "edges": {}}
        found = False
        for number, payload in self.connection.execute(
            "SELECT id,changes FROM revisions WHERE id<=? ORDER BY id", (revision,)
        ):
            found = number == revision
            for kind, changes in json.loads(zlib.decompress(payload)).items():
                for key, (_, after) in changes.items():
                    if kind == "header":
                        state["header"] = json.loads(after)
                    elif after is None:
                        state[kind].pop(key, None)
                    else:
                        state[kind][key] = json.loads(after)
        if not found:
            raise ValueError(f"No edit revision {revision}")
        return state

    def close(self):
        if self.connection is not None:
            self.connection.close()
            self.connection = None
        if self.lock is not None:
            self.lock.close()
            self.lock = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
