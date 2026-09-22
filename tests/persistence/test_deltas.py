"""Storage size and exact historical replay, including the original file format."""

import copy
import hashlib
import json
import sqlite3
import zlib

import pytest

from motile_tracker.persistence.codec import Encoder, dumps
from motile_tracker.persistence.deltas import apply_changes, encode_changes
from motile_tracker.persistence.store import Store, history_path

EMPTY_HISTORY = {"undo_stack": [], "redo_stack": []}


def state(attributes, *, edges=None):
    encoder = Encoder()
    return {
        "header": encoder.encode({"ndim": 3}),
        "nodes": {"1": encoder.encode(attributes)} if attributes is not None else {},
        "edges": {"1,2": encoder.encode(edges)} if edges is not None else {},
    }


def save(store, value):
    return store.commit(value, EMPTY_HISTORY, {}, "test edit")


def legacy_payload(before, after):
    """Version 1 stored JSON strings for whole records, with None for absence."""
    changes = {}
    for kind in ("header", "nodes", "edges"):
        old = {"header": before["header"]} if kind == "header" else before[kind]
        new = {"header": after["header"]} if kind == "header" else after[kind]
        changes[kind] = {
            key: [
                dumps(old[key]) if key in old else None,
                dumps(new[key]) if key in new else None,
            ]
            for key in old.keys() | new.keys()
            if old.get(key) != new.get(key)
        }
    return zlib.compress(dumps(changes).encode())


def make_legacy(path):
    first = state({"node_id": 1, "tracklet_id": 1, "note": None})
    second = state({"node_id": 1, "tracklet_id": 2, "note": None})
    # Table layout is shared by versions 1 and 2. Replace the generated revision
    # bytes explicitly, so the compatibility fixture uses the original encoding.
    with Store(path, create=True) as store:
        save(store, first)
        save(store, second)
        initial = {"header": None, "nodes": {}, "edges": {}}
        payloads = [legacy_payload(initial, first), legacy_payload(first, second)]
        for revision, payload in enumerate(payloads):
            store.connection.execute(
                "UPDATE revisions SET changes=? WHERE id=?", (payload, revision)
            )
        store.connection.execute("PRAGMA user_version=1")
    return first, second, payloads


def test_small_change_does_not_repeat_large_unchanged_record(tmp_path):
    attrs = {
        "tracklet_id": 1,
        **{
            f"property_{n}": hashlib.sha256(str(n).encode()).hexdigest()
            for n in range(128)
        },
    }
    before = state(attrs)
    after = state({**attrs, "tracklet_id": 2})
    with Store(tmp_path / "tracks.geff", create=True) as store:
        save(store, before)
        save(store, after)
        size = store.connection.execute(
            "SELECT length(changes) FROM revisions WHERE id=1"
        ).fetchone()[0]
        assert size < 300  # A single changed integer, independent of the other data.
        assert store.state_at(0) == before
        assert store.state_at(1) == after
        assert store.read()[1] == after


@pytest.mark.parametrize(
    "before,after",
    [
        ({"note": None, "remove": None}, {"note": None, "added": None}),
        ({"value": None}, {"value": []}),
        ({"value": []}, {}),
        ({"value": True}, {"value": 1}),
        ({"value": 1}, {"value": 1.0}),
        ({"value": -0.0}, {"value": 0.0}),
        ({"value": [True, 1]}, {"value": [1, 1.0]}),
        ({}, {"value": None}),
        ({"a": 1, "b": 2}, {"b": 2, "a": 1}),
        ({"a": 1}, {"a": 1, "z": 3, "b": 2}),
        (None, {"node_id": 1}),
        ({"node_id": 1}, None),
    ],
)
def test_replay_preserves_missing_null_order_and_record_lifecycle(
    tmp_path, before, after
):
    first, second = state(before), state(after)
    with Store(tmp_path / "tracks.geff", create=True) as store:
        save(store, first)
        save(store, second)
        save(store, first)
        assert dumps(store.state_at(0)) == dumps(first)
        assert dumps(store.state_at(1)) == dumps(second)
        assert dumps(store.state_at(2)) == dumps(first)


def test_edges_and_header_replay(tmp_path):
    first = state({"node_id": 1}, edges={"solution": True, "weight": None})
    second = state({"node_id": 1}, edges={"solution": False, "weight": 0.5})
    second["header"] = Encoder().encode({"ndim": 3, "custom": None})
    third = state({"node_id": 1})
    with Store(tmp_path / "tracks.geff", create=True) as store:
        for value in (first, second, third):
            save(store, value)
        assert [store.state_at(i) for i in range(3)] == [first, second, third]


def test_inverse_delta_restores_deleted_attribute_position():
    before = state({"a": 1, "b": 2, "c": 3})
    after = state({"a": 1, "c": 3})
    changes = encode_changes(json.loads(zlib.decompress(legacy_payload(before, after))))
    inverse = copy.deepcopy(changes)
    for records in inverse["changes"].values():
        for patch in records.values():
            if "record" in patch:
                patch["record"].reverse()
            else:
                for values in patch["attributes"].values():
                    values.reverse()
    restored = copy.deepcopy(after)
    apply_changes(restored, inverse)
    assert dumps(restored) == dumps(before)


def test_legacy_history_is_readable_and_retained_when_new_edits_are_appended(tmp_path):
    path = tmp_path / "tracks.geff"
    first, second, payloads = make_legacy(path)
    with Store(path, readonly=True) as store:
        assert store.connection.execute("PRAGMA user_version").fetchone()[0] == 1
        assert store.state_at(0) == first
        assert store.state_at(1) == second
    third = state({"node_id": 1, "tracklet_id": 3, "note": None})
    with Store(path) as store:
        save(store, third)
        assert store.connection.execute("PRAGMA user_version").fetchone()[0] == 2
        assert [
            row[0]
            for row in store.connection.execute(
                "SELECT changes FROM revisions WHERE id<2 ORDER BY id"
            )
        ] == payloads
    with Store(path, readonly=True) as store:
        assert [store.state_at(i) for i in range(3)] == [first, second, third]
        assert store.read()[1] == third


def test_failed_first_compact_write_keeps_legacy_version_and_state(tmp_path):
    path = tmp_path / "tracks.geff"
    first, second, payloads = make_legacy(path)
    with Store(path) as store:
        store.connection.execute("""CREATE TEMP TRIGGER fail_revision BEFORE INSERT ON revisions
            BEGIN SELECT RAISE(ABORT, 'write failed'); END""")
        with pytest.raises(sqlite3.IntegrityError, match="write failed"):
            save(store, first)
        assert store.connection.execute("PRAGMA user_version").fetchone()[0] == 1
        assert store.read()[1] == second
        assert [
            row[0]
            for row in store.connection.execute(
                "SELECT changes FROM revisions ORDER BY id"
            )
        ] == payloads


def test_replay_rejects_patch_applied_to_wrong_prior_state(tmp_path):
    path = tmp_path / "tracks.geff"
    with Store(path, create=True) as store:
        save(store, state({"tracklet_id": 1}))
        save(store, state({"tracklet_id": 2}))
        payload = store.connection.execute(
            "SELECT changes FROM revisions WHERE id=1"
        ).fetchone()[0]
        # Repeating the same transition has an invalid before-value (now 2).
        store.connection.execute(
            "UPDATE revisions SET changes=? WHERE id=0", (payload,)
        )
        with pytest.raises(ValueError, match="prior|before|missing"):
            store.state_at(1)


def test_unknown_database_version_is_not_downgraded(tmp_path):
    path = tmp_path / "tracks.geff"
    with Store(path, create=True) as store:
        save(store, state({"tracklet_id": 1}))
        store.connection.execute("PRAGMA user_version=999")
    with pytest.raises(ValueError, match="999"):
        Store(path)
    with sqlite3.connect(history_path(path)) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 999
