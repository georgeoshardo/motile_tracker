"""Reversible record/attribute changes, with support for version 1 history."""

from __future__ import annotations

import json

from .codec import dumps

FORMAT_VERSION = 2


def _fields(record):
    if not isinstance(record, dict) or set(record) != {"type", "items"}:
        return None
    if record["type"] != "dict":
        return None
    pairs = record["items"]
    if not isinstance(pairs, list) or any(
        not isinstance(pair, list) or len(pair) != 2 or not isinstance(pair[0], str)
        for pair in pairs
    ):
        return None
    fields = dict(pairs)
    return fields if len(fields) == len(pairs) else None


def _change(before, after):
    """Absent attributes use []; a present null uses [None]."""
    old = json.loads(before) if before is not None else None
    new = json.loads(after) if after is not None else None
    old_fields, new_fields = _fields(old), _fields(new)
    if old_fields is not None and new_fields is not None:
        # Retain the encoded dictionary's order as well as its values. JSON sorts
        # patch keys, so additions will be appended in sorted order during replay.
        replay_order = [key for key in old_fields if key in new_fields]
        replay_order += sorted(new_fields.keys() - old_fields.keys())
        inverse_order = [key for key in new_fields if key in old_fields]
        inverse_order += sorted(old_fields.keys() - new_fields.keys())
        if replay_order == list(new_fields) and inverse_order == list(old_fields):
            return {
                "attributes": {
                    key: [
                        [old_fields[key]] if key in old_fields else [],
                        [new_fields[key]] if key in new_fields else [],
                    ]
                    for key in old_fields.keys() | new_fields.keys()
                    if (key in old_fields) != (key in new_fields)
                    or dumps(old_fields.get(key)) != dumps(new_fields.get(key))
                }
            }
    # New/deleted records, non-dictionaries and order changes retain whole records.
    return {"record": [old, new]}


def encode_changes(changes):
    return {
        "format": FORMAT_VERSION,
        "changes": {
            kind: {
                key: _change(before, after) for key, (before, after) in records.items()
            }
            for kind, records in changes.items()
        },
    }


def _apply(current, change):
    if set(change) == {"record"}:
        before, after = change["record"]
        if dumps(current) != dumps(before):
            raise ValueError("History record does not match its prior state")
        return after
    if set(change) != {"attributes"}:
        raise ValueError("Unsupported history change")
    fields = _fields(current)
    if fields is None:
        raise ValueError("Attribute history is missing its prior record")
    for key, (before, after) in sorted(change["attributes"].items()):
        if (
            not isinstance(before, list)
            or not isinstance(after, list)
            or len(before) > 1
            or len(after) > 1
        ):
            raise ValueError("Invalid history attribute presence")
        previous = [fields[key]] if key in fields else []
        if dumps(previous) != dumps(before):
            raise ValueError(
                f"History attribute {key!r} does not match its before value"
            )
        if after:
            fields[key] = after[0]
        else:
            fields.pop(key, None)
    return {"type": "dict", "items": [[key, value] for key, value in fields.items()]}


def apply_changes(state, payload):
    """Apply a decompressed revision to an encoded full state, in place."""
    version = payload.get("format", 1)
    if version not in (1, FORMAT_VERSION):
        raise ValueError(f"Unsupported history change format {version}")
    groups = payload if version == 1 else payload["changes"]
    for kind, changes in groups.items():
        if kind not in ("header", "nodes", "edges"):
            raise ValueError(f"Unsupported history record kind {kind!r}")
        for key, change in changes.items():
            current = state["header"] if kind == "header" else state[kind].get(key)
            if version == 1:
                before, after = change
                change = {
                    "record": [
                        json.loads(before) if before is not None else None,
                        json.loads(after) if after is not None else None,
                    ]
                }
            after = _apply(current, change)
            if kind == "header":
                state["header"] = after
            elif after is None:
                state[kind].pop(key, None)
            else:
                state[kind][key] = after
