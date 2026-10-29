"""Versioned, non-executable encoding for graph data and reversible actions."""

from __future__ import annotations

import hashlib
import json
import math
import zlib
from datetime import date, datetime

import numpy as np
import polars as pl
from funtracks import actions
from tracksdata.nodes import Mask


def dumps(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


class Encoder:
    def __init__(self):
        self.blobs = {}

    def encode(self, value):
        if isinstance(value, pl.Series):
            return {
                "type": "series",
                "name": value.name,
                "dtype": encode_dtype(value.dtype),
                "values": self.encode(value.to_list()),
            }
        if isinstance(value, np.ndarray):
            if value.dtype.hasobject or value.dtype.fields:
                raise TypeError(
                    "Object/structured arrays cannot be saved in edit history"
                )
            raw = value.tobytes(order="C")
            key = hashlib.sha256(raw).hexdigest()
            self.blobs.setdefault(key, zlib.compress(raw))
            return {
                "type": "array",
                "dtype": value.dtype.str,
                "shape": list(value.shape),
                "blob": key,
            }
        if isinstance(value, Mask):
            return {
                "type": "mask",
                "mask": self.encode(value.mask),
                "bbox": self.encode(value.bbox),
            }
        if isinstance(value, np.generic):
            return self.encode(value.item())
        if isinstance(value, (datetime, date)):
            return {"type": type(value).__name__, "value": value.isoformat()}
        if isinstance(value, float) and not math.isfinite(value):
            return {"type": "float", "value": str(value)}
        if isinstance(value, tuple):
            return {"type": "tuple", "items": [self.encode(v) for v in value]}
        if isinstance(value, list):
            return [self.encode(v) for v in value]
        if isinstance(value, dict):
            # Wrap dictionaries, so a user attribute named 'type' is unambiguous.
            return {
                "type": "dict",
                "items": [[self.encode(k), self.encode(v)] for k, v in value.items()],
            }
        if value is None or isinstance(value, (str, bool, int, float)):
            return value
        if hasattr(value, "model_dump"):
            return self.encode(value.model_dump(mode="json"))
        raise TypeError(f"Unsupported edit-history value: {type(value).__name__}")


def decode(value, get_blob):
    if isinstance(value, list):
        return [decode(v, get_blob) for v in value]
    if not isinstance(value, dict):
        return value
    kind = value["type"]
    if kind == "series":
        return pl.Series(
            value["name"],
            decode(value["values"], get_blob),
            dtype=decode_dtype(value["dtype"]),
        )
    if kind == "dict":
        return {decode(k, get_blob): decode(v, get_blob) for k, v in value["items"]}
    if kind == "tuple":
        return tuple(decode(v, get_blob) for v in value["items"])
    if kind == "array":
        dtype = np.dtype(value["dtype"])
        if dtype.hasobject or dtype.fields:
            raise ValueError("Unsupported history array dtype")
        raw = zlib.decompress(get_blob(value["blob"]))
        if hashlib.sha256(raw).hexdigest() != value["blob"]:
            raise ValueError("Corrupted edit-history array")
        return np.frombuffer(raw, dtype=dtype).reshape(value["shape"]).copy()
    if kind == "mask":
        return Mask(
            decode(value["mask"], get_blob), bbox=decode(value["bbox"], get_blob)
        )
    if kind == "float":
        return float(value["value"])
    if kind in ("datetime", "date"):
        return {"datetime": datetime, "date": date}[kind].fromisoformat(value["value"])
    raise ValueError(f"Unsupported edit-history type: {kind}")


_DTYPES = {
    name: getattr(pl, name)
    for name in (
        "Boolean",
        "Int8",
        "Int16",
        "Int32",
        "Int64",
        "UInt8",
        "UInt16",
        "UInt32",
        "UInt64",
        "Float32",
        "Float64",
        "String",
        "Binary",
        "Object",
        "Null",
        "Date",
        "Time",
    )
}


def encode_dtype(dtype):
    if isinstance(dtype, pl.Array):
        return ["Array", encode_dtype(dtype.inner), list(dtype.shape)]
    if isinstance(dtype, pl.List):
        return ["List", encode_dtype(dtype.inner)]
    if isinstance(dtype, pl.Datetime):
        return ["Datetime", dtype.time_unit, dtype.time_zone]
    if isinstance(dtype, pl.Duration):
        return ["Duration", dtype.time_unit]
    if str(dtype) in _DTYPES:
        return [str(dtype)]
    raise TypeError(f"Unsupported graph schema dtype: {dtype}")


def decode_dtype(data):
    if data[0] == "Array":
        return pl.Array(decode_dtype(data[1]), shape=tuple(data[2]))
    if data[0] == "List":
        return pl.List(decode_dtype(data[1]))
    if data[0] == "Datetime":
        return pl.Datetime(data[1], data[2])
    if data[0] == "Duration":
        return pl.Duration(data[1])
    return _DTYPES[data[0]]


_ACTION_FIELDS = {
    "AddNode": ("node", "attributes"),
    "DeleteNode": ("node", "attributes"),
    "AddEdge": ("edge", "attributes"),
    "DeleteEdge": ("edge", "attributes"),
    "UpdateNodeSeg": ("node", "mask", "added", "mask_key"),
    "UpdateNodeAttrs": ("node", "prev_attrs", "new_attrs"),
    "UpdateTrackIDs": (
        "start_node",
        "old_tracklet_id",
        "new_tracklet_id",
        "old_lineage_id",
        "new_lineage_id",
    ),
}


def encode_action(action, encoder):
    if isinstance(action, actions.ActionGroup):
        return {
            "kind": "group",
            "label": type(action).__name__,
            "actions": [encode_action(a, encoder) for a in action.actions],
        }
    name = type(action).__name__
    if name == "FeatureChange":
        from .features import FeatureChange

        if type(action) is FeatureChange:
            return {"kind": name, "data": encoder.encode(action.saved)}
    if name not in _ACTION_FIELDS or type(action) is not getattr(actions, name):
        raise TypeError(f"Unsupported history action: {name}")
    return {
        "kind": name,
        "data": encoder.encode(
            {key: getattr(action, key) for key in _ACTION_FIELDS[name]}
        ),
    }


def decode_action(record, tracks, get_blob):
    name = record["kind"]
    if name == "group":
        # ActionGroup does not apply its children; only their constructors do.
        return actions.ActionGroup(
            tracks, [decode_action(a, tracks, get_blob) for a in record["actions"]]
        )
    if name == "FeatureChange":
        from .features import FeatureChange

        action = object.__new__(FeatureChange)
        action.tracks = tracks
        action.saved = decode(record["data"], get_blob)
        return action
    if name not in _ACTION_FIELDS:
        raise ValueError(f"Unsupported history action: {name}")
    fields = decode(record["data"], get_blob)
    if set(fields) != set(_ACTION_FIELDS[name]):
        raise ValueError(f"Invalid fields for history action {name}")
    # Never call a primitive's __init__: that would perform the edit on load.
    action = object.__new__(getattr(actions, name))
    action.tracks = tracks
    action.__dict__.update(fields)
    return action
