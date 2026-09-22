"""Capture and restore the full editing graph, including deleted candidates."""

from __future__ import annotations

import tracksdata as td
from funtracks.data_model import SolutionTracks
from funtracks.features import FeatureDict

from .codec import decode, decode_action, decode_dtype, encode_action, encode_dtype


def capture(tracks, encoder, previous=None, dirty_nodes=None):
    graph = tracks.graph_full
    schemas = {}
    for kind, source in (
        ("node", graph._node_attr_schemas()),
        ("edge", graph._edge_attr_schemas()),
    ):
        schemas[kind] = {
            key: [encode_dtype(schema.dtype), schema.default_value]
            for key, schema in source.items()
            if key not in {"node_id", "edge_id", "source_id", "target_id"}
        }
    header = {
        "schemas": schemas,
        "metadata": dict(graph.metadata),
        "features": tracks.features.dump_json(),
        "scale": tracks.scale,
        "ndim": tracks.ndim,
        "node_id_counter": tracks.node_id_counter,
        "max_tracklet_id": tracks.track_annotator.max_tracklet_id,
        "max_lineage_id": tracks.track_annotator.max_lineage_id,
    }
    from motile_tracker.motile.backend.motile_run import MotileRun

    if isinstance(tracks, MotileRun):
        header["run"] = {
            key: getattr(tracks, key)
            for key in ("run_name", "input_points", "gaps", "status", "time")
        }
        params = tracks.solver_params
        header["run"]["solver_params"] = params.model_dump() if params else None
    encoded_header = encoder.encode(header)
    if previous:
        before = dict(previous["header"]["items"])["schemas"]
        after = dict(encoded_header["items"])["schemas"]
        if before != after:
            dirty_nodes = None
    old_nodes = previous["nodes"] if previous else {}
    ids = graph.node_ids()
    nodes = {str(n): old_nodes[str(n)] for n in ids if str(n) in old_nodes}
    changed = (
        ids
        if dirty_nodes is None
        else [n for n in ids if n in dirty_nodes or str(n) not in old_nodes]
    )
    if changed:
        import polars as pl

        rows = graph.node_attrs().filter(pl.col("node_id").is_in(changed))
        for attrs in rows.iter_rows(named=True):
            nodes[str(int(attrs["node_id"]))] = encoder.encode(attrs)
    edges = {}
    for attrs in graph.edge_attrs().iter_rows(named=True):
        source, target = int(attrs["source_id"]), int(attrs["target_id"])
        attrs.pop("edge_id", None)
        edges[f"{source},{target}"] = encoder.encode(attrs)
    return {"header": encoded_header, "nodes": nodes, "edges": edges}


def restore(state, get_blob, into=None):
    header = decode(state["header"], get_blob)
    graph = td.graph.IndexedRXGraph()
    for key, (dtype, default) in header["schemas"]["node"].items():
        if key in graph.node_attr_keys():
            continue
        graph.add_node_attr_key(key, decode_dtype(dtype), default_value=default)
    for key, (dtype, default) in header["schemas"]["edge"].items():
        graph.add_edge_attr_key(key, decode_dtype(dtype), default_value=default)
    nodes, ids = [], []
    for key, encoded in state["nodes"].items():
        attrs = decode(encoded, get_blob)
        attrs.pop("node_id", None)
        nodes.append(attrs)
        ids.append(int(key))
    graph.bulk_add_nodes(nodes=nodes, indices=ids)
    graph.bulk_add_edges([decode(v, get_blob) for v in state["edges"].values()])
    graph._update_metadata(**header["metadata"])
    features = FeatureDict.from_json(header["features"])
    if "run" in header:
        from motile_tracker.motile.backend.motile_run import MotileRun
        from motile_tracker.motile.backend.solver_params import SolverParams

        run = header["run"]
        if run["solver_params"] is not None:
            run["solver_params"] = SolverParams(**run["solver_params"])
        tracks = into if into is not None else object.__new__(MotileRun)
        MotileRun.__init__(
            tracks,
            graph,
            scale=header["scale"],
            ndim=header["ndim"],
            _features=features,
            **run,
        )
    else:
        tracks = into if into is not None else object.__new__(SolutionTracks)
        SolutionTracks.__init__(
            tracks, graph, scale=header["scale"], ndim=header["ndim"], features=features
        )
    tracks.node_id_counter = header["node_id_counter"]
    tracks.track_annotator.max_tracklet_id = header["max_tracklet_id"]
    tracks.track_annotator.max_lineage_id = header["max_lineage_id"]
    return tracks


def capture_history(history, encoder, cache=None):
    cache = {} if cache is None else cache

    def encode_once(action):
        if action not in cache:
            cache[action] = encode_action(action, encoder)
        return cache[action]

    return {
        key: [encode_once(a) for a in getattr(history, key)]
        for key in ("undo_stack", "redo_stack")
    }


def restore_history(history, data, tracks, get_blob):
    for key in ("undo_stack", "redo_stack"):
        setattr(history, key, [decode_action(a, tracks, get_blob) for a in data[key]])
