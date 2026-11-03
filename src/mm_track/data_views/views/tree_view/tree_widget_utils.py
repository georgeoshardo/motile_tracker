from __future__ import annotations

import napari.layers
import numpy as np
import pandas as pd
import polars as pl
import tracksdata as td
from funtracks.data_model import Tracks
from tracksdata.constants import DEFAULT_ATTR_KEYS

from mm_track.data_views.node_type import NodeType


def get_tracklets(
    parent_to_children: dict[int, list[int]],
    child_to_parent: dict[int, int],
    node_ids: list[int],
    dividing_node_set: set[int],
    node_to_track_id: dict[int, int],
) -> list[set[int]]:
    """Group nodes into tracklets by BFS, cutting at division nodes.

    A tracklet is a maximal linear segment of the track graph — it does not
    cross a division point. The returned sets contain node IDs; callers are
    responsible for sorting by time if needed.

    Args:
        parent_to_children: maps each parent node_id to its list of child node_ids.
        child_to_parent: maps each child node_id to its single parent node_id.
        node_ids: all node IDs to partition.
        dividing_node_set: set of node IDs that have ≥2 children (division nodes).
        node_to_track_id: maps each node_id to its pre-computed tracklet ID.

    Returns:
        List of sets, one set of node IDs per tracklet.
    """
    visited: set[int] = set()
    tracklets: list[set[int]] = []
    for start_node in node_ids:
        if start_node in visited:
            continue
        component: set[int] = set()
        queue = [start_node]
        while queue:
            node = queue.pop()
            if node in visited:
                continue
            visited.add(node)
            component.add(node)
            pred = child_to_parent.get(node)
            if (
                pred is not None
                and pred not in visited
                and pred not in dividing_node_set
                and node_to_track_id.get(pred) == node_to_track_id.get(node)
            ):
                queue.append(pred)
            if node not in dividing_node_set:
                for succ in parent_to_children.get(node, []):
                    if succ not in visited and node_to_track_id.get(
                        succ
                    ) == node_to_track_id.get(node):
                        queue.append(succ)
        tracklets.append(component)
    return tracklets


def extract_sorted_tracks(
    tracks: Tracks,
    colormap: napari.utils.CyclicLabelColormap,
) -> pd.DataFrame | None:
    """
    Extract the information of individual tracks required for constructing the tree
    plot. Follows the same logic as the relabel_segmentation function from the Motile
    toolbox.

    Args:
        tracks (funtracks.data_model.Tracks): A tracks object containing a graph
            to be converted into a dataframe.
        colormap (napari.utils.CyclicLabelColormap): The colormap to use to
            extract the color of each node from the track ID

    Returns:
        pd.DataFrame | None: data frame with all the information needed to
        construct the tree plot. Columns are: 't', 'node_id', 'track_id',
        'color', 'x', 'y', ('z'), 'index', 'parent_id', 'parent_track_id',
        'state', 'symbol', and 'x_axis_pos' (the tracklet's lane, see
        :func:`trench_lanes`)
    """

    if tracks is None or tracks.graph is None:
        return None

    solution_nx_graph = tracks.graph
    time_key = tracks.features.time_key
    tracklet_key = tracks.features.tracklet_key

    # Batch-fetch all node attributes in one SQL query instead of per-node calls.
    node_feature_keys = [
        key
        for key, feature in tracks.features.items()
        if feature.get("feature_type") != "edge"
        and key in solution_nx_graph.node_attr_keys()
    ]
    all_keys = list(
        {DEFAULT_ATTR_KEYS.NODE_ID, time_key, tracklet_key} | set(node_feature_keys)
    )
    if len(solution_nx_graph.node_ids()) != 0:
        df_attrs = solution_nx_graph.node_attrs(attr_keys=all_keys)
    else:
        df_attrs = pl.DataFrame(schema=all_keys)
    node_ids_list = df_attrs[DEFAULT_ATTR_KEYS.NODE_ID].to_list()
    node_to_time = dict(zip(node_ids_list, df_attrs[time_key].to_list(), strict=True))
    node_to_track_id = dict(
        zip(node_ids_list, df_attrs[tracklet_key].to_list(), strict=True)
    )
    feat_cols = {key: df_attrs[key].to_list() for key in node_feature_keys}
    node_to_feat = {
        node: {key: feat_cols[key][i] for key in node_feature_keys}
        for i, node in enumerate(node_ids_list)
    }

    # Batch-fetch all edges in one query and build adjacency maps.
    # This replaces all per-node predecessors/successors/in_degree/out_degree calls.
    edge_df = solution_nx_graph.edge_attrs(
        attr_keys=[DEFAULT_ATTR_KEYS.EDGE_SOURCE, DEFAULT_ATTR_KEYS.EDGE_TARGET]
    )
    sources = edge_df[DEFAULT_ATTR_KEYS.EDGE_SOURCE].to_list()
    targets = edge_df[DEFAULT_ATTR_KEYS.EDGE_TARGET].to_list()
    child_to_parent: dict[int, int] = {}
    parent_to_children: dict[int, list[int]] = {}
    for src, tgt in zip(sources, targets, strict=True):
        child_to_parent[tgt] = src
        parent_to_children.setdefault(src, []).append(tgt)

    track_list = []
    parent_mapping = []

    # Identify parent nodes (nodes with more than one child) and end nodes
    parent_nodes = [n for n in node_ids_list if len(parent_to_children.get(n, [])) > 1]
    end_nodes = [n for n in node_ids_list if n not in parent_to_children]

    # BFS to collect tracklets, cutting edges at division (parent) nodes
    tracklets = get_tracklets(
        parent_to_children,
        child_to_parent,
        node_ids_list,
        set(parent_nodes),
        node_to_track_id,
    )

    # Map every track id to its color in one vectorized colormap.map call, then
    # look up per tracklet. colormap.map has a large fixed per-call overhead, so
    # a single array call is far faster than calling it once per tracklet.
    unique_track_ids = list(set(node_to_track_id.values()))
    tid_to_color = dict(
        zip(unique_track_ids, colormap.map(np.asarray(unique_track_ids)), strict=True)
    )

    # each tracklet's first node: where it starts in time and in the trench
    first_nodes = [
        min(node_set, key=node_to_time.__getitem__) for node_set in tracklets
    ]
    rows = tracks.get_positions(first_nodes)[:, -2] if first_nodes else np.empty(0)
    starts = {
        node_to_track_id[node]: (node_to_time[node], float(row))
        for node, row in zip(first_nodes, rows, strict=True)
    }

    for node_set in tracklets:
        # Sort nodes in each tracklet by time using the precomputed dict
        sorted_nodes = sorted(node_set, key=lambda node: node_to_time[node])

        # track_id and color are the same for all nodes in a node_set
        parent_track_id = None
        track_id = node_to_track_id[sorted_nodes[0]]
        color = np.concatenate((tid_to_color[track_id][:3] * 255, [255]))

        for node in sorted_nodes:
            if node in parent_nodes:
                state = NodeType.SPLIT
                symbol = "t1"
            elif node in end_nodes:
                state = NodeType.END
                symbol = "x"
            else:
                state = NodeType.CONTINUE
                symbol = "o"

            track_dict = {
                "t": node_to_time[node],
                "node_id": node,
                "track_id": track_id,
                "color": color,
                "parent_id": 0,
                "parent_track_id": 0,
                "state": state,
                "symbol": symbol,
            }

            for feature_key, feature in tracks.features.items():
                if feature.get("feature_type") == "edge":
                    continue
                if feature_key not in node_to_feat[node]:
                    continue
                display_name = feature.get("display_name", feature_key)
                value_names = feature.get("value_names", None)
                val = node_to_feat[node][feature_key]
                num_values = feature.get("num_values", 1)
                if num_values > 1:
                    for i in range(num_values):
                        v = val[i]
                        if isinstance(display_name, list | tuple):
                            name = display_name[i]
                        elif (
                            isinstance(value_names, list)
                            and len(value_names) == num_values
                        ):
                            name = f"{value_names[i]}"
                        else:
                            name = f"{display_name}_{i}"
                        track_dict[name] = v
                else:
                    track_dict[display_name] = val

            # Determine parent_id and parent_track_id
            parent_id = child_to_parent.get(node)
            if parent_id is not None:
                track_dict["parent_id"] = parent_id

                if parent_track_id is None:
                    parent_track_id = node_to_track_id[parent_id]
                track_dict["parent_track_id"] = parent_track_id

            else:
                parent_track_id = 0
                track_dict["parent_id"] = 0
                track_dict["parent_track_id"] = parent_track_id

            track_list.append(track_dict)

        parent_mapping.append(
            {"track_id": track_id, "parent_track_id": parent_track_id, "node_id": node}
        )

    lanes = trench_lanes(
        starts, {m["track_id"]: m["parent_track_id"] for m in parent_mapping}
    )
    x_axis_order = sorted(lanes, key=lanes.get)
    for node in track_list:
        node["x_axis_pos"] = lanes[node["track_id"]]

    df = pd.DataFrame(track_list)
    return df, x_axis_order


def find_root(track_id: int, parent_map: dict) -> int:
    """Function to find the root associated with a track by tracing its lineage"""

    # Keep traversing a track is found where parent_track_id == 0 (i.e., it's a root)
    current_track = track_id
    while parent_map.get(current_track) != 0:
        current_track = parent_map.get(current_track)
    return current_track


# Where a parent sits between its daughters, as a fraction of the way from the
# mother (the daughter nearer the closed end) to the other daughter: near the mother,
# so the mother's line barely moves at each division and the daughter drops away.
TREE_SPLIT = 1 / 8


def trench_lanes(
    starts: dict[int, tuple[float, float]], parents: dict[int, int]
) -> dict[int, float]:
    """The lane of each tracklet in the lineage tree, in trench order.

    Lineages are stacked as the cells sit in the trench: founders present from the
    start first, from the closed end down, then those that appear later, one empty
    lane between lineages. Within a lineage the daughter nearer the closed end (the
    mother, which keeps the old pole) goes above her sister, and the parent sits
    ``TREE_SPLIT`` of the way from her towards the sister. Every tracklet without
    daughters gets a whole-numbered lane of its own.

    Args:
        starts: track id -> (first time point, row position when first seen); row 0
            is the trench's closed end.
        parents: track id -> parent track id (0 or a missing id for none).

    Returns:
        track id -> lane.
    """
    children: dict[int, list[int]] = {}
    for track_id, parent in parents.items():
        if parent in starts:
            children.setdefault(parent, []).append(track_id)

    def by_position(track_id):
        t0, y0 = starts[track_id]
        return (t0, y0, track_id)

    lanes: dict[int, float] = {}
    next_lane = 0

    def place(track_id):
        nonlocal next_lane
        kids = sorted(children.get(track_id, []), key=lambda k: (starts[k][1], k))
        if not kids:
            lanes[track_id] = next_lane
            next_lane += 1
            return
        for kid in kids:
            place(kid)
        keep = lanes[kids[0]]
        lanes[track_id] = keep + TREE_SPLIT * (lanes[kids[-1]] - keep)

    roots = sorted((t for t in starts if parents.get(t) not in starts), key=by_position)
    for root in roots:
        place(root)
        next_lane += 1  # gap between lineages
    return lanes


def extract_lineage_tree(graph: td.GraphView, node_id: str) -> list[str]:
    """Extract the entire lineage tree including horizontal relations for a given node"""

    # Walk up to root — one SQL call per step (unavoidable for linear parent chains)
    root_node = int(node_id)
    while True:
        preds = graph.predecessors(root_node)
        if not preds:
            break
        root_node = int(preds[0])

    # BFS downward batched by level — O(num_levels) SQL calls instead of O(N_descendants)
    nodes: set[int] = set()
    level = [root_node]
    while level:
        nodes.update(level)
        children_map: dict[int, list[int]] = graph.successors(level)
        next_level = [
            int(child)
            for children in children_map.values()
            for child in children
            if int(child) not in nodes
        ]
        level = next_level

    return list(nodes)


def get_features_from_tracks(
    tracks: Tracks | None = None, features_to_ignore: list[str] | None = None
) -> list[str]:
    """Extract the regionprops feature display names currently activated on Tracks.

    Args:
        tracks (Tracks | None): the Tracks instance to extract features from

    Returns:
        features_to_plot (list[str]): list of the feature names to plot, or an empty list
        if tracks is None
    """

    if features_to_ignore is None:
        features_to_ignore = []
    features_to_plot = []
    if tracks is not None:
        for key, feature in tracks.features.items():
            # Skip edge features - only show node features in dropdown
            if feature["feature_type"] == "edge":
                continue
            name = feature.get("display_name", key)
            if feature["value_type"] in ("float", "int"):
                if feature["num_values"] > 1:
                    value_names = feature.get("value_names", None)
                    for i in range(feature["num_values"]):
                        if isinstance(name, list | tuple):
                            features_to_plot.append(name[i])
                        elif (
                            value_names is not None
                            and len(value_names) == feature["num_values"]
                        ):
                            features_to_plot.append(value_names[i])
                        else:
                            features_to_plot.append(f"{name}_{i}")
                else:
                    features_to_plot.append(name)

    features_to_plot = [
        feature
        for feature in features_to_plot
        if not any(ig in feature for ig in features_to_ignore)
    ]

    return features_to_plot
