"""The status bar text for the cell under the cursor: its IDs as the dataset saves them.

napari's status bar shows a layer's value at the cursor: a mask's label, but a
point's index in the points layer, which names no cell. The point and mask layers
show the cell instead, as ``node 3990, track 213, lineage 20``: its node id, tracklet
and lineage, the numbers the tree, the table and the saved GEFF use.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from funtracks.data_model import Tracks


def describe_cell(tracks: Tracks | None, node) -> str | None:
    """``node N, track T, lineage L`` for a cell of the tracks, None for no cell."""
    if tracks is None or node is None:
        return None
    try:
        node = int(node)
    except (TypeError, ValueError):
        return None
    graph = tracks.graph_solution
    if node <= 0 or not graph.has_node(node):
        return None
    parts = [f"node {node}"]
    keys = graph.node_attr_keys()
    for label, key in (
        ("track", tracks.features.tracklet_key),
        ("lineage", tracks.features.lineage_key),
    ):
        if key is not None and key in keys:
            value = tracks.get_node_attr(node, key)
            if value is not None:
                parts.append(f"{label} {int(value)}")
    return ", ".join(parts)


def with_cell(status: dict, text: str | None, *, keep_details: bool) -> dict:
    """The layer's status with its value replaced by the cell's IDs.

    Args:
        status: The status napari's layer made.
        text: The cell's description (:func:`describe_cell`); None leaves the
            status as it is.
        keep_details: Keep what napari added after the value (a mask's colouring
            property, say).
    """
    if text is None:
        return status
    value = str(status.get("value", ""))
    details = value.split("; ", 1)
    suffix = f"; {details[1]}" if keep_details and len(details) > 1 else ""
    status["value"] = text + suffix
    coords = status.get("coords", "")
    status["coordinates"] = f"{coords}: {text}{suffix}" if coords else text + suffix
    return status


class CellStatusPoints:
    """Mixin for a points layer whose points are cells (``self.nodes``, in point
    order): the status bar names the cell under the cursor, not the point index."""

    def get_status(
        self,
        position=None,
        *,
        view_direction=None,
        dims_displayed=None,
        world: bool = False,
    ) -> dict:
        status = super().get_status(
            position,
            view_direction=view_direction,
            dims_displayed=dims_displayed,
            world=world,
        )
        if position is None:
            return status
        index = self.get_value(
            position,
            view_direction=view_direction,
            dims_displayed=dims_displayed,
            world=world,
        )
        nodes = getattr(self, "nodes", None)
        node = None
        if index is not None and nodes is not None and 0 <= int(index) < len(nodes):
            node = nodes[int(index)]
        return with_cell(
            status, describe_cell(self.tracks_viewer.tracks, node), keep_details=False
        )
