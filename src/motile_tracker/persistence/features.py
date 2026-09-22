"""Make custom group schema changes reversible, just like membership edits."""

from funtracks.actions import Action

from .codec import decode_dtype, encode_dtype


def _feature_state(tracks, name):
    definition = tracks.features.get(name)
    schema = tracks.graph_full._node_attr_schemas().get(name)
    values = {}
    if schema:
        values = {
            int(row["node_id"]): row[name]
            for row in tracks.graph_full.node_attrs(
                attr_keys=["node_id", name]
            ).iter_rows(named=True)
        }
    return {
        "name": name,
        "definition": dict(definition) if definition else None,
        "schema": [encode_dtype(schema.dtype), schema.default_value]
        if schema
        else None,
        "values": values,
    }


class FeatureChange(Action):
    def __init__(self, tracks, target):
        super().__init__(tracks)
        name = target["name"]
        self.saved = _feature_state(tracks, name)
        managed = name in tracks.annotators.all_features
        if name in tracks.features:
            if managed:
                tracks.disable_features([name])
            else:
                del tracks.features[name]
        if name in tracks.graph_full.node_attr_keys():
            tracks.graph_solution.remove_node_attr_key(name)
        if target["definition"] is not None:
            if managed:
                tracks.enable_features([name], recompute=False)
            else:
                tracks.add_feature(name, target["definition"])
        elif target.get("schema"):
            dtype, default = target["schema"]
            tracks.graph_solution.add_node_attr_key(
                name, decode_dtype(dtype), default_value=default
            )
        if target["values"]:
            tracks.graph_full.update_node_attrs(
                attrs={name: list(target["values"].values())},
                node_ids=list(target["values"]),
            )

    def inverse(self):
        return FeatureChange(self.tracks, self.saved)


def change_feature(tracks, name, definition):
    action = FeatureChange(
        tracks, {"name": name, "definition": definition, "values": {}}
    )
    tracks.action_history.add_new_action(action)
    tracks.refresh.emit(None)


def toggle_feature(tracks, name, enabled):
    if (name in tracks.features) == enabled:
        return
    action = object.__new__(FeatureChange)
    action.tracks = tracks
    action.saved = _feature_state(tracks, name)
    if enabled:
        tracks.enable_features([name])
    else:
        tracks.disable_features([name])
    tracks.action_history.add_new_action(action)
    tracks.refresh.emit(None)
