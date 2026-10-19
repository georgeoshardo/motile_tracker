"""Keep a GEFF dataset in its own format while it is edited.

A dataset written by another tool (for example the Trackastra trench bundle on
the BioImage Archive) has its own property names, id numbering, label array and
metadata. funtracks loads it into its own model: time becomes ``t``, ids are
offset by one so that no cell has label 0, and masks live in the graph. Writing
that model back as it is would change the dataset's format, which is what
happened before this module existed.

When an editing session starts, the dataset's format is recorded
(:func:`capture`): the property names and array encodings, the metadata, the
node and edge order, the label array, and the original values of every cell.
Each publication rebuilds the dataset in that format from the edited tracks
(:func:`write`):

- cells keep their original ids; new cells get ids above the highest ever used;
- the label array is rewritten from the edited masks, with each cell's label
  property (``seg_id``); new cells get the lowest free label of their frame;
- positions keep their original value while a cell is unchanged, and take the
  position funtracks computes for an edited or new cell;
- tracklets and lineages follow the GEFF definitions, keeping the original
  numbers wherever a tracklet or lineage survives (on a join, the earlier one's);
- properties that repeat another (``label`` == ``seg_id``, ``coords`` == the
  position) are kept in step with it;
- metadata, encodings, property order and JSON layout are the original ones.

:func:`check` is the gate a publication must pass before it replaces the
previous one: the GEFF validator including its data checks, the dataset's
format, and agreement between the label array and the masks.

Only datasets whose format can be reproduced are handled; :func:`capture` raises
:class:`UnsupportedFormat` otherwise, and the caller must then not publish over
the source.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import zarr

FORMAT_VERSION = 1
HISTORY_GROUP = "edit_history"
_JSON_STYLES = (
    {"indent": 2},
    {"indent": 4},
    {"indent": 2, "sort_keys": True},
    {"indent": 4, "sort_keys": True},
)


class UnsupportedFormat(ValueError):
    """The dataset's format cannot be reproduced faithfully."""


class PublicationRejected(ValueError):
    """A publication failed the gate and was not written."""

    def __init__(self, problems):
        self.problems = list(problems)
        shown = "; ".join(self.problems[:5])
        more = f" (+{len(self.problems) - 5} more)" if len(self.problems) > 5 else ""
        super().__init__(f"Publication rejected: {shown}{more}")


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def is_native(attrs: dict) -> bool:
    """Whether a GEFF was written by funtracks (this application) itself."""
    return "funtracks" in ((attrs.get("geff") or {}).get("extra") or {})


def _labels_object(geff_attrs: dict) -> dict | None:
    for obj in geff_attrs.get("related_objects") or []:
        if obj.get("type") == "labels":
            return obj
    return None


def _label_prop(obj: dict | None) -> str | None:
    if obj is None:
        return None
    return obj.get("node_prop") or obj.get("label_prop")


def _dumps(value, style: dict) -> str:
    text = json.dumps(
        value, indent=style.get("indent", 2), sort_keys=style.get("sort_keys", False)
    )
    return text + style.get("newline", "")


def _json_style(text: str) -> dict:
    value = json.loads(text)
    for style in _JSON_STYLES:
        for newline in ("", "\n"):
            candidate = {**style, "newline": newline}
            if _dumps(value, candidate) == text:
                return candidate
    return {"indent": 2, "newline": "", "inexact": True}


def mask_digest(flat_indices: np.ndarray) -> str:
    """A fingerprint of a cell's pixels (flat indices within its frame)."""
    flat = np.sort(np.asarray(flat_indices, dtype=np.int64))
    return hashlib.sha1(flat.tobytes()).hexdigest()


def _mask_flat(mask, frame_shape) -> np.ndarray:
    """Flat pixel indices of a tracksdata Mask within its frame."""
    local = np.asarray(mask.mask, dtype=bool)
    offsets = np.asarray(mask.bbox, dtype=np.int64)[: local.ndim]
    coords = [
        axis + offset for axis, offset in zip(np.nonzero(local), offsets, strict=True)
    ]
    return np.ravel_multi_index(coords, frame_shape)


def _array_dirs(group: Path) -> list[Path]:
    return sorted(
        p.parent for p in group.rglob(".zarray") if HISTORY_GROUP not in p.parts
    )


def _read(array_dir: Path) -> np.ndarray:
    return zarr.open_array(str(array_dir), mode="r")[:]


# ----------------------------------------------------------------------
# capture
# ----------------------------------------------------------------------
def read_layout(geff_path) -> dict:
    """The parts of a GEFF's format that do not depend on its cells.

    That is the root metadata text and its JSON style, the group metadata files,
    every array's encoding, the node and edge property names, and the metadata
    files of the arrays it refers to (``related_objects``), keyed by object type.

    Raises:
        UnsupportedFormat: if the GEFF is not a zarr v2 group with fixed-length
            properties.
    """
    geff_path = Path(geff_path).resolve()
    root_file = geff_path / ".zattrs"
    if not root_file.is_file() or not (geff_path / ".zgroup").is_file():
        raise UnsupportedFormat("only zarr v2 GEFF groups keep their format")
    root_text = root_file.read_text()

    files, arrays = {}, {}
    for file in sorted(geff_path.rglob("*")):
        rel = file.relative_to(geff_path)
        if rel.parts and rel.parts[0] == HISTORY_GROUP or not file.is_file():
            continue
        if file.name in (".zgroup", ".zattrs") and rel != Path(".zattrs"):
            files[rel.as_posix()] = file.read_text()
    for array_dir in _array_dirs(geff_path):
        rel = array_dir.relative_to(geff_path).as_posix()
        arrays[rel] = {
            "zarray": json.loads((array_dir / ".zarray").read_text()),
            "zattrs": (array_dir / ".zattrs").read_text()
            if (array_dir / ".zattrs").exists()
            else None,
        }
    for required in ("nodes/ids", "edges/ids"):
        if required not in arrays:
            raise UnsupportedFormat(f"missing {required}")

    def props_of(kind):
        names = []
        for rel in arrays:
            parts = rel.split("/")
            if parts[0] == kind and len(parts) == 4 and parts[1] == "props":
                if parts[3] not in ("values", "missing"):
                    raise UnsupportedFormat(f"variable-length property {parts[2]!r}")
                if parts[3] == "values":
                    names.append(parts[2])
        return names

    related = {}
    geff = (json.loads(root_text).get("geff")) or {}
    for obj in geff.get("related_objects") or []:
        target = (geff_path / obj.get("path", "")).resolve()
        if obj.get("path") and target.is_dir() and obj.get("type") not in related:
            related[obj["type"]] = {
                name: (target / name).read_text()
                for name in ("zarr.json", ".zarray", ".zattrs")
                if (target / name).is_file()
            }
    return {
        "root_text": root_text,
        "root_style": _json_style(root_text),
        "files": files,
        "arrays": arrays,
        "node_props": props_of("nodes"),
        "edge_props": props_of("edges"),
        "related": related,
    }


def capture(geff_path, tracks, *, check_masks: bool = True) -> dict:
    """Record a GEFF dataset's format and its cells as they are now.

    Args:
        geff_path: The GEFF group (zarr v2) on disk, in its original format.
        tracks: The tracks loaded from it, before any edit. Used to verify that
            ids, times and masks line up with the files, and to learn the id
            offset funtracks applied on import.
        check_masks: Verify every cell's mask against the label array.

    Raises:
        UnsupportedFormat: if the format cannot be reproduced.
    """
    geff_path = Path(geff_path).resolve()
    if (geff_path / ".zattrs").is_file() and is_native(
        json.loads((geff_path / ".zattrs").read_text())
    ):
        raise UnsupportedFormat("written by funtracks; uses the native format")
    layout = read_layout(geff_path)
    root_text, files, arrays = layout["root_text"], layout["files"], layout["arrays"]
    node_props, edge_props = layout["node_props"], layout["edge_props"]
    geff = json.loads(root_text).get("geff") or {}
    node_ids = _read(geff_path / "nodes/ids").astype(np.int64)
    edge_ids = _read(geff_path / "edges/ids").astype(np.int64).reshape(-1, 2)
    values = {p: _read(geff_path / f"nodes/props/{p}/values") for p in node_props}
    missing = {
        p: _read(geff_path / f"nodes/props/{p}/missing").astype(bool)
        for p in node_props
        if f"nodes/props/{p}/missing" in arrays
    }
    edge_values = {p: _read(geff_path / f"edges/props/{p}/values") for p in edge_props}

    axes = geff.get("axes") or []
    time_props = [a["name"] for a in axes if a.get("type") == "time"]
    space_props = [a["name"] for a in axes if a.get("type") == "space"]
    if len(time_props) != 1 or not space_props:
        raise UnsupportedFormat("needs one time axis and spatial axes in the metadata")
    time_prop = time_props[0]
    for name in [time_prop, *space_props]:
        if name not in values or name in missing:
            raise UnsupportedFormat(f"axis property {name!r} missing or incomplete")
    track_props = geff.get("track_node_props") or {}
    tracklet_prop, lineage_prop = (
        track_props.get("tracklet"),
        track_props.get("lineage"),
    )
    labels_obj = _labels_object(geff)
    label_prop = _label_prop(labels_obj)
    for name in (tracklet_prop, lineage_prop, label_prop):
        if name is not None and (name not in values or name in missing):
            raise UnsupportedFormat(f"property {name!r} missing or incomplete")

    # id offset applied by funtracks on import, verified against the loaded tracks
    loaded = sorted(int(n) for n in tracks.graph_solution.node_ids())
    offset = 1 if (len(node_ids) and node_ids.min() == 0) else 0
    if sorted((node_ids + offset).tolist()) != loaded:
        raise UnsupportedFormat("node ids do not match the loaded tracks")
    loaded_times = dict(
        zip(loaded, (int(t) for t in tracks.get_times(loaded)), strict=True)
    )
    for node, t in zip(node_ids, values[time_prop], strict=True):
        if loaded_times[int(node) + offset] != int(t):
            raise UnsupportedFormat("node times do not match the loaded tracks")

    # roles of the other properties
    roles = {}
    position = np.stack([values[p] for p in space_props], axis=1).astype(float)
    for name in node_props:
        if name in (time_prop, *space_props, tracklet_prop, lineage_prop, label_prop):
            continue
        array = values[name]
        if name in missing:
            roles[name] = "passthrough"
        elif (
            label_prop is not None
            and array.shape == values[label_prop].shape
            and np.array_equal(array, values[label_prop])
        ):
            roles[name] = "label_alias"
        elif (
            array.ndim == 2
            and array.shape == position.shape
            and np.array_equal(array.astype(float), position)
        ):
            roles[name] = "position_alias"
        else:
            roles[name] = "passthrough"

    # label array, masks and the position convention
    labels = None
    digests = {}
    if labels_obj is not None:
        labels_path = (geff_path / labels_obj["path"]).resolve()
        if not labels_path.exists():
            raise UnsupportedFormat(f"label array {labels_obj['path']!r} not found")
        meta_files = {
            name: (labels_path / name).read_text()
            for name in ("zarr.json", ".zarray", ".zattrs")
            if (labels_path / name).is_file()
        }
        label_array = zarr.open_array(str(labels_path), mode="r")
        labels = {
            "path": labels_obj["path"],
            "files": meta_files,
            "shape": list(label_array.shape),
            "dtype": label_array.dtype.str,
        }
        if check_masks:
            data = label_array[:]
            frame_shape = data.shape[1:]
            graph = tracks.graph_solution
            masks = graph.node_attrs(attr_keys=["node_id", "mask"])
            loaded_masks = dict(
                zip(masks["node_id"].to_list(), masks["mask"].to_list(), strict=True)
            )
            by_frame = defaultdict(list)
            for i in range(len(node_ids)):
                by_frame[int(values[time_prop][i])].append(i)
            for t, indices in by_frame.items():
                frame = data[t].ravel()
                order = np.argsort(frame, kind="stable")
                sorted_frame = frame[order]
                for i in indices:
                    lab = values[label_prop][i]
                    lo, hi = np.searchsorted(sorted_frame, [lab, lab + 1])
                    flat = order[lo:hi]
                    digest = mask_digest(flat)
                    node = int(node_ids[i])
                    mask = loaded_masks.get(node + offset)
                    if (
                        mask is None
                        or mask_digest(_mask_flat(mask, frame_shape)) != digest
                    ):
                        raise UnsupportedFormat(
                            f"mask of node {node} differs from the label array"
                        )
                    digests[node] = digest

    archived = {}
    for i, node in enumerate(node_ids):
        entry = {
            "t": int(values[time_prop][i]),
            "pos": [float(v) for v in position[i]],
        }
        if label_prop is not None:
            entry["label"] = int(values[label_prop][i])
        if tracklet_prop is not None:
            entry["tracklet"] = int(values[tracklet_prop][i])
        if lineage_prop is not None:
            entry["lineage"] = int(values[lineage_prop][i])
        if int(node) in digests:
            entry["mask"] = digests[int(node)]
        passthrough = {
            name: (
                None
                if name in missing and missing[name][i]
                else values[name][i].tolist()
            )
            for name, role in roles.items()
            if role == "passthrough"
        }
        if passthrough:
            entry["props"] = passthrough
        archived[str(int(node))] = entry
    edge_archived = {}
    if edge_props:
        for j, (s, t) in enumerate(edge_ids):
            edge_archived[f"{int(s)},{int(t)}"] = {
                p: edge_values[p][j].tolist() for p in edge_props
            }

    return {
        "version": FORMAT_VERSION,
        "root_text": root_text,
        "root_style": layout["root_style"],
        "files": files,
        "arrays": arrays,
        "node_props": node_props,
        "edge_props": edge_props,
        "id_offset": offset,
        "node_order": [int(n) for n in node_ids],
        "edge_order": [[int(s), int(t)] for s, t in edge_ids],
        "roles": {
            "time": time_prop,
            "space": space_props,
            "label": label_prop,
            "tracklet": tracklet_prop,
            "lineage": lineage_prop,
            "other": roles,
        },
        "labels": labels,
        "archived": archived,
        "edge_archived": edge_archived,
    }


# ----------------------------------------------------------------------
# rebuilding the dataset from edited tracks
# ----------------------------------------------------------------------
def _tracklets(order, succ, pred):
    """GEFF tracklets: maximal chains where a node continues its predecessor's
    tracklet exactly when it has one predecessor and that predecessor has one
    successor. Returns lists of nodes in path order."""
    continues = {n: len(pred[n]) == 1 and len(succ[pred[n][0]]) == 1 for n in order}
    chains = []
    for head in order:
        if continues[head]:
            continue
        chain = [head]
        while len(succ[chain[-1]]) == 1 and continues[succ[chain[-1]][0]]:
            chain.append(succ[chain[-1]][0])
        chains.append(chain)
    return chains


def _components(order, succ, pred):
    seen, components = set(), []
    for start in order:
        if start in seen:
            continue
        stack, members = [start], []
        seen.add(start)
        while stack:
            n = stack.pop()
            members.append(n)
            for m in succ[n] + pred[n]:
                if m not in seen:
                    seen.add(m)
                    stack.append(m)
        components.append(members)
    return components


def _number(groups, times, archived_value, existing_max):
    """Number groups of nodes, keeping archived numbers where they survive.

    Groups are visited by their earliest node (time, then id); each takes the
    archived number of its earliest archived node unless an earlier group has
    already taken it. Other groups get numbers above the archived maximum.
    """
    keyed = sorted(groups, key=lambda g: min((times[n], n) for n in g))
    taken, numbers, next_number = set(), {}, existing_max + 1
    for group in keyed:
        candidates = sorted(
            (times[n], n) for n in group if archived_value(n) is not None
        )
        number = archived_value(candidates[0][1]) if candidates else None
        if number is None or number in taken:
            number = next_number
            next_number += 1
        taken.add(number)
        for n in group:
            numbers[n] = number
    return numbers


def build(tracks, fmt) -> dict:
    """Compute the dataset's content in its own format from the edited tracks."""
    offset = fmt["id_offset"]
    archived = fmt["archived"]
    roles = fmt["roles"]
    graph = tracks.graph_solution
    time_key = tracks.features.time_key
    labels_info = fmt.get("labels")
    frame_shape = tuple(labels_info["shape"][1:]) if labels_info else None
    keys = ["node_id", time_key] + (["mask"] if frame_shape else [])
    table = graph.node_attrs(attr_keys=keys)
    internal = [int(n) for n in table["node_id"].to_list()]
    times = {
        n - offset: int(t)
        for n, t in zip(internal, table[time_key].to_list(), strict=True)
    }
    masks = (
        {n - offset: m for n, m in zip(internal, table["mask"].to_list(), strict=True)}
        if frame_shape
        else {}
    )
    positions = {
        n - offset: [float(v) for v in np.ravel(p)]
        for n, p in zip(internal, tracks.get_positions(internal), strict=True)
    }
    present = set(times)
    order = [n for n in fmt["node_order"] if n in present]
    order += sorted(n for n in present if str(n) not in archived)
    edges = {(int(s) - offset, int(t) - offset) for s, t in graph.edge_list()}
    edge_order = [tuple(e) for e in fmt["edge_order"] if tuple(e) in edges]
    edge_order += sorted(edges - set(edge_order))

    # masks and positions: the original position while a cell is unchanged, else
    # the one funtracks computed for the edited cell
    flats, pos, changed = {}, {}, set()
    for n in order:
        a = archived.get(str(n))
        flat = None
        if frame_shape:
            if masks.get(n) is None:
                raise PublicationRejected([f"node {n} has no mask"])
            flat = _mask_flat(masks[n], frame_shape)
        flats[n] = flat
        if flat is not None:
            same = (
                a is not None
                and a["t"] == times[n]
                and a.get("mask") == mask_digest(flat)
            )
        else:
            same = (
                a is not None
                and a["t"] == times[n]
                and np.allclose(a["pos"], positions[n], rtol=0, atol=1e-9)
            )
        if same:
            pos[n] = a["pos"]
        else:
            changed.add(n)
            pos[n] = positions[n]

    # labels: archived label while unique in its frame, else the lowest free one
    label = {}
    if roles["label"] is not None:
        used = defaultdict(set)
        for n in order:
            a = archived.get(str(n))
            if (
                a is not None
                and a.get("label") is not None
                and a["label"] not in used[times[n]]
            ):
                label[n] = a["label"]
                used[times[n]].add(a["label"])
        limit = (
            np.iinfo(np.dtype(labels_info["dtype"])).max if labels_info else 2**63 - 1
        )
        for n in order:
            if n in label:
                continue
            free = 1
            while free in used[times[n]]:
                free += 1
            if free > limit:
                raise PublicationRejected(
                    [f"no free label for node {n} in frame {times[n]}"]
                )
            label[n] = free
            used[times[n]].add(free)

    succ, pred = defaultdict(list), defaultdict(list)
    for s, t in edge_order:
        succ[s].append(t)
        pred[t].append(s)
    tracklet = lineage = None
    archived_numbers = [
        a.get("tracklet") for a in archived.values() if a.get("tracklet") is not None
    ]
    if roles["tracklet"] is not None:
        tracklet = _number(
            _tracklets(order, succ, pred),
            times,
            lambda n: (archived.get(str(n)) or {}).get("tracklet"),
            max(archived_numbers, default=0),
        )
    archived_lineages = [
        a.get("lineage") for a in archived.values() if a.get("lineage") is not None
    ]
    if roles["lineage"] is not None:
        lineage = _number(
            _components(order, succ, pred),
            times,
            lambda n: (archived.get(str(n)) or {}).get("lineage"),
            max(archived_lineages, default=0),
        )
    return {
        "order": order,
        "edge_order": edge_order,
        "times": times,
        "pos": pos,
        "flats": flats,
        "label": label,
        "tracklet": tracklet,
        "lineage": lineage,
        "changed": changed,
    }


def _template_chunks(template, shape):
    if list(template["chunks"]) == list(template["shape"]):
        return tuple(max(1, s) for s in shape)
    return tuple(template["chunks"])


def _write_array(path: Path, data: np.ndarray, template: dict) -> None:
    import numcodecs

    meta = template["zarray"]
    compressor = (
        numcodecs.get_codec(dict(meta["compressor"]))
        if meta.get("compressor")
        else None
    )
    filters = (
        [numcodecs.get_codec(dict(f)) for f in meta["filters"]]
        if meta.get("filters")
        else None
    )
    array = zarr.create_array(
        store=str(path),
        shape=data.shape,
        chunks=_template_chunks(meta, data.shape),
        dtype=np.dtype(meta["dtype"]),
        zarr_format=2,
        compressors=compressor,
        filters=filters,
        fill_value=meta["fill_value"],
        order=meta.get("order", "C"),
        chunk_key_encoding={
            "name": "v2",
            "separator": meta.get("dimension_separator", "."),
        },
        overwrite=True,
    )
    if data.size:
        array[...] = data
    if template.get("zattrs") is not None:
        (path / ".zattrs").write_text(template["zattrs"])


def _missing_template(values_template: dict) -> dict:
    meta = dict(values_template["zarray"])
    meta.update(dtype="|b1", fill_value=False)
    meta["shape"] = meta["shape"][:1]
    meta["chunks"] = meta["chunks"][:1]
    return {"zarray": meta, "zattrs": values_template.get("zattrs")}


def write(tracks, fmt, geff_out, labels_out=None, *, drop_extra=False) -> dict:
    """Write the edited tracks in the dataset's format.

    Args:
        tracks: The edited tracks (their solution graph is written).
        fmt: The format recorded by :func:`capture`.
        geff_out: A new directory for the GEFF group.
        labels_out: A new directory for the label array, if the dataset has one.
        drop_extra: Leave the ``extra`` metadata out.

    Returns:
        The content written (see :func:`build`), for :func:`check`.
    """
    content = build(tracks, fmt)
    geff_out = Path(geff_out)
    geff_out.mkdir(parents=True)
    roles, archived, arrays = fmt["roles"], fmt["archived"], fmt["arrays"]
    order, edge_order = content["order"], content["edge_order"]

    for rel, text in fmt["files"].items():
        target = geff_out / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)

    def node_column(name):
        if name == roles["time"]:
            return [content["times"][n] for n in order], None
        if name in roles["space"]:
            axis = roles["space"].index(name)
            return [content["pos"][n][axis] for n in order], None
        if name == roles["label"]:
            return [content["label"][n] for n in order], None
        if name == roles["tracklet"]:
            return [content["tracklet"][n] for n in order], None
        if name == roles["lineage"]:
            return [content["lineage"][n] for n in order], None
        role = roles["other"][name]
        if role == "label_alias":
            return [content["label"][n] for n in order], None
        if role == "position_alias":
            return [content["pos"][n] for n in order], None
        column, missing = [], []
        template = arrays[f"nodes/props/{name}/values"]["zarray"]
        filler = np.zeros(template["shape"][1:], dtype=template["dtype"]).tolist()
        for n in order:
            a = archived.get(str(n))
            value = None if a is None or n in content["changed"] else a["props"][name]
            missing.append(value is None)
            column.append(filler if value is None else value)
        return column, missing if any(missing) else None

    ids_template = arrays["nodes/ids"]
    _write_array(
        geff_out / "nodes/ids",
        np.asarray(order, dtype=ids_template["zarray"]["dtype"]),
        ids_template,
    )
    for name in fmt["node_props"]:
        template = arrays[f"nodes/props/{name}/values"]
        column, missing = node_column(name)
        dtype = np.dtype(template["zarray"]["dtype"])
        trailing = tuple(template["zarray"]["shape"][1:])
        data = np.asarray(column, dtype=dtype).reshape((len(order), *trailing))
        _write_array(geff_out / f"nodes/props/{name}/values", data, template)
        missing_rel = f"nodes/props/{name}/missing"
        if missing is not None or missing_rel in arrays:
            m_template = arrays.get(missing_rel) or _missing_template(template)
            mask = np.asarray(
                missing if missing is not None else [False] * len(order), dtype=bool
            )
            _write_array(geff_out / missing_rel, mask, m_template)

    edge_template = arrays["edges/ids"]
    edge_data = np.asarray(edge_order, dtype=edge_template["zarray"]["dtype"]).reshape(
        -1, 2
    )
    _write_array(geff_out / "edges/ids", edge_data, edge_template)
    for name in fmt["edge_props"]:
        template = arrays[f"edges/props/{name}/values"]
        column, missing = [], []
        filler = np.zeros(
            template["zarray"]["shape"][1:], dtype=template["zarray"]["dtype"]
        ).tolist()
        for s, t in edge_order:
            value = (fmt["edge_archived"].get(f"{s},{t}") or {}).get(name)
            missing.append(value is None)
            column.append(filler if value is None else value)
        dtype = np.dtype(template["zarray"]["dtype"])
        data = np.asarray(column, dtype=dtype).reshape(
            (len(edge_order), *template["zarray"]["shape"][1:])
        )
        _write_array(geff_out / f"edges/props/{name}/values", data, template)
        missing_rel = f"edges/props/{name}/missing"
        if any(missing) or missing_rel in arrays:
            m_template = arrays.get(missing_rel) or _missing_template(template)
            _write_array(
                geff_out / missing_rel, np.asarray(missing, dtype=bool), m_template
            )

    # root metadata: the original, with axis ranges brought up to date
    attrs = json.loads(fmt["root_text"])
    geff = attrs["geff"]
    for axis in geff.get("axes") or []:
        if axis["name"] == roles["time"]:
            column = [content["times"][n] for n in order]
        else:
            index = roles["space"].index(axis["name"])
            column = [content["pos"][n][index] for n in order]
        if column and axis.get("min") is not None:
            axis["min"] = float(min(column))
        if column and axis.get("max") is not None:
            axis["max"] = float(max(column))
    if drop_extra:
        geff.pop("extra", None)
    (geff_out / ".zattrs").write_text(_dumps(attrs, fmt["root_style"]))

    if labels_out is not None and fmt.get("labels"):
        write_labels(content, fmt, labels_out)
    return content


def write_labels(content, fmt, labels_out) -> None:
    info = fmt["labels"]
    labels_out = Path(labels_out)
    labels_out.mkdir(parents=True)
    for name, text in info["files"].items():
        (labels_out / name).write_text(text)
    data = np.zeros(info["shape"], dtype=np.dtype(info["dtype"]))
    for n in content["order"]:
        frame = data[content["times"][n]].reshape(-1)
        frame[content["flats"][n]] = content["label"][n]
    zarr.open_array(str(labels_out), mode="r+")[...] = data


# ----------------------------------------------------------------------
# the gate
# ----------------------------------------------------------------------
def check(geff_dir, labels_dir, fmt, content, *, drop_extra=False) -> list[str]:
    """Problems with a written publication; empty if it may replace the last one."""
    from geff import validate_structure
    from geff.core_io import read_to_memory
    from geff.validate.data import ValidationConfig, validate_data

    problems = []
    geff_dir = Path(geff_dir)
    roles = fmt["roles"]
    try:
        validate_structure(str(geff_dir))
        memory = read_to_memory(str(geff_dir))
        validate_data(
            memory,
            ValidationConfig(
                graph=True,
                tracklet=roles["tracklet"] is not None,
                lineage=roles["lineage"] is not None,
            ),
        )
    except Exception as exc:  # noqa: BLE001 -- every validator failure is a problem
        problems.append(f"GEFF validation: {exc}")

    # same format: properties, encodings, metadata keys
    written = {
        p.parent.relative_to(geff_dir).as_posix(): json.loads((p).read_text())
        for p in geff_dir.rglob(".zarray")
    }
    for rel, template in fmt["arrays"].items():
        meta = written.get(rel)
        if meta is None:
            problems.append(f"array {rel} missing")
            continue
        for key, value in template["zarray"].items():
            if key not in ("shape", "chunks") and meta.get(key) != value:
                problems.append(f"array {rel}: {key} {meta.get(key)!r} != {value!r}")
    for rel in written:
        if rel not in fmt["arrays"] and not rel.endswith("/missing"):
            problems.append(f"unexpected array {rel}")
    original = json.loads(fmt["root_text"])["geff"]
    current = json.loads((geff_dir / ".zattrs").read_text())["geff"]
    expected_keys = set(original) - ({"extra"} if drop_extra else set())
    if set(current) != expected_keys:
        problems.append(f"metadata keys {sorted(current)} != {sorted(expected_keys)}")

    # the analysis contract: a forest moving forward in time
    order, times = content["order"], content["times"]
    indeg, outdeg = defaultdict(int), defaultdict(int)
    for s, t in content["edge_order"]:
        indeg[t] += 1
        outdeg[s] += 1
        if times[t] <= times[s]:
            problems.append(f"link {s}->{t} does not go forward in time")
    if any(v > 1 for v in indeg.values()):
        problems.append("a cell has two parents")
    if any(v > 2 for v in outdeg.values()):
        problems.append("a cell has more than two children")

    # label array agrees with the masks, cell by cell
    if labels_dir is not None and fmt.get("labels"):
        data = zarr.open_array(str(labels_dir), mode="r")[:]
        per_frame = defaultdict(set)
        for n in order:
            per_frame[times[n]].add(content["label"][n])
        if len(
            set(map(tuple, ((times[n], content["label"][n]) for n in order)))
        ) != len(order):
            problems.append("two cells share a label in one frame")
        for t in range(data.shape[0]):
            found = set(np.unique(data[t]).tolist()) - {0}
            if found != per_frame.get(t, set()):
                problems.append(f"frame {t}: labels in the array differ from the cells")
                break
        for n in order:
            frame = data[times[n]].reshape(-1)
            flat = content["flats"][n]
            if not np.all(frame[flat] == content["label"][n]) or np.count_nonzero(
                frame == content["label"][n]
            ) != len(flat):
                problems.append(
                    f"node {n}: array pixels differ from its mask (overlap?)"
                )
                break
    return problems


# ----------------------------------------------------------------------
# changing a dataset's format to another dataset's
# ----------------------------------------------------------------------
_ROOT_VALUES = ("extra",)  # metadata that describes the dataset, not its format


def conform_array_json(own_text: str, template_text: str) -> str:
    """An array's ``zarr.json`` with the template's attributes and layout.

    The array itself (shape, data type, chunks, codecs) must be the same; only
    its attributes (e.g. ``axis_types``) and JSON layout follow the template.
    """
    own, template = json.loads(own_text), json.loads(template_text)
    if {**own, "attributes": None} == {**template, "attributes": None}:
        return template_text
    same_kind = {k: v for k, v in own.items() if k not in ("attributes", "shape")} == {
        k: v for k, v in template.items() if k not in ("attributes", "shape")
    }
    if not same_kind:
        raise UnsupportedFormat("arrays are encoded differently; cannot conform")
    return _dumps(
        {**own, "attributes": template.get("attributes", {})},
        _json_style(template_text),
    )


def conform(fmt: dict, template: dict, *, drop_extra: bool = False) -> dict:
    """The dataset format `fmt` changed to follow `template` (see :func:`read_layout`).

    Afterwards the dataset is written exactly like the template's dataset:
    properties the template lacks are dropped together with their values,
    property and id data types, array encodings, group files, metadata layout
    and property order are the template's, and the label array's metadata is the
    template's. What describes this dataset rather than the format is kept: its
    cells, axis ranges, and its ``extra`` metadata unless `drop_extra`.

    Raises:
        UnsupportedFormat: if the template needs a property the dataset does not
            have, or the two describe their axes, tracks or related arrays
            differently.
    """
    import copy

    own_root = json.loads(fmt["root_text"])
    new_root = json.loads(template["root_text"])
    own, new = own_root["geff"], new_root["geff"]

    def axes_kind(geff):
        return [
            {k: v for k, v in axis.items() if k not in ("min", "max")}
            for axis in geff.get("axes") or []
        ]

    for key, describe in (
        ("axes", axes_kind),
        ("track_node_props", lambda g: g.get("track_node_props")),
        ("related_objects", lambda g: g.get("related_objects")),
    ):
        if describe(own) != describe(new):
            raise UnsupportedFormat(f"{key} differ from the template's")
    missing = [p for p in template["node_props"] if p not in fmt["node_props"]]
    missing += [p for p in template["edge_props"] if p not in fmt["edge_props"]]
    if missing:
        raise UnsupportedFormat(
            f"the template has properties this dataset lacks: {missing}"
        )

    # the template's metadata with this dataset's own values
    for axis, own_axis in zip(
        new.get("axes") or [], own.get("axes") or [], strict=True
    ):
        for bound in ("min", "max"):
            if bound in axis:
                axis[bound] = own_axis.get(bound)
    for key in _ROOT_VALUES:
        if key in own and not drop_extra:
            new[key] = own[key]
        else:
            new.pop(key, None)

    out = copy.deepcopy(fmt)
    out["root_text"] = _dumps(new_root, template["root_style"])
    out["root_style"] = template["root_style"]
    out["files"] = dict(template["files"])
    out["arrays"] = copy.deepcopy(template["arrays"])
    out["node_props"] = list(template["node_props"])
    out["edge_props"] = list(template["edge_props"])
    kept = set(template["node_props"])
    out["roles"]["other"] = {
        name: role for name, role in fmt["roles"]["other"].items() if name in kept
    }
    for entry in out["archived"].values():
        props = {k: v for k, v in (entry.get("props") or {}).items() if k in kept}
        if props:
            entry["props"] = props
        else:
            entry.pop("props", None)
    edge_kept = set(template["edge_props"])
    out["edge_archived"] = (
        {
            key: {k: v for k, v in values.items() if k in edge_kept}
            for key, values in fmt["edge_archived"].items()
        }
        if edge_kept
        else {}
    )
    labels_template = template["related"].get("labels")
    if out.get("labels") and labels_template:
        out["labels"]["files"] = {
            name: conform_array_json(text, labels_template[name])
            if name in labels_template
            else text
            for name, text in out["labels"]["files"].items()
        }
        dtype = json.loads(labels_template.get("zarr.json", "{}")).get("data_type")
        if dtype is not None and np.dtype(dtype) != np.dtype(out["labels"]["dtype"]):
            raise UnsupportedFormat("label arrays have different data types")
    return out


def rebase(fmt, source, destination, labels_target=None):
    """A copy of the format for a dataset copied from `source` to `destination`.

    Relative references in the metadata are rewritten so that they still point at
    the same objects; the label array may be redirected to `labels_target`.
    """
    import copy
    import os

    fmt = copy.deepcopy(fmt)
    attrs = json.loads(fmt["root_text"])
    for obj in attrs["geff"].get("related_objects") or []:
        reference = obj.get("path")
        if not reference or "://" in reference:
            continue
        if obj.get("type") == "labels" and labels_target is not None:
            obj["path"] = os.path.relpath(labels_target, destination)
        else:
            target = (Path(source) / reference).resolve()
            obj["path"] = os.path.relpath(target, destination)
    fmt["root_text"] = _dumps(attrs, fmt["root_style"])
    if fmt.get("labels") and labels_target is not None:
        fmt["labels"]["path"] = os.path.relpath(labels_target, destination)
    elif fmt.get("labels"):
        target = (Path(source) / fmt["labels"]["path"]).resolve()
        fmt["labels"]["path"] = os.path.relpath(target, destination)
    return fmt


def labels_path(geff_path, fmt) -> Path | None:
    info = fmt.get("labels") if fmt else None
    return (Path(geff_path) / info["path"]).resolve() if info else None
