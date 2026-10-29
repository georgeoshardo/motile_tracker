"""Publish committed state as conventional GEFF without touching its history."""

from __future__ import annotations

import hashlib
import os
import shutil
from contextlib import nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory

import zarr

from . import source_format
from .state import restore
from .store import Store

# Signatures that also cover the label array carry this prefix; older ones cover
# the graph only and are still accepted for the datasets that recorded them.
LABELS_SIGNATURE = "v2:"


def graph_signature(path, labels=None):
    """Detect content changes without rejecting copies with new file timestamps.

    With `labels` (the dataset's label array) the signature covers that array too.
    """
    path = Path(path)
    digest = hashlib.sha256()
    targets = [(path, member) for member in ("nodes", "edges", ".zattrs", "zarr.json")]
    if labels is not None:
        targets.append((Path(labels).parent, Path(labels).name))
    for base, member in targets:
        target = base / member
        files = sorted(target.rglob("*")) if target.is_dir() else [target]
        for file in files:
            if file.is_file():
                digest.update(str(file.relative_to(base)).encode())
                with file.open("rb") as handle:
                    while chunk := handle.read(1024 * 1024):
                        digest.update(chunk)
    return (LABELS_SIGNATURE if labels is not None else "") + digest.hexdigest()


def dataset_signature(path, fmt):
    """The signature of a dataset: graph, plus label array if its format has one."""
    return graph_signature(path, source_format.labels_path(path, fmt))


def check_external_change(store):
    path = store.path
    marker = path / "edit_history" / "publishing"
    expected = store.metadata("graph_signature")
    if marker.exists():
        lines = marker.read_text().splitlines()
        if len(lines) < 2:
            return  # An interrupted publication can be rebuilt from the journal.
        expected = lines[1]
    intact = all((path / name).is_dir() for name in ("nodes", "edges"))
    if not (expected and intact):
        return
    if expected.startswith(LABELS_SIGNATURE):
        labels = source_format.labels_path(path, store.metadata("source_format"))
        current = graph_signature(path, labels)
    else:
        current = graph_signature(path)
    if current != expected:
        raise ValueError(
            "GEFF changed outside this editing session; keep it intact and resolve the separate edits"
        )


def rebase_references(attrs, source, destination):
    import copy

    attrs = copy.deepcopy(attrs)
    for obj in attrs.get("geff", {}).get("related_objects") or []:
        reference = obj.get("path")
        if reference and "://" not in reference:
            target = (Path(source) / reference).resolve()
            try:
                # Members inside the GEFF are copied with it; keep those links
                # inside the new folder rather than pointing back to the source.
                obj["path"] = target.relative_to(Path(source).resolve()).as_posix()
            except ValueError:
                obj["path"] = os.path.relpath(target, destination)
    return attrs


def write_empty_geff(tracks, path, zarr_format):
    """GEFF's array writer supports empties; funtracks' position splitter does not."""
    import numpy as np
    from geff.core_io import write_arrays
    from geff_spec import GeffMetadata

    position = tracks.features.position_key
    spatial = (
        list(position) if isinstance(position, (list, tuple)) else tracks.axis_names
    )
    axes = [{"name": tracks.features.time_key, "type": "time"}]
    axes += [{"name": name, "type": "space"} for name in spatial]
    for axis, scale in zip(axes, tracks.scale or [1] * len(axes), strict=True):
        axis["scale"] = scale
    metadata = GeffMetadata(
        directed=True,
        axes=axes,
        node_props_metadata={},
        edge_props_metadata={},
        extra={
            "funtracks": {"features": tracks.features.dump_json()},
            "tracksdata": {
                k: v for k, v in tracks.graph_full.metadata.items() if k != "geff"
            },
        },
    )
    write_arrays(
        path,
        node_ids=np.empty(0, dtype=np.uint64),
        edge_ids=np.empty((0, 2), dtype=np.uint64),
        node_props={},
        edge_props={},
        metadata=metadata,
        zarr_format=zarr_format,
    )


def publish(path):
    """Only the session's single worker may call this while its lock is held.

    Publication spans multiple files and is deliberately not claimed to be atomic.
    The committed database remains untouched and can always rebuild these groups.
    """
    from funtracks.import_export import write_to_geff

    from mm_track.motile.backend.motile_run import MotileRun

    path = Path(path)
    # The session holds the writer lock; this independent connection belongs only
    # to the projection thread and never mutates revision data.
    with Store(path, readonly=True) as source:
        check_external_change(source)
        revision, state, _ = source.read(include_history=False)
        tracks = restore(state, source.get_blob)
        original_attrs = source.metadata("original_attrs", {})
        fmt = source.metadata("source_format")
    if fmt is not None:
        return publish_in_source_format(path, tracks, fmt, revision)
    zarr_format = 3 if (path / "zarr.json").exists() else 2
    with TemporaryDirectory(prefix="motile-projection-", dir=path.parent) as temporary:
        staged = Path(temporary) / "tracks.geff"
        if tracks.graph_solution.num_nodes():
            write_to_geff(tracks, staged, zarr_format=zarr_format)
        else:
            write_empty_geff(tracks, staged, zarr_format)
        if isinstance(tracks, MotileRun):
            from mm_track.motile.backend.motile_run import (
                GAPS_FILENAME,
                IN_POINTS_FILENAME,
            )

            tracks._save_params(staged)
            tracks._save_attrs(staged)
            if tracks.input_points is not None:
                tracks._save_array(staged, IN_POINTS_FILENAME, tracks.input_points)
            tracks._save_list(tracks.gaps, staged, GAPS_FILENAME)
        root = zarr.open_group(staged, mode="a")
        generated = dict(root.attrs)
        metadata = {**original_attrs, **generated}
        # Keep application extras and related raw-image references. The generated
        # axes/property descriptions must match the current graph.
        old_geff = original_attrs.get("geff", {})
        geff = {**old_geff, **generated.get("geff", {})}
        geff["extra"] = {**old_geff.get("extra", {}), **geff.get("extra", {})}
        if old_geff.get("related_objects") is not None:
            # Embedded node masks now describe the edited segmentation. Keep raw
            # image links but do not advertise the unchanged input labels as the
            # edited result. Their original references remain in journal metadata.
            geff["related_objects"] = [
                obj
                for obj in old_geff["related_objects"]
                if obj.get("type") != "labels"
            ]
        metadata["geff"] = geff
        root.attrs.update(metadata)
        with Store(path, readonly=True) as source:
            check_external_change(source)
        # Mark before replacing anything. A killed writer leaves this recoverable
        # marker; reopening can distinguish interrupted publication from outsiders.
        marker = path / "edit_history" / "publishing"
        with marker.open("w") as handle:
            handle.write(str(revision))
            handle.flush()
            os.fsync(handle.fileno())
        for member in staged.iterdir():
            target = path / member.name
            if member.is_dir():
                if target.exists():
                    shutil.rmtree(target)
                os.replace(member, target)
            else:
                os.replace(member, target)
        signature = graph_signature(path)
        marker.write_text(f"{revision}\n{signature}")
        return revision, signature


def publish_in_source_format(path, tracks, fmt, revision):
    """Publish a dataset in the format it arrived in (see :mod:`.source_format`).

    The graph and the label array are written to staging directories next to
    their targets and checked by the gate; only then do they replace the
    published ones. A rejected publication leaves the previous one in place and
    raises :class:`~.source_format.PublicationRejected`.
    """
    path = Path(path)
    labels_target = source_format.labels_path(path, fmt)
    with (
        TemporaryDirectory(prefix=".motile-publish-", dir=path.parent) as staging,
        TemporaryDirectory(prefix=".motile-labels-", dir=labels_target.parent)
        if labels_target is not None
        else nullcontext() as label_staging,
    ):
        staged = Path(staging) / path.name
        staged_labels = (
            Path(label_staging) / labels_target.name
            if labels_target is not None
            else None
        )
        content = source_format.write(tracks, fmt, staged, staged_labels)
        problems = source_format.check(staged, staged_labels, fmt, content)
        if problems:
            raise source_format.PublicationRejected(problems)
        with Store(path, readonly=True) as source:
            check_external_change(source)
        # Mark before replacing anything. A killed writer leaves this recoverable
        # marker; reopening can distinguish interrupted publication from outsiders.
        marker = path / "edit_history" / "publishing"
        with marker.open("w") as handle:
            handle.write(str(revision))
            handle.flush()
            os.fsync(handle.fileno())
        for member in staged.iterdir():
            target = path / member.name
            if member.is_dir() and target.exists():
                shutil.rmtree(target)
            os.replace(member, target)
        if labels_target is not None:
            previous = Path(label_staging) / "previous"
            if labels_target.exists():
                os.replace(labels_target, previous)
            os.replace(staged_labels, labels_target)
        signature = graph_signature(path, labels_target)
        marker.write_text(f"{revision}\n{signature}")
        return revision, signature
