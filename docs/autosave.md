# Automatic saving and persistent history

This fork saves completed track edits automatically. The Tracks List shows each
dataset's destination and save status. Opening a writable GEFF attaches the
editing session to that folder. New results and CSV imports get unique names in
the application's data directory. A read-only dataset opens as a labelled local
working copy. Changing the copy-directory fields does not redirect autosave.

The floppy-disk control saves a separate copy, including history. Choose a new
filename; existing copies are not overwritten. Editing continues in the original
session until the copy is opened. Relative raw-image references are adjusted so
the copied dataset still finds the images. Copying a GEFF does not copy external raw
image volumes it references.

## What survives reopening

Completed node, link, segmentation, split/merge, group and measurement edits,
including undo and redo, are committed to `edit_history/history.sqlite`. The
initial full graph and deleted candidates are preserved, so an edit made in an
earlier session can still be undone. Editing after undo retains the earlier
operations, following funtracks' existing history behaviour. View navigation,
selection and display settings are not data edits.

Each revision records its operation, UTC timestamp and session identifier.
Sessions record the installed application and library versions. The schema has
an explicit version; unsupported schema versions or action types are rejected.
Serialization uses an allowlist of action fields and typed arrays/masks, without
pickle or executing action constructors during restoration.

## Persistence and recovery

One application writer owns a GEFF at a time. SQLite uses rollback journaling,
FULL synchronization and fullfsync where supported. Every completed edit's
revision is committed synchronously. These guarantees depend on the filesystem
honouring its locking and flush operations; use a local working directory.

The database stores the current graph for fast reopening, compressed immutable
before/after changes, deduplicated binary array payloads and action records, and
undo/redo stack transitions. Earlier graphs can be reconstructed with
`session.store.state_at(revision)`. Checkpoints do not prune history.

A single background worker rebuilds conventional GEFF data from a committed
database snapshot. It never reads the graph while the UI is mutating it. Rapid
edits can share one graph publication, but each has its own durable history
transaction. Conventional GEFF readers see the latest published solution; this
fork restores the latest committed revision. A publication marker allows this
fork to recover interrupted node/edge replacement, including an empty solution.

The history lives in a recognised Zarr group. Publication replaces only graph
members and preserves the history, extra files and raw-image references. Input
label images are not overwritten. Their stale label references are removed from
the published GEFF because its embedded node masks hold the edited segmentation;
the original references remain recorded in history metadata.

The editor detects external changes to the published graph by content fingerprint
and stops instead of overwriting them. Timestamp-only changes, such as copying or
unpacking a folder, do not invalidate history. External changes require resolving
the separate versions; this implementation does not merge concurrent edits.

If a journal write or undo fails, the in-memory graph and history return to the
last committed state and autosave reports a stopped state. Resolve the failure
and reopen the dataset before continuing. A projection failure retains the
committed history and is reported explicitly. Normal application shutdown drains
pending publications; after an abnormal exit, reopen in this fork to recover
before using the GEFF in other software. Allow the application to finish closing
before copying a live dataset with a file manager.

## Verification

The persistence tests cover save/reopen/undo/redo, split-created cells, deleted
candidates, editing after undo, groups, measurements, Zarr v2/v3, 3D masks,
initially empty tracks, deletion of the final cell, copies, read-only working
copies, external modification, failed writes, interrupted undo, failed loading,
and abrupt subprocess exit without Save or close. Tests use temporary data. The
full suite passed 631 tests (4 optional-data skips and 1 expected failure); the
four optional real-data tests also passed separately on copied datasets.

A development measurement on a temporary copy of trench_0165 (3,309 nodes) found
an initial save of approximately 0.68 seconds and a median of 107 ms for 20 link
deletions/undos including the edit and durable commit. The maximum was 279 ms;
reopening took approximately 0.16 seconds. Compressed revision storage grew by
about 2.8 MB. These are measurements of one local dataset and machine, not a
latency guarantee for larger volumes or network storage.
