Importing Externally Generated Tracks
=====================================

Usage Overview
**************

It is also possible to view tracks that were not created from the motile widget using
the synchronized Lineage View and napari layers. Bringing them in is an *import* rather
than a load: because the application cannot assume anything about how another tool
wrote the data, you have to describe its layout as part of importing it. See
:ref:`save-load-vs-import-export` for how this differs from loading tracks you saved
here yourself.

To import, navigate to the ``Results List`` tab and
select ``External tracks from CSV`` or ``External tracks from geff`` in the dropdown menu at the bottom of the widgets, and click ``Load``.
A pop up menu will allow you to select a CSV file or geff zarr folder and map its columns to the required default attributes and optional additional attributes.
You may also provide the accompanying segmentation and specify scaling information.

Once imported, tracks appear in the results list like any other set of tracks, and can
be edited and then saved in the application's own format so that later sessions can load
them back without repeating the column mapping.

The following columns have to be selected:

- time: representing the position of the object in the time dimension.
- x: x centroid coordinate of the object.
- y: y centroid coordinate of the object.
- z (optional): z centroid coordinate of the object, if it is a 3D object.
- id: unique id of the object.
- parent_id: id of the directly connected predecessor (parent) of the object. Should be empty if the object is at the start of a lineage.
- seg_id: label value in the segmentation image data (if provided) that corresponds to the object id.

From this, a `SolutionTracks object`_ is generated, containing a networkx graph representing the tracking result, and optionally
a segmentation. The networkx graph is directed, with nodes representing detections and
edges going from a detection in time t to the same object in t+n (edges go forward in time).
Nodes must have an attribute representing time, by default named "t" but a different name
can be stored in the ``Tracks.time_attr`` attribute. Nodes must also have one or more attributes
representing position. The default way of storing positions on nodes is an attribute called
"pos" containing a list of position values, but dimensions can also be stored in separate attributes
(e.g. "x" and "y", each with one value). The name or list of names of the position attributes
should be specified in ``Tracks.pos_attr``.

The segmentation is expected to be a numpy array with time as the first dimension, followed
by the position dimensions in the same order as the ``Tracks.pos_attr``. The segmentation
must have unique label ids across all time points - there is a helper function in the
funtracks called ensure_unique_labels that relabels a segmentation to be unique
across time if needed. If a segmentation is provided, the node ids in the graph should
match label id of the corresponding segmentation.

An example script that loads a tracks object from a CSV and segmentation array
is provided in ``scripts/view_external_tracks.py``.

Loading GEFF groups with their images (kymograph view)
******************************************************

Datasets that keep one GEFF per field of view next to the image and label arrays
it describes (for example one group per mother machine trench) can be loaded
without any column mapping from the ``Kymograph Data`` tab, as long as the GEFF
metadata points at the arrays through ``related_objects``::

    trenches.zarr/
        trench_0165/
            images                 (T, Y, X) raw image stack
            segmentation           (T, Y, X) labels, label value == GEFF seg_id
            tracking_graph.geff    related_objects -> ../segmentation, ../images
        trench_0187/
            ...

Select the store with ``Browse…``: every GEFF group in it is listed by the name
of the group that holds it. ``Load`` adds the image as a napari layer, adds the
tracks to the Tracks List (so they can be edited, saved and exported like any
other tracks) and, when ``Open in kymograph view`` is checked, switches the
viewer to the kymograph view of that group with the image behind it. The arrow
buttons step through the groups; by default the image layers of earlier loads
are removed to keep the layer list short (the tracks stay in the Tracks List).

The property names are taken from the GEFF metadata: the typed ``axes`` give the
time and position properties, ``track_node_props`` give the tracklet and lineage
ids, and the ``label_prop`` of the labels object gives the segmentation ids.

The same can be done from the command line, which is convenient for opening one
trench straight away::

    motile_tracker --data trenches.zarr --group trench_0165

Without ``--group`` the first group is loaded; ``--spatial`` shows it in the
regular spatial view instead of the kymograph.

Once you have a Tracks object in the format described above, the following code
will view it in the Tree View and create synchronized napari layers (Points,
Labels, and Tracks) to visualize the provided tracks:

.. code-block:: python

    tracks_viewer = TracksViewer.get_instance(viewer)
    tracks_viewer.tracks_list.add_tracks(tracks, "example")

We plan to incorporate loaders from standard formats in the future to make this process easier,
and incorporate the loading into the user interface.

.. _SolutionTracks object: https://funkelab.github.io/funtracks/latest/reference/funtracks/data_model/solution_tracks/#funtracks.data_model.solution_tracks.SolutionTracks
